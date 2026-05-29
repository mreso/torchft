# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stress test for torchft + RDMATransport on CIFAR-10 with a small CNN.

Layout
------
- 1 LighthouseServer in the parent process.
- 4 worker processes, one per replica group (world_size=1 per replica).
- A chaos-monkey thread in the parent randomly kills a worker every
  [chaos_min_steps, chaos_max_steps] cluster steps. With 4 replicas and
  manager_min_replica_size=2, killing one still leaves 3 healthy and the
  killed worker heals from a peer via RDMATransport (HTTP fallback on this
  CPU-only box).

Run
---
    python examples/stress_test_cifar10.py [--total-steps N]

Use --help to see all knobs.
"""

import argparse
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import timedelta
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Multiprocessing must use "spawn" so workers don't share the parent's
# Lighthouse gRPC server / file descriptors.
# ---------------------------------------------------------------------------
mp.set_start_method("spawn", force=True)

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar_data")
LOG_FORMAT = "%(asctime)s [%(processName)s] %(message)s"

# IPC channel between worker -> parent (results + heartbeats)
EVT_STEP = "step"
EVT_EVAL = "eval"
EVT_DONE = "done"
EVT_ERROR = "error"
EVT_HEAL = "heal"
EVT_READY = "ready"


# ---------------------------------------------------------------------------
# Tiny CNN for CIFAR-10 (32x32). Trains in well under a second per step on
# CPU at batch_size=64. Plenty of capacity to climb above random on a
# couple-hundred-step run, which is what we need to demonstrate that the
# torchft fault-tolerance machinery doesn't wreck convergence.
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pools: 32 -> 16 -> 8 -> 8 (no pool on last conv)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Construct the training model.

    ``smallcnn`` is the tiny default (a few MB — checkpoint transfer is
    negligible, recovery time is dominated by quorum/timeout overhead).
    ``resnet50`` is a ~25M-param model (~98 MB fp32 weights; with SGD momentum
    the checkpoint is ~196 MB) so the RDMA vs HTTP bulk-transfer difference is
    actually measurable. The torchvision ResNet-50 is adapted for 32x32 CIFAR
    input with the standard CIFAR stem (3x3 stride-1 conv, no max-pool) so the
    spatial map isn't collapsed before it reaches the residual stages.
    """
    if model_name == "smallcnn":
        return SmallCNN(num_classes)
    if model_name == "resnet50":
        m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    if model_name in LLAMA_CONFIGS:
        return LlamaLM(LLAMA_CONFIGS[model_name])
    raise ValueError(
        f"unknown --model {model_name!r} "
        f"(smallcnn|resnet50|{'|'.join(LLAMA_CONFIGS)})"
    )


def is_lm_model(model_name: str) -> bool:
    return model_name in LLAMA_CONFIGS


# ---------------------------------------------------------------------------
# Self-contained Llama-style decoder (no transformers/torchtitan dependency).
# RMSNorm + RoPE + (optional GQA) attention via SDPA + SwiGLU MLP. Random init
# — this is a *checkpoint-transfer* benchmark (RDMA GPUDirect vs HTTP), not a
# convergence run, so weights/data are synthetic. ``llama_7b`` is the standard
# Llama-2 7B shape (~6.7B params; bf16 checkpoint ≈ params + SGD momentum ≈
# 27 GB), large enough that bulk transfer dominates recovery time.
# ---------------------------------------------------------------------------
LLAMA_CONFIGS: Dict[str, Dict[str, int]] = {
    "llama_1b": dict(
        dim=2048, n_layers=16, n_heads=16, n_kv_heads=16,
        intermediate=5632, vocab=32000, max_seq=2048,
    ),
    "llama_7b": dict(
        dim=4096, n_layers=32, n_heads=32, n_kv_heads=32,
        intermediate=11008, vocab=32000, max_seq=2048,
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.to(dt)) * self.weight


def _precompute_rope(head_dim: int, seq: int, theta: float = 10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq).float()
    freqs = torch.outer(t, inv)  # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class LlamaAttention(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.n_kv = cfg["n_kv_heads"]
        self.head_dim = cfg["dim"] // cfg["n_heads"]
        self.wq = nn.Linear(cfg["dim"], self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg["dim"], self.n_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg["dim"], self.n_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg["dim"], bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(o)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg["dim"], cfg["intermediate"], bias=False)  # gate
        self.w3 = nn.Linear(cfg["dim"], cfg["intermediate"], bias=False)  # up
        self.w2 = nn.Linear(cfg["intermediate"], cfg["dim"], bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.attn = LlamaAttention(cfg)
        self.mlp = LlamaMLP(cfg)
        self.n1 = RMSNorm(cfg["dim"])
        self.n2 = RMSNorm(cfg["dim"])

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class LlamaLM(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab"], cfg["dim"])
        self.blocks = nn.ModuleList(
            [LlamaBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = RMSNorm(cfg["dim"])
        self.head = nn.Linear(cfg["dim"], cfg["vocab"], bias=False)
        head_dim = cfg["dim"] // cfg["n_heads"]
        cos, sin = _precompute_rope(head_dim, cfg["max_seq"])
        # Non-persistent: recomputable from config, kept out of the state_dict
        # so the checkpoint is pure weights (no constant rope tables to ship).
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        x = self.tok(idx)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm(x)
        return self.head(x)


class SyntheticTokens(Dataset):
    """Random next-token sequences. Deterministic per index (spawn-safe)."""

    def __init__(self, length: int, seq_len: int, vocab: int) -> None:
        self.length = length
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        ids = torch.randint(0, self.vocab, (self.seq_len + 1,), generator=g)
        return ids[:-1], ids[1:]


# ---------------------------------------------------------------------------
# Synthetic CIFAR-shaped dataset (used when real CIFAR-10 isn't on disk).
# Each class has a fixed prototype 3x32x32 tensor; samples are prototype +
# Gaussian noise. Noise scale is high enough that the task is non-trivial
# (a tiny CNN should reach ~50-70% in a few hundred steps, not 100% in 5).
# ---------------------------------------------------------------------------
class SyntheticCIFAR(Dataset):
    NUM_CLASSES = 10

    def __init__(
        self,
        length: int,
        train: bool,
        noise_scale: float = 1.5,
        seed: int = 12345,
    ) -> None:
        self.length = length
        self.train = train
        self.noise_scale = noise_scale
        g = torch.Generator().manual_seed(seed)
        # Prototypes shared across train/test so the test set is in-distribution.
        self.prototypes = torch.randn(self.NUM_CLASSES, 3, 32, 32, generator=g)
        self.split_offset = 0 if train else 10_000_000

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        label = idx % self.NUM_CLASSES
        g = torch.Generator().manual_seed(idx + self.split_offset)
        noise = torch.randn(3, 32, 32, generator=g) * self.noise_scale
        x = self.prototypes[label] + noise
        # Per-channel normalize so values look like a real image.
        x = (x - x.mean()) / (x.std() + 1e-5)
        return x, label


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
def _try_real_cifar(train_tf, test_tf):
    """Try to load real CIFAR-10 from disk (no download). Returns
    (trainset, testset) or None if not available."""
    try:
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=False, transform=train_tf
        )
        testset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=False, transform=test_tf
        )
        return trainset, testset
    except Exception:
        return None


def make_loaders(
    batch_size: int,
    replica_id: int,
    num_replicas: int,
    model_name: str = "smallcnn",
    seq_len: int = 256,
):
    # Language-model task: synthetic next-token sequences (no tokenizer / no
    # download). Each replica gets a disjoint shard via a per-replica index
    # offset so sequences differ across replicas.
    if is_lm_model(model_name):
        vocab = LLAMA_CONFIGS[model_name]["vocab"]
        trainset = SyntheticTokens(length=100_000, seq_len=seq_len, vocab=vocab)
        test_set = SyntheticTokens(length=256, seq_len=seq_len, vocab=vocab)
        gen = torch.Generator().manual_seed(1000 + replica_id)
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True, generator=gen,
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return train_loader, test_loader, False

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    real = _try_real_cifar(train_tf, test_tf)
    if real is not None:
        trainset, testset = real
        using_real = True
    else:
        trainset = SyntheticCIFAR(length=10_000, train=True)
        testset = SyntheticCIFAR(length=2_000, train=False)
        using_real = False

    # Each replica sees a disjoint shard of the training set so we don't
    # double-count samples across replicas. Use a per-replica shuffle seed
    # so reordering is independent across replicas.
    indices = list(range(replica_id, len(trainset), num_replicas))
    train_subset = torch.utils.data.Subset(trainset, indices)

    gen = torch.Generator().manual_seed(1000 + replica_id)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=gen,
    )
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader, using_real


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int = 8,
    device: object = None,
    is_lm: bool = False,
) -> float:
    model.eval()
    with torch.no_grad():
        if is_lm:
            # Report a (0,1] pseudo-score = exp(-mean CE) so the harness's
            # "higher is better / improved" logic stays coherent; the raw LM
            # loss is logged separately via EVT_STEP.
            tot, n = 0.0, 0
            for i, (x, y) in enumerate(loader):
                if i >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), y.view(-1)
                )
                tot += float(loss.item())
                n += 1
            model.train()
            return math.exp(-(tot / max(1, n)))
        correct, total = 0, 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(1, total)


# ---------------------------------------------------------------------------
# Worker: one replica group, world_size=1
# ---------------------------------------------------------------------------
def worker_main(
    replica_id: int,
    num_replicas: int,
    lighthouse_address: str,
    total_steps: int,
    batch_size: int,
    eval_every: int,
    log_every: int,
    event_q: mp.Queue,
    healed: bool,
    repo_root: str,
    manager_min_replica_size: int,
    seed: int,
    cluster_step,  # mp.Value('i') shared with the parent
    device_type: str = "cpu",
    transport_mode: str = "rdma",
    model_name: str = "smallcnn",
    seq_len: int = 256,
    grad_sync_every: int = 1,
    max_gpu_snapshot_gb: float = 4.0,
) -> None:
    # `spawn` workers don't inherit the parent's cwd in sys.path, so the
    # bundled site-packages torchft (which lacks rdma_transport) wins over
    # the local checkout. Force-prepend the repo root so the worker imports
    # the same torchft the parent does.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Imports inside the worker so spawn doesn't drag the parent state in.
    import torch.distributed as dist
    from torchft import (
        Manager,
        Optimizer,
        ProcessGroupGloo,
    )
    from torchft.checkpointing.rdma_transport import RDMATransport
    from torchft.checkpointing.http_transport import HTTPTransport

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    log = logging.getLogger(f"replica-{replica_id}")
    random.seed(seed + replica_id)
    torch.manual_seed(seed + replica_id)

    if device_type == "cuda":
        # One replica per GPU: replica i -> cuda:i.
        torch.cuda.set_device(replica_id)
        device = torch.device(f"cuda:{replica_id}")
        log.info("using device %s for replica %d", device, replica_id)
    else:
        device = torch.device("cpu")

    # Per-worker TCPStore so dist init in Manager has somewhere to publish.
    store = dist.TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )

    is_lm = is_lm_model(model_name)
    model = build_model(model_name).to(device)
    if is_lm and device.type == "cuda":
        # bf16 keeps a 7B model + grads + SGD momentum well within one MI300X
        # while still producing a ~27 GB checkpoint for the transfer benchmark.
        model = model.to(torch.bfloat16)
    log.info(
        "model=%s params=%d dtype=%s",
        model_name,
        sum(p.numel() for p in model.parameters()),
        next(model.parameters()).dtype,
    )
    lr = 1e-3 if is_lm else 0.05
    base_opt = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    def state_dict() -> Dict[str, object]:
        return {
            "model": model.state_dict(),
            "optim": base_opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

    # Holds the wall-clock start of the quorum/heal that is currently in
    # flight (set right before optimizer.zero_grad()). load_state_dict runs
    # synchronously inside that call, so (now - t0) is the end-to-end recovery
    # latency: quorum + checkpoint transfer (over RDMA or HTTP) + apply.
    heal_timer = {"t0": None}

    def load_state_dict(sd: Dict[str, object]) -> None:
        model.load_state_dict(sd["model"])
        base_opt.load_state_dict(sd["optim"])
        scheduler.load_state_dict(sd["scheduler"])
        log.info("loaded state from peer (model + optim + scheduler restored)")
        t0 = heal_timer.get("t0")
        heal_secs = (time.monotonic() - t0) if t0 is not None else float("nan")
        try:
            event_q.put({
                "type": EVT_HEAL,
                "replica": replica_id,
                "step": manager.current_step(),
                "heal_secs": heal_secs,
                "transport": transport_mode,
            })
        except Exception:
            pass

    pg = ProcessGroupGloo(timeout=timedelta(seconds=30))
    if transport_mode == "http":
        transport = HTTPTransport(timeout=timedelta(seconds=30), num_chunks=0)
        rdma_active = False
    else:
        transport = RDMATransport(
            device=device,
            timeout=timedelta(seconds=30),
            # Keep large snapshots resident on the GPU so the RDMA read path is
            # GPU->GPU (GPUDirect), not spilled to pinned CPU. Needed for big
            # LM checkpoints; harmless for small models.
            max_gpu_snapshot_bytes=int(max_gpu_snapshot_gb * (1 << 30)),
        )
        # _fallback is set (non-None) only when torchcomms RDMA was unavailable
        # and the transport silently downgraded to HTTP. Assert real RDMA so a
        # misconfigured run (missing NCCL_IB_GID_INDEX etc.) fails loudly
        # instead of quietly measuring HTTP.
        rdma_active = getattr(transport, "_fallback", None) is None
        if not rdma_active:
            log.error(
                "RDMA requested but transport fell back to HTTP "
                "(RdmaTransport.supported() is False) — check NCCL_IB_GID_INDEX / env"
            )
    log.info("transport=%s rdma_active=%s", transport_mode, rdma_active)

    # port=0 lets the OS pick a free port on every (re)spawn; this avoids
    # TIME_WAIT collisions when a chaos kill is followed by an immediate
    # restart on the same rank.
    manager = Manager(
        pg=pg,
        min_replica_size=manager_min_replica_size,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"replica_{replica_id}",
        store_addr="localhost",
        store_port=store.port,
        rank=0,
        world_size=1,
        lighthouse_addr=lighthouse_address,
        port=0,
        timeout=timedelta(seconds=60),
        quorum_timeout=timedelta(seconds=60),
        connect_timeout=timedelta(seconds=60),
        checkpoint_transport=transport,
        # Healed state must be applied before the next forward pass. With
        # async quorum a restarted worker can compute gradients from a fresh
        # random init and only load the checkpoint inside should_commit(),
        # corrupting the cluster on the first post-heal step.
        use_async_quorum=False,
    )

    log.info(
        "started: lighthouse=%s healed=%s pid=%d",
        lighthouse_address,
        healed,
        os.getpid(),
    )
    try:
        event_q.put({
            "type": EVT_READY,
            "replica": replica_id,
            "healed": healed,
            "transport": transport_mode,
            "rdma_active": rdma_active,
        })
    except Exception:
        pass

    # NOTE: we deliberately do NOT wrap `model` in torchft.DistributedDataParallel.
    # Each replica group has world_size=1, so there is no within-group allreduce
    # to perform. The cross-replica allreduce is driven manually below via
    # `manager.allreduce(p.grad)` after `loss.backward()`. Using the DDP wrapper
    # here is actively harmful: when a peer dies mid-step, the comm_hook raises
    # an AssertionError out of DDP's Reducer (because `_ManagedFuture._fut` is
    # left unset on a failed wait). DDP's bucket-reduction state then stays
    # corrupted forever, producing the
    #   "Expected to have finished reduction in the prior iteration..."
    # error on every subsequent forward pass. Driving allreduce manually keeps
    # the reducer state machine out of the picture entirely, so a
    # `manager.allreduce` failure is just a normal transient that the next
    # quorum cycle clears.
    optimizer = Optimizer(manager, base_opt)

    train_loader, test_loader, using_real = make_loaders(
        batch_size, replica_id, num_replicas, model_name, seq_len
    )
    if is_lm:
        log.info("dataset: synthetic tokens (seq_len=%d)", seq_len)
    else:
        log.info("dataset: %s CIFAR-10", "real" if using_real else "synthetic")

    last_loss = float("nan")
    try:
        # Outer loop in case the dataloader is exhausted before total_steps.
        while manager.current_step() < total_steps:
            for x, y in train_loader:
                # Pre-quorum value, only used if zero_grad itself raises
                # before we get a chance to refresh `step` post-heal.
                step = manager.current_step()

                # Per-step try so a transient peer-error during allreduce
                # (e.g. a chaos kill) doesn't kill this worker; the next
                # iteration will trigger a fresh quorum.
                try:
                    # Mark the start of this step's quorum so load_state_dict
                    # can report end-to-end recovery latency if a heal fires here.
                    heal_timer["t0"] = time.monotonic()
                    # zero_grad triggers start_quorum, which (with
                    # use_async_quorum=False) synchronously heals from a peer
                    # if this worker is behind — restoring model/optim state
                    # AND advancing manager._step to the cluster step. We must
                    # re-read `step` AFTER this so the local view matches the
                    # Manager's view; otherwise a freshly respawned worker
                    # would carry step=0 through the entire iteration, hiding
                    # progress in EVT_STEP logs and skipping the
                    # total_steps cutoff after a heal.
                    optimizer.zero_grad()
                    step = manager.current_step()
                    if step >= total_steps:
                        break
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    if is_lm:
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.float().view(-1, logits.size(-1)), y.view(-1)
                        )
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                    loss.backward()
                    # Cross-replica gradient averaging via the Manager. If the
                    # quorum is broken (e.g. a peer was just killed), the
                    # Manager swallows the error and `should_commit()` will
                    # return False on the next line, skipping the optimizer
                    # step. The next iteration's `zero_grad()` will trigger a
                    # fresh quorum and clear the manager's error state.
                    #
                    # The cross-replica allreduce runs over ProcessGroupGloo
                    # (CPU-only), so grads are staged through CPU. For a 7B model
                    # that is ~14 GB through CPU/Gloo per step, which would
                    # dwarf step time and make the chaos/recovery benchmark
                    # impractical — so for the LM task we sync only every
                    # ``grad_sync_every`` steps (0 = never; weight averaging
                    # disabled — this is a checkpoint-transfer benchmark, not a
                    # convergence run) and run a tiny proxy collective on the
                    # other steps so the Manager can still detect peer failure
                    # and drive should_commit. Vision keeps full sync every step.
                    sync_now = (
                        True
                        if not is_lm
                        else (grad_sync_every > 0 and step % grad_sync_every == 0)
                    )
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if sync_now and grads:
                        # Model + checkpoint stay on GPU, so the RDMA recovery
                        # path still exercises GPUDirect; only the grad sync is
                        # staged through CPU (Gloo limitation).
                        flat = torch.cat(
                            [g.detach().float().view(-1) for g in grads]
                        ).cpu()
                        work = manager.allreduce(flat)
                        work.wait()
                        flat = flat.to(device)
                        offset = 0
                        for g in grads:
                            n = g.numel()
                            g.copy_(flat[offset : offset + n].view_as(g).to(g.dtype))
                            offset += n
                    else:
                        # Tiny managed collective: keeps the Manager's quorum /
                        # failure-detection / commit accounting alive without a
                        # multi-GB CPU transfer.
                        manager.allreduce(torch.zeros(1)).wait()
                    optimizer.step()
                    new_step = manager.current_step()
                    if new_step > step:
                        scheduler.step()
                        # Publish progress to the parent immediately so the
                        # chaos monkey can fire on real cluster step counts
                        # rather than waiting for an EVT_STEP throttle.
                        try:
                            with cluster_step.get_lock():
                                if new_step > cluster_step.value:
                                    cluster_step.value = new_step
                        except Exception:
                            pass
                    last_loss = float(loss.item())
                except (RuntimeError, AssertionError) as e:
                    log.warning(
                        "transient step error at step %d: %r — continuing",
                        step,
                        e,
                    )
                    event_q.put(
                        {
                            "type": EVT_ERROR,
                            "replica": replica_id,
                            "error": repr(e),
                        }
                    )
                    continue

                if step % log_every == 0:
                    event_q.put(
                        {
                            "type": EVT_STEP,
                            "replica": replica_id,
                            "step": step,
                            "loss": last_loss,
                            "lr": base_opt.param_groups[0]["lr"],
                        }
                    )

                if step > 0 and step % eval_every == 0:
                    acc = evaluate(model, test_loader, device=device, is_lm=is_lm, max_batches=(2 if is_lm else 8))
                    event_q.put(
                        {
                            "type": EVT_EVAL,
                            "replica": replica_id,
                            "step": step,
                            "acc": acc,
                            "loss": last_loss,
                        }
                    )
    except Exception as e:
        log.exception("worker failed")
        event_q.put({"type": EVT_ERROR, "replica": replica_id, "error": repr(e)})
        raise
    finally:
        try:
            final_acc = evaluate(model, test_loader, device=device, is_lm=is_lm, max_batches=(2 if is_lm else 8))
        except Exception:
            final_acc = float("nan")
        event_q.put(
            {
                "type": EVT_DONE,
                "replica": replica_id,
                "loss": last_loss,
                "acc": final_acc,
                "step": manager.current_step(),
            }
        )
        try:
            manager.shutdown(wait=False)
        except Exception:
            pass
        # torchcomms holds folly Singletons (CtranIbSingleton / RegCache) that
        # abort the process during atexit teardown if any RDMA-registered
        # memory is still referenced. Flush the event queue and hard-exit to
        # bypass that crash (it would otherwise mark a clean worker as failed).
        try:
            event_q.close()
            event_q.join_thread()
        except Exception:
            pass
        os._exit(0)


# ---------------------------------------------------------------------------
# Parent: lighthouse + supervisor + chaos monkey
# ---------------------------------------------------------------------------
def spawn_worker(
    replica_id: int,
    num_replicas: int,
    lighthouse_address: str,
    total_steps: int,
    batch_size: int,
    eval_every: int,
    log_every: int,
    event_q: mp.Queue,
    healed: bool,
    repo_root: str,
    manager_min_replica_size: int,
    seed: int,
    cluster_step,
    device_type: str = "cpu",
    transport_mode: str = "rdma",
    model_name: str = "smallcnn",
    seq_len: int = 256,
    grad_sync_every: int = 1,
    max_gpu_snapshot_gb: float = 4.0,
) -> mp.Process:
    p = mp.Process(
        target=worker_main,
        name=f"replica-{replica_id}",
        args=(
            replica_id,
            num_replicas,
            lighthouse_address,
            total_steps,
            batch_size,
            eval_every,
            log_every,
            event_q,
            healed,
            repo_root,
            manager_min_replica_size,
            seed,
            cluster_step,
            device_type,
            transport_mode,
            model_name,
            seq_len,
            grad_sync_every,
            max_gpu_snapshot_gb,
        ),
        daemon=False,
    )
    p.start()
    return p


def chaos_monkey(
    procs: Dict[int, mp.Process],
    procs_lock,
    stop_event,
    cluster_step,  # mp.Value('i')
    crash_log: List[Tuple[int, int]],
    min_step_interval: int,
    max_step_interval: int,
    log: logging.Logger,
):
    """Kill a random worker every [min_step_interval, max_step_interval] cluster
    steps. The supervisor thread is responsible for respawning dead workers.
    """
    log.info(
        "chaos monkey: armed; first kill once cluster reaches step %d",
        min_step_interval,
    )
    last_diag = 0.0

    def diag(msg: str) -> None:
        nonlocal last_diag
        now = time.time()
        if now - last_diag >= 3.0:
            log.info("chaos monkey: %s", msg)
            last_diag = now

    # Wait for first eligible kill.
    while not stop_event.is_set():
        cs = cluster_step.value
        if cs >= min_step_interval:
            break
        diag(f"waiting for first kill (cluster_step={cs} target={min_step_interval})")
        if stop_event.wait(0.25):
            return

    next_kill_step = cluster_step.value + random.randint(
        min_step_interval, max_step_interval
    )
    log.info(
        "chaos monkey: scheduled first kill at cluster_step=%d (now=%d)",
        next_kill_step,
        cluster_step.value,
    )

    while not stop_event.is_set():
        # Wait for cluster step to reach next_kill_step.
        while not stop_event.is_set():
            cs = cluster_step.value
            if cs >= next_kill_step:
                break
            diag(f"awaiting kill at step={next_kill_step} (now={cs})")
            if stop_event.wait(0.25):
                return

        with procs_lock:
            alive_ids = [rid for rid, p in procs.items() if p.is_alive()]
        if len(alive_ids) <= 2:
            log.warning(
                "chaos monkey: only %d alive replicas — postponing kill at step %d",
                len(alive_ids),
                cluster_step.value,
            )
            next_kill_step = cluster_step.value + random.randint(
                min_step_interval, max_step_interval
            )
            continue

        victim_id = random.choice(alive_ids)
        with procs_lock:
            victim = procs[victim_id]
        kill_step = cluster_step.value
        log.warning(
            "CHAOS: killing replica %d (pid=%d) at cluster_step=%d",
            victim_id,
            victim.pid,
            kill_step,
        )
        crash_log.append((kill_step, victim_id))
        try:
            victim.kill()
            victim.join(timeout=10)
        except Exception as e:
            log.exception("chaos monkey: failed to kill %d: %r", victim_id, e)

        next_kill_step = cluster_step.value + random.randint(
            min_step_interval, max_step_interval
        )
        log.info(
            "chaos monkey: next kill scheduled at cluster_step=%d (now=%d)",
            next_kill_step,
            cluster_step.value,
        )


def supervisor(
    procs: Dict[int, mp.Process],
    procs_lock,
    spawn_args,
    stop_event,
    restart_delay_min: float,
    restart_delay_max: float,
    cascade_log: List[Tuple[float, int]],
    log: logging.Logger,
):
    """Watch for dead workers (chaos victims OR cascade failures) and respawn
    them after a short delay. With min_replica_size=2 and 4 replicas, the
    cluster keeps committing while a peer is rebooting, then the rebooted
    peer heals from the cluster via Manager's checkpoint transport.
    """
    last_pids: Dict[int, int] = {}
    while not stop_event.is_set():
        with procs_lock:
            snapshot = list(procs.items())
        for rid, p in snapshot:
            if not p.is_alive() and last_pids.get(rid) != p.pid:
                last_pids[rid] = p.pid
                exit_code = p.exitcode
                cascade_log.append((time.time(), rid))
                delay = random.uniform(restart_delay_min, restart_delay_max)
                log.warning(
                    "SUPERVISOR: replica %d dead (pid=%d, exit=%s); "
                    "respawning in %.1fs",
                    rid,
                    p.pid,
                    exit_code,
                    delay,
                )
                if stop_event.wait(delay):
                    return
                new_proc = spawn_worker(*spawn_args(rid, healed=True))
                with procs_lock:
                    procs[rid] = new_proc
                # Intentionally leave last_pids[rid] pointing at the OLD (dead)
                # pid. If we set it to new_proc.pid here, then when chaos kills
                # *this* freshly-respawned worker later, procs[rid].pid will
                # equal last_pids[rid] and the `last_pids.get(rid) != p.pid`
                # guard above will skip the respawn — leaving the replica dead
                # forever. The first `last_pids[rid] = p.pid` above is what
                # prevents double-handling the same dead process.
                log.warning(
                    "SUPERVISOR: replica %d respawned as pid=%d",
                    rid,
                    new_proc.pid,
                )
        if stop_event.wait(0.5):
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-replicas", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="device for model + checkpoint staging; 'cuda' puts replica i on cuda:i",
    )
    parser.add_argument(
        "--transport",
        choices=["rdma", "http"],
        default="rdma",
        help="checkpoint/recovery transport: rdma (torchcomms RDMA) or http",
    )
    parser.add_argument(
        "--model",
        choices=["smallcnn", "resnet50", "llama_1b", "llama_7b"],
        default="smallcnn",
        help="model to train; resnet50 (~196 MB) and llama_7b (~27 GB bf16 "
        "checkpoint) make the RDMA-GPUDirect vs HTTP transfer cost measurable",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="sequence length for the LM task (llama_* models)",
    )
    parser.add_argument(
        "--grad-sync-every",
        type=int,
        default=1,
        help="cross-replica grad-average interval. 1 = every step (vision "
        "default). For large LMs the grad allreduce goes through CPU/Gloo and "
        "is huge, so use 0 (never; weight-averaging off — transfer benchmark) "
        "or a large N. A tiny proxy collective runs on non-sync steps so the "
        "Manager still detects failures and heals.",
    )
    parser.add_argument(
        "--max-gpu-snapshot-gb",
        type=float,
        default=4.0,
        help="max GPU-resident checkpoint snapshot before spilling to pinned "
        "CPU. Raise it (e.g. 40) for big LM checkpoints so the RDMA read stays "
        "GPU->GPU (GPUDirect) instead of CPU-staged",
    )
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-chaos", action="store_true")
    parser.add_argument(
        "--chaos-min-steps",
        type=int,
        default=20,
        help="min step delta between chaos kills",
    )
    parser.add_argument(
        "--chaos-max-steps",
        type=int,
        default=40,
        help="max step delta between chaos kills",
    )
    parser.add_argument("--restart-delay-min", type=float, default=1.0)
    parser.add_argument("--restart-delay-max", type=float, default=3.0)
    parser.add_argument(
        "--lighthouse-min-replicas",
        type=int,
        default=2,
        help="min replicas for the lighthouse to form a quorum",
    )
    parser.add_argument(
        "--manager-min-replica-size",
        type=int,
        default=2,
        help="min replica size required for the manager to commit",
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=0.30,
        help="success threshold for final test accuracy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="base seed for chaos timing and worker RNGs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    log = logging.getLogger("main")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # ------------------------------------------------------------------
    # Lighthouse (the rendezvous server). min_replicas determines the
    # quorum size. With 4 replicas and chaos killing one at a time, a
    # quorum of 2 lets training keep going while a peer is healing.
    # ------------------------------------------------------------------
    from torchft._torchft import LighthouseServer

    lighthouse = LighthouseServer(
        bind="[::]:0",
        min_replicas=args.lighthouse_min_replicas,
        join_timeout_ms=10_000,
        quorum_tick_ms=200,
        heartbeat_timeout_ms=5_000,
    )
    log.info(
        "lighthouse up at %s (min_replicas=%d, num_replicas=%d)",
        lighthouse.address(),
        args.lighthouse_min_replicas,
        args.num_replicas,
    )

    event_q: mp.Queue = mp.Queue()
    procs: Dict[int, mp.Process] = {}
    procs_lock = mp.Lock()
    crash_log: List[Tuple[int, int]] = []

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cluster_step = mp.Value("i", 0)

    def spawn_args(rid: int, healed: bool) -> Tuple:
        return (
            rid,
            args.num_replicas,
            lighthouse.address(),
            args.total_steps,
            args.batch_size,
            args.eval_every,
            args.log_every,
            event_q,
            healed,
            repo_root,
            args.manager_min_replica_size,
            args.seed,
            cluster_step,
            args.device,
            args.transport,
            args.model,
            args.seq_len,
            args.grad_sync_every,
            args.max_gpu_snapshot_gb,
        )

    for rid in range(args.num_replicas):
        procs[rid] = spawn_worker(*spawn_args(rid, healed=False))
        log.info("spawned replica %d as pid=%d", rid, procs[rid].pid)

    # ------------------------------------------------------------------
    # Chaos monkey + supervisor threads
    # ------------------------------------------------------------------
    import threading

    stop_chaos = threading.Event()
    cascade_log: List[Tuple[float, int]] = []
    chaos_thread = None
    supervisor_thread = threading.Thread(
        target=supervisor,
        name="supervisor",
        args=(
            procs,
            procs_lock,
            spawn_args,
            stop_chaos,
            args.restart_delay_min,
            args.restart_delay_max,
            cascade_log,
            log,
        ),
        daemon=True,
    )
    supervisor_thread.start()

    if not args.no_chaos:
        chaos_thread = threading.Thread(
            target=chaos_monkey,
            name="chaos-monkey",
            args=(
                procs,
                procs_lock,
                stop_chaos,
                cluster_step,
                crash_log,
                args.chaos_min_steps,
                args.chaos_max_steps,
                log,
            ),
            daemon=True,
        )
        chaos_thread.start()
        log.info(
            "chaos monkey armed: kill every %d-%d cluster steps",
            args.chaos_min_steps,
            args.chaos_max_steps,
        )
    else:
        log.info("chaos monkey DISABLED")

    # ------------------------------------------------------------------
    # Drain events; track per-replica progress and bail out when the
    # cluster has crossed total_steps.
    # ------------------------------------------------------------------
    losses_by_replica: Dict[int, float] = {}
    accs_by_replica: Dict[int, float] = {}
    last_steps: Dict[int, int] = {}
    loss_history: List[Tuple[int, int, float]] = []  # (step, replica, loss)
    eval_history: List[Tuple[int, int, float]] = []  # (step, replica, acc)
    final_step_by_replica: Dict[int, int] = {}
    heal_count = 0
    error_count = 0
    ready_count = 0
    heal_secs_list: List[float] = []  # end-to-end recovery latencies (s)
    rdma_active_seen = None  # set from EVT_READY: did workers use real RDMA?
    last_print = time.time()

    def all_dead() -> bool:
        with procs_lock:
            return all(not p.is_alive() for p in procs.values())

    try:
        while True:
            try:
                ev = event_q.get(timeout=2.0)
            except Exception:
                ev = None

            if ev is not None:
                t = ev["type"]
                rid = ev["replica"]
                if t == EVT_STEP:
                    last_steps[rid] = ev["step"]
                    losses_by_replica[rid] = ev["loss"]
                    loss_history.append((ev["step"], rid, ev["loss"]))
                    if time.time() - last_print > 4.0:
                        log.info(
                            "progress: cluster_step=%d | %s",
                            cluster_step.value,
                            " | ".join(
                                f"r{r}=step{last_steps.get(r, 0)} "
                                f"loss={losses_by_replica.get(r, float('nan')):.3f}"
                                for r in sorted(procs.keys())
                            ),
                        )
                        last_print = time.time()
                elif t == EVT_EVAL:
                    accs_by_replica[rid] = ev["acc"]
                    eval_history.append((ev["step"], rid, ev["acc"]))
                    log.info(
                        "[r%d step %d] EVAL acc=%.3f loss=%.3f",
                        rid,
                        ev["step"],
                        ev["acc"],
                        ev["loss"],
                    )
                elif t == EVT_HEAL:
                    heal_count += 1
                    secs = ev.get("heal_secs", float("nan"))
                    if secs == secs and secs > 0:  # not NaN
                        heal_secs_list.append(secs)
                    log.warning(
                        "HEAL: replica %d restored from peer at step=%s "
                        "via %s in %.3fs",
                        rid,
                        ev.get("step", "unknown"),
                        ev.get("transport", "?"),
                        secs,
                    )
                elif t == EVT_READY:
                    ready_count += 1
                    if ev.get("rdma_active") is not None:
                        rdma_active_seen = ev.get("rdma_active")
                    log.info(
                        "READY: replica %d (healed=%s, transport=%s, rdma_active=%s)"
                        " — total ready events: %d",
                        rid,
                        ev.get("healed"),
                        ev.get("transport"),
                        ev.get("rdma_active"),
                        ready_count,
                    )
                elif t == EVT_DONE:
                    final_step_by_replica[rid] = ev["step"]
                    if not (ev["acc"] != ev["acc"]):  # not NaN
                        accs_by_replica[rid] = ev["acc"]
                    losses_by_replica[rid] = ev["loss"]
                    log.info(
                        "DONE: replica %d final_loss=%.3f final_acc=%.3f step=%d",
                        rid,
                        ev["loss"],
                        ev["acc"],
                        ev["step"],
                    )
                elif t == EVT_ERROR:
                    error_count += 1
                    log.error("ERROR from replica %d: %s", rid, ev["error"])

            if cluster_step.value >= args.total_steps:
                log.info(
                    "cluster reached step %d (>= %d), stopping",
                    cluster_step.value,
                    args.total_steps,
                )
                break
            if all_dead():
                log.error("all worker processes are dead — aborting")
                break
    finally:
        stop_chaos.set()
        if chaos_thread is not None:
            chaos_thread.join(timeout=5)
        supervisor_thread.join(timeout=5)

        # Reap workers.
        with procs_lock:
            for rid, p in procs.items():
                if p.is_alive():
                    log.info("terminating replica %d (pid=%d)", rid, p.pid)
                    p.terminate()
                    p.join(timeout=10)
                    if p.is_alive():
                        log.warning("replica %d still alive — killing", rid)
                        p.kill()
                        p.join(timeout=5)

        # Drain any final events that arrived during shutdown so the
        # summary picks them up.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                ev = event_q.get(timeout=0.2)
            except Exception:
                break
            t = ev["type"]
            rid = ev["replica"]
            if t == EVT_DONE:
                final_step_by_replica[rid] = ev["step"]
                if not (ev["acc"] != ev["acc"]):
                    accs_by_replica[rid] = ev["acc"]
                losses_by_replica[rid] = ev["loss"]
            elif t == EVT_EVAL:
                accs_by_replica[rid] = ev["acc"]
                eval_history.append((ev["step"], rid, ev["acc"]))
            elif t == EVT_HEAL:
                heal_count += 1

        try:
            lighthouse.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    final_loss = (
        sum(losses_by_replica.values()) / len(losses_by_replica)
        if losses_by_replica
        else float("nan")
    )
    latest_eval_step = max((step for step, _, _ in eval_history), default=None)
    first_eval_step = min((step for step, _, _ in eval_history), default=None)
    starting_acc = (
        max(acc for step, _, acc in eval_history if step == first_eval_step)
        if first_eval_step is not None
        else None
    )
    latest_eval_acc = (
        max(acc for step, _, acc in eval_history if step == latest_eval_step)
        if latest_eval_step is not None
        else (max(accs_by_replica.values()) if accs_by_replica else float("nan"))
    )
    best_acc = (
        max((acc for _, _, acc in eval_history), default=float("nan"))
        if eval_history
        else (max(accs_by_replica.values()) if accs_by_replica else float("nan"))
    )
    chaos_crashes = len(crash_log)
    cascade_respawns = max(0, len(cascade_log) - chaos_crashes)
    final_steps = cluster_step.value
    latest_acc_valid = latest_eval_acc == latest_eval_acc  # not NaN
    improved = (
        latest_acc_valid
        and (
            (starting_acc is not None and latest_eval_acc > starting_acc + 0.05)
            or latest_eval_acc >= args.target_acc
        )
    )
    min_expected_kills = (
        0
        if args.no_chaos
        else max(3, min(5, args.total_steps // max(1, args.chaos_max_steps)))
    )
    enough_chaos = chaos_crashes >= min_expected_kills

    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"seed:                  {args.seed}")
    print(f"replicas:              {args.num_replicas}")
    print(f"requested steps:       {args.total_steps}")
    print(f"max cluster steps:     {final_steps}")
    print(f"final train loss:      {final_loss:.4f}")
    if starting_acc is not None:
        print(f"first eval acc:        {starting_acc:.4f}")
    else:
        print("first eval acc:        N/A")
    if latest_eval_step is not None:
        print(f"latest eval step:      {latest_eval_step}")
    else:
        print("latest eval step:      N/A")
    print(f"latest eval acc:       {latest_eval_acc:.4f}")
    print(f"best test accuracy:    {best_acc:.4f}")
    print(f"device:                {args.device}")
    print(f"model:                 {args.model}")
    print(f"transport:             {args.transport}")
    print(f"rdma active:           {rdma_active_seen}")
    print(f"chaos kills:           {chaos_crashes}")
    print(f"cascade respawns:      {cascade_respawns}")
    print(f"successful heals:      {heal_count}")
    if heal_secs_list:
        import statistics as _stats
        _mean = _stats.mean(heal_secs_list)
        _median = _stats.median(heal_secs_list)
        _mn, _mx = min(heal_secs_list), max(heal_secs_list)
        print(
            f"recovery time (s):     mean={_mean:.3f} median={_median:.3f} "
            f"min={_mn:.3f} max={_mx:.3f}  (n={len(heal_secs_list)})"
        )
        print(f"recovery samples (s):  {[round(s, 3) for s in heal_secs_list]}")
    else:
        print("recovery time (s):     no timed heals recorded")
    print(f"transient errors:      {error_count}")
    print(f"accuracy improved:     {improved}")
    print(f"enough chaos events:   {enough_chaos} (need >= {min_expected_kills})")
    print("-" * 70)
    print("per-replica final step (covers chaos kills & restarts):")
    for rid in sorted(procs.keys()):
        step = final_step_by_replica.get(rid, last_steps.get(rid, 0))
        loss = losses_by_replica.get(rid, float("nan"))
        acc = accs_by_replica.get(rid, float("nan"))
        print(
            f"  replica {rid}: step={step:4d}  loss={loss:.3f}  acc={acc:.3f}"
        )
    if eval_history:
        print("-" * 70)
        print("eval history (step, replica, acc):")
        for step, rid, acc in sorted(eval_history):
            print(f"  step {step:4d}  r{rid}  acc={acc:.3f}")
    if loss_history:
        print("-" * 70)
        print("loss trajectory (step, replica, loss):")
        for step, rid, loss in sorted(loss_history):
            print(f"  step {step:4d}  r{rid}  loss={loss:.3f}")
    if crash_log:
        print("-" * 70)
        print("chaos crash log (cluster_step, victim_replica):")
        for step, rid in crash_log:
            print(f"  step {step:4d}  r{rid}")
    print("-" * 70)

    reached_target = final_steps >= int(args.total_steps * 0.9)
    success = (
        improved
        and reached_target
        and enough_chaos
        and (chaos_crashes == 0 or heal_count >= max(1, chaos_crashes // 2))
    )
    verdict = "SUCCESS" if success else "FAIL"
    print(f"VERDICT:               {verdict}")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
