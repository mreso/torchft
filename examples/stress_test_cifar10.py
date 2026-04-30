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


def make_loaders(batch_size: int, replica_id: int, num_replicas: int):
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


def evaluate(model: nn.Module, loader: DataLoader, max_batches: int = 8) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
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

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    log = logging.getLogger(f"replica-{replica_id}")
    random.seed(seed + replica_id)
    torch.manual_seed(seed + replica_id)

    device = torch.device("cpu")

    # Per-worker TCPStore so dist init in Manager has somewhere to publish.
    store = dist.TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )

    model = SmallCNN().to(device)
    base_opt = optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    def state_dict() -> Dict[str, object]:
        return {
            "model": model.state_dict(),
            "optim": base_opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

    def load_state_dict(sd: Dict[str, object]) -> None:
        model.load_state_dict(sd["model"])
        base_opt.load_state_dict(sd["optim"])
        scheduler.load_state_dict(sd["scheduler"])
        log.info("loaded state from peer (model + optim + scheduler restored)")
        try:
            event_q.put({"type": EVT_HEAL, "replica": replica_id, "step": manager.current_step()})
        except Exception:
            pass

    pg = ProcessGroupGloo(timeout=timedelta(seconds=30))
    transport = RDMATransport(device=device, timeout=timedelta(seconds=30))

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
        event_q.put({"type": EVT_READY, "replica": replica_id, "healed": healed})
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
        batch_size, replica_id, num_replicas
    )
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
                    # Flatten all grads into one buffer so this is a single
                    # collective per step instead of one per parameter — much
                    # cheaper on CPU/Gloo and keeps step time low enough that
                    # chaos timing stays meaningful.
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if grads:
                        flat = torch.cat([g.detach().view(-1) for g in grads])
                        work = manager.allreduce(flat)
                        work.wait()
                        offset = 0
                        for g in grads:
                            n = g.numel()
                            g.copy_(flat[offset : offset + n].view_as(g))
                            offset += n
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
                    acc = evaluate(model, test_loader)
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
            final_acc = evaluate(model, test_loader)
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
                    log.warning(
                        "HEAL: replica %d restored from peer at step=%s",
                        rid,
                        ev.get("step", "unknown"),
                    )
                elif t == EVT_READY:
                    ready_count += 1
                    log.info(
                        "READY: replica %d (healed=%s) — total ready events: %d",
                        rid,
                        ev.get("healed"),
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
    print(f"chaos kills:           {chaos_crashes}")
    print(f"cascade respawns:      {cascade_respawns}")
    print(f"successful heals:      {heal_count}")
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
