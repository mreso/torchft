# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RAID5 Global FSDP Training Example
====================================

Demonstrates global FSDP training with RAID5-style fault tolerance.
FSDP spans ALL workers (one big sharding group), with XOR parity
protecting against single-GPU failures. Each worker is a single GPU,
launched independently (no torchrun), and registered as a separate
torchft "replica" for health monitoring.

Three process groups per worker:
  1. FSDP PG (NCCL): from dist.init_process_group("nccl"), handles
     all-gather (forward) and reduce-scatter (backward).
  2. Manager PG (torchft ProcessGroupGloo): reconfigurable, for quorum
     coordination (start_quorum, should_commit).
  3. RAID5 PG (torchft ProcessGroupNCCL): reconfigurable, for NCCL
     allgather in parity computation.

Launch:
  Terminal 1 (Lighthouse):
    torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000

  Terminal 2-5 (Workers 0-3, one per GPU):
    MASTER_ADDR=localhost MASTER_PORT=29500 RANK=0 WORLD_SIZE=4 \\
      CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 \\
      python train_raid5_fsdp.py

    MASTER_ADDR=localhost MASTER_PORT=29500 RANK=1 WORLD_SIZE=4 \\
      CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 \\
      python train_raid5_fsdp.py

    MASTER_ADDR=localhost MASTER_PORT=29500 RANK=2 WORLD_SIZE=4 \\
      CUDA_VISIBLE_DEVICES=2 TORCHFT_LIGHTHOUSE=http://localhost:29510 \\
      python train_raid5_fsdp.py

    MASTER_ADDR=localhost MASTER_PORT=29500 RANK=3 WORLD_SIZE=4 \\
      CUDA_VISIBLE_DEVICES=3 TORCHFT_LIGHTHOUSE=http://localhost:29510 \\
      python train_raid5_fsdp.py
"""

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import init_device_mesh
from torchdata.stateful_dataloader import StatefulDataLoader
from torchft import DistributedSampler, Manager, Optimizer, ProcessGroupGloo, ProcessGroupNCCL
from torchft.raid import RAID5FSDP

logging.basicConfig(level=logging.INFO)


def main() -> None:
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

    # Each worker sees one GPU via CUDA_VISIBLE_DEVICES, mapped to device 0
    torch.cuda.set_device(0)

    # --- Dataset ---
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar", train=True, download=True, transform=transform
    )

    # Each worker gets a different data shard (no replica groups)
    sampler = DistributedSampler(
        trainset,
        replica_rank=RANK,
        num_replica_groups=WORLD_SIZE,
        group_rank=0,
        num_replicas=1,
        shuffle=True,
    )

    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )

    # --- Model ---
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.cnn(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # --- Initialize default PG for FSDP ---
    dist.init_process_group("nccl")

    # FSDP mesh spanning all workers
    fsdp_mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("dp_shard",))

    device = "cuda"
    m = Net().to(device)

    # FSDP shards model across all workers — no set_all_reduce_hook needed
    fully_shard(m, mesh=fsdp_mesh)

    raw_optimizer = optim.AdamW(m.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint callbacks ---
    def load_state_dict(state_dict: dict) -> None:
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optim"])

    def state_dict() -> dict:
        return {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }

    # --- Manager PG (torchft Gloo, reconfigurable via quorum) ---
    manager_pg = ProcessGroupGloo(timeout=timedelta(seconds=30))

    # Each worker is its own torchft "replica" (group of size 1)
    manager = Manager(
        pg=manager_pg,
        min_replica_size=1,
        use_async_quorum=False,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"raid5_fsdp_{RANK}",
        timeout=timedelta(seconds=30),
        rank=0,  # each worker is rank 0 in its 1-worker group
        world_size=1,
        init_sync=False,  # different FSDP shards, can't sync weights
    )
    optimizer = Optimizer(manager, raw_optimizer)

    # --- RAID5 PG (torchft ProcessGroupNCCL, reconfigurable) ---
    raid5_pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    raid5_pg.configure(
        store_addr=f"localhost:{MASTER_PORT}/raid5_parity",
        replica_id=str(RANK),
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    print(m)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"[Rank {RANK}] Total number of parameters: {num_params}")

    # --- Training loop with RAID5 context ---
    with RAID5FSDP(manager, m, raw_optimizer, fsdp_mesh, raid5_pg):
        while True:
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Quorum computation is triggered via the torchft Optimizer wrapper
                optimizer.zero_grad()

                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()

                # The torchft Optimizer calls should_commit() and conditionally
                # steps the raw optimizer, which triggers RAID5 parity hooks.
                optimizer.step()

                step = manager.current_step()
                if step % 10 == 0 or step == 1:
                    print(
                        f"[Rank {RANK}, step {step}] loss = {loss.item()}",
                        flush=True,
                    )

                if manager.current_step() >= 10000:
                    return


if __name__ == "__main__":
    main()
