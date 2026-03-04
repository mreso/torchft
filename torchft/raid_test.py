# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from typing import List, Tuple

import torch
from torch.distributed import TCPStore

from torchft.process_group import ProcessGroupGloo
from torchft.raid import RAID5FSDP, ParityManager, StateFlattener


class TestStateFlattener(unittest.TestCase):
    def test_roundtrip_single_dtype(self) -> None:
        """Flatten and unflatten float32 tensors, verify bit-exact equality."""
        flattener = StateFlattener()
        tensors = [
            torch.randn(3, 4),
            torch.randn(5),
            torch.randn(2, 3, 4),
        ]
        flat = flattener.flatten(tensors)
        self.assertEqual(flat.dtype, torch.uint8)
        self.assertTrue(flat.is_contiguous())

        recovered = flattener.unflatten(flat)
        self.assertEqual(len(recovered), len(tensors))
        for orig, rec in zip(tensors, recovered):
            self.assertEqual(orig.shape, rec.shape)
            self.assertEqual(orig.dtype, rec.dtype)
            self.assertTrue(torch.equal(orig, rec))

    def test_roundtrip_mixed_dtype(self) -> None:
        """Flatten mixed-dtype tensors, verify bit-exact equality."""
        flattener = StateFlattener()
        tensors = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randint(0, 100, (5, 2), dtype=torch.int64),
            torch.randn(2, 3, dtype=torch.float16),
            torch.tensor([True, False, True], dtype=torch.bool),
        ]
        flat = flattener.flatten(tensors)
        recovered = flattener.unflatten(flat)

        self.assertEqual(len(recovered), len(tensors))
        for orig, rec in zip(tensors, recovered):
            self.assertEqual(orig.shape, rec.shape)
            self.assertEqual(orig.dtype, rec.dtype)
            self.assertTrue(torch.equal(orig, rec))

    def test_empty(self) -> None:
        """Flatten empty tensor list."""
        flattener = StateFlattener()
        flat = flattener.flatten([])
        self.assertEqual(flat.numel(), 0)
        recovered = flattener.unflatten(flat)
        self.assertEqual(len(recovered), 0)

    def test_single_element(self) -> None:
        """Flatten a single scalar tensor."""
        flattener = StateFlattener()
        t = torch.tensor(42.0)
        flat = flattener.flatten([t])
        recovered = flattener.unflatten(flat)
        self.assertEqual(len(recovered), 1)
        self.assertTrue(torch.equal(t, recovered[0]))


def _run_parity_worker(
    rank: int,
    world_size: int,
    store_port: int,
    all_data: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Worker function for parity tests. Returns (primary_stripe, secondary_stripe, stripe_size)."""
    store_addr = f"localhost:{store_port}/parity_test"

    pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg.configure(store_addr, "0", rank, world_size)

    local_flat = all_data[rank].clone()

    pm = ParityManager(rank, world_size, torch.device("cpu"))
    pm.compute_parity(local_flat, pg)

    assert pm.primary_stripe is not None
    assert pm.secondary_stripe is not None
    return pm.primary_stripe.clone(), pm.secondary_stripe.clone(), pm.stripe_size


def _run_parity_and_reconstruct_worker(
    rank: int,
    world_size: int,
    store_port: int,
    all_data: List[torch.Tensor],
    failed_rank: int,
    run_id: int,
) -> torch.Tensor:
    """
    Worker that computes parity, then reconstructs the failed rank's data.

    All ranks participate in Phase 1 (parity computation). Only surviving
    ranks participate in Phase 2 (reconstruction).
    """
    store_addr = f"localhost:{store_port}/parity_test_{run_id}"

    # Phase 1: compute parity with all ranks
    pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg.configure(store_addr, "0", rank, world_size)

    local_flat = all_data[rank].clone()
    pm = ParityManager(rank, world_size, torch.device("cpu"))
    pm.compute_parity(local_flat, pg)

    if rank == failed_rank:
        # The failed rank doesn't participate in reconstruction
        return torch.tensor([])

    # Phase 2: reconstruct with surviving ranks only
    surviving_ranks = [r for r in range(world_size) if r != failed_rank]
    new_rank = surviving_ranks.index(rank)
    new_world_size = len(surviving_ranks)

    store_addr2 = f"localhost:{store_port}/reconstruct_test_{run_id}"
    pg2 = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg2.configure(store_addr2, "0", new_rank, new_world_size)

    # Update ParityManager's rank for the new PG
    pm._rank = new_rank

    reconstructed = pm.reconstruct(failed_rank, local_flat, pg2)
    return reconstructed


class TestParityManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

    def test_parity_compute(self) -> None:
        """Verify parity computation produces correct stripes with 2/N overhead."""
        world_size = 3
        data_size = 100  # bytes

        all_data = [
            torch.randint(0, 256, (data_size,), dtype=torch.uint8)
            for _ in range(world_size)
        ]

        with ThreadPoolExecutor(max_workers=world_size) as executor:
            futures = []
            for rank in range(world_size):
                futures.append(
                    executor.submit(
                        _run_parity_worker,
                        rank,
                        world_size,
                        self.store.port,
                        all_data,
                    )
                )

            results = [f.result() for f in futures]

        stripe_size = math.ceil(data_size / world_size)
        padded_size = stripe_size * world_size

        # Compute expected XOR(all ranks) with padding
        xor_all = torch.zeros(padded_size, dtype=torch.uint8)
        for d in all_data:
            padded = torch.zeros(padded_size, dtype=torch.uint8)
            padded[:data_size] = d
            xor_all ^= padded

        for rank in range(world_size):
            pri, sec, ss = results[rank]
            self.assertEqual(ss, stripe_size)
            # Each stripe should have size stripe_size
            self.assertEqual(pri.numel(), stripe_size)
            self.assertEqual(sec.numel(), stripe_size)

            # Primary stripe = XOR(all) at stripe index 'rank'
            pri_start = rank * stripe_size
            expected_pri = xor_all[pri_start : pri_start + stripe_size]
            self.assertTrue(
                torch.equal(pri, expected_pri),
                f"Rank {rank} primary stripe mismatch",
            )

            # Secondary stripe = XOR(all) at stripe index '(rank+1) % N'
            sec_idx = (rank + 1) % world_size
            sec_start = sec_idx * stripe_size
            expected_sec = xor_all[sec_start : sec_start + stripe_size]
            self.assertTrue(
                torch.equal(sec, expected_sec),
                f"Rank {rank} secondary stripe mismatch",
            )

    def test_parity_memory_overhead(self) -> None:
        """Verify parity storage is 2/N of data size, not 100%."""
        world_size = 8
        data_size = 1000

        stripe_size = math.ceil(data_size / world_size)
        expected_per_rank = 2 * stripe_size  # primary + secondary

        # Should be ~25% for 8 GPUs (250 bytes vs 1000 bytes)
        self.assertLess(expected_per_rank, data_size)
        # For world_size=8, 2/8 = 25%
        self.assertAlmostEqual(expected_per_rank / data_size, 2 / world_size, places=1)

    def test_parity_roundtrip_reconstruct(self) -> None:
        """
        Compute parity with N ranks, simulate failure of one rank,
        reconstruct, and verify the result matches the original data.
        """
        world_size = 4
        data_size = 200  # bytes

        all_data = [
            torch.randint(0, 256, (data_size,), dtype=torch.uint8)
            for _ in range(world_size)
        ]

        # Test reconstruction for each possible failed rank
        for failed_rank in range(world_size):
            with self.subTest(failed_rank=failed_rank):
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=world_size, mp_context=context
                ) as executor:
                    futures = []
                    for rank in range(world_size):
                        futures.append(
                            executor.submit(
                                _run_parity_and_reconstruct_worker,
                                rank,
                                world_size,
                                self.store.port,
                                all_data,
                                failed_rank,
                                failed_rank,  # run_id to disambiguate store prefixes
                            )
                        )

                    results = [f.result() for f in futures]

                # All surviving ranks should produce the same reconstruction
                # (the failed rank returns empty tensor)
                expected = all_data[failed_rank]
                for rank, result in enumerate(results):
                    if rank == failed_rank:
                        self.assertEqual(result.numel(), 0)
                    else:
                        self.assertTrue(
                            torch.equal(result, expected),
                            f"Rank {rank} reconstruction mismatch for failed_rank={failed_rank}",
                        )

    def test_parity_small_buffer(self) -> None:
        """Test parity with a very small buffer (smaller than world_size)."""
        world_size = 3
        data_size = 2  # fewer bytes than ranks

        all_data = [
            torch.randint(0, 256, (data_size,), dtype=torch.uint8)
            for _ in range(world_size)
        ]

        failed_rank = 1
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=world_size, mp_context=context
        ) as executor:
            futures = []
            for rank in range(world_size):
                futures.append(
                    executor.submit(
                        _run_parity_and_reconstruct_worker,
                        rank,
                        world_size,
                        self.store.port,
                        all_data,
                        failed_rank,
                        100,  # run_id
                    )
                )
            results = [f.result() for f in futures]

        expected = all_data[failed_rank]
        for rank, result in enumerate(results):
            if rank != failed_rank:
                self.assertTrue(
                    torch.equal(result, expected),
                    f"Rank {rank} reconstruction mismatch",
                )


def _run_raid5_fsdp_worker(
    rank: int,
    world_size: int,
    store_port: int,
) -> bool:
    """
    Worker for RAID5FSDP integration test.
    Sets up a simple model, optimizer, runs a few steps with RAID5,
    and verifies parity is computed.
    """
    import os
    from datetime import timedelta
    from unittest.mock import Mock

    import torch
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

    from torchft.process_group import ProcessGroupGloo
    from torchft.raid import RAID5FSDP

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port + 100)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    store_addr = f"localhost:{store_port}/raid5_test"

    pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg.configure(store_addr, "0", rank, world_size)

    # Simple model (no actual FSDP, just testing the parity wrapper logic)
    model = nn.Linear(16, 8, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Mock manager
    manager = Mock()
    manager.start_quorum = Mock()
    manager.should_commit = Mock(return_value=True)

    # Mock DeviceMesh
    mesh = Mock(spec=DeviceMesh)
    mesh.device_type = "cpu"
    mesh.get_local_rank = Mock(return_value=rank)
    mesh.size = Mock(return_value=world_size)

    raid5 = RAID5FSDP(manager, model, optimizer, mesh, pg)

    with raid5:
        for step in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

    # Verify parity was computed
    return raid5._parity_initialized


class TestRAID5FSDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

    def test_raid5_fsdp_no_failure(self) -> None:
        """Test RAID5FSDP wrapper runs without errors and computes parity."""
        world_size = 3
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=world_size, mp_context=context
        ) as executor:
            futures = []
            for rank in range(world_size):
                futures.append(
                    executor.submit(
                        _run_raid5_fsdp_worker,
                        rank,
                        world_size,
                        self.store.port,
                    )
                )
            results = [f.result() for f in futures]

        for rank, parity_initialized in enumerate(results):
            self.assertTrue(
                parity_initialized,
                f"Rank {rank} did not initialize parity",
            )

    def test_compute_new_shard_shape(self) -> None:
        """Test shard shape computation after resharding."""
        # 4 GPUs → 3 GPUs, shard dim0 = 8, full dim0 = 32
        new_shape = RAID5FSDP._compute_new_shard_shape(
            torch.Size([8, 16]), old_world_size=4, new_world_size=3
        )
        # full_dim0 = 8 * 4 = 32, new_dim0 = ceil(32/3) = 11
        self.assertEqual(new_shape, torch.Size([11, 16]))

        # Scalar case
        new_shape = RAID5FSDP._compute_new_shard_shape(
            torch.Size([]), old_world_size=4, new_world_size=3
        )
        self.assertEqual(new_shape, torch.Size([]))

        # 1-D case
        new_shape = RAID5FSDP._compute_new_shard_shape(
            torch.Size([10]), old_world_size=4, new_world_size=3
        )
        # full_dim0 = 10 * 4 = 40, new_dim0 = ceil(40/3) = 14
        self.assertEqual(new_shape, torch.Size([14]))


if __name__ == "__main__":
    unittest.main()
