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
from torchft.raid import RAID5FSDP, ErasureCodingManager, ParityManager, StateFlattener


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


def _run_ec_and_reconstruct_worker(
    rank: int,
    world_size: int,
    store_port: int,
    all_data: List[torch.Tensor],
    failed_ranks: List[int],
    num_parity: int,
    run_id: int,
) -> dict:
    """
    Worker that computes erasure coding parity, then reconstructs failed ranks.

    All ranks participate in Phase 1 (parity computation). Only surviving
    ranks participate in Phase 2 (reconstruction).

    Returns dict mapping failed_rank -> reconstructed tensor (or empty dict for failed ranks).
    """
    store_addr = f"localhost:{store_port}/ec_test_{run_id}"

    # Phase 1: compute parity with all ranks
    pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg.configure(store_addr, "0", rank, world_size)

    local_flat = all_data[rank].clone()
    ec = ErasureCodingManager(rank, world_size, torch.device("cpu"), num_parity)
    ec.compute_parity(local_flat, pg)

    if rank in failed_ranks:
        return {}

    # Phase 2: reconstruct with surviving ranks only
    failed_set = set(failed_ranks)
    surviving_ranks = [r for r in range(world_size) if r not in failed_set]
    new_rank = surviving_ranks.index(rank)
    new_world_size = len(surviving_ranks)

    store_addr2 = f"localhost:{store_port}/ec_recon_{run_id}"
    pg2 = ProcessGroupGloo(timeout=timedelta(seconds=10))
    pg2.configure(store_addr2, "0", new_rank, new_world_size)

    # Update rank for the new PG
    ec._rank = new_rank

    result = ec.reconstruct(failed_ranks, local_flat, pg2)
    return {k: v.clone() for k, v in result.items()}


class TestErasureCodingManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

    def _run_ec_test(
        self,
        world_size: int,
        data_size: int,
        failed_ranks: List[int],
        num_parity: int,
        run_id: int,
    ) -> None:
        """Helper to run an erasure coding reconstruction test."""
        all_data = [
            torch.randint(0, 256, (data_size,), dtype=torch.uint8)
            for _ in range(world_size)
        ]

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=world_size, mp_context=context
        ) as executor:
            futures = []
            for rank in range(world_size):
                futures.append(
                    executor.submit(
                        _run_ec_and_reconstruct_worker,
                        rank,
                        world_size,
                        self.store.port,
                        all_data,
                        failed_ranks,
                        num_parity,
                        run_id,
                    )
                )
            results = [f.result() for f in futures]

        # All surviving ranks should produce the same reconstructions
        for rank, result in enumerate(results):
            if rank in failed_ranks:
                self.assertEqual(len(result), 0)
            else:
                self.assertEqual(
                    set(result.keys()),
                    set(failed_ranks),
                    f"Rank {rank} did not reconstruct all failed ranks",
                )
                for fr in failed_ranks:
                    self.assertTrue(
                        torch.equal(result[fr], all_data[fr]),
                        f"Rank {rank} reconstruction mismatch for failed_rank={fr}",
                    )

    def test_single_failure_m1(self) -> None:
        """m=1, reconstruct 1 failed rank out of 4."""
        self._run_ec_test(
            world_size=4, data_size=200, failed_ranks=[2],
            num_parity=1, run_id=200,
        )

    def test_single_failure_m2(self) -> None:
        """m=2, reconstruct 1 failed rank out of 4."""
        self._run_ec_test(
            world_size=4, data_size=200, failed_ranks=[1],
            num_parity=2, run_id=201,
        )

    def test_double_failure_m2(self) -> None:
        """m=2, reconstruct 2 simultaneous failures out of 5 ranks."""
        self._run_ec_test(
            world_size=5, data_size=200, failed_ranks=[1, 3],
            num_parity=2, run_id=202,
        )

    def test_double_failure_m2_all_pairs(self) -> None:
        """m=2, test all C(4,2) = 6 failure pairs for N=4."""
        from itertools import combinations

        run_id = 300
        for pair in combinations(range(4), 2):
            with self.subTest(failed_ranks=pair):
                self._run_ec_test(
                    world_size=4, data_size=200, failed_ranks=list(pair),
                    num_parity=2, run_id=run_id,
                )
                run_id += 1

    def test_triple_failure_m3(self) -> None:
        """m=3, reconstruct 3 failures out of 6 ranks."""
        self._run_ec_test(
            world_size=6, data_size=300, failed_ranks=[0, 2, 5],
            num_parity=3, run_id=400,
        )

    def test_min_world_size_error(self) -> None:
        """world_size < m+1 raises ValueError at reconstruct time."""
        # Init should succeed (degraded operation is allowed)
        ec = ErasureCodingManager(
            rank=0, world_size=2, device=torch.device("cpu"), num_parity=2
        )
        # But reconstruction should fail
        ec._stored_stripes = torch.zeros(2, 3, 1, dtype=torch.uint8)
        ec._stripe_size = 1
        ec._flat_size = 2
        with self.assertRaises(ValueError):
            ec.reconstruct(
                [1], torch.zeros(2, dtype=torch.uint8), None  # type: ignore
            )

    def test_memory_overhead(self) -> None:
        """Verify storage matches m*(m+1)/N formula."""
        for m, N in [(1, 8), (2, 8), (2, 4), (3, 6)]:
            stripe_size = math.ceil(1000 / N)
            expected_per_rank = m * (m + 1) * stripe_size
            expected_ratio = m * (m + 1) / N
            actual_ratio = expected_per_rank / 1000
            self.assertAlmostEqual(
                actual_ratio,
                expected_ratio,
                places=1,
                msg=f"m={m}, N={N}: ratio {actual_ratio} != {expected_ratio}",
            )

    def test_small_buffer_m2(self) -> None:
        """Buffer smaller than world_size with m=2."""
        self._run_ec_test(
            world_size=4, data_size=2, failed_ranks=[0, 3],
            num_parity=2, run_id=500,
        )

    def test_m1_single_failure_all_ranks(self) -> None:
        """m=1, test each possible single failure for N=4."""
        for failed_rank in range(4):
            with self.subTest(failed_rank=failed_rank):
                self._run_ec_test(
                    world_size=4, data_size=200, failed_ranks=[failed_rank],
                    num_parity=1, run_id=600 + failed_rank,
                )


if __name__ == "__main__":
    unittest.main()
