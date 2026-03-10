# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Erasure-Coded Fault Tolerance for FSDP
=======================================

This module implements intra-group fault tolerance for FSDP training using
erasure coding. Each GPU stores parity syndrome stripes, and when GPUs fail,
surviving GPUs reconstruct their data and reshard.

Two levels of fault tolerance are provided:

- **RAID5 / XOR parity (m=1):** single failure tolerance, 2/N overhead per GPU.
  Uses simple XOR parity (``ParityManager`` / ``RAID5FSDP``).

- **Reed-Solomon erasure coding (m>=1):** tolerate up to m simultaneous
  GPU failures using m parity syndromes over GF(2^8). Memory overhead per GPU:
  ``m*(m+1)/N`` of the protected data size. (``ErasureCodingManager`` /
  ``ErasureCodingFSDP``).
"""

import logging
import math
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import torch
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import AllgatherOptions
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle

from torchft.process_group import ProcessGroup

if TYPE_CHECKING:
    from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _TensorMeta:
    """Metadata for reconstructing a tensor from a flat uint8 buffer."""

    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: tuple[int, ...]
    nbytes: int


class StateFlattener:
    """
    Handles flattening a list of tensors into a single contiguous uint8 buffer
    and unflattening back to the original tensors.

    This operates on raw byte representations, preserving exact bit patterns
    across all dtypes.
    """

    def __init__(self) -> None:
        self._metas: List[_TensorMeta] = []

    def flatten(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten a list of tensors into a single contiguous uint8 buffer.

        Args:
            tensors: list of tensors to flatten. Must all be on the same device.

        Returns:
            A 1-D uint8 tensor containing all tensor data concatenated.
        """
        if len(tensors) == 0:
            self._metas = []
            return torch.tensor([], dtype=torch.uint8)

        device = tensors[0].device
        self._metas = []
        parts: List[torch.Tensor] = []

        for t in tensors:
            # Make contiguous so storage faithfully represents the data
            t_contig = t.contiguous()
            self._metas.append(
                _TensorMeta(
                    shape=t_contig.shape,
                    dtype=t_contig.dtype,
                    storage_offset=0,
                    stride=t_contig.stride(),
                    nbytes=t_contig.nelement() * t_contig.element_size(),
                )
            )
            # View the underlying storage as uint8
            raw = torch.tensor(
                t_contig.untyped_storage(), dtype=torch.uint8, device=device
            )
            # Slice to exactly the number of bytes for this tensor's elements
            parts.append(raw[: self._metas[-1].nbytes])

        return torch.cat(parts)

    def unflatten(self, flat: torch.Tensor) -> List[torch.Tensor]:
        """
        Reconstruct tensors from a flat uint8 buffer using stored metadata.

        Args:
            flat: the flat uint8 buffer previously produced by flatten().

        Returns:
            List of tensors with original shapes and dtypes.
        """
        if len(self._metas) == 0:
            return []

        result: List[torch.Tensor] = []
        offset = 0

        for meta in self._metas:
            raw = flat[offset : offset + meta.nbytes]
            # Reinterpret the bytes as the original dtype
            t = raw.view(meta.dtype)
            t = t.reshape(meta.shape)
            result.append(t)
            offset += meta.nbytes

        return result


class ParityManager:
    """
    Manages XOR parity across a group of GPUs for fault tolerance using a
    duplicate-stripe approach.

    After computing XOR(all ranks) via allreduce, the result is divided into
    N stripes. Each rank stores two stripes: a primary (stripe r) and a
    secondary (stripe (r+1) % N). This ensures that when any single rank
    dies, all N stripes of XOR(all ranks) remain available on surviving ranks.

    Memory overhead per GPU: ``2 * ceil(flat_size / N)`` bytes, i.e. 2/N of the
    protected data size. For 8 GPUs this is 25%.

    Reconstruction identity:
      data[f] = XOR(all ranks) ^ XOR(surviving ranks)
    """

    def __init__(self, rank: int, world_size: int, device: torch.device) -> None:
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._primary_stripe: Optional[torch.Tensor] = None
        self._secondary_stripe: Optional[torch.Tensor] = None
        self._stripe_size: int = 0
        self._flat_size: int = 0

    @property
    def primary_stripe(self) -> Optional[torch.Tensor]:
        return self._primary_stripe

    @property
    def secondary_stripe(self) -> Optional[torch.Tensor]:
        return self._secondary_stripe

    @property
    def stripe_size(self) -> int:
        return self._stripe_size

    def compute_parity(self, local_flat: torch.Tensor, pg: ProcessGroup) -> None:
        """
        Compute XOR parity across all ranks.

        Uses allgather to collect all ranks' data, then computes XOR locally
        on GPU. Each rank stores two stripes of the global XOR(all ranks):
        a primary stripe at index ``rank`` and a secondary stripe at index
        ``(rank + 1) % world_size``.

        Communication cost: one allgather of size flat_size × world_size.

        Args:
            local_flat: this rank's flat uint8 data buffer.
            pg: the process group to use for communication.
        """
        assert local_flat.dtype == torch.uint8

        self._flat_size = local_flat.numel()
        self._stripe_size = math.ceil(self._flat_size / self._world_size)
        padded_size = self._stripe_size * self._world_size

        # Pad local data to uniform size
        buf = torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
        buf[: self._flat_size].copy_(local_flat)

        # Allgather all ranks' data
        gathered = [
            torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
            for _ in range(self._world_size)
        ]
        work = pg.allgather([gathered], [buf], AllgatherOptions())
        work.wait()

        # XOR all gathered buffers locally
        xor_all = torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
        for g in gathered:
            xor_all ^= g

        # Store primary stripe (index = rank) and secondary stripe (index = (rank+1) % N)
        p_start = self._rank * self._stripe_size
        self._primary_stripe = xor_all[p_start : p_start + self._stripe_size].clone()

        s_start = ((self._rank + 1) % self._world_size) * self._stripe_size
        self._secondary_stripe = xor_all[s_start : s_start + self._stripe_size].clone()

    def reconstruct(
        self,
        failed_rank: int,
        local_flat: torch.Tensor,
        pg: ProcessGroup,
    ) -> torch.Tensor:
        """
        Reconstruct the failed rank's full flat buffer from surviving data + parity.

        Must be called only on surviving ranks (not the failed rank).

        Algorithm:
        1. Allgather primary and secondary stripes from surviving ranks.
        2. Reassemble full XOR(all ranks) from the gathered stripes (every
           stripe has at least one surviving copy).
        3. Allgather surviving data, compute XOR(surviving) locally.
        4. data[f] = XOR(all ranks) ^ XOR(surviving).

        Communication cost: one allgather of 2*stripe_size + one allgather of flat_size.

        Args:
            failed_rank: the rank that died.
            local_flat: this surviving rank's flat uint8 data buffer.
            pg: process group configured with surviving ranks only.

        Returns:
            The reconstructed flat uint8 buffer for the failed rank.
        """
        assert self._primary_stripe is not None, "must call compute_parity first"
        assert self._secondary_stripe is not None
        assert local_flat.dtype == torch.uint8

        new_world_size = self._world_size - 1
        padded_size = self._stripe_size * self._world_size

        # Step 1: Allgather stripes from surviving ranks.
        # Each surviving rank sends its primary and secondary concatenated.
        local_pair = torch.cat([self._primary_stripe, self._secondary_stripe])
        gathered_pairs = [
            torch.zeros_like(local_pair) for _ in range(new_world_size)
        ]
        opts_ag = AllgatherOptions()
        work = pg.allgather([gathered_pairs], [local_pair], opts_ag)
        work.wait()

        # Step 2: Reassemble XOR(all ranks) from gathered stripes.
        # Map new_rank -> original_rank for surviving ranks.
        surviving_ranks = [r for r in range(self._world_size) if r != failed_rank]

        xor_all = torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
        filled = [False] * self._world_size

        for i, orig_rank in enumerate(surviving_ranks):
            pair = gathered_pairs[i]
            pri = pair[: self._stripe_size]
            sec = pair[self._stripe_size :]

            # Primary stripe index = orig_rank
            pri_idx = orig_rank
            if not filled[pri_idx]:
                start = pri_idx * self._stripe_size
                xor_all[start : start + self._stripe_size] = pri
                filled[pri_idx] = True

            # Secondary stripe index = (orig_rank + 1) % world_size
            sec_idx = (orig_rank + 1) % self._world_size
            if not filled[sec_idx]:
                start = sec_idx * self._stripe_size
                xor_all[start : start + self._stripe_size] = sec
                filled[sec_idx] = True

        assert all(filled), (
            f"Not all stripes recovered: {filled}, failed_rank={failed_rank}"
        )

        # Step 3: Compute XOR(surviving) via allgather + local XOR
        surviving_buf = torch.zeros(
            padded_size, dtype=torch.uint8, device=self._device
        )
        surviving_buf[: self._flat_size].copy_(local_flat)

        gathered_surviving = [
            torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
            for _ in range(new_world_size)
        ]
        work = pg.allgather(
            [gathered_surviving], [surviving_buf], AllgatherOptions()
        )
        work.wait()

        xor_surviving = torch.zeros(
            padded_size, dtype=torch.uint8, device=self._device
        )
        for g in gathered_surviving:
            xor_surviving ^= g

        # Step 4: data[f] = XOR(all ranks) ^ XOR(surviving)
        result = xor_all ^ xor_surviving
        return result[: self._flat_size].clone()


class ErasureCodingManager:
    """
    Reed-Solomon erasure coding manager for multi-failure fault tolerance.

    Computes ``num_parity`` (m) syndromes over GF(2^8) using a Vandermonde
    encoding matrix. Each syndrome is split into N stripes, and each rank
    stores (m+1) consecutive stripe positions to tolerate up to m failures.

    Syndrome computation:
      ``S_j = XOR_{i=0}^{N-1}(gf_mul(g^(i*j), data_i))`` for j = 0..m-1

    - j=0: S_0 = XOR(all data) (pure XOR, same as RAID5)
    - j>=1: weighted GF(2^8) sums

    Memory overhead per GPU: ``m * (m+1) * stripe_size`` bytes, where
    stripe_size = ceil(flat_size / N). This gives m*(m+1)/N of the data size.

    When m=1, this is equivalent to ``ParityManager`` (XOR parity, 2/N overhead).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        num_parity: int = 1,
    ) -> None:
        if num_parity < 1:
            raise ValueError(f"num_parity must be >= 1, got {num_parity}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._num_parity = num_parity

        # Positions this rank stores for each syndrome: (m+1) consecutive
        self._stored_positions = [
            (rank + offset) % world_size for offset in range(num_parity + 1)
        ]

        # Stored syndrome stripes: shape [m, m+1, stripe_size] (set after compute)
        self._stored_stripes: Optional[torch.Tensor] = None
        self._stripe_size: int = 0
        self._flat_size: int = 0

    @property
    def stripe_size(self) -> int:
        return self._stripe_size

    @property
    def stored_stripes(self) -> Optional[torch.Tensor]:
        return self._stored_stripes

    def compute_parity(self, local_flat: torch.Tensor, pg: ProcessGroup) -> None:
        """
        Compute m erasure coding syndromes across all ranks.

        Uses one allgather to collect all ranks' data, then computes m
        syndromes locally. Each rank stores (m+1) stripes per syndrome.

        Args:
            local_flat: this rank's flat uint8 data buffer.
            pg: the process group to use for communication.
        """
        from torchft.gf256 import GF_GENERATOR, gf_mul_vec, gf_pow

        assert local_flat.dtype == torch.uint8

        N = self._world_size
        m = self._num_parity
        self._flat_size = local_flat.numel()
        self._stripe_size = math.ceil(self._flat_size / N)
        padded_size = self._stripe_size * N

        # Pad local data to uniform size
        buf = torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
        buf[: self._flat_size].copy_(local_flat)

        # Allgather all ranks' data
        gathered = [
            torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
            for _ in range(N)
        ]
        work = pg.allgather([gathered], [buf], AllgatherOptions())
        work.wait()

        # Compute m syndromes
        # S_j = XOR_{i=0}^{N-1}(gf_mul(g^(i*j), data_i))
        syndromes = torch.zeros(
            m, padded_size, dtype=torch.uint8, device=self._device
        )
        for j in range(m):
            for i in range(N):
                coeff = gf_pow(GF_GENERATOR, i * j)
                if coeff == 1:
                    # j=0 always has coeff=1 (pure XOR); j>0 has coeff=1 for i=0
                    syndromes[j] ^= gathered[i]
                else:
                    syndromes[j] ^= gf_mul_vec(coeff, gathered[i])

        # Store (m+1) stripes per syndrome for this rank
        num_stored = len(self._stored_positions)
        self._stored_stripes = torch.zeros(
            m, num_stored, self._stripe_size,
            dtype=torch.uint8, device=self._device,
        )
        for j in range(m):
            for s_idx, pos in enumerate(self._stored_positions):
                start = pos * self._stripe_size
                self._stored_stripes[j, s_idx] = syndromes[
                    j, start : start + self._stripe_size
                ]

    def reconstruct(
        self,
        failed_ranks: List[int],
        local_flat: torch.Tensor,
        pg: ProcessGroup,
    ) -> Dict[int, torch.Tensor]:
        """
        Reconstruct failed ranks' data from surviving data + parity syndromes.

        Must be called only on surviving ranks. Uses k syndromes to recover
        k failed ranks (k <= num_parity).

        Algorithm:
        1. Allgather stored stripes from survivors, reassemble k syndromes.
        2. Allgather surviving data, compute residuals for each syndrome.
        3. Build k×k Vandermonde submatrix, invert in GF(2^8).
        4. Apply inverse to residuals to recover failed ranks' data.

        Args:
            failed_ranks: sorted list of ranks that died (len <= num_parity).
            local_flat: this surviving rank's flat uint8 data buffer.
            pg: process group configured with surviving ranks only.

        Returns:
            Dict mapping each failed rank to its reconstructed flat uint8 buffer.
        """
        from torchft.gf256 import (
            GF_GENERATOR,
            gf_matrix_inv,
            gf_mul_vec,
            gf_pow,
        )

        assert self._stored_stripes is not None, "must call compute_parity first"
        assert local_flat.dtype == torch.uint8

        k = len(failed_ranks)
        m = self._num_parity
        N = self._world_size
        if k > m:
            raise ValueError(
                f"Cannot recover {k} failures with {m} parity syndromes"
            )
        if N < m + 1:
            raise ValueError(
                f"world_size ({N}) must be >= num_parity + 1 "
                f"({m + 1}) to recover from failures"
            )
        if k == 0:
            return {}

        failed_set = set(failed_ranks)
        new_world_size = N - k
        padded_size = self._stripe_size * N

        # -- Step 1: Allgather stored stripes from survivors --
        # Each survivor sends its stored_stripes flattened
        num_stored = len(self._stored_positions)
        local_stripes_flat = self._stored_stripes.reshape(-1)
        gathered_stripes = [
            torch.zeros_like(local_stripes_flat) for _ in range(new_world_size)
        ]
        work = pg.allgather(
            [gathered_stripes], [local_stripes_flat], AllgatherOptions()
        )
        work.wait()

        # Identify surviving ranks in original ordering
        surviving_ranks = [r for r in range(N) if r not in failed_set]

        # Reassemble k syndromes (we use syndrome indices 0..k-1)
        # Each survivor stored positions [(orig_rank + offset) % N for offset in 0..m]
        syndromes = torch.zeros(
            k, padded_size, dtype=torch.uint8, device=self._device
        )
        filled = [[False] * N for _ in range(k)]

        for i, orig_rank in enumerate(surviving_ranks):
            stripes_i = gathered_stripes[i].reshape(m, num_stored, self._stripe_size)
            survivor_positions = [
                (orig_rank + offset) % N for offset in range(num_stored)
            ]
            for j in range(k):
                for s_idx, pos in enumerate(survivor_positions):
                    if not filled[j][pos]:
                        start = pos * self._stripe_size
                        syndromes[j, start : start + self._stripe_size] = (
                            stripes_i[j, s_idx]
                        )
                        filled[j][pos] = True

        for j in range(k):
            assert all(filled[j]), (
                f"Syndrome {j} not fully recovered: {filled[j]}, "
                f"failed_ranks={failed_ranks}"
            )

        # -- Step 2: Allgather surviving data, compute residuals --
        surviving_buf = torch.zeros(
            padded_size, dtype=torch.uint8, device=self._device
        )
        surviving_buf[: self._flat_size].copy_(local_flat)

        gathered_data = [
            torch.zeros(padded_size, dtype=torch.uint8, device=self._device)
            for _ in range(new_world_size)
        ]
        work = pg.allgather(
            [gathered_data], [surviving_buf], AllgatherOptions()
        )
        work.wait()

        # Residual_j = S_j ^ contribution_of_surviving_ranks
        # contribution_of_surviving for syndrome j = XOR of gf_mul(g^(i*j), data_i) for surviving i
        residuals = syndromes.clone()
        for idx, orig_rank in enumerate(surviving_ranks):
            for j in range(k):
                coeff = gf_pow(GF_GENERATOR, orig_rank * j)
                if coeff == 1:
                    residuals[j] ^= gathered_data[idx]
                else:
                    residuals[j] ^= gf_mul_vec(coeff, gathered_data[idx])

        # -- Step 3: Build and invert k×k Vandermonde submatrix --
        # Matrix[j][f_idx] = g^(failed_ranks[f_idx] * j)
        vand_sub = [
            [gf_pow(GF_GENERATOR, failed_ranks[f_idx] * j) for f_idx in range(k)]
            for j in range(k)
        ]
        inv_matrix = gf_matrix_inv(vand_sub, k)

        # -- Step 4: Apply inverse to residuals --
        # data[failed_ranks[f_idx]] = XOR_{j=0}^{k-1}(gf_mul(inv[f_idx][j], residual_j))
        result: Dict[int, torch.Tensor] = {}
        for f_idx in range(k):
            recovered = torch.zeros(
                padded_size, dtype=torch.uint8, device=self._device
            )
            for j in range(k):
                coeff = inv_matrix[f_idx][j]
                if coeff == 0:
                    continue
                elif coeff == 1:
                    recovered ^= residuals[j]
                else:
                    recovered ^= gf_mul_vec(coeff, residuals[j])
            result[failed_ranks[f_idx]] = recovered[: self._flat_size].clone()

        return result


def _extract_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract the local tensor from a DTensor, or return the tensor itself."""
    if isinstance(t, DTensor):
        return t.to_local()
    return t


class ErasureCodingFSDP:
    """
    Erasure-coded fault tolerance wrapper for FSDP training.

    A context manager (following the LocalSGD/DiLoCo pattern) that integrates
    with optimizer step hooks to maintain erasure coding parity of model
    parameters and optimizer state. When GPUs fail, surviving GPUs can
    reconstruct the dead ranks' data and reshard.

    With ``num_parity=1`` (default), this is equivalent to RAID5 XOR parity
    (single failure tolerance). Higher values of ``num_parity`` use
    Reed-Solomon erasure coding over GF(2^8) for multi-failure tolerance.

    Parity is computed asynchronously on a separate CUDA stream after each
    optimizer step, overlapping with the next step's forward pass.

    Usage::

        ec = ErasureCodingFSDP(manager, model, optimizer, fsdp_mesh, pg, num_parity=2)
        with ec:
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch).sum()
                loss.backward()
                optimizer.step()
    """

    def __init__(
        self,
        manager: "Manager",
        model: nn.Module,
        optimizer: optim.Optimizer,
        fsdp_mesh: DeviceMesh,
        intra_group_pg: ProcessGroup,
        num_parity: int = 1,
    ) -> None:
        """
        Args:
            manager: The torchft Manager for fault tolerance coordination.
            model: The FSDP-wrapped model.
            optimizer: The optimizer used for training.
            fsdp_mesh: The DeviceMesh used for FSDP sharding.
            intra_group_pg: The ProcessGroup for intra-replica-group communication.
            num_parity: Number of parity syndromes (m). m=1 is RAID5/XOR,
                m=2 tolerates 2 simultaneous failures, etc.
        """
        from torchft.manager import Manager

        self._manager: Manager = manager
        self._model = model
        self._optimizer = optimizer
        self._fsdp_mesh = fsdp_mesh
        self._pg = intra_group_pg
        self._num_parity = num_parity

        device = fsdp_mesh.device_type
        rank = fsdp_mesh.get_local_rank()
        world_size = fsdp_mesh.size()

        if device != "cpu":
            self._device = torch.device(device, torch.cuda.current_device())
        else:
            self._device = torch.device("cpu")
        self._rank = rank
        self._world_size = world_size

        self._parity_mgr = ErasureCodingManager(
            rank, world_size, self._device, num_parity
        )
        self._flattener = StateFlattener()

        self._hooks: List[RemovableHandle] = []

        # Async parity computation stream
        self._parity_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device)
            if torch.cuda.is_available() and device != "cpu"
            else None
        )
        self._parity_event: Optional[torch.cuda.Event] = None

        # Track whether parity has been computed at least once
        self._parity_initialized = False

    def __enter__(self) -> "ErasureCodingFSDP":
        self._hooks.append(
            self._optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return False

    def _collect_protected_tensors(self) -> List[torch.Tensor]:
        """
        Collect all tensors that need parity protection:
        model parameters and their optimizer state (exp_avg, exp_avg_sq).
        """
        tensors: List[torch.Tensor] = []

        for param in self._model.parameters():
            local_p = _extract_local_tensor(param.data)
            tensors.append(local_p)

        # Collect optimizer state tensors
        for param in self._model.parameters():
            state = self._optimizer.state.get(param)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    t = state[key]
                    tensors.append(_extract_local_tensor(t))

        return tensors

    def _do_update_parity(self) -> None:
        """Flatten protected tensors and compute parity."""
        tensors = self._collect_protected_tensors()
        if len(tensors) == 0:
            return
        flat = self._flattener.flatten(tensors)
        self._parity_mgr.compute_parity(flat, self._pg)
        self._parity_initialized = True

    def _update_parity_async(self) -> None:
        """Launch parity computation on a separate CUDA stream."""
        if self._parity_stream is not None:
            # Record event on current stream so parity stream waits for optimizer.step()
            event = torch.cuda.Event()
            event.record()
            self._parity_stream.wait_event(event)

            with torch.cuda.stream(self._parity_stream):
                self._do_update_parity()

            # Record completion event for synchronization before reconstruction
            self._parity_event = torch.cuda.Event()
            self._parity_event.record(self._parity_stream)
        else:
            self._do_update_parity()

    def _ensure_parity_complete(self) -> None:
        """Block until async parity computation finishes."""
        if self._parity_event is not None:
            self._parity_event.synchronize()
            self._parity_event = None

    def _step_post_hook(
        self,
        _optim: optim.Optimizer,
        _args: Tuple[Any, ...],
        _kwargs: Dict[str, Any],
    ) -> None:
        """Called after optimizer.step(). Launches async parity computation."""
        self._update_parity_async()

    def handle_failure(self, failed_ranks: List[int]) -> None:
        """
        Reconstruct dead rank data, reshard model+optimizer, reconfigure PG.

        Called by the Manager when an intra-group failure is detected.

        Args:
            failed_ranks: sorted list of ranks that failed (up to num_parity).
        """
        if len(failed_ranks) > self._num_parity:
            raise ValueError(
                f"Cannot recover {len(failed_ranks)} failures with "
                f"{self._num_parity} parity syndromes"
            )
        logger.info(
            f"ErasureCoding(m={self._num_parity}): "
            f"handling failure of ranks {failed_ranks}"
        )

        # Ensure the last parity computation is complete
        self._ensure_parity_complete()

        if not self._parity_initialized:
            raise RuntimeError(
                "Cannot reconstruct: parity has not been computed yet. "
                "At least one optimizer.step() must complete before failure recovery."
            )

        # Get current local data
        tensors = self._collect_protected_tensors()
        flat = self._flattener.flatten(tensors)

        # Reconstruct the failed ranks' data
        reconstructed = self._parity_mgr.reconstruct(failed_ranks, flat, self._pg)

        # Reshard across N-k GPUs
        self._reshard_multi(reconstructed, failed_ranks)

        logger.info(
            f"ErasureCoding: recovery complete, now running on "
            f"{self._world_size} GPUs"
        )

    def _reshard_multi(
        self,
        reconstructed: Dict[int, torch.Tensor],
        failed_ranks: List[int],
    ) -> None:
        """
        Redistribute model parameters and optimizer state across N-k GPUs
        after reconstructing failed ranks' data.

        Args:
            reconstructed: dict mapping each failed rank to its reconstructed
                flat uint8 buffer.
            failed_ranks: sorted list of ranks that failed.
        """
        failed_set = set(failed_ranks)
        old_world_size = self._world_size
        k = len(failed_ranks)
        new_world_size = old_world_size - k

        # Compute new rank: shift down by the count of failed ranks below this one
        new_rank = self._rank - sum(1 for f in failed_ranks if f < self._rank)

        # Unflatten reconstructed data for each failed rank
        reconstructed_tensors: Dict[int, List[torch.Tensor]] = {}
        for fr, flat_data in reconstructed.items():
            reconstructed_tensors[fr] = self._flattener.unflatten(flat_data)

        # Split reconstructed tensors in the same order as _collect_protected_tensors
        recon_idx = 0

        # Reshard model parameters
        for param in self._model.parameters():
            local_p = _extract_local_tensor(param.data)
            recon_shards = {
                fr: reconstructed_tensors[fr][recon_idx] for fr in failed_ranks
            }
            recon_idx += 1

            new_shard = self._reshard_parameter_multi(
                local_p, recon_shards, failed_ranks, old_world_size,
                new_rank, new_world_size,
            )

            if isinstance(param.data, DTensor):
                param.data._local_tensor.copy_(new_shard)
            else:
                param.data.copy_(new_shard)

        # Reshard optimizer state
        for param in self._model.parameters():
            state = self._optimizer.state.get(param)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    local_t = _extract_local_tensor(state[key])
                    recon_shards = {
                        fr: reconstructed_tensors[fr][recon_idx]
                        for fr in failed_ranks
                    }
                    recon_idx += 1

                    new_shard = self._reshard_parameter_multi(
                        local_t, recon_shards, failed_ranks, old_world_size,
                        new_rank, new_world_size,
                    )

                    if isinstance(state[key], DTensor):
                        state[key]._local_tensor.copy_(new_shard)
                    else:
                        state[key].copy_(new_shard)

        # Update internal state for N-k configuration
        self._rank = new_rank
        self._world_size = new_world_size

        # Build new DeviceMesh excluding the failed ranks
        old_mesh_1d = self._fsdp_mesh.mesh.tolist()
        if isinstance(old_mesh_1d[0], list):
            raise NotImplementedError(
                "Multi-dimensional mesh resharding not yet supported"
            )
        surviving_devices = [
            d for i, d in enumerate(old_mesh_1d) if i not in failed_set
        ]
        new_mesh = DeviceMesh(
            self._fsdp_mesh.device_type,
            surviving_devices,
        )
        self._fsdp_mesh = new_mesh

        # Update DTensor specs to point to the new mesh
        for param in self._model.parameters():
            if isinstance(param.data, DTensor):
                from torch.distributed.tensor import _DTensorSpec

                param.data._spec = _DTensorSpec(
                    mesh=new_mesh,
                    placements=param.data._spec.placements,
                )

        # Update ErasureCodingManager for the new configuration
        self._parity_mgr = ErasureCodingManager(
            new_rank, new_world_size, self._device, self._num_parity
        )

        # Recompute parity for N-k configuration
        self._do_update_parity()

    def _reshard_parameter_multi(
        self,
        local_shard: torch.Tensor,
        reconstructed_shards: Dict[int, torch.Tensor],
        failed_ranks: List[int],
        old_world_size: int,
        new_rank: int,
        new_world_size: int,
    ) -> torch.Tensor:
        """
        Reshard a single parameter/state tensor from old_world_size to new_world_size.

        Each surviving rank has its own shard and the reconstructed shards for
        failed ranks. We allgather surviving shards, form the full tensor, re-chunk.

        Args:
            local_shard: this rank's current shard.
            reconstructed_shards: dict of {failed_rank: reconstructed_shard}.
            failed_ranks: sorted list of failed ranks.
            old_world_size: previous world size.
            new_rank: this rank's new index in the N-k group.
            new_world_size: new world size (N-k).

        Returns:
            The new local shard for this rank.
        """
        failed_set = set(failed_ranks)

        # Allgather surviving shards
        gathered = [torch.zeros_like(local_shard) for _ in range(new_world_size)]
        opts = AllgatherOptions()
        work = self._pg.allgather([gathered], [local_shard], opts)
        work.wait()

        # Build the full list of old shards in original rank order
        all_shards: List[torch.Tensor] = []
        surviving_idx = 0
        for old_rank in range(old_world_size):
            if old_rank in failed_set:
                all_shards.append(reconstructed_shards[old_rank])
            else:
                all_shards.append(gathered[surviving_idx])
                surviving_idx += 1

        # Concatenate to form the full (unsharded) tensor
        full_tensor = torch.cat([s.flatten() for s in all_shards])

        # Re-chunk for N-k GPUs
        chunks = full_tensor.chunk(new_world_size)
        new_shard = chunks[new_rank].clone()

        # Reshape to match expected shard shape
        return new_shard.reshape(
            self._compute_new_shard_shape(
                local_shard.shape, old_world_size, new_world_size
            )
        )

    @staticmethod
    def _compute_new_shard_shape(
        old_shard_shape: torch.Size,
        old_world_size: int,
        new_world_size: int,
    ) -> torch.Size:
        """
        Compute new shard shape after resharding.

        FSDP shards along dimension 0. The new shard size along dim 0 is
        ceil(full_dim0 / new_world_size).
        """
        if len(old_shard_shape) == 0:
            return old_shard_shape

        # Full dimension 0 = old_shard_dim0 * old_world_size
        full_dim0 = old_shard_shape[0] * old_world_size
        new_dim0 = math.ceil(full_dim0 / new_world_size)
        return torch.Size([new_dim0] + list(old_shard_shape[1:]))


class RAID5FSDP(ErasureCodingFSDP):
    """
    RAID5-style fault tolerance wrapper for FSDP training.

    Backward-compatible alias for ``ErasureCodingFSDP(num_parity=1)``.

    A context manager (following the LocalSGD/DiLoCo pattern) that integrates
    with optimizer step hooks to maintain XOR parity of model parameters and
    optimizer state. When a GPU fails, surviving GPUs can reconstruct the dead
    rank's data and reshard across N-1 GPUs.

    Parity is computed asynchronously on a separate CUDA stream after each
    optimizer step, overlapping with the next step's forward pass.

    Usage::

        raid5 = RAID5FSDP(manager, model, optimizer, fsdp_mesh, pg)
        with raid5:
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch).sum()
                loss.backward()
                optimizer.step()
    """

    def __init__(
        self,
        manager: "Manager",
        model: nn.Module,
        optimizer: optim.Optimizer,
        fsdp_mesh: DeviceMesh,
        intra_group_pg: ProcessGroup,
    ) -> None:
        """
        Args:
            manager: The torchft Manager for fault tolerance coordination.
            model: The FSDP-wrapped model.
            optimizer: The optimizer used for training.
            fsdp_mesh: The DeviceMesh used for FSDP sharding.
            intra_group_pg: The ProcessGroup for intra-replica-group communication.
        """
        super().__init__(
            manager, model, optimizer, fsdp_mesh, intra_group_pg, num_parity=1
        )

    def handle_failure(self, failed_ranks: List[int]) -> None:
        """
        Reconstruct dead rank data, reshard model+optimizer, reconfigure PG.

        Called by the Manager when an intra-group failure is detected.

        Args:
            failed_ranks: list of ranks that failed (only single failure
                supported for RAID5).
        """
        if len(failed_ranks) != 1:
            raise ValueError(
                f"RAID5 only supports single-failure recovery, "
                f"got {len(failed_ranks)} failures"
            )
        super().handle_failure(failed_ranks)
