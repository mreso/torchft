# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RAID5-Style Fault Tolerance for FSDP
=====================================

This module implements intra-group fault tolerance for FSDP training using
RAID5-style XOR parity. Each GPU stores a small parity buffer, and when one
GPU dies, surviving GPUs reconstruct its data via XOR and reshard across N-1
GPUs, eliminating the need for external checkpoint recovery for single-GPU
failures.
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


def _extract_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract the local tensor from a DTensor, or return the tensor itself."""
    if isinstance(t, DTensor):
        return t.to_local()
    return t


class RAID5FSDP:
    """
    RAID5-style fault tolerance wrapper for FSDP training.

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
        from torchft.manager import Manager

        self._manager: Manager = manager
        self._model = model
        self._optimizer = optimizer
        self._fsdp_mesh = fsdp_mesh
        self._pg = intra_group_pg

        device = fsdp_mesh.device_type
        rank = fsdp_mesh.get_local_rank()
        world_size = fsdp_mesh.size()

        if device != "cpu":
            self._device = torch.device(device, torch.cuda.current_device())
        else:
            self._device = torch.device("cpu")
        self._rank = rank
        self._world_size = world_size

        self._parity_mgr = ParityManager(rank, world_size, self._device)
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

    def __enter__(self) -> "RAID5FSDP":
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
            failed_ranks: list of ranks that failed (currently only single
                failure is supported).
        """
        if len(failed_ranks) != 1:
            raise ValueError(
                f"RAID5 only supports single-failure recovery, got {len(failed_ranks)} failures"
            )
        failed_rank = failed_ranks[0]
        logger.info(f"RAID5: handling failure of rank {failed_rank}")

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

        # Reconstruct the failed rank's data
        reconstructed_flat = self._parity_mgr.reconstruct(failed_rank, flat, self._pg)
        reconstructed_tensors = self._flattener.unflatten(reconstructed_flat)

        # Reshard across N-1 GPUs
        self._reshard(reconstructed_tensors, failed_rank)

        logger.info(
            f"RAID5: recovery complete, now running on {self._world_size} GPUs"
        )

    def _reshard(
        self, reconstructed_tensors: List[torch.Tensor], failed_rank: int
    ) -> None:
        """
        Redistribute model parameters and optimizer state across N-1 GPUs
        after reconstructing a failed rank's data.

        Args:
            reconstructed_tensors: the failed rank's tensors (same order as
                _collect_protected_tensors produces).
            failed_rank: the rank that failed.
        """
        old_world_size = self._world_size

        # Compute new rank: ranks above failed_rank shift down by 1
        new_world_size = old_world_size - 1
        if self._rank > failed_rank:
            new_rank = self._rank - 1
        else:
            new_rank = self._rank

        # Split reconstructed tensors in the same order as _collect_protected_tensors
        recon_idx = 0

        # Reshard model parameters
        for param in self._model.parameters():
            local_p = _extract_local_tensor(param.data)
            recon_p = reconstructed_tensors[recon_idx]
            recon_idx += 1

            new_shard = self._reshard_parameter(
                local_p, recon_p, failed_rank, old_world_size, new_rank, new_world_size
            )

            # Update parameter in-place
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
                    recon_t = reconstructed_tensors[recon_idx]
                    recon_idx += 1

                    new_shard = self._reshard_parameter(
                        local_t,
                        recon_t,
                        failed_rank,
                        old_world_size,
                        new_rank,
                        new_world_size,
                    )

                    if isinstance(state[key], DTensor):
                        state[key]._local_tensor.copy_(new_shard)
                    else:
                        state[key].copy_(new_shard)

        # Update internal state for N-1 configuration
        self._rank = new_rank
        self._world_size = new_world_size

        # Build new DeviceMesh excluding the failed rank
        old_mesh_1d = self._fsdp_mesh.mesh.tolist()
        if isinstance(old_mesh_1d[0], list):
            # multi-dim mesh, flatten to get device IDs
            raise NotImplementedError(
                "Multi-dimensional mesh resharding not yet supported"
            )
        surviving_devices = [d for i, d in enumerate(old_mesh_1d) if i != failed_rank]
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

        # Update ParityManager for the new configuration
        self._parity_mgr = ParityManager(new_rank, new_world_size, self._device)

        # Recompute parity for N-1 configuration
        self._do_update_parity()

    def _reshard_parameter(
        self,
        local_shard: torch.Tensor,
        reconstructed_shard: torch.Tensor,
        failed_rank: int,
        old_world_size: int,
        new_rank: int,
        new_world_size: int,
    ) -> torch.Tensor:
        """
        Reshard a single parameter/state tensor from old_world_size to new_world_size.

        Each surviving rank has its own shard and the reconstructed shard for the
        failed rank. We allgather all shards, form the full tensor, and re-chunk.

        Args:
            local_shard: this rank's current shard.
            reconstructed_shard: the dead rank's shard (just reconstructed).
            failed_rank: rank that failed.
            old_world_size: previous world size.
            new_rank: this rank's new index in the N-1 group.
            new_world_size: new world size (N-1).

        Returns:
            The new local shard for this rank.
        """
        # Allgather surviving shards
        gathered = [torch.zeros_like(local_shard) for _ in range(new_world_size)]
        opts = AllgatherOptions()
        work = self._pg.allgather([gathered], [local_shard], opts)
        work.wait()

        # Build the full list of old shards in original rank order
        all_shards: List[torch.Tensor] = []
        surviving_idx = 0
        for old_rank in range(old_world_size):
            if old_rank == failed_rank:
                all_shards.append(reconstructed_shard)
            else:
                all_shards.append(gathered[surviving_idx])
                surviving_idx += 1

        # Concatenate to form the full (unsharded) tensor
        full_tensor = torch.cat([s.flatten() for s in all_shards])

        # Re-chunk for N-1 GPUs
        chunks = full_tensor.chunk(new_world_size)
        new_shard = chunks[new_rank].clone()

        # Reshape to match expected shard shape
        # After rechunking the shard may have a different shape
        return new_shard.reshape(
            self._compute_new_shard_shape(local_shard.shape, old_world_size, new_world_size)
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
