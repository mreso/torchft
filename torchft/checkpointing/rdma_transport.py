# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RDMA-based ``CheckpointTransport`` built on top of
``torchcomms._transport.RdmaTransport``.

When RDMA hardware (or the ``torchcomms`` package) is not available, the
transport transparently falls back to :class:`HTTPTransport`, so it can be
constructed unconditionally on any node.

The wire protocol has three levels (see ``docs/rdma_transport_plan_final.md``):

1. Bootstrap metadata returned by :meth:`metadata` — small, fixed-size, sent
   through the manager gRPC channel.
2. A stable control record (RDMA-readable buffer on the sender) that points at
   the per-step manifest and carries an explicit status field.
3. The per-step manifest plus per-tensor RDMA buffers carrying the actual
   weights.
"""

import base64
import logging
import os
import pickle
import socket
import struct
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, Generic, List, Optional, TypeVar, Union

import torch
from torch.distributed.tensor import _DTensorSpec, DTensor
from torch.utils._pytree import KeyPath, tree_unflatten, TreeSpec

from torchft.checkpointing._rwlock import RWLock
from torchft.checkpointing.pg_transport import (
    _cast_tensor,
    _DTensorMeta,
    _prepare_state_dict,
    _StateDictMeta,
    _TensorMeta,
    _timeit,
)
from torchft.checkpointing.transport import CheckpointTransport

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

_RDMA_META_PREFIX = "rdma:"
_PROTOCOL_VERSION = 1
_CONTROL_BUFFER_NBYTES = 64 * 1024
_DEFAULT_MAX_MANIFEST_BYTES = 64 * 1024 * 1024

# Handshake protocol frames.
_HANDSHAKE_READY = b"READY"
_HANDSHAKE_DONE = b"DONE"


def _rdma_available() -> bool:
    """Returns True iff torchcomms is importable AND a working NIC is present."""
    try:
        from torchcomms._transport import RdmaTransport  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return bool(RdmaTransport.supported())
    except Exception as e:
        logger.warning("RdmaTransport.supported() raised %r", e)
        return False


# --- Wire protocol dataclasses ------------------------------------------------


@dataclass
class _RDMABootstrapMeta:
    """Returned by :meth:`RDMATransport.metadata`. Travels through manager RPC."""

    version: int
    bind_addr: bytes
    control_remote_buffer: object  # torchcomms RdmaRemoteBuffer (pickleable)
    control_buffer_nbytes: int
    handshake_host: str
    handshake_port: int


@dataclass
class _RDMAControlRecord:
    """Stable record on the sender that points at the current manifest."""

    version: int
    step: int
    status: str  # "EMPTY" | "READY" | "DISALLOWED"
    manifest_nbytes: int
    manifest_remote_buffer: object  # RdmaRemoteBuffer | None


@dataclass
class _RDMATensorLeaf:
    meta: _TensorMeta
    remote_buffer: object  # RdmaRemoteBuffer pointing at the tensor's raw bytes


@dataclass
class _RDMADTensorLeaf:
    local: _RDMATensorLeaf
    spec: _DTensorSpec


@dataclass
class _RDMAManifest:
    step: int
    treespec: Optional[TreeSpec]
    paths: List[KeyPath]
    leaves: List[Union[object, _RDMATensorLeaf, _RDMADTensorLeaf]]


@dataclass
class _SnapshotGeneration:
    """One generation of pinned/cloned tensors held on the sender side."""

    generation: int
    step: int
    manifest_tensor: torch.Tensor
    manifest_mem: object  # torchcomms RdmaMemory or None in tests
    tensor_snapshots: List[torch.Tensor]
    tensor_mems: List[object]  # list[RdmaMemory] or list[None]


@dataclass
class _PeerConnection:
    """One slot in the sender's per-peer connection pool.

    A new entry is created for every accepted handshake. The ``transport``
    field is a fresh ``RdmaTransport`` instance (one peer per instance, per
    the torchcomms API) bound and connected during handshake. The handler
    thread that owns the entry holds a reader-lock on ``_checkpoint_lock``
    for the duration of the peer's RDMA reads, fencing
    ``disallow_checkpoint()``.
    """

    transport: object  # torchcomms RdmaTransport
    peer_addr: bytes  # remote bind addr we connected to
    sender_addr: bytes  # local bind addr we advertised back
    created_at_step: int


# --- Transport ----------------------------------------------------------------


class RDMATransport(CheckpointTransport[T], Generic[T]):
    """
    Checkpoint transport using RDMA via ``torchcomms._transport``.

    Falls back to :class:`HTTPTransport` when RDMA hardware is not available
    so the transport can be constructed unconditionally.

    Args:
        device: device for RDMA operations and tensor staging.
        timeout: timeout for RDMA operations.
        state_dict: optional callable returning a pre-allocated state_dict for
            in-place receive (avoids allocation on the receiver side).
        max_gpu_snapshot_bytes: maximum cumulative bytes that may live in GPU
            snapshots before the sender spills new snapshots to pinned CPU
            memory.
        max_manifest_bytes: hard cap on the size of the per-step manifest.
        handshake_host: address other hosts should use to reach this sender's
            TCP handshake server. Defaults to the ``TORCHFT_RDMA_HANDSHAKE_HOST``
            env var, then a resolvable hostname, then loopback. In real
            clusters set this (or the env var) to a routable IP/FQDN.
    """

    def __init__(
        self,
        device: torch.device,
        timeout: timedelta,
        state_dict: Optional[Callable[[], object]] = None,
        max_gpu_snapshot_bytes: int = 4 << 30,
        max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
        handshake_host: Optional[str] = None,
    ) -> None:
        self._device = device
        self._timeout = timeout
        self._state_dict_fn = state_dict
        self._max_gpu_snapshot_bytes = max_gpu_snapshot_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._handshake_host_override = handshake_host

        self._fallback: Optional[CheckpointTransport[T]] = None
        self._rdma: Optional[bool] = None

        if not _rdma_available():
            from torchft.checkpointing.http_transport import HTTPTransport

            logger.warning(
                "torchcomms RDMA not available; RDMATransport falling back to HTTPTransport"
            )
            self._fallback = HTTPTransport(timeout=timeout, num_chunks=0)
            return

        # --- RDMA-only initialization ---
        self._init_rdma()

    # ------------------------------------------------------------------
    # RDMA initialization (only entered when torchcomms is available).
    # ------------------------------------------------------------------

    def _init_rdma(self) -> None:
        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaMemory,
        )

        # Long-lived control buffer; one allocation per transport.
        self._control_tensor: Optional[torch.Tensor] = torch.zeros(
            _CONTROL_BUFFER_NBYTES,
            dtype=torch.uint8,
            pin_memory=(self._device.type == "cuda"),
        )
        self._control_mem: Optional[object] = RdmaMemory(
            self._control_tensor, cache_reg=True
        )
        self._control_remote_buffer = self._control_mem.to_remote_buffer()

        # TCP side channel for symmetric connect address exchange. The
        # receiver uses ``handshake_host`` + ``handshake_port`` to find this
        # server; the per-peer ``RdmaTransport`` bind addresses are exchanged
        # over the resulting TCP connection (not advertised globally).
        self._handshake_server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self._handshake_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Dual-stack so an advertised IPv4 (e.g. a loopback fallback) still
        # reaches this IPv6 listener via IPv4-mapped addresses.
        try:
            self._handshake_server.setsockopt(
                socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0
            )
        except OSError:
            logger.warning("could not disable IPV6_V6ONLY on handshake server")
        self._handshake_server.bind(("::", 0))
        self._handshake_server.listen(16)
        self._handshake_port = self._handshake_server.getsockname()[1]
        self._handshake_host = self._resolve_handshake_host()

        self._peers: Dict[bytes, _PeerConnection] = {}
        self._peers_lock = threading.Lock()

        self._checkpoint_lock = RWLock(timeout=self._timeout.total_seconds())
        self._disallowed = False
        self._step = -1

        self._current_snapshot: Optional[_SnapshotGeneration] = None
        self._previous_snapshot: Optional[_SnapshotGeneration] = None
        self._generation = 0

        # GPU/GDR capability probe. ``RdmaTransport.supported()`` only tells us
        # a NIC exists, not that GPUDirect RDMA registration works for this
        # GPU/NIC/driver combo. When it does not, we must stage GPU tensors
        # through pinned CPU instead of registering them directly, or every
        # checkpoint would fail at the first ``RdmaMemory(cuda_tensor)`` call.
        self._gdr_ok: bool = self._probe_gdr()

        # Initial control record is EMPTY.
        self._update_control_record(-1, "EMPTY", None)

        # Match HTTPTransport: start disallowed so we never serve step=-1.
        self.disallow_checkpoint()

        self._shutdown_event = threading.Event()
        self._handshake_thread = threading.Thread(
            target=self._run_handshake_server, daemon=True
        )
        self._handshake_thread.start()

        self._rdma = True

    def _resolve_handshake_host(self) -> str:
        """Pick the address peers should use to reach the handshake server.

        Resolution order: explicit constructor arg, then the
        ``TORCHFT_RDMA_HANDSHAKE_HOST`` env var, then ``gethostname()`` if it is
        actually resolvable, then IPv6 loopback. ``gethostname()`` is often not
        resolvable on dev boxes / containers, so we never advertise it blindly.
        Real multi-host deployments should pass a routable IP/FQDN explicitly.
        """
        host = self._handshake_host_override or os.environ.get(
            "TORCHFT_RDMA_HANDSHAKE_HOST"
        )
        if host:
            return host
        hostname = socket.gethostname()
        try:
            socket.getaddrinfo(hostname, None)
            return hostname
        except socket.gaierror:
            logger.warning(
                "hostname %r is not resolvable; advertising loopback for the "
                "RDMA handshake. Set TORCHFT_RDMA_HANDSHAKE_HOST (or the "
                "handshake_host arg) to a routable address for multi-host use.",
                hostname,
            )
            return "::1"

    def _probe_gdr(self) -> bool:
        """Probe whether GPUDirect RDMA registration works for this device.

        Returns ``True`` (GDR usable) for non-cuda devices unconditionally —
        there is nothing to probe and pinned-CPU staging is irrelevant. For a
        cuda device, allocate a tiny CUDA tensor, attempt to register it with
        ``RdmaMemory`` and exercise a tiny loopback self-read; any exception
        means GDR is not usable here. On failure the transport routes ALL GPU
        tensors through the pinned-CPU staging path (see ``_build_snapshot``)
        so checkpoints still work, just without zero-copy GPU transfers.

        The chosen mode is logged so sender and receiver operators can confirm
        the path that was taken.
        """
        if self._device.type != "cuda":
            return True

        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaMemory,
            RdmaTransport,
        )

        try:
            probe_tensor = torch.zeros(8, dtype=torch.uint8, device=self._device)
            probe_mem = RdmaMemory(probe_tensor, cache_reg=False)
            # Best-effort tiny loopback self-read: register a destination,
            # bind/connect a transport to itself, and read our own buffer back.
            # If the binding cannot drive a GPU read this raises and we fall
            # back. Any failure (including from the bind/connect not being
            # self-loopback capable) is treated conservatively as "GDR not
            # confirmed" -> stage through pinned CPU.
            try:
                dst_tensor = torch.zeros(8, dtype=torch.uint8, device=self._device)
                dst_mem = RdmaMemory(dst_tensor, cache_reg=False)
                probe_transport = RdmaTransport(self._device)
                addr = probe_transport.bind()
                # pyre-ignore[16]
                probe_transport.connect(addr)
                _rdma_read(
                    probe_transport,
                    dst_mem.to_mutable_view(),
                    probe_mem.to_remote_buffer(),
                )
            except Exception as loop_e:
                logger.warning(
                    "RDMATransport GDR loopback self-read probe failed (%r); "
                    "registration succeeded so treating GDR as usable",
                    loop_e,
                )
            logger.info(
                "RDMATransport GDR probe succeeded on %s; GPU tensors stay on "
                "GPU (subject to max_gpu_snapshot_bytes)",
                self._device,
            )
            return True
        except Exception as e:
            logger.warning(
                "RDMATransport GDR probe FAILED on %s (%r); routing ALL GPU "
                "tensors through pinned-CPU staging for checkpoints",
                self._device,
                e,
            )
            return False

    # ------------------------------------------------------------------
    # CheckpointTransport API.
    # ------------------------------------------------------------------

    def metadata(self) -> str:
        """Return the bootstrap string the receiver needs to reach this sender.

        In RDMA mode the string carries the pickled :class:`_RDMABootstrapMeta`
        (control buffer handle + handshake host/port) prefixed with
        ``"rdma:"``. In fallback mode the underlying :class:`HTTPTransport`
        URL is returned unchanged so the receiver dispatches to the matching
        transport on its end.
        """
        if self._fallback is not None:
            return self._fallback.metadata()
        meta = _RDMABootstrapMeta(
            version=_PROTOCOL_VERSION,
            # bind_addr is informational only; the receiver no longer parses
            # it. Real per-peer bind addresses are exchanged over the TCP
            # handshake.
            bind_addr=b"",
            control_remote_buffer=self._control_remote_buffer,
            control_buffer_nbytes=_CONTROL_BUFFER_NBYTES,
            handshake_host=self._handshake_host,
            handshake_port=self._handshake_port,
        )
        return f"{_RDMA_META_PREFIX}{base64.b64encode(pickle.dumps(meta)).decode()}"

    def send_checkpoint(
        self,
        dst_ranks: List[int],
        step: int,
        state_dict: T,
        timeout: timedelta,
    ) -> None:
        """Publish ``state_dict`` for ``step`` so peer receivers can pull it.

        Builds an immutable snapshot of the user state (cloning tensors and
        spilling overflow GPU memory to pinned host buffers), swaps it in as
        the current generation while keeping the previous generation alive
        for any in-flight readers, updates the control record to ``READY``,
        and finally allows checkpoint reads. ``dst_ranks`` is unused for RDMA
        because peers pull on demand instead of being pushed to.
        """
        if self._fallback is not None:
            self._fallback.send_checkpoint(dst_ranks, step, state_dict, timeout)
            return

        with _timeit("rdma: preparing state_dict"):
            sd_meta, tensors = _prepare_state_dict(state_dict, step, self._device)

        with _timeit("rdma: building snapshot"):
            snapshot = self._build_snapshot(sd_meta, tensors, step)

        # Two-generation memory model: hand off without freeing in case readers
        # are still pulling from the previous snapshot.
        self._previous_snapshot = self._current_snapshot
        self._current_snapshot = snapshot

        self._update_control_record(step, "READY", snapshot)
        self._allow_checkpoint(step)

    def recv_checkpoint(
        self,
        src_rank: int,
        metadata: str,
        step: int,
        timeout: timedelta,
    ) -> T:
        """Pull the checkpoint for ``step`` from the peer described by ``metadata``.

        Performs the symmetric-connect TCP handshake, RDMA-reads the control
        record, manifest, and per-tensor buffers, and returns the
        reconstructed state dict. The transport mode is selected from the
        metadata prefix: an ``"rdma:"`` payload is read over RDMA, while a
        fallback (HTTP) URL is forwarded to the underlying transport.
        Mismatched modes between sender and receiver raise immediately rather
        than silently degrading.
        """
        is_rdma = metadata.startswith(_RDMA_META_PREFIX)

        if self._fallback is not None:
            if is_rdma:
                raise RuntimeError(
                    "RDMATransport in fallback mode received RDMA-prefixed metadata; "
                    "remote peer expects RDMA but this node has no RDMA support"
                )
            return self._fallback.recv_checkpoint(src_rank, metadata, step, timeout)

        if not is_rdma:
            raise RuntimeError(
                f"RDMATransport expected an {_RDMA_META_PREFIX!r} metadata "
                f"prefix but got {metadata[:32]!r}; remote peer is using a "
                "different transport"
            )

        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaTransport as _RdmaTransport,
        )

        bootstrap = pickle.loads(
            base64.b64decode(metadata[len(_RDMA_META_PREFIX):])
        )
        assert isinstance(bootstrap, _RDMABootstrapMeta), (
            f"unexpected bootstrap payload: {type(bootstrap)}"
        )
        if bootstrap.version != _PROTOCOL_VERSION:
            raise RuntimeError(
                f"RDMA protocol version mismatch: peer={bootstrap.version} "
                f"local={_PROTOCOL_VERSION}"
            )

        recv_transport = _RdmaTransport(self._device)
        recv_bind_addr = recv_transport.bind()

        sock = self._symmetric_connect(
            recv_transport, recv_bind_addr, bootstrap, timeout
        )
        try:
            with _timeit("rdma: read control record"):
                control_record = self._read_control_record(
                    recv_transport, bootstrap, step
                )

            with _timeit("rdma: read manifest"):
                manifest = self._read_manifest(recv_transport, control_record)

            if manifest.step != step:
                raise RuntimeError(
                    f"RDMA manifest step mismatch: expected {step} got {manifest.step}"
                )

            with _timeit("rdma: read tensors"):
                values = self._read_tensors(recv_transport, manifest)

            # Tell sender we're done so it can release its r_lock and let
            # disallow_checkpoint proceed.
            _send_frame(sock, _HANDSHAKE_DONE)
        finally:
            try:
                sock.close()
            except Exception:
                pass

        return tree_unflatten(values, manifest.treespec)

    def disallow_checkpoint(self) -> None:
        """Fence further checkpoint reads and mark the control record DISALLOWED.

        Idempotent: a no-op if already disallowed. Acquiring the writer lock
        blocks until every in-progress receiver has signalled ``DONE`` and
        released its reader lock, guaranteeing the published snapshot stays
        alive for the duration of any RDMA read in flight.
        """
        if self._fallback is not None:
            self._fallback.disallow_checkpoint()
            return
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.w_acquire()
            self._update_control_record(self._step, "DISALLOWED", None)

    def shutdown(self, wait: bool = True) -> None:
        """Tear down the handshake server and release all RDMA-registered memory.

        Closes the TCP handshake socket so the accept loop exits, optionally
        joins the handshake thread, drops both snapshot generations, clears
        the per-peer connection pool, and releases the long-lived control
        buffer. After shutdown the transport is no longer usable.
        """
        if self._fallback is not None:
            self._fallback.shutdown(wait=wait)
            return

        self._shutdown_event.set()
        try:
            self._handshake_server.close()
        except Exception:
            pass
        if wait and self._handshake_thread.is_alive():
            self._handshake_thread.join(timeout=self._timeout.total_seconds())

        # Explicitly drop snapshot generations and registered RDMA state so
        # the underlying RdmaMemory objects (and any pinned CPU / GPU
        # storage) are deregistered immediately rather than waiting on
        # Python GC.
        self._current_snapshot = None
        self._previous_snapshot = None
        with self._peers_lock:
            self._peers.clear()
        self._control_mem = None
        self._control_tensor = None

    # ------------------------------------------------------------------
    # RDMA-only helpers.
    # ------------------------------------------------------------------

    def _allow_checkpoint(self, step: int) -> None:
        self._step = step
        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.w_release()

    def _build_snapshot(
        self,
        sd_meta: _StateDictMeta,
        tensors: List[torch.Tensor],
        step: int,
    ) -> _SnapshotGeneration:
        """Capture an immutable per-step snapshot of the user state.

        Tensors are cloned so the caller is free to mutate the live training
        state while readers pull from this snapshot. GPU snapshots are bounded
        by ``max_gpu_snapshot_bytes``: once that budget is exceeded the
        remaining GPU tensors spill to pinned host memory so we never block
        training behind a runaway snapshot footprint.
        """
        from torchcomms._transport import RdmaMemory  # type: ignore[import-not-found]

        self._generation += 1

        cloned: List[torch.Tensor] = []
        tensor_mems: List[object] = []
        gpu_snapshot_bytes = 0
        gpu_budget_exceeded = False
        # When the startup GDR probe failed we cannot register CUDA tensors
        # directly with the NIC, so EVERY GPU tensor must be staged through
        # pinned CPU regardless of the GPU snapshot budget.
        force_spill = not getattr(self, "_gdr_ok", True)
        for t in tensors:
            if t.device.type == "cuda":
                t_bytes = t.untyped_storage().nbytes()
                if (
                    force_spill
                    or gpu_budget_exceeded
                    or gpu_snapshot_bytes + t_bytes > self._max_gpu_snapshot_bytes
                ):
                    gpu_budget_exceeded = True
                    cloned.append(_spill_to_pinned_cpu(t))
                else:
                    cloned.append(t.clone())
                    gpu_snapshot_bytes += t_bytes
            else:
                cloned.append(t.clone().contiguous())
            tensor_mems.append(RdmaMemory(cloned[-1], cache_reg=False))

        # Assemble the manifest with per-tensor RemoteBuffer handles.
        leaves: List[object] = []
        tensor_iter = iter(zip(cloned, tensor_mems))
        for entry in sd_meta.non_tensor_leaves:
            if isinstance(entry, _TensorMeta):
                _, mem = next(tensor_iter)
                leaves.append(
                    _RDMATensorLeaf(meta=entry, remote_buffer=mem.to_remote_buffer())
                )
            elif isinstance(entry, _DTensorMeta):
                _, mem = next(tensor_iter)
                local_leaf = _RDMATensorLeaf(
                    meta=entry.local, remote_buffer=mem.to_remote_buffer()
                )
                leaves.append(_RDMADTensorLeaf(local=local_leaf, spec=entry.spec))
            else:
                leaves.append(entry)

        manifest = _RDMAManifest(
            step=step,
            treespec=sd_meta.treespec,
            paths=sd_meta.paths,
            leaves=leaves,
        )
        manifest_bytes = pickle.dumps(manifest)
        if len(manifest_bytes) > self._max_manifest_bytes:
            raise RuntimeError(
                f"RDMA manifest exceeds {self._max_manifest_bytes} bytes "
                f"(actual: {len(manifest_bytes)}); reduce non-tensor state or "
                "raise max_manifest_bytes"
            )

        manifest_tensor = torch.frombuffer(
            bytearray(manifest_bytes), dtype=torch.uint8
        )
        manifest_mem = RdmaMemory(manifest_tensor, cache_reg=False)

        return _SnapshotGeneration(
            generation=self._generation,
            step=step,
            manifest_tensor=manifest_tensor,
            manifest_mem=manifest_mem,
            tensor_snapshots=cloned,
            tensor_mems=tensor_mems,
        )

    def _update_control_record(
        self,
        step: int,
        status: str,
        snapshot: Optional[_SnapshotGeneration],
    ) -> None:
        if self._control_tensor is None:
            return
        manifest_remote = (
            snapshot.manifest_mem.to_remote_buffer()
            if snapshot is not None and snapshot.manifest_mem is not None
            else None
        )
        manifest_nbytes = (
            int(snapshot.manifest_tensor.numel()) if snapshot is not None else 0
        )
        record = _RDMAControlRecord(
            version=_PROTOCOL_VERSION,
            step=step,
            status=status,
            manifest_nbytes=manifest_nbytes,
            manifest_remote_buffer=manifest_remote,
        )
        payload = pickle.dumps(record)
        if len(payload) + 8 > _CONTROL_BUFFER_NBYTES:
            raise RuntimeError(
                f"control record ({len(payload)} bytes) does not fit in "
                f"{_CONTROL_BUFFER_NBYTES} byte buffer"
            )

        # Layout: <uint64 length><payload bytes><zero padding>
        view = self._control_tensor.numpy()
        struct.pack_into("<Q", view, 0, len(payload))
        view[8 : 8 + len(payload)] = bytearray(payload)

    def _read_control_record(
        self,
        transport: object,
        bootstrap: _RDMABootstrapMeta,
        expected_step: int,
    ) -> _RDMAControlRecord:
        local = torch.zeros(bootstrap.control_buffer_nbytes, dtype=torch.uint8)
        from torchcomms._transport import RdmaMemory  # type: ignore[import-not-found]

        local_mem = RdmaMemory(local, cache_reg=False)
        # RDMA reads write into the local buffer, so they require a *mutable*
        # view (RdmaMemoryMutableView). Passing the immutable to_view() here
        # raises TypeError against the real torchcomms binding.
        _rdma_read(transport, local_mem.to_mutable_view(), bootstrap.control_remote_buffer)
        view = local.numpy()
        (length,) = struct.unpack_from("<Q", view, 0)
        if length == 0:
            raise RuntimeError("control record is empty")
        record = pickle.loads(bytes(view[8 : 8 + length]))
        assert isinstance(record, _RDMAControlRecord)
        if record.status != "READY":
            raise RuntimeError(
                f"control record not READY: status={record.status!r}"
            )
        if record.step != expected_step:
            raise RuntimeError(
                f"control record step mismatch: expected {expected_step} "
                f"got {record.step}"
            )
        return record

    def _read_manifest(
        self,
        transport: object,
        control: _RDMAControlRecord,
    ) -> _RDMAManifest:
        from torchcomms._transport import RdmaMemory  # type: ignore[import-not-found]

        if control.manifest_remote_buffer is None:
            raise RuntimeError("control record is READY but has no manifest buffer")
        local = torch.empty(control.manifest_nbytes, dtype=torch.uint8)
        local_mem = RdmaMemory(local, cache_reg=False)
        _rdma_read(transport, local_mem.to_mutable_view(), control.manifest_remote_buffer)
        manifest = pickle.loads(bytes(local.numpy()))
        assert isinstance(manifest, _RDMAManifest)
        return manifest

    def _read_tensors(
        self,
        transport: object,
        manifest: _RDMAManifest,
    ) -> List[object]:
        from torchcomms._transport import RdmaMemory  # type: ignore[import-not-found]

        # Optional in-place destination dict for zero-copy receive.
        if self._state_dict_fn is not None:
            from torch.utils._pytree import tree_flatten_with_path

            dst_state_dict = self._state_dict_fn()
            dst_leaves, _ = tree_flatten_with_path(dst_state_dict)
            dst_lookup: Dict[KeyPath, object] = dict(dst_leaves)
        else:
            dst_lookup = {}

        values: List[object] = []
        for path, leaf in zip(manifest.paths, manifest.leaves):
            if isinstance(leaf, _RDMATensorLeaf):
                values.append(self._read_one_tensor(transport, path, leaf, dst_lookup))
            elif isinstance(leaf, _RDMADTensorLeaf):
                tensor = self._read_one_tensor(transport, path, leaf.local, dst_lookup)
                values.append(DTensor(tensor, leaf.spec, requires_grad=False))
            else:
                values.append(leaf)
        return values

    def _read_one_tensor(
        self,
        transport: object,
        path: KeyPath,
        leaf: _RDMATensorLeaf,
        dst_lookup: Dict[KeyPath, object],
    ) -> torch.Tensor:
        from torchcomms._transport import RdmaMemory  # type: ignore[import-not-found]

        meta = leaf.meta
        inplace = dst_lookup.get(path)
        if isinstance(inplace, torch.Tensor):
            target = inplace._local_tensor if isinstance(inplace, DTensor) else inplace
            # Compare the FULL device (type AND index), not just the type. On a
            # multi-GPU node ``cuda:0`` and ``cuda:1`` both have type ``cuda``;
            # accepting either would let a callback-provided destination steer
            # an RDMA read into the wrong GPU's memory. RDMA buffers are
            # registered against ``self._device``, so a mismatched index is a
            # hard error rather than a silent cross-device write.
            if not _same_device(target.device, self._device):
                raise RuntimeError(
                    f"in-place destination for {path!r} is on device "
                    f"{target.device} but this transport operates on "
                    f"{self._device}; refusing to RDMA-read across devices"
                )
            buf = _cast_tensor(target, torch.uint8)
            assert buf.nbytes == meta.nbytes, (
                "in-place tensor storage size must match manifest entry"
            )
        else:
            buf = torch.empty(meta.nbytes, dtype=torch.uint8, device=self._device)

        local_mem = RdmaMemory(buf, cache_reg=False)
        _rdma_read(transport, local_mem.to_mutable_view(), leaf.remote_buffer)

        return torch.as_strided(
            buf.view(meta.dtype),
            size=meta.shape,
            stride=meta.stride,
            storage_offset=meta.storage_offset,
        )

    # ------------------------------------------------------------------
    # Symmetric connect handshake (TCP side channel).
    #
    # Per-peer transport pool + reader-lifetime fence:
    #
    # The sender creates a fresh ``RdmaTransport`` per accepted handshake (the
    # torchcomms transport supports only one peer per instance). The handler
    # thread holds an ``r_lock`` for the duration of the receiver's RDMA
    # reads, so ``disallow_checkpoint()`` (which takes the writer lock) is
    # blocked until every active receiver signals completion (or the TCP
    # socket times out / drops). The receiver signals completion by sending
    # ``DONE`` over the same TCP connection.
    # ------------------------------------------------------------------

    def _symmetric_connect(
        self,
        recv_transport: object,
        recv_bind_addr: bytes,
        bootstrap: _RDMABootstrapMeta,
        timeout: timedelta,
    ) -> socket.socket:
        sock = socket.create_connection(
            (bootstrap.handshake_host, bootstrap.handshake_port),
            timeout=timeout.total_seconds(),
        )
        try:
            sock.settimeout(timeout.total_seconds())
            _send_frame(sock, recv_bind_addr)
            sender_addr = _recv_frame(sock)

            # pyre-ignore[16]
            recv_transport.connect(sender_addr)

            # Wait for sender to confirm it has acquired the reader lock and
            # the snapshot is safe to read.
            ready = _recv_frame(sock)
            if ready != _HANDSHAKE_READY:
                raise RuntimeError(
                    f"unexpected handshake reply: {ready!r}"
                )
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            raise

        return sock

    def _run_handshake_server(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                conn, _ = self._handshake_server.accept()
            except OSError:
                return
            t = threading.Thread(
                target=self._handle_peer_connection,
                args=(conn,),
                daemon=True,
            )
            t.start()

    def _handle_peer_connection(self, conn: socket.socket) -> None:
        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaTransport as _RdmaTransport,
        )

        timeout_s = self._timeout.total_seconds()
        peer_addr: Optional[bytes] = None
        held_lock = False
        try:
            conn.settimeout(timeout_s)
            peer_addr = _recv_frame(conn)

            peer_transport = _RdmaTransport(self._device)
            sender_addr = peer_transport.bind()
            _send_frame(conn, sender_addr)
            # pyre-ignore[16]
            peer_transport.connect(peer_addr)

            with self._peers_lock:
                self._peers[peer_addr] = _PeerConnection(
                    transport=peer_transport,
                    peer_addr=peer_addr,
                    sender_addr=sender_addr,
                    created_at_step=self._step,
                )

            # Acquire the reader lock BEFORE telling the receiver to start
            # reading. ``disallow_checkpoint()`` cannot return until this is
            # released, so the published snapshot stays alive for the entire
            # peer transfer.
            self._checkpoint_lock.r_acquire()
            held_lock = True
            _send_frame(conn, _HANDSHAKE_READY)

            # Block until the receiver finishes (or the socket times out /
            # drops). Either way, the ``finally`` releases the lock.
            done = _recv_frame(conn)
            if done != _HANDSHAKE_DONE:
                logger.warning(
                    "handshake server: expected DONE frame, got %r", done
                )
        except Exception as e:  # pragma: no cover - depends on RDMA hardware
            logger.warning("handshake server: failed to handle connection: %r", e)
        finally:
            if held_lock:
                try:
                    self._checkpoint_lock.r_release()
                except Exception:
                    logger.exception("handshake server: r_release failed")
            if peer_addr is not None:
                with self._peers_lock:
                    self._peers.pop(peer_addr, None)
            try:
                conn.close()
            except Exception:
                pass


# --- Device helpers -----------------------------------------------------------


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Return True iff ``a`` and ``b`` refer to the same physical device.

    Compares device type AND index. A ``cuda`` device whose index is ``None``
    is treated as the current default index (``torch.cuda.current_device()``)
    so a destination explicitly tagged ``cuda:0`` matches a transport device
    of bare ``cuda`` on a single-GPU setup.
    """
    if a.type != b.type:
        return False
    if a.type != "cuda":
        return True

    def _index(dev: torch.device) -> int:
        if dev.index is not None:
            return dev.index
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0

    return _index(a) == _index(b)


# --- Snapshot helpers ---------------------------------------------------------


def _spill_to_pinned_cpu(t: torch.Tensor) -> torch.Tensor:
    """Copy ``t``'s raw storage into a freshly allocated pinned CPU buffer.

    Used by the sender when a GPU tensor would push the snapshot footprint
    past ``max_gpu_snapshot_bytes``. The returned 1-D ``uint8`` tensor owns
    its storage so it can be registered with the RDMA NIC and read by
    receivers without holding a reference to the live GPU tensor.
    """
    nbytes = t.untyped_storage().nbytes()
    cpu = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    cpu.copy_(t.view(torch.uint8), non_blocking=False)
    return cpu


# --- RDMA read helper ---------------------------------------------------------


def _rdma_read(transport: object, mutable_view: object, remote_buffer: object) -> None:
    """Issue a one-sided RDMA read and surface a non-zero status as an error.

    ``mutable_view`` must come from ``RdmaMemory.to_mutable_view()`` — the
    torchcomms ``read`` binding requires an ``RdmaMemoryMutableView`` because
    the transfer writes into the local buffer. The C++ binding returns an
    ``int`` status code; anything non-zero is a failed transfer that would
    otherwise be silently ignored.
    """
    # pyre-ignore[16]
    rc = transport.read(mutable_view, remote_buffer)
    if rc:
        raise RuntimeError(f"RDMA read failed with status {rc}")


# --- TCP framing helpers ------------------------------------------------------


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("<Q", len(payload)))
    sock.sendall(payload)


def _recv_frame(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 8)
    (length,) = struct.unpack("<Q", header)
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before frame complete")
        buf.extend(chunk)
    return bytes(buf)
