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
import time
from contextlib import nullcontext
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

# Absolute safety deadline for a single peer transfer once the reader-lifetime
# fence has been acquired. This is intentionally *much* larger than the per-step
# ``timeout`` (which bounds control-plane RPCs, not the bulk RDMA read). The
# fence is released on DONE / dead-connection detection long before this fires;
# it exists only as a backstop so a permanently wedged-but-connected peer cannot
# pin a snapshot generation forever. See BLOCKER B2.
_DEFAULT_MAX_TRANSFER_SECONDS = 3600.0

# Polling interval used while the handshake handler waits for DONE. Each poll
# blocks on the socket for at most this long; TCP keepalive failures, EOF, or a
# DONE frame end the wait immediately. Kept short so liveness (a dropped socket)
# is detected promptly without busy-waiting.
_FENCE_POLL_INTERVAL_SECONDS = 1.0


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
    # CUDA event recorded on the copy stream after the snapshot's D2H/clone
    # work was enqueued. ``send_checkpoint`` must wait on it before publishing
    # the control record as READY so receivers never RDMA-read partially
    # produced bytes. ``None`` on the CPU path (no async work to fence).
    cuda_event: object = None


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
        max_transfer_seconds: absolute safety deadline (in seconds) for a single
            peer transfer once the reader-lifetime fence is held. Distinct from
            ``timeout`` (which bounds per-step control-plane RPCs). The fence is
            normally released on DONE or on a detected dead connection (TCP
            keepalive failure / socket error / EOF); this deadline is only a
            backstop against a wedged-but-still-connected peer. Defaults to a
            generous value (1 hour) so a healthy but slow transfer over a large
            checkpoint / slow fabric is never released prematurely. See
            BLOCKER B2.
    """

    def __init__(
        self,
        device: torch.device,
        timeout: timedelta,
        state_dict: Optional[Callable[[], object]] = None,
        max_gpu_snapshot_bytes: int = 4 << 30,
        max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
        handshake_host: Optional[str] = None,
        max_transfer_seconds: float = _DEFAULT_MAX_TRANSFER_SECONDS,
    ) -> None:
        self._device = device
        self._timeout = timeout
        self._state_dict_fn = state_dict
        self._max_gpu_snapshot_bytes = max_gpu_snapshot_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._handshake_host_override = handshake_host
        self._max_transfer_seconds = max_transfer_seconds

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

        # Dedicated copy stream for the snapshot's GPU clone / D2H spill so
        # the staging is stream-ordered against a stream we control and can
        # fence with an event before publishing READY (mirrors
        # ``HTTPTransport``). ``None`` on the CPU path, where copies are
        # synchronous and need no fence.
        self._copy_stream: Optional[object] = (
            torch.cuda.Stream() if self._device.type == "cuda" else None
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

        # The reader fence may legitimately be held for an entire (slow) peer
        # transfer, so the RWLock timeout must cover the absolute transfer
        # safety deadline -- NOT the per-step ``timeout``. Bounding it by the
        # short per-step timeout would make ``disallow_checkpoint()``'s
        # ``w_acquire`` raise on a healthy-but-slow transfer (BLOCKER B2),
        # defeating the fence. A small margin is added so the handler's own
        # backstop fires first and releases cleanly. (See ``_wait_for_done``.)
        self._checkpoint_lock = RWLock(
            timeout=self._max_transfer_seconds + _FENCE_POLL_INTERVAL_SECONDS + 5.0
        )
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
        there is nothing to probe and pinned-CPU staging is irrelevant.

        For a cuda device the probe is **registration-only**: it allocates a
        tiny CUDA tensor and attempts ``RdmaMemory(t)``. Registration is the
        load-bearing signal — it is exactly what ``_build_snapshot`` does for
        every GPU tensor, and it is what fails first when GDR is unavailable
        (IB NIC present, GPUDirect path not usable for this GPU/NIC/driver).

        We deliberately do **not** attempt a self-loopback ``connect()`` +
        ``read()`` here: torchcomms ``connect()`` expects a *separate* peer
        connecting from the other end (server/client each ``bind()`` then
        ``connect()`` to the other's URL), so pointing a transport at its own
        bind address has no second endpoint and can block indefinitely rather
        than raise — which would wedge ``__init__`` on every GPU node. A real
        end-to-end GPU read is validated by the hardware test
        (``examples/rdma_transport_sim.py --real --device cuda``), not at
        construction time.

        On registration failure the transport routes ALL GPU tensors through
        the pinned-CPU staging path (see ``_build_snapshot``) so checkpoints
        still work, just without zero-copy GPU transfers. The chosen mode is
        logged so operators can confirm the path that was taken.
        """
        if self._device.type != "cuda":
            return True

        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaMemory,
        )

        try:
            probe_tensor = torch.zeros(8, dtype=torch.uint8, device=self._device)
            # Register and immediately drop it. Success means GDR registration
            # works for this device; that is the only thing we assert here.
            RdmaMemory(probe_tensor, cache_reg=False)
            logger.info(
                "RDMATransport GDR registration probe succeeded on %s; GPU "
                "tensors stay on GPU (subject to max_gpu_snapshot_bytes)",
                self._device,
            )
            return True
        except Exception as e:
            logger.warning(
                "RDMATransport GDR registration probe FAILED on %s (%r); "
                "routing ALL GPU tensors through pinned-CPU staging for "
                "checkpoints",
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

        # B1: the snapshot's GPU clone / D2H spill is stream-ordered and may
        # still be in flight. Wait on the recorded copy-stream event before we
        # flip the control record to READY so a receiver can never RDMA-read
        # partially produced bytes. No READY write may precede this wait.
        self._wait_snapshot_ready(snapshot)

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

    def _wait_snapshot_ready(self, snapshot: _SnapshotGeneration) -> None:
        """Block until the snapshot's CUDA copy work has fully completed.

        Synchronizes on the event recorded over the copy stream in
        ``_build_snapshot``. This is the B1 invariant: the caller must invoke
        this before publishing a READY control record so receivers never see
        partially produced bytes. A no-op on the CPU path (no event).
        """
        event = snapshot.cuda_event
        if event is not None:
            event.synchronize()

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

        # On CUDA, run all clones / D2H spills on the dedicated copy stream so
        # the work is stream-ordered against a stream we own. We make the copy
        # stream wait on the current (producer) stream first, so it observes
        # the latest writes to the source tensors, then record an event after
        # enqueuing the copies. ``send_checkpoint`` waits on that event before
        # publishing READY (see B1 in the blockers plan). On CPU there is no
        # async work and ``_copy_stream`` is ``None``.
        is_cuda = self._device.type == "cuda"
        copy_stream = self._copy_stream if is_cuda else None
        if copy_stream is not None:
            copy_stream.wait_stream(torch.cuda.current_stream())
        stream_ctx = (
            torch.cuda.stream(copy_stream) if copy_stream is not None else nullcontext()
        )

        cloned: List[torch.Tensor] = []
        tensor_mems: List[object] = []
        gpu_snapshot_bytes = 0
        gpu_budget_exceeded = False
        # When the startup GDR probe failed we cannot register CUDA tensors
        # directly with the NIC, so EVERY GPU tensor must be staged through
        # pinned CPU regardless of the GPU snapshot budget (B3).
        force_spill = not getattr(self, "_gdr_ok", True)
        with stream_ctx:
            for t in tensors:
                if t.device.type == "cuda":
                    t_bytes = t.untyped_storage().nbytes()
                    if (
                        force_spill
                        or gpu_budget_exceeded
                        or gpu_snapshot_bytes + t_bytes
                        > self._max_gpu_snapshot_bytes
                    ):
                        gpu_budget_exceeded = True
                        # Only thread the copy stream through on the real CUDA
                        # path; on the CPU device path ``copy_stream`` is None
                        # and the spill stays synchronous (single-arg call).
                        if copy_stream is not None:
                            cloned.append(_spill_to_pinned_cpu(t, copy_stream))
                        else:
                            cloned.append(_spill_to_pinned_cpu(t))
                    else:
                        cloned.append(t.clone())
                        gpu_snapshot_bytes += t_bytes
                else:
                    cloned.append(t.clone().contiguous())
                tensor_mems.append(RdmaMemory(cloned[-1], cache_reg=False))

        # Record the fence event on the copy stream after all snapshot copies
        # have been enqueued. The publishing path must wait on it before any
        # READY control-record write becomes visible to receivers.
        cuda_event: object = None
        if is_cuda:
            cuda_event = torch.cuda.Event()
            cuda_event.record(copy_stream)

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
            cuda_event=cuda_event,
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

        # Layout: <uint64 length><payload bytes><zero padding>.
        #
        # B6 — atomic publication. The length word is the commit point: a
        # reader treats length==0 as "no record" and only decodes the payload
        # once a non-zero length is observed. We therefore (1) clear the length
        # to 0 so a reader racing mid-write never sees a stale length over a
        # half-written payload, (2) write the full payload, and (3) write the
        # real length LAST. Because the length is a single aligned 8-byte word,
        # a reader observes either the old (zeroed) or the new length, never a
        # torn value, so it decodes either nothing or the complete new record.
        view = self._control_tensor.numpy()
        struct.pack_into("<Q", view, 0, 0)
        view[8 : 8 + len(payload)] = bytearray(payload)
        struct.pack_into("<Q", view, 0, len(payload))

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
            # Keepalive on the receiver side too: if the sender dies mid
            # transfer the receiver's blocking reads surface an error instead
            # of hanging, and the sender's matching fence socket sees the drop.
            _enable_tcp_keepalive(sock)
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
        # CUDA current device is thread-local; this handler runs in a fresh
        # thread and constructs torchcomms objects bound to ``self._device``.
        # Pin the device before any of that so a multi-GPU sender binds the
        # RdmaTransport / RdmaMemory to the right GPU (RISK B5a). CPU is a
        # no-op.
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        from torchcomms._transport import (  # type: ignore[import-not-found]
            RdmaTransport as _RdmaTransport,
        )

        timeout_s = self._timeout.total_seconds()
        peer_addr: Optional[bytes] = None
        held_lock = False
        try:
            # Keepalive turns a dead peer into a socket error / EOF on the
            # blocking DONE wait below, which is the liveness oracle that
            # releases the fence (BLOCKER B2). The per-step ``timeout`` only
            # bounds the small control-plane frames (peer_addr handshake), not
            # the bulk transfer.
            _enable_tcp_keepalive(conn)
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

            # Wait for the receiver's DONE. We deliberately do NOT release the
            # fence on a short idle timeout (a healthy but slow transfer would
            # be killed mid-read -> use-after-free). Instead we hold the fence
            # until one of:
            #   * DONE arrives                  -> released promptly,
            #   * the connection is detected
            #     dead (keepalive failure /
            #     socket error / EOF)           -> liveness release,
            #   * the absolute safety deadline  -> backstop release.
            done = self._wait_for_done(conn, peer_addr)
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

    def _wait_for_done(
        self, conn: socket.socket, peer_addr: bytes
    ) -> Optional[bytes]:
        """Hold the fence until DONE, a dead connection, or the safety deadline.

        Polls the socket with a short per-read timeout so a *dropped* peer is
        detected promptly (EOF / connection error -> the fence is released by
        the caller's ``finally``), while a *slow-but-alive* peer keeps blocking
        without tripping any idle deadline. The only time-based release is the
        generous absolute ``max_transfer_seconds`` backstop. Returns the DONE
        frame on success, otherwise ``None`` (caller treats both
        dead-connection and deadline as "release the fence").
        """
        deadline = time.monotonic() + self._max_transfer_seconds
        while not self._shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "handshake server: peer %r exceeded max_transfer_seconds "
                    "(%.1fs); releasing reader fence as a safety backstop",
                    peer_addr,
                    self._max_transfer_seconds,
                )
                return None
            conn.settimeout(min(_FENCE_POLL_INTERVAL_SECONDS, remaining))
            try:
                return _recv_frame(conn)
            except socket.timeout:
                # Idle but the connection is still up (keepalive would have
                # raised otherwise): the peer is alive, keep holding the fence.
                continue
            except (ConnectionError, OSError) as e:
                # Dead/dropped connection (incl. keepalive failure or EOF from
                # _recv_exact). Liveness says release the fence now.
                logger.warning(
                    "handshake server: peer %r connection lost before DONE "
                    "(%r); releasing reader fence",
                    peer_addr,
                    e,
                )
                return None
        return None


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


def _spill_to_pinned_cpu(
    t: torch.Tensor, copy_stream: Optional[object] = None
) -> torch.Tensor:
    """Copy ``t``'s raw storage into a freshly allocated pinned CPU buffer.

    Used by the sender when a GPU tensor would push the snapshot footprint
    past ``max_gpu_snapshot_bytes``. The returned 1-D ``uint8`` tensor owns
    its storage so it can be registered with the RDMA NIC and read by
    receivers without holding a reference to the live GPU tensor.

    When ``copy_stream`` is provided (the CUDA path), the D2H copy is issued
    asynchronously (``non_blocking=True``) on the caller's already-active copy
    stream; completion is fenced by the event recorded in ``_build_snapshot``,
    so the snapshot is only published READY after the copy has finished. With
    no copy stream (the CPU path) the copy is synchronous and complete on
    return.
    """
    nbytes = t.untyped_storage().nbytes()
    cpu = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    cpu.copy_(t.view(torch.uint8), non_blocking=copy_stream is not None)
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


# --- TCP keepalive ------------------------------------------------------------


def _enable_tcp_keepalive(
    sock: socket.socket,
    idle_s: int = 5,
    interval_s: int = 2,
    count: int = 3,
) -> None:
    """Turn on TCP keepalive so a *dead* peer is detected on the fence socket.

    This is the liveness oracle for the reader-lifetime fence (BLOCKER B2): a
    *slow-but-alive* peer keeps the connection (and therefore the fence) up,
    while a peer whose host has died / dropped off the network trips keepalive
    and surfaces a socket error to the blocked ``recv``, releasing the fence.

    ``TCP_KEEPIDLE``/``TCP_KEEPINTVL``/``TCP_KEEPCNT`` (and the macOS
    ``TCP_KEEPALIVE``) are best-effort: any that the platform lacks are
    skipped. ``SO_KEEPALIVE`` alone still works, just with OS-default timing.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        logger.warning("could not enable SO_KEEPALIVE on handshake socket")
        return
    # Per-connection tuning where the platform exposes it. Linux uses
    # TCP_KEEPIDLE/INTVL/CNT; macOS uses TCP_KEEPALIVE for the idle time.
    for opt_name, value in (
        ("TCP_KEEPIDLE", idle_s),
        ("TCP_KEEPALIVE", idle_s),  # macOS spelling of the idle time
        ("TCP_KEEPINTVL", interval_s),
        ("TCP_KEEPCNT", count),
    ):
        opt = getattr(socket, opt_name, None)
        if opt is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, opt, value)
        except OSError:
            logger.debug("could not set %s on handshake socket", opt_name)


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
