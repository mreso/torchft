# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import gc
import pickle
import socket
import struct
import sys
import threading
import time
import types
import weakref
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, List, Optional
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.checkpointing.pg_transport import (
    _prepare_state_dict,
    _StateDictMeta,
    _TensorMeta,
)
from torchft.checkpointing.rdma_transport import (
    _HANDSHAKE_DONE,
    _HANDSHAKE_READY,
    _PROTOCOL_VERSION,
    _RDMA_META_PREFIX,
    _RDMABootstrapMeta,
    _RDMAControlRecord,
    _RDMAManifest,
    _RDMATensorLeaf,
    _recv_frame,
    _send_frame,
    _SnapshotGeneration,
    RDMATransport,
)
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import (
    assertStateDictEqual,
    run_multi_recovery_test,
)


@dataclass
class _MockRdmaRemoteBuffer:
    """Pickleable stand-in for torchcomms RdmaRemoteBuffer used in tests."""

    addr: int
    nbytes: int
    rkey: int = 0


class _MockRdmaMemoryView:
    """Immutable view, mirrors torchcomms ``RdmaMemoryView``.

    Returned by ``RdmaMemory.to_view()`` and accepted by ``write()``. The real
    binding rejects this type for ``read()`` (reads write into the local
    buffer), so the mock does too — that is what catches the to_view/
    to_mutable_view contract bug.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def size(self) -> int:
        return self._tensor.numel() * self._tensor.element_size()


class _MockRdmaMemoryMutableView:
    """Mutable view, mirrors torchcomms ``RdmaMemoryMutableView``.

    Returned by ``RdmaMemory.to_mutable_view()`` and required by ``read()``.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def size(self) -> int:
        return self._tensor.numel() * self._tensor.element_size()


# ---------------------------------------------------------------------------
# Mocks for the torchcomms._transport module so we can drive the RDMA path
# without RDMA hardware. Memory is faked by an in-process registry mapping
# remote-buffer addrs to local tensor bytes; ``read()`` copies between them.
# ---------------------------------------------------------------------------


class _MockRdmaMemory:
    """Minimal mock of torchcomms RdmaMemory.

    The ``_registry`` is a ``WeakValueDictionary`` so it does not keep mocks
    alive. When the owning ``_SnapshotGeneration`` is dropped (or
    ``deregister()`` is called explicitly), the registry entry vanishes — this
    mirrors the production lifetime of ``RdmaMemory`` and lets tests detect
    stale-buffer reads against retired snapshots.
    """

    _next_addr: int = 1
    _addr_lock: threading.Lock = threading.Lock()
    _registry: "weakref.WeakValueDictionary[int, _MockRdmaMemory]" = (
        weakref.WeakValueDictionary()
    )

    # When True, constructing an ``RdmaMemory`` from a CUDA-backed tensor
    # raises, mirroring a node where GPUDirect RDMA registration is
    # unavailable (IB NIC present, GDR not usable). Safe default: do not
    # raise, so every existing test is unaffected. Tests flip this to drive
    # the B3 GDR-probe-failure fallback path.
    raise_on_cuda: bool = False

    def __init__(self, tensor: torch.Tensor, cache_reg: bool = False) -> None:
        if _MockRdmaMemory.raise_on_cuda and getattr(tensor, "device", None) is not None:
            if tensor.device.type == "cuda":
                raise RuntimeError(
                    "mock: GPUDirect RDMA registration unavailable for CUDA tensor"
                )
        self.tensor = tensor
        self.cache_reg = cache_reg
        with _MockRdmaMemory._addr_lock:
            self.addr = _MockRdmaMemory._next_addr
            _MockRdmaMemory._next_addr += 1
            _MockRdmaMemory._registry[self.addr] = self

    def to_remote_buffer(self) -> _MockRdmaRemoteBuffer:
        return _MockRdmaRemoteBuffer(
            addr=self.addr,
            nbytes=self.tensor.numel() * self.tensor.element_size(),
        )

    def to_view(self) -> "_MockRdmaMemoryView":
        return _MockRdmaMemoryView(self.tensor)

    def to_mutable_view(self) -> "_MockRdmaMemoryMutableView":
        return _MockRdmaMemoryMutableView(self.tensor)

    def deregister(self) -> None:
        """Drop this buffer from the global registry so reads against it fail."""
        with _MockRdmaMemory._addr_lock:
            _MockRdmaMemory._registry.pop(self.addr, None)

    @classmethod
    def reset_registry(cls) -> None:
        with cls._addr_lock:
            cls._registry.clear()
        cls.raise_on_cuda = False


@dataclass
class _MockRdmaTransportRecord:
    """Records what a ``MockRdmaTransport`` instance has been asked to do."""

    bind_addrs: List[bytes] = field(default_factory=list)
    connect_calls: List[bytes] = field(default_factory=list)
    reads: List[tuple] = field(default_factory=list)


class _MockRdmaTransport:
    """Mock of torchcomms RdmaTransport. Tracks bind/connect/read calls."""

    _next_id: int = 0
    _id_lock: threading.Lock = threading.Lock()
    instances: List["_MockRdmaTransport"] = []
    instances_lock: threading.Lock = threading.Lock()

    # Injectable per-read delay (seconds). Default 0 keeps existing tests
    # unaffected; tests that need a slow transfer set this to simulate a read
    # that outlasts the per-step timeout (BLOCKER B2).
    read_delay_s: float = 0.0

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.record = _MockRdmaTransportRecord()
        with _MockRdmaTransport.instances_lock:
            _MockRdmaTransport.instances.append(self)

    @staticmethod
    def supported() -> bool:
        return True

    def bind(self) -> bytes:
        with _MockRdmaTransport._id_lock:
            _MockRdmaTransport._next_id += 1
            addr = f"mock://transport/{_MockRdmaTransport._next_id}".encode()
        self.record.bind_addrs.append(addr)
        return addr

    def connect(self, peer_addr: bytes) -> int:
        self.record.connect_calls.append(peer_addr)
        self._connected = True
        return 0

    def connected(self) -> bool:
        return getattr(self, "_connected", False)

    def read(
        self, local_view: "_MockRdmaMemoryMutableView", remote_buffer: object
    ) -> int:
        # The real torchcomms ``read`` binding requires an
        # ``RdmaMemoryMutableView``; passing an immutable ``RdmaMemoryView``
        # (from ``to_view()``) raises TypeError. Model that contract so the
        # mock catches the bug instead of silently accepting either type.
        assert isinstance(local_view, _MockRdmaMemoryMutableView), (
            f"read() requires a mutable view (RdmaMemoryMutableView), got "
            f"{type(local_view).__name__}; use to_mutable_view()"
        )
        assert isinstance(remote_buffer, _MockRdmaRemoteBuffer), (
            f"unexpected remote_buffer type: {type(remote_buffer)}"
        )
        # Simulate a slow fabric / large transfer if a delay was injected. The
        # registry lookup happens AFTER the delay so a test can assert the
        # source buffer is still registered for the whole read.
        delay = getattr(self, "read_delay_s", 0.0)
        if delay:
            time.sleep(delay)
        src_mem = _MockRdmaMemory._registry.get(remote_buffer.addr)
        assert src_mem is not None, (
            f"mock read against unregistered addr {remote_buffer.addr}"
        )
        nbytes = remote_buffer.nbytes
        src_view = src_mem.tensor.view(torch.uint8)[:nbytes]
        local_view._tensor.view(torch.uint8)[:nbytes].copy_(src_view)
        self.record.reads.append((remote_buffer.addr, nbytes))
        return 0

    def write(
        self, local_view: "_MockRdmaMemoryView", remote_buffer: object
    ) -> int:
        assert isinstance(local_view, _MockRdmaMemoryView), (
            f"write() requires an immutable view (RdmaMemoryView), got "
            f"{type(local_view).__name__}; use to_view()"
        )
        assert isinstance(remote_buffer, _MockRdmaRemoteBuffer)
        dst_mem = _MockRdmaMemory._registry.get(remote_buffer.addr)
        assert dst_mem is not None, (
            f"mock write against unregistered addr {remote_buffer.addr}"
        )
        nbytes = remote_buffer.nbytes
        dst_mem.tensor.view(torch.uint8)[:nbytes].copy_(
            local_view._tensor.view(torch.uint8)[:nbytes]
        )
        return 0

    @classmethod
    def reset(cls) -> None:
        with cls.instances_lock:
            cls.instances.clear()
        with cls._id_lock:
            cls._next_id = 0
        # Clear any injected per-read delay so it does not leak across tests.
        cls.read_delay_s = 0.0


def _install_torchcomms_mock() -> object:
    """Insert mock ``torchcomms._transport`` module into ``sys.modules``."""
    pkg = sys.modules.get("torchcomms")
    created_pkg = pkg is None
    if pkg is None:
        pkg = types.ModuleType("torchcomms")
        sys.modules["torchcomms"] = pkg

    mod = types.ModuleType("torchcomms._transport")
    mod.RdmaTransport = _MockRdmaTransport
    mod.RdmaMemory = _MockRdmaMemory
    sys.modules["torchcomms._transport"] = mod
    pkg._transport = mod  # type: ignore[attr-defined]

    return ("torchcomms._transport", "torchcomms" if created_pkg else None)


def _uninstall_torchcomms_mock(token: object) -> None:
    transport_key, pkg_key = token  # type: ignore[misc]
    sys.modules.pop(transport_key, None)
    if pkg_key is not None:
        sys.modules.pop(pkg_key, None)


# ---------------------------------------------------------------------------
# Existing tests, preserved.
# ---------------------------------------------------------------------------


class TestRDMATransportProtocol(TestCase):
    """Tests for the wire-protocol dataclasses."""

    def test_protocol_dataclass_roundtrip(self) -> None:
        control_buf = _MockRdmaRemoteBuffer(addr=0xDEAD, nbytes=4096, rkey=42)
        manifest_buf = _MockRdmaRemoteBuffer(addr=0xBEEF, nbytes=2048, rkey=43)
        tensor_buf = _MockRdmaRemoteBuffer(addr=0xCAFE, nbytes=128, rkey=44)

        bootstrap = _RDMABootstrapMeta(
            version=_PROTOCOL_VERSION,
            bind_addr=b"",
            control_remote_buffer=control_buf,
            control_buffer_nbytes=4096,
            handshake_host="host.example.com",
            handshake_port=5555,
        )
        decoded_bootstrap = pickle.loads(pickle.dumps(bootstrap))
        self.assertEqual(decoded_bootstrap, bootstrap)
        self.assertEqual(decoded_bootstrap.control_remote_buffer, control_buf)
        self.assertEqual(decoded_bootstrap.handshake_host, "host.example.com")

        control = _RDMAControlRecord(
            version=_PROTOCOL_VERSION,
            step=7,
            status="READY",
            manifest_nbytes=2048,
            manifest_remote_buffer=manifest_buf,
        )
        decoded_control = pickle.loads(pickle.dumps(control))
        self.assertEqual(decoded_control, control)

        tensor_meta = _TensorMeta(
            shape=torch.Size([4, 4]),
            dtype=torch.float32,
            storage_offset=0,
            stride=(4, 1),
            nbytes=64,
        )
        leaf = _RDMATensorLeaf(meta=tensor_meta, remote_buffer=tensor_buf)
        manifest = _RDMAManifest(
            step=7,
            treespec=None,
            paths=[("rank",)],
            leaves=[leaf, "non-tensor", 1234],
        )
        decoded_manifest = pickle.loads(pickle.dumps(manifest))
        self.assertEqual(decoded_manifest.step, 7)
        self.assertEqual(decoded_manifest.leaves[1], "non-tensor")
        self.assertEqual(decoded_manifest.leaves[2], 1234)
        self.assertEqual(decoded_manifest.leaves[0].meta, tensor_meta)
        self.assertEqual(decoded_manifest.leaves[0].remote_buffer, tensor_buf)

    def test_metadata_format(self) -> None:
        transport = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            meta = transport.metadata()
            # In fallback mode the metadata is the HTTP URL — no rdma: prefix.
            self.assertFalse(meta.startswith(_RDMA_META_PREFIX))
            self.assertTrue(meta.startswith("http://"))
        finally:
            transport.shutdown()

        # Now exercise the explicit encoding helpers used by the RDMA codepath.
        bootstrap = _RDMABootstrapMeta(
            version=_PROTOCOL_VERSION,
            bind_addr=b"",
            control_remote_buffer=_MockRdmaRemoteBuffer(
                addr=0x100, nbytes=4096, rkey=1
            ),
            control_buffer_nbytes=4096,
            handshake_host="host.example.com",
            handshake_port=6789,
        )
        encoded = f"{_RDMA_META_PREFIX}{base64.b64encode(pickle.dumps(bootstrap)).decode()}"
        self.assertTrue(encoded.startswith(_RDMA_META_PREFIX))

        body = encoded.removeprefix(_RDMA_META_PREFIX)
        decoded = pickle.loads(base64.b64decode(body))
        self.assertEqual(decoded, bootstrap)


class TestRDMATransportFallback(TestCase):
    """Verify the HTTPTransport fallback when RDMA isn't available."""

    def test_fallback_to_http(self) -> None:
        # On this machine torchcomms isn't installed -> fallback is automatic.
        transport: RDMATransport[Dict[str, object]] = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            self.assertIsNotNone(transport._fallback)
            self.assertIsInstance(transport._fallback, HTTPTransport)
            self.assertIsNone(transport._rdma)

            metadata = transport.metadata()
            self.assertEqual(metadata, transport._fallback.metadata())

            state_dict: Dict[str, object] = {
                "tensor": torch.tensor([1.0, 2.0, 3.0]),
                "scalar": 99,
            }

            transport.send_checkpoint(
                dst_ranks=[],
                step=42,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )

            recovered = transport.recv_checkpoint(
                src_rank=0,
                metadata=metadata,
                step=42,
                timeout=timedelta(seconds=10),
            )
            assertStateDictEqual(self, recovered, state_dict)

            # disallow_checkpoint shouldn't crash and should re-enter the disallowed state.
            transport.disallow_checkpoint()
            self.assertTrue(transport._fallback._disallowed)
        finally:
            transport.shutdown()


class TestStateDictPreparation(TestCase):
    """Sanity check: we depend on _prepare_state_dict from pg_transport."""

    def test_state_dict_preparation(self) -> None:
        device = torch.device("cpu")
        state_dict = {
            "weights": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "nested": {
                "bias": torch.zeros(4),
                "step": 42,
            },
            "name": "model",
        }
        meta, tensors = _prepare_state_dict(state_dict, step=5, device=device)

        self.assertEqual(meta.step, 5)
        # Two tensor leaves, one int, one string.
        self.assertEqual(len(meta.paths), 4)
        tensor_metas = [m for m in meta.non_tensor_leaves if isinstance(m, _TensorMeta)]
        self.assertEqual(len(tensor_metas), 2)
        self.assertEqual(len(tensors), 2)

        # Each prepared tensor is a uint8 view of the same storage as the original.
        for t in tensors:
            self.assertEqual(t.dtype, torch.uint8)

        # Non-tensor leaves are passed through.
        self.assertIn("model", meta.non_tensor_leaves)
        self.assertIn(42, meta.non_tensor_leaves)


class TestRWLockLifecycle(TestCase):
    """Verify the disallow/allow lifecycle through the fallback transport."""

    def test_rwlock_lifecycle(self) -> None:
        transport: RDMATransport[Dict[str, object]] = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            # The HTTPTransport fallback starts disallowed.
            fallback = transport._fallback
            self.assertTrue(fallback._disallowed)
            self.assertTrue(fallback._checkpoint_lock.w_locked())

            # send_checkpoint releases the writer lock.
            state_dict = {"tensor": torch.tensor([1.0, 2.0])}
            transport.send_checkpoint(
                dst_ranks=[],
                step=1,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )
            self.assertFalse(fallback._disallowed)
            self.assertFalse(fallback._checkpoint_lock.w_locked())

            # A reader can hold the lock without blocking.
            reader_started = threading.Event()
            release_reader = threading.Event()

            def reader() -> None:
                with fallback._checkpoint_lock.r_lock():
                    reader_started.set()
                    release_reader.wait(timeout=5)

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            self.assertTrue(reader_started.wait(timeout=2))

            # disallow_checkpoint must block until the reader releases.
            disallow_done = threading.Event()

            def disallow() -> None:
                transport.disallow_checkpoint()
                disallow_done.set()

            d = threading.Thread(target=disallow, daemon=True)
            d.start()
            # Reader is still active -> writer cannot acquire yet.
            self.assertFalse(disallow_done.wait(timeout=0.2))

            release_reader.set()
            t.join(timeout=2)
            self.assertTrue(disallow_done.wait(timeout=2))
            d.join(timeout=2)

            self.assertTrue(fallback._disallowed)
            self.assertTrue(fallback._checkpoint_lock.w_locked())
        finally:
            transport.shutdown()


class TestSnapshotGeneration(TestCase):
    """Build a _SnapshotGeneration without RDMA hardware."""

    def test_snapshot_generation_build(self) -> None:
        device = torch.device("cpu")
        state_dict = {
            "weights": torch.arange(8, dtype=torch.float32),
            "bias": torch.tensor([1.0, 2.0]),
        }
        meta, tensors = _prepare_state_dict(state_dict, step=3, device=device)

        # Mimic what _build_snapshot does without depending on RdmaMemory: clone
        # tensors and capture their storage in a generation record.
        cloned = [t.clone() for t in tensors]
        manifest = _RDMAManifest(
            step=3,
            treespec=meta.treespec,
            paths=meta.paths,
            leaves=[
                _RDMATensorLeaf(meta=tm, remote_buffer=_MockRdmaRemoteBuffer(addr=i, nbytes=tm.nbytes))
                if isinstance(tm, _TensorMeta)
                else tm
                for i, tm in enumerate(meta.non_tensor_leaves)
            ],
        )
        manifest_bytes = pickle.dumps(manifest)
        manifest_tensor = torch.frombuffer(manifest_bytes, dtype=torch.uint8).clone()

        snapshot = _SnapshotGeneration(
            generation=1,
            step=3,
            manifest_tensor=manifest_tensor,
            manifest_mem=None,
            tensor_snapshots=cloned,
            tensor_mems=[None] * len(cloned),
        )

        self.assertEqual(snapshot.generation, 1)
        self.assertEqual(snapshot.step, 3)
        self.assertEqual(len(snapshot.tensor_snapshots), 2)
        # Snapshots are independent copies.
        snapshot.tensor_snapshots[0].zero_()
        self.assertNotEqual(snapshot.tensor_snapshots[0].sum().item(), tensors[0].sum().item())

        # Manifest roundtrips without loss.
        decoded = pickle.loads(bytes(snapshot.manifest_tensor.numpy()))
        self.assertEqual(decoded.step, 3)
        self.assertEqual(len(decoded.leaves), len(meta.non_tensor_leaves))


class TestRDMATransportE2E(TestCase):
    """Drive the full transport through the multi-recovery harness."""

    def test_fallback_e2e_with_transport_test_harness(self) -> None:
        device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            return RDMATransport[Dict[str, object]](
                device=device, timeout=timedelta(seconds=10)
            )

        run_multi_recovery_test(self, init, device=device)


class TestRDMAAvailabilityMock(TestCase):
    """Verify _rdma_available correctly reports False when torchcomms is missing."""

    def test_rdma_available_false_when_torchcomms_missing(self) -> None:
        from torchft.checkpointing import rdma_transport as mod

        # On this machine torchcomms isn't installed.
        self.assertFalse(mod._rdma_available())

    def test_fallback_is_used_when_rdma_unavailable(self) -> None:
        from torchft.checkpointing import rdma_transport as mod

        with patch.object(mod, "_rdma_available", return_value=False):
            transport = RDMATransport(
                device=torch.device("cpu"), timeout=timedelta(seconds=10)
            )
            try:
                self.assertIsInstance(transport._fallback, HTTPTransport)
            finally:
                transport.shutdown()


# ---------------------------------------------------------------------------
# RDMA-path tests using the torchcomms mock. These cover the codepaths that
# only run when ``_rdma_available()`` returns True.
# ---------------------------------------------------------------------------


class TestRDMAPathMocked(TestCase):
    """Drive the RDMA codepath end-to-end with a mocked torchcomms module."""

    def setUp(self) -> None:
        _MockRdmaTransport.reset()
        self._mock_token = _install_torchcomms_mock()
        from torchft.checkpointing import rdma_transport as mod

        self._available_patch = patch.object(mod, "_rdma_available", return_value=True)
        self._available_patch.start()

    def tearDown(self) -> None:
        self._available_patch.stop()
        _uninstall_torchcomms_mock(self._mock_token)
        _MockRdmaTransport.reset()
        _MockRdmaMemory.reset_registry()

    def _new_transport(self) -> RDMATransport:
        return RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )

    def test_metadata_returns_rdma_prefix(self) -> None:
        transport = self._new_transport()
        try:
            self.assertTrue(transport._rdma)
            meta = transport.metadata()
            self.assertTrue(meta.startswith(_RDMA_META_PREFIX))

            decoded = pickle.loads(
                base64.b64decode(meta[len(_RDMA_META_PREFIX):])
            )
            self.assertIsInstance(decoded, _RDMABootstrapMeta)
            # The advertised host is whatever the transport resolved (a
            # resolvable hostname, or loopback when it is not resolvable). It
            # must be non-empty and match the live transport value.
            self.assertTrue(decoded.handshake_host)
            self.assertEqual(decoded.handshake_host, transport._handshake_host)
            self.assertGreater(decoded.handshake_port, 0)
            self.assertEqual(decoded.control_buffer_nbytes, 64 * 1024)
            self.assertIsInstance(decoded.control_remote_buffer, _MockRdmaRemoteBuffer)
        finally:
            transport.shutdown()

    def test_handshake_host_override(self) -> None:
        """An explicit handshake_host is advertised verbatim for cross-host use."""
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            handshake_host="10.1.2.3",
        )
        try:
            self.assertEqual(transport._handshake_host, "10.1.2.3")
            decoded = pickle.loads(
                base64.b64decode(transport.metadata()[len(_RDMA_META_PREFIX):])
            )
            self.assertEqual(decoded.handshake_host, "10.1.2.3")
        finally:
            transport.shutdown()

    def test_handshake_host_env_override(self) -> None:
        """TORCHFT_RDMA_HANDSHAKE_HOST is honored when no arg is passed."""
        with patch.dict("os.environ", {"TORCHFT_RDMA_HANDSHAKE_HOST": "host.example"}):
            transport = RDMATransport(
                device=torch.device("cpu"), timeout=timedelta(seconds=10)
            )
            try:
                self.assertEqual(transport._handshake_host, "host.example")
            finally:
                transport.shutdown()

    def test_snapshot_build_creates_rdma_memory(self) -> None:
        transport = self._new_transport()
        try:
            state_dict = {
                "w": torch.arange(8, dtype=torch.float32),
                "b": torch.tensor([1.0, 2.0]),
                "step": 7,
            }
            transport.send_checkpoint(
                dst_ranks=[1],
                step=7,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )
            snap = transport._current_snapshot
            self.assertIsNotNone(snap)
            self.assertEqual(snap.step, 7)
            self.assertEqual(len(snap.tensor_snapshots), 2)
            self.assertEqual(len(snap.tensor_mems), 2)
            for m in snap.tensor_mems:
                self.assertIsInstance(m, _MockRdmaMemory)
            self.assertIsInstance(snap.manifest_mem, _MockRdmaMemory)
            self.assertGreater(snap.manifest_tensor.numel(), 0)
        finally:
            transport.shutdown()

    def test_control_record_state_transitions(self) -> None:
        transport = self._new_transport()
        try:
            # After init we must be DISALLOWED at step -1.
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "DISALLOWED")
            self.assertEqual(rec.step, -1)

            transport.send_checkpoint(
                dst_ranks=[1],
                step=11,
                state_dict={"t": torch.tensor([3.14])},
                timeout=timedelta(seconds=10),
            )
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "READY")
            self.assertEqual(rec.step, 11)
            self.assertGreater(rec.manifest_nbytes, 0)
            self.assertIsNotNone(rec.manifest_remote_buffer)

            transport.disallow_checkpoint()
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "DISALLOWED")
            self.assertEqual(rec.step, 11)
            self.assertIsNone(rec.manifest_remote_buffer)
        finally:
            transport.shutdown()

    def _read_control(self, transport: RDMATransport) -> _RDMAControlRecord:
        view = transport._control_tensor.numpy()
        (length,) = struct.unpack_from("<Q", view, 0)
        self.assertGreater(length, 0)
        return pickle.loads(bytes(view[8 : 8 + length]))

    def test_per_peer_transport_creation(self) -> None:
        """Two concurrent handshakes -> two distinct sender RdmaTransport instances."""
        transport = self._new_transport()
        try:
            # Allow snapshot reading by publishing one.
            transport.send_checkpoint(
                dst_ranks=[1, 2],
                step=3,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            sender_count_before = len(_MockRdmaTransport.instances)

            # Issue two TCP handshakes from different ephemeral "peers".
            sockets = [
                socket.create_connection(
                    ("localhost", transport._handshake_port), timeout=5
                )
                for _ in range(2)
            ]
            try:
                sender_addrs = []
                for i, s in enumerate(sockets):
                    s.settimeout(5)
                    _send_frame(s, f"mock://peer/{i}".encode())
                    sender_addrs.append(_recv_frame(s))
                    self.assertEqual(_recv_frame(s), _HANDSHAKE_READY)

                # Each handshake should have created a fresh sender-side
                # RdmaTransport, distinct from any others. The receiver-side
                # transports in this test are simulated by raw TCP sockets,
                # so only the sender's per-peer instances appear in the
                # MockRdmaTransport registry.
                self.assertEqual(
                    len(_MockRdmaTransport.instances) - sender_count_before, 2
                )
                self.assertEqual(len(set(sender_addrs)), 2)

                with transport._peers_lock:
                    self.assertEqual(len(transport._peers), 2)

                for s in sockets:
                    _send_frame(s, _HANDSHAKE_DONE)
            finally:
                for s in sockets:
                    try:
                        s.close()
                    except Exception:
                        pass

            # After receivers signal DONE, peers entries should drain.
            self._wait_for(lambda: len(transport._peers) == 0, timeout=5)
        finally:
            transport.shutdown()

    def test_reader_lifecycle_holds_rwlock(self) -> None:
        """Active receiver handshake fences ``disallow_checkpoint()``."""
        transport = self._new_transport()
        try:
            transport.send_checkpoint(
                dst_ranks=[1],
                step=4,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            # Start a "receiver" by opening the handshake socket and not
            # sending DONE yet.
            sock = socket.create_connection(
                ("localhost", transport._handshake_port), timeout=5
            )
            try:
                sock.settimeout(5)
                _send_frame(sock, b"mock://reader/1")
                _ = _recv_frame(sock)  # sender_addr
                self.assertEqual(_recv_frame(sock), _HANDSHAKE_READY)

                # Reader is "in flight"; w_lock should be held by the reader
                # via r_acquire (which acquires the underlying w_lock with
                # the first reader).
                self._wait_for(
                    lambda: transport._checkpoint_lock.w_locked(), timeout=2
                )

                # disallow_checkpoint must block on the active reader.
                disallow_done = threading.Event()

                def disallow() -> None:
                    transport.disallow_checkpoint()
                    disallow_done.set()

                t = threading.Thread(target=disallow, daemon=True)
                t.start()
                self.assertFalse(disallow_done.wait(timeout=0.3))

                # Now release the reader.
                _send_frame(sock, _HANDSHAKE_DONE)

                self.assertTrue(disallow_done.wait(timeout=5))
                t.join(timeout=2)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            transport.shutdown()

    def test_shutdown_clears_state(self) -> None:
        transport = self._new_transport()
        transport.send_checkpoint(
            dst_ranks=[1],
            step=5,
            state_dict={"t": torch.tensor([1.0])},
            timeout=timedelta(seconds=10),
        )
        self.assertIsNotNone(transport._current_snapshot)
        self.assertIsNotNone(transport._control_mem)

        transport.shutdown()

        self.assertIsNone(transport._current_snapshot)
        self.assertIsNone(transport._previous_snapshot)
        self.assertIsNone(transport._control_mem)
        self.assertIsNone(transport._control_tensor)
        self.assertEqual(len(transport._peers), 0)

    def test_recv_rejects_non_rdma_metadata(self) -> None:
        transport = self._new_transport()
        try:
            with self.assertRaisesRegex(RuntimeError, _RDMA_META_PREFIX):
                transport.recv_checkpoint(
                    src_rank=0,
                    metadata="http://example.com/checkpoint/",
                    step=1,
                    timeout=timedelta(seconds=1),
                )
        finally:
            transport.shutdown()

    def test_recv_rejects_version_mismatch(self) -> None:
        transport = self._new_transport()
        try:
            bogus = _RDMABootstrapMeta(
                version=_PROTOCOL_VERSION + 99,
                bind_addr=b"",
                control_remote_buffer=_MockRdmaRemoteBuffer(addr=1, nbytes=64),
                control_buffer_nbytes=64,
                handshake_host="localhost",
                handshake_port=transport._handshake_port,
            )
            meta = (
                f"{_RDMA_META_PREFIX}"
                f"{base64.b64encode(pickle.dumps(bogus)).decode()}"
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                transport.recv_checkpoint(
                    src_rank=0,
                    metadata=meta,
                    step=1,
                    timeout=timedelta(seconds=1),
                )
        finally:
            transport.shutdown()

    def test_recv_checkpoint_full_path(self) -> None:
        """End-to-end RDMA path with a single sender and receiver mock."""
        sender = self._new_transport()
        try:
            state_dict = {
                "w": torch.arange(8, dtype=torch.float32),
                "b": torch.tensor([10.0, 20.0, 30.0]),
                "step": 99,
                "name": "my-model",
            }
            sender.send_checkpoint(
                dst_ranks=[1],
                step=99,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )

            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=99,
                    timeout=timedelta(seconds=10),
                )
                # Don't use assertStateDictEqual since it imports DTensor;
                # check key tensors directly.
                torch.testing.assert_close(got["w"], state_dict["w"])
                torch.testing.assert_close(got["b"], state_dict["b"])
                self.assertEqual(got["step"], 99)
                self.assertEqual(got["name"], "my-model")
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    @staticmethod
    def _wait_for(pred, timeout: float) -> None:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.01)
        raise AssertionError(f"condition not met within {timeout}s")


# ---------------------------------------------------------------------------
# Phase 2: locking & concurrency refinements.
# ---------------------------------------------------------------------------


class _RDMAMockBase(TestCase):
    """Common setup/teardown for tests that drive the mocked RDMA path."""

    def setUp(self) -> None:
        _MockRdmaTransport.reset()
        self._mock_token = _install_torchcomms_mock()
        from torchft.checkpointing import rdma_transport as mod

        self._available_patch = patch.object(
            mod, "_rdma_available", return_value=True
        )
        self._available_patch.start()

    def tearDown(self) -> None:
        self._available_patch.stop()
        _uninstall_torchcomms_mock(self._mock_token)
        _MockRdmaTransport.reset()
        _MockRdmaMemory.reset_registry()

    def _new_transport(
        self,
        timeout: timedelta = timedelta(seconds=10),
        state_dict_fn: Optional[Callable[[], object]] = None,
        max_transfer_seconds: Optional[float] = None,
    ) -> RDMATransport:
        kwargs = {}
        if max_transfer_seconds is not None:
            kwargs["max_transfer_seconds"] = max_transfer_seconds
        return RDMATransport(
            device=torch.device("cpu"),
            timeout=timeout,
            state_dict=state_dict_fn,
            **kwargs,
        )

    def _open_handshake(
        self,
        transport: RDMATransport,
        peer_addr: bytes = b"mock://peer",
    ) -> socket.socket:
        sock = socket.create_connection(
            ("localhost", transport._handshake_port), timeout=5
        )
        sock.settimeout(5)
        _send_frame(sock, peer_addr)
        _ = _recv_frame(sock)  # sender_addr
        ready = _recv_frame(sock)
        if ready != _HANDSHAKE_READY:
            raise AssertionError(f"unexpected handshake reply: {ready!r}")
        return sock

    def _read_control(self, transport: RDMATransport) -> _RDMAControlRecord:
        view = transport._control_tensor.numpy()
        (length,) = struct.unpack_from("<Q", view, 0)
        if length == 0:
            raise AssertionError("control record is empty")
        return pickle.loads(bytes(view[8 : 8 + length]))

    @staticmethod
    def _wait_for(pred, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.01)
        raise AssertionError(f"condition not met within {timeout}s")


class TestRDMAPhase2Locking(_RDMAMockBase):
    """Phase 2: stress-tests for the reader fence + lifecycle transitions."""

    def test_recv_checkpoint_holds_rlock_through_full_path(self) -> None:
        """Sender's r_lock is held for the entire ``recv_checkpoint`` body.

        The earlier locking tests stopped after the TCP ``READY`` frame, which
        meant a regression that sent ``DONE`` too early — or skipped one of the
        post-handshake reads — would still be green. This drives the full
        receive path (control record + manifest + tensors + ``DONE``) through
        a real ``recv_checkpoint`` call, with the receiver wedged inside
        ``_read_tensors`` so the test can observe the sender's reader fence
        while RDMA reads are in flight.
        """
        sender = self._new_transport()
        try:
            payload = {
                "w": torch.arange(8, dtype=torch.float32),
                "b": torch.tensor([10.0, 20.0, 30.0]),
                "step": 99,
            }
            sender.send_checkpoint(
                dst_ranks=[1],
                step=99,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            # ``state_dict_fn`` runs inside ``_read_tensors`` after the
            # control record and manifest have been pulled but before the
            # receiver sends ``DONE``. Wedging it here exercises every read
            # in the post-handshake path while the sender is still holding
            # the reader lock.
            entered = threading.Event()
            release = threading.Event()

            def gated_state_dict() -> Dict[str, torch.Tensor]:
                entered.set()
                if not release.wait(timeout=10):
                    raise AssertionError("test bug: gate never released")
                return {
                    "w": torch.zeros(8, dtype=torch.float32),
                    "b": torch.zeros(3, dtype=torch.float32),
                }

            recv_result: Dict[str, object] = {}
            recv_done = threading.Event()
            recv_error: List[BaseException] = []

            receiver = self._new_transport(state_dict_fn=gated_state_dict)
            try:
                def do_recv() -> None:
                    try:
                        recv_result["v"] = receiver.recv_checkpoint(
                            src_rank=0,
                            metadata=metadata,
                            step=99,
                            timeout=timedelta(seconds=10),
                        )
                    except BaseException as exc:  # pragma: no cover - test guard
                        recv_error.append(exc)
                    finally:
                        recv_done.set()

                t = threading.Thread(target=do_recv, daemon=True)
                t.start()

                # Wait until the receiver is wedged inside ``_read_tensors``.
                self.assertTrue(entered.wait(timeout=5))

                # The handshake completed, but the receiver hasn't sent
                # ``DONE`` yet — the sender's reader-lock must still be held
                # so a concurrent ``disallow_checkpoint`` cannot proceed.
                self.assertTrue(sender._checkpoint_lock.w_locked())

                disallow_done = threading.Event()

                def disallow() -> None:
                    sender.disallow_checkpoint()
                    disallow_done.set()

                d = threading.Thread(target=disallow, daemon=True)
                d.start()
                self.assertFalse(
                    disallow_done.wait(timeout=0.3),
                    "disallow returned while a recv was still in flight",
                )

                # Release the receiver. It will finish ``_read_tensors`` and
                # send ``DONE``, the sender drops the r_lock, and disallow
                # finally returns.
                release.set()

                self.assertTrue(recv_done.wait(timeout=10))
                t.join(timeout=2)
                if recv_error:
                    raise recv_error[0]

                self.assertTrue(disallow_done.wait(timeout=5))
                d.join(timeout=2)

                got = recv_result["v"]
                torch.testing.assert_close(got["w"], payload["w"])
                torch.testing.assert_close(got["b"], payload["b"])
                self.assertEqual(got["step"], 99)

                # And once everything has settled, the sender is back in the
                # disallowed state with the writer lock held.
                self.assertTrue(sender._disallowed)
                self.assertTrue(sender._checkpoint_lock.w_locked())
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_concurrent_receivers_block_disallow(self) -> None:
        """N concurrent receivers all hold the reader lock; disallow blocks until each releases."""
        transport = self._new_transport()
        try:
            transport.send_checkpoint(
                dst_ranks=[1, 2, 3, 4, 5],
                step=1,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            N = 5
            sockets = [
                self._open_handshake(transport, f"mock://peer/{i}".encode())
                for i in range(N)
            ]
            try:
                # All N readers hold r_lock -> writer lock is held.
                self._wait_for(
                    lambda: transport._checkpoint_lock.w_locked(), timeout=2
                )
                self._wait_for(
                    lambda: len(transport._peers) == N, timeout=2
                )

                disallow_done = threading.Event()

                def disallow() -> None:
                    transport.disallow_checkpoint()
                    disallow_done.set()

                d = threading.Thread(target=disallow, daemon=True)
                d.start()

                # Release readers one at a time; disallow stays blocked until
                # the last one signals DONE.
                for i in range(N - 1):
                    _send_frame(sockets[i], _HANDSHAKE_DONE)
                    self.assertFalse(
                        disallow_done.wait(timeout=0.2),
                        f"disallow returned with {N - 1 - i} readers still active",
                    )

                _send_frame(sockets[N - 1], _HANDSHAKE_DONE)
                self.assertTrue(disallow_done.wait(timeout=5))
                d.join(timeout=2)

                # Control record should now reflect DISALLOWED.
                self.assertEqual(self._read_control(transport).status, "DISALLOWED")
            finally:
                for s in sockets:
                    try:
                        s.close()
                    except Exception:
                        pass
        finally:
            transport.shutdown()

    def test_generation_retirement(self) -> None:
        """Two-generation memory model: previous snapshot retired on each new send."""
        transport = self._new_transport()
        try:
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"a": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )
            gen1 = transport._current_snapshot
            self.assertIsNotNone(gen1)
            self.assertEqual(gen1.step, 1)
            self.assertIsNone(transport._previous_snapshot)
            initial_generation = gen1.generation

            transport.disallow_checkpoint()
            transport.send_checkpoint(
                dst_ranks=[1],
                step=2,
                state_dict={"a": torch.tensor([2.0])},
                timeout=timedelta(seconds=10),
            )
            gen2 = transport._current_snapshot
            self.assertIsNotNone(gen2)
            self.assertEqual(gen2.step, 2)
            # Previous now holds gen1.
            self.assertIs(transport._previous_snapshot, gen1)
            self.assertEqual(gen2.generation, initial_generation + 1)

            transport.disallow_checkpoint()
            transport.send_checkpoint(
                dst_ranks=[1],
                step=3,
                state_dict={"a": torch.tensor([3.0])},
                timeout=timedelta(seconds=10),
            )
            gen3 = transport._current_snapshot
            self.assertEqual(gen3.step, 3)
            # gen2 is now the previous; gen1 is no longer referenced by the
            # transport (it was the previous before this send).
            self.assertIs(transport._previous_snapshot, gen2)
            self.assertIsNot(transport._previous_snapshot, gen1)
            self.assertIsNot(transport._current_snapshot, gen1)
            self.assertEqual(gen3.generation, initial_generation + 2)
        finally:
            transport.shutdown()

    def test_retired_snapshot_remote_buffer_unreadable(self) -> None:
        """A read against a retired generation's remote_buffer must fail.

        Until the mock's WeakValueDictionary registry was wired up, dropped
        ``RdmaMemory`` objects would still resolve in tests, masking
        use-after-retirement bugs in production.
        """
        transport = self._new_transport()
        try:
            # Send 1: gen1 created.
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"a": torch.tensor([1.0, 2.0, 3.0])},
                timeout=timedelta(seconds=10),
            )
            gen1 = transport._current_snapshot
            self.assertIsNotNone(gen1)
            gen1_tensor_buf = gen1.tensor_mems[0].to_remote_buffer()
            gen1_manifest_buf = gen1.manifest_mem.to_remote_buffer()
            self.assertIn(gen1_tensor_buf.addr, _MockRdmaMemory._registry)
            self.assertIn(gen1_manifest_buf.addr, _MockRdmaMemory._registry)

            # Send 2: gen1 -> previous, gen2 -> current. gen1 still alive.
            transport.disallow_checkpoint()
            transport.send_checkpoint(
                dst_ranks=[1],
                step=2,
                state_dict={"a": torch.tensor([4.0, 5.0, 6.0])},
                timeout=timedelta(seconds=10),
            )
            self.assertIn(gen1_tensor_buf.addr, _MockRdmaMemory._registry)

            # Send 3: gen2 -> previous, gen3 -> current. gen1 falls off the
            # transport entirely.
            transport.disallow_checkpoint()
            transport.send_checkpoint(
                dst_ranks=[1],
                step=3,
                state_dict={"a": torch.tensor([7.0, 8.0, 9.0])},
                timeout=timedelta(seconds=10),
            )
            # Drop our local handle so gen1 has no strong references.
            del gen1
            gc.collect()

            # Both of gen1's buffers are now gone from the registry.
            self.assertNotIn(gen1_tensor_buf.addr, _MockRdmaMemory._registry)
            self.assertNotIn(gen1_manifest_buf.addr, _MockRdmaMemory._registry)

            # A real RDMA read against the stale buffer fails the mock's
            # registry check.
            from torchcomms._transport import (  # type: ignore[import-not-found]
                RdmaMemory as _MockMem,
                RdmaTransport as _MockT,
            )

            recv_t = _MockT(torch.device("cpu"))
            local = torch.zeros(gen1_tensor_buf.nbytes, dtype=torch.uint8)
            local_mem = _MockMem(local, cache_reg=False)
            with self.assertRaisesRegex(AssertionError, "unregistered addr"):
                recv_t.read(local_mem.to_mutable_view(), gen1_tensor_buf)
        finally:
            transport.shutdown()

    def test_shutdown_drops_snapshot_buffers(self) -> None:
        """Shutdown clears both snapshots; their RDMA buffers are deregistered."""
        transport = self._new_transport()
        transport.send_checkpoint(
            dst_ranks=[1],
            step=1,
            state_dict={"a": torch.tensor([1.0, 2.0])},
            timeout=timedelta(seconds=10),
        )
        snap = transport._current_snapshot
        self.assertIsNotNone(snap)
        tensor_addr = snap.tensor_mems[0].to_remote_buffer().addr
        self.assertIn(tensor_addr, _MockRdmaMemory._registry)

        # Drop our handle and shut down — both snapshot slots go away, the
        # mems get GC'd, and the WeakValueDictionary entries vanish.
        del snap
        transport.shutdown()
        gc.collect()

        self.assertNotIn(tensor_addr, _MockRdmaMemory._registry)

    def test_disallow_send_disallow_cycle(self) -> None:
        """DISALLOWED -> READY -> DISALLOWED -> READY -> DISALLOWED with control record matching."""
        transport = self._new_transport()
        try:
            # Initial state from __init__ -> DISALLOWED.
            self.assertEqual(self._read_control(transport).status, "DISALLOWED")
            self.assertTrue(transport._disallowed)
            self.assertTrue(transport._checkpoint_lock.w_locked())

            transport.send_checkpoint(
                dst_ranks=[1],
                step=10,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "READY")
            self.assertEqual(rec.step, 10)
            self.assertIsNotNone(rec.manifest_remote_buffer)
            self.assertGreater(rec.manifest_nbytes, 0)
            self.assertFalse(transport._disallowed)
            self.assertFalse(transport._checkpoint_lock.w_locked())

            transport.disallow_checkpoint()
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "DISALLOWED")
            # Step is preserved across the disallow -- it just blocks new
            # readers.
            self.assertEqual(rec.step, 10)
            self.assertIsNone(rec.manifest_remote_buffer)
            self.assertEqual(rec.manifest_nbytes, 0)
            self.assertTrue(transport._disallowed)
            self.assertTrue(transport._checkpoint_lock.w_locked())

            # disallow is idempotent: a second call shouldn't double-acquire.
            transport.disallow_checkpoint()
            self.assertTrue(transport._disallowed)
            self.assertTrue(transport._checkpoint_lock.w_locked())

            transport.send_checkpoint(
                dst_ranks=[1],
                step=11,
                state_dict={"t": torch.tensor([2.0])},
                timeout=timedelta(seconds=10),
            )
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "READY")
            self.assertEqual(rec.step, 11)
            self.assertFalse(transport._disallowed)
            self.assertFalse(transport._checkpoint_lock.w_locked())

            transport.disallow_checkpoint()
            rec = self._read_control(transport)
            self.assertEqual(rec.status, "DISALLOWED")
            self.assertEqual(rec.step, 11)
            self.assertTrue(transport._disallowed)
            self.assertTrue(transport._checkpoint_lock.w_locked())
        finally:
            transport.shutdown()

    def test_hung_receiver_times_out_after_ready(self) -> None:
        """A hung-but-CONNECTED receiver holds the fence until the safety deadline.

        Updated for BLOCKER B2: the fence is no longer released on a mere
        per-step idle timeout (that would free a snapshot mid-read on a
        healthy-but-slow transfer -> use-after-free). A receiver that reaches
        READY and then goes silent *while keeping its socket open* is treated
        as a slow-but-alive peer: the fence stays held until the absolute
        ``max_transfer_seconds`` backstop fires, well past the short per-step
        ``timeout``. We use a small ``max_transfer_seconds`` here so the test
        can observe (a) the fence surviving the per-step timeout and (b) the
        backstop eventually releasing it.

        (Liveness for a *dropped* socket is covered separately by
        ``TestRDMAFenceLiveness.test_dropped_receiver_releases_fence_promptly``.)
        """
        per_step_timeout = timedelta(milliseconds=200)
        backstop_s = 2.0
        transport = self._new_transport(
            timeout=per_step_timeout, max_transfer_seconds=backstop_s
        )
        try:
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            sock = self._open_handshake(transport, b"mock://hung")
            try:
                # Reader is now holding r_lock.
                self._wait_for(
                    lambda: transport._checkpoint_lock.w_locked(), timeout=2
                )

                # The connection is alive (we keep the socket open) but idle.
                # The handler must NOT release the fence merely because the
                # per-step timeout elapsed -- wait well past it and confirm the
                # fence is still held.
                time.sleep(per_step_timeout.total_seconds() * 4)
                self.assertTrue(
                    transport._checkpoint_lock.w_locked(),
                    "fence released on idle per-step timeout (B2 regression)",
                )
                self.assertEqual(len(transport._peers), 1)

                # Eventually the absolute safety backstop fires and releases
                # the fence even though the peer never sent DONE.
                self._wait_for(
                    lambda: not transport._checkpoint_lock.w_locked(),
                    timeout=backstop_s + 3,
                )
                self._wait_for(
                    lambda: len(transport._peers) == 0, timeout=2
                )
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

            # disallow_checkpoint should now succeed because the backstop
            # released the reader lock.
            disallow_done = threading.Event()

            def disallow() -> None:
                transport.disallow_checkpoint()
                disallow_done.set()

            d = threading.Thread(target=disallow, daemon=True)
            d.start()
            self.assertTrue(disallow_done.wait(timeout=2))
            d.join(timeout=2)
        finally:
            transport.shutdown()

    def test_hung_receiver_before_handshake_does_not_block_disallow(self) -> None:
        """A receiver that opens TCP but never sends peer_addr never holds the lock."""
        transport = self._new_transport(timeout=timedelta(milliseconds=400))
        try:
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            # Open TCP connection but don't send the peer_addr frame. The
            # handler will block in _recv_frame and time out -- but it never
            # acquires the r_lock, so disallow can proceed immediately.
            sock = socket.create_connection(
                ("localhost", transport._handshake_port), timeout=5
            )
            try:
                disallow_done = threading.Event()

                def disallow() -> None:
                    transport.disallow_checkpoint()
                    disallow_done.set()

                d = threading.Thread(target=disallow, daemon=True)
                d.start()

                # disallow should not block on this receiver since the handler
                # hasn't taken the r_lock yet.
                self.assertTrue(disallow_done.wait(timeout=2))
                d.join(timeout=2)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            transport.shutdown()

    def test_concurrent_send_and_disallow_no_deadlock(self) -> None:
        """Racing send_checkpoint + disallow_checkpoint never deadlocks; lock state stays consistent."""
        transport = self._new_transport()
        try:
            # Move out of the initial DISALLOWED state.
            transport.send_checkpoint(
                dst_ranks=[1],
                step=0,
                state_dict={"t": torch.tensor([0.0])},
                timeout=timedelta(seconds=10),
            )

            for i in range(8):
                step = i + 1
                send_done = threading.Event()
                disallow_done = threading.Event()

                def sender(step: int = step) -> None:
                    transport.send_checkpoint(
                        dst_ranks=[1],
                        step=step,
                        state_dict={"t": torch.tensor([float(step)])},
                        timeout=timedelta(seconds=10),
                    )
                    send_done.set()

                def disallower() -> None:
                    transport.disallow_checkpoint()
                    disallow_done.set()

                s = threading.Thread(target=sender, daemon=True)
                d = threading.Thread(target=disallower, daemon=True)
                s.start()
                d.start()

                self.assertTrue(send_done.wait(timeout=5), f"iter {i}: sender stuck")
                self.assertTrue(
                    disallow_done.wait(timeout=5), f"iter {i}: disallow stuck"
                )
                s.join(timeout=2)
                d.join(timeout=2)

                # Internal invariant: _disallowed iff w_lock held.
                self.assertEqual(
                    transport._disallowed,
                    transport._checkpoint_lock.w_locked(),
                    f"iter {i}: lock state inconsistent with _disallowed",
                )
                # Snapshot was published despite the race.
                self.assertEqual(transport._current_snapshot.step, step)

                # Reset to DISALLOWED for the next iteration.
                if not transport._disallowed:
                    transport.disallow_checkpoint()
                self.assertTrue(transport._disallowed)
                self.assertTrue(transport._checkpoint_lock.w_locked())
        finally:
            transport.shutdown()


# ---------------------------------------------------------------------------
# Phase 3: in-place receive, DTensor, strided, and mixed state_dict.
# ---------------------------------------------------------------------------


class TestRDMAPhase3InPlaceReceive(_RDMAMockBase):
    """Phase 3.1: in-place receive via the state_dict callback."""

    def test_inplace_receive_writes_into_callback_tensors(self) -> None:
        """Pre-allocated destination tensors are filled in-place; storage is shared."""
        sender = self._new_transport()
        try:
            payload = {
                "w": torch.arange(8, dtype=torch.float32),
                "b": torch.tensor([10.0, 20.0, 30.0]),
                "step": 99,
            }
            sender.send_checkpoint(
                dst_ranks=[1],
                step=99,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            preallocated = {
                "w": torch.zeros(8, dtype=torch.float32),
                "b": torch.zeros(3, dtype=torch.float32),
                "step": 0,
            }
            preallocated_w_ptr = preallocated["w"].data_ptr()
            preallocated_b_ptr = preallocated["b"].data_ptr()

            receiver = self._new_transport(state_dict_fn=lambda: preallocated)
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=99,
                    timeout=timedelta(seconds=10),
                )
                torch.testing.assert_close(got["w"], payload["w"])
                torch.testing.assert_close(got["b"], payload["b"])
                self.assertEqual(got["step"], 99)

                # Pre-allocated tensors were updated in place.
                torch.testing.assert_close(preallocated["w"], payload["w"])
                torch.testing.assert_close(preallocated["b"], payload["b"])

                # Returned tensors share storage with the pre-allocated ones.
                self.assertEqual(got["w"].data_ptr(), preallocated_w_ptr)
                self.assertEqual(got["b"].data_ptr(), preallocated_b_ptr)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_inplace_receive_falls_back_when_destination_missing(self) -> None:
        """Tensors absent from the callback's state_dict are allocated fresh."""
        sender = self._new_transport()
        try:
            payload = {
                "present": torch.arange(4, dtype=torch.float32),
                "missing": torch.tensor([1.0, 2.0]),
            }
            sender.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            preallocated = {
                "present": torch.zeros(4, dtype=torch.float32),
                # No "missing" key -- recv must allocate it.
            }
            preallocated_present_ptr = preallocated["present"].data_ptr()

            receiver = self._new_transport(state_dict_fn=lambda: preallocated)
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=1,
                    timeout=timedelta(seconds=10),
                )
                torch.testing.assert_close(got["present"], payload["present"])
                torch.testing.assert_close(got["missing"], payload["missing"])

                # "present" was filled in place; "missing" is a fresh tensor.
                self.assertEqual(
                    got["present"].data_ptr(), preallocated_present_ptr
                )
                torch.testing.assert_close(preallocated["present"], payload["present"])
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_inplace_receive_called_per_recv(self) -> None:
        """state_dict callback runs on every recv_checkpoint, allowing fresh destinations."""
        sender = self._new_transport()
        try:
            payload = {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}

            call_count = {"n": 0}
            destinations: List[torch.Tensor] = []

            def make_state_dict() -> Dict[str, torch.Tensor]:
                call_count["n"] += 1
                t = torch.zeros(4, dtype=torch.float32)
                destinations.append(t)
                return {"x": t}

            receiver = self._new_transport(state_dict_fn=make_state_dict)
            try:
                for step in (1, 2, 3):
                    sender.disallow_checkpoint()
                    sender.send_checkpoint(
                        dst_ranks=[1],
                        step=step,
                        state_dict=payload,
                        timeout=timedelta(seconds=10),
                    )
                    metadata = sender.metadata()
                    got = receiver.recv_checkpoint(
                        src_rank=0,
                        metadata=metadata,
                        step=step,
                        timeout=timedelta(seconds=10),
                    )
                    torch.testing.assert_close(got["x"], payload["x"])

                self.assertEqual(call_count["n"], 3)
                self.assertEqual(len(destinations), 3)
                # Each destination was filled correctly.
                for dst in destinations:
                    torch.testing.assert_close(dst, payload["x"])
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()


class TestRDMAPhase3DTensor(_RDMAMockBase):
    """Phase 3.2: DTensor handling end-to-end through the RDMA pipeline."""

    def setUp(self) -> None:
        super().setUp()
        # DTensor needs a default process group. Use a single-rank Gloo PG.
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", rank=0, world_size=1, store=dist.HashStore()
            )
            self._owns_pg = True
        else:
            self._owns_pg = False

    def tearDown(self) -> None:
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def _make_dtensor(self, data: torch.Tensor) -> DTensor:
        mesh = DeviceMesh("cpu", torch.tensor([0]))
        return distribute_tensor(data, mesh, [Replicate()])

    def test_dtensor_roundtrip(self) -> None:
        """A real DTensor (Replicate on a single-rank CPU mesh) round-trips."""
        local = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        dtensor = self._make_dtensor(local)

        sender = self._new_transport()
        try:
            payload = {"weights": dtensor, "step": 5}
            sender.send_checkpoint(
                dst_ranks=[1],
                step=5,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=5,
                    timeout=timedelta(seconds=10),
                )
                self.assertIsInstance(got["weights"], DTensor)
                torch.testing.assert_close(
                    got["weights"]._local_tensor,
                    dtensor._local_tensor,
                )
                self.assertEqual(got["weights"]._spec, dtensor._spec)
                self.assertEqual(got["step"], 5)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_dtensor_inplace_receive(self) -> None:
        """In-place receive into a pre-allocated DTensor uses the same local storage."""
        sent_local = torch.tensor([7.0, 8.0, 9.0, 10.0])
        dtensor = self._make_dtensor(sent_local)

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=2,
                state_dict={"d": dtensor},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            # Pre-allocate a destination DTensor with zeroed local storage.
            dst_local = torch.zeros(4, dtype=torch.float32)
            dst_dtensor = self._make_dtensor(dst_local)
            dst_local_ptr = dst_dtensor._local_tensor.data_ptr()

            receiver = self._new_transport(
                state_dict_fn=lambda: {"d": dst_dtensor}
            )
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=2,
                    timeout=timedelta(seconds=10),
                )
                self.assertIsInstance(got["d"], DTensor)
                torch.testing.assert_close(
                    got["d"]._local_tensor,
                    dtensor._local_tensor,
                )
                # Same local-tensor storage as the destination DTensor.
                self.assertEqual(
                    got["d"]._local_tensor.data_ptr(), dst_local_ptr
                )
                # Pre-allocated DTensor's local was filled in place too.
                torch.testing.assert_close(
                    dst_dtensor._local_tensor,
                    dtensor._local_tensor,
                )
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()


class TestRDMAPhase3StridedTensors(_RDMAMockBase):
    """Phase 3.3: strided / non-contiguous tensors round-trip via underlying storage."""

    def test_transposed_tensor_roundtrip(self) -> None:
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        transposed = base.T  # shape (4, 3), stride (1, 4), non-contiguous

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": transposed},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=1,
                    timeout=timedelta(seconds=10),
                )
                self.assertEqual(got["t"].shape, transposed.shape)
                self.assertEqual(got["t"].stride(), transposed.stride())
                torch.testing.assert_close(got["t"], transposed)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_strided_slice_roundtrip(self) -> None:
        base = torch.arange(16, dtype=torch.float32)
        sliced = base[::2]  # shape (8,), stride (2,)

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=2,
                state_dict={"s": sliced},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=2,
                    timeout=timedelta(seconds=10),
                )
                self.assertEqual(got["s"].shape, sliced.shape)
                self.assertEqual(got["s"].stride(), sliced.stride())
                torch.testing.assert_close(got["s"], sliced)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_offset_view_roundtrip(self) -> None:
        """A view with non-zero storage offset preserves shape, stride, and offset."""
        base = torch.arange(10, dtype=torch.float32)
        offset_view = base[3:8]  # shape (5,), stride (1,), storage_offset=3

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=3,
                state_dict={"o": offset_view},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=3,
                    timeout=timedelta(seconds=10),
                )
                self.assertEqual(got["o"].shape, offset_view.shape)
                self.assertEqual(got["o"].stride(), offset_view.stride())
                self.assertEqual(
                    got["o"].storage_offset(), offset_view.storage_offset()
                )
                torch.testing.assert_close(got["o"], offset_view)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_inplace_receive_into_transposed_destination(self) -> None:
        """In-place receive into a pre-allocated, non-contiguous destination.

        Exercises the strided + ``state_dict`` callback path: ``_cast_tensor``
        recasts the non-contiguous destination's underlying storage to uint8,
        the RDMA read writes into that storage, and ``torch.as_strided``
        rebuilds the transposed view on top.
        """
        src_base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        src_view = src_base.T  # (4, 3), stride (1, 4), non-contiguous

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": src_view},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            # Pre-allocate a destination with matching storage layout. We pass
            # the transposed (non-contiguous) view through the state_dict
            # callback so the receiver's in-place path has to handle the
            # non-contiguous case explicitly.
            dst_base = torch.zeros(3, 4, dtype=torch.float32)
            dst_view = dst_base.T  # (4, 3), stride (1, 4), non-contiguous
            self.assertFalse(dst_view.is_contiguous())
            dst_storage_ptr = dst_base.untyped_storage().data_ptr()
            dst_view_ptr = dst_view.data_ptr()

            receiver = self._new_transport(state_dict_fn=lambda: {"t": dst_view})
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=1,
                    timeout=timedelta(seconds=10),
                )
                self.assertIsInstance(got["t"], torch.Tensor)
                self.assertEqual(got["t"].shape, src_view.shape)
                self.assertEqual(got["t"].stride(), src_view.stride())
                torch.testing.assert_close(got["t"], src_view)

                # Pre-allocated destination was filled in place: dst_base now
                # holds the source storage contents, and dst_view (which is
                # dst_base.T) reflects them.
                torch.testing.assert_close(dst_view, src_view)
                torch.testing.assert_close(dst_base, src_base)

                # Returned tensor shares storage with the destination.
                self.assertEqual(
                    got["t"].untyped_storage().data_ptr(), dst_storage_ptr
                )
                self.assertEqual(got["t"].data_ptr(), dst_view_ptr)
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()

    def test_inplace_receive_into_offset_view_destination(self) -> None:
        """In-place receive into a pre-allocated offset view (non-zero storage_offset)."""
        # Source: offset view into a larger contiguous buffer. The sender
        # transfers the FULL underlying storage (40 bytes), not just the 5
        # visible elements, so the destination must have a matching 40-byte
        # storage.
        src_base = torch.arange(10, dtype=torch.float32)
        src_view = src_base[3:8]  # shape (5,), stride (1,), storage_offset=3

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=2,
                state_dict={"o": src_view},
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            dst_base = torch.zeros(10, dtype=torch.float32)
            dst_view = dst_base[3:8]
            self.assertEqual(dst_view.storage_offset(), 3)
            dst_storage_ptr = dst_base.untyped_storage().data_ptr()

            receiver = self._new_transport(state_dict_fn=lambda: {"o": dst_view})
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=2,
                    timeout=timedelta(seconds=10),
                )
                self.assertEqual(got["o"].shape, src_view.shape)
                self.assertEqual(got["o"].stride(), src_view.stride())
                self.assertEqual(
                    got["o"].storage_offset(), src_view.storage_offset()
                )
                torch.testing.assert_close(got["o"], src_view)

                # The pre-allocated view was filled in place.
                torch.testing.assert_close(dst_view, src_view)
                self.assertEqual(
                    got["o"].untyped_storage().data_ptr(), dst_storage_ptr
                )
                self.assertEqual(got["o"].data_ptr(), dst_view.data_ptr())
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()


class TestRDMAPhase3MixedStateDict(_RDMAMockBase):
    """Phase 3.4: mixed state_dict with tensors of various sizes/dtypes, scalars, nested dicts."""

    def test_mixed_state_dict_roundtrip(self) -> None:
        torch.manual_seed(0)
        payload: Dict[str, object] = {
            "weights": {
                "fc1": torch.randn(8, 16, dtype=torch.float32),
                "fc2": torch.randn(16, 4, dtype=torch.float32),
                "bias_f64": torch.zeros(4, dtype=torch.float64),
                "bias_f16": torch.tensor([0.5, -0.5], dtype=torch.float16),
            },
            "ints": torch.arange(10, dtype=torch.int32),
            "longs": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "bool_mask": torch.tensor([True, False, True, True]),
            "optimizer": {
                "step": 1234,
                "lr": 0.001,
                "betas": (0.9, 0.999),
            },
            "metadata": {
                "name": "model_v1",
                "tags": ["fp32", "cpu"],
                "nested": {"depth": 3, "value": "leaf"},
            },
            "scalar_int": 42,
            "scalar_string": "hello",
            "scalar_float": 3.14,
            "scalar_none": None,
            "small_tensor": torch.tensor([1, 2, 3], dtype=torch.int32),
            "scalar_tensor": torch.tensor(7.0),
        }

        sender = self._new_transport()
        try:
            sender.send_checkpoint(
                dst_ranks=[1],
                step=42,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            receiver = self._new_transport()
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=42,
                    timeout=timedelta(seconds=10),
                )

                # Tensors.
                torch.testing.assert_close(
                    got["weights"]["fc1"], payload["weights"]["fc1"]
                )
                torch.testing.assert_close(
                    got["weights"]["fc2"], payload["weights"]["fc2"]
                )
                torch.testing.assert_close(
                    got["weights"]["bias_f64"], payload["weights"]["bias_f64"]
                )
                torch.testing.assert_close(
                    got["weights"]["bias_f16"], payload["weights"]["bias_f16"]
                )
                torch.testing.assert_close(got["ints"], payload["ints"])
                torch.testing.assert_close(got["longs"], payload["longs"])
                torch.testing.assert_close(got["bool_mask"], payload["bool_mask"])
                torch.testing.assert_close(
                    got["small_tensor"], payload["small_tensor"]
                )
                torch.testing.assert_close(
                    got["scalar_tensor"], payload["scalar_tensor"]
                )

                # Non-tensors and nested dicts.
                self.assertEqual(got["optimizer"], payload["optimizer"])
                self.assertEqual(got["metadata"], payload["metadata"])
                self.assertEqual(got["scalar_int"], 42)
                self.assertEqual(got["scalar_string"], "hello")
                self.assertEqual(got["scalar_float"], 3.14)
                self.assertIsNone(got["scalar_none"])
            finally:
                receiver.shutdown()
        finally:
            sender.shutdown()


# ---------------------------------------------------------------------------
# Phase 4: GPU snapshot spill-to-CPU.
# ---------------------------------------------------------------------------


class _FakeCudaTensor:
    """Quacks like a ``torch.Tensor`` on CUDA for ``_build_snapshot``.

    The backing storage is a real CPU tensor — only ``.device`` lies. This
    lets us drive the GPU-spill code path on machines without a CUDA build
    while still exercising the real cloning / copy code in production.
    """

    def __init__(self, nbytes: int) -> None:
        self._t: torch.Tensor = torch.zeros(nbytes, dtype=torch.uint8)
        self.device: torch.device = torch.device("cuda:0")

    def untyped_storage(self) -> "torch.UntypedStorage":
        return self._t.untyped_storage()

    def clone(self) -> torch.Tensor:
        return self._t.clone()

    def view(self, dtype: torch.dtype) -> torch.Tensor:
        return self._t.view(dtype)


def _make_cuda_only_state_dict_meta(nbytes_per_tensor: List[int]) -> _StateDictMeta:
    """Build a ``_StateDictMeta`` whose leaves are all CUDA tensor metadata."""
    non_tensor_leaves = [
        _TensorMeta(
            shape=torch.Size([n]),
            dtype=torch.uint8,
            storage_offset=0,
            stride=(1,),
            nbytes=n,
        )
        for n in nbytes_per_tensor
    ]
    return _StateDictMeta(
        step=0,
        treespec=None,
        paths=[(f"t{i}",) for i in range(len(nbytes_per_tensor))],
        non_tensor_leaves=non_tensor_leaves,
    )


class TestRDMAGpuSnapshotSpill(_RDMAMockBase):
    """Verify GPU snapshots spill to pinned CPU once the budget is exceeded."""

    def _patch_pin_memory(self) -> "patch._patch":
        """Strip ``pin_memory=True`` from the ``torch.empty`` call.

        Pinned-memory allocation requires a CUDA backend, which this CI
        machine does not have. The patch lets us run the spill code path
        and still observe — via the wrapping spill helper — that the
        production code requested pinning.
        """
        from torchft.checkpointing import rdma_transport as mod

        real_spill = mod._spill_to_pinned_cpu

        def stub_spill(t: torch.Tensor) -> torch.Tensor:
            nbytes = t.untyped_storage().nbytes()
            cpu = torch.empty(nbytes, dtype=torch.uint8)  # no pin_memory
            cpu.copy_(t.view(torch.uint8), non_blocking=False)
            return cpu

        # Wrap the stub so the test can count how many tensors were spilled.
        stub_spill.real = real_spill  # type: ignore[attr-defined]
        return patch.object(mod, "_spill_to_pinned_cpu", side_effect=stub_spill)

    def test_all_tensors_fit_no_spill(self) -> None:
        """Every GPU tensor stays on GPU when cumulative bytes fit the budget."""
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            max_gpu_snapshot_bytes=10_000,
        )
        try:
            tensors = [_FakeCudaTensor(600), _FakeCudaTensor(600), _FakeCudaTensor(600)]
            sd_meta = _make_cuda_only_state_dict_meta([600, 600, 600])

            with self._patch_pin_memory() as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            spill_mock.assert_not_called()
            self.assertEqual(len(snap.tensor_snapshots), 3)
            self.assertEqual(len(snap.tensor_mems), 3)
        finally:
            transport.shutdown()

    def test_spill_when_budget_exceeded_midway(self) -> None:
        """Once cumulative GPU bytes exceed the budget, remaining tensors spill."""
        # Budget=1000, tensors=[600, 600, 600].
        # Tensor 0: 0+600 <= 1000 -> kept on GPU.
        # Tensor 1: 600+600 > 1000 -> spilled.
        # Tensor 2: budget already exceeded -> spilled.
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            max_gpu_snapshot_bytes=1000,
        )
        try:
            tensors = [_FakeCudaTensor(600), _FakeCudaTensor(600), _FakeCudaTensor(600)]
            sd_meta = _make_cuda_only_state_dict_meta([600, 600, 600])

            with self._patch_pin_memory() as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            self.assertEqual(spill_mock.call_count, 2)
            self.assertEqual(len(snap.tensor_snapshots), 3)
            for ts in snap.tensor_snapshots:
                self.assertEqual(ts.dtype, torch.uint8)
                self.assertEqual(ts.numel(), 600)
        finally:
            transport.shutdown()

    def test_zero_budget_spills_everything(self) -> None:
        """With ``max_gpu_snapshot_bytes=0`` every GPU tensor spills."""
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            max_gpu_snapshot_bytes=0,
        )
        try:
            tensors = [_FakeCudaTensor(128) for _ in range(4)]
            sd_meta = _make_cuda_only_state_dict_meta([128, 128, 128, 128])

            with self._patch_pin_memory() as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            self.assertEqual(spill_mock.call_count, 4)
            self.assertEqual(len(snap.tensor_snapshots), 4)
        finally:
            transport.shutdown()

    def test_spill_remains_sticky_after_first_overflow(self) -> None:
        """A small tensor after a spill still spills (simple, predictable rule)."""
        # Budget=1000, tensors=[600, 600 (spills), 100 (would fit but spills)].
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            max_gpu_snapshot_bytes=1000,
        )
        try:
            tensors = [_FakeCudaTensor(600), _FakeCudaTensor(600), _FakeCudaTensor(100)]
            sd_meta = _make_cuda_only_state_dict_meta([600, 600, 100])

            with self._patch_pin_memory() as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            self.assertEqual(spill_mock.call_count, 2)
            self.assertEqual(len(snap.tensor_snapshots), 3)
        finally:
            transport.shutdown()

    def test_cpu_tensors_never_count_toward_gpu_budget(self) -> None:
        """CPU tensors never count toward the GPU budget."""
        # Budget=500. CPU tensors are huge but irrelevant; one GPU tensor of
        # 200 fits, second GPU tensor of 400 spills (200+400 > 500).
        transport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timedelta(seconds=10),
            max_gpu_snapshot_bytes=500,
        )
        try:
            cpu_t = torch.zeros(10_000, dtype=torch.uint8)  # huge CPU tensor
            tensors = [_FakeCudaTensor(200), cpu_t, _FakeCudaTensor(400)]
            sd_meta = _StateDictMeta(
                step=0,
                treespec=None,
                paths=[("t0",), ("t1",), ("t2",)],
                non_tensor_leaves=[
                    _TensorMeta(
                        shape=torch.Size([200]),
                        dtype=torch.uint8,
                        storage_offset=0,
                        stride=(1,),
                        nbytes=200,
                    ),
                    _TensorMeta(
                        shape=torch.Size([10_000]),
                        dtype=torch.uint8,
                        storage_offset=0,
                        stride=(1,),
                        nbytes=10_000,
                    ),
                    _TensorMeta(
                        shape=torch.Size([400]),
                        dtype=torch.uint8,
                        storage_offset=0,
                        stride=(1,),
                        nbytes=400,
                    ),
                ],
            )

            with self._patch_pin_memory() as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            # Only the 3rd tensor (400 bytes after 200 already used) spills.
            self.assertEqual(spill_mock.call_count, 1)
            self.assertEqual(len(snap.tensor_snapshots), 3)
        finally:
            transport.shutdown()


# ---------------------------------------------------------------------------
# BLOCKER B2: lifetime fence tied to RDMA liveness/completion, not idle timeout.
# ---------------------------------------------------------------------------


class TestRDMAFenceLiveness(_RDMAMockBase):
    """The reader-lifetime fence survives a slow transfer and releases on drop.

    These tests pin down the BLOCKER B2 contract: a healthy-but-slow RDMA read
    (longer than the per-step ``timeout``) must NOT release the fence, while a
    receiver that drops its socket must release it promptly via TCP liveness.
    """

    def test_slow_read_holds_fence_until_done(self) -> None:
        """A read slower than the per-step ``timeout`` keeps the fence held.

        The receiver runs a real ``recv_checkpoint`` whose RDMA reads are
        injected with a delay well beyond the sender's per-step ``timeout``.
        While that read is in flight a concurrent ``disallow_checkpoint()`` must
        block (the fence is held), and the snapshot's ``RdmaMemory`` must stay
        registered for the whole read. Once the receiver sends ``DONE`` the
        fence releases and ``disallow`` returns.
        """
        # Per-step timeout is short; the read deliberately outlasts it. The
        # safety backstop is large so it never fires during the test.
        per_step = timedelta(milliseconds=300)
        read_delay = 1.2  # > 4x the per-step timeout
        sender = self._new_transport(
            timeout=per_step, max_transfer_seconds=120.0
        )
        try:
            payload = {"w": torch.arange(8, dtype=torch.float32)}
            sender.send_checkpoint(
                dst_ranks=[1],
                step=7,
                state_dict=payload,
                timeout=timedelta(seconds=10),
            )
            metadata = sender.metadata()

            snap = sender._current_snapshot
            self.assertIsNotNone(snap)
            tensor_addr = snap.tensor_mems[0].to_remote_buffer().addr
            self.assertIn(tensor_addr, _MockRdmaMemory._registry)

            # Inject the slow read for every mock transport (sender + receiver
            # per-peer instances).
            _MockRdmaTransport.read_delay_s = read_delay

            recv_result: Dict[str, object] = {}
            recv_error: List[BaseException] = []
            recv_done = threading.Event()

            receiver = self._new_transport(
                timeout=per_step, max_transfer_seconds=120.0
            )
            try:
                def do_recv() -> None:
                    try:
                        recv_result["v"] = receiver.recv_checkpoint(
                            src_rank=0,
                            metadata=metadata,
                            step=7,
                            timeout=timedelta(seconds=10),
                        )
                    except BaseException as exc:  # pragma: no cover - guard
                        recv_error.append(exc)
                    finally:
                        recv_done.set()

                rt = threading.Thread(target=do_recv, daemon=True)
                rt.start()

                # Wait until the sender's fence is held (handshake reached
                # READY and the handler took the reader lock).
                self._wait_for(
                    lambda: sender._checkpoint_lock.w_locked(), timeout=5
                )

                disallow_done = threading.Event()

                def disallow() -> None:
                    sender.disallow_checkpoint()
                    disallow_done.set()

                d = threading.Thread(target=disallow, daemon=True)
                d.start()

                # Across a span longer than the per-step timeout (but shorter
                # than the read delay), the fence must stay held and the
                # snapshot buffer must stay registered: the slow read is still
                # in flight, so disallow cannot proceed.
                deadline = time.monotonic() + (
                    per_step.total_seconds() * 2
                )
                while time.monotonic() < deadline:
                    self.assertFalse(
                        disallow_done.is_set(),
                        "disallow returned mid-read (B2: fence released on "
                        "idle timeout)",
                    )
                    self.assertIn(
                        tensor_addr,
                        _MockRdmaMemory._registry,
                        "snapshot RdmaMemory deregistered during in-flight read",
                    )
                    time.sleep(0.05)

                # The read eventually completes and the receiver sends DONE,
                # the fence releases, and disallow returns.
                self.assertTrue(recv_done.wait(timeout=10))
                if recv_error:
                    raise recv_error[0]
                self.assertTrue(disallow_done.wait(timeout=5))
                d.join(timeout=2)
                rt.join(timeout=2)

                torch.testing.assert_close(recv_result["v"]["w"], payload["w"])
                self.assertTrue(sender._disallowed)
                self.assertEqual(
                    self._read_control(sender).status, "DISALLOWED"
                )
            finally:
                receiver.shutdown()
        finally:
            _MockRdmaTransport.read_delay_s = 0.0
            sender.shutdown()

    def test_dropped_receiver_releases_fence_promptly(self) -> None:
        """A receiver that drops its socket without DONE releases the fence.

        This is the liveness half of B2: even though the fence is no longer
        released on an idle timeout, a *dead* connection (the receiver closes
        its socket mid-transfer) must still release it promptly so
        ``disallow_checkpoint()`` can proceed -- far sooner than the generous
        ``max_transfer_seconds`` backstop.
        """
        # Large backstop so a "prompt" release can only come from liveness
        # (EOF on the dropped socket), not from the safety deadline.
        transport = self._new_transport(
            timeout=timedelta(milliseconds=300), max_transfer_seconds=600.0
        )
        try:
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )

            sock = self._open_handshake(transport, b"mock://dropper")
            # Fence is now held by the handler.
            self._wait_for(
                lambda: transport._checkpoint_lock.w_locked(), timeout=2
            )

            disallow_done = threading.Event()

            def disallow() -> None:
                transport.disallow_checkpoint()
                disallow_done.set()

            d = threading.Thread(target=disallow, daemon=True)
            d.start()
            # While the receiver is connected and silent, disallow is blocked.
            self.assertFalse(disallow_done.wait(timeout=0.5))

            # Drop the socket WITHOUT sending DONE. The handler's blocking read
            # sees EOF -> liveness release, well before the 600s backstop.
            sock.close()

            self.assertTrue(
                disallow_done.wait(timeout=5),
                "dropped receiver did not release the fence promptly",
            )
            d.join(timeout=2)
            self._wait_for(lambda: len(transport._peers) == 0, timeout=2)
            self.assertEqual(self._read_control(transport).status, "DISALLOWED")
        finally:
            transport.shutdown()


# ---------------------------------------------------------------------------
# RISK B5a: handshake handler threads must pin the CUDA device.
# ---------------------------------------------------------------------------


class TestRDMAHandshakeCudaDevice(_RDMAMockBase):
    """``_handle_peer_connection`` pins the CUDA device before torchcomms use."""

    def test_handshake_handler_sets_cuda_device(self) -> None:
        """When device.type == cuda the handler calls ``torch.cuda.set_device``.

        CUDA current device is thread-local and the handler runs in a fresh
        thread, so it must pin ``self._device`` before constructing the
        per-peer ``RdmaTransport`` (RISK B5a). We patch ``torch.cuda.set_device``
        (so no real GPU is touched) and drive a single handshake.

        To keep the test CPU-simulatable and deterministic we build a normal
        CPU transport (so all buffers / snapshots stay on host memory and the
        ``.numpy()`` control-record path works) and then flip ``_device`` to a
        fake ``cuda`` device just before driving the handshake. The handler
        reads ``self._device`` to decide whether to pin, exactly as it would on
        a real GPU node, while the mock ``RdmaTransport`` happily accepts the
        cuda device.
        """
        recorded: List[object] = []
        cuda_device = torch.device("cuda:0")

        with patch.object(
            torch.cuda, "set_device", side_effect=lambda d: recorded.append(d)
        ):
            transport = self._new_transport(max_transfer_seconds=120.0)
            try:
                transport.send_checkpoint(
                    dst_ranks=[1],
                    step=1,
                    state_dict={"t": torch.tensor([1.0])},
                    timeout=timedelta(seconds=10),
                )

                # Now make the handler take the cuda branch. Snapshot/control
                # buffers were already built on CPU above.
                transport._device = cuda_device

                sock = self._open_handshake(transport, b"mock://cuda-peer")
                try:
                    # The handler thread must have pinned the cuda device
                    # before constructing the per-peer RdmaTransport.
                    self._wait_for(lambda: len(recorded) >= 1, timeout=5)
                    self.assertEqual(recorded[0], cuda_device)
                    _send_frame(sock, _HANDSHAKE_DONE)
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
            finally:
                transport.shutdown()

    def test_cpu_handshake_does_not_set_cuda_device(self) -> None:
        """The CPU path never calls ``torch.cuda.set_device`` (guarded)."""
        recorded: List[object] = []

        with patch.object(
            torch.cuda, "set_device", side_effect=lambda d: recorded.append(d)
        ):
            transport = self._new_transport()
            try:
                transport.send_checkpoint(
                    dst_ranks=[1],
                    step=1,
                    state_dict={"t": torch.tensor([1.0])},
                    timeout=timedelta(seconds=10),
                )
                sock = self._open_handshake(transport, b"mock://cpu-peer")
                try:
                    # Give the handler a moment to run; it must NOT touch CUDA.
                    time.sleep(0.3)
                    self.assertEqual(recorded, [])
                    _send_frame(sock, _HANDSHAKE_DONE)
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
            finally:
                transport.shutdown()


# ---------------------------------------------------------------------------
# B1: GPU snapshot must be synchronized before RDMA exposure.
#
# These tests fake ``torch.cuda`` (Stream / Event / current_stream / stream)
# with recorders so the CUDA branches of ``_build_snapshot`` /
# ``send_checkpoint`` run on a machine with no GPU. The single assertion that
# matters: the snapshot's copy-stream event is waited on (synchronized) BEFORE
# the control record is flipped to READY. If the synchronization is removed,
# these tests fail.
# ---------------------------------------------------------------------------


class _FakeCudaEvent:
    """Records ``record()`` / ``synchronize()`` against a shared call log."""

    def __init__(self, log: List[str]) -> None:
        self._log = log
        self.recorded = False
        self.synchronized = False

    def record(self, stream: object = None) -> None:
        self.recorded = True
        self._log.append("event.record")

    def synchronize(self) -> None:
        self.synchronized = True
        self._log.append("event.synchronize")


class _FakeCudaStream:
    """Stand-in for ``torch.cuda.Stream`` used by the copy-stream path."""

    def __init__(self, log: List[str]) -> None:
        self._log = log

    def wait_stream(self, other: object) -> None:
        self._log.append("stream.wait_stream")

    def synchronize(self) -> None:
        self._log.append("stream.synchronize")


class _FakeCudaModule:
    """Minimal fake of ``torch.cuda`` that records the ordering of operations.

    Only the entry points touched by the RDMA snapshot path are implemented:
    ``Stream``, ``Event``, ``current_stream``, and the ``stream(...)`` context
    manager. Every interesting call appends to ``log`` so a test can assert
    the relative order of the copy-stream fence and the READY publish.
    """

    def __init__(self) -> None:
        self.log: List[str] = []
        self.events: List[_FakeCudaEvent] = []

    def set_device(self, device: object) -> None:
        # Used by ``_restore_cuda_device`` (and send/recv) to re-pin HIP's
        # current device after RDMA registration on ROCm. A no-op for the fake;
        # logged so ordering can be inspected if needed.
        self.log.append("set_device")

    def synchronize(self, device: object = None) -> None:
        self.log.append("synchronize")

    def Stream(self) -> _FakeCudaStream:  # noqa: N802 - mirrors torch API
        return _FakeCudaStream(self.log)

    def Event(self) -> _FakeCudaEvent:  # noqa: N802 - mirrors torch API
        ev = _FakeCudaEvent(self.log)
        self.events.append(ev)
        return ev

    def current_stream(self) -> _FakeCudaStream:
        return _FakeCudaStream(self.log)

    def stream(self, stream: object) -> "object":
        log = self.log

        class _Ctx:
            def __enter__(self_inner) -> object:
                log.append("stream.enter")
                return stream

            def __exit__(self_inner, *exc: object) -> bool:
                log.append("stream.exit")
                return False

        return _Ctx()


class TestRDMAGpuSnapshotSyncBeforePublish(_RDMAMockBase):
    """B1: the snapshot's CUDA work is fenced before the READY publish."""

    def _make_cuda_transport(
        self, fake_cuda: _FakeCudaModule
    ) -> "RDMATransport":
        """Build a transport that believes its device is CUDA.

        ``pin_memory=True`` allocations and ``torch.cuda`` calls are routed to
        fakes so the CUDA branches execute on a CPU-only host.
        """
        import torch as _torch
        from torchft.checkpointing import rdma_transport as mod

        real_zeros = _torch.zeros
        real_empty = _torch.empty

        def zeros_no_pin(*args: object, **kwargs: object) -> "_torch.Tensor":
            kwargs.pop("pin_memory", None)
            return real_zeros(*args, **kwargs)

        def empty_no_pin(*args: object, **kwargs: object) -> "_torch.Tensor":
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        self._patches = [
            patch.object(mod.torch, "cuda", fake_cuda),
            patch.object(mod.torch, "zeros", side_effect=zeros_no_pin),
            patch.object(mod.torch, "empty", side_effect=empty_no_pin),
        ]
        for p in self._patches:
            p.start()
        try:
            return RDMATransport(
                device=torch.device("cuda:0"),
                timeout=timedelta(seconds=10),
                max_gpu_snapshot_bytes=1 << 30,
            )
        except Exception:
            for p in self._patches:
                p.stop()
            raise

    def _teardown_patches(self) -> None:
        for p in getattr(self, "_patches", []):
            p.stop()
        self._patches = []

    def test_event_synchronize_precedes_ready_publish(self) -> None:
        fake_cuda = _FakeCudaModule()
        transport = self._make_cuda_transport(fake_cuda)
        try:
            from torchft.checkpointing import rdma_transport as mod

            # Drive the CUDA path: ``_prepare_state_dict`` returns CUDA-like
            # tensors plus matching metadata.
            tensors = [_FakeCudaTensor(256), _FakeCudaTensor(128)]
            sd_meta = _make_cuda_only_state_dict_meta([256, 128])

            # Mark the publish point in the shared call log so we can assert it
            # happens strictly after the event synchronize.
            real_update = transport._update_control_record

            def logging_update(step: int, status: str, snapshot: object) -> None:
                if status == "READY":
                    fake_cuda.log.append("control.READY")
                return real_update(step, status, snapshot)

            with patch.object(
                mod, "_prepare_state_dict", return_value=(sd_meta, tensors)
            ), patch.object(
                transport, "_update_control_record", side_effect=logging_update
            ):
                transport.send_checkpoint(
                    dst_ranks=[1],
                    step=7,
                    state_dict={"unused": 0},
                    timeout=timedelta(seconds=10),
                )

            log = fake_cuda.log
            self.assertIn("event.synchronize", log)
            self.assertIn("control.READY", log)
            # The fence must come before the READY publish.
            self.assertLess(
                log.index("event.synchronize"),
                log.index("control.READY"),
                f"event must be synchronized before READY publish; log={log}",
            )
            # The event recorded on the snapshot was actually synchronized.
            self.assertEqual(len(fake_cuda.events), 1)
            self.assertTrue(fake_cuda.events[0].recorded)
            self.assertTrue(fake_cuda.events[0].synchronized)
        finally:
            self._teardown_patches()
            transport.shutdown()

    def test_copy_stream_waits_on_producer_stream(self) -> None:
        """The copy stream waits on the producer stream before copies start."""
        fake_cuda = _FakeCudaModule()
        transport = self._make_cuda_transport(fake_cuda)
        try:
            from torchft.checkpointing import rdma_transport as mod

            tensors = [_FakeCudaTensor(64)]
            sd_meta = _make_cuda_only_state_dict_meta([64])
            with patch.object(
                mod, "_prepare_state_dict", return_value=(sd_meta, tensors)
            ):
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            log = fake_cuda.log
            # Ordering inside _build_snapshot: wait on producer stream, enter
            # the copy-stream context, then record the fence event.
            self.assertIn("stream.wait_stream", log)
            self.assertIn("event.record", log)
            self.assertLess(
                log.index("stream.wait_stream"), log.index("event.record")
            )
            self.assertIsNotNone(snap.cuda_event)
        finally:
            self._teardown_patches()
            transport.shutdown()

    def test_removing_sync_would_break_invariant(self) -> None:
        """Guard test: if ``_wait_snapshot_ready`` no-ops, READY precedes sync.

        This pins the regression the B1 fix guards against. We monkeypatch the
        wait to do nothing (simulating the unfixed code) and assert that the
        synchronize no longer precedes the READY publish — proving the primary
        test above is actually exercising the synchronization.
        """
        fake_cuda = _FakeCudaModule()
        transport = self._make_cuda_transport(fake_cuda)
        try:
            from torchft.checkpointing import rdma_transport as mod

            tensors = [_FakeCudaTensor(32)]
            sd_meta = _make_cuda_only_state_dict_meta([32])

            real_update = transport._update_control_record

            def logging_update(step: int, status: str, snapshot: object) -> None:
                if status == "READY":
                    fake_cuda.log.append("control.READY")
                return real_update(step, status, snapshot)

            with patch.object(
                mod, "_prepare_state_dict", return_value=(sd_meta, tensors)
            ), patch.object(
                transport, "_update_control_record", side_effect=logging_update
            ), patch.object(
                transport, "_wait_snapshot_ready", return_value=None
            ):
                transport.send_checkpoint(
                    dst_ranks=[1],
                    step=3,
                    state_dict={"unused": 0},
                    timeout=timedelta(seconds=10),
                )

            log = fake_cuda.log
            # With the sync removed, no synchronize was issued before READY.
            self.assertIn("control.READY", log)
            ready_idx = log.index("control.READY")
            self.assertNotIn("event.synchronize", log[:ready_idx])
        finally:
            self._teardown_patches()
            transport.shutdown()


# ---------------------------------------------------------------------------
# B6: control-record publication atomicity.
#
# The control buffer is published payload-first, length-last; the length word
# is the commit point. A reader observing a mid-write buffer must decode either
# the previous record or the new one, never a torn mix.
# ---------------------------------------------------------------------------


class TestRDMAControlRecordAtomicity(_RDMAMockBase):
    """B6: publication is atomic at the length-commit boundary."""

    def _decode(self, buf: "torch.Tensor") -> Optional[_RDMAControlRecord]:
        """Decode the control buffer the way ``_read_control_record`` does.

        Returns ``None`` when length==0 (the "not yet committed" signal),
        otherwise the decoded record. Raises if the bytes are torn — which is
        exactly what must never be observed.
        """
        view = buf.numpy()
        (length,) = struct.unpack_from("<Q", view, 0)
        if length == 0:
            return None
        rec = pickle.loads(bytes(view[8 : 8 + length]))
        assert isinstance(rec, _RDMAControlRecord)
        return rec

    def test_length_is_committed_last(self) -> None:
        """The length word is written after the payload (commit-last order)."""
        transport = self._new_transport()
        try:
            # Publish an initial record so the buffer holds a known good record.
            transport.send_checkpoint(
                dst_ranks=[1],
                step=1,
                state_dict={"t": torch.tensor([1.0])},
                timeout=timedelta(seconds=10),
            )
            old_rec = self._read_control(transport)
            self.assertEqual(old_rec.status, "READY")
            self.assertEqual(old_rec.step, 1)

            # Capture the byte-write order by recording every store to offset 0
            # (the length word) vs. the payload region. We intercept numpy
            # ``struct.pack_into`` via a wrapper on the control tensor.
            order: List[str] = []
            real_pack_into = struct.pack_into

            def tracking_pack_into(fmt: str, buf: object, offset: int, *vals: object):
                if offset == 0:
                    order.append(f"len={vals[0]}")
                return real_pack_into(fmt, buf, offset, *vals)

            import torchft.checkpointing.rdma_transport as mod

            with patch.object(mod.struct, "pack_into", side_effect=tracking_pack_into):
                transport.disallow_checkpoint()

            # disallow writes a DISALLOWED record: length is first zeroed, then
            # written non-zero LAST (after the payload copy in between).
            self.assertEqual(order[0], "len=0", f"length not cleared first: {order}")
            self.assertNotEqual(
                order[-1], "len=0", f"length not committed last: {order}"
            )
            self.assertEqual(self._read_control(transport).status, "DISALLOWED")
        finally:
            transport.shutdown()

    def test_concurrent_reader_never_sees_torn_record(self) -> None:
        """A reader polling during repeated publishes only sees whole records.

        A background thread hammers the control buffer with alternating
        publishes while the main thread decodes it in a tight loop. Every
        successful decode must be a complete, valid record (one of the values
        the publisher actually wrote) — never a torn mix of two records.
        """
        transport = self._new_transport()
        try:
            # Seed a first record.
            transport.send_checkpoint(
                dst_ranks=[1],
                step=0,
                state_dict={"t": torch.tensor([0.0])},
                timeout=timedelta(seconds=10),
            )

            stop = threading.Event()
            errors: List[BaseException] = []
            seen_statuses: set = set()

            def publisher() -> None:
                try:
                    i = 0
                    while not stop.is_set():
                        i += 1
                        # Alternate between a READY snapshot and DISALLOWED so
                        # the payload contents (and length) change every write.
                        if i % 2 == 1:
                            snap = transport._build_snapshot(
                                *_prepare_state_dict(
                                    {"t": torch.arange(i % 7 + 1, dtype=torch.float32)},
                                    i,
                                    torch.device("cpu"),
                                ),
                                step=i,
                            )
                            transport._update_control_record(i, "READY", snap)
                        else:
                            transport._update_control_record(i, "DISALLOWED", None)
                except BaseException as exc:  # pragma: no cover - guard
                    errors.append(exc)

            t = threading.Thread(target=publisher, daemon=True)
            t.start()
            try:
                # Decode many times concurrently with the publisher.
                deadline = time.monotonic() + 1.5
                decodes = 0
                while time.monotonic() < deadline:
                    rec = self._decode(transport._control_tensor)
                    if rec is not None:
                        # A successfully decoded record must be internally
                        # consistent — these asserts would blow up on torn
                        # bytes (bad pickle / wrong type).
                        self.assertEqual(rec.version, _PROTOCOL_VERSION)
                        self.assertIn(rec.status, ("READY", "DISALLOWED"))
                        seen_statuses.add(rec.status)
                        decodes += 1
            finally:
                stop.set()
                t.join(timeout=5)

            if errors:
                raise errors[0]
            self.assertGreater(decodes, 0, "reader never decoded a record")
        finally:
            transport.shutdown()


# ---------------------------------------------------------------------------
# B3: GPU/GDR capability probe + staged pinned-CPU fallback.
# ---------------------------------------------------------------------------


def _fake_cuda_zeros(real_zeros: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Return a ``torch.zeros`` replacement that fakes CUDA allocations.

    On this CPU-only CI box ``torch.zeros(..., device="cuda")`` and
    ``pin_memory=True`` both fail. This wrapper strips ``pin_memory`` and,
    for a ``cuda`` device request, returns a real CPU tensor whose ``.device``
    is overridden to report cuda (via ``_FakeCudaTensor``). That lets us drive
    the GDR probe and control-buffer allocation on a "cuda" transport without
    real hardware.
    """

    def wrapper(*args: object, **kwargs: object) -> torch.Tensor:
        device = kwargs.pop("device", None)
        kwargs.pop("pin_memory", None)
        if device is not None and torch.device(device).type == "cuda":
            # Size is the first positional arg in the call sites we patch.
            nbytes = int(args[0]) if args else int(kwargs.get("size", 0))
            return _FakeCudaTensor(nbytes)  # type: ignore[return-value]
        return real_zeros(*args, **kwargs)

    return wrapper


class TestRDMAGdrProbe(_RDMAMockBase):
    """B3: startup GDR probe selects GPU-direct vs pinned-CPU staging."""

    def _new_cuda_transport(
        self,
        max_gpu_snapshot_bytes: int = 4 << 30,
    ) -> RDMATransport:
        """Construct an RDMATransport on a (faked) cuda device.

        Patches ``torch.zeros`` / ``torch.empty`` (strip ``pin_memory`` and fake
        the cuda allocations for the probe + control buffer) and ``torch.cuda``
        (a ``_FakeCudaModule`` providing the copy-stream / event machinery added
        by B1) so both construction and the post-construction ``_build_snapshot``
        calls run on a CPU-only host. The patches stay active until
        ``_teardown_patches`` (called from each test's ``finally``).
        """
        from torchft.checkpointing import rdma_transport as mod

        real_zeros = torch.zeros
        real_empty = torch.empty

        def empty_no_pin(*args: object, **kwargs: object) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        self._patches = [
            patch.object(mod.torch, "cuda", _FakeCudaModule()),
            patch.object(
                mod.torch, "zeros", side_effect=_fake_cuda_zeros(real_zeros)
            ),
            patch.object(mod.torch, "empty", side_effect=empty_no_pin),
        ]
        for p in self._patches:
            p.start()
        try:
            return RDMATransport(
                device=torch.device("cuda:0"),
                timeout=timedelta(seconds=10),
                max_gpu_snapshot_bytes=max_gpu_snapshot_bytes,
            )
        except Exception:
            self._teardown_patches()
            raise

    def _teardown_patches(self) -> None:
        for p in getattr(self, "_patches", []):
            p.stop()
        self._patches = []

    def test_probe_success_keeps_gpu_tensors_on_gpu(self) -> None:
        """When RdmaMemory accepts CUDA tensors, the probe passes (gdr_ok)."""
        _MockRdmaMemory.raise_on_cuda = False
        transport = self._new_cuda_transport(max_gpu_snapshot_bytes=10_000)
        try:
            self.assertTrue(transport._gdr_ok)

            tensors = [_FakeCudaTensor(600), _FakeCudaTensor(600)]
            sd_meta = _make_cuda_only_state_dict_meta([600, 600])

            from torchft.checkpointing import rdma_transport as mod

            with patch.object(mod, "_spill_to_pinned_cpu") as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            # GDR ok + within budget -> nothing spilled.
            spill_mock.assert_not_called()
            self.assertEqual(len(snap.tensor_snapshots), 2)
        finally:
            transport.shutdown()
            self._teardown_patches()

    def test_probe_failure_routes_all_gpu_tensors_through_pinned_cpu(self) -> None:
        """Probe failure forces EVERY GPU tensor through the spill path.

        Even with a generous ``max_gpu_snapshot_bytes`` that would normally
        keep all tensors on GPU, a failed GDR probe must stage every GPU
        tensor through pinned CPU so checkpoints still succeed without GDR.
        """
        _MockRdmaMemory.raise_on_cuda = True
        # Huge budget: without the probe-failure override nothing would spill.
        transport = self._new_cuda_transport(max_gpu_snapshot_bytes=1 << 40)
        try:
            self.assertFalse(transport._gdr_ok)

            tensors = [_FakeCudaTensor(128) for _ in range(4)]
            sd_meta = _make_cuda_only_state_dict_meta([128, 128, 128, 128])

            # ``raise_on_cuda`` only affects CUDA-backed tensors; the spilled
            # pinned-CPU buffers and manifest are CPU so RdmaMemory accepts
            # them. Wrap the real spill to count how many GPU tensors spilled.
            from torchft.checkpointing import rdma_transport as mod

            real_spill = mod._spill_to_pinned_cpu

            def counting_spill(
                t: object, copy_stream: object = None
            ) -> torch.Tensor:
                nbytes = t.untyped_storage().nbytes()
                cpu = torch.empty(nbytes, dtype=torch.uint8)  # no pin on CI
                cpu.copy_(t.view(torch.uint8), non_blocking=False)
                return cpu

            with patch.object(
                mod, "_spill_to_pinned_cpu", side_effect=counting_spill
            ) as spill_mock:
                snap = transport._build_snapshot(sd_meta, tensors, step=1)

            # All 4 GPU tensors were staged through pinned CPU.
            self.assertEqual(spill_mock.call_count, 4)
            self.assertEqual(len(snap.tensor_snapshots), 4)
            for ts in snap.tensor_snapshots:
                self.assertEqual(ts.dtype, torch.uint8)
                self.assertEqual(ts.device.type, "cpu")
            del real_spill
        finally:
            transport.shutdown()
            self._teardown_patches()

    def test_cpu_device_probe_is_noop(self) -> None:
        """On a CPU device the probe never runs and gdr_ok stays True."""
        # Even with raise_on_cuda set, a CPU transport must construct fine and
        # report gdr_ok=True (the probe short-circuits for non-cuda devices).
        _MockRdmaMemory.raise_on_cuda = True
        transport = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            self.assertTrue(transport._gdr_ok)
        finally:
            transport.shutdown()


# ---------------------------------------------------------------------------
# B5b: in-place receive must check the FULL device (type AND index).
# ---------------------------------------------------------------------------


def _fake_device_tensor(numel: int, device: torch.device) -> torch.Tensor:
    """Return a real ``torch.Tensor`` whose ``.device`` reports ``device``.

    Backed by CPU storage so it works on CI, but ``.device`` is overridden via
    a lightweight subclass to simulate an in-place destination living on a
    specific cuda index without real GPUs. Note ``type(t) is torch.Tensor`` is
    False for this subclass, so ``_cast_tensor`` (which only accepts standard
    tensors) rejects it — we use that as the "device check passed" signal in
    the acceptance tests below.
    """

    class _FakeDeviceTensor(torch.Tensor):
        @property
        def device(self) -> torch.device:  # type: ignore[override]
            return device

    return _FakeDeviceTensor(torch.zeros(numel, dtype=torch.float32))


class TestRDMAInPlaceFullDeviceCheck(_RDMAMockBase):
    """B5b: reject in-place destinations on a different cuda index."""

    def _make_leaf(self, numel: int) -> _RDMATensorLeaf:
        nbytes = numel * 4
        meta = _TensorMeta(
            shape=torch.Size([numel]),
            dtype=torch.float32,
            storage_offset=0,
            stride=(1,),
            nbytes=nbytes,
        )
        # remote_buffer is unused: the cross-device check fires before any read.
        return _RDMATensorLeaf(
            meta=meta, remote_buffer=_MockRdmaRemoteBuffer(addr=1, nbytes=nbytes)
        )

    def test_inplace_wrong_cuda_index_is_rejected(self) -> None:
        """A destination on cuda:1 is refused when the transport is on cuda:0."""
        transport = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            # Force the transport device to cuda:0 so the device comparison is
            # meaningful (no allocation happens in this path).
            transport._device = torch.device("cuda:0")

            leaf = self._make_leaf(4)
            path = ("w",)
            wrong = _fake_device_tensor(4, torch.device("cuda:1"))
            dst_lookup = {path: wrong}

            mock_transport = _MockRdmaTransport(torch.device("cuda:0"))
            with self.assertRaisesRegex(RuntimeError, "across devices"):
                transport._read_one_tensor(
                    mock_transport, path, leaf, dst_lookup
                )
        finally:
            transport.shutdown()

    def test_inplace_matching_cuda_index_is_accepted(self) -> None:
        """A destination on the same cuda index passes the device check.

        We can't run a real cuda RDMA read on CI, so we prove the device check
        passed by observing that control flow reached ``_cast_tensor`` — which
        rejects our non-standard tensor subclass with a distinct assertion —
        rather than being rejected earlier as cross-device.
        """
        transport = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            transport._device = torch.device("cuda:0")

            leaf = self._make_leaf(4)
            path = ("w",)
            same_index = _fake_device_tensor(4, torch.device("cuda:0"))
            dst_lookup = {path: same_index}

            mock_transport = _MockRdmaTransport(torch.device("cuda:0"))
            # Reaching _cast_tensor (which rejects the non-standard subclass)
            # proves the device check accepted the same-index destination.
            with self.assertRaisesRegex(
                AssertionError, "can only cast standard tensors"
            ):
                transport._read_one_tensor(
                    mock_transport, path, leaf, dst_lookup
                )
        finally:
            transport.shutdown()

    def test_inplace_bare_cuda_matches_indexed_destination(self) -> None:
        """A bare ``cuda`` transport device matches a ``cuda:0`` destination.

        ``_same_device`` resolves a ``None`` index to the current device, so a
        destination explicitly on ``cuda:0`` is accepted by a transport whose
        device is bare ``cuda`` (single-GPU default).
        """
        transport = RDMATransport(
            device=torch.device("cpu"), timeout=timedelta(seconds=10)
        )
        try:
            transport._device = torch.device("cuda")  # no index

            leaf = self._make_leaf(4)
            path = ("w",)
            dst = _fake_device_tensor(4, torch.device("cuda:0"))
            dst_lookup = {path: dst}

            mock_transport = _MockRdmaTransport(torch.device("cuda"))
            with patch("torch.cuda.current_device", return_value=0):
                with self.assertRaisesRegex(
                    AssertionError, "can only cast standard tensors"
                ):
                    transport._read_one_tensor(
                        mock_transport, path, leaf, dst_lookup
                    )
        finally:
            transport.shutdown()
