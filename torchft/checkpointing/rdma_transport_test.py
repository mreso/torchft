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

    def __init__(self, tensor: torch.Tensor, cache_reg: bool = False) -> None:
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
    ) -> RDMATransport:
        return RDMATransport(
            device=torch.device("cpu"),
            timeout=timeout,
            state_dict=state_dict_fn,
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
        """A receiver that gets to READY but never sends DONE is timed out and the lock released."""
        # Short timeout so the test doesn't take forever.
        transport = self._new_transport(timeout=timedelta(milliseconds=400))
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
                # Handler is blocked in _recv_frame waiting for DONE; it should
                # time out per the transport timeout and release the r_lock.
                self._wait_for(
                    lambda: not transport._checkpoint_lock.w_locked(),
                    timeout=5,
                )
                # Peer entry should be cleaned up after timeout.
                self._wait_for(
                    lambda: len(transport._peers) == 0, timeout=2
                )
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

            # disallow_checkpoint should now succeed quickly because the
            # reader released its lock on timeout.
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
