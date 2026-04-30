# Engineering Plan: TorchComms RDMA Checkpoint Transport (Final)

This plan merges two independent design proposals (Plan A and Plan B) for implementing an RDMA-based `CheckpointTransport` using `torchcomms._transport.RdmaTransport`. Where the plans disagree, resolutions are noted inline.

---

## 1. Architecture & Connection Model

### Goal

Implement `RDMATransport(CheckpointTransport[T])` that uses `torchcomms._transport.RdmaTransport` and `RdmaMemory` for zero-copy RDMA checkpoint transfers during fault-tolerant healing. This avoids the HTTP serialization overhead of `HTTPTransport` and the ProcessGroup dependency of `PGTransport`.

### Connection Establishment via Existing Manager Metadata Flow

The existing manager gRPC metadata path is sufficient -- no new RPCs or protocol changes required:

1. **Sender** calls `transport.metadata()` which returns bootstrap info (RDMA bind address).
2. Manager stores it via `checkpoint_metadata` in the quorum RPC (`manager.py:647`, `manager.rs:359-361`).
3. **Receiver** fetches it via `primary_client._checkpoint_metadata(group_rank)` (`manager.py:775-776`).
4. Receiver passes the metadata string to `transport.recv_checkpoint()` (`manager.py:792`).

This is exactly how `HTTPTransport` works today (it passes an HTTP URL). For RDMA, the metadata string carries a serialized `_RDMABootstrapMeta`.

### One Transport Instance Per Peer

**Resolution (Plan B wins):** The `RdmaTransport` C++ implementation uses a single virtual circuit slot (`kDummyRank = 0` in `RdmaTransport.cpp:18`). `bind()` + `connect()` establishes exactly one peer connection per instance. This is confirmed by `TransportTest.py:33-53` where both sides call `bind()` and `connect()` symmetrically.

Therefore, the sender must maintain a **pool of `RdmaTransport` instances** -- one per connected peer. Plan A's assumption that a single `bind()` can accept multiple connections is incorrect.

### Symmetric Connection

Both sides must call `bind()` to get a URL, exchange URLs, and then both call `connect(peer_url)`. After connection, both sides can issue `read()` and `write()`. The receiver creates its transport in `recv_checkpoint()` and connects to the sender's advertised address. The sender must also connect back to the receiver's address, which requires the receiver to advertise its own bind address.

### Connection Flow

```
Sender (healthy replica)                         Receiver (recovering replica)
─────────────────────────                        ──────────────────────────────
RDMATransport.__init__():
  creates peer pool (empty)
  metadata() returns serialized
    _RDMABootstrapMeta (bind addr,
    control buffer handle)
        │
        ├──── quorum RPC ──── ManagerServer stores metadata
        │                           │
        │                     checkpoint_metadata RPC
        │                           │
        │                     recv_checkpoint(metadata=bootstrap_meta)
        │                           │
        │                     creates RdmaTransport(device)
        │                       .bind() -> recv_addr
        │                       .connect(sender_addr)
        │                           │
  send_checkpoint():                │
    get_or_create_peer(rank):       │
      .connect(recv_addr)  ◄───── exchange addrs via control channel ─────►
        │                           │
  RDMA: receiver reads ◄────────── RDMA read(local_buf, remote_buffer)
    control record, manifest,       │
    tensor data                     reconstruct state_dict
```

### Key Design Decision: Pull-Based Transfer

**Resolution (both plans agree):** The transfer model is pull-based. The sender registers memory and makes it available; receivers issue RDMA reads to pull data. This is inherently parallel for multiple receivers without sender-side threading overhead -- each receiver independently reads from the same immutable snapshot.

---

## 2. Wire Protocol

### Three-Level Protocol

**Resolution (Plan B wins on control record indirection; Plan A wins on simplicity):**

Plan A embedded `RemoteBuffer` handles directly in the pickled metadata sent through Phase 1. Plan B added a control record + manifest indirection layer. Plan B's approach is better because:

1. The manager metadata string has a practical size limit (it's stored in gRPC/protobuf). Embedding N `RdmaRemoteBuffer` objects (one per tensor) in this string doesn't scale.
2. A stable control record lets the receiver discover manifest location without re-fetching manager metadata.
3. The control record enables atomic publish: the sender can prepare the full manifest + tensor memory, then flip the control record to `READY` in one write.

However, Plan B's control record is overly complex for V1. We simplify to two levels:

#### Level 1: Bootstrap Metadata (via manager gRPC)

`metadata()` returns a base64-encoded pickle of:

```python
@dataclass
class _RDMABootstrapMeta:
    version: int                           # protocol version for forward compat
    bind_addr: bytes                       # RdmaTransport.bind() result
    control_remote_buffer: RdmaRemoteBuffer  # handle to the stable control buffer
    control_buffer_nbytes: int
```

This is small and fixed-size -- safe for the manager metadata string.

#### Level 2: Control Record (stable RDMA-readable buffer on sender)

A fixed-size CPU-pinned buffer registered once in `__init__`:

```python
@dataclass
class _RDMAControlRecord:
    version: int
    step: int
    status: str  # "EMPTY" | "READY" | "DISALLOWED"
    manifest_nbytes: int
    manifest_remote_buffer: RdmaRemoteBuffer  # points to per-step manifest
```

The receiver RDMA-reads this control record using the `control_remote_buffer` from bootstrap metadata, then checks `status == "READY"` and `step == expected_step` before proceeding.

#### Level 3: Per-Step Manifest + Tensor Data

The manifest contains the full state dict structure plus `RdmaRemoteBuffer` handles for each tensor:

```python
@dataclass
class _RDMATensorLeaf:
    meta: _TensorMeta                # shape, dtype, stride, storage_offset, nbytes
    remote_buffer: RdmaRemoteBuffer  # RDMA handle to the tensor's raw bytes

@dataclass
class _RDMADTensorLeaf:
    local: _RDMATensorLeaf
    spec: _DTensorSpec

@dataclass
class _RDMAManifest:
    step: int
    treespec: TreeSpec
    paths: list[KeyPath]
    leaves: list[Union[object, _RDMATensorLeaf, _RDMADTensorLeaf]]
```

**Transfer sequence:**
1. Receiver RDMA-reads the control record (small, fixed size).
2. Receiver validates step + status, then RDMA-reads the manifest (variable size, from `manifest_remote_buffer`).
3. Receiver unpickles the manifest, pre-allocates destination tensors, then RDMA-reads each tensor using the per-leaf `remote_buffer`.

### State Dict Decomposition

Reuse `PGTransport`'s existing `_prepare_state_dict()` (`pg_transport.py:106-146`) which decomposes a `state_dict` into `(_StateDictMeta, list[torch.Tensor])`. The `_RDMAManifest` extends this with `RdmaRemoteBuffer` handles per tensor.

---

## 3. Memory Management

### Sender-Side Snapshot

**Resolution (Plan B wins):** Plan A exposed live tensor references directly via RDMA. Plan B's immutable snapshot approach is safer: `send_checkpoint()` receives a materialized `state_dict` from `Manager._manager_state_dict()` (`manager.py:761`), which already reads under `_state_dict_lock.r_lock()` (`manager.py:958-965`). The snapshot policy:

```python
@dataclass
class _SnapshotGeneration:
    generation: int
    step: int
    manifest_tensor: torch.Tensor       # CPU-pinned, holds pickled manifest
    manifest_mem: RdmaMemory
    tensor_snapshots: list[torch.Tensor] # clones or staged copies
    tensor_mems: list[RdmaMemory]        # registered memory for each
```

- For GPU tensors: clone on the same device (fast, keeps GPU-direct RDMA path).
- For CPU tensors: clone to contiguous CPU memory.
- If cumulative GPU snapshot bytes exceed a configurable `max_gpu_snapshot_bytes`, fall back to pinned CPU staging with async D2H copy (similar to `HTTPTransport.send_checkpoint()` at `http_transport.py:224-233`).

### Generation Lifecycle

**Resolution (Plan B wins):** Plan A didn't address multi-generation memory. Plan B's two-generation model is correct:

- Keep at most two generations alive: current (`READY`) and previous (`RETIRED`).
- Free the previous generation when the next `send_checkpoint()` successfully publishes a new one.
- This prevents use-after-free if a receiver is still reading from the previous generation during the transition.

### RdmaMemory Lifetime

`RdmaMemory` objects must stay alive while RDMA reads are in flight. The `_SnapshotGeneration` dataclass holds all references. `RdmaMemory(cache_reg=True)` should be used for the long-lived control buffer; per-step snapshot buffers use `cache_reg=False` and are deregistered when the generation is freed.

### Receiver-Side Allocation

The receiver pre-allocates destination tensors from `_TensorMeta.nbytes` in the manifest. If an `state_dict` callback is provided (like `PGTransport`'s `state_dict` parameter at `pg_transport.py:190`), reuse existing tensor storage for zero-copy receive into pre-allocated tensors.

---

## 4. Concurrency

### Multi-Destination Send (Sender Side)

The sender must maintain a peer pool since each `RdmaTransport` supports only one connection:

```python
@dataclass
class _PeerConnection:
    transport: RdmaTransport
    bind_addr: bytes
    connected: bool
    lock: threading.Lock
```

`send_checkpoint(dst_ranks, ...)`:
1. Build one immutable snapshot generation (shared across all receivers).
2. Update the control record to `READY`.
3. No per-receiver data push needed -- receivers pull via RDMA reads.

The sender does **not** pre-connect to receivers in `send_checkpoint()` since it doesn't know receiver addresses yet. Instead, connections are established lazily when the receiver reaches out via the control channel (or a side-channel for symmetric connect address exchange).

### Receiver-Side Transfer

`recv_checkpoint()`:
1. Create a fresh `RdmaTransport`, `bind()`, `connect()` to sender.
2. Exchange bind addresses with sender for symmetric connection (required by the API).
3. RDMA-read control record, validate step/status.
4. RDMA-read manifest, unpickle.
5. For each tensor leaf: allocate local buffer, wrap as `RdmaMemory`, RDMA-read from `remote_buffer`.
6. Reconstruct state_dict via `tree_unflatten()`.

**Initial implementation: serial per-tensor reads.** The Python binding is synchronous (`read()` blocks). Parallelism comes from multiple receivers reading concurrently from the same immutable snapshot.

### Symmetric Connect Challenge

The `RdmaTransport` API requires symmetric `bind()` + `connect()` on both sides. This creates a bootstrapping challenge: the receiver knows the sender's bind address (from manager metadata), but the sender doesn't know the receiver's address until the receiver connects.

**Solution:** Use a small TCP side-channel for address exchange during the symmetric connect handshake. The sender listens on a TCP port (included in `_RDMABootstrapMeta`), the receiver connects and sends its RDMA bind address, then both sides call `connect()`. This is a one-time setup cost per peer pair.

```python
@dataclass
class _RDMABootstrapMeta:
    version: int
    bind_addr: bytes
    control_remote_buffer: RdmaRemoteBuffer
    control_buffer_nbytes: int
    handshake_port: int  # TCP port for symmetric connect address exchange
```

---

## 5. Locking: `disallow_checkpoint()` / `allow_checkpoint()`

### Resolution (hybrid approach)

Plan A reused `HTTPTransport`'s `RWLock` directly. Plan B proposed a custom generation-based state machine (EMPTY/READY/DISALLOWED/RETIRED). The right answer is a hybrid:

**Use `RWLock` from `torchft/checkpointing/_rwlock.py` for the reader/writer fence** (proven, tested, matches the `CheckpointTransport` contract) **plus the control record status field for RDMA-level visibility**.

```python
class RDMATransport(CheckpointTransport[T]):
    def __init__(self, ...):
        self._checkpoint_lock = RWLock(timeout=timeout.total_seconds())
        self._disallowed = False
        self.disallow_checkpoint()  # start disallowed, like HTTPTransport:68

    def send_checkpoint(self, dst_ranks, step, state_dict, timeout):
        # Build snapshot, register memory, write manifest
        snapshot = self._build_snapshot(state_dict, step)
        # Atomically update control record to READY
        self._update_control_record(step, "READY", snapshot)
        self._current_snapshot = snapshot
        self._allow_checkpoint(step)

    def disallow_checkpoint(self):
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.w_acquire()
            # After lock acquired: no active readers
            self._update_control_record(self._step, "DISALLOWED", None)
            # Previous snapshot can now be retired (next send will free it)

    def _allow_checkpoint(self, step):
        self._step = step
        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.w_release()
```

Receivers acquire `r_lock()` during RDMA reads. When `disallow_checkpoint()` is called, `w_acquire()` blocks until all active reads complete. After the lock is held, the control record is flipped to `DISALLOWED` and the snapshot can be retired on the next step.

**Why not Plan B's "no need to wait for readers" approach?** Plan B argued that since snapshots are immutable clones, `disallow_checkpoint()` doesn't need to wait. However, this creates a subtle bug: if a receiver is mid-read when the sender frees the snapshot's `RdmaMemory` objects, the RDMA NIC will access deregistered memory. The `RWLock` fence ensures all reads complete before deregistration.

---

## 6. DTensor Handling

Both plans agree: reuse `PGTransport`'s DTensor handling unchanged.

- Flatten: extract `v._local_tensor`, prepare as `_TensorMeta`, wrap in `_RDMADTensorLeaf` with `v._spec` (`pg_transport.py:120-130`).
- Transfer: only the local shard bytes go over RDMA.
- Reconstruct: `DTensor(tensor, spec, requires_grad=False)` (`pg_transport.py:296-298`).

No cross-rank re-sharding. This is checkpoint transport, not redistribution.

---

## 7. Non-Tensor State

Both plans agree: non-tensor leaves (optimizer scalars, step counters, RNG state, strings, ints) are stored directly in the manifest's `leaves` list and travel as part of the pickled manifest (Level 3). No RDMA read needed for these -- they are small enough to fit in the manifest buffer.

Guardrail: add a manifest size cap (default 64 MiB) with a clear error if exceeded.

---

## 8. Manager Integration

### Resolution (Plan B wins on identifying the issue; Plan A's "no changes" is almost correct)

**Plan B correctly identified** that `Manager.__init__` constructs the default transport at line 277-281, before `self._client` is created at line 337. However, Plan B's proposed `peer_metadata_lookup` callback is unnecessary.

**The RDMA transport does NOT need a `peer_metadata_lookup` callback.** Here's why:

- The `recv_checkpoint()` method already receives the sender's metadata string as a parameter (`metadata: str`), passed by the Manager at `manager.py:792-794`. The receiver doesn't need to look up peer metadata itself.
- The `send_checkpoint()` method doesn't need peer metadata either -- it publishes its own snapshot and waits for receivers to connect.
- The only metadata exchange happens through the existing quorum + `checkpoint_metadata` RPC path, which the Manager already handles.

**Manager changes needed: zero.** The transport is injected via `checkpoint_transport` parameter (`manager.py:182, 277-285`). Users pass it explicitly:

```python
manager = Manager(
    pg=pg,
    checkpoint_transport=RDMATransport(device=device, timeout=timeout),
    ...
)
```

The only gap is that if `checkpoint_transport is None` and we want RDMA as default, we'd need an opt-in mechanism. But that's a follow-up -- the initial implementation uses explicit construction.

---

## 9. Shared Helpers: Extract or Import?

### Resolution (Plan A wins)

Plan B proposed extracting `_TensorMeta`, `_DTensorMeta`, `_StateDictMeta`, `_prepare_state_dict`, `_cast_tensor`, and `_timeit` into a new `_state_dict_meta.py` module. Plan A imports them directly from `pg_transport.py`.

**Plan A's approach is better for V1:**
- These functions are already module-level in `pg_transport.py` (lines 32-165) and are importable.
- Extracting to a new module is a refactoring change that touches `pg_transport.py` imports and requires updating existing tests.
- The functions are small, stable, and don't have circular dependencies.
- A future cleanup PR can extract them if a third transport needs them.

```python
from torchft.checkpointing.pg_transport import (
    _cast_tensor,
    _DTensorMeta,
    _prepare_state_dict,
    _prepare_tensor,
    _StateDictMeta,
    _TensorMeta,
    _timeit,
)
```

---

## 10. Fallback: No RDMA Hardware

### Detection

```python
def _rdma_available() -> bool:
    try:
        from torchcomms._transport import RdmaTransport
        return RdmaTransport.supported()
    except ImportError:
        return False
```

`RdmaTransport.supported()` is a static method that checks for a working InfiniBand NIC. It's called once and cached globally (`RdmaTransport.cpp:167-200`).

### Transparent Fallback

```python
class RDMATransport(CheckpointTransport[T]):
    def __init__(self, device, timeout, ...):
        if not _rdma_available():
            from torchft.checkpointing.http_transport import HTTPTransport
            logger.warning("RDMA not available, falling back to HTTPTransport")
            self._fallback = HTTPTransport(timeout=timeout, num_chunks=0)
            self._rdma = None
            return
        self._fallback = None
        # ... RDMA init ...
```

All methods delegate to `self._fallback` when set. The metadata string is prefixed (`rdma:` for RDMA, `http://` for HTTP) so receivers can detect protocol mismatch.

### No Mixed Mode

**Resolution (both plans agree):** Don't support mixed RDMA/HTTP in the same quorum. If the sender advertises `rdma:` metadata but the receiver can't do RDMA, raise a clear error. Keep it simple for V1.

---

## 11. Testing Strategy

### Unit Tests (`torchft/checkpointing/rdma_transport_test.py`)

```python
class RDMATransportTest(TestCase):
    @skipUnless(_rdma_available(), "RDMA hardware not available")
    def test_rdma_transport_cpu(self):
        device = torch.device("cpu")
        def init(rank, world_size):
            return RDMATransport(timeout=timedelta(seconds=10), device=device)
        run_multi_recovery_test(self, init, device=device)

    @skipUnless(_rdma_available() and torch.cuda.device_count() >= 3, "need RDMA + 3 GPUs")
    def test_rdma_transport_cuda(self):
        device = torch.device("cuda")
        def init(rank, world_size):
            torch.cuda.set_device(rank)
            return RDMATransport(timeout=timedelta(seconds=10), device=device)
        run_multi_recovery_test(self, init, device=device)

    def test_fallback_to_http(self):
        # Mock RdmaTransport.supported() -> False, verify HTTPTransport used
        ...

    def test_bootstrap_meta_roundtrip(self):
        # Encode/decode _RDMABootstrapMeta through base64 pickle
        ...

    def test_control_record_lifecycle(self):
        # EMPTY -> READY -> DISALLOWED transitions
        ...

    def test_manifest_roundtrip(self):
        # Encode/decode _RDMAManifest with tensors, DTensors, non-tensor leaves
        ...

    def test_step_mismatch(self):
        # Verify step validation in recv_checkpoint
        ...

    def test_disallow_blocks_new_reads(self):
        # Verify RWLock behavior
        ...
```

The `run_multi_recovery_test` harness (`transport_test.py:50-160`) already tests 3-node recovery, 2-node recovery, timeout behavior, and sequential rounds with `disallow_checkpoint()` between them.

### Integration Tests (extend `manager_integ_test.py`)

```python
@skipUnless(TORCHCOMMS_AVAILABLE and _rdma_available(), "need torchcomms + RDMA")
def test_ddp_rdma_transport_recovery(self):
    """Full end-to-end: Manager + DDP + RDMATransport healing."""
    ...
```

### Benchmarks (later milestone)

**Resolution (Plan B wins):** Don't block initial landing on performance benchmarks. Functional correctness and fallback safety first. Benchmarks (`rdma_transport_bench.py`) follow the style of `http_transport_bench.py` / `pg_transport_bench.py` in a follow-up.

---

## 12. File Layout

### New Files

```
torchft/
├── checkpointing/
│   ├── rdma_transport.py           # RDMATransport implementation + protocol dataclasses
│   └── rdma_transport_test.py      # Unit tests
```

**Resolution (Plan A wins):** Plan B proposed splitting protocol dataclasses into `_rdma_protocol.py`. This is unnecessary -- the protocol types are small and tightly coupled to the transport implementation. Keep them in one file for V1.

### Modified Files

- **`torchft/checkpointing/__init__.py`**: Add `RDMATransport` to exports.
- **`torchft/checkpointing/pg_transport.py`**: No changes. RDMA transport imports shared helpers directly.
- **`torchft/manager.py`**: No changes. Transport is injected via `checkpoint_transport` parameter.
- **`torchft/manager_integ_test.py`**: Add RDMA integration test behind availability guard.

### Files NOT Changed

- `torchft/proto/torchft.proto` -- existing `string checkpoint_metadata` field is sufficient.
- `torchft/src/manager.rs` -- already stores/returns opaque metadata strings.
- `torchft/torchcomms.py` -- wraps collective comms, not transport bootstrap.

---

## 13. Class Skeleton

```python
# torchft/checkpointing/rdma_transport.py

import base64
import logging
import pickle
import socket
import struct
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Optional, TypeVar, Union

import torch
from torch.distributed.tensor._dt_init import _DTensorSpec
from torch.utils._pytree import KeyPath, tree_unflatten, TreeSpec

from torchft.checkpointing._rwlock import RWLock
from torchft.checkpointing.pg_transport import (
    _cast_tensor,
    _DTensorMeta,
    _prepare_state_dict,
    _prepare_tensor,
    _StateDictMeta,
    _TensorMeta,
    _timeit,
)
from torchft.checkpointing.transport import CheckpointTransport

logger = logging.getLogger(__name__)
T = TypeVar("T")

_RDMA_META_PREFIX = "rdma:"
_PROTOCOL_VERSION = 1
_CONTROL_BUFFER_NBYTES = 64 * 1024  # 64 KiB


def _rdma_available() -> bool:
    try:
        from torchcomms._transport import RdmaTransport
        return RdmaTransport.supported()
    except ImportError:
        return False


# --- Protocol dataclasses ---

@dataclass
class _RDMABootstrapMeta:
    version: int
    bind_addr: bytes
    control_remote_buffer: object  # RdmaRemoteBuffer (pickleable)
    control_buffer_nbytes: int
    handshake_port: int


@dataclass
class _RDMAControlRecord:
    version: int
    step: int
    status: str  # "EMPTY" | "READY" | "DISALLOWED"
    manifest_nbytes: int
    manifest_remote_buffer: object  # RdmaRemoteBuffer | None


@dataclass
class _RDMATensorLeaf:
    meta: _TensorMeta
    remote_buffer: object  # RdmaRemoteBuffer


@dataclass
class _RDMADTensorLeaf:
    local: _RDMATensorLeaf
    spec: _DTensorSpec


@dataclass
class _RDMAManifest:
    step: int
    treespec: TreeSpec
    paths: list[KeyPath]
    leaves: list[Union[object, _RDMATensorLeaf, _RDMADTensorLeaf]]


@dataclass
class _SnapshotGeneration:
    generation: int
    step: int
    manifest_tensor: torch.Tensor
    manifest_mem: object  # RdmaMemory
    tensor_snapshots: list[torch.Tensor]
    tensor_mems: list[object]  # list[RdmaMemory]


@dataclass
class _PeerConnection:
    transport: object  # RdmaTransport
    bind_addr: bytes
    connected: bool
    lock: threading.Lock


class RDMATransport(CheckpointTransport[T]):
    """
    Checkpoint transport using RDMA via torchcomms._transport.

    Falls back to HTTPTransport when RDMA hardware is not available.

    Args:
        device: device for RDMA operations and tensor staging
        timeout: timeout for RDMA operations
        state_dict: optional callable returning a pre-allocated state_dict
            for inplace receive (avoids allocation on receiver side)
        max_gpu_snapshot_bytes: max GPU memory for tensor snapshots before
            falling back to CPU staging
    """

    def __init__(
        self,
        device: torch.device,
        timeout: timedelta,
        state_dict: Optional[Callable[[], object]] = None,
        max_gpu_snapshot_bytes: int = 4 << 30,
    ) -> None:
        self._device = device
        self._timeout = timeout
        self._state_dict_fn = state_dict
        self._max_gpu_snapshot_bytes = max_gpu_snapshot_bytes

        if not _rdma_available():
            from torchft.checkpointing.http_transport import HTTPTransport
            logger.warning("RDMA not available, falling back to HTTPTransport")
            self._fallback: Optional[object] = HTTPTransport(
                timeout=timeout, num_chunks=0
            )
            self._rdma = None
            return

        self._fallback = None

        from torchcomms._transport import RdmaMemory, RdmaTransport as _RdmaTransport

        # Long-lived control buffer (CPU-pinned, registered once)
        self._control_tensor = torch.zeros(
            _CONTROL_BUFFER_NBYTES, dtype=torch.uint8, pin_memory=True
        )
        self._control_mem = RdmaMemory(self._control_tensor, cache_reg=True)
        self._control_remote_buffer = self._control_mem.to_remote_buffer()

        # TCP handshake server for symmetric RDMA connect
        self._handshake_server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self._handshake_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._handshake_server.bind(("::", 0))
        self._handshake_server.listen(16)
        self._handshake_port = self._handshake_server.getsockname()[1]

        # Primary RDMA transport for this node (used to generate bind address)
        self._primary_transport = _RdmaTransport(device)
        self._bind_addr: bytes = self._primary_transport.bind()

        # Peer connection pool: rank -> _PeerConnection
        self._peers: dict[int, _PeerConnection] = {}
        self._peers_lock = threading.Lock()

        # Locking
        self._checkpoint_lock = RWLock(timeout=timeout.total_seconds())
        self._disallowed = False
        self._step = -1

        # Snapshot generations
        self._current_snapshot: Optional[_SnapshotGeneration] = None
        self._previous_snapshot: Optional[_SnapshotGeneration] = None
        self._generation = 0

        # Write initial EMPTY control record
        self._update_control_record(-1, "EMPTY", None)

        # Start disallowed (like HTTPTransport)
        self.disallow_checkpoint()

        # Start handshake listener thread
        self._shutdown_event = threading.Event()
        self._handshake_thread = threading.Thread(
            target=self._run_handshake_server, daemon=True
        )
        self._handshake_thread.start()

    def metadata(self) -> str:
        if self._fallback is not None:
            return self._fallback.metadata()
        meta = _RDMABootstrapMeta(
            version=_PROTOCOL_VERSION,
            bind_addr=self._bind_addr,
            control_remote_buffer=self._control_remote_buffer,
            control_buffer_nbytes=_CONTROL_BUFFER_NBYTES,
            handshake_port=self._handshake_port,
        )
        return f"{_RDMA_META_PREFIX}{base64.b64encode(pickle.dumps(meta)).decode()}"

    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        if self._fallback is not None:
            return self._fallback.send_checkpoint(dst_ranks, step, state_dict, timeout)

        from torchcomms._transport import RdmaMemory

        with _timeit("rdma: preparing state_dict"):
            sd_meta, tensors = _prepare_state_dict(state_dict, step, self._device)

        with _timeit("rdma: building snapshot"):
            snapshot = self._build_snapshot(sd_meta, tensors, step)

        # Retire previous snapshot, install new one
        self._previous_snapshot = self._current_snapshot
        self._current_snapshot = snapshot

        # Update control record atomically to READY
        self._update_control_record(step, "READY", snapshot)
        self._allow_checkpoint(step)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        if self._fallback is not None:
            return self._fallback.recv_checkpoint(src_rank, metadata, step, timeout)

        from torchcomms._transport import RdmaMemory, RdmaTransport as _RdmaTransport

        bootstrap = pickle.loads(
            base64.b64decode(metadata.removeprefix(_RDMA_META_PREFIX))
        )
        assert isinstance(bootstrap, _RDMABootstrapMeta)

        # Create fresh transport, establish symmetric connection
        recv_transport = _RdmaTransport(self._device)
        recv_bind_addr = recv_transport.bind()

        # Exchange addresses via TCP handshake
        sender_addr = self._symmetric_connect(
            recv_transport, recv_bind_addr, bootstrap
        )

        # Phase 1: Read control record
        control_record = self._read_control_record(
            recv_transport, bootstrap, step, timeout
        )

        # Phase 2: Read manifest
        manifest = self._read_manifest(recv_transport, control_record)
        assert manifest.step == step

        # Phase 3: Read tensor data
        values = self._read_tensors(recv_transport, manifest)

        return tree_unflatten(values, manifest.treespec)

    def disallow_checkpoint(self) -> None:
        if self._fallback is not None:
            return self._fallback.disallow_checkpoint()
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.w_acquire()
            self._update_control_record(self._step, "DISALLOWED", None)

    def _allow_checkpoint(self, step: int) -> None:
        self._step = step
        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.w_release()

    def shutdown(self, wait: bool = True) -> None:
        if self._fallback is not None:
            return self._fallback.shutdown(wait=wait)
        self._shutdown_event.set()
        self._handshake_server.close()
        if wait:
            self._handshake_thread.join()

    # --- Internal methods (sketched) ---

    def _build_snapshot(
        self, sd_meta: _StateDictMeta, tensors: list[torch.Tensor], step: int
    ) -> _SnapshotGeneration:
        """Clone tensors into immutable snapshot, register with RdmaMemory."""
        ...

    def _update_control_record(
        self, step: int, status: str, snapshot: Optional[_SnapshotGeneration]
    ) -> None:
        """Serialize control record into the stable control buffer."""
        ...

    def _read_control_record(
        self, transport: object, bootstrap: _RDMABootstrapMeta,
        expected_step: int, timeout: timedelta
    ) -> _RDMAControlRecord:
        """RDMA-read and validate the sender's control record."""
        ...

    def _read_manifest(
        self, transport: object, control: _RDMAControlRecord
    ) -> _RDMAManifest:
        """RDMA-read and unpickle the per-step manifest."""
        ...

    def _read_tensors(
        self, transport: object, manifest: _RDMAManifest
    ) -> list[object]:
        """RDMA-read each tensor leaf, reconstruct with as_strided."""
        ...

    def _symmetric_connect(
        self, recv_transport: object, recv_bind_addr: bytes,
        bootstrap: _RDMABootstrapMeta
    ) -> bytes:
        """Exchange RDMA addresses via TCP, then both sides call connect()."""
        ...

    def _run_handshake_server(self) -> None:
        """Accept TCP connections for symmetric RDMA connect setup."""
        ...
```

---

## 14. Implementation Order

### Phase 1: Core Transport (target: functional correctness)
1. Protocol dataclasses (`_RDMABootstrapMeta`, `_RDMAControlRecord`, `_RDMAManifest`, `_RDMATensorLeaf`, `_RDMADTensorLeaf`).
2. Symmetric connect handshake (TCP side-channel for address exchange).
3. `send_checkpoint` with snapshot build + control record publish.
4. `recv_checkpoint` with control record read + manifest read + tensor reads.
5. Fallback to `HTTPTransport` when RDMA unavailable.
6. Unit tests: `run_multi_recovery_test` with CPU tensors.

### Phase 2: Locking & Lifecycle
1. `RWLock` integration for `disallow_checkpoint()` / `_allow_checkpoint()`.
2. Generation lifecycle (two-generation model, proper cleanup).
3. Locking tests (concurrent readers, disallow blocks new reads).

### Phase 3: In-Place Receive & DTensor
1. Support `state_dict` callback for zero-copy receive into pre-allocated tensors.
2. DTensor leaf handling (pass-through from `_prepare_state_dict`).
3. Strided tensor reconstruction via `torch.as_strided`.

### Phase 4: Integration & Polish
1. Add to `__init__.py` exports.
2. Manager integration test (extend `manager_integ_test.py`).
3. GPU tensor tests (CUDA device, GPU-direct RDMA path).
4. GPU snapshot spill-to-CPU when exceeding `max_gpu_snapshot_bytes`.

### Phase 5: Performance (follow-up)
1. Benchmark (`rdma_transport_bench.py`) comparing against HTTPTransport.
2. Chunked reads for very large tensors (using `to_view(offset, length)`).
3. Parallel RDMA reads within a single receiver (bounded parallelism).

---

## 15. Disagreement Resolution Summary

| Topic | Plan A | Plan B | Resolution |
|-------|--------|--------|------------|
| **Connection model** | Single bind, multiple receivers | One transport per peer | **Plan B.** C++ uses single VC slot (`kDummyRank=0`). |
| **Wire protocol** | Embed remote_buffers in pickled metadata | Control record + manifest indirection | **Plan B** (with simplification). Indirection avoids metadata size limits and enables atomic publish. |
| **Locking** | Reuse HTTPTransport's RWLock directly | Custom generation state machine, no reader wait | **Hybrid.** RWLock for reader fence (prevents use-after-free of `RdmaMemory`) + control record status for RDMA visibility. |
| **Manager changes** | Zero changes needed | `peer_metadata_lookup` callback, construction order fix | **Plan A.** Manager already passes metadata to `recv_checkpoint()`. No callback needed. |
| **Shared helpers** | Import from `pg_transport.py` | Extract to `_state_dict_meta.py` | **Plan A.** Avoid unnecessary refactoring in V1. |
| **Protocol file layout** | All in `rdma_transport.py` | Split into `_rdma_protocol.py` | **Plan A.** Keep in one file -- types are small and tightly coupled. |
| **Multi-generation memory** | Not addressed | Two-generation model | **Plan B.** Prevents use-after-free during generation transitions. |
| **Snapshot model** | Expose live tensor references | Clone into immutable snapshot | **Plan B.** Safer -- no live reference leakage over RDMA. |
| **Benchmarks** | Include in initial landing | Defer to follow-up | **Plan B.** Correctness first, performance second. |
| **Class name** | `RDMATransport` | `TorchCommsRDMATransport` | **`RDMATransport`** (Plan A). Shorter, consistent with `HTTPTransport` / `PGTransport` naming. |

---

## 16. Open Questions & Risk Areas

1. **Symmetric connect bootstrapping**: The TCP side-channel for address exchange adds complexity. If the `torchcomms` API later supports asymmetric connect (bind-only on sender, connect-only on receiver), this can be simplified. Monitor torchcomms development.

2. **RdmaMemory lifetime guarantees**: Verify that holding a Python reference to `RdmaMemory` is sufficient to keep the underlying buffer registered. The C++ destructor calls `RegCache::globalDeregister()` -- Python GC must not collect `RdmaMemory` while views/remote_buffers are in use.

3. **Timeout semantics**: The Python `read()`/`write()` bindings call `.get()` on the C++ `SemiFuture` with no timeout. Long-running RDMA operations may block indefinitely. Consider adding a Python-level timeout wrapper using `threading.Timer` + transport teardown.

4. **Error recovery**: If an RDMA read fails mid-transfer (sender crash, NIC error), the Python `read()` call should raise an exception. Verify this propagates cleanly to `Manager._async_quorum()`'s exception handler (`manager.py:805-807`).

5. **Manifest size cap**: Very large state dicts with many small tensors could produce large manifests. Add a configurable cap (default 64 MiB) and raise a clear error if exceeded.

6. **Control record atomicity**: The control record update (step + status + manifest_remote_buffer) must appear atomic to receivers. Since it's a single `pickle.dumps` into a pre-registered buffer followed by a memory fence, this should be safe, but verify with concurrent reader tests.
