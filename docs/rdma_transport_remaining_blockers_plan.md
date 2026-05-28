# RDMATransport — Remaining Blockers & Fix Plan

Status as of this commit. Tracks what was just fixed, what is still blocking
real GPU+RDMA deployment, and — for each blocker — the fix approach, how to
**simulate it on a machine without RDMA hardware**, and how to verify it on
real hardware.

The simulator (`examples/rdma_transport_sim.py`) and the contract-faithful fake
torchcomms in `rdma_transport_test.py` are the two tools used throughout. The
fake now enforces the real torchcomms API shape (mutable vs immutable views,
int return codes), so a whole class of API-contract bugs fails locally instead
of only on hardware.

---

## Already fixed in this commit

| # | Issue | Fix |
|---|-------|-----|
| F1 | **RDMA reads passed `to_view()` (immutable) where the binding requires `to_mutable_view()` → guaranteed `TypeError` on real hardware.** | All three read sites now use `to_mutable_view()` via a new `_rdma_read()` helper. |
| F2 | `read()` status code ignored. | `_rdma_read()` raises on non-zero status. |
| F3 | Mock didn't model the mutable/immutable view distinction, so it masked F1. | Mock now has `RdmaMemoryView` / `RdmaMemoryMutableView`; `read()` requires mutable, `write()` requires immutable, both return `int`. Reproduction test confirms `to_view()` reads are now rejected. |
| F4 | Handshake advertised `socket.gethostname()` unconditionally (often unresolvable). | `handshake_host` arg + `TORCHFT_RDMA_HANDSHAKE_HOST` env override; falls back to a resolvable hostname, then loopback. Listener is now dual-stack (`IPV6_V6ONLY=0`). **Partial** — see B4 for full cross-host hardening. |
| F5 | `_read_manifest` didn't validate the manifest buffer was present. | Raises if a READY control record has no manifest buffer. |

Verification: `pytest torchft/checkpointing/rdma_transport_test.py` → 43 passed
(2 failures are pre-existing `HTTPTransport` fallback tests that also rely on a
resolvable `gethostname()` — `http_transport.py:173` — and are unrelated to this
feature). `examples/rdma_transport_sim.py` → 3/3 scenarios pass.

---

## BLOCKER B1 — GPU snapshot not synchronized before RDMA exposure

**Severity: blocker (silent data corruption on GPU).**

Root cause: `_build_snapshot` (`rdma_transport.py`) clones GPU tensors / spills
to pinned CPU, then `send_checkpoint` immediately publishes `READY` and releases
the lock. CUDA clones and async D2H copies are *stream-ordered*, not complete.
A receiver can RDMA-read partially-produced bytes. `HTTPTransport` synchronizes
its staging stream; this path does not. `Manager` runs send on
`self._recovery_stream`, which is not synchronized against the stream that last
wrote the model tensors.

Fix:
1. In `_build_snapshot`, after cloning/spilling, record a `torch.cuda.Event` on
   the current stream and `event.synchronize()` (or block the publishing stream
   on it) **before** `_update_control_record(..., "READY", ...)`.
2. Make `send_checkpoint` first make the recovery/copy stream wait on the
   stream(s) that produced the source tensors (accept an optional producer
   stream/event, or synchronize the default stream).
3. Use a dedicated copy stream for the spill D2H and synchronize it (mirrors
   `HTTPTransport`), which also sets up B7 (overlap).

Simulate without hardware:
- Add a unit test that patches `torch.cuda` with a fake stream/event recorder
  and asserts the publish sequence calls `synchronize()` **before** the control
  record flips to `READY`. This catches ordering regressions on CPU.
- Property: in `send_checkpoint`, no `READY` write may be observed before the
  recorded event has been waited on.

Verify on hardware:
- Stress test: one thread mutates the source tensors in a tight loop while a
  peer repeatedly pulls; assert every received checkpoint hashes to a value the
  sender actually published (no torn reads). Run with/without the sync to prove
  the test detects the race.

---

## BLOCKER B2 — Lifetime fence keyed to TCP idle timeout, not RDMA completion

**Severity: blocker (use-after-free of RDMA memory).**

Root cause: the sender's handshake handler holds the reader lock only until it
receives `DONE`, with the socket bounded by `self._timeout`. A healthy but slow
transfer (large checkpoint / slow fabric) that exceeds `self._timeout` causes
`_recv_frame` to time out, the `finally` releases `r_lock`, and the next
generation rotation can free the `RdmaMemory` while the NIC is still reading.
The blocking `transport.read()` calls also have no per-op timeout.

Fix (prefer liveness over idle timeout):
1. Do **not** use an application idle timeout as the completion oracle. Enable
   `SO_KEEPALIVE` (+ `TCP_KEEPIDLE/INTVL/CNT`) on both handshake sockets so a
   *dead* peer is detected, but a *slow-but-alive* peer keeps the fence.
2. Keep a generous absolute safety deadline (≫ expected transfer time), distinct
   from the per-step `timeout`, configurable.
3. Add an explicit per-operation timeout/abort story for `transport.read()`
   (torchcomms read has a timeout internally; surface and tie it to the fence).
4. Alternative/complement: chunk the receiver read loop and have it emit a
   progress frame per chunk; the sender resets its deadline on each frame.

Simulate without hardware:
- Extend the fake transport with an injectable per-read delay. Test that a read
  slower than the old `self._timeout` does **not** release the fence (i.e.
  `disallow_checkpoint()` still blocks) and that the snapshot buffer stays
  registered for the whole read.
- Test that a *dropped* receiver socket releases the fence promptly (liveness).

Verify on hardware:
- Throttle the fabric / use a multi-GB checkpoint so a real read exceeds the old
  timeout; assert no use-after-free (ASAN / registration refcount stays > 0 for
  the duration) and data integrity holds.

---

## BLOCKER B3 — No GPU/GDR capability probe, no staged fallback

**Severity: blocker (works on CPU NIC, fails after deploy on GPU).**

Root cause: `_rdma_available()` only checks `RdmaTransport.supported()` ("a NIC
exists"). The code then assumes `RdmaMemory(cuda_tensor)` and
`RdmaTransport(device=cuda)` work. "IB NIC present" ≠ "GPUDirect RDMA usable for
this GPU/NIC/driver combo". If GDR registration is unavailable, failures surface
only at first checkpoint.

Fix:
1. Add a startup GPU-RDMA probe: allocate a tiny CUDA tensor, attempt
   `RdmaMemory(t)` + a loopback self-`read`, catch failures.
2. On probe failure with `device.type == cuda`: either refuse cuda (explicit
   error) or automatically route all GPU tensors through the pinned-CPU staging
   path (force the spill path regardless of budget).
3. Surface the chosen mode in `metadata()`/logs so sender and receiver agree.

Simulate without hardware:
- Fake `RdmaMemory` gains a flag to raise on CUDA-backed tensors; assert the
  probe detects it and the transport falls back to pinned-CPU staging (using
  `_FakeCudaTensor`-style stand-ins already in the tests).

Verify on hardware:
- Run on a node with GDR enabled (probe passes, GPU stays on GPU) and a node
  with GDR disabled (probe fails, auto-stages through pinned CPU); both produce
  correct checkpoints.

---

## BLOCKER B4 — Cross-host handshake addressing (finish what F4 started)

**Severity: blocker for multi-host; partially mitigated.**

Done in F4: configurable `handshake_host` + env var, dual-stack listener,
loopback fallback. Remaining:
1. Auto-select a routable IP when no override is given (enumerate interfaces;
   prefer the one on the cluster/RDMA subnet) instead of falling back to
   loopback silently for multi-host runs.
2. IPv4-only environments: bind path when IPv6 is unavailable.
3. Document required port reachability (the handshake TCP port is ephemeral;
   provide a way to pin a port / range for firewalls/security groups).

Simulate without hardware:
- Unit-test the IP-selection helper with a faked interface list; assert it
  prefers a given subnet and that an explicit override always wins.
- Two-process (not two-thread) localhost run of the simulator to exercise a real
  TCP connect across process boundaries (add `--procs 2` mode).

Verify on hardware:
- Real 2-host run with a pinned handshake port behind a firewall rule.

---

## RISK B5 — Multi-GPU correctness

Root cause: (a) handshake threads construct `RdmaTransport(self._device)` without
`torch.cuda.set_device`, and CUDA current device is thread-local; (b) in-place
receive accepts any destination where `inplace.device.type == self._device.type`
— `cuda:0` and `cuda:1` both pass.

Fix: set the device in handshake threads when `device.type == cuda`; compare full
device (type **and** index) in `_read_one_tensor`, failing fast on mismatch.

Simulate: unit test asserting `_read_one_tensor` rejects a destination whose
device index differs (using fake CUDA-like tensors carrying an index).

Verify on hardware: 2-GPU node, per-peer handshake threads pinned to the right
device.

---

## RISK B6 — Control-record publish atomicity

Root cause: `_update_control_record` writes the length first, then the payload.
A racing reader could observe a new length over a partially-written payload.
Currently mitigated only because `Manager` publishes while disallowed.

Fix: publish payload first and length last (length is the commit point), or use
a sequence number / double-buffered control record the reader validates before
and after. Optionally enforce the "must be disallowed during publish" contract
in code.

Simulate: a test that interleaves a publish with a concurrent control-record
read in the fake and asserts the reader never sees a torn record (with the fix,
it either sees the old or the new record, never a mix).

Verify on hardware: covered by the B1/B2 stress test.

---

## RISK B7 — Performance (after correctness)

- Spill uses blocking `cpu.copy_(non_blocking=False)` — no D2H overlap. Use a
  dedicated copy stream + event (ties into B1).
- Per-tensor serial reads on the receiver — pipeline / batch reads.
- Strided/offset views transfer full underlying storage (`untyped_storage`).
  Acceptable for correctness; revisit if checkpoints carry large sliced views.
- Control/manifest reads re-allocate + re-register a CPU buffer every receive —
  cache a per-receiver scratch buffer.

Simulate: the simulator already times each scenario; add a bytes/sec report and
a large-tensor scenario to track regressions.

Verify on hardware: bandwidth benchmark vs `HTTPTransport` on the same model.

---

## TODO B8 — Real GPU+RDMA test coverage

No test currently exercises real `RdmaTransport(device=cuda)`, real
`RdmaMemory(cuda_tensor)`, a cross-host handshake, or GDR. Add a hardware-gated
test (`@skipUnless(torch.cuda.is_available() and _rdma_available())`) and wire
`examples/rdma_transport_sim.py --real --device cuda` into the hardware CI lane.
It should cover: GPU-resident checkpoint, spilled checkpoint, cross-host
handshake, and a timeout/teardown scenario.

---

## Suggested order

1. B1 (GPU stream sync) — correctness, highest risk.
2. B2 (lifetime fence) — correctness, use-after-free.
3. B3 (GDR probe + staged fallback) — required before any GPU run.
4. B8 (real hardware test) — lock in 1–3 on hardware.
5. B4 (cross-host addressing) — required for multi-host.
6. B5, B6 — multi-GPU + atomicity hardening.
7. B7 — performance.

## How to run the simulator

```bash
# Simulated (any machine, CPU): drives the real transport via the faithful fake.
python examples/rdma_transport_sim.py

# On a real RDMA node (torchcomms installed + NIC):
python examples/rdma_transport_sim.py --real --device cuda
```

On a dev box where the torchft Rust extension isn't built, put a stub for
`torchft._torchft` on `PYTHONPATH` (see `examples/`/CI notes) so the pure-Python
checkpointing modules import without the native build.
