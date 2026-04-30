# GPU/RDMA Readiness Audit for `RDMATransport`

## Scope

Read and audited:

- `torchft/checkpointing/rdma_transport.py`
- `torchft/checkpointing/rdma_transport_test.py`
- `torchft/checkpointing/rdma_manager_integ_test.py`
- `examples/stress_test_cifar10.py`
- `docs/rdma_transport_plan_final.md`
- `torchft/checkpointing/pg_transport.py`
- `torchft/checkpointing/http_transport.py`

Additional context used:

- `torchft/manager.py`
- `torchft/checkpointing/_rwlock.py`
- `torchstore/torchstore/transport/torchcomms/{buffer.py,cache.py}`

I could not read the exact external file requested at `~/Projects/torchft_torchstore_workspace/torchcomms/comms/torchcomms/transport/tests/py/TransportTest.py` because that checkout is not present in this workspace. For torchcomms API assumptions, I used the in-repo `torchstore` torchcomms transport code as the closest available real usage.

## Bottom Line

Current status: **not GPU/RDMA ready**.

The CPU/fallback path is well covered. The real GPU+RDMA path is not. The biggest correctness problems are:

1. GPU snapshot publication is not synchronized with CUDA stream execution.
2. The sender can release snapshot lifetime protection based on a TCP idle timeout even while RDMA reads may still be in flight.
3. The cross-host handshake address advertisement is not robust enough for real clusters.
4. The current `transport.read(...)` usage likely does not match the torchcomms API shape used elsewhere in this workspace.

## Findings

### BLOCKER: GPU snapshot publication is not synchronized before RDMA exposure

- References:
  - `torchft/checkpointing/rdma_transport.py:462-498` in `_build_snapshot`
  - `torchft/checkpointing/rdma_transport.py:318` in `send_checkpoint`
  - `torchft/manager.py:748-761` in the recovery-stream send path
  - `torchft/checkpointing/http_transport.py:225-232` for the CPU-staging path that does synchronize
- What happens:
  - GPU tensors are cloned with `t.clone()` in `_build_snapshot()` (`rdma_transport.py:494`).
  - If the GPU budget is exceeded, `_spill_to_pinned_cpu()` copies GPU bytes into pinned host memory (`rdma_transport.py:492`, `810-820`).
  - `send_checkpoint()` immediately registers those buffers, publishes the control record as `READY`, and releases the writer lock (`rdma_transport.py:314-318`, `456-460`).
  - `Manager` runs this on `self._recovery_stream` (`manager.py:748-761`), but there is no event or stream wait against the stream that last wrote the model tensors.
- Why this is a blocker:
  - CUDA copies and clones are stream-ordered, not globally complete by default.
  - The sender can expose GPU or staged host memory to RDMA before the producing CUDA work is complete.
  - That is a real race on a GPU machine: receivers can read partially produced bytes.
- Evidence from tests:
  - There are no real CUDA tests in `rdma_transport_test.py`.
  - The only “GPU” coverage is fake CPU-backed CUDA-like tensors (`rdma_transport_test.py:1966-2164`), which cannot catch stream-ordering bugs.
- Fix direction:
  - Record an event after snapshot clone / spill work and wait for it before `_update_control_record(..., "READY", ...)`.
  - If `send_checkpoint()` runs on a dedicated recovery stream, it must first wait on the stream(s) that produced the source tensors.
  - Mirror the explicit synchronization discipline already used by `HTTPTransport`.

### BLOCKER: The sender may retire RDMA-visible memory while a real transfer is still running

- References:
  - `torchft/checkpointing/rdma_transport.py:701-734` in `_symmetric_connect`
  - `torchft/checkpointing/rdma_transport.py:749-797` in `_handle_peer_connection`
  - `torchft/checkpointing/rdma_transport.py:405-418` in `disallow_checkpoint`
  - `torchft/checkpointing/rdma_transport.py:441-446` in `shutdown`
  - `torchft/checkpointing/rdma_transport_test.py:1264-1308` in `test_hung_receiver_times_out_after_ready`
- What happens:
  - The sender holds `r_lock` only until it receives TCP `DONE` from the receiver.
  - The sender-side TCP socket uses `self._timeout` and drops the lock on timeout (`rdma_transport.py:754`, `784-797`).
- Why this is a blocker:
  - TCP idle timeout is not proof that RDMA reads are done.
  - A large checkpoint or a slow fabric can legitimately take longer than `self._timeout`.
  - Worse, the Python `transport.read()` calls have no explicit timeout wrapper in this code, so the receiver can still be blocked in RDMA while the sender times out the TCP side channel and releases the lifetime fence.
  - After that, `disallow_checkpoint()` or the next generation rotation can retire `RdmaMemory` while the NIC is still reading it.
- Evidence from tests:
  - The tests explicitly encode that timeout releases the lock (`rdma_transport_test.py:1264-1308`), which is fine for CPU mocks but unsafe as a proxy for “RDMA finished”.
- Fix direction:
  - Do not use TCP socket idle timeout as the lifetime oracle for RDMA buffers.
  - Either keep the generation alive until explicit receiver completion is proven, or add protocol-level progress/keepalive so the sender never times out a healthy long transfer.
  - Add a real timeout story for `transport.read()` itself before relying on `DONE`.

### BLOCKER: Cross-host symmetric-connect bootstrapping is not production-safe

- References:
  - `torchft/checkpointing/rdma_transport.py:227-232` in `_init_rdma`
  - `torchft/checkpointing/rdma_transport.py:278-283` in `metadata`
  - `torchft/checkpointing/rdma_transport.py:707-709` in `_symmetric_connect`
  - `torchft/checkpointing/rdma_transport_test.py:543`, `626-680`, `866-870` where tests only use `socket.gethostname()` or `localhost`
  - `torchft/checkpointing/rdma_manager_integ_test.py:8-15` and `examples/stress_test_cifar10.py:17`, `244`, `279`
- What happens:
  - The sender advertises `handshake_host = socket.gethostname()` (`rdma_transport.py:232`).
  - The TCP side channel listens on an ephemeral port bound to `("::", 0)` (`rdma_transport.py:227-231`).
  - The receiver connects to `(bootstrap.handshake_host, bootstrap.handshake_port)` (`rdma_transport.py:707-709`).
- Why this is a blocker:
  - `socket.gethostname()` is often not routable from other hosts, especially in containerized or multi-NIC clusters.
  - There is no configuration knob for “advertise this specific IP/FQDN”.
  - The code assumes an IPv6 listener and an arbitrary resolver result from `create_connection`; dual-stack behavior is not guaranteed.
  - The port is random and there is no story for firewalls or security groups.
- Evidence from tests:
  - The integration tests are explicitly fallback-only (`rdma_manager_integ_test.py:8-15`).
  - The stress example forces `device=torch.device("cpu")` and notes HTTP fallback on this machine (`stress_test_cifar10.py:17`, `244`, `279`).
  - No test exercises a real cross-host handshake.
- Fix direction:
  - Add an explicit advertised handshake address config.
  - Prefer using a known routable manager/listener address instead of `gethostname()`.
  - Make the listener family configurable or dual-stack-safe.
  - Document and test required port reachability across hosts.

### BLOCKER: RDMA read API usage likely does not match the torchcomms contract used elsewhere

- References:
  - `torchft/checkpointing/rdma_transport.py:592`, `620`, `678`
  - `torchstore/torchstore/transport/torchcomms/buffer.py:235`
  - `torchstore/torchstore/transport/torchcomms/buffer.py:288`
- What happens:
  - This transport calls `transport.read(local_mem.to_view(), remote_buffer)` for control, manifest, and tensor reads.
  - In the in-repo torchstore torchcomms integration, reads use `to_mutable_view()` and writes use `to_view()`.
- Why this is a blocker:
  - If the real API distinguishes immutable and mutable views, `rdma_transport.py` is using the wrong one for every read.
  - The current RDMA tests cannot catch this because `_MockRdmaMemory` only exposes `to_view()` and does not model mutability requirements.
  - `rdma_transport.py` also ignores the return value from `read()`, while `torchstore` checks for non-zero error codes on both `read()` and `write()`.
- Fix direction:
  - Verify the real torchcomms API and switch reads to `to_mutable_view()` if required.
  - Handle and surface non-zero return codes or exceptions consistently.
  - Update the RDMA mocks to model the real API shape so this cannot regress silently.

### BLOCKER: CUDA/GDR support is never probed, only “NIC exists”

- References:
  - `torchft/checkpointing/rdma_transport.py:64-73` in `_rdma_available`
  - `torchft/checkpointing/rdma_transport.py:216-218` in `_init_rdma`
  - `torchft/checkpointing/rdma_transport.py:485-498` in `_build_snapshot`
  - `torchft/checkpointing/rdma_transport.py:665-678` in `_read_one_tensor`
  - `torchft/checkpointing/rdma_transport_test.py:1966-2164`
- What happens:
  - `_rdma_available()` only checks `RdmaTransport.supported()`.
  - The transport then assumes `RdmaMemory(cuda_tensor)` and `RdmaTransport(device=cuda)` are valid for the configured device.
- Why this is a blocker:
  - “InfiniBand NIC present” is not the same as “GPUDirect RDMA path is usable for this GPU/NIC/driver/runtime combination”.
  - If GDR registration is unsupported or partially supported, `RdmaTransport(device)` or `RdmaMemory(tensor)` may fail only after deployment.
  - There is no fallback from “RDMA available on CPU” to “stage GPU to pinned CPU because GPU registration is unavailable”.
- Evidence from tests:
  - The only GPU-path tests use `_FakeCudaTensor` and patched spill helpers (`rdma_transport_test.py:1966-2164`).
  - There is zero real CUDA registration, zero real `RdmaMemory(cuda_tensor)`, and zero real GDR coverage.
- Fix direction:
  - Add an explicit GPU-RDMA capability probe at startup.
  - If GPU registration fails, either refuse `device=cuda` or automatically stage tensors through pinned CPU.
  - Add a real CUDA+RDMA integration test before calling this production-ready.

### RISK: Handshake threads may touch the wrong CUDA device/context

- References:
  - `torchft/checkpointing/rdma_transport.py:749-765` in `_handle_peer_connection`
- What happens:
  - Each accepted handshake starts a fresh Python thread that constructs `_RdmaTransport(self._device)`.
  - There is no per-thread `torch.cuda.set_device(...)`.
- Why this is a risk:
  - CUDA current device is thread-local.
  - If torchcomms relies on the current CUDA context instead of only the passed `torch.device`, this can break on multi-GPU nodes or initialize work on the wrong device.
- Fix direction:
  - If `self._device.type == "cuda"`, set the device explicitly in handshake threads before constructing torchcomms objects.
  - Add a multi-GPU test that verifies per-peer handshake threads use the intended device.

### RISK: In-place receive only checks device type, not device index

- References:
  - `torchft/checkpointing/rdma_transport.py:664-669` in `_read_one_tensor`
- What happens:
  - The in-place path accepts any destination where `inplace.device.type == self._device.type`.
- Why this is a risk:
  - `cuda:0` and `cuda:1` both pass that check.
  - On a multi-GPU node, the transport can write into a tensor on the wrong GPU while the transport itself was constructed for a different device.
- Fix direction:
  - Compare full devices, not only `.type`.
  - Fail fast if the callback returns a tensor on a different CUDA device.

### RISK: `_cast_tensor()` is a storage reinterpretation trick with no real CUDA coverage

- References:
  - `torchft/checkpointing/pg_transport.py:149-164` in `_cast_tensor`
  - `torchft/checkpointing/pg_transport.py:106-146` in `_prepare_state_dict`
  - `torchft/checkpointing/rdma_transport.py:308`, `669`
  - `torchft/checkpointing/rdma_transport_test.py:1419-1852`, `1966-2164`
- What happens:
  - `_cast_tensor()` builds a new tensor directly from `tensor.untyped_storage()` and asserts storage identity.
  - RDMA send and in-place receive both rely on that behavior.
- Why this is a risk:
  - It is an internal-storage reinterpretation path, not a normal tensor copy.
  - It is covered for CPU and for fake-CUDA tests, but not with real CUDA storage.
  - If PyTorch changes the semantics of `torch.tensor(storage, device=...)` on CUDA, both PG and RDMA receive paths can fail.
- Fix direction:
  - Add real CUDA tests for `_prepare_state_dict()` and `_cast_tensor()`.
  - Consider isolating this dependency behind a smaller helper with explicit failure handling.

### RISK: Control-record publication is only conditionally atomic

- References:
  - `torchft/checkpointing/rdma_transport.py:546-579` in `_update_control_record`
  - `torchft/checkpointing/rdma_transport.py:314-318` in `send_checkpoint`
  - `docs/rdma_transport_plan_final.md:876`
- What happens:
  - The writer stores `<length><payload>` into the pinned control buffer.
  - It writes the length first, then copies the payload bytes.
- Why this is a risk:
  - A reader that races this update can observe a new length with partially written payload bytes.
  - The current `Manager` lifecycle usually calls `send_checkpoint()` while the transport is disallowed, which reduces exposure, but the transport itself does not enforce that precondition.
- Fix direction:
  - Write payload first and publish length last, or use a versioned/double-buffered control record.
  - If “must be disallowed during publish” is the contract, enforce it in code.

### RISK: GPU spill path is correctness-friendly but leaves overlap on the floor

- References:
  - `torchft/checkpointing/rdma_transport.py:485-492`, `810-820`
  - `torchft/checkpointing/http_transport.py:225-232`, `269-279`
  - `docs/rdma_transport_plan_final.md:168`
- What happens:
  - `_spill_to_pinned_cpu()` allocates pinned CPU memory and does a blocking `cpu.copy_(..., non_blocking=False)`.
  - `HTTPTransport` at least has a dedicated CUDA stream and then synchronizes it.
- Why this is a risk:
  - The spill behavior is probably correct once the stream-ordering bug above is fixed, but it gives up the async D2H overlap the plan promised.
  - On large models this can erase much of the hoped-for RDMA advantage.
- Fix direction:
  - Use a dedicated copy stream plus event synchronization.
  - If the sender must spill many tensors, pipeline clone / D2H / registration instead of serial blocking copies.

### RISK: Non-contiguous and offset views transfer full underlying storage

- References:
  - `torchft/checkpointing/pg_transport.py:99-101` in `_prepare_tensor`
  - `torchft/checkpointing/rdma_transport.py:680-684` in `_read_one_tensor`
  - `torchft/checkpointing/rdma_transport_test.py:1670-1852`
- What happens:
  - The transfer size is `tensor.untyped_storage().nbytes()`, not logical tensor bytes.
  - Reconstruction uses `torch.as_strided(...)` to recover shape/stride/offset.
- Assessment:
  - Correctness looks good. The CPU tests for transposed and offset views are solid.
  - On GPU, though, this can register and transfer a much larger buffer than the logical tensor requires.
- Impact:
  - This is mostly a performance risk, not a correctness bug.
- Fix direction:
  - Track this as a performance TODO unless checkpoints regularly contain large sliced views.

### TODO: There is no real GPU+RDMA test coverage in this repo

- References:
  - `torchft/checkpointing/rdma_transport_test.py:1966-2164`
  - `torchft/checkpointing/rdma_manager_integ_test.py:8-15`
  - `examples/stress_test_cifar10.py:17`, `244`, `279`
  - `docs/rdma_transport_plan_final.md:407-412`, `837-838`
- Current state:
  - No `@skipUnless(torch.cuda.is_available() and _rdma_available(), ...)` tests exist in the actual checked-in RDMA transport tests.
  - The manager integration tests are explicitly fallback-only.
  - The stress test runs on CPU on this machine.
  - The plan doc calls out real CUDA/RDMA testing as future work.
- Needed:
  - At least one real 2-host or 2-process GPU+RDMA test that exercises:
    - `RdmaTransport(device=cuda)`
    - `RdmaMemory(cuda_tensor)`
    - the symmetric handshake cross-host
    - one checkpoint that stays on GPU
    - one checkpoint that spills to pinned CPU
    - a timeout / teardown scenario

### TODO: Add explicit error handling around malformed control/manifest state

- References:
  - `torchft/checkpointing/rdma_transport.py:596-608`
  - `torchft/checkpointing/rdma_transport.py:617-623`
  - `torchft/checkpointing/rdma_transport.py:676-678`
- Current state:
  - The happy-path structure is fine.
  - But malformed `manifest_remote_buffer`, `manifest_nbytes`, or partial reads rely on whatever exception comes back from `pickle.loads()` or `transport.read()`.
- Needed:
  - Validate `manifest_remote_buffer is not None` before read.
  - Surface read failures with consistent context including `step`, tensor path, and peer metadata.
  - If torchcomms returns status codes, check them.

### TODO: Add an advertised-handshake-address config and document networking requirements

- References:
  - `torchft/checkpointing/rdma_transport.py:227-232`, `278-283`, `707-709`
- Needed:
  - Configurable advertised host/IP.
  - Documentation for firewall and port reachability.
  - Dual-stack or IPv4-only fallback if IPv6 listener setup is unavailable.

### OK: The control-record -> manifest -> tensor read sequence is logically sound

- References:
  - `torchft/checkpointing/rdma_transport.py:378-403` in `recv_checkpoint`
  - `torchft/checkpointing/rdma_transport.py:581-608` in `_read_control_record`
  - `torchft/checkpointing/rdma_transport.py:610-623` in `_read_manifest`
  - `torchft/checkpointing/rdma_transport.py:625-685` in `_read_tensors` / `_read_one_tensor`
- Assessment:
  - The receive order is correct.
  - The code validates protocol version, `READY` state, and `step` both in the control record and manifest.
  - Non-tensor leaves, tensor leaves, and DTensor leaves are reconstructed consistently.
- Caveat:
  - This “OK” is about logical sequencing only. It does not remove the stream-ordering, timeout, or API-contract blockers above.

### OK: The two-generation model is structurally correct for retirement and shutdown

- References:
  - `torchft/checkpointing/rdma_transport.py:314-316`, `441-446`
  - `torchft/checkpointing/rdma_transport_test.py:1061-1201`
- Assessment:
  - Keeping `current` and `previous` snapshots alive is the right shape.
  - The retirement and shutdown tests prove that dropped generations stop resolving in the mock registry.
- Caveat:
  - This only remains correct if the lifetime fence is tied to real RDMA completion. The timeout blocker above currently breaks that assumption.

### OK: CPU-side strided / DTensor reconstruction matches the PG transport design

- References:
  - `torchft/checkpointing/pg_transport.py:106-146`, `149-164`, `285-289`
  - `torchft/checkpointing/rdma_transport.py:503-514`, `680-684`
  - `torchft/checkpointing/rdma_transport_test.py:1556-1852`
- Assessment:
  - Reuse of PG transport metadata helpers is internally consistent.
  - CPU tests cover DTensor round-trip, in-place DTensor receive, transposed views, slices, and offset views.
- Caveat:
  - The missing real CUDA coverage still matters for the GPU case.

## Direct Answers to the Requested Questions

### 1. RDMA path gaps

- `RdmaTransport(device=cuda)`: **BLOCKER/RISK**. The code assumes this works, but never probes GPU/GDR capability and never tests it on real hardware.
- `RdmaMemory(tensor)` with GPU tensors: **BLOCKER/RISK**. The design depends on it for both send and receive, but the repo has no real coverage. If GDR is unavailable, there is no fallback except the sender-side spill path.
- Symmetric connect handshake cross-host: **BLOCKER**. `socket.gethostname()` plus an ephemeral port is not a robust cluster advertisement scheme.
- Control record / manifest / tensor read sequence: **OK** logically, but subject to the control-record atomicity risk and the timeout lifetime blocker.
- Missing error handling: **TODO/RISK**. Read return codes are ignored, malformed manifest state is not validated early, and failures lack peer/tensor context.

### 2. GPU tensor handling

- `_build_snapshot()` cloning: **BLOCKER**. No stream synchronization before exposing cloned GPU memory.
- `max_gpu_snapshot_bytes` spill logic: **RISK** for performance, **probably correct** for bytes copied once stream ordering is fixed. It uses blocking D2H and does not overlap work.
- `_prepare_state_dict()` / `_cast_tensor()` on GPU: **RISK**. The storage reinterpretation is relied on heavily and not tested on real CUDA.
- `torch.as_strided()` reconstruction on GPU: **OK/RISK**. The reconstruction logic is sound; the real uncertainty is whether the underlying GPU buffer contents are complete and on the correct device.

### 3. Memory registration

- GPU registration requirements: **BLOCKER/RISK**. Not probed, not tested, no fallback.
- Two-generation lifetime model: **OK** structurally, but invalidated by the timeout-based early lock release.
- Registered-memory leaks: **RISK** is moderate, not dominant. Snapshot memory is reference-owned and dropped on rotation/shutdown, but only CPython refcounting and object destruction are relied on.

### 4. Concurrency on GPU

- RWLock fence vs async CUDA completion: **BLOCKER**. The lock only protects Python object lifetime, not GPU execution completion.
- Handshake server threads and CUDA context: **RISK**. No per-thread device setup before constructing GPU transports.

### 5. Cross-host networking

- `handshake_host = socket.gethostname()`: **BLOCKER** for production readiness.
- RDMA bind addresses routable across hosts: **RISK**. The code assumes the exchanged `bind()` addresses are usable cross-host but does not validate that beyond localhost mocks.
- Firewall / port issues: **TODO/BLOCKER** depending on environment. No config or docs exist.

### 6. Performance concerns

- Biggest bottlenecks:
  - blocking spill-to-CPU copies
  - per-tensor serial reads
  - full-storage transfer for strided views
  - per-read registration of control/manifest/temp buffers
- Unnecessary CPU copies:
  - sender-side spill is required as fallback, but control/manifest reads still do fresh CPU allocations every receive
  - there is no “GPU registration unavailable -> staged receive” path
- Missing overlap:
  - yes, especially clone/D2H/registration overlap on send

### 7. torchcomms API assumptions

- Direct read of the requested external `TransportTest.py`: **not possible in this workspace**.
- Best available evidence:
  - In-repo torchstore torchcomms code uses `connect()` symmetrically on both sides, which matches this transport's handshake design.
  - In-repo torchstore torchcomms code uses `read(to_mutable_view(), remote)` and `write(to_view(), remote)` and checks return codes, which this transport does not.
- Conclusion:
  - `bind()` / `connect()` usage looks directionally consistent.
  - `read()` usage is likely wrong or at least under-validated.

## Recommended Fix Order

1. Fix GPU stream synchronization before snapshot publish.
2. Fix lifetime fencing so timeout cannot retire memory while RDMA is still active.
3. Verify the real torchcomms read API and update `to_view()` / return-code handling.
4. Add explicit advertised-handshake-address configuration and cross-host tests.
5. Add a real CUDA+RDMA capability probe and a real hardware test.
6. Then optimize spill overlap and per-tensor registration overhead.
