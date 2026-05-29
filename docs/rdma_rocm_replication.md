# torchft + torchcomms RDMA on ROCm — setup & scaling-verification guide

## Purpose

We have validated the torchft `RDMATransport` (torchcomms RDMA / GPUDirect) for
fault-tolerant checkpoint recovery on a **single node with one usable RDMA NIC**.
On that box the four NICs are rail-isolated (cross-NIC RDMA does not route), so
every GPU had to funnel through one NIC and concurrent heals serialized on it.

**This guide is for setting up and verifying the same stack on a machine whose
fabric *does* support cross-NIC (and ideally multi-node) RDMA**, then scaling the
test up in tiers to confirm the recovery-time results hold — and improve — when
more NICs are available:

```
Tier 1  multi-GPU, single NIC      (reproduce our baseline)
Tier 2  multi-GPU, multi-NIC       (the new capability to verify)
Tier 3  multi-node                 (the eventual target)
```

Each tier has a **hardware sanity gate** you must pass before running the
benchmark at that tier. Don't skip the gates — they isolate fabric problems from
software problems.

---

## Part A — One-time environment setup

### A.0 Hardware / OS prerequisites

Reference (the validated box; yours may differ — the env vars in A.3 are the
part most likely to need per-box tuning):

| Component | Reference value |
|-----------|-----------------|
| GPUs | 8× AMD Instinct **MI300X** (`gfx942`), 192 GB each |
| ROCm | 6.4.2 (`/opt/rocm`) |
| torch | `2.13.0.dev*+rocm7.2` (auto-imports torchcomms via `torch.distributed`) |
| RDMA NIC | Mellanox `mlx5_*`, RoCE v2, 200 Gb/s, port `ACTIVE` |
| RoCE GID | working RoCE v2 GID at **index 3** (`NCCL_IB_GID_INDEX=3`) |

> For Tier 2/3 you want a fabric that routes RDMA **between** NICs and **between
> nodes** (all-to-all or rail-optimized-with-routing). The common
> 1-NIC-per-GPU reference design (e.g. 8 GPUs + 8 NICs) is ideal.

### A.1 Repos and branches

```
torchcomms   branch: rocm-transport-support     # ROCm transport build fixes
torchft      branch: feature/rdma_transport_rocm # RDMATransport ROCm fixes + benchmark
```

The `rocm-transport-support` torchcomms branch carries the ROCm build fixes
(transport-layer enablement, `getCuMemDmaBufFd` via HIP, `HSA_STATUS_SUCCESS` +
cuda→hip compat macros, header-only fmt linkage). Keep a backup branch before
rebasing torchcomms onto upstream `main`.

### A.2 Python environment & packages

Use **`pip`**, not `uv`, for the PyTorch wheels (`uv` 0.10 has a zip64 bug that
corrupts large PyTorch wheels).

```bash
python -m venv .venv
.venv/bin/pip install torch==2.13.0.dev*+rocm7.2 torchvision==0.28.0.dev*+rocm7.2
.venv/bin/pip install protoc-wheel-0     # self-contained protoc for torchft's Rust build
```

`torchcomms` links **folly** from a conda env (`comms_vllm` on the reference
box). Point `CONDA_PREFIX` at an env that provides folly + a `libstdc++` new
enough for `GLIBCXX_3.4.30` (that conda `libstdc++` is why `LD_LIBRARY_PATH`
below matters).

### A.3 Runtime environment variables

Needed for **any** RDMA use. Because this torch auto-imports torchcomms inside
`torch.distributed`, even `import torch` needs `LD_LIBRARY_PATH` once torchcomms
is installed. These are the **Tier-1 (single-NIC) baseline** values; Tier 2/3
override the NIC-selection ones (Part C).

```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/comms_vllm/lib:/opt/rocm/lib
export NCCL_IB_GID_INDEX=3              # RoCE v2 routable GID (default => supported()=False)
export NCCL_IB_HCA=mlx5_0               # Tier 1: pin to one NIC
export NCCL_CTRAN_IB_DEVICE_STRIDE=0    # Tier 1: every GPU -> the one NIC
export TORCHFT_RDMA_HANDSHAKE_HOST=::1  # single-node loopback for the TCP handshake
export NCCL_DEBUG=ERROR GLOG_minloglevel=2   # optional: quieter logs
```

- `NCCL_IB_GID_INDEX`: without a valid RoCE v2 GID index,
  `RdmaTransport.supported()` is False and the transport silently falls back to
  HTTP. Find it with
  `cat /sys/class/infiniband/<nic>/ports/1/gid_attrs/types/3` → `RoCE v2`.
- NIC-selection formula (torchcomms `CtranIb.cc`):
  `nic_index = cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE`
  indexing the `NCCL_IB_HCA`-filtered device list. `STRIDE=0` ⇒ every GPU → the
  first NIC. See Part C to spread across NICs.

### A.4 Build torchcomms (transport-only, ROCm)

```bash
cd torchcomms && git checkout rocm-transport-support
( cd comms/utils/cvars && NCCL_CVARS_OUTPUT_DIR=$PWD python extractcvars.py )

LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
CONDA_PREFIX=/path/to/conda/envs/comms_vllm \
USE_NCCL=OFF USE_NCCLX=OFF USE_GLOO=OFF USE_RCCL=OFF USE_RCCLX=OFF \
USE_XCCL=OFF USE_TRANSPORT=ON \
  ../.venv/bin/python -m pip install --no-build-isolation -v -e .
```

**If GitHub is proxy-blocked** (FetchContent deps fmt/glog/gflags/nlohmann_json
fail to clone): pre-clone them and redirect via global git `insteadOf`
(`git config --global url./local/path.insteadOf https://github.com/...`); remove
later with `git config --global --remove-section url.<path>`.

### A.5 Build torchft

```bash
cd torchft && git checkout feature/rdma_transport_rocm
PROTOC=$PWD/../.venv/bin/protoc \
PROTOC_INCLUDE=$PWD/../.venv/lib/python3.*/site-packages/protoc/data/include \
  ../.venv/bin/pip install --no-build-isolation -e '.[dev]'
```

---

## Part B — Verify the software stack (do this once, single-NIC)

### B.1 Real-hardware transport simulator
Drives the full `RDMATransport` send/recv path (handshake, control record,
manifest, per-tensor RDMA reads, reader fence) against the real backend:

```bash
cd torchft
env LD_LIBRARY_PATH=$LD_LIBRARY_PATH NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=mlx5_0 \
    NCCL_CTRAN_IB_DEVICE_STRIDE=0 TORCHFT_RDMA_HANDSHAKE_HOST=::1 \
  ../.venv/bin/python -u examples/rdma_transport_sim.py --real --device cuda
# PASS: "=== 3/3 scenarios passed ===", exit 0
```

### B.2 Unit tests (mock-backed, CPU)
```bash
env LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  ../.venv/bin/python -m pytest torchft/checkpointing/rdma_transport_test.py -q
# PASS: all passed (handshake tests are slow, ~10 min total).
```

If B.1/B.2 fail, fix the software/env before touching the fabric tiers.

---

## Part C — Tiered scaling verification

The benchmark is `examples/stress_test_cifar10.py`: 1 Lighthouse + N replicas
(one GPU each), a chaos monkey kills a random worker every `[chaos-min,
chaos-max]` cluster steps, a supervisor respawns it, and it heals from a peer
over the chosen transport. The SUMMARY prints `rdma active`, successful heals,
and recovery-time stats (mean/median/min/max + per-heal samples). `EVT_READY`
asserts RDMA is actually active and errors loudly if it silently fell back to
HTTP.

Use the large model (`--model llama_7b`, ~27 GB bf16 checkpoint) for the
transfer-bound comparison — small models are dominated by quorum overhead and
won't show the NIC difference. `llama_1b` (~3.8 GB) is a faster smoke variant.

---

### Tier 1 — multi-GPU, single NIC (reproduce the baseline)

**Hardware gate** — single-NIC RDMA loopback works (Part B passed). Confirm a
single heal ≈saturates one NIC:

```bash
# 2-GPU single heal, large model -> read the "rdma: read tensors took ..." line.
env $TIER1_ENV ../.venv/bin/python -u examples/stress_test_cifar10.py \
  --device cuda --transport rdma --model llama_7b \
  --num-replicas 2 --total-steps 20 --no-chaos \
  --batch-size 4 --seq-len 256 --grad-sync-every 0 \
  --max-gpu-snapshot-gb 50 --eval-every 1000 \
  --lighthouse-min-replicas 2 --manager-min-replica-size 2
```
where `TIER1_ENV` = the A.3 baseline (`NCCL_IB_HCA=mlx5_0
NCCL_CTRAN_IB_DEVICE_STRIDE=0`). Expect `bulk transfer ≈ NIC line rate`
(reference box: ~1.5 s for ~13.5 GB ≈ 9 GB/s startup heal; ~1.0–1.3 s for the
full 27 GB mid-training ≈ 21–27 GB/s on a 200 Gb/s NIC).

**Benchmark — RDMA vs HTTP, 8-GPU chaos (the baseline numbers):**
```bash
for T in rdma http; do
env $TIER1_ENV ../.venv/bin/python -u examples/stress_test_cifar10.py \
  --device cuda --transport $T --model llama_7b \
  --num-replicas 8 --total-steps 150 --chaos-min-steps 15 --chaos-max-steps 30 \
  --batch-size 4 --seq-len 256 --grad-sync-every 0 --max-gpu-snapshot-gb 50 \
  --eval-every 1000 --lighthouse-min-replicas 2 --manager-min-replica-size 2
done
```
**Reference result (single NIC):** RDMA heals are fast solo (~1–1.3 s transfer)
but go **bimodal** under concurrent heals (~11–12 s) because they serialize on
the one NIC; HTTP heals are ~25 s each and throttle the cluster. Both complete
with 0 CUDA/recovery errors. This is the bottleneck Tier 2 should remove.

---

### Tier 2 — multi-GPU, multi-NIC (the capability to verify)

**Hardware gate (DO THIS FIRST): confirm cross-NIC RDMA actually routes.**
On the reference box this *fails* (rail isolation); on your target it must pass.
Save as `xnic_check.py`:

```python
import torch
from torchcomms._transport import RdmaMemory, RdmaTransport
assert RdmaTransport.supported()
# With STRIDE=1 + all NICs visible: cuda:0 -> NIC0, cuda:1 -> NIC1
t0 = RdmaTransport(torch.device("cuda:0"))
t1 = RdmaTransport(torch.device("cuda:1"))
u0, u1 = t0.bind(), t1.bind()
assert t0.connect(u1) == 0 and t1.connect(u0) == 0
src = torch.arange(1024, dtype=torch.uint8, device="cuda:0")
dst = torch.zeros(1024, dtype=torch.uint8, device="cuda:1")
m0, m1 = RdmaMemory(src), RdmaMemory(dst)
rc = t1.read(m1.to_mutable_view(), m0.to_remote_buffer())
torch.cuda.synchronize("cuda:1")
print("read rc:", rc, "data_ok:", bool(torch.equal(src.cpu(), dst.cpu())))
import os; os._exit(0)
```
```bash
env LD_LIBRARY_PATH=$LD_LIBRARY_PATH NCCL_IB_GID_INDEX=3 \
    NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3" \
    NCCL_CTRAN_IB_DEVICE_STRIDE=1 NCCL_CTRAN_IB_DEVICES_PER_RANK=1 \
  ../.venv/bin/python -u xnic_check.py
# PASS: "read rc: 0 data_ok: True"
# FAIL (rail-isolated like the reference box): "transport retry counter exceeded"
#   -> cross-NIC does not route; Tier 2 is not possible on this fabric. Stop here.
```

**Spread GPUs across NICs.** Set the HCA list to all NICs and `STRIDE=1` so
GPU *i* → NIC *i*:
```bash
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"  # all your NICs
export NCCL_CTRAN_IB_DEVICE_STRIDE=1
export NCCL_CTRAN_IB_DEVICES_PER_RANK=1
```
> Caveat: the selection formula `cudaDev * DEVICES_PER_RANK * STRIDE` does **not**
> wrap, and torchcomms throws if it indexes past the NIC list. So `STRIDE=1`
> requires **#GPUs ≤ #NICs** (e.g. 8 GPUs + 8 NICs is fine; 8 GPUs + 4 NICs is
> not — GPUs 4–7 would exceed the list). If you have fewer NICs than GPUs and
> cross-NIC routes, pin each replica to `mlx5_{i % num_nics}` per process
> instead (set `NCCL_IB_HCA` before the worker constructs the transport).

**Benchmark — same as Tier 1 but with the multi-NIC env.** Re-run the 8-GPU
chaos RDMA-vs-HTTP commands above with the Tier-2 env.

**What to verify:**
- All replicas still report `rdma active=True`; **0 CUDA/recovery errors**.
- The RDMA recovery-time distribution is **no longer bimodal** — concurrent
  heals run on different NICs in parallel instead of serializing, so the slow
  (~11–12 s on reference) cluster collapses toward the solo ~1 s transfer.
- RDMA's advantage over HTTP **widens** vs Tier 1 (HTTP can't parallelize the
  GPU→CPU→TCP path the same way).
- Optional: confirm aggregate NIC utilization with a fabric counter tool while
  several replicas heal at once.

---

### Tier 3 — multi-node

**Hardware gate: cross-node RDMA routes.** Run `xnic_check.py` style across two
hosts (one binds + advertises its URL over TCP, the other connects + reads), or
use your cluster's standard RoCE reachability check (e.g. `ib_read_bw` /
`perftest` between the two nodes' NICs). Both directions must pass at line rate.

**Configuration changes vs Tier 1/2:**
- `TORCHFT_RDMA_HANDSHAKE_HOST` must be each node's **routable** IP/FQDN (not
  `::1`) so a healing replica on node B can reach the checkpoint source on
  node A's TCP handshake server.
- Run the **Lighthouse on one node** and launch workers on every node pointed at
  that lighthouse address; place replicas across nodes (e.g. node A: replicas
  0–7 on its 8 GPUs, node B: replicas 8–15).
- Keep the per-GPU NIC mapping from Tier 2 so each replica uses its local NIC.

> Note: `stress_test_cifar10.py` as written is **single-host** — it spawns all N
> replicas locally and runs the Lighthouse + chaos monkey in-process. For
> Tier 3 it needs a small launcher change: start the Lighthouse standalone, and
> run the worker spawner per node against the shared lighthouse address (the
> `worker_main`/`spawn_worker` plumbing already takes `lighthouse_address` and a
> `--device cuda` per-replica GPU index; what's missing is a multi-host launcher
> and a global replica-id offset per node). Add that launcher before running
> Tier 3.

**What to verify:** heals succeed across hosts (`rdma active=True`, 0 errors),
recovery time tracks the cross-node NIC bandwidth, and the chaos cluster stays
productive (no HTTP-style throttling) as node/replica count grows.

---

## Reference / troubleshooting

- **Falls back to HTTP** (`rdma active=False`): `NCCL_IB_GID_INDEX` wrong/missing,
  or `RdmaTransport.supported()` False — check the RoCE v2 GID index and that the
  NIC port is `ACTIVE`.
- **`transport retry counter exceeded`**: RDMA packets aren't being delivered —
  cross-NIC/cross-node has no route, or PFC/ECN/traffic-class misconfig on the
  fabric. This is a fabric problem, not a torchft/torchcomms one.
- **`hipErrorInvalidValue` / `CUDA error: invalid argument`**: should be handled
  by `RDMATransport._restore_cuda_device` (ROCm device-perturbation workaround).
  If it resurfaces, a new RDMA↔CUDA boundary may be missing a restore call.
- **`uv` can't install the PyTorch wheels**: zip64 bug — use `pip`.
- Runtime env vars are mandatory at every tier; consider wiring them into
  `.venv/bin/activate`.
