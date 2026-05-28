# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end simulator / smoke test for :class:`RDMATransport`.

This drives the **real** ``RDMATransport`` send/receive code path (TCP
symmetric-connect handshake, control record, manifest, per-tensor RDMA reads,
reader-lifetime fence) without requiring RDMA hardware. By default it injects a
*contract-faithful* fake ``torchcomms._transport`` that enforces the same API
shape as the real binding:

  * ``read()`` requires an ``RdmaMemoryMutableView`` (from ``to_mutable_view()``)
    and returns an ``int`` status — passing the immutable ``to_view()`` raises,
    exactly as the real pybind binding does.
  * ``write()`` requires the immutable ``RdmaMemoryView``.

Because ``RDMATransport`` auto-detects torchcomms, the **same** script becomes a
real-hardware test when run with ``--real`` on a node that has torchcomms +
an RDMA NIC:

    # simulated (any machine, CPU only):
    python examples/rdma_transport_sim.py

    # real RDMA hardware (torchcomms installed, NIC present):
    python examples/rdma_transport_sim.py --real --device cuda

Exit code is non-zero if any scenario fails, so it is CI/hardware-gate friendly.
"""

import argparse
import sys
import threading
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import torch


def _install_fake_torchcomms() -> Tuple[object, object]:
    """Insert the contract-faithful fake torchcomms used by the unit tests.

    Reuses the single source of truth in ``rdma_transport_test`` so the
    simulator and the test suite model the torchcomms contract identically.
    Returns (uninstall_token, available_patch) to be undone by the caller.
    """
    from unittest.mock import patch

    from torchft.checkpointing import rdma_transport as mod
    from torchft.checkpointing.rdma_transport_test import (
        _install_torchcomms_mock,
        _MockRdmaTransport,
    )

    _MockRdmaTransport.reset()
    token = _install_torchcomms_mock()
    available_patch = patch.object(mod, "_rdma_available", return_value=True)
    available_patch.start()
    return token, available_patch


def _uninstall_fake_torchcomms(token: object, available_patch: object) -> None:
    from torchft.checkpointing.rdma_transport_test import _uninstall_torchcomms_mock

    available_patch.stop()  # type: ignore[attr-defined]
    _uninstall_torchcomms_mock(token)


def _make_state_dict(device: torch.device) -> Dict[str, object]:
    """A realistic checkpoint mixing dtypes, shapes, strides and non-tensors."""
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
    return {
        "contiguous": torch.randn(16, 8, device=device),
        "fp16": torch.randn(32, device=device, dtype=torch.float16),
        "int64": torch.arange(10, dtype=torch.int64, device=device),
        "transposed": base.t(),  # non-contiguous view
        "sliced": base[:, 1:5],  # offset + strided view
        "scalar": torch.tensor(3.14, device=device),
        "step": 1234,
        "name": "rdma-sim-model",
        "config": {"lr": 0.1, "layers": [3, 4, 5]},
    }


def _assert_state_dict_close(
    got: Dict[str, object], want: Dict[str, object]
) -> None:
    assert set(got.keys()) == set(want.keys()), (
        f"key mismatch: {set(got) ^ set(want)}"
    )
    for k, w in want.items():
        g = got[k]
        if isinstance(w, torch.Tensor):
            assert isinstance(g, torch.Tensor), f"{k}: expected tensor, got {type(g)}"
            torch.testing.assert_close(
                g.cpu(), w.cpu(), msg=lambda m, k=k: f"{k}: {m}"
            )
        else:
            assert g == w, f"{k}: {g!r} != {w!r}"


def _new_transport(
    device: torch.device,
    timeout: timedelta,
    state_dict_fn: Optional[object] = None,
):
    from torchft.checkpointing.rdma_transport import RDMATransport

    return RDMATransport(
        device=device,
        timeout=timeout,
        state_dict=state_dict_fn,
    )


def scenario_basic_roundtrip(device: torch.device) -> None:
    """Single sender, single receiver, full state-dict roundtrip."""
    timeout = timedelta(seconds=30)
    sender = _new_transport(device, timeout)
    state_dict = _make_state_dict(device)
    try:
        sender.send_checkpoint(
            dst_ranks=[1], step=1234, state_dict=state_dict, timeout=timeout
        )
        metadata = sender.metadata()
        assert metadata.startswith("rdma:"), (
            f"expected rdma metadata, got {metadata[:16]!r} (RDMA not active?)"
        )

        receiver = _new_transport(device, timeout)
        try:
            got = receiver.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timeout
            )
            _assert_state_dict_close(got, state_dict)  # type: ignore[arg-type]
        finally:
            receiver.shutdown()
    finally:
        sender.shutdown()


def scenario_concurrent_receivers(device: torch.device, n: int = 4) -> None:
    """Multiple receivers pull the same published checkpoint concurrently."""
    timeout = timedelta(seconds=30)
    sender = _new_transport(device, timeout)
    state_dict = _make_state_dict(device)
    try:
        sender.send_checkpoint(
            dst_ranks=list(range(1, n + 1)),
            step=1234,
            state_dict=state_dict,
            timeout=timeout,
        )
        metadata = sender.metadata()

        errors: List[BaseException] = []
        errors_lock = threading.Lock()

        def pull(rank: int) -> None:
            receiver = _new_transport(device, timeout)
            try:
                got = receiver.recv_checkpoint(
                    src_rank=0, metadata=metadata, step=1234, timeout=timeout
                )
                _assert_state_dict_close(got, state_dict)  # type: ignore[arg-type]
            except BaseException as e:  # noqa: BLE001 - propagate to main
                with errors_lock:
                    errors.append(e)
            finally:
                receiver.shutdown()

        threads = [
            threading.Thread(target=pull, args=(r,)) for r in range(1, n + 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout.total_seconds())

        assert not errors, f"{len(errors)} concurrent receiver(s) failed: {errors[0]!r}"
    finally:
        sender.shutdown()


def _make_contiguous_state_dict(device: torch.device) -> Dict[str, object]:
    """Contiguous-only checkpoint — the realistic zero-copy in-place target.

    The in-place path requires the destination's *storage* size to match the
    source's underlying storage. That only holds for contiguous tensors, which
    is exactly what real model params / optimizer state are.
    """
    return {
        "w0": torch.randn(16, 8, device=device),
        "w1": torch.randn(8, 4, device=device),
        "fp16": torch.randn(32, device=device, dtype=torch.float16),
        "step": 1234,
        "name": "rdma-sim-model",
    }


def scenario_inplace_receive(device: torch.device) -> None:
    """Receiver supplies a preallocated destination dict (zero-copy receive)."""
    timeout = timedelta(seconds=30)
    sender = _new_transport(device, timeout)
    state_dict = _make_contiguous_state_dict(device)
    try:
        sender.send_checkpoint(
            dst_ranks=[1], step=1234, state_dict=state_dict, timeout=timeout
        )
        metadata = sender.metadata()

        # Preallocated destination with matching shapes/dtypes (contiguous).
        def dst_factory() -> Dict[str, object]:
            return {
                k: (torch.empty_like(v) if isinstance(v, torch.Tensor) else v)
                for k, v in _make_contiguous_state_dict(device).items()
            }

        receiver = _new_transport(device, timeout, state_dict_fn=dst_factory)
        try:
            got = receiver.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timeout
            )
            _assert_state_dict_close(got, state_dict)  # type: ignore[arg-type]
        finally:
            receiver.shutdown()
    finally:
        sender.shutdown()


SCENARIOS = [
    ("basic_roundtrip", scenario_basic_roundtrip),
    ("concurrent_receivers", scenario_concurrent_receivers),
    ("inplace_receive", scenario_inplace_receive),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real torchcomms backend (requires torchcomms + RDMA NIC)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for tensors and RDMA (cpu | cuda | cuda:N)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but CUDA is not available")
        return 2

    mode = "REAL torchcomms" if args.real else "SIMULATED (fake torchcomms)"
    print(f"=== RDMATransport simulator: mode={mode} device={device} ===")

    token = patch_obj = None
    if not args.real:
        token, patch_obj = _install_fake_torchcomms()

    failures = 0
    try:
        for name, fn in SCENARIOS:
            t0 = time.monotonic()
            try:
                fn(device)
                dt = time.monotonic() - t0
                print(f"  [PASS] {name:24s} ({dt*1000:.1f} ms)")
            except BaseException as e:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"  [FAIL] {name:24s} {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
    finally:
        if not args.real:
            _uninstall_fake_torchcomms(token, patch_obj)

    total = len(SCENARIOS)
    print(f"=== {total - failures}/{total} scenarios passed ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
