#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RAID5 FSDP Orchestrator
========================

Orchestrates a full RAID5 FSDP fault tolerance demo:

  1. Starts a lighthouse server and N workers
  2. Workers train for a few steps
  3. Kills one worker (simulating GPU failure)
  4. Remaining workers detect the failure and exit
  5. Restarts training with N-1 workers
  6. After a few more steps, restarts with N workers (adding one back)

Usage:
    python run_raid5_fsdp_demo.py [--num_workers 4] [--steps_before_kill 10]
                                   [--steps_after_kill 10] [--steps_after_rejoin 10]

Requires: torchvision, torchdata (install via: uv pip install torchvision torchdata)
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional


def find_free_port() -> int:
    """Find a free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class OutputMonitor:
    """Thread-safe stdout monitor for worker processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._max_step: Dict[int, int] = {}  # rank -> max step seen
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._step_event = threading.Event()
        self._step_pattern = re.compile(r"\[Rank \d+, step (\d+)\]")
        # Also match quorum commit messages: "step N] should_commit=True"
        self._commit_pattern = re.compile(r"step (\d+)\] should_commit=True")

    @property
    def max_step_seen(self) -> int:
        with self._lock:
            return max(self._max_step.values()) if self._max_step else -1

    def start_monitoring(self, rank: int, proc: subprocess.Popen) -> None:
        """Start a background thread to read and print a worker's stdout."""
        t = threading.Thread(
            target=self._reader_thread,
            args=(rank, proc),
            daemon=True,
        )
        t.start()
        self._threads.append(t)

    def _reader_thread(self, rank: int, proc: subprocess.Popen) -> None:
        """Read lines from a worker's stdout and track step progress."""
        assert proc.stdout is not None
        for line in proc.stdout:
            if self._stop.is_set():
                break
            line = line.rstrip()
            if line:
                print(f"  [worker {rank}] {line}", flush=True)
                match = self._step_pattern.search(line)
                if not match:
                    match = self._commit_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    with self._lock:
                        prev = self._max_step.get(rank, -1)
                        if step > prev:
                            self._max_step[rank] = step
                    self._step_event.set()

    def wait_for_step(self, target_step: int, timeout: float = 120.0) -> bool:
        """Wait until any worker reports reaching target_step."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.max_step_seen >= target_step:
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._step_event.wait(timeout=min(remaining, 1.0))
            self._step_event.clear()
        return self.max_step_seen >= target_step

    def stop(self) -> None:
        """Signal reader threads to stop."""
        self._stop.set()

    def reset(self) -> None:
        """Reset step tracking for a new batch of workers."""
        self.stop()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._stop.clear()
        self._step_event.clear()
        with self._lock:
            self._max_step.clear()


def start_lighthouse(port: int) -> subprocess.Popen:
    """Start the lighthouse server."""
    cmd = [
        "torchft_lighthouse",
        f"--bind=[::]:{ port}",
        "--min_replicas=1",
        "--quorum_tick_ms=100",
        "--join_timeout_ms=10000",
    ]
    print(f"[orchestrator] Starting lighthouse on port {port}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "RUST_BACKTRACE": "1"},
    )
    # Give it a moment to bind
    time.sleep(1)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Lighthouse failed to start:\n{out}")
    print(f"[orchestrator] Lighthouse running (pid={proc.pid})")
    return proc


def start_workers(
    world_size: int,
    master_port: int,
    lighthouse_port: int,
    gpu_ids: List[int],
    monitor: OutputMonitor,
    num_parity: int = 1,
) -> Dict[int, subprocess.Popen]:
    """Start worker processes and begin monitoring their output."""
    workers: Dict[int, subprocess.Popen] = {}
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_raid5_fsdp.py")

    for rank in range(world_size):
        env = {
            **os.environ,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "CUDA_VISIBLE_DEVICES": str(gpu_ids[rank]),
            "TORCHFT_LIGHTHOUSE": f"http://localhost:{lighthouse_port}",
            "NUM_PARITY": str(num_parity),
            # Reduce NCCL timeout for faster failure detection in the demo
            "TORCH_NCCL_NONBLOCKING_TIMEOUT": "30",
            "NCCL_TIMEOUT": "30",
        }
        cmd = [sys.executable, script]
        print(f"[orchestrator] Starting worker rank={rank} (GPU {gpu_ids[rank]})")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        workers[rank] = proc
        monitor.start_monitoring(rank, proc)

    return workers


def kill_worker(workers: Dict[int, subprocess.Popen], rank: int) -> None:
    """Kill a specific worker process."""
    proc = workers[rank]
    if proc.poll() is None:
        print(f"[orchestrator] Killing worker rank={rank} (pid={proc.pid})")
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        print(f"[orchestrator] Worker rank={rank} killed")
    else:
        print(f"[orchestrator] Worker rank={rank} already exited (rc={proc.returncode})")


def wait_for_workers_exit(workers: Dict[int, subprocess.Popen], timeout: float = 60.0) -> None:
    """Wait for all workers to exit."""
    start = time.time()
    while time.time() - start < timeout:
        alive = [r for r, p in workers.items() if p.poll() is None]
        if not alive:
            return
        time.sleep(0.5)

    # Force kill any remaining
    for rank, proc in workers.items():
        if proc.poll() is None:
            print(f"[orchestrator] Force-killing worker rank={rank}")
            proc.kill()
            proc.wait()


def cleanup(
    lighthouse: Optional[subprocess.Popen],
    workers: Dict[int, subprocess.Popen],
    monitor: Optional[OutputMonitor] = None,
) -> None:
    """Clean up all processes."""
    if monitor:
        monitor.stop()
    for rank, proc in workers.items():
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if lighthouse and lighthouse.poll() is None:
        lighthouse.kill()
        lighthouse.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAID5 FSDP fault tolerance demo")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (GPUs)")
    parser.add_argument("--steps_before_kill", type=int, default=10, help="Steps before killing a worker")
    parser.add_argument("--steps_after_kill", type=int, default=10, help="Steps to run with N-1 workers")
    parser.add_argument("--steps_after_rejoin", type=int, default=10, help="Steps after adding worker back")
    parser.add_argument("--kill_rank", type=int, default=None, help="Rank to kill (default: last)")
    parser.add_argument("--num_parity", type=int, default=1, help="Number of parity syndromes (m=1: RAID5, m=2: dual parity, etc.)")
    args = parser.parse_args()

    N = args.num_workers
    gpu_count = int(subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
    ).strip())
    if gpu_count < N:
        print(f"Error: need {N} GPUs but only {gpu_count} available")
        sys.exit(1)
    gpu_ids = list(range(N))

    kill_rank = args.kill_rank if args.kill_rank is not None else N - 1
    if kill_rank >= N:
        print(f"Error: kill_rank={kill_rank} >= num_workers={N}")
        sys.exit(1)

    lighthouse_port = find_free_port()
    lighthouse = None
    workers: Dict[int, subprocess.Popen] = {}
    monitor = OutputMonitor()

    try:
        # ==================== Phase 1: Start lighthouse + N workers ====================
        print(f"\n{'='*70}")
        print(f"PHASE 1: Starting lighthouse + {N} workers")
        print(f"{'='*70}\n")

        lighthouse = start_lighthouse(lighthouse_port)
        master_port = find_free_port()

        workers = start_workers(N, master_port, lighthouse_port, gpu_ids, monitor, args.num_parity)

        print(f"\n[orchestrator] Waiting for workers to reach step {args.steps_before_kill}...")
        reached = monitor.wait_for_step(args.steps_before_kill, timeout=180.0)
        if not reached:
            all_exited = all(p.poll() is not None for p in workers.values())
            if all_exited:
                print("[orchestrator] All workers exited before reaching target step")
            else:
                print(f"[orchestrator] Timeout (max step seen: {monitor.max_step_seen})")
            return

        print(f"[orchestrator] Workers reached step {args.steps_before_kill}")

        # ==================== Phase 2: Kill one worker ====================
        print(f"\n{'='*70}")
        print(f"PHASE 2: Killing worker rank={kill_rank} (simulating GPU failure)")
        print(f"{'='*70}\n")

        kill_worker(workers, kill_rank)

        print("[orchestrator] Waiting for remaining workers to detect failure and exit...")
        print("[orchestrator] (NCCL will timeout since the killed worker stopped responding)")
        wait_for_workers_exit(workers, timeout=120.0)

        # Report exit codes
        print()
        for rank, proc in sorted(workers.items()):
            status = "KILLED (simulated failure)" if rank == kill_rank else f"exit code {proc.returncode}"
            print(f"  Worker rank={rank}: {status}")

        # Clean up monitor for next phase
        monitor.reset()

        # ==================== Phase 3: Restart with N-1 workers ====================
        print(f"\n{'='*70}")
        print(f"PHASE 3: Restarting with {N - 1} workers (recovery)")
        print(f"{'='*70}\n")

        surviving_gpus = [g for i, g in enumerate(gpu_ids) if i != kill_rank]
        master_port = find_free_port()

        workers = start_workers(N - 1, master_port, lighthouse_port, surviving_gpus, monitor, args.num_parity)

        target_step = args.steps_after_kill
        print(f"[orchestrator] Waiting for {N - 1} workers to reach step {target_step}...")
        reached = monitor.wait_for_step(target_step, timeout=180.0)
        if not reached:
            print(f"[orchestrator] Failed to reach step {target_step} (max: {monitor.max_step_seen})")
            return

        print(f"\n[orchestrator] {N - 1} workers successfully trained for {target_step} steps!")

        # Clean up for next phase
        for rank, proc in workers.items():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        wait_for_workers_exit(workers, timeout=30.0)
        monitor.reset()

        # ==================== Phase 4: Restart with N workers ====================
        print(f"\n{'='*70}")
        print(f"PHASE 4: Adding worker back - restarting with {N} workers")
        print(f"{'='*70}\n")

        master_port = find_free_port()
        workers = start_workers(N, master_port, lighthouse_port, gpu_ids, monitor, args.num_parity)

        target_step = args.steps_after_rejoin
        print(f"[orchestrator] Waiting for {N} workers to reach step {target_step}...")
        reached = monitor.wait_for_step(target_step, timeout=180.0)
        if not reached:
            print(f"[orchestrator] Failed to reach step {target_step} (max: {monitor.max_step_seen})")
            return

        print(f"\n[orchestrator] {N} workers successfully trained for {target_step} steps!")

        # ==================== Done ====================
        print(f"\n{'='*70}")
        print("DEMO COMPLETE")
        print(f"{'='*70}")
        print()
        parity_desc = "RAID5/XOR" if args.num_parity == 1 else f"Reed-Solomon (m={args.num_parity})"
        print(f"Summary (parity: {parity_desc}):")
        print(f"  1. Trained with {N} workers for {args.steps_before_kill} steps")
        print(f"  2. Killed worker rank={kill_rank} (simulated GPU failure)")
        print(f"  3. Remaining workers detected failure and exited")
        print(f"  4. Restarted with {N - 1} workers, trained for {args.steps_after_kill} steps")
        print(f"  5. Added worker back, trained with {N} workers for {args.steps_after_rejoin} steps")

    except KeyboardInterrupt:
        print("\n[orchestrator] Interrupted, cleaning up...")
    finally:
        cleanup(lighthouse, workers, monitor)
        print("[orchestrator] All processes cleaned up")


if __name__ == "__main__":
    main()
