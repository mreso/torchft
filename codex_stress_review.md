# CIFAR-10 Stress Test Review

## Findings

1. Healing bug: `examples/stress_test_cifar10.py` was constructing `Manager(..., use_async_quorum=True)`. In torchft, async quorum can stage a recovered checkpoint but defer applying the user state until `should_commit()`. This script was therefore able to:
   - start a step on a freshly respawned worker with random model weights,
   - compute gradients from the wrong model state,
   - then load the healed checkpoint inside commit,
   - and apply those stale gradients to healed weights.

   That is a credible root cause for the severe post-heal regression seen in Claude's Run 4.

2. Chaos/supervisor flow is otherwise structurally correct. The chaos monkey kills a live worker, the supervisor observes exit and respawns the same replica id, and the restarted worker is eligible to heal from surviving peers through torchft checkpoint transport.

3. Data loading is sharded correctly across replicas. Training indices are assigned as `range(replica_id, len(trainset), num_replicas)`, so each replica sees a disjoint slice of the training set. This does not look like the source of the regression.

4. Model state preservation is mostly correct. The script checkpoints and restores model weights, optimizer state, and scheduler state. That is sufficient for correctness here. It does not persist RNG or dataloader position, which can add nondeterminism after restart, but that is secondary to the async-heal bug above.

5. The verdict logic was not robust enough for a stress test. It used the maximum accuracy seen in `accs_by_replica`, which can mix stale and final values across replicas. That can both hide regressions and make the final outcome hard to interpret. For a fault-tolerance stress test, the latest eval window is the right signal.

## Changes made

- Forced synchronous healing before forward passes by changing the manager configuration to `use_async_quorum=False`.
- Added `--seed` and seeded parent/worker RNGs for reproducible chaos timing and worker initialization.
- Included the healed step in `HEAL` events.
- Changed summary/verdict logic to report:
  - first eval accuracy,
  - latest eval step,
  - latest eval accuracy,
  - best observed accuracy,
  - whether enough chaos events occurred.
- Added a printed loss trajectory to the summary.
- Tightened success criteria so chaos-enabled runs require multiple kill events and verdicts use the latest eval path rather than a loose best-ever value.

## Requested rerun

Attempted command:

```bash
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate torchft_dev && cd ~/Projects/torchft_torchstore_workspace/torchft && python -u examples/stress_test_cifar10.py --total-steps 200 --eval-every 50 --chaos-min-steps 20 --chaos-max-steps 40 2>&1 | tee /tmp/codex_stress_run.log
```

Result in this Codex session:

```text
thread 'main' (2005474) panicked at linux-sandbox/src/vendored_bwrap.rs:61:9:
build-time bubblewrap is not available in this build.
codex-linux-sandbox should always compile vendored bubblewrap on Linux targets.
Notes:
- ensure the target OS is Linux
- libcap headers must be available via pkg-config
- bubblewrap sources expected at codex-rs/vendor/bubblewrap (default)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The failure occurs before `python` starts, so there are no runtime `CHAOS`, `HEAL`, `EVAL`, or `VERDICT` logs to report from this session.

## Expected next step outside this sandbox

Re-run the exact command above in a normal shell on the same machine. With the synchronous-heal fix in place, the stress test should now preserve model state before resumed training and should no longer apply gradients computed from a random respawned model.
