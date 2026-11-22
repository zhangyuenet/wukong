# Wukong: notes for Claude Code

Read [docs/CONVENTIONS.md](docs/CONVENTIONS.md) and [docs/PLAN.md](docs/PLAN.md) first.

## Disk space: hard rule

The dev VM has a 78 GB disk and has crashed from a full disk before (a 44 GB PX4 log).

- After every simulation or test session: stop the simulation, check that
  `pgrep -af 'px4|gz sim|MicroXRCE'` prints nothing, then run `./scripts/clean_logs.sh`.
- Keep only what later work needs. Put a `KEEP` file (with the reason) in a run
  directory that proves a milestone or is a baseline to compare against; delete
  diagnostic plots, intermediate data and one-off outputs once they have been looked at.
- Never leave a simulation, `ros2 run`, or a background command running at the end of
  a turn unless the user asked for it.
- Background command output: redirect to a file and cap it (e.g. `| tail -c 5M`, or
  `timeout`), never stream an unbounded log into the scratchpad.
- Before a long run (eval over several scenarios), check `df -h /`; stop if under 10 GB free.

Limits and the automatic guards are described in CONVENTIONS.md section 7.1.
