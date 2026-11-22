#!/usr/bin/env bash
# Run the M2 gutter inspection on several scenarios and score each report.
# 在多个场景上运行 M2 水槽检查，逐个给报告打分。
#
#   ./scripts/eval_m2.sh                    # every sim/scenarios/*.yaml
#   ./scripts/eval_m2.sh basic rotated      # just these
#   OFFLINE=1 ./scripts/eval_m2.sh basic    # no Claude: checks flight and perception only
#
# Each scenario gets a fresh simulation (started and stopped here), so no
# other simulation may be running. Uses the Claude API unless OFFLINE=1:
# about 10 requests per scenario.
# 每个场景都重新启动仿真；运行前不能有别的仿真在运行。
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
set -u

INSTRUCTION="${INSTRUCTION:-检查这栋房子所有的水槽，告诉我哪几段需要清理}"
OUT="$ROOT/log/eval_m2/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
if [ $# -gt 0 ]; then
    scenarios=("$@")
else
    scenarios=()
    for f in "$ROOT"/sim/scenarios/*.yaml; do scenarios+=("$(basename "$f" .yaml)"); done
fi
offline=()
[ -n "${OFFLINE:-}" ] && offline=(--offline)

if pidof px4 >/dev/null || pgrep -f '^gz sim' >/dev/null; then
    echo "[eval] a simulation is already running; stop it first" >&2
    exit 1
fi

stop_sim() {
    local pid
    pid=$(pgrep -f "bash $ROOT/scripts/sim.sh" | head -1 || true)
    [ -n "$pid" ] && kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        pidof px4 >/dev/null || pgrep -f '^gz sim' >/dev/null || return 0
        sleep 1
    done
}
trap stop_sim EXIT INT TERM

declare -A result
for sc in "${scenarios[@]}"; do
    echo "[eval] === $sc ==="
    log="$OUT/$sc"
    mkdir -p "$log"
    (SCENARIO="$sc" HEADLESS=1 setsid "$ROOT/scripts/sim.sh" < /dev/null > "$log/sim.log" 2>&1 &)
    for _ in $(seq 1 180); do
        grep -q 'logger restarted' "$log/sim.log" 2>/dev/null && break
        grep -qE 'Aborted|already running' "$log/sim.log" 2>/dev/null && break
        sleep 1
    done
    if ! grep -q 'logger restarted' "$log/sim.log"; then
        echo "[eval] $sc: simulation did not start (see $log/sim.log)"
        result[$sc]="sim failed"
        stop_sim
        continue
    fi
    sleep 3
    ros2 run wukong_mission run_task "${offline[@]}" --scenario "$sc" --out "$log/mission" \
        "$INSTRUCTION" > "$log/run_task.log" 2>&1 || true
    grep -E '^\[run_task\] (Needs|No |Could)' "$log/run_task.log" || true
    if python3 "$ROOT/sim/evaluate.py" "$log/mission/report.json" \
            "$ROOT/sim/generated/$sc/truth.json" > "$log/score.txt" 2>&1; then
        result[$sc]="PASS"
    else
        result[$sc]="FAIL"
    fi
    tail -1 "$log/score.txt"
    stop_sim
done

echo
echo "[eval] summary (details in $OUT):"
fail=0
for sc in "${scenarios[@]}"; do
    echo "  $sc: ${result[$sc]}"
    [ "${result[$sc]}" = PASS ] || fail=1
done
# Keep the newest runs only; mark a run worth keeping with a KEEP file.
# 只保留最近几次；值得长期保留的运行请在目录里放 KEEP 文件。
"$ROOT/scripts/clean_logs.sh" || true
exit $fail
