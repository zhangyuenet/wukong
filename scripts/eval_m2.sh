#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Run the M2 gutter inspection on several scenarios and score each report.
# 在多个场景上运行 M2 水槽检查，逐个给报告打分。
#
#   ./scripts/eval_m2.sh                    # every sim/scenarios/*.yaml
#   ./scripts/eval_m2.sh basic rotated      # just these
#   OFFLINE=1 ./scripts/eval_m2.sh basic    # no Claude: checks flight and perception only
#   SEEN_FREE=1 ./scripts/eval_m2.sh        # fly only through space seen free (M4 class A)
#
# Each scenario gets a fresh simulation (started and stopped here), so no
# other simulation may be running. Uses the Claude API unless OFFLINE=1:
# about 10 requests per scenario (more for a townhouse unit, whose photos
# along the row are also asked where one home ends).
# The vehicle's true pose is recorded alongside (sim/truth_pose.py), so the
# score checks what each photo really showed, not what the report claims.
# 每个场景都重新启动仿真；运行前不能有别的仿真在运行。同时记录飞机的真实位姿，
# 打分时按照片实际拍到的范围核对，而不是按报告声明的范围。
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
flags=()
[ -n "${OFFLINE:-}" ] && flags=(--offline)
[ -n "${SEEN_FREE:-}" ] && flags+=(--seen-free)

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
    # The truth this run is scored against, kept with it. 本次打分用的真值，与结果放在一起。
    cp "$ROOT/sim/generated/$sc/truth.json" "$log/truth.json"
    python3 "$ROOT/sim/truth_pose.py" "$log/truth_poses.jsonl" > "$log/truth_pose.log" 2>&1 &
    recorder=$!
    ros2 run wukong_mission run_task "${flags[@]}" --scenario "$sc" --out "$log/mission" \
        "$INSTRUCTION" > "$log/run_task.log" 2>&1 || true
    kill -TERM "$recorder" 2>/dev/null || true
    wait "$recorder" 2>/dev/null || true
    grep -E '^\[run_task\] (Needs|No |Could)' "$log/run_task.log" || true
    if python3 "$ROOT/sim/evaluate.py" "$log/mission/report.json" "$log/truth.json" \
            --poses "$log/truth_poses.jsonl" > "$log/score.txt" 2>&1; then
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
