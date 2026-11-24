#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Keep logs and test leftovers under fixed limits.
# 把日志和测试遗留文件控制在固定上限内。
#
#   ./scripts/clean_logs.sh           # delete what is over the limits
#   ./scripts/clean_logs.sh -n        # dry run: only show what would be deleted
#
# Limits (override with environment variables):
#   PX4 flight logs (.ulg)    newest PX4_ULG_KEEP=3 files, at most PX4_ULG_MAX_MB=1024 in total
#   log/eval_m2/<run>/        newest EVAL_KEEP=3 runs, plus runs that contain a KEEP file
#   log/missions/<run>/       newest MISSIONS_KEEP=5 runs, plus runs that contain a KEEP file
#   log/build_*, log/test_*   newest COLCON_LOG_KEEP=2 of each kind (colcon's own logs)
#   ~/.ros/log                files older than ROS_LOG_DAYS=7 days
#
# To keep a run for good (e.g. the run that proved a milestone), put a KEEP file in
# it: echo "M3 passed" > log/eval_m2/<run>/KEEP
# 想永久保留某次运行（例如证明里程碑完成的那次），在目录里放一个 KEEP 文件。
#
# sim.sh runs this on every start; eval_m2.sh runs it when it finishes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/versions.env" 2>/dev/null || true
PX4_DIR="${PX4_DIR:-$HOME/wip/wukong_deps/PX4-Autopilot}"
ULG_DIR="$PX4_DIR/build/px4_sitl_default/rootfs/log"

PX4_ULG_KEEP="${PX4_ULG_KEEP:-3}"
PX4_ULG_MAX_MB="${PX4_ULG_MAX_MB:-1024}"
EVAL_KEEP="${EVAL_KEEP:-3}"
MISSIONS_KEEP="${MISSIONS_KEEP:-5}"
COLCON_LOG_KEEP="${COLCON_LOG_KEEP:-2}"
ROS_LOG_DAYS="${ROS_LOG_DAYS:-7}"

DRY=0
[ "${1:-}" = "-n" ] && DRY=1
freed=0

remove() {  # remove <path>...
    local p kb
    for p in "$@"; do
        [ -e "$p" ] || continue
        kb=$(du -sk "$p" 2>/dev/null | cut -f1)
        freed=$((freed + ${kb:-0}))
        if [ $DRY = 1 ]; then echo "[clean] would delete $p (${kb} KB)"; else rm -rf -- "$p"; fi
    done
}

# Timestamp-named run directories in <dir>, oldest first, minus the newest <keep> and
# those with KEEP. Hand-named runs (e.g. --out log/missions/m3_diag) are never touched.
# 只轮转按时间戳命名的目录；手动命名的目录（--out 指定的名字）不会被删除。
# Sorted by name (YYYYmmdd-HHMMSS), not mtime: adding a KEEP file changes the mtime.
# 按目录名（时间戳）排序而不是修改时间：放入 KEEP 文件会改变目录的修改时间。
old_runs() {  # old_runs <dir> <keep>
    [ -d "$1" ] || return 0
    find "$1" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | sort | head -n -"$2" \
        | while read -r d; do [ -e "$d/KEEP" ] || echo "$d"; done
}

# 1. PX4 flight logs: by count, then by total size (oldest first).
if [ -d "$ULG_DIR" ]; then
    mapfile -t ulgs < <(find "$ULG_DIR" -name '*.ulg' -printf '%T@ %s %p\n' | sort -rn)
    total=0
    for i in "${!ulgs[@]}"; do
        read -r _ size path <<< "${ulgs[$i]}"
        total=$((total + size / 1048576))
        if [ "$i" -ge "$PX4_ULG_KEEP" ] || [ "$total" -gt "$PX4_ULG_MAX_MB" ]; then
            remove "$path"
        fi
    done
    [ $DRY = 1 ] || find "$ULG_DIR" -mindepth 1 -type d -empty -delete
fi

# 2. Our own run outputs.
mapfile -t runs < <(old_runs "$ROOT/log/eval_m2" "$EVAL_KEEP"; old_runs "$ROOT/log/missions" "$MISSIONS_KEEP")
remove "${runs[@]}"

# 3. colcon build/test logs (the latest_* symlinks always point at the newest).
for kind in build test test-result; do
    mapfile -t old < <(find "$ROOT/log" -mindepth 1 -maxdepth 1 -type d -name "${kind}_*" \
        | sort | head -n -"$COLCON_LOG_KEEP")
    remove "${old[@]}"
done

# 4. ROS 2 node logs.
if [ -d "$HOME/.ros/log" ]; then
    mapfile -t old < <(find "$HOME/.ros/log" -mindepth 1 -maxdepth 1 -mtime +"$ROS_LOG_DAYS")
    remove "${old[@]}"
fi

verb=deleted; [ $DRY = 1 ] && verb="would free"
echo "[clean] $verb $((freed / 1024)) MB; now: $(du -sh "$ROOT/log" 2>/dev/null | cut -f1) in log/," \
     "$(du -sh "$ULG_DIR" 2>/dev/null | cut -f1) of PX4 logs," \
     "$(df -h --output=avail / | tail -1 | tr -d ' ') free on /"
