#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Run every check that does not need the simulator, in one go.
# 一次运行所有不需要仿真的检查。
#
#   ./scripts/check.sh
#
# 1. The three ROS 2 packages: build, then their tests (unit tests, flake8, pep257).
# 2. sim/ is not a package, so colcon does not see it: the same flake8 and
#    pep257 rules (ament's configuration), then its pytest tests.
# Exit code 0 only if everything passes.
# 1. 三个 ROS 2 包：编译后运行测试（单元测试、flake8、pep257）。
# 2. sim/ 不是功能包，colcon 不会检查它：用同样的 ament 规则检查风格，再运行它的测试。
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh" >/dev/null
set -u
cd "$ROOT"

PACKAGES=(wukong_flight wukong_perception wukong_mission)
failed=()

step() { echo; echo "[check] $*"; }

step "build ${PACKAGES[*]}"
colcon build --symlink-install --packages-select "${PACKAGES[@]}" 2>&1 | tail -3 \
    || failed+=("build")

step "package tests"
colcon test --packages-select "${PACKAGES[@]}" 2>&1 | tail -2
for p in "${PACKAGES[@]}"; do
    # Per package: the workspace-wide summary also counts third-party packages.
    # 逐个包汇总：整个工作空间的汇总还包括第三方包。
    if ! colcon test-result --test-result-base "build/$p" >/dev/null 2>&1; then
        colcon test-result --test-result-base "build/$p" --verbose 2>&1 | tail -20
        failed+=("$p tests")
    fi
    colcon test-result --test-result-base "build/$p" 2>&1 | tail -1 | sed "s/^/  $p: /"
done

step "sim/: flake8 and pep257"
out=$(ament_flake8 sim/ 2>&1) || { grep -E '^sim/' <<<"$out"; failed+=("sim flake8"); }
out=$(ament_pep257 sim/ 2>&1) || { grep -E '^sim/' <<<"$out"; failed+=("sim pep257"); }

step "sim/: tests"
python3 -m pytest -q sim/ 2>&1 | tail -3 || failed+=("sim tests")

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "[check] all passed"
else
    echo "[check] FAILED: ${failed[*]}"
    exit 1
fi
