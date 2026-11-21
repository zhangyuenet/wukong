#!/usr/bin/env bash
# Check the Wukong development environment. Safe to run any time.
# 检查 Wukong 开发环境，随时可以运行。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/versions.env"

pass=0; fail=0
ok()  { echo "  [OK]   $*"; pass=$((pass+1)); }
bad() { echo "  [FAIL] $*"; fail=$((fail+1)); }
check() { local name=$1; shift; if out=$("$@" 2>/dev/null); then ok "$name ${out:+— $out}"; else bad "$name"; fi; }

echo "System / 系统"
check "Ubuntu 24.04"     bash -c 'lsb_release -rs | grep -qx 24.04 && lsb_release -ds'
check "Architecture"     uname -m
gl=$(glxinfo -B 2>/dev/null | grep -E 'OpenGL core profile version|OpenGL version' | head -1 | sed 's/.*: //')
if [ -z "$gl" ]; then bad "OpenGL (glxinfo missing? install mesa-utils)"
elif echo "$gl" | grep -qE '^([4-9]|3\.[3-9])'; then ok "OpenGL $gl"
else bad "OpenGL $gl (Gazebo needs 3.3+; check VMware 3D acceleration / SVGA_VGPU10)"; fi
if grep -q 'SVGA_VGPU10=0' ~/.profile 2>/dev/null; then bad "~/.profile still sets SVGA_VGPU10=0"; fi

echo "ROS 2"
check "ROS 2 ${ROS_DISTRO}" test -f "/opt/ros/${ROS_DISTRO}/setup.bash"
check "colcon"            bash -c 'command -v colcon'
check "rosdep"            bash -c 'test -d ~/.ros/rosdep/sources.cache'
check "vcs"               bash -c 'command -v vcs'

echo "Simulation / 仿真"
check "Gazebo"            bash -c 'gz sim --versions 2>/dev/null | head -1 | grep .'
check "PX4 source ${PX4_VERSION}" bash -c "git -C '$PX4_DIR' describe --tags --exact-match 2>/dev/null | grep -qx '$PX4_VERSION' && echo '$PX4_DIR'"
check "PX4 SITL build"    test -x "$PX4_DIR/build/px4_sitl_default/bin/px4"
check "MicroXRCEAgent"    bash -c 'command -v MicroXRCEAgent'

echo "Workspace / 工作空间"
check "px4_msgs source"   test -d "$ROOT/src/third_party/px4_msgs"
check "workspace built"   test -f "$ROOT/install/setup.bash"
check "env.sh in ~/.bashrc" grep -qF "scripts/env.sh" ~/.bashrc

echo "Mission layer / 任务层"
check "anthropic SDK ${ANTHROPIC_SDK_VERSION}" bash -c "python3 -c 'import anthropic; print(anthropic.__version__)' | grep -qx '${ANTHROPIC_SDK_VERSION}' && echo ok"
check "Brotli ${BROTLI_VERSION} (for the SDK)" bash -c "python3 -c 'import brotli; print(brotli.__version__)' | grep -qx '${BROTLI_VERSION}' && echo ok"
check "Pillow"            bash -c "python3 -c 'import PIL; print(PIL.__version__)'"
if [ -n "${ANTHROPIC_API_KEY:-}" ] || grep -qE '^ANTHROPIC_API_KEY=.+' "$ROOT/.env" 2>/dev/null; then
    ok "Claude API key (environment or .env)"
else
    echo "  [--]   Claude API key not set: run_task needs it, run_task --offline does not"
fi

echo
echo "Passed: $pass   Failed: $fail"
[ "$fail" -eq 0 ]
