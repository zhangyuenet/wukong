#!/usr/bin/env bash
# Start the simulation with one command: XRCE-DDS Agent + PX4 SITL + Gazebo.
# 一条命令启动仿真：XRCE-DDS Agent + PX4 SITL + Gazebo。
#
#   ./scripts/sim.sh                # default model gz_x500, with Gazebo GUI
#   HEADLESS=1 ./scripts/sim.sh     # no GUI, saves CPU
#   ./scripts/sim.sh gz_x500_depth  # another PX4 model
#
# Ctrl+C (or `shutdown` in the pxh shell) stops everything.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # ROS setup.bash is not `set -u` safe
set -u

MODEL="${1:-gz_x500}"

# Render engine. In the VMware Fusion VM, Gazebo's default ogre2 on the 3D-accelerated
# driver draws only the background (an all-grey view). ogre1 renders correctly on the
# GPU (~10% CPU for the GUI); ogre2 in software also works but costs ~1.5 cores.
# Override with PX4_GZ_SIM_RENDER_ENGINE=ogre2 on a machine with a real GPU.
# 渲染引擎：VMware 虚拟机里 ogre2 + 硬件加速只画出灰色背景，ogre1 正常且省 CPU。
export PX4_GZ_SIM_RENDER_ENGINE="${PX4_GZ_SIM_RENDER_ENGINE:-ogre}"
BIN="$PX4_DIR/build/px4_sitl_default/bin"
ROOTFS="$PX4_DIR/build/px4_sitl_default/rootfs"
LOG_DIR="$WUKONG_ROOT/log/sim"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
    trap - EXIT INT TERM
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    # PX4 catches SIGTERM without exiting; ask it to shut down, then force it.
    if pidof px4 >/dev/null; then
        (cd "$ROOTFS" && timeout 5 "$BIN/px4-shutdown" >/dev/null 2>&1) || true
        sleep 1
        pkill -KILL -x px4 2>/dev/null || true
    fi
    pkill -f '^gz sim' 2>/dev/null || true   # Gazebo server started by PX4
}
trap cleanup EXIT INT TERM

# 1. XRCE-DDS Agent
if pgrep -x MicroXRCEAgent >/dev/null; then
    echo "[sim] MicroXRCEAgent already running, reusing it"
else
    MicroXRCEAgent udp4 -p 8888 > "$LOG_DIR/agent.log" 2>&1 &
    pids+=($!)
    echo "[sim] MicroXRCEAgent started (log: $LOG_DIR/agent.log)"
fi

# 2. Sim-only parameter fix, applied once PX4 has loaded its airframe.
#    PX4 v1.17's gz_x500 sets NAV_DLL_ACT=2, which refuses to arm without a
#    MAVLink ground station. In simulation our ROS 2 nodes are in charge, so
#    turn it off. It must be set after startup: setting it earlier, while it
#    still equals the default 0, is discarded when the airframe changes the
#    default. NEVER copy this to a real drone, which needs the datalink failsafe.
#    仅限仿真：v1.17 的 gz_x500 设置 NAV_DLL_ACT=2，没有地面站就不能解锁。
#    仿真中由 ROS 2 节点控制，所以关掉它。真机绝不能这样做。
(
    cd "$ROOTFS"
    for _ in $(seq 1 120); do
        sleep 1
        "$BIN/px4-uxrce_dds_client" status >/dev/null 2>&1 || continue
        if [ "$("$BIN/px4-param" show -q NAV_DLL_ACT 2>/dev/null)" != "0" ]; then
            "$BIN/px4-param" set NAV_DLL_ACT 0 >/dev/null 2>&1 \
                && echo "[sim] NAV_DLL_ACT set to 0 (sim only, no GCS needed to arm)"
        fi
        break
    done
) &
pids+=($!)

# 3. PX4 SITL + Gazebo (foreground)
cd "$PX4_DIR"
if [ -t 0 ]; then
    make px4_sitl "$MODEL"
else
    # No terminal: run without the interactive pxh shell, which would spin on EOF.
    # 没有终端时不启动交互 shell，否则它会在 EOF 上空转刷屏。
    # Run in background + wait so that a signal triggers cleanup right away.
    (cd "$ROOTFS" && PX4_SIM_MODEL="$MODEL" exec "$BIN/px4" -d) &
    pids+=($!)
    wait "$!"
fi
