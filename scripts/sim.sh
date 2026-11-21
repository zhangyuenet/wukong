#!/usr/bin/env bash
# Start the simulation with one command: XRCE-DDS Agent + PX4 SITL + Gazebo.
# 一条命令启动仿真：XRCE-DDS Agent + PX4 SITL + Gazebo。
#
#   ./scripts/sim.sh                  # x500 with a camera, house world, Gazebo GUI
#   HEADLESS=1 ./scripts/sim.sh       # no GUI, saves CPU
#   ./scripts/sim.sh gz_x500          # another PX4 model (here: no camera)
#   WORLD=default ./scripts/sim.sh    # PX4's empty world instead of the house
#   SCENARIO=rotated ./scripts/sim.sh # house from sim/scenarios/rotated.yaml (M2),
#                                     # depth camera model by default
#
# Worlds come from sim/worlds/ in this repo (falling back to PX4's own worlds).
# When the model has a camera, its image is bridged to the ROS 2 topic /camera/image_raw.
# 仿真世界优先用本仓库 sim/worlds/ 下的文件；带相机的机型会把图像桥接到 /camera/image_raw。
#
# Ctrl+C (or `shutdown` in the pxh shell) stops everything.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # ROS setup.bash is not `set -u` safe
set -u

SCENARIO="${SCENARIO:-}"
if [ -n "$SCENARIO" ]; then
    MODEL="${1:-gz_x500_depth}"
else
    MODEL="${1:-gz_x500_mono_cam}"
fi
WORLD="${WORLD:-wukong_house}"

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

# World and models. PX4 only loads worlds from its own worlds directory
# ($PX4_GZ_WORLDS, set by its gz_env.sh), so link ours in there. Models are found
# through GZ_SIM_RESOURCE_PATH, which PX4 appends to rather than replaces.
# PX4 只从自己的 worlds 目录加载世界文件，所以把本仓库的世界文件链接过去。
PX4_WORLDS="$PX4_DIR/Tools/simulation/gz/worlds"
if [ -n "$SCENARIO" ]; then
    # Generate world, ground truth and rough map: sim/generated/<scenario>/.
    # 生成仿真世界、真值和粗略地图。
    SCENARIO_FILE="$WUKONG_ROOT/sim/scenarios/$SCENARIO.yaml"
    [ -f "$SCENARIO_FILE" ] || { echo "[sim] no scenario $SCENARIO_FILE" >&2; exit 1; }
    WORLD="wukong_$SCENARIO"
    python3 "$WUKONG_ROOT/sim/scenario.py" "$SCENARIO_FILE" >/dev/null
    ln -sf "$WUKONG_ROOT/sim/generated/$SCENARIO/$WORLD.sdf" "$PX4_WORLDS/$WORLD.sdf"
    echo "[sim] scenario $SCENARIO: map for the drone and ground truth in sim/generated/$SCENARIO/"
elif [ -f "$WUKONG_ROOT/sim/worlds/$WORLD.sdf" ]; then
    ln -sf "$WUKONG_ROOT/sim/worlds/$WORLD.sdf" "$PX4_WORLDS/$WORLD.sdf"
elif [ ! -f "$PX4_WORLDS/$WORLD.sdf" ]; then
    echo "[sim] unknown world '$WORLD' (not in sim/worlds/ or $PX4_WORLDS)" >&2
    exit 1
fi
export PX4_GZ_WORLD="$WORLD"
export GZ_SIM_RESOURCE_PATH="$WUKONG_ROOT/sim/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
echo "[sim] model $MODEL, world $WORLD"

# Warn when PX4 flight logs (.ulg) pile up. PX4 日志过多时提醒清理。
PX4_LOG_GB=$(du -s --block-size=1G "$ROOTFS/log" 2>/dev/null | cut -f1 || echo 0)
if [ "${PX4_LOG_GB:-0}" -ge 5 ]; then
    echo "[sim] WARNING: PX4 logs use ${PX4_LOG_GB} GB in $ROOTFS/log"
    echo "[sim]          delete old ones with: rm -rf $ROOTFS/log/*"
fi

# Refuse to start a second simulation. The cleanup below shuts down PX4 and
# Gazebo by name, so a second copy exiting would also stop the first one.
# 已有仿真在运行时拒绝启动：下面的清理按进程名关闭 PX4 和 Gazebo，会误关前一个仿真。
if pidof px4 >/dev/null || pgrep -f '^gz sim' >/dev/null; then
    echo "[sim] a simulation is already running (px4 or gz sim); stop it first:" >&2
    echo "[sim]   Ctrl+C in its terminal, or: pgrep -af 'px4|gz sim'" >&2
    exit 1
fi

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
    # Gazebo server started by PX4. With a camera sensor it hangs on SIGTERM
    # (seen with ogre1 in the VM), so give it 5 s, then kill it.
    # 带相机时 Gazebo 收到 SIGTERM 后会卡住，等 5 秒后强制结束。
    pkill -f '^gz sim' 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        pgrep -f '^gz sim' >/dev/null || break
        sleep 1
    done
    pkill -KILL -f '^gz sim' 2>/dev/null || true
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
        # Log only while armed. PX4 SITL defaults to SDLOG_MODE=1 (from boot until
        # the first disarm), so a sim left idle logs forever: ~1 GB/hour with the
        # SITL profile, which once filled the disk (44 GB). The logger has already
        # started by now, so set the param and restart the logger without -e.
        # 只在解锁期间记录日志。SITL 默认从启动就开始记录、直到第一次上锁，
        # 仿真空转时会一直写下去（约 1 GB/小时），曾经写满过磁盘。
        if [ "$("$BIN/px4-param" show -q SDLOG_MODE 2>/dev/null)" != "0" ]; then
            "$BIN/px4-param" set SDLOG_MODE 0 >/dev/null 2>&1
        fi
        "$BIN/px4-logger" stop >/dev/null 2>&1 || true
        "$BIN/px4-logger" start -b 1000 -t -m all >/dev/null 2>&1 \
            && echo "[sim] PX4 logger restarted: logging only while armed"
        break
    done
) &
pids+=($!)

# 3. Cameras -> ROS 2. The colour image is always /camera/image_raw; the depth
#    camera (x500_depth) adds /depth/image (32FC1, metres) and /depth/camera_info.
#    相机桥接：彩色图像统一发到 /camera/image_raw；深度相机另外发 /depth/...
# Run the executables directly: killing a `ros2 run` wrapper leaves its child running.
# 直接运行可执行文件：杀掉 ros2 run 包装进程时，子进程不会一起退出。
GZ_MODEL="${MODEL#gz_}_0"   # PX4 names the first instance <model>_0
GZ_SENSORS="/world/$WORLD/model/$GZ_MODEL/link/camera_link/sensor"
GZ_COLOR=""
case "$MODEL" in
    *mono_cam*) GZ_COLOR="$GZ_SENSORS/imager/image" ;;
    *x500_depth*) GZ_COLOR="$GZ_SENSORS/IMX214/image" ;;
esac
if [ -n "$GZ_COLOR" ]; then
    IMAGE_BRIDGE="$(ros2 pkg prefix ros_gz_image)/lib/ros_gz_image/image_bridge"
    remaps=(-r "$GZ_COLOR:=/camera/image_raw")
    topics=("$GZ_COLOR")
    if [[ "$MODEL" == *x500_depth* ]]; then
        topics+=(/depth_camera)
        remaps+=(-r "/depth_camera:=/depth/image")
        "$(ros2 pkg prefix ros_gz_bridge)/lib/ros_gz_bridge/parameter_bridge" \
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' \
            --ros-args -r /camera_info:=/depth/camera_info > "$LOG_DIR/camera_info_bridge.log" 2>&1 &
        pids+=($!)
    fi
    "$IMAGE_BRIDGE" "${topics[@]}" --ros-args "${remaps[@]}" > "$LOG_DIR/camera_bridge.log" 2>&1 &
    pids+=($!)
    echo "[sim] camera bridge: ${topics[*]} -> ROS 2 (log: $LOG_DIR/camera_bridge.log)"
fi

# 4. PX4 SITL + Gazebo (foreground)
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
