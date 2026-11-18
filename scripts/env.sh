# Wukong development environment. Source this file; do not execute it.
# Wukong 开发环境变量。用 source 加载，不要直接执行。

WUKONG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WUKONG_ROOT
source "$WUKONG_ROOT/scripts/versions.env"
export ROS_DISTRO PX4_DIR WUKONG_DEPS_DIR

[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ] && source "/opt/ros/${ROS_DISTRO}/setup.bash"
[ -f "$WUKONG_ROOT/install/setup.bash" ] && source "$WUKONG_ROOT/install/setup.bash"

# Keep ROS 2 discovery inside this VM, so it does not talk to other machines on the LAN.
# ROS 2 节点发现限定在本机，不与局域网上的其它设备互相发现。
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# Gazebo's Qt GUI is more reliable under XWayland than native Wayland.
# Gazebo 的 Qt 界面在 Wayland 下走 XWayland 更稳定。
export QT_QPA_PLATFORM=xcb

# Never cap OpenGL in VMware (see scripts/setup/02_px4.sh).
unset SVGA_VGPU10
