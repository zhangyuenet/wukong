#!/usr/bin/env bash
# Step 2: clone PX4, install its dependencies (incl. Gazebo Harmonic), build SITL.
# 第 2 步：获取 PX4 源码，安装依赖（含 Gazebo Harmonic），编译 SITL。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../versions.env"

mkdir -p "$WUKONG_DEPS_DIR"
if [ ! -d "$PX4_DIR/.git" ]; then
    echo "==> Cloning PX4 ${PX4_VERSION} (several GB, takes a while) / 克隆 PX4（数 GB，较慢）"
    git clone --recursive --branch "$PX4_VERSION" \
        https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
else
    echo "==> PX4 already at $PX4_DIR, syncing to ${PX4_VERSION} / 已存在，切到指定版本"
    git -C "$PX4_DIR" fetch --tags
    git -C "$PX4_DIR" checkout "$PX4_VERSION"
    git -C "$PX4_DIR" submodule update --init --recursive
fi

echo "==> Running PX4 ubuntu.sh (simulation only, no NuttX toolchain) / 运行 PX4 依赖安装脚本"
# --no-nuttx: the hardware toolchain is not needed until M5 (real drone).
# --no-nuttx：真机固件工具链到 M5 阶段才需要。
bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx

# ubuntu.sh appends `export SVGA_VGPU10=0` to ~/.profile inside VMware. That was a
# fix for the old Gazebo Classic; it caps OpenGL at 2.1 and breaks Gazebo Harmonic
# (ogre2 needs OpenGL 3.3+). Remove it.
# ubuntu.sh 在 VMware 中会往 ~/.profile 写 SVGA_VGPU10=0，它会把 OpenGL 限制在 2.1，
# 导致 Gazebo Harmonic 无法渲染，这里删掉。
if grep -q 'SVGA_VGPU10=0' ~/.profile 2>/dev/null; then
    sed -i '/SVGA_VGPU10=0/d' ~/.profile
    echo "==> Removed SVGA_VGPU10=0 from ~/.profile / 已从 ~/.profile 删除 SVGA_VGPU10=0"
fi

echo "==> Building PX4 SITL / 编译 PX4 SITL"
make -C "$PX4_DIR" px4_sitl

echo "==> Done. Log out and back in so ~/.profile changes take effect. / 完成，请注销后重新登录。"
