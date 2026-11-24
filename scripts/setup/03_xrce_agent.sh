#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Step 3: build and install the Micro XRCE-DDS Agent (bridge between PX4 and ROS 2).
# 第 3 步：编译安装 Micro XRCE-DDS Agent（PX4 与 ROS 2 之间的桥）。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../versions.env"

# Build outside the ROS environment so the agent uses its own bundled Fast DDS.
# 在 ROS 环境之外编译，让 Agent 使用自带的 Fast DDS。
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION || true

src="$WUKONG_DEPS_DIR/Micro-XRCE-DDS-Agent"
mkdir -p "$WUKONG_DEPS_DIR"
if [ ! -d "$src/.git" ]; then
    git clone --branch "$XRCE_AGENT_VERSION" \
        https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$src"
fi

cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$src/build" -j"$(nproc)"
sudo cmake --install "$src/build"
sudo ldconfig /usr/local/lib/

echo "==> Done: $(command -v MicroXRCEAgent) / 完成"
