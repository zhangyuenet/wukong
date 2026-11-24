#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Step 1: install ROS 2 (Jazzy) and base tools.
# 第 1 步：安装 ROS 2 Jazzy 和基础工具。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../versions.env"

echo "==> Base tools / 基础工具"
sudo apt-get update
sudo apt-get install -y curl wget gnupg lsb-release software-properties-common \
    mesa-utils git build-essential cmake
sudo add-apt-repository -y universe

if [ ! -f /etc/apt/sources.list.d/ros2.sources ] && [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    echo "==> Adding ROS 2 apt source / 添加 ROS 2 软件源"
    ver=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
          | grep -F '"tag_name"' | awk -F'"' '{print $4}')
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/ros2-apt-source.deb" \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${codename}_all.deb"
    sudo dpkg -i "$tmp/ros2-apt-source.deb"
    rm -rf "$tmp"
fi

echo "==> Installing ROS 2 ${ROS_DISTRO} / 安装 ROS 2 ${ROS_DISTRO}"
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
    "ros-${ROS_DISTRO}-desktop" \
    "ros-${ROS_DISTRO}-ros-gz" \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

echo "==> Done. ROS 2 ${ROS_DISTRO} installed. / 完成"
