#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Send a command to the running PX4 SITL from any terminal (same as typing it at pxh>).
# 从任意终端向正在运行的 PX4 SITL 发送命令，效果等同于在 pxh> 中输入。
#
#   ./scripts/px4cmd.sh commander land
#   ./scripts/px4cmd.sh commander rtl
#   ./scripts/px4cmd.sh listener vehicle_status -n 1
#   ./scripts/px4cmd.sh param show NAV_DLL_ACT
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/versions.env"

if [ $# -lt 1 ]; then
    sed -n '2,8p' "$0"; exit 1
fi
BIN="$PX4_DIR/build/px4_sitl_default/bin"
cmd="$1"; shift
if [ ! -x "$BIN/px4-$cmd" ]; then
    echo "Unknown PX4 command: $cmd (see $BIN/px4-*)" >&2; exit 1
fi
cd "$PX4_DIR/build/px4_sitl_default/rootfs"
exec "$BIN/px4-$cmd" "$@"
