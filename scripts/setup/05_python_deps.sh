#!/usr/bin/env bash
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
# Step 5: Python packages for the mission layer (Claude API client).
# 第 5 步：任务层用到的 Python 包（Claude API 客户端）。
#
# Installed with pip into ~/.local, like PX4's own Python tools, so that the
# ROS 2 nodes (run by /usr/bin/python3) can import them. Pinned in versions.env.
# 装在 ~/.local，与 PX4 的 Python 工具一致，这样 ROS 2 节点可以直接 import。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../versions.env"

echo "==> Installing anthropic==${ANTHROPIC_SDK_VERSION} / 安装 Claude API SDK"
pip install --user --break-system-packages "anthropic==${ANTHROPIC_SDK_VERSION}"

# Shadow Ubuntu's python3-brotli 1.1.0, which httpx2 cannot use (see versions.env).
# 用新版 Brotli 覆盖 Ubuntu 自带的 1.1.0。
echo "==> Installing Brotli==${BROTLI_VERSION}"
pip install --user --break-system-packages "Brotli==${BROTLI_VERSION}"

python3 -c "import anthropic, pydantic, brotli; print('anthropic', anthropic.__version__, '/ pydantic', pydantic.VERSION, '/ brotli', brotli.__version__)"
echo "==> Done. Put your API key in .env as ANTHROPIC_API_KEY (see .env.example)."
echo "    完成。把 API 密钥写进 .env 的 ANTHROPIC_API_KEY（参考 .env.example）。"
