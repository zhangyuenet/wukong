#!/usr/bin/env bash
# Step 4: set up the ROS 2 workspace (this repository is the colcon workspace).
# 第 4 步：准备 ROS 2 工作空间（本仓库本身就是 colcon 工作空间）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/versions.env"

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

cd "$ROOT"
mkdir -p src/third_party
echo "==> Importing third-party packages / 导入第三方功能包"
vcs import --input wukong.repos --skip-existing src/third_party
vcs pull src/third_party || true

echo "==> Installing package dependencies / 安装依赖"
rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -y

echo "==> Building / 编译"
colcon build --symlink-install

# Hook env.sh into ~/.bashrc once.
line="source \"$ROOT/scripts/env.sh\""
if ! grep -qF "$line" ~/.bashrc; then
    printf '\n# Wukong dev environment\n%s\n' "$line" >> ~/.bashrc
    echo "==> Added env.sh to ~/.bashrc / 已加入 ~/.bashrc"
fi

echo "==> Done. Open a new terminal. / 完成，请打开新终端。"
