# 开发环境搭建 (Development Setup)

本文档对应 [PLAN.md](PLAN.md) 的 M0 阶段：在 Ubuntu 虚拟机里搭好 PX4 仿真、
Gazebo 和 ROS 2，并让三者连通。

## 1. 版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Ubuntu | 24.04 LTS（ARM64） | VMware Fusion 虚拟机，宿主机为 MacBook Air M2 |
| ROS 2 | Jazzy | 与 Ubuntu 24.04 配套的 LTS 版本 |
| Gazebo | Harmonic | PX4 与 ROS 2 Jazzy 共同支持的版本 |
| PX4 | v1.17.0 | 当前最新稳定版 |
| px4_msgs | `release/1.17` | 必须与 PX4 版本对应 |
| Micro XRCE-DDS Agent | v2.4.3 | PX4 v1.17 不兼容 v3.x |

所有版本号集中写在 [scripts/versions.env](../scripts/versions.env) 和
[wukong.repos](../wukong.repos)，升级时只改这两个文件。

## 2. 目录布局

```
~/wip/
├── wukong/                     本仓库，同时也是 colcon 工作空间
│   ├── src/                    Wukong 自己的 ROS 2 功能包
│   │   └── third_party/        由 wukong.repos 导入的 px4_msgs 等（不提交）
│   ├── build/ install/ log/    colcon 产物（不提交）
│   └── scripts/
└── wukong_deps/                第三方源码，放在仓库之外
    ├── PX4-Autopilot/
    └── Micro-XRCE-DDS-Agent/
```

## 3. 虚拟机准备

开始之前，在 VMware Fusion 的虚拟机设置里确认：

- **显示器 → 加速 3D 图形**：打开。Gazebo 需要 OpenGL 3.3 以上。
- **处理器**：M2 有 8 个核心，可以给虚拟机 6 个，编译会快不少（当前是 4 个）。
- **内存**：保持 12GB 左右即可。
- **磁盘**：至少留出 30GB 空间。PX4 源码加编译产物大约 10GB。

## 4. 安装步骤

每一步都可以重复执行。中途失败的话，修好问题后重跑当前这一步就行。

```bash
cd ~/wip/wukong

# 1. ROS 2 Jazzy、ros_gz 和开发工具（约 15 分钟）
./scripts/setup/01_ros2.sh

# 2. PX4 源码、PX4 依赖（含 Gazebo Harmonic）、编译 SITL（约 30–60 分钟）
./scripts/setup/02_px4.sh
#    完成后注销并重新登录一次

# 3. Micro XRCE-DDS Agent（约 10 分钟）
./scripts/setup/03_xrce_agent.sh

# 4. 导入 px4_msgs / px4_ros_com，编译工作空间，并把 env.sh 加入 ~/.bashrc
./scripts/setup/04_workspace.sh

# 打开一个新终端后自检
./scripts/check_env.sh
```

`04_workspace.sh` 会在 `~/.bashrc` 中加入一行 `source .../scripts/env.sh`。
这个文件负责加载 ROS 2 和本仓库的工作空间，把 ROS 2 节点发现限制在本机，
并让 Gazebo 的界面走 XWayland。

## 5. 连通性测试

打开两个终端。

**终端 1：一条命令启动仿真**

```bash
./scripts/sim.sh               # 带 Gazebo 界面
HEADLESS=1 ./scripts/sim.sh    # 无界面，省 CPU
```

`sim.sh` 会依次启动 XRCE-DDS Agent 和 PX4 SITL + Gazebo。出现 `pxh>` 提示符
后，就可以在这里输入 PX4 命令。按 Ctrl+C 或输入 `shutdown`，会把三个进程全部
停掉。Agent 的日志在 `log/sim/agent.log`。

**终端 2：ROS 2**

```bash
ros2 topic list | grep fmu                          # 应列出约 65 个 /fmu/in/... 和 /fmu/out/... 话题
ros2 topic echo /fmu/out/vehicle_status_v1 --once
ros2 run px4_ros_com offboard_control               # 解锁，起飞到 5 米高度悬停
```

`pxh>` 提示符只在启动完成时打印一次，很快会被后续日志推上去，但 shell 一直在
等待输入。按一下回车就能看到新的提示符。也可以从任意终端用
`./scripts/px4cmd.sh <命令>` 发送 PX4 命令，例如
`./scripts/px4cmd.sh commander land`，效果与在 `pxh>` 中输入相同。

如果无人机起飞并悬停（`/fmu/out/vehicle_local_position_v1` 的 `z` 约为 -5，
NED 坐标系中负值表示向上），就说明 PX4、Gazebo、ROS 2 三者已经连通。之后可以
在 `pxh>` 中输入 `commander land` 让它降落。

## 6. 用 Wukong 自己的节点飞一次

`wukong_flight` 功能包的 `fly_mission` 节点完成起飞、按航点飞行、降落，结束后
自动退出。仿真启动后，在终端 2 中：

```bash
colcon build --symlink-install --packages-select wukong_flight   # 第一次或改动 setup.py 后

ros2 run wukong_flight fly_mission                    # 起飞到 5 米，向北飞 5 米，降落
ros2 run wukong_flight fly_mission --ros-args \
    --params-file install/wukong_flight/share/wukong_flight/config/square.yaml   # 飞一个 10 米正方形
ros2 run wukong_flight fly_mission --ros-args -p waypoints:="[20.0, 0.0, 8.0]"   # 自定义航点
```

航点是“北、东、高度”三个数一组，单位米，相对起飞点。正常结束时退出码为 0。

- 飞行中按 **Ctrl+C**：先刹车悬停，再自动降落，落地后退出。再按一次立即退出，
  由 PX4 的失效保护接管。
- 飞行中在 `pxh>` 里切换模式（例如 `commander mode auto:loiter`），节点会认为有人接管，
  立即停止发指令并退出。

可以连续运行多次，不需要重启仿真。每次都以当前所在位置为起飞点。

> **关于 `NAV_DLL_ACT`**：PX4 v1.17 的 `gz_x500` 机型把 `NAV_DLL_ACT`（数据链
> 丢失时的动作）设为 2（返航）。这样一来，没有 MAVLink 地面站（如 QGroundControl）
> 时，PX4 会报 `Preflight Fail: No connection to the GCS` 并拒绝解锁。仿真中
> 由 ROS 2 节点控制飞行，所以 `sim.sh` 在 PX4 启动后把它改为 0。这个参数必须在
> 启动后设置：用 `PX4_PARAM_NAV_DLL_ACT=0` 环境变量在启动时设置不会生效。
> **这一设置只用于仿真，真机必须保留数据链丢失保护。**

## 7. 常见问题

**Gazebo 三维视图一片灰色，看不到地面和无人机**

这是 VMware 3D 驱动跑 Gazebo 默认渲染引擎 ogre2 时的问题：只画得出背景色，
画不出任何物体。实测对比如下：

| 渲染方式 | 结果 | 界面 CPU 占用 |
|---|---|---|
| ogre2 + VMware 硬件加速 | 全灰 | — |
| ogre2 + 软件渲染（`LIBGL_ALWAYS_SOFTWARE=1`） | 正常 | 约 145% |
| ogre1 + 硬件加速 | 正常 | 约 10% |

因此 `sim.sh` 默认设置 `PX4_GZ_SIM_RENDER_ENGINE=ogre`。换到有独立显卡的机器
后，可以用 `PX4_GZ_SIM_RENDER_ENGINE=ogre2 ./scripts/sim.sh` 改回 ogre2。
M2 阶段加入相机传感器时，需要再验证 ogre1 下相机图像是否正常。

**Gazebo 窗口黑屏、崩溃，或报 OpenGL / ogre2 错误**

1. 运行 `glxinfo -B`，确认 OpenGL 版本在 3.3 以上，并且渲染器不是 `llvmpipe`。
2. 检查 `~/.profile` 里是否还有 `SVGA_VGPU10=0`。PX4 的安装脚本在 VMware 中
   会加上这一行，它会把 OpenGL 降到 2.1。`02_px4.sh` 会自动删掉它，删除后
   需要重新登录才生效。
3. 确认 VMware 的“加速 3D 图形”已经打开。
4. 实在不行，登录界面右下角选择 “Ubuntu on Xorg” 再试。

**`ros2 topic list` 里看不到 `/fmu/...` 话题**

- 确认 Agent 正在运行，端口为 8888。
- 确认当前终端已经加载了 `env.sh`：`echo $WUKONG_ROOT` 有输出。
- 确认 `px4_msgs` 的分支与 PX4 版本对应（见第 1 节）。两边对不上时，需要运行
  PX4 的消息转换节点，见 PX4 文档 *ROS 2 Message Translation Node*。

**colcon 编译 Python 包时报 setuptools 相关的警告或错误**

PX4 的安装脚本会用 pip 往 `~/.local` 安装一批 Python 包，其中可能包括较新的
setuptools，与 ROS 2 用 apt 安装的版本冲突。可以用
`pip list --user | grep -i setuptools` 检查；必要时执行
`pip uninstall --break-system-packages setuptools`，删掉用户目录里的那份。

**PX4 报 `gz_bridge: Task already running` 或 `failed to start and spawn model`**

上一次仿真的 Gazebo 服务器还在运行。`pgrep -af '^gz sim'` 查看，用
`pkill -f '^gz sim'` 结束后再启动。

**在没有终端的环境里运行 PX4（脚本、CI）**

`make px4_sitl` 会启动交互式的 `pxh>` shell。没有终端时，它会在 EOF 上空转，
日志可以在几分钟内涨到几百 MB。`sim.sh` 检测到没有终端时，会自动改用
`px4 -d`（daemon 模式）。

**QGroundControl**

M0 阶段用 ROS 2 节点控制无人机，暂时用不到 QGroundControl。需要时建议装在
Mac 宿主机上，再通过 UDP 连到虚拟机里的 PX4 SITL。
