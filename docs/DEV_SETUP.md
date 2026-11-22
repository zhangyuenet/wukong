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

# 5. 任务层的 Python 包（Claude API SDK），装在 ~/.local
./scripts/setup/05_python_deps.sh

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
./scripts/sim.sh                   # 带相机的 x500 + 房屋世界，带 Gazebo 界面
HEADLESS=1 ./scripts/sim.sh        # 无界面，省 CPU
./scripts/sim.sh gz_x500           # 不带相机的机型
WORLD=default ./scripts/sim.sh     # PX4 自带的空世界
```

`sim.sh` 会依次启动 XRCE-DDS Agent、PX4 SITL + Gazebo，机型带相机时还会启动
相机桥，把图像发到 ROS 2 话题 `/camera/image_raw`。出现 `pxh>` 提示符后，就可以
在这里输入 PX4 命令。按 Ctrl+C 或输入 `shutdown`，会把所有进程停掉。日志在
`log/sim/`。

仿真世界 `wukong_house` 在 [sim/worlds/](../sim/worlds/)，起飞点以北 15 米有一栋
房屋，南侧水槽里有落叶，北侧水槽是干净的。PX4 只从自己的目录加载世界文件，
`sim.sh` 会把本仓库的世界文件链接过去。

**相机需要显示器**：VMware 里用的 ogre1 渲染引擎只能通过 X 显示器（GLX）渲染，
即使 `HEADLESS=1` 也一样。在桌面的终端里运行没有问题；通过 SSH 或在没有
`DISPLAY` 的环境中运行时，Gazebo 会报 `Unable to open display` 并崩溃。这种情况
下先设置 `export DISPLAY=:0`（需要桌面已登录），或者改用不带相机的 `gz_x500`。

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

## 7. 用一句话完成任务（M1）

`wukong_mission` 功能包的 `run_task` 把一句话交给 Claude 规划成技能序列，飞完后
汇总照片判断结果。先把 Claude API 密钥写进仓库根目录的 `.env`（参考
`.env.example`，`.env` 不会被提交）：

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

仿真启动后，在终端 2 中：

```bash
colcon build --symlink-install            # 第一次需要编译 wukong_mission

ros2 run wukong_mission run_task "帮我检查一下面前这栋房子的水槽，需要清理的地方拍照片给我"
ros2 run wukong_mission run_task --offline "任意一句话"   # 不联网：固定计划 + 不判断照片
```

一次任务的过程：

1. 读取无人机当前位置。
2. **在地面上规划**：Claude 根据地图（[sim_house.yaml](../src/wukong_mission/config/sim_house.yaml)）
   把这句话变成技能序列。计划必须通过飞行前检查（高度、距离、与房屋保持 2 米
   以上），不通过就把错误反馈给 Claude 重试一次，仍不通过就**不起飞**。
3. **飞行**：按计划执行。每拍一张照片，就在后台线程里交给 Claude 判断，飞行
   不等待结果。
4. 降落后等待判断结果，写出报告。

照片和报告在 `log/missions/<时间>/`（`report.md` 给人看，`report.json` 给程序用）。
`log/` 不会被提交。

技能有：起飞、飞到某点（可以指定朝向某个点）、悬停、拍照、返航降落、原地降落。
飞行速度限制在 3 m/s。网络或 API 出错时：规划阶段出错就不起飞；判断阶段出错，
报告里这张照片标为“未判断”，飞行不受影响。

## 8. 检查不同房屋的水槽（M2）

M2 中无人机只知道房屋的大概位置。它先绕房屋飞一圈，用深度相机扫描，算出房屋
轮廓和屋檐线，再沿屋檐逐段拍照，每段照片单独交给 Claude 判断。

**仿真场景**在 [sim/scenarios/](../sim/scenarios/)，每个 YAML 文件描述一栋房屋：
位置、尺寸、朝向、屋顶坡度，以及每段水槽的状态（`clean` 干净、`leaves` 有落叶、
`blocked` 堵塞）。启动时 `sim.sh` 会生成三样东西，放在 `sim/generated/<场景>/`
（不提交）：

- 仿真世界文件；
- `truth.json`：真值，用来给报告打分；
- `map.yaml`：给无人机的粗略地图，只有房屋的大概位置（有误差）和一个肯定能
  包住房屋的半径。

```bash
SCENARIO=basic ./scripts/sim.sh          # 终端 1；默认机型 x500_depth（带深度相机）

ros2 run wukong_mission run_task --scenario basic "检查这栋房子的水槽，告诉我哪几段需要清理"
python3 sim/evaluate.py log/missions/<时间>/report.json sim/generated/basic/truth.json
```

一次任务的过程：

1. Claude 只理解要检查哪栋房屋、哪几个方向的水槽，不负责坐标。
2. 起飞前只规划环绕扫描：在粗略禁飞区（整个圆柱）外的 8 个站点各扫描一次深度。
3. 扫描完悬停，后台线程拟合房屋：最小面积外接矩形是轮廓，四条边里最低的两条
   是屋檐。然后生成逐段拍照的视点：距屋檐 4.5 米，比水槽高 0.6 米，正对该段
   中点；换到另一侧时沿房屋外的环形航线绕过去。
4. 新航点必须通过飞行前检查才会执行，这时禁飞区换成实测的房屋框。检查不通过
   就直接返航。
5. 每张照片按拍照时的位姿裁剪到对应的那一段（相邻段的部分调暗），再交给
   Claude 判断。
6. 报告按段列出结果，`report.json` 里带每段的坐标。`evaluate.py` 按位置把它和
   真值对应，逐段打分。

输出目录里还有 `survey_points.npy`（扫描得到的点云）和 `house_estimate.json`
（拟合结果），方便排查。

**关于高度**：PX4 的高度估计会随气压计漂移，在仿真中约 0.2–0.3 米。飞行和拟合
使用同一个高度坐标系，所以不受影响。拟合结果里的 `ground_level` 是从点云中测得
的地面高度，和实际比较时要减去它。

## 9. 复杂房屋（M3）

M3 的场景不告诉无人机房屋在哪里、是什么类型，地图里只有安全限制和搜索半径；
房屋类型（独立屋还是联排、几层、目标户的分界）由 Claude 看照片判断。`run_task`
会根据正在运行的仿真世界自动选择地图，不需要指定场景，命令和 M2 一样：

```bash
SCENARIO=two_storey ./scripts/sim.sh      # 两层独立屋，带车库和门廊
SCENARIO=row_middle ./scripts/sim.sh      # 五户联排，目标是中间户
SCENARIO=row_end ./scripts/sim.sh         # 四户联排，目标是端户（三面水槽）

ros2 run wukong_mission run_task "帮我检查一下旁边这栋房子所有的水槽，告诉我哪几段需要清理"
./scripts/eval_m2.sh two_storey row_end row_middle    # 批量打分
```

`--offline` 不联网时没有 Claude 判断房屋类型，改用仿真生成的真值
（`sim/generated/<场景>/truth.json`）代替，作用和离线时的假照片判断一样，只用于仿真。

一次任务的过程：

1. 在起飞点上方 6 米环视扫描，找出最近的建筑。
2. 对准房屋正面上离起飞点最近的位置（离两端至少 3 米），停稳后拍一张照片，画上
   每米一格的刻度尺和一条中线，由 Claude 判断是独立屋还是联排；联排还要读出中线所在
   那一户的分界。端户靠外的一端直接取整排的尽头。
   （照片比拍照请求晚几百毫秒到达，飞机还在转动时刻度尺会画偏，所以这张照片要像深度
   扫描一样等朝向误差在 1 度以内、静止 1 秒再拍。）
3. 逐站扫描：每扫描一次就重新估计建筑轮廓，只决定下一站。先在高处绕一圈，
   再降低绕一圈。联排只扫描目标户附近，需要时从屋顶上方越过。
4. 用高度图识别所有楼层的屋檐，规划逐段拍照的航线（绕开一切看到的东西），
   拍照、判断、写报告。

不用 Gazebo 也能检查逻辑：

```bash
python3 sim/perception_check.py two_storey     # 只检查屋檐识别
python3 sim/mission_check.py two_storey row_end --drift 0.4   # 整个任务，含位置漂移
```

`mission_check.py` 用光线投射代替深度相机，每个场景十几秒。

## 10. 常见问题

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

**`run_task` 报 “no vehicle position from PX4”，但 PX4 话题都在**

PX4 的位置估计没有接受 GPS：航向没有对准，因为磁场被判为“受到干扰”。用
`./scripts/px4cmd.sh listener estimator_status_flags` 检查，`cs_yaw_align` 为
False、`cs_mag_field_disturbed` 为 True 就是这个问题。

原因是 PX4 会在飞行中学习磁力计偏置，并在上锁时保存。在同一个 PX4 上仿真飞行
几十次后，偏置漂到了 0.25 高斯，约为地磁场的一半，修正后的磁场强度明显偏小。
`sim.sh` 现在每次启动时把 `CAL_MAG0_*OFF` 清零，并关闭 `SENS_MAG_AUTOCAL`。
这只用于仿真，真机必须使用真实的磁力计校准。

**解锁被拒，PX4 报 `Preflight Fail: Battery unhealthy`**

这是 PX4 SITL 的问题，出现过两次，都在带深度相机的仿真运行约 10 分钟、飞过几次
之后。现象是 PX4 里所有按定时器运行的模块同时停止，模拟电池和 `load_mon` 都不再
发布消息；由数据驱动的模块（传感器、位置估计）照常运行。电池状态超过 5 秒没有
更新，PX4 就判定电池故障，拒绝解锁。

用 `./scripts/px4cmd.sh uorb top -1 battery_status cpuload` 检查：两者的
RATE 都是 0，就是这个问题。重启仿真即可。

还没有找到触发条件：空转超过 10 分钟、连续飞 4 次都没有复现。要查清原因，需要
用 gdb 附加到 PX4 进程，而这台虚拟机限制了 ptrace，必须用 sudo。
`scripts/eval_m2.sh` 每个场景都重新启动仿真，不受影响。

**任务中途报 `-> land (lost vehicle state)`，但 PX4 一切正常**

`run_task` 超过 0.5 秒收不到 PX4 的状态或位置消息，就认为失去了联系。M3 中出现过
一次：PX4 的日志显示飞机一直在 offboard 模式下正常悬停，是 `run_task` 这一侧暂时
没有收到消息，原因还没查清。现在的处理：

- 中断不到 1 秒：保持最后的设定点，恢复后继续飞。
- 超过 1 秒：请 PX4 原地降落，并持续重发降落指令，直到重新收到消息、确认已经落地
  上锁；60 秒仍无消息就结束任务，由 PX4 的失效保护处置。报告里写明飞行没有完成，
  没拍到的段标为“未检查”。
- 日志里的 `no fresh vehicle state: status ... position ... old` 记录了哪种消息停了
  多久，再出现时用它查原因。

**磁盘被 PX4 日志写满**

PX4 SITL 的启动脚本默认设置 `SDLOG_MODE=1`，意思是从启动开始记录日志，直到
第一次上锁。仿真启动后如果一直不起飞，日志就会一直写下去，速度约 1 GB/小时。
仿真曾经空转两天，写出了一个 44 GB 的 `.ulg` 文件。

现在 `sim.sh` 在 PX4 启动后把 `SDLOG_MODE` 改为 0，并重启 logger，只在解锁期间
记录日志，每次飞行约 10 MB。日志在
`$PX4_DIR/build/px4_sitl_default/rootfs/log/`，超过 5 GB 时 `sim.sh` 启动时会提醒。
可以随时整个删除。

不用仿真时记得关掉，`pgrep -af 'px4|gz sim'` 可以检查是否还有残留进程。

**在没有终端的环境里运行 PX4（脚本、CI）**

`make px4_sitl` 会启动交互式的 `pxh>` shell。没有终端时，它会在 EOF 上空转，
日志可以在几分钟内涨到几百 MB。`sim.sh` 检测到没有终端时，会自动改用
`px4 -d`（daemon 模式）。

**QGroundControl**

M0 阶段用 ROS 2 节点控制无人机，暂时用不到 QGroundControl。需要时建议装在
Mac 宿主机上，再通过 UDP 连到虚拟机里的 PX4 SITL。
