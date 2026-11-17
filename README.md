# Wukong 悟空

**An open-source autonomous drone platform you can talk to.**
**一个可以用自然语言对话的开源无人机自主工作平台。**

[English](#english) | [中文](#中文)

---

## English

### What is Wukong?

Wukong is an open-source project that explores how a drone can understand a
plain-language request, plan the job, carry it out on its own, and report back.

The name comes from Sun Wukong, the Monkey King of Chinese mythology, who can
transform into whatever a situation calls for. The long-term idea is the same:
one flying platform that becomes a different tool for each task.

### First scenario: gutter inspection

The first target is a small, real household job:

> "Check the gutters on the house in front of you and see if they need
> cleaning. Send me photos of the spots that do."

The drone should then:

1. Take off and identify the nearest house.
2. Fly along the roofline and inspect the gutters section by section.
3. Decide from the images which sections need attention.
4. Save or send photos of those sections.
5. Return to the take-off point and land.

### Why this project exists

This is a learning-by-building project. A drone is an affordable and fun
carrier for the software problems that matter across robotics: perception,
localization, planning, safety, and the split between fast on-board control
and slower large-model reasoning. The work is done with AI assistance and
shared in the open.

### Approach

| Layer | Choice | Role |
|---|---|---|
| Flight control | PX4 | Attitude and position control, failsafes, return-to-launch |
| Robot middleware | ROS 2 | Perception, planning, mission logic on the companion computer |
| Simulation | Gazebo | Everything runs in simulation before it runs on hardware |
| Language and vision reasoning | Large models (cloud first) | Task planning and image assessment, never inside a real-time loop |

Design principles:

- **Simulation first.** Nothing flies for real until it works in simulation.
- **Safety comes from geometry, not recognition.** Obstacle avoidance relies on
  depth sensing and must work without any network connection.
- **Large models are non-blocking.** If the cloud is slow or unreachable, the
  drone holds position or returns home.
- **Privacy by default.** Flight logs, imagery, locations and credentials stay
  out of this repository.

### Status

Early planning. See [docs/PLAN.md](docs/PLAN.md) for the roadmap (in Chinese).

### Safety and regulations

This is experimental software. Flying a drone carries real risk to people and
property, and is regulated in most countries. Anyone using this code is
responsible for following local rules and for keeping a pilot ready to take
manual control at all times.

### License

Copyright 2026 Yue Zhang.

Licensed under the [Apache License, Version 2.0](LICENSE).

---

## 中文

### Wukong 是什么

Wukong 是一个开源项目，探索无人机如何听懂一句日常语言的请求，自己规划任务、
自主执行，并把结果汇报给人。

项目名取自中国神话中的孙悟空。悟空能随场景变化成所需的样子，这个项目的长期
设想也是如此：同一个飞行平台，在不同任务里变成不同的工具。

### 第一个场景：检查屋檐水槽

第一个目标是一件真实的家务小事：

> “帮我检查一下面前这栋房子的 gutter，看看是否需要清理。需要清理的地方拍照片给我。”

无人机接到指令后应当：

1. 起飞，确认距离自己最近的房屋。
2. 沿屋檐飞行，逐段检查水槽。
3. 根据图像判断哪些位置需要清理。
4. 把这些位置的照片保存或发送给用户。
5. 回到起飞点并降落。

### 为什么做这个项目

这是一个边做边学的项目。无人机成本可控、有趣，又能承载机器人领域共通的软件
问题：感知、定位、规划、安全，以及机载快速控制与大模型慢速推理之间的分工。
整个过程在 AI 的协助下完成，并公开分享。

### 技术路线

| 层 | 选择 | 作用 |
|---|---|---|
| 飞行控制 | PX4 | 姿态与位置控制、失效保护、自动返航 |
| 机器人中间件 | ROS 2 | 机载电脑上的感知、规划和任务逻辑 |
| 仿真 | Gazebo | 所有功能先在仿真中跑通，再上真机 |
| 语言与视觉推理 | 大模型（先用云端） | 任务规划和图像判断，不进入任何实时环路 |

设计原则：

- **仿真优先。** 仿真里没跑通的功能，不上真机。
- **安全靠几何，不靠识别。** 避障依赖深度测量，断网时也必须有效。
- **大模型不阻塞。** 云端变慢或失联时，无人机悬停或返航。
- **默认保护隐私。** 飞行日志、影像、位置和密钥不进入本仓库。

### 当前状态

处于早期规划阶段。路线图见 [docs/PLAN.md](docs/PLAN.md)。

### 安全与法规

本项目是实验性软件。无人机飞行对人身和财产有实际风险，并且在大多数国家受到
监管。使用本项目代码的人需要自行遵守当地法规，并始终安排一名可以随时手动接管
的操作员。

### 许可证

Copyright 2026 Yue Zhang.

本项目采用 [Apache License 2.0](LICENSE) 许可证。
