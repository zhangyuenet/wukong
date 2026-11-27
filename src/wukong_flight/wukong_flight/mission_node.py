# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
A ROS 2 node that flies one checked plan, and the loop that runs it.

Used by `fly_mission` (waypoints from parameters) and by wukong_mission's
`run_task` (a plan written by a large model).
fly_mission 和 wukong_mission 的 run_task 都用它来执行计划。

Ctrl+C during flight brakes and lands; a second Ctrl+C exits at once
(PX4 then handles the loss of offboard setpoints with its own failsafe).
飞行中按 Ctrl+C 会先刹车再降落；再按一次立即退出，由 PX4 的失效保护接管。
"""

import os
import re
import threading
from typing import Callable, Optional, Sequence, Tuple, Union

import rclpy
from rclpy.node import Node

from .mission import Mission, MissionConfig, Phase, VehicleState
from .px4_link import Px4Link
from .skills import Decide, Limits, Skill, TakePhoto

# Vehicle pose when a scan or decision is requested: (north, east, altitude, heading,
# roll, pitch), in plan coordinates; angles in radians, heading 0 north, pi/2 east.
Pose = Tuple[float, float, float, float, float, float]
# (label, pose, done): capture a depth frame, then call done(ok). Must not block.
ScanCallback = Callable[[str, Pose, Callable[[bool], None]], None]
# (label, pose) -> (skills or None, new limits or None, note). Runs in a worker
# thread, so it may take seconds (perception); the vehicle holds position meanwhile.
DecideCallback = Callable[[str, Pose], Tuple[Optional[list], Optional[Limits], str]]
# (label, path, pose when the photo was requested)
PhotoCallback = Callable[[str, str, Pose], None]


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_')[:40] or 'photo'


class MissionNode(Node):

    def __init__(self, plan: Sequence[Skill], config: MissionConfig = MissionConfig(),
                 limits: Limits = Limits(), output_dir: str = '.',
                 on_photo: Optional[PhotoCallback] = None, name: str = 'mission',
                 rate_hz: float = 20.0, frame: str = 'takeoff',
                 start: Optional[Tuple[float, float]] = None,
                 on_scan: Optional[ScanCallback] = None,
                 on_decide: Optional[DecideCallback] = None) -> None:
        super().__init__(name)
        # Raises PlanRejected before anything is sent to PX4.
        self.mission = Mission(plan, config, limits, frame, start)
        self.link = Px4Link(self)
        self.output_dir = output_dir
        self.on_photo = on_photo
        self.on_scan = on_scan
        self.on_decide = on_decide
        self._decide_thread: Optional[threading.Thread] = None
        self.camera = None
        # A Decide skill may add photos later, so have the camera ready.
        if any(isinstance(s, (TakePhoto, Decide)) for s in plan):
            from .camera import Camera   # Pillow only needed when taking photos
            self.camera = Camera(self)
        self._last_report = 0.0
        self._had_contact = False
        self.create_timer(1.0 / rate_hz, self._tick)

    def _tick(self) -> None:
        if self.mission.finished:
            return
        state = self.link.state()
        if self._had_contact and not (state.connected and state.position_valid):
            # Say which messages stopped, to find out why. 记录哪些消息停了，便于查原因。
            self.get_logger().warn(f'no fresh vehicle state: {self.link.describe_ages()}')
        self._had_contact = state.connected and state.position_valid
        out = self.mission.step(state)
        if out.setpoint is not None:
            self.link.send_setpoint(out.setpoint)
        for event in out.events:
            # rclpy pins each logging call site to one severity, so use two call sites.
            if self.mission.phase == Phase.ABORTED:
                self.get_logger().error(event)
            else:
                self.get_logger().info(event)
        for command in out.commands:
            self.get_logger().info(f'command: {command.value}')
            self.link.send_command(command)
        if out.photo is not None:
            self._take_photo(out.photo, state)
        if out.scan is not None:
            self._scan(out.scan, state)
        if out.decide is not None:
            self._decide(out.decide, state)

        if state.t - self._last_report >= 2.0 and state.connected:
            self._last_report = state.t
            self.get_logger().info(
                f'[{self.mission.phase.value}] '
                f'pos N={state.x:.1f} E={state.y:.1f} alt={-state.z:.1f} '
                f'speed={state.speed:.1f} armed={state.armed} offboard={state.offboard}')

    def _pose(self, state) -> Pose:
        return (*self.mission.plan_position(state), state.heading, state.roll, state.pitch)

    def pose_at(self, stamp_us: float) -> Union[Pose, str]:
        """
        Return the vehicle's pose in plan coordinates at a camera frame's stamp (us), or why not.

        Not before takeoff (no home yet), nor when the moment cannot be told
        well (see Px4Link.pose_at). 起飞前（还没有起点）或成像时刻的位姿无法确定时，
        返回原因。
        """
        if not self.mission.has_home:
            return 'before takeoff'
        p = self.link.pose_at(stamp_us)
        if isinstance(p, str):
            return p
        x, y, z, heading, roll, pitch = p
        s = VehicleState(t=0.0, connected=True, position_valid=True, x=x, y=y, z=z)
        return (*self.mission.plan_position(s), heading, roll, pitch)

    def _scan(self, label: str, state) -> None:
        if self.on_scan is None:
            self.mission.scan_done(False)
            return
        self.on_scan(label, self._pose(state), self.mission.scan_done)

    def _decide(self, label: str, state) -> None:
        if self.on_decide is None:
            self.mission.decide_done(None, why='no decision maker')
            return
        # Plan from the hold point: the check starts there. 从目标点开始规划，检查也从那里开始。
        pose = (*self.mission.hold_position(state), state.heading, state.roll, state.pitch)

        def work() -> None:
            try:
                skills, limits, note = self.on_decide(label, pose)
            except Exception as e:   # noqa: B902 - any failure: insert nothing, fly on
                self.get_logger().error(f'decide "{label}" failed: {e!r}')
                skills, limits, note = None, None, f'error: {e}'
            self.mission.decide_done(skills, limits, note)

        self.get_logger().info(f'deciding "{label}" (holding position)...')
        self._decide_thread = threading.Thread(target=work, name='decide', daemon=True)
        self._decide_thread.start()

    def _take_photo(self, label: str, state) -> None:
        pose = self._pose(state)
        if self.camera is None:
            self.mission.photo_done(False)
            return
        n = len(self.mission.photos) + 1
        path = os.path.join(self.output_dir, f'photo_{n:02d}_{_slug(label)}.jpg')

        def done(ok: bool, info: str) -> None:
            if ok:
                self.get_logger().info(f'photo "{label}" saved: {info}')
                if self.on_photo is not None:
                    self.on_photo(label, info, pose)
            else:
                self.get_logger().error(f'photo "{label}" failed: {info}')
            self.mission.photo_done(ok)

        self.camera.capture(path, done)


def spin_mission(node: MissionNode) -> bool:
    """
    Spin until the mission finishes. Returns True if it landed as planned.

    Call rclpy.init(signal_handler_options=SignalHandlerOptions.NO) first, so
    that Ctrl+C reaches this loop and the node can still command a landing.
    先用 SignalHandlerOptions.NO 初始化 rclpy，Ctrl+C 才能到达这里，节点才能继续降落。
    """
    interrupted = False
    try:
        while rclpy.ok() and not node.mission.finished:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except KeyboardInterrupt:
                if interrupted:
                    node.get_logger().warn('second Ctrl+C, exiting without waiting')
                    break
                interrupted = True
                node.get_logger().warn('Ctrl+C: landing; press again to exit immediately')
                node.mission.request_land()
    finally:
        if node.mission.failure:
            node.get_logger().error(f'mission failed: {node.mission.failure}')
        elif node.mission.phase == Phase.DONE:
            node.get_logger().info('mission complete')
    return node.mission.phase == Phase.DONE and not node.mission.failure
