# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
ROS 2 node: take off, fly to the configured waypoints, land, then exit.

    ros2 run wukong_flight fly_mission
    ros2 run wukong_flight fly_mission --ros-args \
        --params-file $(ros2 pkg prefix wukong_flight)/share/wukong_flight/config/square.yaml

Ctrl+C during flight switches PX4 to auto-land; a second Ctrl+C exits at once
(PX4 then handles the loss of offboard setpoints with its own failsafe).
飞行中按 Ctrl+C 会让 PX4 自动降落；再按一次立即退出，由 PX4 的失效保护接管。
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .mission import Mission, MissionConfig, Phase, waypoints_from_flat
from .px4_link import Px4Link


class FlyMissionNode(Node):

    def __init__(self) -> None:
        super().__init__('fly_mission')
        p = self.declare_parameter
        config = MissionConfig(
            takeoff_altitude=p('takeoff_altitude', 5.0).value,
            # north, east, altitude triplets relative to the takeoff point
            waypoints=waypoints_from_flat(p('waypoints', [5.0, 0.0, 5.0]).value),
            hold_time=p('hold_time', 3.0).value,
            tolerance=p('tolerance', 0.5).value,
            leg_timeout=p('leg_timeout', 60.0).value,
        )
        rate_hz = p('rate_hz', 20.0).value

        self.mission = Mission(config)
        self.link = Px4Link(self)
        self._last_report = 0.0
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f'mission: takeoff to {config.takeoff_altitude} m, '
            f'{len(config.waypoints)} waypoint(s) {list(config.waypoints)}, then land')

    def _tick(self) -> None:
        if self.mission.finished:
            return
        state = self.link.state()
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

        if state.t - self._last_report >= 2.0 and state.connected:
            self._last_report = state.t
            self.get_logger().info(
                f'[{self.mission.phase.value}] '
                f'pos N={state.x:.1f} E={state.y:.1f} alt={-state.z:.1f} '
                f'armed={state.armed} offboard={state.offboard}')


def main(args=None) -> int:
    # Handle SIGINT ourselves so we can still publish a land command after Ctrl+C.
    # 自己处理 Ctrl+C，这样按下后仍能发出降落指令。
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = FlyMissionNode()
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
        ok = node.mission.phase == Phase.DONE
        if node.mission.failure:
            node.get_logger().error(f'mission failed: {node.mission.failure}')
        elif ok:
            node.get_logger().info('mission complete')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
