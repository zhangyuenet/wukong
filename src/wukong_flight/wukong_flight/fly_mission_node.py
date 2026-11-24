# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
ROS 2 node: take off, fly to the configured waypoints, land, then exit.

    ros2 run wukong_flight fly_mission
    ros2 run wukong_flight fly_mission --ros-args \
        --params-file $(ros2 pkg prefix wukong_flight)/share/wukong_flight/config/square.yaml

Ctrl+C during flight brakes and lands; a second Ctrl+C exits at once.
飞行中按 Ctrl+C 会先刹车再降落；再按一次立即退出。
"""

import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .mission import MissionConfig, PlanRejected, waypoint_plan, waypoints_from_flat
from .mission_node import MissionNode, spin_mission
from .skills import describe


def _read_params():
    """Read parameters with a throwaway node, so MissionNode gets a ready plan."""
    tmp = Node('fly_mission')
    p = tmp.declare_parameter
    plan = waypoint_plan(
        p('takeoff_altitude', 5.0).value,
        # north, east, altitude triplets relative to the takeoff point
        waypoints_from_flat(p('waypoints', [5.0, 0.0, 5.0]).value),
        p('hold_time', 3.0).value)
    config = MissionConfig(tolerance=p('tolerance', 0.5).value,
                           max_speed=p('max_speed', 3.0).value)
    tmp.destroy_node()
    return plan, config


def main(args: Optional[List[str]] = None) -> int:
    # Handle SIGINT ourselves so we can still publish a land command after Ctrl+C.
    # 自己处理 Ctrl+C，这样按下后仍能发出降落指令。
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        plan, config = _read_params()
        try:
            node = MissionNode(plan, config, name='fly_mission')
        except PlanRejected as e:
            print(f'[fly_mission] {e}', file=sys.stderr)
            return 2
        for i, skill in enumerate(plan, 1):
            node.get_logger().info(f'plan {i}: {describe(skill)}')
        ok = spin_mission(node)
        node.destroy_node()
        return 0 if ok else 1
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
