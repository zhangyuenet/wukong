# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Thin ROS 2 wrapper around the PX4 uXRCE-DDS topics.

Everything that knows PX4 topic names, QoS and message fields lives here, so the
rest of Wukong talks in terms of VehicleState / Setpoint / Command.
所有与 PX4 话题名、QoS、消息字段有关的细节都集中在这里。
"""

import math
from typing import Union

from px4_msgs.msg import (OffboardControlMode, TimesyncStatus, TrajectorySetpoint,
                          VehicleAttitude, VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mission import Command, Setpoint, VehicleState
from .pose_history import euler_from_quaternion, Pose6, PoseHistory

# QoS that matches the PX4 uXRCE-DDS bridge.
PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# PX4 sends position at ~100 Hz and status at >= 1 Hz; older than this means lost.
POSITION_STALE_S = 0.5
STATUS_STALE_S = 2.0


def px4_topic(direction: str, name: str, msg_type: type) -> str:
    """
    Return e.g. /fmu/out/vehicle_status_v1.

    PX4 >= 1.16 appends _v<N> to topics whose message has MESSAGE_VERSION > 0.
    PX4 1.16 起，MESSAGE_VERSION 大于 0 的消息，话题名会带 _v<N> 后缀。
    """
    version = getattr(msg_type, 'MESSAGE_VERSION', 0)
    suffix = f'_v{version}' if version else ''
    return f'/fmu/{direction}/{name}{suffix}'


class Px4Link:
    """Subscribes to vehicle state and publishes offboard setpoints and commands."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._status: VehicleStatus | None = None
        self._position: VehicleLocalPosition | None = None
        self._landed = True
        self._roll = 0.0
        self._pitch = 0.0
        self._status_t = -1.0
        self._position_t = -1.0
        # Recent poses by time, for depth frames taken in flight. 最近的位姿，按时间，
        # 用于飞行中的深度帧。
        self.history = PoseHistory()

        node.create_subscription(
            VehicleStatus, px4_topic('out', 'vehicle_status', VehicleStatus),
            self._on_status, PX4_QOS)
        node.create_subscription(
            VehicleLocalPosition,
            px4_topic('out', 'vehicle_local_position', VehicleLocalPosition),
            self._on_position, PX4_QOS)
        node.create_subscription(
            VehicleLandDetected,
            px4_topic('out', 'vehicle_land_detected', VehicleLandDetected),
            self._on_land_detected, PX4_QOS)
        node.create_subscription(
            VehicleAttitude, px4_topic('out', 'vehicle_attitude', VehicleAttitude),
            self._on_attitude, PX4_QOS)
        node.create_subscription(
            TimesyncStatus, px4_topic('out', 'timesync_status', TimesyncStatus),
            self._on_timesync, PX4_QOS)

        self._mode_pub = node.create_publisher(
            OffboardControlMode,
            px4_topic('in', 'offboard_control_mode', OffboardControlMode), PX4_QOS)
        self._setpoint_pub = node.create_publisher(
            TrajectorySetpoint,
            px4_topic('in', 'trajectory_setpoint', TrajectorySetpoint), PX4_QOS)
        self._command_pub = node.create_publisher(
            VehicleCommand, px4_topic('in', 'vehicle_command', VehicleCommand), PX4_QOS)

    # -- input --------------------------------------------------------------

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _now_us(self) -> int:
        return self._node.get_clock().now().nanoseconds // 1000

    def _on_status(self, msg: VehicleStatus) -> None:
        self._status, self._status_t = msg, self._now()

    def _on_position(self, msg: VehicleLocalPosition) -> None:
        self._position, self._position_t = msg, self._now()
        if msg.xy_valid and msg.z_valid:
            self.history.add_position(msg.timestamp_sample, msg.x, msg.y, msg.z)

    def _on_land_detected(self, msg: VehicleLandDetected) -> None:
        self._landed = bool(msg.landed)

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        self._roll, self._pitch, _ = euler_from_quaternion(msg.q)   # Hamilton, FRD -> NED
        self.history.add_attitude(msg.timestamp_sample, msg.q)

    def _on_timesync(self, msg: TimesyncStatus) -> None:
        # Only the DDS link's own: a ground station's MAVLink time sync publishes
        # here too, with its own offset. 只用 DDS 链路自己的；地面站的 MAVLink 时间
        # 同步也发到这里，偏移不同。
        if msg.source_protocol == TimesyncStatus.SOURCE_PROTOCOL_DDS:
            self.history.set_offset(msg.estimated_offset, msg.timestamp)

    def pose_at(self, stamp_us: float) -> Union[Pose6, str]:
        """
        Return the pose (PX4 local NED, heading, roll, pitch) at a frame's stamp, or why not.

        See pose_history for the clocks. 时钟见 pose_history。
        """
        p = self.history.pose_at(stamp_us)
        return self.history.why_not if p is None else p

    def describe_ages(self) -> str:
        """How old the last status and position messages are, for the log."""
        t = self._now()

        def age(msg, stamp):
            return 'never' if msg is None else f'{t - stamp:.2f} s'
        pos = self._position
        valid = '' if pos is None else f', xy_valid={pos.xy_valid} z_valid={pos.z_valid}'
        return (f'status {age(self._status, self._status_t)}, '
                f'position {age(pos, self._position_t)} old{valid}')

    def state(self) -> VehicleState:
        t = self._now()
        st, pos = self._status, self._position
        connected = (st is not None and pos is not None
                     and t - self._status_t < STATUS_STALE_S
                     and t - self._position_t < POSITION_STALE_S)
        if not connected:
            return VehicleState(t=t, connected=False, position_valid=False)
        return VehicleState(
            t=t,
            connected=True,
            position_valid=bool(pos.xy_valid and pos.z_valid),
            x=float(pos.x), y=float(pos.y), z=float(pos.z),
            heading=float(pos.heading),
            speed=math.sqrt(pos.vx ** 2 + pos.vy ** 2 + pos.vz ** 2),
            armed=st.arming_state == VehicleStatus.ARMING_STATE_ARMED,
            offboard=st.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD,
            landing=st.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
            landed=self._landed,
            preflight_ok=bool(st.pre_flight_checks_pass),
            roll=self._roll, pitch=self._pitch,
        )

    # -- output -------------------------------------------------------------

    def send_setpoint(self, sp: Setpoint) -> None:
        """Position setpoint plus the offboard heartbeat. Must be sent at >= 2 Hz."""
        now = self._now_us()
        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = now
        self._mode_pub.publish(mode)

        msg = TrajectorySetpoint()
        msg.position = [sp.x, sp.y, sp.z]
        nan = float('nan')   # NaN = "not controlled" for PX4
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = sp.yaw
        msg.yawspeed = nan
        msg.timestamp = now
        self._setpoint_pub.publish(msg)

    def send_command(self, command: Command) -> None:
        if command == Command.ENGAGE_OFFBOARD:
            # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1, PX4 main mode OFFBOARD=6
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        elif command == Command.ARM:
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        elif command == Command.DISARM:
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        elif command == Command.LAND:
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        else:
            raise ValueError(command)

    def _vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._now_us()
        self._command_pub.publish(msg)
