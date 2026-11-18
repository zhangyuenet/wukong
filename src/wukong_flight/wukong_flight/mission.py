# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Takeoff -> fly to waypoints -> land, as a pure state machine.

This module has no ROS imports on purpose: the node feeds it a VehicleState
snapshot on every tick and executes the Output it returns. That keeps the flight
logic testable without a simulator (see test/test_mission.py).
本模块刻意不依赖 ROS：节点每个周期把飞机状态传进来，再执行返回的输出。
这样飞行逻辑不需要仿真器也能测试。

Frames: PX4 local NED (x north, y east, z down, metres). Mission targets are
given relative to the takeoff point as (north, east, altitude-up), and converted
here, so that nobody has to write a negative number to mean "up".
坐标系：PX4 本地 NED（z 向下）。航点用相对起飞点的（北, 东, 高度）表示，
高度向上为正，在本模块内转换成 NED。
"""

from dataclasses import dataclass, field
import enum
import math
from typing import List, Optional, Sequence, Tuple


class Phase(enum.Enum):
    WAIT_READY = 'wait_ready'   # waiting for PX4 to report a valid, landed vehicle
    PRESTREAM = 'prestream'     # streaming setpoints; PX4 requires this before offboard
    ARMING = 'arming'           # asked for offboard + arm, waiting for both
    TAKEOFF = 'takeoff'         # climbing straight up above the takeoff point
    GOTO = 'goto'               # flying to waypoint [index]
    HOLD = 'hold'               # hovering at waypoint [index]
    BRAKE = 'brake'             # stopping where we are before landing
    LAND = 'land'               # PX4 auto-land, we only watch
    DONE = 'done'               # landed and disarmed
    ABORTED = 'aborted'         # stopped commanding; PX4 or a pilot is in charge


class Command(enum.Enum):
    ENGAGE_OFFBOARD = 'engage_offboard'
    ARM = 'arm'
    DISARM = 'disarm'
    LAND = 'land'


@dataclass(frozen=True)
class VehicleState:
    """What the state machine needs to know about the vehicle, in PX4 local NED."""

    t: float                     # seconds, monotonic
    connected: bool              # status and position received recently
    position_valid: bool
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0         # radians
    speed: float = 0.0           # m/s, 3D
    armed: bool = False
    offboard: bool = False       # nav_state == OFFBOARD
    landing: bool = False        # nav_state == AUTO_LAND
    landed: bool = True
    # PX4 pre_flight_checks_pass. It also covers the *current* mode, so it is
    # false while PX4 sits in offboard mode without setpoints (e.g. right after a
    # previous mission). Only used to explain a failed arm, never as a gate.
    # 它也检查当前模式：上一次任务结束后 PX4 停在 offboard 模式、没有设定点时为 false。
    preflight_ok: bool = False


@dataclass(frozen=True)
class Setpoint:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class Output:
    setpoint: Optional[Setpoint] = None   # None: do not stream offboard setpoints
    commands: List[Command] = field(default_factory=list)
    events: List[str] = field(default_factory=list)   # human-readable log lines


@dataclass(frozen=True)
class MissionConfig:
    takeoff_altitude: float = 5.0          # m above the takeoff point
    # (north, east, altitude) in metres, relative to the takeoff point
    waypoints: Tuple[Tuple[float, float, float], ...] = ()
    hold_time: float = 3.0                 # s to hover at each waypoint
    tolerance: float = 0.5                 # m, distance that counts as "reached"
    prestream_time: float = 1.0            # s of setpoints before asking for offboard
    command_retry: float = 1.0             # s between repeated mode/arm/land commands
    ready_timeout: float = 60.0            # s to wait for PX4 to be ready
    arm_timeout: float = 10.0
    leg_timeout: float = 60.0              # s allowed per takeoff/goto leg
    brake_speed: float = 0.3               # m/s, slow enough to start landing
    brake_timeout: float = 5.0             # s, land anyway after this long

    def __post_init__(self) -> None:
        if self.takeoff_altitude <= 0:
            raise ValueError('takeoff_altitude must be positive')
        if self.tolerance <= 0:
            raise ValueError('tolerance must be positive')
        for wp in self.waypoints:
            if len(wp) != 3 or wp[2] <= 0:
                raise ValueError(f'bad waypoint {wp}: need (north, east, altitude>0)')


def waypoints_from_flat(values: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    """[n1, e1, a1, n2, e2, a2, ...] -> ((n1, e1, a1), ...). ROS params have no nested lists."""
    if len(values) % 3 != 0:
        raise ValueError('waypoints must have 3 numbers per point: north, east, altitude')
    return tuple(
        (float(values[i]), float(values[i + 1]), float(values[i + 2]))
        for i in range(0, len(values), 3))


FLYING_PHASES = (Phase.TAKEOFF, Phase.GOTO, Phase.HOLD, Phase.BRAKE)


class Mission:
    """Takeoff, visit each waypoint, land. Call step() at a fixed rate (>= 10 Hz)."""

    def __init__(self, config: MissionConfig) -> None:
        self.config = config
        self.phase = Phase.WAIT_READY
        self.index = 0                      # current waypoint
        self._phase_start: Optional[float] = None
        self._last_command_t = -math.inf
        self._home: Optional[Setpoint] = None
        self._land_requested = False
        self._brake_point: Optional[Setpoint] = None
        self.failure: Optional[str] = None  # why the mission did not complete

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.ABORTED)

    def request_land(self) -> None:
        """Ask to stop the mission and land (e.g. on Ctrl+C). Applied on the next step."""
        self._land_requested = True

    # -- helpers ------------------------------------------------------------

    def _enter(self, phase: Phase, s: VehicleState, out: Output, why: str = '') -> None:
        self.phase = phase
        self._phase_start = s.t
        self._last_command_t = -math.inf
        label = phase.value + (f' {self.index + 1}' if phase in (Phase.GOTO, Phase.HOLD) else '')
        out.events.append(f'-> {label}' + (f' ({why})' if why else ''))

    def _elapsed(self, s: VehicleState) -> float:
        return s.t - (self._phase_start if self._phase_start is not None else s.t)

    def _retry_due(self, s: VehicleState) -> bool:
        if s.t - self._last_command_t >= self.config.command_retry:
            self._last_command_t = s.t
            return True
        return False

    def _abort(self, s: VehicleState, out: Output, why: str) -> None:
        self.failure = why
        self._enter(Phase.ABORTED, s, out, why)

    def target(self) -> Setpoint:
        """Setpoint for the current phase, in PX4 local NED."""
        assert self._home is not None
        h = self._home
        if self.phase in (Phase.GOTO, Phase.HOLD):
            north, east, alt = self.config.waypoints[self.index]
            return Setpoint(h.x + north, h.y + east, h.z - alt, h.yaw)
        if self.phase == Phase.BRAKE:
            assert self._brake_point is not None
            return self._brake_point
        if self.phase == Phase.TAKEOFF:
            return Setpoint(h.x, h.y, h.z - self.config.takeoff_altitude, h.yaw)
        return h

    @staticmethod
    def _distance(s: VehicleState, sp: Setpoint) -> float:
        return math.dist((s.x, s.y, s.z), (sp.x, sp.y, sp.z))

    def _brake(self, s: VehicleState, out: Output, why: str) -> None:
        """
        Hold the current position until the vehicle has stopped, then land.

        PX4's land mode descends wherever the vehicle is, but keeps its momentum:
        in simulation a stop at 7 m/s drifted 30 m before touchdown.
        PX4 的降落模式不会先刹车，7 m/s 时中断会再滑出约 30 米，所以先悬停再降落。
        """
        assert self._home is not None
        self._brake_point = Setpoint(s.x, s.y, s.z, self._home.yaw)
        self._enter(Phase.BRAKE, s, out, why)

    def _next_leg(self, s: VehicleState, out: Output) -> None:
        """After takeoff or a hold: go to the next waypoint, or land."""
        if self.phase == Phase.HOLD:
            self.index += 1
        if self.index < len(self.config.waypoints):
            self._enter(Phase.GOTO, s, out)
        else:
            self._enter(Phase.LAND, s, out, 'mission complete')

    # -- main loop ----------------------------------------------------------

    def step(self, s: VehicleState) -> Output:
        out = Output()
        if self._phase_start is None:
            self._phase_start = s.t
        cfg = self.config

        # Safety checks while we are the one flying. 飞行中的安全检查。
        if self.phase in FLYING_PHASES:
            if not s.armed:
                self._abort(s, out, 'vehicle disarmed')
            elif not s.offboard:
                # A pilot or a PX4 failsafe switched modes: never fight it.
                # 遥控接管或 PX4 失效保护切换了模式：不要抢回控制权。
                self._abort(s, out, 'left offboard mode, PX4 or pilot in control')
            elif not s.connected or not s.position_valid:
                self._enter(Phase.LAND, s, out, 'lost vehicle state')
            elif self.phase == Phase.BRAKE:
                pass
            elif self._land_requested:
                self._brake(s, out, 'land requested')
            elif self._elapsed(s) > cfg.leg_timeout and self.phase != Phase.HOLD:
                self._brake(s, out, 'leg timed out')
        elif self._land_requested and self.phase in (
                Phase.WAIT_READY, Phase.PRESTREAM, Phase.ARMING):
            if s.armed:
                out.commands.append(Command.DISARM)
            self._abort(s, out, 'cancelled before takeoff')

        if self.phase == Phase.WAIT_READY:
            # PX4 runs its own checks when we ask it to arm. PX4 解锁时自己会做检查。
            if s.connected and s.position_valid and not s.armed and s.landed:
                self._home = Setpoint(s.x, s.y, s.z, s.heading)
                self._enter(Phase.PRESTREAM, s, out,
                            f'home at x={s.x:.1f} y={s.y:.1f} z={s.z:.1f}')
            elif self._elapsed(s) > cfg.ready_timeout:
                self._abort(s, out, 'vehicle not ready (need PX4 connected, landed, '
                                    'disarmed, valid position)')

        elif self.phase == Phase.PRESTREAM:
            if self._elapsed(s) >= cfg.prestream_time:
                self._enter(Phase.ARMING, s, out)

        elif self.phase == Phase.ARMING:
            if s.armed and s.offboard:
                self._enter(Phase.TAKEOFF, s, out)
            elif self._elapsed(s) > cfg.arm_timeout:
                if s.armed:
                    out.commands.append(Command.DISARM)
                hint = '' if s.preflight_ok else ', PX4 preflight checks failing'
                self._abort(s, out, f'could not arm in offboard mode{hint}')
            elif self._retry_due(s):
                if not s.offboard:
                    out.commands.append(Command.ENGAGE_OFFBOARD)
                if not s.armed:
                    out.commands.append(Command.ARM)

        elif self.phase == Phase.TAKEOFF:
            if self._distance(s, self.target()) < cfg.tolerance:
                self._next_leg(s, out)

        elif self.phase == Phase.GOTO:
            if self._distance(s, self.target()) < cfg.tolerance:
                self._enter(Phase.HOLD, s, out)

        elif self.phase == Phase.HOLD:
            if self._elapsed(s) >= cfg.hold_time:
                self._next_leg(s, out)

        elif self.phase == Phase.BRAKE:
            if s.speed < cfg.brake_speed:
                self._enter(Phase.LAND, s, out, f'stopped, speed {s.speed:.1f} m/s')
            elif self._elapsed(s) > cfg.brake_timeout:
                self._enter(Phase.LAND, s, out, f'brake timed out at {s.speed:.1f} m/s')

        if self.phase == Phase.LAND:
            if not s.armed and s.landed:
                self._enter(Phase.DONE, s, out, 'landed and disarmed')
            elif not s.landing and s.armed and self._retry_due(s):
                out.commands.append(Command.LAND)

        # Keep streaming setpoints in every phase where offboard is (or will be) active.
        # 在需要 offboard 的阶段持续发送设定点，PX4 收不到会触发失效保护。
        if self.phase in (Phase.PRESTREAM, Phase.ARMING) + FLYING_PHASES:
            out.setpoint = self.target()
        return out
