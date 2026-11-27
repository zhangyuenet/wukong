# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Execute a plan of skills (see skills.py), as a pure state machine.

This module has no ROS imports on purpose: the node feeds it a VehicleState
snapshot on every tick and executes the Output it returns. That keeps the flight
logic testable without a simulator (see test/test_mission.py).
本模块刻意不依赖 ROS：节点每个周期把飞机状态传进来，再执行返回的输出。
这样飞行逻辑不需要仿真器也能测试。

Frames: PX4 local NED (x north, y east, z down, metres). Plans are written
relative to the takeoff point as (north, east, altitude-up) and converted here.
坐标系：PX4 本地 NED（z 向下）。计划用相对起飞点的（北, 东, 高度）表示，在这里换算。
"""

from dataclasses import dataclass, field, replace
import enum
import math
from typing import List, Optional, Sequence, Tuple

from .skills import (check_plan, Decide, describe, GoTo, Hover, Land, Limits, ReturnHome,
                     Scan, Skill, Takeoff, TakePhoto, TERMINAL)


class Phase(enum.Enum):
    WAIT_READY = 'wait_ready'   # waiting for PX4 to report a valid, landed vehicle
    PRESTREAM = 'prestream'     # streaming setpoints; PX4 requires this before offboard
    ARMING = 'arming'           # asked for offboard + arm, waiting for both
    RUNNING = 'running'         # executing plan[index]
    BRAKE = 'brake'             # stopping where we are before landing
    LAND = 'land'               # ask PX4 to land until it does, then only watch
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
    heading: float = 0.0         # radians, 0 = north, +pi/2 = east
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
    roll: float = 0.0            # radians, body FRD
    pitch: float = 0.0           # radians, nose up positive


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
    photo: Optional[str] = None           # take a photo now, with this label
    scan: Optional[str] = None            # capture a depth frame now, with this label
    decide: Optional[str] = None          # ask the mission layer for the next skills


@dataclass(frozen=True)
class MissionConfig:
    tolerance: float = 0.5                 # m, distance that counts as "reached"
    yaw_tolerance: float = 0.1             # rad (~6 deg)
    max_speed: float = 3.0                 # m/s, how fast the setpoint moves
    settle_speed: float = 0.3              # m/s, must be this still to decide
    # Photos: which part of a gutter photo is judged comes from the pose
    # recorded with it, and the gutter sits only ~6 deg above the frame's
    # bottom edge. "Arrived" (within tolerance) can still mean 1.5 m/s and 7 deg
    # of tilt; at 0.3 m/s a photo was once taken 0.35 m high, braking after a
    # descent, and the gutter's near end fell out of the picture. No tilt limit:
    # wind holds a hovering drone tilted, and the recorded pose has the tilt.
    # 拍照：水槽照片判断哪一部分要用拍照时记录的位姿来算，而水槽离画面底边只有约 6 度。
    # “到达”（在 tolerance 以内）时速度可能还有 1.5 m/s、倾斜 7 度；原来速度低于 0.3 m/s
    # 就拍，曾在下降后刹车、还高出 0.35 米时拍下，水槽近端出了画面。不限制倾斜：
    # 有风时悬停本来就是斜的，记录的位姿里有倾斜角。
    photo_tolerance: float = 0.2           # m from the viewpoint
    photo_settle_speed: float = 0.15       # m/s
    photo_settle_time: float = 0.5         # s, continuously that still
    photo_timeout: float = 5.0             # s, give up on one photo or scan after this
    decide_timeout: float = 30.0           # s, then go on with the plan as it is
    # Depth scans need the vehicle still: braking tilts it by several degrees,
    # which is metres of error at 15 m. They are converted with the pose at
    # the moment of the image, tilt included, so still means not moving, not
    # exactly on target: in Gazebo a hovering x500 bobs at about 0.1 m/s and
    # settles 0.5-1 degree off the heading asked for. 深度扫描要求静止：刹车时机身
    # 倾斜几度，在 15 米处就是几米的误差。扫描按成像时刻的位姿（含倾斜）转换，所以
    # “静止”指不在动，而不是正好对准：Gazebo 中悬停的 x500 上下晃动约 0.1 m/s，
    # 航向停在离目标 0.5–1 度处。
    scan_settle_speed: float = 0.15        # m/s
    scan_yaw_tolerance: float = 0.035      # rad (2 deg)
    scan_settle_time: float = 1.0          # s, continuously still before scanning
    scan_settle_timeout: float = 10.0      # s at the point, then scan anyway
    prestream_time: float = 1.0            # s of setpoints before asking for offboard
    command_retry: float = 1.0             # s between repeated mode/arm/land commands
    ready_timeout: float = 60.0            # s to wait for PX4 to be ready
    arm_timeout: float = 10.0
    leg_timeout: float = 30.0              # s per leg, on top of distance / max_speed
    brake_speed: float = 0.3               # m/s, slow enough to start landing
    brake_timeout: float = 5.0             # s, land anyway after this long
    # A busy computer can delay PX4's messages for a moment: hold the last
    # setpoint that long before treating it as lost contact.
    # 机载电脑偶尔会晚一点处理 PX4 的消息：这段时间内保持最后的设定点，之后才算失联。
    contact_grace: float = 1.0             # s
    # Landing without any word from PX4 for this long: stop, PX4's failsafe is in charge.
    # 降落阶段持续这么久收不到 PX4 的消息：结束任务，由 PX4 的失效保护处置。
    lost_timeout: float = 60.0             # s
    # PX4 must switch to its land mode within this long; after that we stop
    # asking and stop the setpoints, and PX4's offboard-loss failsafe decides.
    # PX4 须在这段时间内进入降落模式；否则不再请求、停发设定点，由 PX4 的失效保护处置。
    land_confirm_timeout: float = 10.0     # s
    # Once PX4 is landing we only watch: stop watching after this long, so a
    # landing that never ends does not hang the mission. 20 m takes about 20 s.
    # 进入降落模式后只观察；超过这段时间仍未落地就结束任务，避免一直挂着。
    land_timeout: float = 120.0            # s

    def __post_init__(self) -> None:
        if self.tolerance <= 0 or self.max_speed <= 0:
            raise ValueError('tolerance and max_speed must be positive')


class PlanRejected(ValueError):
    """The plan failed check_plan(); the vehicle will not arm."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__('plan rejected: ' + '; '.join(errors))
        self.errors = list(errors)


def waypoints_from_flat(values: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    """[n1, e1, a1, n2, e2, a2, ...] -> ((n1, e1, a1), ...). ROS params have no nested lists."""
    if len(values) % 3 != 0:
        raise ValueError('waypoints must have 3 numbers per point: north, east, altitude')
    return tuple(
        (float(values[i]), float(values[i + 1]), float(values[i + 2]))
        for i in range(0, len(values), 3))


def waypoint_plan(takeoff_altitude: float, waypoints: Sequence[Tuple[float, float, float]],
                  hold_time: float) -> List[Skill]:
    """Takeoff, visit each (north, east, altitude) and hover there, then land."""
    plan: List[Skill] = [Takeoff(takeoff_altitude)]
    for north, east, alt in waypoints:
        plan += [GoTo(north, east, alt), Hover(hold_time)]
    plan.append(Land())
    return plan


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


FLYING_PHASES = (Phase.RUNNING, Phase.BRAKE)


class Mission:
    """Run a checked plan. Call step() at a fixed rate (>= 10 Hz)."""

    def __init__(self, plan: Sequence[Skill], config: MissionConfig = MissionConfig(),
                 limits: Limits = Limits(), frame: str = 'takeoff',
                 start: Optional[Tuple[float, float]] = None) -> None:
        """
        Check the plan; raise PlanRejected if it may not fly.

        frame 'takeoff': north/east relative to the takeoff point.
        frame 'local': north/east in PX4's local frame (as on a map); start must
        then be the vehicle's current (north, east), and the mission aborts
        before arming if the vehicle is no longer there.
        frame='local' 时 start 是飞机当前位置；起飞前飞机若已不在那里则放弃任务。
        """
        if frame not in ('takeoff', 'local'):
            raise ValueError(f'unknown frame {frame}')
        if frame == 'local' and start is None:
            raise ValueError("frame 'local' needs the start position")
        self.frame = frame
        self.start = (0.0, 0.0) if frame == 'takeoff' else start
        check = check_plan(plan, limits, self.start)
        if not check.ok:
            raise PlanRejected(check.errors)
        self.plan = list(plan)
        self.limits = limits
        self.config = config
        self.phase = Phase.WAIT_READY
        self.index = 0                      # current skill
        self.failure: Optional[str] = None  # why the mission did not complete
        self.photos: List[Tuple[str, bool]] = []   # (label, taken)
        self.scans: List[Tuple[str, bool]] = []
        # (label, accepted, errors): what happened at each Decide skill
        self.decisions: List[Tuple[str, bool, List[str]]] = []
        self._phase_start: Optional[float] = None
        self._skill_start = 0.0
        self._leg_limit = math.inf
        self._last_command_t = -math.inf
        self._last_t: Optional[float] = None
        self._home: Optional[Setpoint] = None
        self._target: Optional[Setpoint] = None     # where the current skill goes
        self._carrot: Optional[Setpoint] = None     # setpoint sent, moving at max_speed
        self._asked_t: Optional[float] = None        # photo/scan/decide request sent
        self._still_since: Optional[float] = None
        self._answered = False
        self._answer: object = None
        self._land_requested = False
        self._lost_since: Optional[float] = None     # no fresh, valid vehicle state since
        self._landing_since: Optional[float] = None  # PX4 first seen in its land mode

    # -- public -------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.ABORTED)

    def request_land(self) -> None:
        """Ask to stop the mission and land (e.g. on Ctrl+C). Applied on the next step."""
        self._land_requested = True

    def photo_done(self, ok: bool = True) -> None:
        """Tell the mission the photo it asked for has been taken (or failed)."""
        self._answer = ok
        self._answered = True   # set last: may be called from another thread

    def scan_done(self, ok: bool = True) -> None:
        """Tell the mission the depth scan it asked for has been captured (or failed)."""
        self._answer = ok
        self._answered = True   # set last: may be called from another thread

    def decide_done(self, skills: Optional[Sequence[Skill]],
                    limits: Optional[Limits] = None, why: str = '') -> None:
        """
        Answer a Decide skill: the skills to insert, or None to insert nothing.

        Skills ending with ReturnHome or Land replace the rest of the plan
        (the decision then gives its own complete, checked way home); other
        skills are inserted before the rest.

        limits replaces the mission's limits from now on (e.g. a no-fly box
        measured by perception instead of a rough one). Both are checked
        together, on the next step, before anything changes.
        可以同时替换限制（例如用感知测得的禁飞区代替粗略的），在下一步一起检查。
        """
        self._answer = (list(skills) if skills else None, limits, why)
        self._answered = True   # set last: called from a worker thread

    @property
    def has_home(self) -> bool:
        """Tell whether the takeoff point is known (plan coordinates are measured from it)."""
        return self._home is not None

    def plan_position(self, s: VehicleState) -> Tuple[float, float, float]:
        """PX4 local NED -> plan coordinates (north, east, altitude)."""
        assert self._home is not None
        alt = self._home.z - s.z
        if self.frame == 'local':
            return s.x, s.y, alt
        return s.x - self._home.x, s.y - self._home.y, alt

    def hold_position(self, s: VehicleState) -> Tuple[float, float, float]:
        """
        Where the vehicle is meant to be (the current target), in plan coordinates.

        In-flight decisions are checked from here, so plan them from here too:
        the vehicle itself hovers up to the arrival tolerance away.
        飞行中的决定从目标点开始检查，所以也要从目标点开始规划；飞机本身悬停时可能偏开
        到达容差那么远。
        """
        if self._target is None:
            return self.plan_position(s)
        return self.plan_position(replace(s, x=self._target.x, y=self._target.y,
                                          z=self._target.z))

    def to_ned(self, north: float, east: float, altitude: float) -> Tuple[float, float, float]:
        """Plan coordinates -> PX4 local NED setpoint."""
        assert self._home is not None
        if self.frame == 'local':
            return north, east, self._home.z - altitude
        return self._home.x + north, self._home.y + east, self._home.z - altitude

    # -- helpers ------------------------------------------------------------

    def _enter(self, phase: Phase, s: VehicleState, out: Output, why: str = '') -> None:
        self.phase = phase
        self._phase_start = s.t
        self._last_command_t = -math.inf
        out.events.append(f'-> {phase.value}' + (f' ({why})' if why else ''))

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

    def _brake(self, s: VehicleState, out: Output, why: str) -> None:
        """
        Hold the current position until the vehicle has stopped, then land.

        PX4's land mode descends wherever the vehicle is, but keeps its momentum:
        in simulation a stop at 7 m/s drifted 30 m before touchdown.
        PX4 的降落模式不会先刹车，7 m/s 时中断会再滑出约 30 米，所以先悬停再降落。
        """
        assert self._target is not None
        self.failure = self.failure or why      # landing before the plan is done
        here = Setpoint(s.x, s.y, s.z, self._target.yaw)
        self._target = self._carrot = here
        self._enter(Phase.BRAKE, s, out, why)

    def _start_skill(self, s: VehicleState, out: Output) -> None:
        """Begin plan[index]: set the target, or hand over to landing."""
        assert self._target is not None
        skill = self.plan[self.index]
        self._skill_start = s.t
        self._asked_t = None
        self._still_since = None
        self._answered = False
        self._answer = None
        out.events.append(f'skill {self.index + 1}/{len(self.plan)}: {describe(skill)}')
        cur = self._target
        if isinstance(skill, Takeoff):
            x, y, z = self.to_ned(0.0, 0.0, skill.altitude)
            self._target = Setpoint(x, y, z, cur.yaw)
        elif isinstance(skill, GoTo):
            x, y, z = self.to_ned(skill.north, skill.east, skill.altitude)
            yaw = cur.yaw
            if skill.look_at is not None:
                lx, ly, _ = self.to_ned(skill.look_at[0], skill.look_at[1], 0.0)
                if math.hypot(lx - x, ly - y) > 0.1:
                    yaw = math.atan2(ly - y, lx - x)   # NED: 0 = north, pi/2 = east
            self._target = Setpoint(x, y, z, yaw)
        elif isinstance(skill, ReturnHome):
            assert self._home is not None
            self._target = Setpoint(self._home.x, self._home.y, cur.z, cur.yaw)
        elif isinstance(skill, Land):
            self._enter(Phase.LAND, s, out, 'plan complete')
            return
        dist = math.dist((s.x, s.y, s.z), (self._target.x, self._target.y, self._target.z))
        self._leg_limit = self.config.leg_timeout + dist / self.config.max_speed
        if isinstance(skill, Hover):
            self._leg_limit += skill.seconds
        elif isinstance(skill, Decide):
            self._leg_limit += self.config.decide_timeout

    def _next_skill(self, s: VehicleState, out: Output) -> None:
        self.index += 1
        if self.index < len(self.plan):
            self._start_skill(s, out)
        else:   # check_plan guarantees a terminal skill, but be safe
            self._enter(Phase.LAND, s, out, 'plan complete')

    def _arrived(self, s: VehicleState) -> bool:
        t, c = self._target, self._carrot
        assert t is not None and c is not None
        return ((c.x, c.y, c.z) == (t.x, t.y, t.z)
                and math.dist((s.x, s.y, s.z), (t.x, t.y, t.z)) < self.config.tolerance
                and abs(_wrap(s.heading - t.yaw)) < self.config.yaw_tolerance)

    def _move_carrot(self, s: VehicleState) -> None:
        """Move the published setpoint towards the target at max_speed."""
        t, c = self._target, self._carrot
        assert t is not None and c is not None
        dt = 0.0 if self._last_t is None else min(max(s.t - self._last_t, 0.0), 0.2)
        d = math.dist((c.x, c.y, c.z), (t.x, t.y, t.z))
        step = self.config.max_speed * dt
        if d <= step or d < 1e-6:
            self._carrot = t
        else:
            k = step / d
            self._carrot = Setpoint(c.x + k * (t.x - c.x), c.y + k * (t.y - c.y),
                                    c.z + k * (t.z - c.z), t.yaw)

    # -- main loop ----------------------------------------------------------

    def step(self, s: VehicleState) -> Output:
        out = Output()
        if self._phase_start is None:
            self._phase_start = s.t
        cfg = self.config
        if s.connected and s.position_valid:
            self._lost_since = None
        elif self._lost_since is None:
            self._lost_since = s.t
        lost_for = 0.0 if self._lost_since is None else s.t - self._lost_since
        holding = False

        # Safety checks while we are the one flying. 飞行中的安全检查。
        if self.phase in FLYING_PHASES:
            # Lost contact first: without fresh messages, armed/offboard read as
            # False and must not be taken for a disarm or a pilot takeover.
            # 先判断是否失去通信：收不到消息时“已解锁”“offboard”都读作 False，
            # 不能当成上锁或遥控接管。
            if self._lost_since is not None and lost_for < cfg.contact_grace:
                holding = True       # keep streaming the last setpoint, wait
            elif self._lost_since is not None:
                # Cannot brake without a position: let PX4 land where it is.
                # 没有位置就无法刹车：直接让 PX4 原地降落。
                self.failure = self.failure or f'lost vehicle state for {lost_for:.1f} s'
                self._enter(Phase.LAND, s, out, 'lost vehicle state')
            elif not s.armed:
                self._abort(s, out, 'vehicle disarmed')
            elif not s.offboard:
                # A pilot or a PX4 failsafe switched modes: never fight it.
                # 遥控接管或 PX4 失效保护切换了模式：不要抢回控制权。
                self._abort(s, out, 'left offboard mode, PX4 or pilot in control')
            elif self.phase == Phase.BRAKE:
                pass
            elif self._land_requested:
                self._brake(s, out, 'land requested')
            elif s.t - self._skill_start > self._leg_limit:
                self._brake(s, out, f'skill {self.index + 1} timed out')
        elif self._land_requested and self.phase in (
                Phase.WAIT_READY, Phase.PRESTREAM, Phase.ARMING):
            if s.armed:
                out.commands.append(Command.DISARM)
            self._abort(s, out, 'cancelled before takeoff')

        if self.phase == Phase.WAIT_READY:
            # PX4 runs its own checks when we ask it to arm. PX4 解锁时自己会做检查。
            if s.connected and s.position_valid and not s.armed and s.landed:
                if self.frame == 'local' and math.hypot(
                        s.x - self.start[0], s.y - self.start[1]) > 1.0:
                    self._abort(s, out, f'vehicle at ({s.x:.1f}, {s.y:.1f}) is not at the '
                                        f'start the plan was checked for {self.start}')
                    return out
                self._home = Setpoint(s.x, s.y, s.z, s.heading)
                self._target = self._carrot = self._home
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
                self._enter(Phase.RUNNING, s, out)
                self._start_skill(s, out)
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

        elif holding:
            pass

        elif self.phase == Phase.RUNNING:
            self._run_skill(s, out)

        elif self.phase == Phase.BRAKE:
            if s.speed < cfg.brake_speed:
                self._enter(Phase.LAND, s, out, f'stopped, speed {s.speed:.1f} m/s')
            elif self._elapsed(s) > cfg.brake_timeout:
                self._enter(Phase.LAND, s, out, f'brake timed out at {s.speed:.1f} m/s')

        if self.phase == Phase.LAND:
            self._land(s, out, lost_for)

        # Keep streaming setpoints in every phase where offboard is (or will be)
        # active; LAND streams its own until PX4 switches (see _land).
        # 在需要 offboard 的阶段持续发送设定点，PX4 收不到会触发失效保护；
        # 降落阶段在 _land 里自己发送，直到 PX4 切换模式。
        if self.phase in (Phase.PRESTREAM, Phase.ARMING) + FLYING_PHASES:
            if not holding:
                self._move_carrot(s)
            out.setpoint = self._carrot
        self._last_t = s.t
        return out

    def _land(self, s: VehicleState, out: Output, lost_for: float) -> None:
        """
        Ask PX4 to land until it is seen landing, then only watch.

        Until PX4 reports its land mode the vehicle is still ours (offboard):
        hold the last setpoint, so the switch does not trip PX4's offboard-loss
        failsafe, and repeat the land command. Any other mode, before or after,
        means a pilot or a PX4 failsafe took over: stop commanding at once.
        (Regression: a pilot switching to position mode during the landing was
        switched back to land mode every second, indefinitely.)
        PX4 进入降落模式之前飞机仍由我们控制：保持最后的设定点（以免切换时触发
        offboard 丢失保护），并重发降落指令。之前或之后出现任何其他模式，都说明
        遥控接管或 PX4 失效保护介入，立即停止发指令。（曾经在降落中遥控切到位置模式后，
        节点每秒把模式切回降落，没有尽头。）
        """
        cfg = self.config
        if not s.connected:
            # Without messages, armed and landed read as their defaults: they
            # say nothing. Keep asking PX4 to land, unless it already is: then
            # a mode change we cannot see may be a pilot taking over.
            # 收不到消息时“未解锁”“已落地”只是默认值，不能当真。PX4 尚未开始降落时继续
            # 发降落指令；已经在降落时不再发，因为看不到的模式切换可能是遥控接管。
            if lost_for > cfg.lost_timeout:
                self._abort(s, out, f'no word from PX4 for {lost_for:.0f} s while '
                                    f'landing; PX4 failsafe in charge')
            elif self._landing_since is None and self._retry_due(s):
                out.commands.append(Command.LAND)
            return
        if s.landing and self._landing_since is None:
            self._landing_since = s.t
            out.events.append('PX4 is landing')
        if not s.armed:
            if s.landed:
                self._enter(Phase.DONE, s, out, 'landed and disarmed')
            else:
                self._abort(s, out, 'disarmed before touchdown, PX4 or pilot in control')
        elif s.landing:
            if s.t - self._landing_since > cfg.land_timeout:
                self._abort(s, out, f'still landing after {cfg.land_timeout:.0f} s; '
                                    f'left to PX4')
        elif self._landing_since is not None:
            if s.landed:
                # A pilot often switches mode on the ground to disarm by hand.
                # 落地后遥控切模式再手动上锁是常见操作，算正常落地。
                self._enter(Phase.DONE, s, out, 'landed; mode changed before disarming')
            else:
                self._abort(s, out, 'left land mode, PX4 or pilot in control')
        elif not s.offboard:
            self._abort(s, out, 'left offboard mode before landing started, '
                                'PX4 or pilot in control')
        elif self._elapsed(s) > cfg.land_confirm_timeout:
            self._abort(s, out, f'PX4 did not start landing within '
                                f'{cfg.land_confirm_timeout:.0f} s; its failsafe decides')
        else:
            out.setpoint = self._carrot
            if self._retry_due(s):
                out.commands.append(Command.LAND)

    def _settled(self, s: VehicleState, strict: bool) -> bool:
        """Close to the viewpoint and still for a moment; for scans, very still and longer."""
        cfg = self.config
        t = self._target
        assert t is not None
        if strict:
            still = (self._arrived(s) and s.speed < cfg.scan_settle_speed
                     and abs(_wrap(s.heading - t.yaw)) < cfg.scan_yaw_tolerance)
            wait = cfg.scan_settle_time
        else:
            still = (self._arrived(s) and s.speed < cfg.photo_settle_speed
                     and math.dist((s.x, s.y, s.z), (t.x, t.y, t.z)) < cfg.photo_tolerance)
            wait = cfg.photo_settle_time
        if not still:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = s.t
        return s.t - self._still_since >= wait

    def _apply_decision(self, label: str, s: VehicleState, out: Output) -> None:
        """Insert the skills from decide_done() after the current one, if they pass the check."""
        skills, limits, why = self._answer
        if not skills:
            self.decisions.append((label, False, [why or 'nothing to add']))
            out.events.append(f'decide "{label}": nothing added' + (f' ({why})' if why else ''))
            return
        assert self._target is not None
        here = self.hold_position(s)
        # Skills that already end the mission (return home / land) replace what
        # was left; others are inserted before it. Either way the whole rest
        # must pass the check from here, or nothing changes.
        # 以返航或降落结尾的技能替换剩余计划，否则插在剩余计划之前；都必须从当前位置通过检查。
        if isinstance(skills[-1], TERMINAL):
            rest = list(skills)
        else:
            rest = list(skills) + self.plan[self.index + 1:]
        new_limits = limits if limits is not None else self.limits
        check = check_plan(rest, new_limits, self.start, initial=here)
        if not check.ok:
            self.decisions.append((label, False, check.errors))
            out.events.append(f'decide "{label}": rejected by the check, going on with the '
                              f'plan: {"; ".join(check.errors[:3])}')
            return
        self.plan = self.plan[:self.index + 1] + rest
        self.limits = new_limits
        self.decisions.append((label, True, []))
        out.events.append(f'decide "{label}": {len(skills)} skill(s) added, '
                          f'plan now {len(self.plan)} skills')

    def _run_skill(self, s: VehicleState, out: Output) -> None:
        skill = self.plan[self.index]
        cfg = self.config
        if isinstance(skill, (Takeoff, GoTo)):
            if self._arrived(s):
                self._next_skill(s, out)
        elif isinstance(skill, ReturnHome):
            if self._arrived(s):
                self._enter(Phase.LAND, s, out, 'back home')
        elif isinstance(skill, Hover):
            if s.t - self._skill_start >= skill.seconds:
                self._next_skill(s, out)
        elif isinstance(skill, (TakePhoto, Scan)):
            kind, log = (('photo', self.photos) if isinstance(skill, TakePhoto)
                         else ('scan', self.scans))
            if self._asked_t is None:
                if self._settled(s, kind == 'scan' or getattr(skill, 'steady', False)):
                    self._asked_t = s.t
                    setattr(out, kind, skill.label)
                elif (kind == 'scan' and self._arrived(s)
                      and s.t - self._skill_start > cfg.scan_settle_timeout):
                    # Hovering never quite still enough (it bobs at about the
                    # threshold): scan anyway rather than time out and land.
                    # 悬停一直达不到完全静止（上下晃动正好在门限附近）：照样扫描，
                    # 不要等到超时降落。
                    out.events.append(f'scan "{skill.label}": not still after '
                                      f'{cfg.scan_settle_timeout:g} s, scanning anyway')
                    self._asked_t = s.t
                    out.scan = skill.label
            elif self._answered:
                log.append((skill.label, bool(self._answer)))
                if not self._answer:
                    out.events.append(f'{kind} "{skill.label}" failed, continuing')
                self._next_skill(s, out)
            elif s.t - self._asked_t > cfg.photo_timeout:
                log.append((skill.label, False))
                out.events.append(f'{kind} "{skill.label}" timed out, continuing')
                self._next_skill(s, out)
        elif isinstance(skill, Decide):
            if self._asked_t is None:
                if s.speed < cfg.settle_speed and self._arrived(s):
                    self._asked_t = s.t
                    out.decide = skill.label
            elif self._answered:
                self._apply_decision(skill.label, s, out)
                self._next_skill(s, out)
            elif s.t - self._asked_t > cfg.decide_timeout:
                self.decisions.append((skill.label, False, ['timed out']))
                out.events.append(f'decide "{skill.label}" timed out, going on with the plan')
                self._next_skill(s, out)
        elif isinstance(skill, Land):   # handled in _start_skill, kept for safety
            self._enter(Phase.LAND, s, out, 'plan complete')
