# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
The skill library and the pre-flight check of a plan.

A plan is a list of skills. Whoever writes it (a person, a config file or a
large model), it must pass check_plan() before the vehicle arms. The check is
pure geometry: limits on altitude and distance, and no-fly boxes around known
obstacles. It never trusts the author.
技能库和飞行前的计划检查。无论计划是谁写的（人、配置文件或大模型），解锁前都必须
通过 check_plan()。检查只依赖几何：高度、距离限制和已知障碍物周围的禁飞区。

North and east are in metres, either relative to the takeoff point (the
default) or in PX4's local frame (origin where PX4 started, used with a map).
Altitude is always metres above the takeoff point, up positive.
北、东（米）可以相对起飞点（默认），也可以用 PX4 本地坐标系（配合地图使用）。
高度始终是相对起飞点的米数，向上为正。
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Takeoff:
    """Climb straight up above the takeoff point. Must be the first skill."""

    altitude: float


@dataclass(frozen=True)
class GoTo:
    """Fly in a straight line to a point; optionally turn to face another point."""

    north: float
    east: float
    altitude: float
    # Face this (north, east) point on arrival, e.g. the part of a house to
    # photograph. None keeps the current heading. 到达后朝向该点，None 保持航向。
    look_at: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class Hover:
    """Hold position for a number of seconds."""

    seconds: float


@dataclass(frozen=True)
class TakePhoto:
    """
    Settle, then take one photo with the forward camera.

    steady: wait as for a depth scan (heading within a degree, still for a
    second). For photos whose pixels are mapped back onto the world with the
    recorded pose (the ruler on the house overview): the frame arrives a few
    hundred ms after the request, and a drone still turning made the drawn
    ruler miss the house by ~9 degrees. 等到像深度扫描一样稳定再拍：照片要用记录的
    位姿反投影时（全景照上的刻度尺）使用。照片比请求晚几百毫秒到达，飞机还在转动时，
    刻度尺曾与画面偏差约 9 度。
    """

    label: str
    steady: bool = False


@dataclass(frozen=True)
class ReturnHome:
    """Fly back above the takeoff point at the current altitude, then land."""


@dataclass(frozen=True)
class Land:
    """Land where the vehicle is."""


@dataclass(frozen=True)
class Scan:
    """Settle, then capture one depth frame (for perception, not for people)."""

    label: str


@dataclass(frozen=True)
class Decide:
    """
    Hold position while the mission layer works out the next skills.

    Typically after scanning: perception turns the scans into geometry, and
    the next skills are computed from it. They are inserted after this skill
    only if the rest of the plan, from here, passes check_plan(); otherwise
    the mission goes on with the skills that were already planned.
    悬停等待任务层根据感知结果给出后续技能；只有通过检查才插入，否则按原计划继续。
    """

    label: str


Skill = Union[Takeoff, GoTo, Hover, TakePhoto, Scan, Decide, ReturnHome, Land]
TERMINAL = (ReturnHome, Land)


@dataclass(frozen=True)
class Box:
    """An axis-aligned no-fly box: north/east extents and a top height, from the ground up."""

    north_min: float
    north_max: float
    east_min: float
    east_max: float
    top: float
    name: str = ''

    def inflated(self, margin: float) -> 'Box':
        return Box(self.north_min - margin, self.north_max + margin,
                   self.east_min - margin, self.east_max + margin,
                   self.top + margin, self.name)

    def contains(self, north: float, east: float, altitude: float) -> bool:
        return (self.north_min <= north <= self.north_max
                and self.east_min <= east <= self.east_max
                and altitude <= self.top)


@dataclass(frozen=True)
class RotatedBox:
    """
    A no-fly box turned to any heading, e.g. a house that is not aligned north.

    length runs along bearing_deg (compass: 0 north, 90 east), width across it.
    length 沿 bearing_deg 方向（罗盘方位），width 与之垂直。
    """

    center: Tuple[float, float]     # (north, east)
    length: float
    width: float
    bearing_deg: float
    top: float
    name: str = ''

    def inflated(self, margin: float) -> 'RotatedBox':
        return RotatedBox(self.center, self.length + 2 * margin, self.width + 2 * margin,
                          self.bearing_deg, self.top + margin, self.name)

    def contains(self, north: float, east: float, altitude: float) -> bool:
        if altitude > self.top:
            return False
        b = math.radians(self.bearing_deg)
        dn, de = north - self.center[0], east - self.center[1]
        along = dn * math.cos(b) + de * math.sin(b)
        across = -dn * math.sin(b) + de * math.cos(b)
        return abs(along) <= self.length / 2 and abs(across) <= self.width / 2


@dataclass(frozen=True)
class Cylinder:
    """A round no-fly zone: everything within radius of center, up to top."""

    center: Tuple[float, float]     # (north, east)
    radius: float
    top: float
    name: str = ''

    def inflated(self, margin: float) -> 'Cylinder':
        return Cylinder(self.center, self.radius + margin, self.top + margin, self.name)

    def contains(self, north: float, east: float, altitude: float) -> bool:
        return (altitude <= self.top
                and math.hypot(north - self.center[0], east - self.center[1]) <= self.radius)


class HeightField:
    """
    Everything seen above the ground, as a grid of heights: a no-fly zone of any shape.

    cells maps (i, j) -> top height; cell (i, j) covers north origin[0] + i*res
    .. + res, east likewise. Heights are in the plan's altitude frame. Built by
    perception from the survey points (houses, neighbours, trees alike); plain
    Python so the flight layer needs no numpy.
    把地面以上看到的一切表示为高度栅格，作为任意形状的禁飞区（房屋、邻居、树木都一样）。
    """

    def __init__(self, origin: Tuple[float, float], res: float,
                 cells: Dict[Tuple[int, int], float], name: str = 'surroundings') -> None:
        self.origin = (float(origin[0]), float(origin[1]))
        self.res = float(res)
        self.cells = dict(cells)
        self.name = name
        self._inflated: Dict[float, 'HeightField'] = {}

    @property
    def top(self) -> float:
        return max(self.cells.values()) if self.cells else 0.0

    def cell(self, north: float, east: float) -> Tuple[int, int]:
        return (int(math.floor((north - self.origin[0]) / self.res)),
                int(math.floor((east - self.origin[1]) / self.res)))

    def height_at(self, north: float, east: float) -> Optional[float]:
        return self.cells.get(self.cell(north, east))

    def contains(self, north: float, east: float, altitude: float) -> bool:
        h = self.cells.get(self.cell(north, east))
        return h is not None and altitude <= h

    def inflated(self, margin: float) -> 'HeightField':
        """Grow by margin sideways (a disk) and upwards. Cached per margin."""
        key = round(margin, 3)
        if key in self._inflated:
            return self._inflated[key]
        # A cell is blocked if any point of it is within margin of a blocked
        # cell: reach = margin plus one cell diagonal, to stay conservative.
        k = int(math.ceil(margin / self.res + math.sqrt(2)))
        offsets = [(di, dj) for di in range(-k, k + 1) for dj in range(-k, k + 1)
                   if max(0.0, math.hypot(di, dj) - math.sqrt(2)) * self.res <= margin]
        out: Dict[Tuple[int, int], float] = {}
        for (i, j), h in self.cells.items():
            t = h + margin
            for di, dj in offsets:
                c = (i + di, j + dj)
                if out.get(c, -math.inf) < t:
                    out[c] = t
        hf = HeightField(self.origin, self.res, out, self.name)
        self._inflated[key] = hf
        return hf


NoFly = Union[Box, RotatedBox, Cylinder, HeightField]


@dataclass(frozen=True)
class Limits:
    min_altitude: float = 1.5        # m, lowest allowed except while landing
    max_altitude: float = 20.0       # m
    max_distance: float = 50.0       # m, horizontal, from the takeoff point
    max_hover: float = 60.0          # s, per Hover skill
    max_skills: int = 40
    no_fly: Tuple[NoFly, ...] = ()
    margin: float = 2.0              # m, kept from every no-fly box
    sample_step: float = 0.25        # m, resolution of the path check


@dataclass
class PlanCheck:
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _segment_hits(a, b, box: Box, step: float) -> bool:
    """Whether the straight segment a -> b, each (north, east, alt), enters box."""
    n = max(1, int(math.ceil(math.dist(a, b) / step)))
    return any(box.contains(*(a[k] + (b[k] - a[k]) * i / n for k in range(3)))
               for i in range(n + 1))


def check_plan(plan: Sequence[Skill], limits: Limits = Limits(),
               start: Tuple[float, float] = (0.0, 0.0),
               initial: Optional[Tuple[float, float, float]] = None) -> PlanCheck:
    """
    Check a plan before flight. An empty error list means it may fly.

    start is the takeoff point in the plan's frame: (0, 0) when the plan is
    relative to the takeoff point. max_distance is measured from start.
    initial: check the rest of a plan in flight, starting from the vehicle's
    current (north, east, altitude); takeoff is then not allowed.
    start 是起飞点在计划坐标系中的位置。initial 用于飞行中检查剩余计划。
    """
    result = PlanCheck()
    err = result.errors.append
    if not plan:
        err('plan is empty')
        return result
    if len(plan) > limits.max_skills:
        err(f'plan has {len(plan)} skills, the limit is {limits.max_skills}')
    if initial is None and not isinstance(plan[0], Takeoff):
        err('the first skill must be takeoff')
    if not isinstance(plan[-1], TERMINAL):
        err('the last skill must be land or return_home')
    boxes = [b.inflated(limits.margin) for b in limits.no_fly]

    home = (float(start[0]), float(start[1]))
    pos = (home[0], home[1], 0.0) if initial is None else tuple(map(float, initial))
    for i, skill in enumerate(plan, 1):
        where = f'skill {i} ({type(skill).__name__})'
        if isinstance(skill, Takeoff) and (i > 1 or initial is not None):
            err(f'{where}: takeoff is only allowed as the first skill')
        if isinstance(skill, TERMINAL) and i < len(plan):
            err(f'{where}: nothing may follow land or return_home')

        if isinstance(skill, Takeoff):
            nxt = (home[0], home[1], skill.altitude)
        elif isinstance(skill, GoTo):
            nxt = (skill.north, skill.east, skill.altitude)
            if skill.look_at is not None and len(skill.look_at) != 2:
                err(f'{where}: look_at must be (north, east)')
        elif isinstance(skill, ReturnHome):
            nxt = (home[0], home[1], pos[2])
        else:
            nxt = pos
            if isinstance(skill, Hover) and not 0 <= skill.seconds <= limits.max_hover:
                err(f'{where}: hover {skill.seconds} s is outside 0..{limits.max_hover} s')
            if isinstance(skill, (TakePhoto, Scan, Decide)) and not skill.label.strip():
                err(f'{where}: label is empty')

        if nxt != pos:
            if not limits.min_altitude <= nxt[2] <= limits.max_altitude:
                err(f'{where}: altitude {nxt[2]:.1f} m is outside '
                    f'{limits.min_altitude}..{limits.max_altitude} m')
            away = math.hypot(nxt[0] - home[0], nxt[1] - home[1])
            if away > limits.max_distance:
                err(f'{where}: {away:.1f} m from home, the limit is {limits.max_distance} m')
            for box in boxes:
                if _segment_hits(pos, nxt, box, limits.sample_step):
                    err(f'{where}: path from {_fmt(pos)} to {_fmt(nxt)} comes within '
                        f'{limits.margin} m of no-fly box {box.name or "?"}')
        pos = nxt
    # The final descent (land or return_home) goes straight down.
    for box in boxes:
        if _segment_hits(pos, (pos[0], pos[1], 0.0), box, limits.sample_step):
            err(f'landing at {_fmt(pos)} descends into no-fly box {box.name or "?"}')
    return result


def _fmt(p) -> str:
    return f'(N {p[0]:.1f}, E {p[1]:.1f}, alt {p[2]:.1f})'


def describe(skill: Skill) -> str:
    """One line, for logs and for showing a plan to a person."""
    if isinstance(skill, Takeoff):
        return f'takeoff to {skill.altitude:.1f} m'
    if isinstance(skill, GoTo):
        s = f'go to N {skill.north:.1f} E {skill.east:.1f} alt {skill.altitude:.1f}'
        if skill.look_at is not None:
            s += f', facing N {skill.look_at[0]:.1f} E {skill.look_at[1]:.1f}'
        return s
    if isinstance(skill, Hover):
        return f'hover {skill.seconds:.0f} s'
    if isinstance(skill, TakePhoto):
        return f'take photo "{skill.label}"'
    if isinstance(skill, Scan):
        return f'depth scan "{skill.label}"'
    if isinstance(skill, Decide):
        return f'decide next steps ({skill.label})'
    if isinstance(skill, ReturnHome):
        return 'return home and land'
    return 'land'
