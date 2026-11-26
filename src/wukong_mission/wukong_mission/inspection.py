# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Turn the eaves found in flight into photo viewpoints and a way round the house.

plan_inspection(): one viewpoint per gutter segment, facing it, chosen and
joined by the path planner (pathplan.py) on the height map of everything
seen. Geometry only; the plan still goes through check_plan().
把飞行中识别出的屋檐变成逐段拍照的视点和绕行航线：纯几何，仍然要通过 check_plan()。
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from wukong_flight.skills import GoTo, HeightField, Limits, Skill, TakePhoto
from wukong_perception.geometry import Eave, segment_eave, segments_for

if TYPE_CHECKING:
    from wukong_perception.heightmap import HeightMap

    from .pathplan import ViewConfig

Point = Tuple[float, float]


@dataclass(frozen=True)
class InspectionConfig:
    segment_length: float = 3.0    # m, longest gutter segment per photo
    # Photograph this far past each measured end of an eave: measured eaves stop
    # a little short at gable ends, and gutter ends (by the downspouts) are where
    # debris collects. 每条屋檐两端多拍这么远：测得的屋檐在山墙端会短一点，
    # 而水槽两端（落水管旁）正是最容易积杂物的地方。
    end_margin: float = 0.3
    # ...and this far where the eave runs into a taller part of the building (a
    # garage roof against the house wall). There, scans that do not line up
    # spread the taller wall over the lower roof and the measured eave stops
    # short by about their misalignment (under 0.1 m without drift, 1.0-1.2 m
    # in Gazebo). Past the real end the photo only shows the wall.
    # 屋檐一端紧挨更高的楼体（车库屋面接主楼墙）时多拍这么远：扫描没对齐时，高墙在高度图
    # 里会盖到低屋面上，测得的屋檐比实际短，短多少取决于没对齐的程度（无漂移时不到 0.1 米，
    # Gazebo 中 1.0 到 1.2 米）。拍过头的部分只是墙。
    wall_margin: float = 1.2


@dataclass
class PhotoTarget:
    """What a planned photo is meant to show, for cropping and the report."""

    label: str
    gutter: str
    segment: int                   # 1-based
    segments: int
    start: Point
    end: Point
    height: float
    viewpoint: Tuple[float, float, float]
    # One home of a townhouse row (set by HouseTask): a point on this gutter's
    # line inside the home being inspected, and the part of the segment known
    # to be that home's without looking (it may run past a party wall, so the
    # boundary shows in the photo). None: the whole segment is the house's.
    # 联排中的一户：本水槽线上位于目标户内的一点，以及不看照片就能确定属于目标户的那部分
    # （段可能越过共用墙，好让分界出现在照片里）。None 表示整段都属于这栋房子。
    home: Optional[Point] = None
    claim: Optional[Tuple[Point, Point]] = None
    # Whether a party wall bounds the home toward this segment's start and
    # toward its end: an end unit has none toward the row's end, where a
    # corner can look like a boundary. 目标户在本段起点方向和终点方向是否有共用墙：
    # 端户朝整排尽头的方向没有，那里的转角可能被看成分界。
    walls: Tuple[bool, bool] = (True, True)


@dataclass
class Skipped:
    """A gutter segment that will not be photographed, and why."""

    label: str
    gutter: str
    segment: int
    start: Point
    end: Point
    reason: str


def _extended(e: Eave, before: float, after: float) -> Eave:
    """Return the eave lengthened by `before` at its start and `after` at its end."""
    if (before <= 0 and after <= 0) or e.length < 1e-6:
        return e
    dn = (e.end[0] - e.start[0]) / e.length
    de = (e.end[1] - e.start[1]) / e.length
    return Eave(e.name, (round(e.start[0] - dn * before, 3), round(e.start[1] - de * before, 3)),
                (round(e.end[0] + dn * after, 3), round(e.end[1] + de * after, 3)),
                e.height, e.side_bearing_deg)


def runs_into_taller(e: Eave, at_end: bool, surfaces: Sequence[HeightField],
                     reach: float = 1.5, inside: float = 0.3, rise: float = 1.5) -> bool:
    """
    Whether something much taller stands just past one end of the eave, on its roof side.

    That is a lower roof meeting a taller part of the building, where the
    gutter runs on up to the wall. Looked for within `reach` past the end,
    `inside` back from the eave line, at least `rise` above the eave: a roof
    of its own rises less than that so close to its edge.
    屋檐一端外侧（屋面一侧）紧挨着高得多的东西：低屋面接更高的楼体，水槽一直通到墙边。
    在端点外 reach 以内、屋檐线往里 inside 处找，比屋檐高 rise 以上才算；
    自身的屋面离边缘这么近时升不了这么高。
    """
    if e.length < 1e-6:
        return False
    sign = 1.0 if at_end else -1.0
    dn = sign * (e.end[0] - e.start[0]) / e.length
    de = sign * (e.end[1] - e.start[1]) / e.length
    p0 = e.end if at_end else e.start
    o = e.outward
    steps = max(1, int(round(reach / 0.1)))
    for k in range(1, steps + 1):
        t = k * reach / steps
        n = p0[0] + dn * t - o[0] * inside
        east = p0[1] + de * t - o[1] * inside
        for hf in surfaces:
            h = hf.height_at(n, east)
            if h is not None and h > e.height + rise:
                return True
    return False


def plan_inspection(eaves: Sequence[Eave], surfaces: Sequence[HeightField],
                    here: Tuple[float, float, float], home: Point, limits: Limits,
                    cfg: InspectionConfig = InspectionConfig(),
                    view: Optional['ViewConfig'] = None
                    ) -> Tuple[List[Skill], Dict[str, PhotoTarget], List[Skipped]]:
    """
    Photograph every segment of the given eaves, flying around whatever is there.

    surfaces: HeightFields of everything seen (for line of sight); limits must
    already contain them as no-fly zones. Ends at home's position at a safe
    altitude, so a following ReturnHome only has to land.
    对给定屋檐逐段拍照；无法拍摄的段记录原因后跳过。最后回到起飞点上方。
    """
    from .pathplan import choose_viewpoint, Planner, route_home, ViewConfig
    planner = Planner(limits, home=home)
    view = view or ViewConfig()
    skills: List[Skill] = []
    targets: Dict[str, PhotoTarget] = {}
    skipped: List[Skipped] = []

    # Viewpoints for every segment first. 先为每一段选好视点。
    plans = []
    for e in eaves:
        before, after = (max(cfg.end_margin, cfg.wall_margin)
                         if cfg.wall_margin > 0 and runs_into_taller(e, at_end, surfaces)
                         else cfg.end_margin for at_end in (False, True))
        e = _extended(e, before, after)
        n = segments_for(e, cfg.segment_length)
        views = []
        for i, (a, b) in enumerate(segment_eave(e, n), 1):
            label = f'{e.name} segment {i} of {n}'
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            vp = choose_viewpoint(planner, mid, e.height, e.outward, surfaces, view)
            if vp is None:
                skipped.append(Skipped(label, e.name, i, a, b,
                                       'no safe viewpoint that can see it'))
                continue
            views.append((label, i, n, a, b, mid, vp))
        if views:
            plans.append((e, views))

    pos = here
    while plans:
        # Nearest eave next, entered from its nearer end. 先去最近的屋檐，从较近的一端开始。
        best = min(range(len(plans)), key=lambda k: min(
            math.dist(pos, plans[k][1][0][6]), math.dist(pos, plans[k][1][-1][6])))
        e, views = plans.pop(best)
        if math.dist(pos, views[-1][6]) < math.dist(pos, views[0][6]):
            views = views[::-1]
        for label, i, n, a, b, mid, vp in views:
            route = planner.path(pos, vp)
            if route is None:
                skipped.append(Skipped(label, e.name, i, a, b, 'no safe path to its viewpoint'))
                continue
            for p in route[:-1]:
                skills.append(GoTo(*p))
            skills.append(GoTo(*vp, look_at=(round(mid[0], 2), round(mid[1], 2))))
            skills.append(TakePhoto(label))
            targets[label] = PhotoTarget(label, e.name, i, n, a, b, e.height, vp)
            pos = vp
    # Back above home, at an altitude from which the way is clear. 回到起飞点上方。
    route = route_home(planner, pos, home)
    if route is None:
        skipped.append(Skipped('return', '', 0, home, home, 'no safe path home found'))
    else:
        skills += [GoTo(*p) for p in route]
    return skills, targets, skipped


def surfaces_from_heightmap(hm: 'HeightMap', res: float = 0.5, min_height: float = 0.8,
                            name: str = 'surroundings') -> HeightField:
    """
    Convert a perception HeightMap (numpy) to a flight-layer HeightField no-fly zone.

    Cells are max-pooled to about `res` (a whole number of the map's cells, on
    the map's own origin, so each coarse cell covers exactly its fine cells).
    Only cells at least min_height above the ground are kept, and heights are
    put back into the altitude frame the vehicle flies in (ground_level added).
    把感知的高度图按最大值降采样为飞行层的高度场禁飞区，高度换回飞行使用的高度坐标系。
    """
    import numpy as np
    k = max(1, int(round(res / hm.res)))
    top = np.nan_to_num(hm.top, nan=-np.inf)
    r, c = top.shape
    r2, c2 = -(-r // k), -(-c // k)
    pad = np.full((r2 * k, c2 * k), -np.inf)
    pad[:r, :c] = top
    pooled = pad.reshape(r2, k, c2, k).max(axis=(1, 3))
    cells = {(int(i), int(j)): float(pooled[i, j]) + hm.ground_level
             for i, j in zip(*np.nonzero(pooled >= min_height))}
    return HeightField(hm.origin, k * hm.res, cells, name)
