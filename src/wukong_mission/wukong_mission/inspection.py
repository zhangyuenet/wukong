# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Turn "inspect this house's gutters" into flyable skills, in two halves.

1. survey_skills(): before takeoff, from the rough map only. Circle the house
   outside its rough no-fly zone, take a depth scan at each station, then a
   Decide skill.
2. inspection_skills(): at the Decide, from the fitted house. One viewpoint per
   gutter segment, facing it; between sides, go around the house on a ring.

Both are geometry only; the plans still go through check_plan().
两部分都是纯几何：起飞前按粗略地图规划环绕扫描；扫描后按拟合结果规划逐段拍照。
仍然要通过 check_plan()。
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from wukong_flight.skills import (Decide, GoTo, Limits, RotatedBox, Scan, Skill, Takeoff,
                                  TakePhoto)
from wukong_perception.geometry import Eave, HouseEstimate, segment_eave, segments_for

from .world import RoughHouse

Point = Tuple[float, float]


@dataclass(frozen=True)
class InspectionConfig:
    stations: int = 8              # survey stations around the house
    survey_altitude: float = 4.5   # m; low enough to see the eaves with a level camera
    segment_length: float = 3.0    # m, longest gutter segment per photo
    standoff: float = 4.5          # m, horizontal distance of a viewpoint from the eave
    # m above the gutter lip. The camera sits 0.26 m higher still; at 4.5 m this
    # puts the gutter ~14.5 deg below centre, inside the +-21 deg vertical FOV,
    # and looks a little into the channel (from 0.6 m up, clean gutters often
    # showed only their outer face: "unclear").
    # 视点比水槽外沿高 0.9 米（相机再高 0.26 米），水槽在画面中心下方约 14.5 度，
    # 能看到一点槽内；0.6 米时干净的水槽常常只看到外侧面，被判为“看不清”。
    above_gutter: float = 0.9
    clearance_extra: float = 0.5   # m beyond the safety margin for survey stations
    # Photograph this far past each measured end of an eave: measured eaves stop
    # a little short at gable ends, and gutter ends (by the downspouts) are where
    # debris collects. 每条屋檐两端多拍这么远：测得的屋檐在山墙端会短一点，
    # 而水槽两端（落水管旁）正是最容易积杂物的地方。
    end_margin: float = 0.3


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


def _bearing(frm: Point, to: Point) -> float:
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0])) % 360


def survey_skills(house: RoughHouse, limits: Limits, start: Point,
                  cfg: InspectionConfig = InspectionConfig()) -> List[Skill]:
    """Takeoff, circle the rough zone scanning at each station, then Decide."""
    k = cfg.stations
    # Chords between stations must stay outside the zone plus margin.
    # 相邻站点之间的直线也要在禁飞区外。
    r = (house.radius + limits.margin + cfg.clearance_extra) / math.cos(math.pi / k)
    c = house.approx_center
    first = _bearing(c, start)       # start on the side facing the takeoff point
    skills: List[Skill] = [Takeoff(cfg.survey_altitude)]
    for i in range(k):
        b = math.radians(first + 360.0 * i / k)
        p = (c[0] + r * math.cos(b), c[1] + r * math.sin(b))
        skills.append(GoTo(round(p[0], 2), round(p[1], 2), cfg.survey_altitude, look_at=c))
        skills.append(Scan(f'{house.name} station {i + 1}/{k}'))
    skills.append(Decide(f'{house.name} survey done'))
    return skills


class Ring:
    """
    A rectangle around the house, offset outward; flying along it is always clear.

    Corners in order around the house; edges are straight and stay `offset`
    away from the house outline. 环绕房屋的矩形航线，沿它飞行始终与房屋保持距离。
    """

    def __init__(self, est: HouseEstimate, offset: float) -> None:
        b = math.radians(est.ridge_bearing_deg)
        self.a = (math.cos(b), math.sin(b))        # along the ridge
        self.c = (-math.sin(b), math.cos(b))       # across
        self.center = est.center
        self.hu = est.length / 2 + offset
        self.hv = est.width / 2 + offset

    def to_uv(self, p: Point) -> Point:
        d = (p[0] - self.center[0], p[1] - self.center[1])
        return d[0] * self.a[0] + d[1] * self.a[1], d[0] * self.c[0] + d[1] * self.c[1]

    def to_ne(self, u: float, v: float) -> Point:
        return (self.center[0] + u * self.a[0] + v * self.c[0],
                self.center[1] + u * self.a[1] + v * self.c[1])

    def corners(self) -> List[Point]:
        """Corners going around: (+u,+v), (-u,+v), (-u,-v), (+u,-v)."""
        return [self.to_ne(su * self.hu, sv * self.hv)
                for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    def path(self, frm: Point, to: Point) -> List[Point]:
        """Corners to pass, going the shorter way round, from near frm to near to."""
        cs = self.corners()
        per = 2 * (2 * self.hu + 2 * self.hv)

        def pos(p: Point) -> float:
            u, v = self.to_uv(p)
            L, W = 2 * self.hu, 2 * self.hv
            # Distance along the ring, starting at corner 0 (+u,+v), going towards -u.
            cand = [
                (abs(v - self.hv), self.hu - max(-self.hu, min(self.hu, u))),
                (abs(u + self.hu), L + self.hv - max(-self.hv, min(self.hv, v))),
                (abs(v + self.hv), L + W + max(-self.hu, min(self.hu, u)) + self.hu),
                (abs(u - self.hu), 2 * L + W + max(-self.hv, min(self.hv, v)) + self.hv),
            ]
            return min(cand)[1]

        corner_pos = [0.0, 2 * self.hu, 2 * self.hu + 2 * self.hv, 4 * self.hu + 2 * self.hv]
        p0, p1 = pos(frm), pos(to)
        fwd = (p1 - p0) % per
        if fwd <= per - fwd:
            return [cs[i] for i in self._between(corner_pos, p0, fwd, per, +1)]
        return [cs[i] for i in self._between(corner_pos, p0, per - fwd, per, -1)]

    @staticmethod
    def _between(corner_pos, p0, dist, per, direction) -> List[int]:
        order = range(4) if direction > 0 else range(3, -1, -1)
        hits = []
        for i in order:
            d = ((corner_pos[i] - p0) * direction) % per
            if 1e-6 < d < dist - 1e-6:
                hits.append((d, i))
        return [i for _, i in sorted(hits)]


def inspection_skills(est: HouseEstimate, here: Point, home: Point, limits: Limits,
                      cfg: InspectionConfig = InspectionConfig(),
                      gutters: Optional[Sequence[str]] = None
                      ) -> Tuple[List[Skill], Limits, Dict[str, PhotoTarget]]:
    """
    Photograph every segment of every eave (or of the named gutters).

    Returns the skills to insert, the limits with the measured house box in
    place of rough zones, and what each photo is meant to show.
    返回要插入的技能、用实测房屋框替换粗略禁飞区后的限制、以及每张照片的拍摄对象。
    """
    box = RotatedBox(est.center, est.length, est.width, est.ridge_bearing_deg,
                     est.ridge_height, 'house (measured)')
    new_limits = Limits(min_altitude=limits.min_altitude, max_altitude=limits.max_altitude,
                        max_distance=limits.max_distance, max_hover=limits.max_hover,
                        max_skills=max(limits.max_skills, 80),
                        no_fly=tuple(z for z in limits.no_fly
                                     if not getattr(z, 'name', '').startswith('house'))
                        + (box,),
                        margin=limits.margin, sample_step=limits.sample_step)
    eaves: List[Eave] = [e for e in est.eaves
                         if gutters is None or e.name in gutters]
    ring = Ring(est, cfg.standoff)
    # Visit the nearest eave first. 先去最近的一侧。
    eaves.sort(key=lambda e: math.dist(here, e.point(0.5)))

    skills: List[Skill] = []
    targets: Dict[str, PhotoTarget] = {}
    pos = here
    alt = round(max(e.height for e in eaves) + cfg.above_gutter, 2) if eaves else 0.0
    for e in eaves:
        n = segments_for(e, cfg.segment_length)
        segs = segment_eave(e, n)
        out = e.outward
        views = []
        for i, (a, b) in enumerate(segs, 1):
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            vp = (round(mid[0] + out[0] * cfg.standoff, 2),
                  round(mid[1] + out[1] * cfg.standoff, 2))
            views.append((i, a, b, mid, vp))
        # Walk the eave from the end nearer to where we are. 从较近的一端开始。
        if math.dist(pos, views[-1][4]) < math.dist(pos, views[0][4]):
            views.reverse()
        for corner in ring.path(pos, views[0][4]):
            skills.append(GoTo(round(corner[0], 2), round(corner[1], 2), alt))
        for i, a, b, mid, vp in views:
            label = f'{e.name} segment {i} of {n}'
            skills.append(GoTo(vp[0], vp[1], alt, look_at=(round(mid[0], 2), round(mid[1], 2))))
            skills.append(TakePhoto(label))
            targets[label] = PhotoTarget(label, e.name, i, n, a, b, e.height, (*vp, alt))
        pos = views[-1][4]
    # Finish at the ring point nearest home, so the way home is clear.
    # 最后停在环上离家最近的一侧，保证返航直线畅通。
    u, v = ring.to_uv(home)
    exit_pt = ring.to_ne(max(-ring.hu, min(ring.hu, u)), max(-ring.hv, min(ring.hv, v)))
    for corner in ring.path(pos, exit_pt):
        skills.append(GoTo(round(corner[0], 2), round(corner[1], 2), alt))
    skills.append(GoTo(round(exit_pt[0], 2), round(exit_pt[1], 2), alt))
    return skills, new_limits, targets


# -- M3: eaves of any building, paths around anything ---------------------------------------

@dataclass
class Skipped:
    """A gutter segment that will not be photographed, and why."""

    label: str
    gutter: str
    segment: int
    start: Point
    end: Point
    reason: str


def _extended(e: Eave, margin: float) -> Eave:
    """Return the eave lengthened by margin at both ends."""
    if margin <= 0 or e.length < 1e-6:
        return e
    dn = (e.end[0] - e.start[0]) / e.length
    de = (e.end[1] - e.start[1]) / e.length
    return Eave(e.name, (round(e.start[0] - dn * margin, 3), round(e.start[1] - de * margin, 3)),
                (round(e.end[0] + dn * margin, 3), round(e.end[1] + de * margin, 3)),
                e.height, e.side_bearing_deg)


def plan_inspection(eaves: Sequence[Eave], surfaces, here: Tuple[float, float, float],
                    home: Point, limits: Limits, cfg: InspectionConfig = InspectionConfig(),
                    view=None) -> Tuple[List[Skill], Dict[str, PhotoTarget], List[Skipped]]:
    """
    Photograph every segment of the given eaves, flying around whatever is there.

    surfaces: HeightFields of everything seen (for line of sight); limits must
    already contain them as no-fly zones. Ends at home's position at a safe
    altitude, so a following ReturnHome only has to land.
    对给定屋檐逐段拍照；无法拍摄的段记录原因后跳过。最后回到起飞点上方。
    """
    from .pathplan import choose_viewpoint, Planner, route_home, ViewConfig
    planner = Planner(limits)
    view = view or ViewConfig()
    skills: List[Skill] = []
    targets: Dict[str, PhotoTarget] = {}
    skipped: List[Skipped] = []

    # Viewpoints for every segment first. 先为每一段选好视点。
    plans = []
    for e in eaves:
        e = _extended(e, cfg.end_margin)
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


def surfaces_from_heightmap(hm, res: float = 0.5, min_height: float = 0.8,
                            name: str = 'surroundings'):
    """
    Convert a perception HeightMap (numpy) to a flight-layer HeightField no-fly zone.

    Cells are max-pooled to about `res` (a whole number of the map's cells, on
    the map's own origin, so each coarse cell covers exactly its fine cells).
    Only cells at least min_height above the ground are kept, and heights are
    put back into the altitude frame the vehicle flies in (ground_level added).
    把感知的高度图按最大值降采样为飞行层的高度场禁飞区，高度换回飞行使用的高度坐标系。
    """
    import numpy as np
    from wukong_flight.skills import HeightField
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
