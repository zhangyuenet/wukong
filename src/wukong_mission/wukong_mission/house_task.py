# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
M3 gutter inspection: find the nearest building, survey it in layers, inspect every eave.

The map gives no house: only limits and a search radius. In flight (callbacks
from MissionNode, decisions in a worker thread):
  1. discovery: climb above the takeoff point, scan all around       -> Decide
  2. survey, one station at a time: after every scan, refit the building
     and choose only the next station along the ring (high ring first: the
     far side is first visited high and far off, and seen from above before
     the low ring comes near it)
                                                                      -> Decide
  3. height map of everything seen, eaves on every level, viewpoints  -> photos
Choosing one station at a time matters: from the takeoff point only the near
side is visible, so a ring planned from it could put far-side stations inside
the unseen part of the house. Facing the building at every station, the drone
sees how far it extends before going there.
逐站规划：起飞点只能看到房屋的近侧，一次排好整圈站点可能让远侧站点落进没看到的部分。
M3 水槽检查：地图不给房屋。起飞后环视找到最近的建筑，分两层环绕扫描，
再识别所有楼层的屋檐并逐段拍照。
"""

import dataclasses
from dataclasses import asdict, dataclass
import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from wukong_flight.skills import (Decide, GoTo, HeightField, Limits, ReturnHome, Scan, Skill,
                                  Takeoff, TakePhoto)
from wukong_perception.geometry import merge
from wukong_perception.heightmap import (building_rectangle, buildings, eave_lines, edge_cells,
                                         FreeSpace, ground_height, HeightMap, Rectangle)
from wukong_perception.register import merge_registered, register_scans

from .gutter_task import GutterRequest, GutterTask, matching_gutters
from .inspection import InspectionConfig, plan_inspection, surfaces_from_heightmap
from .pathplan import Planner, route_home
from .world import WorldMap

DISCOVERY = 'discovery'
CLASSIFY = 'what kind of house'
OVERVIEW = 'house overview'
SURVEY_STEP = 'survey step'


@dataclass(frozen=True)
class SurveyConfig:
    discovery_altitude: float = 6.0     # m above the takeoff point, climbing straight up
    discovery_headings: int = 8
    min_building_height: float = 2.5    # m; lower things (cars, fences) are not houses
    min_building_area: float = 20.0     # m^2
    low_offset: float = 7.0             # m out from the outline, low ring
    low_altitude: float = 4.5
    high_offset: float = 10.0           # high ring: far enough to see the near roof edge
    high_above_top: float = 2.5         # m above the building's highest point
    spacing: float = 9.0                # m between stations along a ring
    min_step_deg: float = 15.0          # each new station at least this far round
    # Townhouse rows: survey only this far either side of the takeoff point's
    # position along the row (the target unit and its neighbours' boundaries).
    # 联排：只扫描起飞点沿房屋长轴的投影前后各这么远。
    row_window: float = 12.0
    seen_fill: float = 0.6              # m; holes this small in seen ground count as seen
    row_extension: float = 15.0         # m; a row may go on this far past what is seen
    visit_radius: float = 3.0           # m around where the drone has been: known free
    max_stations: int = 40
    # Photograph this far past a party wall read from the overview photo, so
    # that the boundary between the homes shows in the picture (in Gazebo the
    # reading was up to 1.1 m off; 1.0 m missed one). The part past it is judged
    # only if the photo shows the boundary further out. 在读出的共用墙外多拍这么远，
    # 好让两户的分界出现在照片里（Gazebo 中读数最多差 1.1 米，1.0 米时漏过一次）；
    # 这部分只有照片显示分界更靠外时才判断。
    party_wall_margin: float = 1.5


def clip_to_window(eaves, window: Rectangle, margin: float = 1.0, min_length: float = 1.0,
                   across_margin: Optional[float] = None):
    """
    Keep the parts of eaves within a townhouse survey window (along the row's long axis).

    Eaves across the row (end units' sides) are kept whole if inside.
    只保留扫描窗口内（沿联排长轴方向）的屋檐部分。
    """
    from wukong_perception.geometry import Eave
    lo, hi = -window.length / 2 - margin, window.length / 2 + margin
    # An end unit's side eave lies past the unit's end by the roof overhang.
    # 端户侧面的屋檐比户界多出屋顶挑出的距离。
    am = margin if across_margin is None else across_margin
    alo, ahi = -window.length / 2 - am, window.length / 2 + am
    out = []
    for e in eaves:
        u0, v0 = window.to_uv(e.start)
        u1, v1 = window.to_uv(e.end)
        # Across the row (an end unit's side): judged by direction, not by exact
        # coordinates, since a measured eave is never exactly square to the row.
        # 横跨联排的屋檐（端户侧面）按方向判断：测得的屋檐不会与长轴严格垂直。
        if abs(u1 - u0) < 0.5 * e.length:
            if alo <= (u0 + u1) / 2 <= ahi:
                out.append(e)
            continue
        if max(u0, u1) < lo or min(u0, u1) > hi:
            continue

        def at(u):
            t = (u - u0) / (u1 - u0)
            return (round(e.start[0] + t * (e.end[0] - e.start[0]), 3),
                    round(e.start[1] + t * (e.end[1] - e.start[1]), 3))
        a, b = max(min(u0, u1), lo), min(max(u0, u1), hi)
        if b - a < min_length:
            continue
        s, t = (at(a), at(b)) if u0 < u1 else (at(b), at(a))
        out.append(Eave(e.name, s, t, e.height, e.side_bearing_deg))
    return out


def _long_axis(r: Rectangle) -> Rectangle:
    """Return the same rectangle with u along its longer side (a row's long axis)."""
    if r.width > r.length:
        return Rectangle(r.center, r.width, r.length, (r.bearing_deg + 90) % 360)
    return r


def _dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if di * di + dj * dj <= k * k:
                out |= np.roll(np.roll(mask, di, 0), dj, 1)
    return out


def ring_stations(rect: Rectangle, offset: float, spacing: float,
                  end_offsets: Optional[Tuple[float, float]] = None
                  ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Stations around an outline: (position, point to face), corners included.

    Each station faces the nearest point of the outline, so the depth camera
    looks straight at the wall and roof edge in front of it. end_offsets
    (at -u, at +u) replace offset along the length: a townhouse window ends at
    a party wall with a neighbour beyond it, so the ring must not grow there.
    沿外扩矩形布站，每个站点朝向建筑轮廓上离它最近的点。end_offsets 单独指定长轴两端的
    外扩距离：联排窗口在分户墙处结束，墙外是邻居，那一端不应外扩。
    """
    lo, hi = end_offsets if end_offsets is not None else (offset, offset)
    hv = rect.width / 2 + offset
    u_lo, u_hi = -rect.length / 2 - lo, rect.length / 2 + hi
    corners = [(u_hi, hv), (u_lo, hv), (u_lo, -hv), (u_hi, -hv)]
    out = []
    for k in range(4):
        (u0, v0), (u1, v1) = corners[k], corners[(k + 1) % 4]
        n = max(1, int(math.ceil(math.hypot(u1 - u0, v1 - v0) / spacing)))
        for i in range(n):
            u, v = u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * i / n
            face = (max(-rect.length / 2, min(rect.length / 2, u)),
                    max(-rect.width / 2, min(rect.width / 2, v)))
            out.append((rect.to_ne(u, v), rect.to_ne(*face)))
    return out


class HouseTask(GutterTask):
    """State of one M3 inspection; answers MissionNode's callbacks."""

    def __init__(self, world: WorldMap, request: GutterRequest, start: Tuple[float, float],
                 out_dir: str, judge, instruction: str,
                 cfg: InspectionConfig = InspectionConfig(),
                 survey: SurveyConfig = SurveyConfig(), log=print) -> None:
        # Not GutterTask.__init__: there is no rough house on this map.
        from concurrent.futures import ThreadPoolExecutor
        import threading
        self.world, self.request, self.start = world, request, start
        self.out_dir, self.judge, self.instruction = out_dir, judge, instruction
        self.cfg, self.survey, self.log = cfg, survey, log
        self.depth = None
        self.estimate = None
        self.targets = {}
        self.skipped = []
        self.decide_note = ''
        # Aligning scans to each other (register.py) made things worse on real
        # Gazebo data; off until better localisation (M4). 扫描配准在 Gazebo 数据上
        # 反而更差，等 M4 有更好的定位后再用。
        self.register = False
        self.rect: Optional[Rectangle] = None          # what the survey goes round
        self.full_rect: Optional[Rectangle] = None     # the whole building seen so far
        # Progress round the building is measured about this point, fixed after
        # discovery: the outline's centre moves as more of the building is seen.
        # 绕行进度以发现阶段确定的固定点为参考；外接矩形的中心会随着看到更多而移动。
        self.anchor: Optional[Tuple[float, float]] = None
        self.visited: List[Tuple[float, float]] = [start]   # where the drone has been
        # Photo-based judgement of the house (classify.py); None: use the site
        # configuration only. 根据照片判断房屋类型；为 None 时只用场地配置。
        self.classifier = None
        self.overview: Optional[Tuple[str, tuple]] = None   # (photo path, pose)
        self.house = None                                    # Classification
        self.is_row = world.townhouse
        self.look: Optional[Tuple[float, float]] = None     # where the overview photo looked
        self.refusal: Optional[str] = None                   # why nothing was inspected
        self.free = FreeSpace(start)                         # where rays went unhindered
        self._rays_added = 0
        self.building_top = 0.0
        self.ring = 'high'             # high ring first, then low
        self.progress = 0.0            # radians travelled round the building in this ring
        self.stations = 0
        self.eaves = []
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='judge')
        self._lock = threading.Lock()
        self._futures = []

    # -- before takeoff ------------------------------------------------------------------

    def plan(self) -> List[Skill]:
        """Climb above the takeoff point and look all around; the rest is decided in flight."""
        s = self.survey
        n0, e0 = self.start
        alt = s.discovery_altitude
        skills: List[Skill] = [Takeoff(alt)]
        for k in range(s.discovery_headings):
            b = 2 * math.pi * k / s.discovery_headings
            look = (round(n0 + 10 * math.cos(b), 2), round(e0 + 10 * math.sin(b), 2))
            skills += [GoTo(n0, e0, alt, look_at=look), Scan(f'look around {k + 1}')]
        return skills + [Decide(DISCOVERY), ReturnHome()]

    # -- in flight -----------------------------------------------------------------------

    def _heightmap(self, survey_only: bool = False) -> Optional[HeightMap]:
        """
        Height map of the scans; survey_only leaves out the discovery scans.

        The discovery scans are far away, coarse, and taken right after takeoff,
        when the position estimate is least settled: good for finding the house,
        not for measuring its eaves. 环视扫描距离远、分辨率低，又是刚起飞时拍的，
        位置估计最不稳定：用来找房子可以，用来测量屋檐不行。

        The ground level still comes from all scans. The depth camera looks
        straight ahead, so a station facing a wall a few metres away sees no
        ground at all; only the discovery scans (and stations looking past the
        building) do. Without them the tallest peak near zero was a wall band
        (1.3 m), and every eave came out 1.3 m too low.
        地面高度仍用全部扫描估计：深度相机水平朝前，正对几米外墙面的站点完全看不到地面，
        只有环视扫描（以及从房屋旁边看过去的站点）能看到。不用它们时，0 附近最高的峰是墙面
        （1.3 米），所有屋檐都低了 1.3 米。
        """
        if not self.depth:
            return None
        merged = merge_registered if self.register else merge
        clouds = list(self.depth.clouds)
        ground = None
        if survey_only:
            labels = list(getattr(self.depth, 'labels', []))
            kept = [c for c, lab in zip(clouds, labels) if not str(lab).startswith('look around')]
            if len(kept) >= 4:
                everything = merged(clouds)
                if everything is not None:
                    ground = ground_height(everything[:, 2])
                clouds = kept
        pts = merged(clouds)
        if pts is None:
            return None
        return HeightMap.from_points(pts, res=0.2, ground_level=ground)

    def on_decide(self, label: str, pose) -> Tuple[Optional[list], Optional[Limits], str]:
        self.visited.append((pose[0], pose[1]))
        if label == DISCOVERY:
            return self._after_discovery(pose)
        if label == SURVEY_STEP:
            return self._survey_step(pose)
        if label == CLASSIFY:
            why = self._classify()
            if why:
                return self._refuse(why)
            return self._survey_step(pose)
        return None, None, f'unknown decision {label}'

    def _after_discovery(self, pose):
        s = self.survey
        hm = self._heightmap()
        if hm is None:
            return None, None, 'no depth scans'
        found = [b for b in buildings(hm, self.start, min_height=s.min_building_height,
                                      min_area=s.min_building_area)
                 if b.distance <= self.world.search_radius]
        if not found:
            self.decide_note = f'no building within {self.world.search_radius:g} m'
            return None, None, self.decide_note
        target = found[0]
        self._update_building(hm, target)
        r = self.rect
        self.anchor = r.center
        self.log(f'[house] nearest building {target.distance:.1f} m away: about '
                 f'{r.length:.0f} x {r.width:.0f} m so far, at least {target.top:.1f} m high')
        if self.classifier is None:
            return self._refuse('no way to judge what kind of house this is')
        # Face the building, take one photo, and judge what kind of house it is
        # before choosing what to survey. 正对建筑拍一张照片，先判断房屋类型再决定扫描范围。
        here = (round(pose[0], 2), round(pose[1], 2), round(pose[2], 2))
        look = self.look = self._front_point()
        # Steady: the ruler is drawn with the recorded pose. 刻度尺按记录的位姿绘制，须等稳定。
        skills = [GoTo(*here, look_at=look), TakePhoto(OVERVIEW, steady=True), Decide(CLASSIFY)]
        planner = Planner(self._survey_limits(hm))
        home = route_home(planner, here, self.start) or []
        return (skills + [GoTo(*p) for p in home] + [ReturnHome()], self._survey_limits(hm),
                'photographing the house to judge what kind it is')

    def _front_point(self, keep_in: float = 3.0) -> Tuple[float, float]:
        """
        Return the point of the building's front nearest the takeoff point, for the overview.

        The photo's centre line marks the home to inspect, so it must fall on the
        home in front of the takeoff point. Aiming at the centre of what discovery
        saw put it on a neighbour when more of the row showed on one side. Kept
        keep_in metres from the corners so it falls on a home, not on its edge
        (discovery measures the ends to within a couple of metres).
        照片中线标出要检查的那一户，所以必须落在起飞点正前方的那一户上。对准环视所见部分的
        中心时，若一侧看到的更多，中线会落在邻居家。离两端至少 keep_in 米，避免落在墙角上
        （环视测出的两端可能差两米）。
        """
        from .classify import front_of
        a, b = (np.array(p, float) for p in front_of(self.full_rect, self.start))
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return (round(float(a[0]), 2), round(float(a[1]), 2))
        t = float((np.array(self.start, float) - a) @ d) / length
        t = min(max(t, min(keep_in, length / 2)), max(length - keep_in, length / 2))
        p = a + d * (t / length)
        return (round(float(p[0]), 2), round(float(p[1]), 2))

    def _target(self, hm: HeightMap):
        s = self.survey
        found = [b for b in buildings(hm, self.start, min_height=s.min_building_height,
                                      min_area=s.min_building_area)
                 if b.distance <= self.world.search_radius]
        return found[0] if found else None

    def _update_building(self, hm: HeightMap, target) -> None:
        self.full_rect = building_rectangle(hm, target)
        self.rect = self.full_rect
        if self.is_row:
            self.rect = self._row_window(self.full_rect)
        self.building_top = target.top + hm.ground_level

    def _row_window(self, r: Rectangle, around_unit: float = 2.0) -> Rectangle:
        """
        Return the part of a townhouse row to survey, as a rectangle.

        The target unit and around_unit metres either side when the photo told
        where it is; otherwise row_window either side of the takeoff point.
        已知户界时取该户两侧各外扩 around_unit 米；否则取起飞点两侧各 row_window 米。
        """
        # The row's long axis is the rectangle's longer side. 长轴是矩形的长边。
        if r.width > r.length:
            r = Rectangle(r.center, r.width, r.length, (r.bearing_deg + 90) % 360)
        unit = self._unit_range(r)
        if unit is not None:
            lo, hi = unit[0] - around_unit, unit[1] + around_unit
        else:
            u0, _ = r.to_uv(self.start)
            lo, hi = u0 - self.survey.row_window, u0 + self.survey.row_window
        lo = max(-r.length / 2, lo)
        hi = min(r.length / 2, hi)
        if hi - lo >= r.length - 1e-6:
            return r
        c = r.to_ne((lo + hi) / 2, 0.0)
        return Rectangle(c, hi - lo, r.width, r.bearing_deg)

    def on_photo(self, label: str, path: str, pose) -> None:
        if label == OVERVIEW:              # for judging the house, not a gutter photo
            self.overview = (path, pose)
            return
        super().on_photo(label, path, pose)

    def _refuse(self, why: str):
        """Insert nothing: the plan goes on with its checked way home, nothing inspected."""
        self.refusal = why
        self.decide_note = f'not inspecting: {why}'
        self.log(f'[house] {self.decide_note}; flying home')
        return None, None, self.decide_note

    def _classify(self) -> Optional[str]:
        """
        Judge the house from the overview photo; return why not to inspect it, if so.

        Fails closed. Without knowing the kind of house (and, in a townhouse row,
        where the home to inspect begins and ends) the drone would inspect the
        neighbours' gutters: a row taken for one long house, or a guessed
        stretch of it. Then it flies home without inspecting anything.
        不确定时不检查：不知道房屋类型、或联排中目标户的户界时，可能把联排当成一栋长房子，
        或按猜测的范围检查到邻居家的水槽。这时直接返航，什么都不检查。
        """
        from .classify import front_of
        if self.overview is None or self.full_rect is None:
            return 'no photo of the house to judge it from'
        path, pose = self.overview
        front = self._wall_line(front_of(self.full_rect, self.start))
        # Draw the ruler 1.5 m below the camera: across the front, a little below
        # the middle of the picture, numbers in view. (At 2.5 m above the ground,
        # seen from 6 m and 10 m away, it fell off the bottom of the picture.)
        # 刻度尺画在相机下方 1.5 米：横穿房屋正面、位于画面中部偏下，数字可见。
        height = pose[2] + 0.26 - 1.5
        try:
            c = self.classifier.classify_house(path, pose, front, height, self.out_dir)
        except Exception as e:   # noqa: B902 - network, model, image: fly home
            return f'could not judge what kind of house this is ({e})'
        self.house = c
        self.log(f'[house] {c.describe()}: {c.notes}')
        if c.kind == 'not_a_house':
            return 'the nearest building does not look like a house'
        if c.confidence == 'low':
            return 'unsure what kind of house this is'
        self.is_row = c.kind == 'townhouse_row'
        if not self.is_row:
            return None
        if c.unit is None:
            return 'could not tell where the home to inspect begins and ends in the row'
        # The home read off the ruler must be the one the drone looked at.
        # 从刻度尺上读出的那一户必须是无人机正对的那一户。
        r = _long_axis(self.full_rect)
        lo, hi = self._unit_range(r)
        u = r.to_uv(self.look)[0] if self.look is not None else (lo + hi) / 2
        if not lo - 0.3 <= u <= hi + 0.3:
            return (f'the home read from the photo ({lo:+.1f}..{hi:+.1f} m along the row) is '
                    f'not the one the drone looked at ({u:+.1f} m)')
        return None

    def _wall_line(self, front):
        """
        Move the front line from the roof's outer edge back onto the wall.

        Unit boundaries (colour changes, downspouts) are on the wall, which sits
        behind the overhang and gutter; a ruler drawn on the outer edge reads
        them off by up to a metre when seen at an angle (parallax).
        户界（墙面颜色变化、落水管）在墙上，比屋檐外沿靠里；刻度尺画在外沿时，
        斜着看会因视差偏差一米左右。所以把刻度尺平移到墙面上。
        """
        if not self.depth or not self.depth.clouds:
            return front
        a, b = np.array(front[0], float), np.array(front[1], float)
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return front
        d /= length
        inward = np.array([-d[1], d[0]])
        if (np.array(self.full_rect.center) - a) @ inward < 0:
            inward = -inward
        pts = np.vstack(self.depth.clouds)
        g = self._ground_level()
        low = pts[(pts[:, 2] - g > 0.8) & (pts[:, 2] - g < 2.5)]
        if not len(low):
            return front
        rel = low[:, :2] - a
        along = rel @ d
        depth_in = rel @ inward
        near = (along > 0) & (along < length) & (depth_in > -0.5) & (depth_in < 3.0)
        if near.sum() < 50:
            return front
        inset = float(np.clip(np.percentile(depth_in[near], 10), 0.0, 1.5))
        return (tuple(a + inward * inset), tuple(b + inward * inset))

    def _ground_level(self) -> float:
        """Ground level in the vehicle's altitude frame (from the latest height map)."""
        hm = self._heightmap()
        return hm.ground_level if hm is not None else 0.0

    def _unit_range(self, r: Rectangle) -> Optional[Tuple[float, float]]:
        """
        Return the target unit's extent along r's long axis, if known.

        An end within a metre of the row's end is the row's end (with its roof
        overhang), not a party wall. 离整排末端不到 1 米的一端就是整排的尽头（含挑出的屋檐）。
        """
        if self.house is None or self.house.unit is None:
            return None
        u0, _ = r.to_uv(self.house.unit[0])
        u1, _ = r.to_uv(self.house.unit[1])
        lo, hi = min(u0, u1), max(u0, u1)
        # An end unit: its outer end (the one nearer a row end) is the row's end,
        # however far off the reading is. Ruler and measured outline each err by a
        # metre or two (pose, drift); a 3 m limit once left an end unit's gable,
        # gutter and all, outside the survey. 端户：靠近整排末端的那一端就是整排的末端，
        # 不论读数差多少。刻度尺和实测轮廓各有一两米误差，曾因 3 米的限制把端户的山墙
        # 连同水槽一起排除在扫描范围外。
        if self.house.end_unit:
            if lo + r.length / 2 <= r.length / 2 - hi:
                lo = -r.length / 2
            else:
                hi = r.length / 2
            return lo, hi
        if lo + r.length / 2 < 1.0:
            lo = -r.length / 2
        if r.length / 2 - hi < 1.0:
            hi = r.length / 2
        return lo, hi

    def _unit_rect(self) -> Optional[Rectangle]:
        r = self.full_rect
        if r is None:
            return None
        if r.width > r.length:
            r = Rectangle(r.center, r.width, r.length, (r.bearing_deg + 90) % 360)
        unit = self._unit_range(r)
        if unit is None:
            return None
        lo, hi = max(-r.length / 2, unit[0]), min(r.length / 2, unit[1])
        return Rectangle(r.to_ne((lo + hi) / 2, 0.0), hi - lo, r.width, r.bearing_deg)

    def _limits(self, zones) -> Limits:
        lim = self.world.limits
        return Limits(min_altitude=lim.min_altitude, max_altitude=lim.max_altitude,
                      max_distance=lim.max_distance, max_hover=lim.max_hover,
                      max_skills=max(lim.max_skills, 200), no_fly=tuple(zones),
                      margin=lim.margin, sample_step=lim.sample_step)

    def _survey_limits(self, hm: HeightMap) -> Limits:
        """
        Everything seen, plus the unseen parts of the building as tall as allowed.

        Inside the building's outline (grown by a metre), cells with no data at
        all may hide anything up to the tallest building the map allows; cells
        already seen count at their measured height. Unknown space is an
        obstacle. 外接矩形内没有任何数据的格子按地图允许的最高建筑高度禁飞，
        已看到的格子按实测高度：未知空间就是障碍。
        """
        seen = surfaces_from_heightmap(hm, name='surroundings (seen)')
        unknown = self._unknown_cells(hm, seen.res)
        return self._limits((seen, unknown))

    def _unknown_cells(self, hm: HeightMap, res: float) -> HeightField:
        """
        Mark the building's unseen cells as tall as the map allows (or as rays show).

        Covers the outline grown by a metre across, and along a townhouse row's
        long axis by row_extension: a row goes on past what has been seen. A ray
        that crossed a cell unhindered caps what can stand there. Unseen space
        elsewhere (near-ground blind spots, trees) is M4's obstacle work.
        建筑外接矩形范围内没看到的格子按允许的最高建筑高度禁飞；联排沿长轴方向再向两侧
        延伸，因为联排会在看到的部分之外继续。射线无阻经过的格子按射线高度封顶。
        其他未知空间（近地盲区、树木）属于 M4 的避障工作。
        """
        s = self.survey
        top = self.world.max_building_height + hm.ground_level
        self._update_free_space()
        seen = _dilate_mask(np.isfinite(hm.top), max(1, int(round(s.seen_fill / hm.res))))
        r = self.full_rect
        along = s.row_extension if self.is_row else 1.0
        if r.width > r.length:             # long axis = the longer side
            r = Rectangle(r.center, r.width, r.length, (r.bearing_deg + 90) % 360)
        hu, hv = r.length / 2 + along, r.width / 2 + 1.0
        cells = {}
        rows, cols = hm.top.shape
        for a in range(int(2 * hu / res) + 1):
            for b in range(int(2 * hv / res) + 1):
                n, e = r.to_ne(-hu + a * res, -hv + b * res)
                i, j = hm.cell(n, e)
                if 0 <= i < rows and 0 <= j < cols and seen[i, j]:
                    continue                     # seen: its measured height applies
                if any(math.hypot(n - v[0], e - v[1]) <= s.visit_radius for v in self.visited):
                    continue                     # the drone has been here
                h = min(top, self.free.ceiling_at(n, e) - 0.3)
                if h <= hm.ground_level + 0.5:
                    continue                     # rays show it empty down to the ground
                cells[(int(np.floor((n - hm.origin[0]) / res)),
                       int(np.floor((e - hm.origin[1]) / res)))] = h
        return HeightField(hm.origin, res, cells, 'house (not seen yet)')

    def _update_free_space(self) -> None:
        rays = getattr(self.depth, 'rays', [])
        for cam, ends, hit in rays[self._rays_added:]:
            self.free.add(cam, ends, hit)
        self._rays_added = len(rays)

    def _survey_step(self, pose):
        hm = self._heightmap()
        target = self._target(hm) if hm is not None else None
        if target is None:
            self.decide_note = 'lost the building during the survey'
            return None, None, self.decide_note
        self._update_building(hm, target)
        return self._next_station(pose, hm)

    def _ring_end_offsets(self, offset: float, tol: float = 0.5) -> Tuple[float, float]:
        """
        Return how far the ring reaches past each end of the survey window (at -u, at +u).

        A detached house: offset at both ends, all round. A townhouse window: offset
        only at an end that is the row's own end (an end unit's gable); at a party
        wall it stops, or it would put stations in front of every neighbour.
        独立屋两端都外扩。联排窗口只在整排真正的尽头（端户的山墙）外扩；在分户墙处不外扩，
        否则会在每一户邻居门前布站。
        """
        if not self.is_row or self.full_rect is None:
            return offset, offset
        f, w = self.full_rect, self.rect
        if f.width > f.length:             # long axis = the longer side
            f = Rectangle(f.center, f.width, f.length, (f.bearing_deg + 90) % 360)
        ends = sorted(w.to_uv(f.to_ne(s * f.length / 2, 0.0))[0] for s in (-1, 1))
        lo = offset if ends[0] >= -w.length / 2 - tol else 0.0
        hi = offset if ends[1] <= w.length / 2 + tol else 0.0
        return lo, hi

    def _angle(self, p) -> float:
        c = self.anchor
        return math.atan2(p[1] - c[1], p[0] - c[0])

    def _next_station(self, pose, hm: HeightMap):
        """
        Fly to the next station round the building, scan, and decide again.

        Stations are recomputed from the latest outline. The next one is the
        nearest station counter-clockwise, at least min_step_deg on; after a
        full turn the low ring is done and the high ring starts; after that the
        inspection is planned. 每次按最新轮廓重新计算站点，逆时针取下一个；
        低圈绕满一圈后换高圈，高圈完成后规划巡检。
        """
        s, lim = self.survey, self.world.limits
        if self.stations >= s.max_stations:
            return self._after_survey(pose)
        ground = hm.ground_level
        if self.ring == 'low':
            offset, alt = s.low_offset, ground + s.low_altitude
        else:
            offset = s.high_offset
            alt = min(max(self.building_top + s.high_above_top, ground + s.low_altitude + 1.0),
                      lim.max_altitude)
        limits = self._survey_limits(hm)
        planner = Planner(limits)
        here = (pose[0], pose[1], pose[2])
        a0 = self._angle(here)
        # The very first station may be straight ahead; later ones must move on.
        # Counted in stations, not degrees: the first station can sit at 0 deg.
        # 按站数判断而不是角度：第一站可能正好在 0 度处。
        first = self.ring == 'high' and self.stations == 0
        min_step = 0.0 if first else math.radians(s.min_step_deg)
        candidates = []
        keep_off = self.world.limits.margin + 1.0
        for (n, e), face in ring_stations(self.rect, offset, s.spacing,
                                          self._ring_end_offsets(offset)):
            # Never a station over the building (a townhouse row continues past
            # the survey window). 站点不放在建筑上方（联排在扫描窗口外还在延伸）。
            u, v = self.full_rect.to_uv((n, e))
            if (abs(u) < self.full_rect.length / 2 + keep_off
                    and abs(v) < self.full_rect.width / 2 + keep_off):
                continue
            d = (self._angle((n, e)) - a0) % (2 * math.pi)
            if d < min_step:
                continue
            candidates.append((d, (round(n, 2), round(e, 2), round(alt, 2)), face))
        candidates.sort()
        for d, goal, face in candidates:
            if self.progress + d > 2 * math.pi + math.radians(5):
                break                                  # would go past the start
            route = planner.path(here, goal)
            if route is None:
                continue
            # Bring a checked way home from the new station: it replaces the
            # rest of the plan, and is what the drone flies if a later
            # decision fails. 附带从新站点安全回家的路线；后续判断失败时就按它返航。
            home = route_home(planner, goal, self.start)
            if home is None:
                continue
            self.progress += d
            self.stations += 1
            skills = [GoTo(*p) for p in route[:-1]]
            skills += [GoTo(*goal, look_at=(round(face[0], 2), round(face[1], 2))),
                       Scan(f'{self.ring} ring station {self.stations}'), Decide(SURVEY_STEP)]
            skills += [GoTo(*p) for p in home] + [ReturnHome()]
            note = (f'{self.ring} ring: station {self.stations}, '
                    f'{math.degrees(self.progress):.0f} deg round')
            self.decide_note = note
            return skills, limits, note
        # This ring is done (or nothing more is reachable). 这一圈完成。
        if self.ring == 'high':
            self.ring, self.progress = 'low', 0.0
            self.log(f'[house] high ring done after {self.stations} stations; going lower')
            return self._next_station(pose, hm)
        self.log(f'[house] survey done: {self.stations} stations')
        return self._after_survey(pose)

    def _after_survey(self, pose):
        hm = self._heightmap(survey_only=True)
        if hm is None:
            return None, None, 'no depth scans'
        aligned, reg = (register_scans(self.depth.clouds) if self.register
                        else (self.depth.clouds, None))
        np.save(os.path.join(self.out_dir, 'survey_points.npy'),
                np.vstack(aligned).astype(np.float32))
        with open(os.path.join(self.out_dir, 'registration.json'), 'w', encoding='utf-8') as f:
            json.dump({'labels': list(getattr(self.depth, 'labels', [])),
                       'shifts': reg.shifts if reg else [], 'scores': reg.scores if reg else []},
                      f, indent=1)
        # Every scan on its own, with the pose it was converted with, for checking
        # how well scans line up. 每次扫描单独保存，连同转换时用的位姿，便于检查对齐。
        np.savez_compressed(
            os.path.join(self.out_dir, 'scans.npz'),
            **{f'scan{k:02d}': c.astype(np.float32) for k, c in enumerate(self.depth.clouds)},
            poses=np.array(getattr(self.depth, 'poses', []) or [[0.0] * 6]),
            labels=np.array(getattr(self.depth, 'labels', [])))
        target = self._target(hm)
        if target is None:
            self.decide_note = 'the building was lost after the survey'
            return None, None, self.decide_note
        eaves = eave_lines(hm, edge_cells(hm, target.cells))
        if self.is_row:
            # Outside the survey window the row was only glimpsed past the target
            # unit: fragments seen from there are not reliable eaves.
            # 测绘窗口外的部分只是斜着瞥见，拼出的零碎屋檐不可靠，直接裁掉。
            eaves = clip_to_window(eaves, self.rect, margin=1.0)
        self.eaves = eaves
        g = hm.ground_level
        self.log(f'[house] {len(eaves)} eave(s): '
                 + ', '.join(f'{e.name} ({e.length:.1f} m at {e.height - g:.1f} m)'
                             for e in eaves))
        with open(os.path.join(self.out_dir, 'eaves.json'), 'w', encoding='utf-8') as f:
            json.dump({'ground_level': g, 'eaves': [asdict(e) for e in eaves]}, f, indent=2)
        unit_rect = self._unit_rect() if self.is_row else None
        if self.is_row and unit_rect is None:
            return self._refuse('no unit boundaries to keep the inspection to the target home')
        walls: List[Tuple[float, float]] = []
        if unit_rect is not None:
            walls = self._party_walls()
            eaves = self._unit_eaves(eaves, unit_rect, walls)
        names = matching_gutters(eaves, self.request.sides)
        chosen = [e for e in eaves if names is None or e.name in names]
        if not chosen:
            self.decide_note = (f'no gutter faces {", ".join(self.request.sides)}; found '
                                + ', '.join(e.name for e in eaves))
            return None, None, self.decide_note
        hf = surfaces_from_heightmap(hm)
        # What has not been seen is still an obstacle while photographing: a
        # townhouse row's far ends were only glimpsed from afar, and a way round
        # them through unseen space ran along their walls.
        # 拍照阶段没看到的空间同样是障碍：联排的两端只在远处看过，
        # 穿过未知空间绕行的航线曾贴着它们的墙飞。
        unknown = self._unknown_cells(hm, hf.res)
        limits = self._limits((hf, unknown))
        cfg = self.cfg
        if unit_rect is not None:
            cfg = dataclasses.replace(cfg, end_margin=0.0)    # _unit_eaves has done it
        skills, targets, skipped = plan_inspection(chosen, [hf], tuple(pose[:3]), self.start,
                                                   limits, cfg)
        skills.append(ReturnHome())          # ends at home already: replaces the rest
        if unit_rect is not None:
            self._mark_home(targets, walls)
        self.targets, self.skipped = targets, skipped
        for sk in skipped:
            self.log(f'[house] skipping {sk.label}: {sk.reason}')
        self.decide_note = (f'{len(targets)} gutter segment(s) to photograph'
                            + (f', {len(skipped)} skipped' if skipped else ''))
        return skills, limits, self.decide_note

    def _party_walls(self) -> List[Tuple[float, float]]:
        """
        Return where the party walls were read, as points on the front.

        One for an end unit (its other end is the row's end), two otherwise.
        Points, not positions along the fitted outline: carried to the back
        along the outline, they landed 1.8 m off in Gazebo (outline 4 deg off).
        返回读出的共用墙位置（正面上的点）。端户只有一面共用墙。用点而不是外接矩形长轴上的
        位置：沿外接矩形搬到背面时，Gazebo 中偏了 1.8 米（矩形方向偏 4 度）。
        """
        r = _long_axis(self.full_rect)
        lo, hi = self._unit_range(r)
        a, b = sorted(self.house.unit, key=lambda p: r.to_uv(p)[0])
        walls = []
        if lo > -r.length / 2 + 1e-6:
            walls.append(tuple(a))
        if hi < r.length / 2 - 1e-6:
            walls.append(tuple(b))
        return walls

    def _home_point(self) -> Tuple[float, float]:
        if self.look is not None:
            return self.look
        return tuple(np.mean(np.array(self.house.unit, float), axis=0))

    def _along_row(self, p, q) -> bool:
        r = _long_axis(self.full_rect)
        length = math.dist(p, q)
        return length > 1e-6 and abs(r.to_uv(q)[0] - r.to_uv(p)[0]) >= 0.5 * length

    def _span(self, p, q, walls):
        """
        Return (a, d, t_home, lower, upper) for the line p -> q.

        Walls and the home's point are carried onto the line square to it (each
        eave's own direction, measured, not the outline's); lower/upper are the
        walls on either side of the home, in metres from p (+-inf: the row's end).
        把共用墙和目标户的参考点垂直投影到这条线上（用屋檐自己的方向，而不是外接矩形的）。
        """
        a = np.array(p, float)
        d = np.array(q, float) - a
        d /= np.linalg.norm(d)
        t_home = float((np.array(self._home_point(), float) - a) @ d)
        tws = [float((np.array(w, float) - a) @ d) for w in walls]
        lower = max([x for x in tws if x < t_home], default=-math.inf)
        upper = min([x for x in tws if x > t_home], default=math.inf)
        return a, d, t_home, lower, upper

    def _unit_eaves(self, eaves, unit_rect: Rectangle, walls):
        """
        Keep the target home's part of each eave, and party_wall_margin past a party wall.

        Ends at the row's end (an end unit's corner) get end_margin instead, as
        every eave end does on a detached house; side gutters across the row are
        kept whole if they are at the home's end. The part past a party wall is
        photographed so the boundary shows; it is judged only if a photo shows
        the boundary there. 保留每条屋檐属于目标户的部分，共用墙外多留 party_wall_margin
        （好让分界出现在照片里，只有照片显示分界在那里时才判断）；整排尽头（端户转角）
        与独立屋一样外延 end_margin；横跨联排的侧面水槽在目标户一端时整条保留。
        """
        from wukong_perception.geometry import Eave
        m, extra = self.survey.party_wall_margin, self.cfg.end_margin
        across, out = [], []

        def eave(e, a, d, lo, hi):
            p, q = a + d * lo, a + d * hi
            return Eave(e.name, (round(float(p[0]), 3), round(float(p[1]), 3)),
                        (round(float(q[0]), 3), round(float(q[1]), 3)), e.height,
                        e.side_bearing_deg)
        for e in eaves:
            if e.length < 1e-6:
                continue
            if not self._along_row(e.start, e.end):
                across.append(e)
                continue
            a, d, _, lower, upper = self._span(e.start, e.end, walls)
            if min(upper, e.length) - max(lower, 0.0) < 0.3:
                continue                      # not along the target home at all
            lo = max(lower - m, 0.0) if lower > -math.inf else -extra
            hi = min(upper + m, e.length) if upper < math.inf else e.length + extra
            out.append(eave(e, a, d, lo, hi))
        for e in clip_to_window(across, unit_rect, margin=m, across_margin=1.0):
            a = np.array(e.start, float)
            d = (np.array(e.end, float) - a) / e.length
            out.append(eave(e, a, d, -extra, e.length + extra))
        return out

    def _mark_home(self, targets, walls) -> None:
        """
        Tell each photo along the row where the home is and which part is surely its.

        home: the point of the gutter's line square to where the overview photo
        looked (inside the home). claim: the part of the segment between the
        party walls as read (or up to the row's end). 告诉每张沿联排方向的照片：
        目标户在哪里（总览照片正对处），以及按读数确定属于目标户的那一部分。
        """
        for t in targets.values():
            if not self._along_row(t.start, t.end):
                continue                      # across the row: all of it is the home's
            a, d, t_home, lower, upper = self._span(t.start, t.end, walls)
            length = math.dist(t.start, t.end)

            def at(s, a=a, d=d):
                p = a + d * s
                return (round(float(p[0]), 3), round(float(p[1]), 3))
            t.home = at(t_home)
            t.walls = (lower > -math.inf, upper < math.inf)
            ca, cb = max(0.0, lower), min(length, upper)
            t.claim = (at(ca), at(ca)) if cb <= ca else (at(ca), at(cb))
