# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
M2 gutter inspection of a house whose position is known only roughly.

Survey the house, find its eaves, photograph each gutter segment, and judge
the photos segment by segment.

Flow in flight (all callbacks from MissionNode):
  Scan     -> DepthCamera.capture           (depth frame -> points)
  Decide   -> on_decide: fit_house + inspection_skills  (worker thread)
  photo    -> on_photo: crop to the segment, judge in the background
M2 水槽检查：环绕扫描粗略已知的房屋，找出屋檐，逐段拍照，逐段判断。
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from wukong_flight.skills import Limits, ReturnHome, Skill
from wukong_perception.geometry import (fit_house, FitError, HouseEstimate, merge, project,
                                        RGB_X500_DEPTH)

from .inspection import inspection_skills, InspectionConfig, PhotoTarget, survey_skills
from .judge import Verdict
from .world import RoughHouse, WorldMap

COMPASS_BEARING = {'north': 0, 'northeast': 45, 'east': 90, 'southeast': 135, 'south': 180,
                   'southwest': 225, 'west': 270, 'northwest': 315}

SEGMENT_QUESTION = (
    'This picture is cropped to ONE segment of a roof gutter: the white channel running '
    'across the picture along the bottom edge of the roof right in front of the camera, '
    'seen from the side and slightly above. Judge only that gutter, the one closest to the '
    'camera. Ignore any other gutter further away, for example one going away from the '
    'camera around a corner of the roof: it is inspected separately. The darkened strips at '
    'the left and right edges belong to the neighbouring segments: ignore anything that is '
    'only in those strips. Is there anything in this gutter segment (leaves, debris, a pile '
    'of material) that means it needs cleaning? Seen from this angle, debris in a gutter '
    'usually sticks up above its outer lip and shows against the roof behind it: anything '
    "sitting along the gutter's edge counts, even if most of it is seen against the roof. "
    'If debris reaches into the bright part even a little, the answer is yes. Answer unclear '
    'only when the gutter itself cannot be seen well enough. Ignore the roof surface and the '
    'wall themselves.')


@dataclass
class GutterRequest:
    """What the user asked for, as the planner understood it."""

    understood: str
    house: str
    sides: List[str] = field(default_factory=list)   # compass words; empty = all gutters


def matching_gutters(eaves: Sequence, sides: Sequence[str]) -> Optional[List[str]]:
    """Names of the eaves facing the requested directions; None means all."""
    if not sides:
        return None
    out = []
    for e in eaves:
        for side in sides:
            b = COMPASS_BEARING.get(side.lower().strip())
            if b is None:
                continue
            d = abs((e.side_bearing_deg - b + 180) % 360 - 180)
            if d <= 67.5:
                out.append(e.name)
                break
    return out


Box = Tuple[int, int, int, int]


def crop_box(target: PhotoTarget, pose: Sequence[float], size=(1920, 1080),
             margin: float = 0.08) -> Optional[Tuple[Box, Tuple[int, int]]]:
    """
    Pixel box (left, top, right, bottom) around the segment in a photo.

    From the vehicle pose when the photo was taken, both segment ends at gutter
    height (left/right), and points 0.6 m above and 0.9 m below its middle
    (top/bottom). Also returns the segment's own columns (u0, u1) inside the
    box; the margin outside them shows the neighbours. None if the segment is
    not in front of the camera.
    用拍照时的位姿把水槽段的两端投影到图像上，求出裁剪框。
    """
    mid = ((target.start[0] + target.end[0]) / 2, (target.start[1] + target.end[1]) / 2)
    # Top 0.6 m above the gutter. Not much more: a gutter further away (round
    # a corner) appears higher in the picture and should be cut off (at 0.7 m,
    # in M2, one was judged). Not less: the altitude estimate is up to ~0.45 m
    # off the truth in Gazebo, and a vehicle lower than it believes sees the
    # gutter higher; at 0.4 m a gutter was once 1 px above the crop.
    # 上边缘在水槽上方 0.6 米。不能再大：远处的水槽（转角后面的）在画面中更靠上，应当
    # 裁掉（M2 时取 0.7 米，曾判断了这样的水槽）。不能再小：Gazebo 中高度估计与真值
    # 相差可达约 0.45 米，飞机实际比估计低时水槽在画面中更靠上；取 0.4 米时水槽曾高出
    # 裁剪框 1 个像素。
    # Bottom 0.9 m below: a gutter measured a little too far away (a skewed
    # eave line) appears lower in the picture than predicted. 下边缘放宽到 0.9 米：
    # 位置估计偏远的水槽在画面中比预测的更低。
    pts = np.array([[*target.start, target.height], [*target.end, target.height],
                    [*mid, target.height + 0.6], [*mid, target.height - 0.9]])
    uvd = project(pts, RGB_X500_DEPTH, pose)
    if np.any(uvd[:, 2] <= 0.5) or not np.all(np.isfinite(uvd)):
        return None
    u0, u1 = sorted(uvd[:2, 0])
    pad = (u1 - u0) * margin
    left, right = max(0, int(u0 - pad)), min(size[0], int(math.ceil(u1 + pad)))
    top, bottom = max(0, int(uvd[2, 1])), min(size[1], int(math.ceil(uvd[3, 1])))
    if right - left < 64 or bottom - top < 32:
        return None
    return (left, top, right, bottom), (int(round(u0)) - left, int(round(u1)) - left)


def _samples(target: PhotoTarget, pose, step: float = 0.05):
    """(s, u): points every step along the segment (metres from its start) and their columns."""
    a, b = np.array(target.start, float), np.array(target.end, float)
    length = float(np.linalg.norm(b - a))
    s = np.linspace(0.0, length, max(2, int(length / step) + 1))
    xy = a + np.outer(s / max(length, 1e-9), b - a)
    uvd = project(np.column_stack([xy, np.full(len(s), target.height)]), RGB_X500_DEPTH, pose)
    ok = (uvd[:, 2] > 0.5) & np.isfinite(uvd[:, 0])
    return s[ok], uvd[ok, 0]


def stretch_in_columns(target: PhotoTarget, pose, c0: float,
                       c1: float) -> Optional[Tuple[float, float]]:
    """Part of the segment that appears between columns c0 and c1: (from, to) in metres."""
    s, u = _samples(target, pose)
    inside = (u >= c0 - 0.5) & (u <= c1 + 0.5)
    if not inside.any():
        return None
    return float(s[inside].min()), float(s[inside].max())


def along_at_column(target: PhotoTarget, pose, c: float) -> Optional[float]:
    """Where along the segment (metres from its start) column c falls; None if off it."""
    s, u = _samples(target, pose)
    if len(s) < 2 or not (min(u) <= c <= max(u)):
        return None
    order = np.argsort(u)          # a straight line projects to a monotonic run of columns
    return float(np.interp(c, u[order], s[order]))


HOME_CLEARANCE = 0.75   # m: a boundary this close to the home's own point is in doubt


def _along_line(target: PhotoTarget, p) -> float:
    """Where p lies along the segment's line, in metres from its start (may be outside it)."""
    a, b = np.array(target.start, float), np.array(target.end, float)
    length = float(np.linalg.norm(b - a))
    return float((np.array(p, float) - a) @ (b - a) / length) if length > 1e-9 else 0.0


def point_along(target: PhotoTarget, s: float) -> Tuple[float, float]:
    a, b = np.array(target.start, float), np.array(target.end, float)
    length = float(np.linalg.norm(b - a))
    p = a + (b - a) * (s / length if length > 1e-9 else 0.0)
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def column_of(p, height: float, pose) -> Optional[float]:
    """Image column of a point at the given height, None if behind the camera."""
    u, _, d = project(np.array([[p[0], p[1], height]]), RGB_X500_DEPTH, pose)[0]
    return float(u) if d > 0.5 and np.isfinite(u) else None


def _wall_rows(target: PhotoTarget, pose, size) -> Tuple[int, int]:
    """Rows from just above the gutter to 2.5 m below it: the wall shows where homes meet."""
    mid = ((target.start[0] + target.end[0]) / 2, (target.start[1] + target.end[1]) / 2)
    uvd = project(np.array([[*mid, target.height + 0.4], [*mid, target.height - 2.5]]),
                  RGB_X500_DEPTH, pose)
    top = int(max(0, min(size[1] - 1, uvd[0, 1])))
    bottom = int(max(top + 1, min(size[1], math.ceil(uvd[1, 1]))))
    return top, bottom


def _save_crop(path: str, box, judged, suffix: str = '_segment.jpg') -> Optional[str]:
    """
    Save the box of the photo, darkening outside the judged columns (full-image columns).

    The darkened strips show the neighbouring stretches for context; the judge
    is told to ignore them. 调暗判断范围以外的部分：保留上下文，但不参与判断。
    """
    from PIL import Image, ImageDraw
    try:
        with Image.open(path) as img:
            crop = img.convert('RGB').crop(tuple(int(v) for v in box))
    except OSError:
        return None
    if judged is not None:
        w, h = crop.size
        u0, u1 = judged[0] - box[0], judged[1] - box[0]
        mask = Image.new('L', crop.size, 0)
        d = ImageDraw.Draw(mask)
        if u0 > 1:
            d.rectangle([0, 0, int(u0) - 1, h], fill=150)
        if u1 < w - 1:
            d.rectangle([int(math.ceil(u1)) + 1, 0, w, h], fill=150)
        crop = Image.composite(Image.new('RGB', crop.size, (0, 0, 0)), crop, mask)
    out = path[:-4] + suffix
    crop.save(out, quality=92)
    return out


def apply_boundaries(verdicts: List[Verdict]) -> List[Verdict]:
    """
    Let a boundary between homes seen in one photo bound the others on that gutter.

    A stretch lying wholly past a boundary (seen from the home being inspected)
    is the neighbour's and is set aside; one partly past it may have been judged
    on the neighbour's gutter, so it becomes unclear and keeps only its own part.
    一张照片里看到的户界同样约束同一条水槽的其他照片：完全在分界之外的段属于邻居，
    剔除；部分越过分界的段可能判断的是邻居家的水槽，改为“看不清”，只保留自家部分。
    """
    seen: Dict[str, List[Tuple[Tuple[float, float], str]]] = {}
    for v in verdicts:
        if v.extra.get('boundary'):
            seen.setdefault(v.extra['gutter'], []).append((v.extra['boundary'], v.label))
    for v in verdicts:
        x = v.extra
        if not x.get('home') or v.needs_attention in ('skipped', 'neighbour'):
            continue
        a, b = np.array(x['from'], float), np.array(x['to'], float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        d = (b - a) / length
        t_home = float((np.array(x['home'], float) - a) @ d)
        lo, hi, why = 0.0, length, ''
        for p, other in seen.get(x['gutter'], []):
            if other == v.label:
                continue                      # its own boundary is already applied
            tb = float((np.array(p, float) - a) @ d)
            if tb >= t_home and tb < hi:
                hi, why = tb, other
            elif tb < t_home and tb > lo:
                lo, why = tb, other
        if hi - lo < 0.3:
            v.needs_attention, v.confidence = 'neighbour', 'medium'
            v.findings = f'Past the boundary between homes seen in "{why}": the neighbour\'s.'
        elif lo > 0.05 or hi < length - 0.05:
            v.needs_attention, v.confidence = 'unclear', 'low'
            v.findings += (f' Part of this stretch lies past the boundary between homes seen in '
                           f'"{why}", so the judgement may be about the neighbour\'s gutter.')
            p0, p1 = a + d * lo, a + d * hi
            x['from'] = [round(float(p0[0]), 3), round(float(p0[1]), 3)]
            x['to'] = [round(float(p1[0]), 3), round(float(p1[1]), 3)]
    return verdicts


class GutterTask:
    """Holds the state of one gutter inspection and answers MissionNode's callbacks."""

    def __init__(self, world: WorldMap, request: GutterRequest, start: Tuple[float, float],
                 out_dir: str, judge, instruction: str,
                 cfg: InspectionConfig = InspectionConfig(), log=print) -> None:
        houses = {h.name: h for h in world.rough_houses}
        if request.house not in houses:
            raise ValueError(f'no house "{request.house}" on the map '
                             f'(have: {", ".join(houses) or "none"})')
        self.world, self.request, self.start = world, request, start
        self.house: RoughHouse = houses[request.house]
        self.out_dir, self.judge, self.instruction = out_dir, judge, instruction
        self.cfg, self.log = cfg, log
        self.depth = None                       # DepthCamera, set by attach()
        self.estimate: Optional[HouseEstimate] = None
        self.targets: Dict[str, PhotoTarget] = {}
        self.decide_note = ''
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='judge')
        self._lock = threading.Lock()
        self._futures: List[Tuple[str, str, Future]] = []

    # -- before takeoff ---------------------------------------------------------

    def plan(self) -> List[Skill]:
        return survey_skills(self.house, self.world.limits, self.start, self.cfg) + [ReturnHome()]

    def attach(self, node) -> None:
        """Wire the callbacks into a MissionNode (needs a ROS node for the depth topics)."""
        from wukong_perception.depth_camera import DepthCamera
        self.depth = DepthCamera(node)
        node.on_scan = self.depth.capture
        node.on_decide = self.on_decide
        node.on_photo = self.on_photo

    # -- in flight ----------------------------------------------------------------

    def on_decide(self, label: str, pose) -> Tuple[Optional[list], Optional[Limits], str]:
        pts = merge(self.depth.clouds) if self.depth else None
        if pts is None:
            return None, None, 'no depth scans'
        np.save(os.path.join(self.out_dir, 'survey_points.npy'), pts.astype(np.float32))
        try:
            est = fit_house(pts, self.house.approx_center, self.house.radius)
        except FitError as e:
            self.decide_note = f'house not found: {e}'
            return None, None, self.decide_note
        self.estimate = est
        with open(os.path.join(self.out_dir, 'house_estimate.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(est), f, indent=2)
        g = est.ground_level
        self.log(f'[gutters] house: {est.length:.1f} x {est.width:.1f} m, ridge '
                 f'{est.ridge_bearing_deg:.0f} deg, {est.ridge_height - g:.1f} m high; eaves: '
                 + ', '.join(f'{e.name} ({e.length:.1f} m at {e.height - g:.1f} m)'
                             for e in est.eaves)
                 + (f'; notes: {"; ".join(est.notes)}' if est.notes else ''))
        names = matching_gutters(est.eaves, self.request.sides)
        if names is not None and not names:
            self.decide_note = (f'no gutter faces {", ".join(self.request.sides)}; found '
                                + ', '.join(e.name for e in est.eaves))
            return None, None, self.decide_note
        skills, limits, targets = inspection_skills(
            est, (pose[0], pose[1]), self.start, self.world.limits, self.cfg, names)
        self.targets = targets
        self.decide_note = f'{len(targets)} gutter segment(s) to photograph'
        return skills, limits, self.decide_note

    def on_photo(self, label: str, path: str, pose) -> None:
        import time
        target = self.targets.get(label)
        fut = self._pool.submit(self._inspect_one, label, path, target, tuple(pose), time.time())
        with self._lock:
            self._futures.append((label, path, fut))

    def _inspect_one(self, label: str, path: str, target: Optional[PhotoTarget], pose,
                     t: float) -> Verdict:
        """
        Crop the photo to its segment and judge it; the report says what was judged.

        The stretch reported ('from'/'to') is the part of the segment inside the
        judged columns, by the recorded pose: not the plan, which may run off the
        picture or past a party wall. A segment not in the picture at all is
        reported as not inspected rather than judged from the whole photo.
        报告的范围是按记录的位姿、实际落在判断区域内的那一段，不是计划的范围。
        段完全不在画面里时报“未检查”，不再用整张照片代替。
        """
        if target is None:
            v = self.judge.judge(self.instruction, label, 'Does this need attention?', path)
            v.path = path
            return v
        extra = {'gutter': target.gutter, 'segment': target.segment,
                 'segments': target.segments, 'from': list(target.start),
                 'to': list(target.end), 'planned_from': list(target.start),
                 'planned_to': list(target.end), 'height': target.height,
                 'full_photo': path, 't': t, 'pose': list(pose)}
        if target.home is not None:
            extra['home'] = list(target.home)
        try:
            from PIL import Image
            with Image.open(path) as img:
                size = img.size
        except OSError as e:
            return Verdict(label, path, 'skipped', 'low', f'not inspected: no photo ({e})',
                           extra=extra)
        found = crop_box(target, pose, size)
        if found is None:
            return Verdict(label, path, 'skipped', 'low', 'not inspected: the segment is not '
                           'in the picture (by the recorded pose)', extra=extra)
        box, (u0, u1) = found
        bright = (max(0.0, box[0] + u0), min(float(size[0]), box[0] + u1))
        judged, note = bright, ''
        if target.home is not None:
            judged, note, boundary, status = self._home_side(path, target, pose, bright, size)
            if boundary is not None:
                extra['boundary'] = list(boundary)
            if status == 'neighbour':
                return Verdict(label, path, 'neighbour', 'medium', note, extra=extra)
            if status == 'doubt':
                return Verdict(label, path, 'unclear', 'low', note, extra=extra)
        stretch = stretch_in_columns(target, pose, *judged)
        if stretch is None:
            return Verdict(label, path, 'skipped', 'low', 'not inspected: the segment is not '
                           'in the judged part of the picture', extra=extra)
        extra['from'], extra['to'] = (list(point_along(target, s)) for s in stretch)
        extra['columns'] = [round(judged[0], 1), round(judged[1], 1)]
        extra['rows'] = [box[1], box[3]]
        to_judge = _save_crop(path, box, judged)
        if to_judge is None:
            return Verdict(label, path, 'skipped', 'low', 'not inspected: could not crop the '
                           'photo', extra=extra)
        v = self.judge.judge(self.instruction, label, SEGMENT_QUESTION, to_judge)
        v.path = to_judge
        v.findings = (v.findings + ' ' + note).strip()
        v.extra = extra
        return v

    def _home_side(self, path: str, target: PhotoTarget, pose, bright, size):
        """
        Return (columns, note, boundary, status): the inspected home's part of the strip.

        Asks where one home ends in a crop showing the wall below the gutter.
        A boundary seen: keep the home's side of it, wherever the plan put the
        party wall. None seen: keep only the claimed part (the plan's reading of
        the boundary). status: 'ok'; 'neighbour' (the whole strip is past the
        boundary); 'doubt' (a boundary right at the point known to be the home's).
        找出照片里两户的分界：看到分界就只保留目标户一侧（不论计划把共用墙放在哪里）；
        没看到就只保留计划认定属于目标户的部分。
        """
        c0, c1 = bright
        width = max(c1 - c0, 1.0)
        u_home = column_of(target.home, target.height, pose)
        wall = _wall_rows(target, pose, size)
        bpath = _save_crop(path, (int(c0), wall[0], int(math.ceil(c1)), wall[1]), None,
                           '_boundary.jpg')
        at, why = (None, '') if bpath is None or u_home is None else \
            self.judge.find_boundary(bpath, (u_home - c0) / width)
        if at is not None:
            # Toward the row's end there is no neighbour: whatever was seen there
            # (in Gazebo, the end corner) is not a party wall. 朝整排尽头没有邻居：
            # 在那边看到的（Gazebo 中是尽头的转角）不是共用墙。
            s_b = along_at_column(target, pose, c0 + at * width)
            s_home = _along_line(target, target.home)
            if s_b is not None and not target.walls[0 if s_b < s_home else 1]:
                at = None
        if at is None:
            if target.claim is None:
                return bright, '', None, 'ok'
            cols = [column_of(p, target.height, pose) for p in target.claim]
            if any(c is None for c in cols):
                return bright, '', None, 'ok'
            lo, hi = max(c0, min(cols)), min(c1, max(cols))
            if hi - lo < 1.0:
                return None, ('Past the boundary between homes read from the overview photo: '
                              "the neighbour's, not inspected."), None, 'neighbour'
            return (lo, hi), '', None, 'ok'
        bc = c0 + at * width
        s_b = along_at_column(target, pose, bc)
        boundary = point_along(target, s_b) if s_b is not None else None
        where = f'A boundary between homes shows {at:.0%} across this stretch ({why})'
        s_home = _along_line(target, target.home)
        if s_b is None or abs(s_b - s_home) < HOME_CLEARANCE:
            # At the point known to be inside the home: the reading is wrong, or
            # the answer is. (A boundary just anywhere in a picture that also shows
            # that point is fine: in Gazebo one was 1.1 m from it, correctly.)
            # 分界就在已知属于目标户的那一点上：要么户界读错了，要么回答错了。
            return None, (where + ', where the home being inspected should be: where this '
                          'home ends is in doubt, so this stretch was not judged.'), boundary, \
                'doubt'
        side = (c0, bc) if u_home < bc else (bc, c1)
        if side[1] - side[0] < 0.05 * width:
            return None, where + ": the rest is the neighbour's, not inspected.", boundary, \
                'neighbour'
        return side, where + "; only this home's side of it was judged.", boundary, 'ok'

    # -- after landing --------------------------------------------------------------

    def pending(self) -> int:
        with self._lock:
            return sum(not f.done() for _, _, f in self._futures)

    def collect(self, timeout: float) -> List[Verdict]:
        import time
        end = time.monotonic() + timeout
        out = []
        with self._lock:
            futures = list(self._futures)
        for label, path, fut in futures:
            try:
                out.append(fut.result(timeout=max(0.0, end - time.monotonic())))
            except Exception as e:   # noqa: B902 - report, go on
                note = 'timed out' if not fut.done() else f'failed: {e}'
                out.append(Verdict(label, path, 'pending', 'low', f'judgement {note}'))
        self._pool.shutdown(wait=False, cancel_futures=True)
        return apply_boundaries(out)
