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
    height (left/right), and points 0.4 m above and 0.9 m below its middle
    (top/bottom). Also returns the segment's own columns (u0, u1) inside the
    box; the margin outside them shows the neighbours. None if the segment is
    not in front of the camera.
    用拍照时的位姿把水槽段的两端投影到图像上，求出裁剪框。
    """
    mid = ((target.start[0] + target.end[0]) / 2, (target.start[1] + target.end[1]) / 2)
    # Top only 0.4 m above the gutter: a gutter further away (round a corner)
    # appears higher in the picture and is mostly cut off. 上边缘只到水槽上方 0.4 米：
    # 远处的水槽（转角后面的）在画面中更靠上，大部分会被裁掉。
    # Bottom 0.9 m below: a gutter measured a little too far away (a skewed
    # eave line) appears lower in the picture than predicted. 下边缘放宽到 0.9 米：
    # 位置估计偏远的水槽在画面中比预测的更低。
    pts = np.array([[*target.start, target.height], [*target.end, target.height],
                    [*mid, target.height + 0.4], [*mid, target.height - 0.9]])
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
        target = self.targets.get(label)
        to_judge = path
        if target is not None:
            to_judge = self._crop(path, target, pose) or path
        fut = self._pool.submit(self._judge_one, label, path, to_judge, target)
        with self._lock:
            self._futures.append((label, path, fut))

    def _crop(self, path: str, target: PhotoTarget, pose) -> Optional[str]:
        from PIL import Image
        try:
            with Image.open(path) as img:
                found = crop_box(target, pose, img.size)
                if found is None:
                    return None
                box, (u0, u1) = found
                crop = img.convert('RGB').crop(box)
                # Darken the neighbours' strips. 把相邻段的部分调暗。
                w, h = crop.size
                dark = Image.new('RGB', crop.size, (0, 0, 0))
                mask = Image.new('L', crop.size, 0)
                from PIL import ImageDraw
                d = ImageDraw.Draw(mask)
                if u0 > 0:
                    d.rectangle([0, 0, max(0, u0 - 1), h], fill=150)
                if u1 < w:
                    d.rectangle([min(w - 1, u1 + 1), 0, w, h], fill=150)
                crop = Image.composite(dark, crop, mask)
                out = path[:-4] + '_segment.jpg'
                crop.save(out, quality=92)
                return out
        except OSError:
            return None

    def _judge_one(self, label, path, to_judge, target) -> Verdict:
        q = SEGMENT_QUESTION if target is not None else 'Does this need attention?'
        v = self.judge.judge(self.instruction, label, q, to_judge)
        v.path = to_judge
        if target is not None:
            v.extra = {'gutter': target.gutter, 'segment': target.segment,
                       'segments': target.segments, 'from': list(target.start),
                       'to': list(target.end), 'height': target.height,
                       'full_photo': path}
        return v

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
        return out
