# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Paths around buildings of any shape, and viewpoints that can see a gutter.

Everything is checked against the inflated no-fly zones of a Limits (the same
ones check_plan() uses), so a path found here also passes the check.
Plain Python: grids here are at most a few hundred cells a side.
任意形状建筑周围的路径规划和视点选择；使用与 check_plan() 相同的禁飞区，
所以这里找到的路径也能通过检查。
"""

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

from wukong_flight.skills import HeightField, Limits

Point3 = Tuple[float, float, float]      # north, east, altitude
CAMERA_ABOVE_BODY = 0.26                 # m, x500_depth camera mount


class Planner:
    """Path finding and visibility against one set of limits."""

    def __init__(self, limits: Limits, grid: float = 0.5, step: float = 0.25) -> None:
        self.limits = limits
        self.zones = [z.inflated(limits.margin) for z in limits.no_fly]
        self.grid = grid
        self.step = step

    # -- collision --------------------------------------------------------------------

    def free(self, p: Point3) -> bool:
        lim = self.limits
        if not lim.min_altitude <= p[2] <= lim.max_altitude:
            return False
        return not any(z.contains(*p) for z in self.zones)

    def segment_free(self, a: Point3, b: Point3) -> bool:
        n = max(1, int(math.ceil(math.dist(a, b) / self.step)))
        return all(self.free(tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3)))
                   for i in range(n + 1))

    # -- paths ----------------------------------------------------------------------

    def path(self, start: Point3, goal: Point3, search_margin: float = 20.0
             ) -> Optional[List[Point3]]:
        """
        Waypoints from start to goal (start excluded, goal included), or None.

        Straight if clear. Otherwise change altitude where a vertical move is
        clear (at the start or at the goal) and go round obstacles at one
        altitude, on a grid (A*), shortened by line-of-sight.
        直线可行就走直线；否则在起点或终点处垂直升降，在一个高度上用 A* 绕行，再拉直。
        """
        if not self.free(goal):
            return None
        if self.segment_free(start, goal):
            return [goal]
        # Goal altitude first, then the start's, then over the top of everything
        # in the way (e.g. across a townhouse row). 先试目标高度、起点高度，再试从上方飞越。
        alts = [goal[2], start[2]]
        over = self._overfly_altitude(start, goal)
        if over is not None:
            alts.append(over)
        for alt in dict.fromkeys(alts):
            a = (start[0], start[1], alt)
            b = (goal[0], goal[1], alt)
            if not (self.segment_free(start, a) and self.segment_free(b, goal)):
                continue
            mid = self._astar(a, b, search_margin)
            if mid is None:
                continue
            route = self._shorten(start, [a] + mid + [b, goal])
            # Verify every leg exactly as check_plan() will. 按 check_plan 的方式逐段复核。
            legs = [start] + route
            if all(self.segment_free(p, q) for p, q in zip(legs, legs[1:])):
                return route
        return None

    def _overfly_altitude(self, a: Point3, b: Point3) -> Optional[float]:
        """Lowest altitude clear of every zone along the straight line a-b, or None."""
        n = max(1, int(math.ceil(math.dist(a[:2], b[:2]) / self.step)))
        top = -math.inf
        for i in range(n + 1):
            p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            for z in self.zones:
                h = getattr(z, 'height_at', None)
                if h is not None:
                    t = h(*p)
                    if t is not None:
                        top = max(top, t)
                elif z.contains(p[0], p[1], -math.inf):
                    top = max(top, z.top)
        if top == -math.inf:
            return None
        alt = round(top + 0.5, 2)
        return alt if alt <= self.limits.max_altitude else None

    def _astar(self, a: Point3, b: Point3, margin: float) -> Optional[List[Point3]]:
        g, alt = self.grid, a[2]
        n0 = min(a[0], b[0]) - margin
        e0 = min(a[1], b[1]) - margin
        rows = int((max(a[0], b[0]) + margin - n0) / g) + 1
        cols = int((max(a[1], b[1]) + margin - e0) / g) + 1

        def center(c):
            return (n0 + (c[0] + 0.5) * g, e0 + (c[1] + 0.5) * g, alt)

        def cell(p):
            return (int((p[0] - n0) / g), int((p[1] - e0) / g))
        blocked: Dict[Tuple[int, int], bool] = {}

        def ok(c):
            if not (0 <= c[0] < rows and 0 <= c[1] < cols):
                return False
            if c not in blocked:
                blocked[c] = not self.free(center(c))
            return not blocked[c]
        s, t = cell(a), cell(b)
        blocked[s] = blocked[t] = False
        frontier = [(0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while frontier:
            _, c = heapq.heappop(frontier)
            if c == t:
                out = []
                while c is not None and c != s:
                    out.append(center(c))
                    c = came[c]
                return out[::-1]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    nb = (c[0] + di, c[1] + dj)
                    if not ok(nb):
                        continue
                    # No corner cutting: a diagonal step needs both side cells free.
                    # 斜向移动时两侧的格子也必须可飞，避免擦过障碍物的角。
                    if di and dj and not (ok((c[0] + di, c[1])) and ok((c[0], c[1] + dj))):
                        continue
                    nc = cost[c] + math.hypot(di, dj)
                    if nc < cost.get(nb, math.inf):
                        cost[nb] = nc
                        came[nb] = c
                        heapq.heappush(frontier, (nc + math.hypot(nb[0] - t[0], nb[1] - t[1]),
                                                  nb))
        return None

    def _shorten(self, start: Point3, pts: List[Point3]) -> List[Point3]:
        """Greedy line-of-sight: from each point, jump to the farthest visible one."""
        out, cur, k = [], start, 0
        while k < len(pts):
            j = len(pts) - 1
            while j > k and not self.segment_free(cur, pts[j]):
                j -= 1
            cur = pts[j]
            out.append((round(cur[0], 2), round(cur[1], 2), round(cur[2], 2)))
            k = j + 1
        return out


def route_home(planner: Planner, pos: Point3, home: Tuple[float, float],
               climbs: Sequence[float] = (0.0, 2.0, 4.0)) -> Optional[List[Point3]]:
    """
    Plan a checked route from pos to above home, at the current altitude or higher.

    The ReturnHome that follows then only descends. 规划从当前位置回到起飞点上方的
    安全路线，之后的 ReturnHome 只需要下降。
    """
    for up in climbs:
        route = planner.path(pos, (home[0], home[1], round(pos[2] + up, 2)))
        if route is not None:
            return route
    return None


# -- seeing a gutter ------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewConfig:
    standoffs: Tuple[float, ...] = (4.5, 5.5, 6.5, 8.0)   # m out from the eave
    aboves: Tuple[float, ...] = (0.9, 1.2, 1.6, 2.0)      # m above the gutter
    max_down_deg: float = 16.0      # camera half vertical FOV is ~21 deg; keep a margin
    shifts: Tuple[float, ...] = (0.0, 1.0, -1.0)          # m along the eave


def line_of_sight(cam: Point3, target: Point3, surfaces: Sequence[HeightField],
                  stop_short: float = 0.6, step: float = 0.2) -> bool:
    """Return True if nothing seen in the survey blocks the view from cam to target."""
    d = math.dist(cam, target)
    n = int((d - stop_short) / step)
    for i in range(1, n + 1):
        t = i * step / d
        p = tuple(cam[k] + (target[k] - cam[k]) * t for k in range(3))
        for hf in surfaces:
            h = hf.height_at(p[0], p[1])
            if h is not None and p[2] <= h:
                return False
    return True


def choose_viewpoint(planner: Planner, mid: Tuple[float, float], height: float,
                     outward: Tuple[float, float], surfaces: Sequence[HeightField],
                     cfg: ViewConfig = ViewConfig()) -> Optional[Point3]:
    """
    Find the first workable viewpoint for a gutter segment, nearest and lowest first.

    Workable: free of every no-fly zone, the gutter less than max_down_deg below
    the camera's level view, and nothing in the way. 视点可用的条件：不在禁飞区内，
    水槽在相机水平视线下方不超过 max_down_deg，中间没有遮挡。
    """
    along = (-outward[1], outward[0])
    target = (mid[0], mid[1], height + 0.05)
    for s in cfg.standoffs:
        for a in cfg.aboves:
            if math.degrees(math.atan2(a + CAMERA_ABOVE_BODY, s)) > cfg.max_down_deg:
                continue
            for sh in cfg.shifts:
                vp = (round(mid[0] + outward[0] * s + along[0] * sh, 2),
                      round(mid[1] + outward[1] * s + along[1] * sh, 2),
                      round(height + a, 2))
                if not planner.free(vp):
                    continue
                cam = (vp[0], vp[1], vp[2] + CAMERA_ABOVE_BODY)
                if line_of_sight(cam, target, surfaces):
                    return vp
    return None
