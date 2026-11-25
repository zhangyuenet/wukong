# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Fly only through space seen to be empty, facing the way the drone goes.

docs/OBSTACLE_AVOIDANCE.md, principles 3 and 4, L2. Routes are still planned
on the optimistic map (pathplan.Planner: known buildings and the unseen part
of the target are no-fly, other space is not), then cut to the part whose
swept volume lies in voxels seen free (voxels.VoxelMap). The drone flies that
part facing the way it goes, looks ahead, and decides again: what it then
sees decides whether the rest can follow.
航线仍在乐观的地图上规划（已知建筑和目标没看到的部分禁飞，其余不禁），再截取扫过的
空间全部在确认为空的体素里的那一段。飞机朝着前进方向飞完这一段，向前看，再重新决定：
看到的结果决定后面能不能继续。
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

from wukong_flight.skills import GoTo
from wukong_perception.voxels import VoxelMap

Point3 = Tuple[float, float, float]      # north, east, altitude


@dataclass(frozen=True)
class ExploreConfig:
    # What a move sweeps must be seen free: an upright cylinder, the x500's
    # ~0.5 m plus 1 m for position error, sway and overshoot across, and its
    # ~0.15 m plus 0.6 m (the barometer drifts ~0.45 m) above and below. Not a
    # 1.5 m sphere: a level camera sees at most 29 deg up, and on a 20 deg climb
    # the top of such a sphere comes into view only ~9.6 m ahead (4.2 m for
    # this cylinder). 移动时扫过、必须确认为空的竖直圆柱：水平方向机体约 0.5 米加 1 米
    # （定位误差、晃动、超调），竖直方向约 0.15 米加 0.6 米（气压计漂移约 0.45 米）。不用
    # 1.5 米的球：水平相机向上最多看 29 度，20 度爬升时球顶要到前方约 9.6 米才进入视野
    # （这个圆柱是 4.2 米）。
    radius: float = 1.5
    half_height: float = 0.75
    stop_short: float = 1.0     # m, stop this far before what is not seen free
    min_step: float = 1.0       # m, a shorter partial move is not worth making
    body: float = 0.5           # m, where the drone has been is free this far round
    turn_deg: float = 10.0      # turn on the spot first when the heading changes more


def reach(voxels: VoxelMap, start: Sequence[float], route: Sequence[Point3],
          cfg: ExploreConfig = ExploreConfig()) -> Tuple[List[Point3], bool]:
    """
    Return the part of route that may be flown now, and whether that is all of it.

    route: the waypoints after start, as pathplan.Planner.path gives them. The
    part ends at the last waypoint whose leg is seen free all the way, or
    stop_short before the first voxel that is not, if that is at least
    min_step on. 返回现在可以飞的那一段，以及是否就是整条航线。
    """
    out: List[Point3] = []
    cur = tuple(float(v) for v in start)
    for p in route:
        d = voxels.first_blocked(cur, p, cfg.radius, cfg.half_height)
        if d is None:
            out.append(tuple(p))
            cur = tuple(p)
            continue
        keep = d - cfg.stop_short
        length = math.dist(cur, p)
        if keep >= cfg.min_step and length > 1e-6:
            k = keep / length
            out.append(tuple(round(cur[i] + k * (p[i] - cur[i]), 2) for i in range(3)))
        return out, False
    return out, True


def legs_facing_ahead(start: Sequence[float], heading: float, waypoints: Sequence[Point3],
                      face: Optional[Tuple[float, float]] = None,
                      cfg: ExploreConfig = ExploreConfig()) -> List[GoTo]:
    """
    Return GoTo skills that fly each leg facing the way it goes.

    The depth camera looks straight ahead, so the drone sees each leg as it
    flies it (L3). GoTo turns towards look_at as its leg starts, so a turn of
    more than turn_deg is made on the spot first. A leg with no horizontal
    part (a climb above the takeoff point) keeps the heading. face: the point
    to look at after the last waypoint, or None. heading: the drone's heading
    at start, radians from north. 每段都朝着前进方向飞：深度相机朝前，边飞边看到
    这一段。GoTo 在开始时转向 look_at，所以转角超过 turn_deg 时先原地转向。没有水平
    分量的一段（起飞点上方的升降）保持航向。
    """
    out: List[GoTo] = []
    cur = tuple(float(v) for v in start[:3])
    for p in waypoints:
        dn, de = p[0] - cur[0], p[1] - cur[1]
        if math.hypot(dn, de) < 0.1:
            out.append(GoTo(*p))
            cur = tuple(p)
            continue
        ahead = math.atan2(de, dn)
        look = (round(p[0] + 10.0 * math.cos(ahead), 2), round(p[1] + 10.0 * math.sin(ahead), 2))
        turn = abs((ahead - heading + math.pi) % (2 * math.pi) - math.pi)
        if turn > math.radians(cfg.turn_deg):
            here = tuple(round(v, 2) for v in cur)
            out.append(GoTo(*here, look_at=(round(here[0] + 10.0 * math.cos(ahead), 2),
                                            round(here[1] + 10.0 * math.sin(ahead), 2))))
        out.append(GoTo(*p, look_at=look))
        heading, cur = ahead, tuple(p)
    if face is not None:
        out.append(GoTo(*cur, look_at=face))
    return out


def back_along(trail: Sequence[Point3], here: Sequence[float]) -> List[Point3]:
    """
    Return the way back from here along the trail flown, ending above the takeoff point.

    Class A goes home the way it came (docs/OBSTACLE_AVOIDANCE.md, L5): every
    leg was flown, so it is known to be clear, and flown in reverse it is
    seen ahead again. trail: the points flown through in order, the first
    above the takeoff point. Points the drone is already at, and points on
    a straight line between their neighbours, are left out.
    A 级原路返回：每一段都飞过，已知可飞；倒着飞时也朝前看得见。trail 是按顺序飞过的
    点，第一个在起飞点上方。省略飞机已经所在的点和夹在一条直线中间的点。
    """
    path = [tuple(float(v) for v in here[:3])]      # here, then the waypoints
    for p in (tuple(q) for q in reversed(trail)):
        if math.dist(path[-1], p) < 0.1:
            continue
        if len(path) > 1 and _between(path[-2], path[-1], p):
            path[-1] = p                               # straight on: no stop needed
        else:
            path.append(p)
    return path[1:]


def _between(a: Point3, b: Point3, c: Point3, tol: float = 0.05) -> bool:
    """Return whether b lies on the straight segment a -> c (within tol metres)."""
    ac = [c[i] - a[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in ac))
    if length < 1e-9:
        return False
    t = sum((b[i] - a[i]) * ac[i] for i in range(3)) / (length * length)
    if not 0.0 <= t <= 1.0:
        return False
    return math.dist(b, tuple(a[i] + t * ac[i] for i in range(3))) <= tol
