# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
The known map: houses, their gutters, flight limits and the camera.

One file feeds two consumers: the planner (as text for the large model) and the
pre-flight check (as no-fly boxes). The model is told about the boxes, but the
check does not rely on it having listened.
同一份地图有两个用途：给大模型的文字描述，和飞行前检查用的禁飞区。
"""

from dataclasses import dataclass
from typing import List, Tuple

from wukong_flight.skills import Box, Cylinder, Limits
import yaml


@dataclass(frozen=True)
class Gutter:
    name: str
    start: Tuple[float, float]     # (north, east)
    end: Tuple[float, float]
    height: float

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)


@dataclass(frozen=True)
class House:
    name: str
    center: Tuple[float, float]
    depth: float                   # north-south
    width: float                   # east-west
    overhang: float
    eave_height: float
    ridge_height: float
    ridge_direction: str
    gutters: Tuple[Gutter, ...]

    def box(self) -> Box:
        """Footprint plus roof overhang, up to the ridge."""
        n, e = self.center
        hd, hw = self.depth / 2 + self.overhang, self.width / 2 + self.overhang
        return Box(n - hd, n + hd, e - hw, e + hw, self.ridge_height, self.name)


@dataclass(frozen=True)
class RoughHouse:
    """
    A house known only roughly: somewhere within radius of approx_center.

    Its exact shape comes from perception in flight (M2). Until then the whole
    circle, up to max_height, is a no-fly zone.
    只知道大概位置的房屋：精确形状在飞行中由感知得到，之前整个圆柱都是禁飞区。
    """

    name: str
    approx_center: Tuple[float, float]
    radius: float
    max_height: float

    def zone(self) -> Cylinder:
        return Cylinder(self.approx_center, self.radius, self.max_height, self.name)


@dataclass(frozen=True)
class WorldMap:
    name: str
    houses: Tuple[House, ...]
    limits: Limits
    camera_hfov_deg: float
    camera_vfov_deg: float
    rough_houses: Tuple[RoughHouse, ...] = ()
    # M3: no house given; the drone inspects the nearest building it finds
    # within search_radius of the takeoff point. M3：地图不给房屋，无人机自己找最近的建筑。
    search_radius: float = 0.0
    max_building_height: float = 12.0
    # Confirmed once by the homeowner for a real site (sim: from the scenario):
    # the target is one unit of a townhouse row. 用户确认过的场地信息：目标是联排中的一户。
    townhouse: bool = False

    @property
    def find_house(self) -> bool:
        return not self.houses and not self.rough_houses and self.search_radius > 0

    def describe(self) -> str:
        """Plain-text map for the planner prompt."""
        lines = [f'Map: {self.name}. Coordinates are (north, east) in metres in the '
                 'local frame; heights are metres above the ground.']
        for h in self.houses:
            b = h.box()
            lines.append(
                f'- {h.name}: footprint centred at {h.center}, {h.depth:g} m north-south by '
                f'{h.width:g} m east-west, roof overhang {h.overhang:g} m, eaves at '
                f'{h.eave_height:g} m, ridge at {h.ridge_height:g} m running '
                f'{h.ridge_direction}. Including the overhang it occupies north '
                f'{b.north_min:g}..{b.north_max:g}, east {b.east_min:g}..{b.east_max:g}.')
            for g in h.gutters:
                lines.append(f'  - {g.name}: from {g.start} to {g.end} at {g.height:g} m, '
                             f'midpoint {g.midpoint}.')
        for r in self.rough_houses:
            lines.append(f'- {r.name}: exact shape unknown; it lies within {r.radius:g} m of '
                         f'{r.approx_center} and is at most {r.max_height:g} m tall. Its shape '
                         f'and gutters are measured in flight by the survey_house skill.')
        if self.find_house:
            lines.append(f'No house positions are known. In flight the drone looks around from '
                         f'above the takeoff point and inspects the building nearest to it '
                         f'(within {self.search_radius:g} m): "the house next to you", "this '
                         f'house" mean that building. It can find gutters on every level of '
                         f'it, and tell which compass direction each one faces.')
        lim = self.limits
        lines.append(
            f'Flight limits: altitude {lim.min_altitude:g}..{lim.max_altitude:g} m above the '
            f'takeoff point, at most {lim.max_distance:g} m horizontally from the takeoff '
            f'point, and at least {lim.margin:g} m from every house (including its overhang, '
            f'and above its ridge). Paths between points are straight lines and must keep '
            f'that distance too.')
        lines.append(
            f'Camera: fixed, facing forward and level, {self.camera_hfov_deg:g} deg '
            f'horizontal by {self.camera_vfov_deg:g} deg vertical field of view. Point it '
            f'with look_at.')
        return '\n'.join(lines)


def _pair(v) -> Tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f'expected [north, east], got {v!r}')
    return float(v[0]), float(v[1])


def load_world(path: str) -> WorldMap:
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    houses: List[House] = []
    rough: List[RoughHouse] = []
    for h in data.get('houses', []):
        if 'approx_center' in h:
            rough.append(RoughHouse(h['name'], _pair(h['approx_center']), float(h['radius']),
                                    float(h['max_height'])))
            continue
        gutters = tuple(Gutter(g['name'], _pair(g['from']), _pair(g['to']), float(g['height']))
                        for g in h.get('gutters', []))
        houses.append(House(
            name=h['name'], center=_pair(h['center']),
            depth=float(h['depth_north_south']), width=float(h['width_east_west']),
            overhang=float(h.get('roof_overhang', 0.0)),
            eave_height=float(h['eave_height']), ridge_height=float(h['ridge_height']),
            ridge_direction=str(h.get('ridge_direction', '')), gutters=gutters))
    lim = data.get('limits', {})
    limits = Limits(
        min_altitude=float(lim.get('min_altitude', Limits.min_altitude)),
        max_altitude=float(lim.get('max_altitude', Limits.max_altitude)),
        max_distance=float(lim.get('max_distance', Limits.max_distance)),
        margin=float(lim.get('margin', Limits.margin)),
        no_fly=tuple(h.box() for h in houses) + tuple(r.zone() for r in rough))
    cam = data.get('camera', {})
    return WorldMap(name=str(data.get('name', 'map')), houses=tuple(houses), limits=limits,
                    camera_hfov_deg=float(cam.get('horizontal_fov_deg', 90)),
                    camera_vfov_deg=float(cam.get('vertical_fov_deg', 60)),
                    rough_houses=tuple(rough),
                    search_radius=float(data.get('search_radius', 0.0)),
                    max_building_height=float(data.get('max_building_height', 12.0)),
                    townhouse=bool((data.get('site') or {}).get('townhouse', False)))
