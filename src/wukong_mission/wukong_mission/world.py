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

from wukong_flight.skills import Box, Limits
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
class WorldMap:
    name: str
    houses: Tuple[House, ...]
    limits: Limits
    camera_hfov_deg: float
    camera_vfov_deg: float

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
    for h in data.get('houses', []):
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
        no_fly=tuple(h.box() for h in houses))
    cam = data.get('camera', {})
    return WorldMap(name=str(data.get('name', 'map')), houses=tuple(houses), limits=limits,
                    camera_hfov_deg=float(cam.get('horizontal_fov_deg', 90)),
                    camera_vfov_deg=float(cam.get('vertical_fov_deg', 60)))
