#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Generate a simulation scenario: a gabled house with segmented gutters.

    python3 sim/scenario.py sim/scenarios/basic.yaml   # -> sim/generated/basic/

Writes, for scenario <name>:
  world.sdf   Gazebo world (PX4 default world + the house, built inline)
  truth.json  ground truth: eave lines, gutter segments and their state
  map.yaml    what the drone is told: the house's rough position only
  spawn_pose  where scripts/sim.sh puts the drone: x,y,z,roll,pitch,yaw (Gazebo)

Frames: Gazebo world is ENU (x east, y north); PX4's local frame starts at the
world origin, so local north = world y, local east = world x. Truth and map use
(north, east). 坐标：Gazebo 是 ENU，PX4 本地坐标原点与世界原点重合。

No ROS imports: runs with plain python3 + PyYAML. 只依赖 python3 和 PyYAML。
"""

import argparse
import json
import math
import os
import random
import re
import sys
from typing import List, Optional, Tuple

import yaml

STATES = ('clean', 'leaves', 'blocked')
HERE = os.path.dirname(os.path.abspath(__file__))
COMPASS = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

DEFAULTS = {
    'length': 10.0,          # m, along the ridge
    'depth': 8.0,            # m, across the ridge
    'wall_height': 3.0,
    'roof_pitch_deg': 30.0,
    'overhang': 0.5,
    'center': [15.0, 0.0],   # north, east
    'ridge_bearing_deg': 90.0,   # compass bearing of the ridge: 0 north, 90 east
    'map_error': 1.0,        # m, max error of the rough position in map.yaml
    'seed': 1,
}


def compass(bearing_deg: float) -> str:
    return COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


class House:
    """Geometry in the model frame: x along the ridge, y across, z up; origin on the ground."""

    def __init__(self, cfg: dict) -> None:
        c = {**DEFAULTS, **cfg.get('house', {})}
        self.length, self.depth = float(c['length']), float(c['depth'])
        self.wall_h, self.over = float(c['wall_height']), float(c['overhang'])
        self.pitch = math.radians(float(c['roof_pitch_deg']))
        self.center = (float(c['center'][0]), float(c['center'][1]))
        self.bearing = float(c['ridge_bearing_deg'])
        # Model x axis in ENU, angle from east counter-clockwise.
        self.yaw = math.radians(90.0 - self.bearing)
        hd = self.depth / 2
        self.ridge_h = self.wall_h + hd * math.tan(self.pitch)
        self.eave_h = self.wall_h - self.over * math.tan(self.pitch)
        self.gutter_y = hd + self.over + 0.08       # gutter centre line, |y|
        self.gutter_h = self.eave_h - 0.04          # top of the gutter bottom plate
        self.gutter_len = self.length + 2 * self.over

    def to_ne(self, x: float, y: float) -> Tuple[float, float]:
        """Model frame (x, y) -> local (north, east)."""
        e = x * math.cos(self.yaw) - y * math.sin(self.yaw)
        n = x * math.sin(self.yaw) + y * math.cos(self.yaw)
        return round(self.center[0] + n, 3), round(self.center[1] + e, 3)

    def side_bearing(self, sign: int) -> float:
        """Compass bearing of the outward normal of the eave at y = sign * ..."""
        # model +y in ENU = yaw + 90 deg; bearing = 90 - enu angle
        enu = math.degrees(self.yaw) + (90 if sign > 0 else -90)
        return (90 - enu) % 360


def _box(name, pose, size, rgb, collide=False):
    x, y, z, r, p, yw = pose
    color = ' '.join(f'{v:.3f}' for v in rgb) + ' 1'
    geo = f'<geometry><box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size></box></geometry>'
    pose_s = f'<pose>{x:.3f} {y:.3f} {z:.3f} {r:.4f} {p:.4f} {yw:.4f}</pose>'
    out = (f'      <visual name="{name}">{pose_s}{geo}'
           f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>\n')
    if collide:
        out += f'      <collision name="{name}_c">{pose_s}{geo}</collision>\n'
    return out


def _debris(rng, x0, x1, y, z, state):
    """
    Boxes standing in the gutter between x0 and x1, sticking out above the lip.

    The lip is 0.13 m above the gutter bottom. Viewed from ~15 degrees above,
    it hides the bottom, so debris must stand above it to be visible at all.
    外沿高 0.13 米；从约 15 度俯角看不到水槽底部，杂物必须高出外沿才看得见。
    """
    parts = []
    browns = [(0.40, 0.30, 0.10), (0.45, 0.32, 0.12), (0.30, 0.22, 0.08)]
    greens = [(0.25, 0.40, 0.12), (0.30, 0.45, 0.15)]
    if state == 'leaves':     # a few separate clumps
        for _ in range(rng.randint(2, 3)):
            ln = rng.uniform(0.25, 0.5)
            xc = rng.uniform(x0 + ln / 2, x1 - ln / 2)
            h = rng.uniform(0.18, 0.26)
            parts.append((xc, ln, h, rng.choice(browns + greens), rng.uniform(-0.3, 0.3)))
    elif state == 'blocked':  # an almost continuous pile
        x = x0 + 0.1
        while x < x1 - 0.3:
            ln = min(rng.uniform(0.5, 0.9), x1 - 0.1 - x)
            h = rng.uniform(0.26, 0.36)
            parts.append((x + ln / 2, ln, h, rng.choice(browns + greens),
                          rng.uniform(-0.15, 0.15)))
            x += ln + rng.uniform(0.0, 0.1)
    return [(xc, y, z + h / 2, ln, 0.14, h, col, yw) for xc, ln, h, col, yw in parts]


def build(cfg: dict, name: str) -> Tuple[str, dict, dict]:
    rng = random.Random(cfg.get('seed', DEFAULTS['seed']))
    h = House(cfg)
    hd, hl = h.depth / 2, h.length / 2
    t = math.tan(h.pitch)
    wall = (0.85, 0.80, 0.70)
    roof = tuple(cfg.get('house', {}).get('roof_color', (0.45, 0.15, 0.12)))
    white = (0.92, 0.92, 0.92)

    v = _box('walls', (0, 0, h.wall_h / 2, 0, 0, 0), (h.length, h.depth, h.wall_h), wall, True)
    slab_w = (hd + h.over) / math.cos(h.pitch)
    zc = (h.eave_h + h.ridge_h) / 2
    for sign, nm in ((1, 'roof_a'), (-1, 'roof_b')):
        v += _box(nm, (0, sign * (hd + h.over) / 2, zc, -sign * h.pitch, 0, 0),
                  (h.length + 2 * h.over, slab_w, 0.15), roof, True)
    # Gable end walls: stacked boxes approximating the triangle, so end views are closed.
    steps = 6
    for i in range(steps):
        z0 = h.wall_h + (h.ridge_h - h.wall_h) * i / steps
        dz = (h.ridge_h - h.wall_h) / steps
        # Width at the top of the step, so no corner pokes through the roof.
        w = 2 * (h.ridge_h - z0 - dz) / t if t > 0 else h.depth
        for sx in (1, -1):
            v += _box(f'gable_{sx}_{i}', (sx * (hl - 0.05), 0, z0 + dz / 2, 0, 0, 0),
                      (0.1, max(w, 0.1), dz), wall)

    truth = {
        'name': name,
        'house': {
            'center': list(h.center), 'length': h.length, 'depth': h.depth,
            'ridge_bearing_deg': h.bearing, 'wall_height': h.wall_h,
            'eave_height': round(h.eave_h, 3), 'ridge_height': round(h.ridge_h, 3),
            'overhang': h.over,
            'footprint_with_overhang': [h.to_ne(sx * (hl + h.over), sy * (hd + h.over))
                                        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]},
        'gutters': []}

    gutter_cfg = cfg.get('gutters', {})
    used_names = set()
    for sign, key in ((1, 'a'), (-1, 'b')):
        y = sign * h.gutter_y
        gname = compass(h.side_bearing(sign)) + ' gutter'
        if gname in used_names:
            gname += f' ({key})'
        used_names.add(gname)
        # Gutter: bottom plate + outer lip. 水槽：底板加外沿。
        v += _box(f'gutter_{key}_bottom', (0, y, h.gutter_h - 0.01, 0, 0, 0),
                  (h.gutter_len, 0.16, 0.02), white)
        v += _box(f'gutter_{key}_lip', (0, y + sign * 0.08, h.gutter_h + 0.06, 0, 0, 0),
                  (h.gutter_len, 0.02, 0.14), white)
        states = gutter_cfg.get(key, gutter_cfg.get(gname.split()[0], ['clean']))
        if isinstance(states, str):
            states = [states]
        bad = [s for s in states if s not in STATES]
        if bad:
            raise ValueError(f'unknown gutter state(s) {bad}; use {STATES}')
        n = len(states)
        seg_len = h.gutter_len / n
        g = {'name': gname, 'side_bearing_deg': round(h.side_bearing(sign), 1),
             'height': round(h.gutter_h, 3),
             'from': h.to_ne(-h.gutter_len / 2, y), 'to': h.to_ne(h.gutter_len / 2, y),
             'segments': []}
        g['debris'] = []    # where debris really is, for scoring by position
        for i, state in enumerate(states):
            x0 = -h.gutter_len / 2 + i * seg_len
            x1 = x0 + seg_len
            for j, (xc, yy, zz, ln, wd, ht, col, yw) in enumerate(
                    _debris(rng, x0, x1, y, h.gutter_h, state)):
                v += _box(f'debris_{key}_{i}_{j}', (xc, yy, zz, 0, 0, yw), (ln, wd, ht), col)
                g['debris'].append({'from': h.to_ne(xc - ln / 2, y),
                                    'to': h.to_ne(xc + ln / 2, y), 'segment': i + 1})
            g['segments'].append({'index': i + 1, 'state': state,
                                  'from': h.to_ne(x0, y), 'to': h.to_ne(x1, y)})
        truth['gutters'].append(g)

    ce, cn = h.center[1], h.center[0]
    model = (f'  <model name="house">\n    <static>true</static>\n'
             f'    <pose>{ce:.3f} {cn:.3f} 0 0 0 {h.yaw:.5f}</pose>\n'
             f'    <link name="link">\n{v}    </link>\n  </model>\n')

    # The rough map: position off by up to map_error, and a radius that surely
    # contains the house. 粗略地图：位置有误差，半径保证包住整栋房屋。
    err = float(cfg.get('house', {}).get('map_error', DEFAULTS['map_error']))
    a = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(0, err)
    approx = [round(cn + r * math.cos(a), 1), round(ce + r * math.sin(a), 1)]
    radius = math.hypot(hl + h.over, hd + h.over) + err + 0.5
    rough = {
        'name': f'scenario {name}',
        'houses': [{'name': 'house', 'approx_center': approx,
                    'radius': round(math.ceil(radius * 2) / 2, 1),
                    'max_height': round(math.ceil(h.ridge_h + 1.0), 1)}],
        'limits': {'min_altitude': 1.5, 'max_altitude': 15.0, 'max_distance': 45.0,
                   'margin': 2.0},
        'camera': {'horizontal_fov_deg': 69, 'vertical_fov_deg': 42},
    }
    return model, truth, rough


def world_sdf(world_name: str, model: str) -> str:
    """PX4's default world with the world renamed and the house added."""
    src = os.path.join(os.environ.get('PX4_DIR', os.path.expanduser(
        '~/wip/wukong_deps/PX4-Autopilot')), 'Tools/simulation/gz/worlds/default.sdf')
    with open(src, encoding='utf-8-sig') as f:
        s = f.read()
    s = s.replace('<world name="default">', f'<world name="{world_name}">', 1)
    return s.replace('    <light name="sunUTC"', model + '    <light name="sunUTC"', 1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('scenario', help='scenario YAML file')
    ap.add_argument('--out', default=os.path.join(HERE, 'generated'))
    args = ap.parse_args(argv)
    with open(args.scenario, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    name = os.path.splitext(os.path.basename(args.scenario))[0]
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        print(f'scenario file name must be letters, digits and _: {name}', file=sys.stderr)
        return 2
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    if 'buildings' in cfg:      # M3: blocks, several roof types, townhouses
        sys.path.insert(0, HERE)
        import buildings
        model, truth, rough = buildings.build(cfg, name, out)
    else:                       # M2: one gabled house
        model, truth, rough = build(cfg, name)
    # The way the drone is set down facing, a compass bearing (default north):
    # the operator points it at the house, and discovery moves on that way when
    # the house is out of depth range. Gazebo's yaw turns anticlockwise from
    # east; scripts/sim.sh spawns the drone with spawn_pose.
    # 起飞前机头朝向（罗盘方位，默认正北）：操作员把飞机朝着房屋放下，房屋超出有效深度
    # 距离时，发现阶段朝这个方向前进。Gazebo 的偏航角自东起逆时针；sim.sh 按 spawn_pose
    # 放置飞机。
    facing = float(cfg.get('takeoff_facing_deg', 0.0))
    truth['takeoff_facing_deg'] = facing
    with open(os.path.join(out, 'spawn_pose'), 'w', encoding='utf-8') as f:
        f.write(f'0,0,0,0,0,{math.radians(90.0 - facing):.4f}\n')
    world_name = f'wukong_{name}'
    with open(os.path.join(out, f'{world_name}.sdf'), 'w', encoding='utf-8') as f:
        f.write(world_sdf(world_name, model))
    with open(os.path.join(out, 'truth.json'), 'w', encoding='utf-8') as f:
        json.dump(truth, f, indent=2)
    with open(os.path.join(out, 'map.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'# Generated from {os.path.basename(args.scenario)}: the rough map the '
                'drone gets.\n')
        yaml.safe_dump(rough, f, sort_keys=False, allow_unicode=True)
    print(f'{out}/{world_name}.sdf')
    return 0


if __name__ == '__main__':
    sys.exit(main())
