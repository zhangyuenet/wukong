# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Scenario generator v2 (M3): buildings made of blocks, several roof types, townhouses.

A scenario YAML with a `buildings:` list is generated here (sim/scenario.py
dispatches to build()). Each building is either a `house` made of blocks (main
house, garage, porch...) or a `row` of townhouse units under one roof.

Roofs are written as OBJ meshes (sim/generated/<name>/meshes/), so hip roofs and
blocks at any heading are exact. Every eave gets a gutter with per-segment state.
房屋由若干“体块”组成（主屋、车库、门廊），屋顶用 OBJ 网格表示；每条屋檐都有水槽。

Block frame: x along the block's length (its ridge for gable/hip roofs), y across,
z up, origin at the centre of the footprint on the ground. A block's `side` names:
  +y = left, -y = right (looking along +x), +x = end, -x = start.
"""

import math
import os
import random
from typing import List, Optional, Sequence, Tuple

STATES = ('clean', 'leaves', 'blocked')
COMPASS = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
RGB = Tuple[float, float, float]
Point = Tuple[float, float]
BuildingOut = Tuple[str, List[dict], dict]   # (model SDF, gutters' truth, building info)

ROOF_COLOR = (0.45, 0.15, 0.12)
WALL_COLORS = [(0.85, 0.80, 0.70), (0.78, 0.82, 0.86), (0.86, 0.76, 0.66), (0.80, 0.84, 0.74),
               (0.88, 0.84, 0.80), (0.74, 0.70, 0.66)]
WHITE = (0.92, 0.92, 0.92)
DOOR = (0.35, 0.20, 0.10)
WINDOW = (0.30, 0.45, 0.60)


def compass(bearing_deg: float) -> str:
    return COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


# -- frames -----------------------------------------------------------------------

class Frame:
    """A horizontal frame: origin (north, east), x axis at compass bearing."""

    def __init__(self, origin: Tuple[float, float], bearing_deg: float) -> None:
        self.origin = origin
        self.bearing = bearing_deg % 360
        b = math.radians(self.bearing)
        self.ax = (math.cos(b), math.sin(b))        # x axis in (north, east)
        lb = math.radians(self.bearing - 90)        # y axis: to the left of x
        self.ay = (math.cos(lb), math.sin(lb))

    def ne(self, x: float, y: float) -> Tuple[float, float]:
        return (self.origin[0] + x * self.ax[0] + y * self.ay[0],
                self.origin[1] + x * self.ax[1] + y * self.ay[1])

    def enu(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        n, e = self.ne(x, y)
        return e, n, z

    @property
    def yaw_enu(self) -> float:
        """Gazebo yaw of the frame's x axis (ENU, counter-clockwise from east)."""
        return math.radians(90.0 - self.bearing)

    def bearing_of(self, dx: float, dy: float) -> float:
        n = dx * self.ax[0] + dy * self.ay[0]
        e = dx * self.ax[1] + dy * self.ay[1]
        return math.degrees(math.atan2(e, n)) % 360

    def child(self, x: float, y: float, turn_deg: float = 0.0) -> 'Frame':
        return Frame(self.ne(x, y), self.bearing + turn_deg)


# -- SDF and mesh output ---------------------------------------------------------------

class Writer:
    """Collects SDF visuals/collisions and OBJ meshes for one building model."""

    def __init__(self, mesh_dir: str, prefix: str) -> None:
        self.mesh_dir, self.prefix = mesh_dir, prefix
        self.parts: List[str] = []
        self.n = 0

    def _name(self, base: str) -> str:
        self.n += 1
        return f'{base}_{self.n}'

    @staticmethod
    def _mat(rgb) -> str:
        c = ' '.join(f'{v:.3f}' for v in rgb) + ' 1'
        return f'<material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>'

    def box(self, f: Frame, x: float, y: float, z: float, size: Sequence[float], rgb: RGB,
            collide: bool = False, base: str = 'box', yaw_extra: float = 0.0) -> None:
        """Box centred at (x, y, z) in frame f, aligned with it."""
        e, n, zz = f.enu(x, y, z)
        yaw = f.yaw_enu + yaw_extra
        pose = f'<pose>{e:.3f} {n:.3f} {zz:.3f} 0 0 {yaw:.5f}</pose>'
        geo = (f'<geometry><box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size>'
               f'</box></geometry>')
        nm = self._name(base)
        self.parts.append(f'      <visual name="{nm}">{pose}{geo}{self._mat(rgb)}</visual>\n')
        if collide:
            self.parts.append(f'      <collision name="{nm}_c">{pose}{geo}</collision>\n')

    def prism(self, f: Frame, poly: Sequence[Tuple[float, float, float]], thickness: float,
              rgb: RGB, base: str = 'roof') -> None:
        """Add a planar polygon (block-frame points), extruded below its plane: closed mesh."""
        pts = [f.enu(*p) for p in poly]
        # Normal of the polygon (Newell), pointed upwards (or anywhere if vertical).
        nx = ny = nz = 0.0
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        top = pts
        bot = [(p[0] - nx * thickness, p[1] - ny * thickness, p[2] - nz * thickness)
               for p in pts]
        k = len(pts)
        verts = top + bot
        faces = []
        for i in range(1, k - 1):
            faces.append((0, i, i + 1))
            faces.append((k, k + i + 1, k + i))
        for i in range(k):
            j = (i + 1) % k
            faces.append((i, k + i, k + j))
            faces.append((i, k + j, j))
        # Both windings, so the mesh renders from either side regardless of culling.
        faces += [(a, c, b) for a, b, c in faces]
        nm = self._name(base)
        path = os.path.join(self.mesh_dir, f'{self.prefix}_{nm}.obj')
        # Every face needs a normal: Gazebo's physics (dartsim) drops meshes
        # without normals, and the empty collision shape then aborts the server
        # (ODE "vertices" assertion). 每个面都要写法向量：没有法向量时物理引擎丢弃网格，
        # 空的碰撞体会让 Gazebo 服务端崩溃。
        with open(path, 'w', encoding='utf-8') as fh:
            for v in verts:
                fh.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
            for a, b, c in faces:
                n = _face_normal(verts[a], verts[b], verts[c])
                fh.write(f'vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n')
            for k, (a, b, c) in enumerate(faces, 1):
                fh.write(f'f {a + 1}//{k} {b + 1}//{k} {c + 1}//{k}\n')
        geo = f'<geometry><mesh><uri>file://{path}</uri></mesh></geometry>'
        self.parts.append(f'      <visual name="{nm}">{geo}{self._mat(rgb)}</visual>\n'
                          f'      <collision name="{nm}_c">{geo}</collision>\n')

    def model(self, name: str) -> str:
        return (f'  <model name="{name}">\n    <static>true</static>\n'
                f'    <link name="link">\n{"".join(self.parts)}    </link>\n  </model>\n')


def _face_normal(a, b, c):
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
    return n[0] / ln, n[1] / ln, n[2] / ln


# -- blocks ---------------------------------------------------------------------------------

class Block:
    """
    One box-shaped part of a building with its own roof.

    roof: gable (ridge along x, eaves on left/right), hip (eaves on all four
    sides, needs length >= width), shed (one slope falling towards `low_side`,
    one eave), flat-less porch is a shed on posts.
    """

    def __init__(self, frame: Frame, cfg: dict, name: str) -> None:
        self.f = frame
        self.name = name
        self.L, self.W = float(cfg['size'][0]), float(cfg['size'][1])
        self.H = float(cfg.get('wall_height', 3.0))
        self.pitch = math.radians(float(cfg.get('pitch', 30.0)))
        self.over = float(cfg.get('overhang', 0.5))
        self.roof = cfg.get('roof', 'gable')
        self.posts = bool(cfg.get('posts', False))      # open porch: posts, no walls
        self.low_side = cfg.get('low_side', 'right')    # shed only
        if self.roof == 'hip' and self.L < self.W:
            raise ValueError(f'{name}: a hip roof needs length >= width')
        t = math.tan(self.pitch)
        self.eave_h = self.H - self.over * t
        if self.roof == 'shed':
            self.ridge_h = self.H + self.W * t
        else:
            self.ridge_h = self.H + self.W / 2 * t

    def eaves(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float], float]]:
        """(side, start, end, outward bearing) in block coordinates, start->end along the eave."""
        L2, W2, o = self.L / 2 + self.over, self.W / 2 + self.over, self.over
        out = []
        if self.roof in ('gable', 'hip'):
            out.append(('left', (-L2, W2), (L2, W2), self.f.bearing_of(0, 1)))
            out.append(('right', (-L2, -W2), (L2, -W2), self.f.bearing_of(0, -1)))
        if self.roof == 'hip':
            out.append(('end', (L2, -W2), (L2, W2), self.f.bearing_of(1, 0)))
            out.append(('start', (-L2, -W2), (-L2, W2), self.f.bearing_of(-1, 0)))
        if self.roof == 'shed':
            y = -W2 if self.low_side == 'right' else W2
            sgn = -1 if self.low_side == 'right' else 1
            out.append((self.low_side, (-L2 + o, y), (L2 - o, y), self.f.bearing_of(0, sgn)))
        return out

    def build(self, w: Writer, wall_rgb: RGB, details: bool = True) -> None:
        f, L, W, H, o = self.f, self.L, self.W, self.H, self.over
        L2, W2 = L / 2, W / 2
        if self.posts:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    w.box(f, sx * (L2 - 0.1), sy * (W2 - 0.1), H / 2, (0.15, 0.15, H), WHITE,
                          True, 'post')
        else:
            w.box(f, 0, 0, H / 2, (L, W, H), wall_rgb, True, 'walls')
        eh = self.eave_h
        if self.roof == 'gable':
            for s in (1, -1):
                w.prism(f, [(-L2 - o, s * (W2 + o), eh), (L2 + o, s * (W2 + o), eh),
                            (L2 + o, 0, self.ridge_h), (-L2 - o, 0, self.ridge_h)],
                        0.12, ROOF_COLOR)
            for sx in (1, -1):   # gable end walls
                w.prism(f, [(sx * L2, -W2, H), (sx * L2, W2, H),
                            (sx * L2, 0, self.ridge_h - 0.05)],
                        0.12, wall_rgb, 'gable')
        elif self.roof == 'hip':
            rx = L2 - W2      # ridge half-length
            for s in (1, -1):
                w.prism(f, [(-L2 - o, s * (W2 + o), eh), (L2 + o, s * (W2 + o), eh),
                            (rx, 0, self.ridge_h), (-rx, 0, self.ridge_h)], 0.12, ROOF_COLOR)
            for sx in (1, -1):
                w.prism(f, [(sx * (L2 + o), -W2 - o, eh), (sx * (L2 + o), W2 + o, eh),
                            (sx * rx, 0, self.ridge_h)], 0.12, ROOF_COLOR)
        elif self.roof == 'shed':
            low = -1 if self.low_side == 'right' else 1
            w.prism(f, [(-L2, low * (W2 + o), eh), (L2, low * (W2 + o), eh),
                        (L2, -low * W2, self.ridge_h), (-L2, -low * W2, self.ridge_h)],
                    0.12, ROOF_COLOR)
        if details and not self.posts:
            # A door and windows on the right (-y) face, so photos are not featureless.
            w.box(f, 0, -W2 - 0.01, 1.05, (1.0, 0.02, 2.1), DOOR, base='door')
            for x in (-L2 * 0.6, L2 * 0.6):
                if abs(x) > 1.5:
                    w.box(f, x, -W2 - 0.01, min(1.6, H - 1.2), (1.2, 0.02, 1.0), WINDOW,
                          base='window')
                    if H > 5:
                        w.box(f, x, -W2 - 0.01, H - 1.3, (1.2, 0.02, 1.0), WINDOW, base='window')


# -- gutters and debris ----------------------------------------------------------------------

def _debris(rng, x0, x1, state):
    """(centre, length, height, colour, yaw) of boxes in the gutter between x0 and x1."""
    browns = [(0.40, 0.30, 0.10), (0.45, 0.32, 0.12), (0.30, 0.22, 0.08)]
    greens = [(0.25, 0.40, 0.12), (0.30, 0.45, 0.15)]
    out = []
    if state == 'leaves':
        for _ in range(rng.randint(2, 3)):
            ln = min(rng.uniform(0.25, 0.5), (x1 - x0) / 2)
            xc = rng.uniform(x0 + ln / 2, x1 - ln / 2)
            out.append((xc, ln, rng.uniform(0.18, 0.26), rng.choice(browns + greens),
                        rng.uniform(-0.3, 0.3)))
    elif state == 'blocked':
        x = x0 + 0.1
        while x < x1 - 0.3:
            ln = min(rng.uniform(0.5, 0.9), x1 - 0.1 - x)
            out.append((x + ln / 2, ln, rng.uniform(0.26, 0.36), rng.choice(browns + greens),
                        rng.uniform(-0.15, 0.15)))
            x += ln + rng.uniform(0.0, 0.1)
    return out


def add_gutter(w: Writer, rng: random.Random, block_frame: Frame, start: Point, end: Point,
               height: float, outward_bearing: float, states: Sequence[str], label: str,
               owner: str) -> dict:
    """Gutter along an eave (block coordinates) with debris per segment; returns its truth."""
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy)
    # A frame along the gutter: x from start to end, y pointing outwards.
    dn, de = _dir_ne(block_frame, sx, sy, ex, ey)
    along = math.degrees(math.atan2(de, dn)) % 360
    gf = Frame(block_frame.ne((sx + ex) / 2, (sy + ey) / 2), along)
    # Which way is outwards in the gutter frame? y (left) or -y (right).
    left_bearing = (along - 90) % 360
    out_sign = 1 if abs(((outward_bearing - left_bearing) + 180) % 360 - 180) < 90 else -1
    gy = out_sign * 0.08
    gh = height - 0.04
    w.box(gf, 0, gy, gh - 0.01, (length, 0.16, 0.02), WHITE, base='gutter')
    w.box(gf, 0, gy + out_sign * 0.08, gh + 0.06, (length, 0.02, 0.14), WHITE, base='lip')
    n = len(states)
    seg = length / n
    truth = {'name': label, 'owner': owner, 'height': round(gh, 3),
             'side_bearing_deg': round(outward_bearing % 360, 1),
             'from': _r(gf.ne(-length / 2, gy)), 'to': _r(gf.ne(length / 2, gy)),
             'segments': [], 'debris': []}
    for i, state in enumerate(states):
        if state not in STATES:
            raise ValueError(f'unknown gutter state {state}; use {STATES}')
        x0 = -length / 2 + i * seg
        x1 = x0 + seg
        for xc, ln, ht, col, yw in _debris(rng, x0, x1, state):
            w.box(gf, xc, gy, gh + ht / 2, (ln, 0.14, ht), col, base='debris', yaw_extra=yw)
            truth['debris'].append({'from': _r(gf.ne(xc - ln / 2, gy)),
                                    'to': _r(gf.ne(xc + ln / 2, gy)), 'segment': i + 1})
        truth['segments'].append({'index': i + 1, 'state': state,
                                  'from': _r(gf.ne(x0, gy)), 'to': _r(gf.ne(x1, gy))})
    return truth


def _dir_ne(f: Frame, sx, sy, ex, ey):
    a, b = f.ne(sx, sy), f.ne(ex, ey)
    return b[0] - a[0], b[1] - a[1]


def _r(p):
    return [round(p[0], 3), round(p[1], 3)]


def _states(spec, side, default_n):
    s = (spec or {}).get(side)
    if s is None:
        return ['clean'] * default_n
    return [s] if isinstance(s, str) else list(s)


# -- buildings -------------------------------------------------------------------------------

def build_house(cfg: dict, idx: int, rng: random.Random, mesh_dir: str,
                target: bool) -> BuildingOut:
    """Build a detached house: blocks placed relative to the house frame."""
    name = cfg.get('name', f'house{idx + 1}')
    hf = Frame(tuple(cfg['center']), float(cfg.get('bearing', 90)))
    w = Writer(mesh_dir, name)
    wall = tuple(cfg.get('wall_color', WALL_COLORS[idx % len(WALL_COLORS)]))
    gutters, footprints = [], []
    blocks = []
    for bi, bc in enumerate(cfg['blocks']):
        off = bc.get('offset', [0, 0])
        bf = hf.child(float(off[0]), float(off[1]), float(bc.get('turn', 0)))
        blocks.append((bc, bf, Block(bf, bc, f'{name}/{bc.get("name", f"block{bi + 1}")}')))
    for bi, (bc, bf, b) in enumerate(blocks):
        bname = bc.get('name', f'block{bi + 1}')
        b.build(w, wall, details=bc.get('details', bi == 0))
        footprints.append(_footprint(b))
        others = [o for k, (_, _, o) in enumerate(blocks) if k != bi]
        for side, s, e, out in b.eaves():
            # A gutter stops where it meets the wall of another, taller part of
            # the house (not where it passes above a lower roof or a porch).
            # 水槽碰到同一栋房屋中更高部分的墙就截止；从较低的屋顶或门廊上方经过不截断。
            walls = [o for o in others if not o.posts and o.H >= b.eave_h - 0.2]
            s, e = _clip_to_outside(bf, s, e, walls)
            if s is None:
                continue
            length = math.dist(s, e)
            states = _states(bc.get('gutters'), side, max(1, round(length / 3.0)))
            label = f'{bname} {compass(out)} gutter'
            g = add_gutter(w, rng, bf, s, e, b.eave_h, out, states, label, name)
            # The stretch visible from above: not under a higher roof of the
            # house (its overhang and gutter). 从上方可见的部分：不在更高屋顶（含屋檐和水槽）之下。
            higher = [o for o in others if o.eave_h > b.eave_h + 0.5]
            vs, ve = s, e
            if higher:
                vs, ve = _clip_to_outside(bf, s, e, higher, grow=max(o.over for o in higher)
                                          + 0.25)
            if vs is not None:
                g['visible'] = [_r(bf.ne(*vs)), _r(bf.ne(*ve))]
            gutters.append(g)
    info = {'name': name, 'kind': 'house', 'target': target,
            'height': round(max(_block_top(bc) for bc in cfg['blocks']), 2),
            'footprints': footprints}
    return w.model(name), gutters, info


def build_row(cfg: dict, idx: int, rng: random.Random, mesh_dir: str,
              target_unit: Optional[int]) -> BuildingOut:
    """Build a row of townhouse units under one roof (gable or hip ends)."""
    name = cfg.get('name', f'row{idx + 1}')
    n = int(cfg['units'])
    uw = float(cfg.get('unit_width', 6.0))
    depth = float(cfg.get('depth', 10.0))
    rf = Frame(tuple(cfg['center']), float(cfg.get('bearing', 90)))
    roof = cfg.get('roof', 'gable')
    blk = {'size': [n * uw, depth], 'wall_height': cfg.get('wall_height', 5.6),
           'pitch': cfg.get('pitch', 30), 'overhang': cfg.get('overhang', 0.5), 'roof': roof}
    b = Block(rf, blk, name)
    w = Writer(mesh_dir, name)
    b.build(w, WALL_COLORS[0], details=False)
    # Per unit: its own wall colour strip on the front, a door, windows, a party-wall line.
    for u in range(n):
        x0 = -n * uw / 2 + u * uw
        xc = x0 + uw / 2
        col = WALL_COLORS[u % len(WALL_COLORS)]
        for s in (1, -1):
            w.box(rf, xc, s * (depth / 2 + 0.005), b.H / 2, (uw - 0.1, 0.01, b.H), col,
                  base='facade')
        w.box(rf, xc - uw * 0.25, -depth / 2 - 0.02, 1.05, (1.0, 0.02, 2.1), DOOR, base='door')
        for z in (1.6, b.H - 1.3):
            w.box(rf, xc + uw * 0.2, -depth / 2 - 0.02, z, (1.4, 0.02, 1.0), WINDOW, base='window')
        if u > 0:   # downspout at each party wall
            for s in (1, -1):
                w.box(rf, x0, s * (depth / 2 + 0.06), b.eave_h / 2, (0.08, 0.08, b.eave_h), WHITE,
                      base='downspout')
    gutters = []
    units_truth = []
    L2 = n * uw / 2 + b.over
    for side, s, e, out in b.eaves():
        # Split each long eave at the party walls, so every unit owns its stretch.
        if side in ('left', 'right'):
            y = s[1]
            for u in range(n):
                x0 = -n * uw / 2 + u * uw - (b.over if u == 0 else 0)
                x1 = -n * uw / 2 + (u + 1) * uw + (b.over if u == n - 1 else 0)
                owner = f'{name}/unit{u + 1}'
                spec = (cfg.get('unit_gutters') or {}).get(u + 1, {})
                states = _states(spec, side, max(1, round((x1 - x0) / 3.0)))
                label = f'unit {u + 1} {compass(out)} gutter'
                gutters.append(add_gutter(w, rng, rf, (x0, y), (x1, y), b.eave_h, out, states,
                                          label, owner))
        else:   # hip ends belong to the end units
            u = n if side == 'end' else 1
            owner = f'{name}/unit{u}'
            spec = (cfg.get('unit_gutters') or {}).get(u, {})
            states = _states(spec, side, max(1, round(math.dist(s, e) / 3.0)))
            label = f'unit {u} {compass(out)} gutter'
            gutters.append(add_gutter(w, rng, rf, s, e, b.eave_h, out, states, label, owner))
    for u in range(n):
        x0 = -n * uw / 2 + u * uw
        units_truth.append({'unit': u + 1, 'from': _r(rf.ne(x0, 0)), 'to': _r(rf.ne(x0 + uw, 0)),
                            'target': target_unit == u + 1})
    info = {'name': name, 'kind': 'row', 'units': units_truth, 'target': target_unit is not None,
            'target_unit': target_unit, 'height': round(b.ridge_h, 2),
            'footprints': [_footprint(b)], 'row_extent': [_r(rf.ne(-L2, 0)), _r(rf.ne(L2, 0))]}
    return w.model(name), gutters, info


def _inside_walls(b: Block, ne, grow: float = 0.0) -> bool:
    dn, de = ne[0] - b.f.origin[0], ne[1] - b.f.origin[1]
    u = dn * b.f.ax[0] + de * b.f.ax[1]
    v = dn * b.f.ay[0] + de * b.f.ay[1]
    return abs(u) < b.L / 2 + grow - 1e-6 and abs(v) < b.W / 2 + grow - 1e-6


def _clip_to_outside(f: Frame, s, e, others, step: float = 0.05, grow: float = 0.0):
    """Longest stretch of the eave s -> e (block coordinates) outside the others' outlines."""
    n = max(1, int(math.dist(s, e) / step))
    out = [not any(_inside_walls(o, f.ne(s[0] + (e[0] - s[0]) * i / n,
                                         s[1] + (e[1] - s[1]) * i / n), grow) for o in others)
           for i in range(n + 1)]
    best, run = (0, -1), None
    for i, ok in enumerate(out + [False]):
        if ok and run is None:
            run = i
        elif not ok and run is not None:
            if i - 1 - run > best[1] - best[0]:
                best = (run, i - 1)
            run = None
    if best[1] - best[0] < 2:
        return None, None
    t0, t1 = best[0] / n, best[1] / n
    return ((s[0] + (e[0] - s[0]) * t0, s[1] + (e[1] - s[1]) * t0),
            (s[0] + (e[0] - s[0]) * t1, s[1] + (e[1] - s[1]) * t1))


def _block_top(bc) -> float:
    W = float(bc['size'][1])
    t = math.tan(math.radians(float(bc.get('pitch', 30))))
    H = float(bc.get('wall_height', 3.0))
    return H + (W * t if bc.get('roof') == 'shed' else W / 2 * t)


def _footprint(b: Block):
    L2, W2 = b.L / 2 + b.over, b.W / 2 + b.over
    return {'block': b.name, 'corners': [_r(b.f.ne(sx * L2, sy * W2))
                                         for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))],
            'eave_height': round(b.eave_h, 3), 'top': round(b.ridge_h, 3)}


def build(cfg: dict, name: str, out_dir: str) -> Tuple[str, dict, dict]:
    """Return (models SDF, truth dict, rough map dict) for a v2 scenario."""
    rng = random.Random(cfg.get('seed', 1))
    mesh_dir = os.path.join(out_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    for f in os.listdir(mesh_dir):
        if f.endswith('.obj'):
            os.remove(os.path.join(mesh_dir, f))
    models, gutters, infos = [], [], []
    target_name = cfg.get('target')            # building name, or "row1:3" style for a unit
    for i, bc in enumerate(cfg['buildings']):
        bname = bc.get('name', f'b{i + 1}')
        bc = {**bc, 'name': bname}
        if bc.get('type', 'house') == 'row':
            tu = None
            if target_name and target_name.startswith(bname + ':'):
                tu = int(target_name.split(':')[1])
            m, g, info = build_row(bc, i, rng, mesh_dir, tu)
        else:
            m, g, info = build_house(bc, i, rng, mesh_dir, target_name == bname)
        models.append(m)
        gutters += g
        infos.append(info)
    target_owner = target_name.replace(':', '/unit') if target_name and ':' in target_name \
        else target_name
    for g in gutters:
        g['target'] = g['owner'] == target_owner
    truth = {'name': name, 'version': 2, 'target': target_name, 'buildings': infos,
             'gutters': gutters}
    rough = {'name': f'scenario {name}', 'search_radius': float(cfg.get('search_radius', 30.0)),
             'max_building_height': float(cfg.get('max_building_height', 12.0)),
             'limits': {'min_altitude': 1.5, 'max_altitude': 18.0, 'max_distance': 45.0,
                        'margin': 2.0},
             'camera': {'horizontal_fov_deg': 69, 'vertical_fov_deg': 42}}
    return ''.join(models), truth, rough
