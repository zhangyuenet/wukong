# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Geometry tests with synthetic depth images and point clouds."""

import math

import numpy as np
import pytest

from wukong_perception.geometry import (compass, depth_to_points, Eave, Intrinsics,
                                        segment_eave, segments_for)

K = Intrinsics(432.5, 432.5, 320.0, 240.0)


def test_depth_to_points_flat_wall_ahead():
    # Facing east (heading 90 deg) from (0, 0, 5): a wall 10 m ahead fills the view.
    depth = np.full((480, 640), 10.0, np.float32)
    pts = depth_to_points(depth, K, (0.0, 0.0, 5.0, math.pi / 2), stride=8)
    assert np.allclose(pts[:, 1], 10.0 + 0.132)          # east = range + camera offset
    centre = pts[np.argmin(np.abs(pts[:, 0]) + np.abs(pts[:, 2] - 5.26))]
    assert centre[2] == pytest.approx(5.26, abs=0.15)    # camera 0.26 m above the body
    # The top rows of the image are higher up. 图像上方对应更高的点。
    assert pts[:, 2].max() > 8 and pts[:, 2].min() < 2


def test_depth_to_points_drops_missing_and_far():
    depth = np.full((480, 640), np.inf, np.float32)
    depth[:10, :10] = 25.0          # beyond max_range
    depth[-10:, -10:] = 5.0
    pts = depth_to_points(depth, K, (0.0, 0.0, 3.0, 0.0), stride=2)
    assert len(pts) == 25 and np.all(pts[:, 0] > 5.0)


def synthetic_house(center=(15.0, 0.0), bearing=90.0, length=10.0, depth=8.0, wall=3.0,
                    pitch=30.0, over=0.5, step=0.15, sides='all', noise=0.0, seed=0):
    """Points on walls, both roof slopes and gutters of a gabled house (north, east, height)."""
    rng = np.random.default_rng(seed)
    t = math.tan(math.radians(pitch))
    hd = depth / 2
    pts = []
    # Walls (model frame: x along the ridge, y across). 墙面。
    for x in np.arange(-length / 2, length / 2, step):
        for z in np.arange(0, wall, step):
            pts += [(x, hd, z), (x, -hd, z)]
    for y in np.arange(-hd, hd, step):
        for z in np.arange(0, wall + (hd - abs(y)) * t, step):
            pts += [(length / 2, y, z), (-length / 2, y, z)]
    # Roof slopes, with overhang. 屋面。
    for x in np.arange(-length / 2 - over, length / 2 + over, step):
        for y in np.arange(-hd - over, hd + over, step):
            pts.append((x, y, wall + (hd - abs(y)) * t))
    # Gutters just outside the eaves. 水槽。
    gy = hd + over + 0.08
    for x in np.arange(-length / 2 - over, length / 2 + over, step / 2):
        for dy in (-0.08, 0.0, 0.08):
            pts += [(x, gy + dy, wall - over * t), (x, -gy - dy, wall - over * t)]
    p = np.array(pts)
    if sides != 'all':        # keep only what a viewer on that model side would see
        p = p[(p[:, 1] < 0.5) if sides == '-y' else (p[:, 1] > -0.5)]
    yaw = math.radians(90 - bearing)    # model x in ENU
    e = p[:, 0] * math.cos(yaw) - p[:, 1] * math.sin(yaw)
    n = p[:, 0] * math.sin(yaw) + p[:, 1] * math.cos(yaw)
    out = np.column_stack([center[0] + n, center[1] + e, p[:, 2]])
    ground = np.column_stack([rng.uniform(0, 30, 3000), rng.uniform(-15, 15, 3000),
                              np.zeros(3000)])
    out = np.vstack([out, ground])
    return out + rng.normal(0, noise, out.shape) if noise else out


def test_segments():
    south = Eave('south gutter', (10.42, -5.5), (10.42, 5.5), 2.7, 180.0)
    n = segments_for(south, 3.0)
    assert n == 4
    segs = segment_eave(south, n)
    assert segs[0][0] == pytest.approx(south.start)
    assert segs[-1][1] == pytest.approx(south.end)


def test_compass():
    assert compass(0) == 'north' and compass(350) == 'north'
    assert compass(135) == 'southeast' and compass(-90) == 'west'


def test_tilt_is_compensated():
    # Nose 5 deg down: a level camera model would put the far ground too high.
    # Synthesize the depth a tilted camera sees of flat ground 4.5 m below.
    pitch = math.radians(-5.0)
    h, w = 480, 640
    v, _ = np.mgrid[0:h, 0:w]
    y = (v - K.cy) / K.fy         # optical down per unit depth
    # Ray in body FRD (fwd=1, right, down=y), rotated by pitch into the level frame;
    # with no roll the sideways part does not change the height.
    down_level = -math.sin(pitch) * 1.0 + math.cos(pitch) * y
    cam_h = 4.5 + 0.26
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(down_level > 0.02, cam_h / down_level, np.inf).astype(np.float32)
    level = depth_to_points(depth, K, (0, 0, 4.5, 0.0), offset_frd=(0, 0, -0.26))
    tilted = depth_to_points(depth, K, (0, 0, 4.5, 0.0, 0.0, pitch), offset_frd=(0, 0, -0.26))
    assert np.abs(tilted[:, 2]).max() < 0.05
    assert np.abs(level[:, 2]).max() > 0.5


def test_project_is_the_inverse_of_depth_to_points():
    from wukong_perception.geometry import project
    pose = (3.0, -2.0, 4.5, 0.7, 0.02, -0.03)
    depth = np.full((480, 640), 8.0, np.float32)
    pts = depth_to_points(depth, K, pose, stride=40)
    uvd = project(pts, K, pose)
    v, u = np.mgrid[0:480:40, 0:640:40]
    assert uvd[:, 0] == pytest.approx(u.ravel(), abs=1e-6)
    assert uvd[:, 1] == pytest.approx(v.ravel(), abs=1e-6)
    assert uvd[:, 2] == pytest.approx(8.0, abs=1e-6)


def test_rgb_intrinsics():
    from wukong_perception.geometry import RGB_X500_DEPTH
    assert RGB_X500_DEPTH.cx == 960 and RGB_X500_DEPTH.cy == 540
    assert RGB_X500_DEPTH.fx == pytest.approx(960 / math.tan(0.602))
