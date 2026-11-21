# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Geometry tests with synthetic depth images and point clouds."""

import math

import numpy as np
import pytest

from wukong_perception.geometry import (compass, depth_to_points, fit_house, FitError,
                                        Intrinsics, segment_eave, segments_for)

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


def test_fit_axis_aligned_house():
    est = fit_house(synthetic_house(), approx_center=(15.5, -0.2), radius=9.0)
    assert est.center[0] == pytest.approx(15.0, abs=0.15)
    assert est.center[1] == pytest.approx(0.0, abs=0.15)
    assert est.ridge_bearing_deg % 180 == pytest.approx(90.0, abs=1.0)
    assert est.length == pytest.approx(11.0, abs=0.3)
    assert est.width == pytest.approx(9.32, abs=0.3)     # 8 + 2 x 0.5 + gutters
    assert est.ridge_height == pytest.approx(5.31, abs=0.15)
    names = sorted(e.name for e in est.eaves)
    assert names == ['north gutter', 'south gutter']
    south = next(e for e in est.eaves if e.name == 'south gutter')
    assert south.start[0] == pytest.approx(10.42, abs=0.2)
    assert south.height == pytest.approx(2.7, abs=0.35)
    assert south.side_bearing_deg == pytest.approx(180.0, abs=1.0)


@pytest.mark.parametrize('bearing', [0.0, 30.0, 60.0, 135.0])
def test_fit_rotated_house(bearing):
    pts = synthetic_house(center=(16.0, 6.0), bearing=bearing, length=13.0, depth=7.0,
                          pitch=35.0, noise=0.03)
    est = fit_house(pts, approx_center=(15.0, 5.0), radius=10.5)
    diff = (est.ridge_bearing_deg - bearing) % 180
    assert min(diff, 180 - diff) < 1.5
    assert math.dist(est.center, (16.0, 6.0)) < 0.25
    assert est.length == pytest.approx(14.0, abs=0.3)
    # Each eave faces away from the house, perpendicular to the ridge.
    for e in est.eaves:
        d = (e.side_bearing_deg - bearing) % 180
        assert min(d, 180 - d) == pytest.approx(90, abs=1.5)
        mid = e.point(0.5)
        out = e.outward
        assert (mid[0] - est.center[0]) * out[0] + (mid[1] - est.center[1]) * out[1] > 3


def test_too_few_points_raise():
    with pytest.raises(FitError):
        fit_house(synthetic_house(), approx_center=(40.0, 40.0), radius=5.0)


def test_segments():
    est = fit_house(synthetic_house(), approx_center=(15.0, 0.0), radius=9.0)
    south = next(e for e in est.eaves if e.name == 'south gutter')
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


def test_altitude_drift_is_measured_from_the_ground():
    # The vehicle's altitude estimate is 0.3 m high: everything appears 0.3 m up.
    pts = synthetic_house() + np.array([0.0, 0.0, 0.3])
    est = fit_house(pts, approx_center=(15.0, 0.0), radius=9.0)
    assert est.ground_level == pytest.approx(0.3, abs=0.02)
    south = next(e for e in est.eaves if e.name == 'south gutter')
    assert south.height - est.ground_level == pytest.approx(2.71, abs=0.1)


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
