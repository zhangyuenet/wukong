# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Voxel map: only what rays crossed is free; occupied wins; swept-volume checks."""

import math

import numpy as np
import pytest

from wukong_perception.voxels import (Ball, Column, FREE, OCCUPIED, UNKNOWN, VoxelConfig,
                                      VoxelMap)

CAM = (0.0, 0.0, 5.0)


def _fan(cam=CAM, half_deg=40.0, n=61):
    """Return unit directions in a square cone around north."""
    a = np.radians(np.linspace(-half_deg, half_deg, n))
    yaw, pitch = np.meshgrid(a, a)
    d = np.column_stack([np.cos(pitch.ravel()) * np.cos(yaw.ravel()),
                         np.cos(pitch.ravel()) * np.sin(yaw.ravel()),
                         np.sin(pitch.ravel())])
    return d


def _wall_scan(m, wall_north=8.1, cam=CAM):
    """Add one frame of rays that all end on a wall across the north."""
    d = _fan(cam)
    t = (wall_north - cam[0]) / d[:, 0]
    ends = np.asarray(cam) + t[:, None] * d
    m.add_rays(cam, ends, np.ones(len(d), bool))


def test_free_up_to_the_wall_occupied_at_it_unknown_behind_and_outside():
    m = VoxelMap((0.0, 0.0))
    _wall_scan(m)
    got = m.states(np.array([(4.0, 0.0, 5.0), (8.2, 0.0, 5.0), (10.0, 0.0, 5.0),
                             (4.0, 0.0, 9.5), (-3.0, 0.0, 5.0), (80.0, 0.0, 5.0)]))
    assert list(got) == [FREE, OCCUPIED, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN]


def test_a_branch_above_an_empty_ray_stays_unknown():
    """A ray under a branch says nothing about the branch (the old column cap did)."""
    m = VoxelMap((0.0, 0.0))
    ends = np.array([(12.0, 0.0, 5.0)])           # one level ray, ending on something far
    m.add_rays(CAM, ends, np.array([True]))
    below, branch = m.states(np.array([(6.0, 0.0, 5.0), (6.0, 0.0, 6.0)]))
    assert below == FREE and branch == UNKNOWN


def test_occupied_wins_until_enough_free_rays_cross_it():
    cfg = VoxelConfig(clear_ratio=3)
    m = VoxelMap((0.0, 0.0), cfg)
    target = np.array([(6.1, 0.1, 5.0)])
    m.add_rays(CAM, target, np.array([True]))      # hit once
    through = np.array([(12.0, 0.2, 5.0)])          # then rays straight through it
    for k in range(cfg.clear_ratio):
        assert m.states(target)[0] == OCCUPIED, k
        m.add_rays(CAM, through, np.array([True]))
    assert m.states(target)[0] == FREE


def test_a_hit_after_free_crossings_makes_the_voxel_occupied_at_once():
    """Something may have come in: what was seen there before does not outweigh a new hit."""
    cfg = VoxelConfig(clear_ratio=3)
    m = VoxelMap((0.0, 0.0), cfg)
    target = np.array([(6.1, 0.1, 5.0)])
    through = np.array([(12.0, 0.2, 5.0)])
    m.assume_free(Ball(CAM, 8.0))                  # the open column counts as seen free too
    for _ in range(cfg.clear_ratio):
        m.add_rays(CAM, through, np.array([True]))
    assert m.states(target)[0] == FREE
    m.add_rays(CAM, target, np.array([True]))
    # Cleared again only by crossings after the hit. 只有击中之后的空射线才能再清除。
    for k in range(cfg.clear_ratio):
        assert m.states(target)[0] == OCCUPIED, k
        m.add_rays(CAM, through, np.array([True]))
    assert m.states(target)[0] == FREE


def test_a_voxel_hit_in_a_frame_is_not_also_free_in_it():
    m = VoxelMap((0.0, 0.0))
    ends = np.array([(6.1, 0.1, 5.0), (12.0, 0.2, 5.0)])   # one ray stops where the other passes
    m.add_rays(CAM, ends, np.array([True, True]))
    assert m.states(ends[:1])[0] == OCCUPIED


@pytest.mark.parametrize('reach, free_at, unknown_at', [
    (18.5, [3.0, 15.0], [19.5]),          # Gazebo: no return means nothing within the clip
    (2.0, [1.5], [3.0, 15.0]),            # a real stereo camera: only as far as trusted
])
def test_rays_without_a_return_clear_only_as_far_as_configured(reach, free_at, unknown_at):
    m = VoxelMap((0.0, 0.0), VoxelConfig(no_return_free=reach))
    d = _fan()
    m.add_rays(CAM, np.asarray(CAM) + 18.5 * d, np.zeros(len(d), bool))
    pts = lambda xs: np.array([(x, 0.0, 5.0) for x in xs])   # noqa: E731
    assert all(m.states(pts(free_at)) == FREE)
    assert all(m.states(pts(unknown_at)) == UNKNOWN)


def test_swept_volume_must_lie_in_free_voxels():
    m = VoxelMap((0.0, 0.0))
    m.assume_free(Ball(CAM, 2.0))                  # around the drone: seen before
    _wall_scan(m)
    assert m.first_blocked(CAM, (4.0, 0.0, 5.0), radius=1.0) is None
    # Towards the wall: blocked where the sphere first touches it, about 1 m short.
    d = m.first_blocked(CAM, (7.5, 0.0, 5.0), radius=1.0)
    assert d == pytest.approx(8.1 - 1.0, abs=0.5)
    # Up out of the cone: blocked at once, as nothing there was seen. 向上出了视场，立即受阻。
    assert m.first_blocked(CAM, (0.0, 0.0, 9.0), radius=1.0) < 1.0


def _first_touch(m, a, b, voxel, radius=1.5, half_height=0.75):
    """Return where the body moving a -> b first touches the voxel (by dense sampling), or None."""
    a, b = np.asarray(a), np.asarray(b)
    mid = m.origin + (np.floor((np.asarray(voxel) - m.origin) / m.res) + 0.5) * m.res
    if (mid - a) @ (b - a) / np.linalg.norm(b - a) < -m.res / 2:
        return None                                  # behind the drone: never entered
    t = np.linspace(0.0, 1.0, 4001)[:, None]
    gap = np.maximum(np.abs(a + t * (b - a) - mid) - m.res / 2, 0.0)
    touch = (np.linalg.norm(gap[:, :2], axis=1) < radius) & (gap[:, 2] < half_height)
    return float(t[np.argmax(touch), 0]) * float(np.linalg.norm(b - a)) if touch.any() else None


def _one_hit(voxel):
    """Return a small map, free all round CAM except one occupied voxel."""
    m = VoxelMap((0.0, 0.0), VoxelConfig(size=20.0, top=12.0))
    m.assume_free(Ball(CAM, 7.0))
    flat, _ = m._index(np.array([voxel]))
    m.hits.reshape(-1)[flat] = 1
    m.frees.reshape(-1)[flat] = 0
    return m


def test_a_voxel_touched_between_samples_still_blocks():
    """On a slanted move the body can brush a voxel's corner between two samples."""
    voxel, b = (2.875, 1.125, 5.125), (3.0, 0.35, 3.5)        # 26 deg down
    m = _one_hit(voxel)
    touch = _first_touch(m, CAM, b, voxel)
    assert touch == pytest.approx(1.684, abs=0.01)
    d = m.first_blocked(CAM, b, radius=1.5, half_height=0.75)
    assert d is not None and touch - 0.2 <= d <= touch


def test_the_swept_check_never_reports_a_contact_late():
    rng = np.random.default_rng(1)
    for _ in range(150):
        b = np.asarray(CAM) + (3.0, rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 2.0))
        voxel = np.asarray(CAM) + rng.uniform(0.2, 0.9) * (b - CAM) + rng.normal(0.0, 1.0, 3)
        m = _one_hit(voxel)
        touch = _first_touch(m, CAM, b, voxel)
        d = m.first_blocked(CAM, b, radius=1.5, half_height=0.75)
        if touch is not None:
            assert d is not None and d <= touch + 1e-9, (b, voxel, touch, d)


def test_something_behind_the_drone_does_not_block_moving_away():
    m = VoxelMap((0.0, 0.0))
    m.assume_free(Ball(CAM, 2.0))
    _wall_scan(m)
    behind = (-0.7, 0.0, 5.0)                       # within the margin, behind the drone
    m.add_rays((-3.0, 0.0, 5.0), np.array([behind]), np.array([True]))
    assert m.first_blocked(CAM, (4.0, 0.0, 5.0), radius=1.0) is None
    assert m.first_blocked(CAM, (-2.0, 0.0, 5.0), radius=1.0) == 0.0     # backing into it


def test_beside_and_above_the_drone_is_unseen_by_a_level_camera():
    """Without knowing its surroundings, even the first metre forward is blocked."""
    m = VoxelMap((0.0, 0.0))
    _wall_scan(m)
    assert m.first_blocked(CAM, (4.0, 0.0, 5.0), radius=1.0) == 0.0


def test_the_open_column_above_the_takeoff_point_is_free():
    m = VoxelMap((0.0, 0.0))
    m.assume_free(Column((0.0, 0.0), 2.0, 0.0, 7.0))
    assert m.first_blocked((0.0, 0.0, 0.5), (0.0, 0.0, 6.0), radius=1.0) is None
    assert m.first_blocked((0.0, 0.0, 0.5), (0.0, 0.0, 6.5), radius=1.0) is not None
    # A later hit still counts: something has come into the column. 之后的击中照样算数。
    m.add_rays((0.0, 0.0, 5.0), np.array([(0.5, 0.0, 5.0)]), np.array([True]))
    assert m.states(np.array([(0.5, 0.0, 5.0)]))[0] == OCCUPIED


def test_a_full_frame_is_quick():
    """One 160 x 120 frame (stride 4) to 18.5 m: fast enough to map every scan."""
    import time
    m = VoxelMap((0.0, 0.0))
    d = _fan(n=139)                                 # ~19000 rays
    t0 = time.perf_counter()
    m.add_rays(CAM, np.asarray(CAM) + 18.5 * d, np.zeros(len(d), bool))
    assert time.perf_counter() - t0 < 5.0
    assert m.scans == 1 and math.isclose(m.res, 0.25)
