# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Exploring: fly only the part of a route seen free, facing the way the drone goes."""

import math

import numpy as np
import pytest
from wukong_flight.skills import GoTo
from wukong_mission.explore import back_along, ExploreConfig, legs_facing_ahead, reach
from wukong_perception.voxels import Ball, VoxelMap

CAM = (0.0, 0.0, 5.0)


def _seen_north(wall_north=None, depth=18.5):
    """Return a map after one frame looking north from CAM, ending on a wall or nothing."""
    m = VoxelMap((0.0, 0.0))
    # Known round the drone, as above the takeoff point (the operator's open
    # column): a level camera cannot see its own surroundings. 飞机周围已知，如同
    # 起飞点上方操作员确认的竖直通道：水平的相机看不到自己周围。
    m.assume_free(Ball(CAM, 3.0))
    a = np.radians(np.linspace(-40, 40, 81))
    yaw, pitch = np.meshgrid(a, a)
    d = np.column_stack([np.cos(pitch.ravel()) * np.cos(yaw.ravel()),
                         np.cos(pitch.ravel()) * np.sin(yaw.ravel()), np.sin(pitch.ravel())])
    if wall_north is None:
        m.add_rays(CAM, np.asarray(CAM) + depth * d, np.zeros(len(d), bool))
    else:
        t = (wall_north - CAM[0]) / d[:, 0]
        m.add_rays(CAM, np.asarray(CAM) + t[:, None] * d, np.ones(len(d), bool))
    return m


def test_a_route_seen_free_all_the_way_is_flown_whole():
    m = _seen_north()
    route = [(5.0, 0.0, 5.0), (10.0, 0.0, 5.0)]
    assert reach(m, CAM, route) == (route, True)


def test_a_route_into_a_wall_stops_short_of_it():
    cfg = ExploreConfig(radius=1.5, stop_short=1.0)
    m = _seen_north(wall_north=9.0)
    part, whole = reach(m, CAM, [(14.0, 0.0, 5.0)], cfg)
    assert not whole and len(part) == 1
    # The sphere touches the wall 1.5 m short of it; 1 m before that. 球体在墙前 1.5 米处碰到墙。
    assert part[0][0] == pytest.approx(9.0 - 1.5 - 1.0, abs=0.4)
    assert part[0][1:] == (0.0, 5.0)


def test_the_unseen_leg_of_a_route_waits_for_a_look():
    m = _seen_north()
    part, whole = reach(m, CAM, [(6.0, 0.0, 5.0), (6.0, 12.0, 5.0)])   # then off to the east
    assert not whole
    assert part[0] == (6.0, 0.0, 5.0)
    assert len(part) in (1, 2)                     # perhaps a little way east, inside the cone


def test_nothing_to_fly_when_the_way_is_unseen_at_once():
    m = _seen_north()
    assert reach(m, CAM, [(0.0, -10.0, 5.0)]) == ([], False)      # due west, never looked at


def test_each_leg_is_flown_facing_ahead_after_turning_on_the_spot():
    legs = legs_facing_ahead(CAM, 0.0, [(6.0, 0.0, 5.0), (6.0, 8.0, 5.0)], face=(12.0, 8.0))
    assert all(isinstance(g, GoTo) for g in legs)
    north, turn, east, face = legs
    assert (north.north, north.east) == (6.0, 0.0) and north.look_at[1] == 0.0  # no turn first
    assert (turn.north, turn.east) == (6.0, 0.0)                               # on the spot
    assert turn.look_at[0] == pytest.approx(6.0) and turn.look_at[1] > 6.0     # now east
    assert (east.north, east.east) == (6.0, 8.0) and east.look_at[1] > 8.0
    assert (face.north, face.east, face.look_at) == (6.0, 8.0, (12.0, 8.0))


def test_a_climb_keeps_the_heading():
    legs = legs_facing_ahead(CAM, math.pi / 2, [(0.0, 0.0, 9.0)])
    assert legs == [GoTo(0.0, 0.0, 9.0)]


def test_the_way_back_is_the_trail_in_reverse_without_needless_stops():
    trail = [(0.0, 0.0, 6.0), (3.0, 0.0, 6.0), (6.0, 0.0, 6.0), (6.0, 4.0, 6.0), (9.0, 4.0, 5.0)]
    back = [(6.0, 4.0, 6.0), (6.0, 0.0, 6.0), (0.0, 0.0, 6.0)]
    assert back_along(trail, (9.0, 4.0, 5.0)) == back
    # From somewhere past the end of the trail, first back to its last point.
    assert back_along(trail[:3], (8.0, 0.0, 6.0)) == [(0.0, 0.0, 6.0)]
    assert back_along(trail[:3], (8.0, 1.0, 6.0)) == [(6.0, 0.0, 6.0), (0.0, 0.0, 6.0)]


def test_a_climb_can_be_checked_further_with_a_flat_body_than_a_ball():
    """With the real camera (29 deg up, 10 m) the top of a 1.5 m ball stays out of view."""
    from wukong_perception.voxels import VoxelConfig
    m = VoxelMap((0.0, 0.0), VoxelConfig(no_return_free=10.0))
    # The camera sees the sides of the 1.5 m body only from ~2 m ahead: up to
    # there the surroundings must already be known (~3.5 m round, e.g. the
    # operator's open column). 相机要到前方约 2 米处才看得到 1.5 米机体的两侧：在那之前
    # 周围必须已知（约 3.5 米，例如操作员确认的竖直通道）。
    m.assume_free(Ball(CAM, 3.5))
    h = np.radians(np.linspace(-36.5, 36.5, 74))
    v = np.radians(np.linspace(-29.0, 29.0, 59))
    yaw, pitch = np.meshgrid(h, v)
    d = np.column_stack([np.cos(pitch.ravel()) * np.cos(yaw.ravel()),
                         np.cos(pitch.ravel()) * np.sin(yaw.ravel()), np.sin(pitch.ravel())])
    m.add_rays(CAM, np.asarray(CAM) + 10.0 * d, np.zeros(len(d), bool))
    up = math.radians(20.0)
    route = [(12.0 * math.cos(up), 0.0, 5.0 + 12.0 * math.sin(up))]

    def gained(cfg):
        part, _ = reach(m, CAM, route, cfg)
        return math.dist(CAM, part[-1]) if part else 0.0
    flat = gained(ExploreConfig())
    ball = gained(ExploreConfig(half_height=1.5))
    # From a hover the flat body still gets only ~2 m: the top of what it sweeps
    # 3-4 m ahead is above the view from here. Climbs must be seen from further
    # back along the same line, i.e. while flying (depth frames along the leg).
    # 从悬停开始，扁圆柱也只能前进约 2 米：前方 3-4 米处扫过空间的顶部在这里看不到。
    # 爬升要从同一条线更靠后的地方看到，也就是边飞边看（沿途的深度帧）。
    assert flat >= 1.5 and ball == 0.0, (flat, ball)
