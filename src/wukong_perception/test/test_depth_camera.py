# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""DepthCamera without ROS running: scans, frames in flight, and what goes to the voxel map."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
from wukong_perception import depth_camera as dc

STILL = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0)


class _Node:
    """Stands in for a ROS node: keeps nothing, starts nothing."""

    def __init__(self):
        self.warnings = []

    def create_subscription(self, *args):
        return None

    def create_timer(self, *args):
        return None

    def get_logger(self):
        return self

    def warning(self, text):
        self.warnings.append(text)


def _image(stamp_s=100.0, depth=5.0):
    """Return a small 32FC1 frame of a wall depth metres ahead."""
    msg = Image()
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(stamp_s), int(stamp_s % 1 * 1e9)
    msg.height, msg.width, msg.encoding, msg.step = 48, 64, '32FC1', 64 * 4
    msg.data = np.full((48, 64), depth, np.float32).tobytes()
    return msg


def _camera(pose_at, monkeypatch, now):
    """Return a DepthCamera with intrinsics, on a clock the test moves (now[0], s)."""
    monkeypatch.setattr(dc.time, 'monotonic', lambda: now[0])
    node = _Node()
    cam = dc.DepthCamera(node, pose_at, 'px4')
    info = CameraInfo()
    info.k = [40.0, 0.0, 32.0, 0.0, 40.0, 24.0, 0.0, 0.0, 1.0]
    cam._on_info(info)
    return cam, node


def test_a_scan_with_the_pose_at_its_moment_is_evidence_of_free_space(monkeypatch):
    now = [10.0]
    cam, _ = _camera(lambda stamp, clock: STILL, monkeypatch, now)
    done = []
    cam.capture('station 1', (9.0, 9.0, 9.0, 0.0), done.append)    # the request's pose unused
    cam._on_image(_image())
    now[0] += dc.FLIGHT_DELAY + 0.01
    cam._drain()
    assert done == [True] and cam.poses == [STILL] and len(cam.rays) == 1
    frames = cam.take_frames()
    assert [f[3] for f in frames] == [True]
    assert cam.take_frames() == []          # taken frames are let go 取走的帧不再保留


def test_a_scan_never_still_and_without_a_pose_shapes_the_house_but_frees_nothing(monkeypatch):
    # The mission scanned anyway (never still), and PX4's clock offset kept
    # changing: no frame has a pose at its moment. 任务照样扫描（一直不够静止），
    # PX4 的时钟偏移一直在变：没有一帧有成像时刻的位姿。
    now = [10.0]
    cam, node = _camera(lambda stamp, clock: 'imaged as the clock offset changed',
                        monkeypatch, now)
    done = []
    request = (1.0, 2.0, 5.0, 0.5)
    cam.capture('station 1', request, done.append)
    while not done:
        cam._on_image(_image())
        now[0] += 0.2
        cam._drain()
        assert now[0] < 10.0 + dc.SCAN_POSE_WAIT + 1.0
    assert now[0] > 10.0 + dc.SCAN_POSE_WAIT                 # it waited for one first
    assert done == [True] and cam.poses == [request] and len(cam.clouds) == 1
    assert cam.rays == [] and cam.at_request == 1          # no free-space evidence
    assert [f[3] for f in cam.take_frames()] == [False]    # hits only, in the voxel map
    assert 'not as free space' in node.warnings[0]
    assert cam.dropped['imaged as the clock offset changed'] > 0


def test_frames_in_flight_wait_in_a_bounded_backlog(monkeypatch):
    now = [10.0]
    north = [0.0]

    def moving(stamp, clock):
        north[0] += 2.0                    # a new viewpoint every frame
        return (north[0], 0.0, 5.0, 0.0, 0.0, 0.0)
    cam, _ = _camera(moving, monkeypatch, now)
    for _ in range(dc.MAP_BACKLOG + 5):
        cam._on_image(_image())
        now[0] += dc.FLIGHT_DELAY + 0.01
        cam._drain()
    assert cam.backlog_dropped == 5
    frames = cam.take_frames()
    assert len(frames) == dc.MAP_BACKLOG and all(f[3] for f in frames)
    assert frames[0][0][0] == pytest.approx(cam.flight_poses[5][1], abs=0.5)   # the oldest went
    assert 'never mapped (backlog full)' in cam.describe_flight()
