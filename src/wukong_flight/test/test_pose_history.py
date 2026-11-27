# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pose history: the pose at a camera frame's stamp."""

import math

import pytest
from wukong_flight.pose_history import euler_from_quaternion, PoseHistory

# PX4 time = sent stamp + offset; the sent stamps are on the companion's clock.
OFFSET = -1_795_747_827_684_719            # us, as seen in SITL
T0 = 1_795_747_929_000_000                 # us, a sent stamp


def _yaw_q(yaw: float, pitch: float = 0.0):
    """Quaternion (w, x, y, z) of yaw then pitch (ZYX)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def _history(n=50, step_us=10_000, speed=2.0, yaw0=0.0, yaw_rate=0.0, t0=T0, offset=OFFSET):
    """Flying north at speed m/s and turning at yaw_rate rad/s, a sample every step_us."""
    h = PoseHistory()
    h.set_offset(offset, t0 - 1_000_000)
    for k in range(n):
        t = t0 + k * step_us
        h.add_position(t, speed * k * step_us * 1e-6, 0.0, -5.0)
        h.add_attitude(t, _yaw_q(yaw0 + yaw_rate * k * step_us * 1e-6))
    return h


def test_euler_from_quaternion():
    roll, pitch, yaw = euler_from_quaternion(_yaw_q(math.radians(30), math.radians(10)))
    assert roll == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(math.radians(10))
    assert yaw == pytest.approx(math.radians(30))


def test_a_gazebo_stamp_is_on_px4s_clock_and_is_interpolated():
    h = _history()
    # 125 ms after the first sample, stamped by Gazebo on the simulation clock.
    stamp = T0 + 125_000 + OFFSET
    x, y, z, yaw, roll, pitch = h.pose_at(stamp)
    assert x == pytest.approx(0.25)
    assert (y, z) == pytest.approx((0.0, -5.0))
    assert (yaw, roll, pitch) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_a_real_camera_stamp_is_on_the_companions_clock():
    h = _history()
    assert h.pose_at(T0 + 125_000)[0] == pytest.approx(0.25)
    assert h.pose_at(T0 + 3600e6) is None      # on neither clock
    assert 'neither clock' in h.why_not


def test_heading_is_interpolated_the_short_way_across_south():
    h = _history(yaw0=math.radians(175), yaw_rate=math.radians(100))   # through 180 deg
    yaw = h.pose_at(T0 + 105_000)[3]
    assert math.degrees(yaw) % 360 == pytest.approx(185.5)


def test_no_pose_outside_the_samples_or_across_a_gap():
    h = _history(n=10)
    assert h.pose_at(T0 - 1_000) is None
    assert h.pose_at(T0 + 95_000) is None      # after the last sample
    gappy = _history(n=10, step_us=200_000)              # 5 Hz: too far apart
    assert gappy.pose_at(T0 + 300_000) is None


def test_no_pose_just_after_the_offset_changes():
    h = _history()
    # An update while converging, sent at T0 + 200 ms: from then on the samples
    # are sent with the new offset. 收敛过程中的一次更新：此后的样本按新偏移发出。
    h.set_offset(OFFSET + 30_000, T0 + 200_000)
    assert h.pose_at(T0 + 220_000) is None                # less than 40 ms after
    assert 'offset changed' in h.why_not
    assert h.pose_at(T0 + 250_000) is not None
    h.set_offset(OFFSET + 30_500, T0 + 300_000)           # under 1 ms: no change
    assert h.pose_at(T0 + 310_000) is not None
    assert h.pose_at(T0 + 125_000) is None                # before the change


def test_a_jump_of_the_sent_clock_starts_afresh():
    h = _history(n=20)
    # PX4 reset its offset estimate: the sent stamps jump 9.8 s ahead.
    later = T0 + 200_000 + 9_800_000
    for k in range(10):
        h.add_position(later + k * 10_000, 10.0, 0.0, -5.0)
        h.add_attitude(later + k * 10_000, _yaw_q(0.0))
    assert h.pose_at(T0 + 100_000) is None      # before the jump: gone
    assert h.pose_at(later + 45_000)[0] == pytest.approx(10.0)
    # And back again: stamps earlier than the last one also start afresh.
    h.add_position(T0 + 500_000, 3.0, 0.0, -5.0)
    h.add_position(T0 + 510_000, 3.0, 0.0, -5.0)
    h.add_attitude(T0 + 500_000, _yaw_q(0.0))
    h.add_attitude(T0 + 510_000, _yaw_q(0.0))
    assert h.pose_at(T0 + 505_000)[0] == pytest.approx(3.0)
