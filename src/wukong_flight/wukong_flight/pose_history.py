# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Where the vehicle was at a given moment: its recent poses, by time.

A depth frame taken in flight must be converted with the pose at the moment
it was imaged, not the pose when it arrived. 飞行中的深度帧要用成像时刻的位姿转换，
不能用收到时的位姿。

Clocks. PX4 stamps its samples with its own clock. uXRCE-DDS sends them
converted to the companion computer's clock, by an offset it keeps
re-estimating (timesync_status.estimated_offset: PX4 time = sent stamp +
offset). A real camera stamps its frames with the companion's clock, so they
compare directly with the sent stamps. Gazebo stamps them with the simulation
clock, which PX4 runs on in SITL, so they are converted with the offset first.
Which of the two a frame uses is told from how near its stamp is to each.
When the simulation runs slower than real time, the offset drifts, and every
10 s or so PX4 resets its estimate: the sent stamps jump by seconds, then
move by tens of milliseconds at each update for a few seconds. A sample is
sent with the offset of the moment it is sent, and each timesync_status
says, by its own stamp, from when its offset applies. So a frame is used
only if it was imaged far enough after the offset last changed that the
samples around it were all sent with the offset it is converted with.
时钟：PX4 用自己的时钟打时间戳；uXRCE-DDS 发出时按它不断重新估计的偏移换成机载
电脑的时钟（PX4 时间 = 发出的时间戳 + 偏移）。真实相机用机载电脑的时钟，可以直接比较；
Gazebo 用仿真时钟，也就是 SITL 中 PX4 的时钟，要先按偏移换算。按时间戳离哪个更近
判断用的是哪一个。仿真慢于实时时偏移会漂移，PX4 大约每 10 秒重置一次估计：发出的
时间戳跳几秒，之后几秒内每次更新再动几十毫秒。样本按发出时的偏移换算，每条
timesync_status 用它自己的时间戳说明新偏移从何时起生效；所以只用成像时刻离上次
偏移变化足够远的帧，这样它前后的样本都是用同一个偏移换算的。
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoseHistoryConfig:
    keep: float = 3.0          # s of samples kept
    max_gap: float = 0.1       # s: the samples either side of a moment at most this far apart
    settle: float = 0.04       # s after the offset changes before frames are used again
    same_clock: float = 5.0    # s: a stamp this near to one clock is on that clock
    jump: float = 0.5          # s: a gap between samples longer than this starts afresh


# x, y, z (PX4 local NED, m), heading, roll, pitch (rad)
Pose6 = Tuple[float, float, float, float, float, float]


def euler_from_quaternion(q: Sequence[float]) -> Tuple[float, float, float]:
    """Return (roll, pitch, yaw) of a Hamilton quaternion (w, x, y, z), body FRD to NED."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _angle_between(a: float, b: float, f: float) -> float:
    """Return the angle a fraction f of the way from a to b, the short way round."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + f * d


class _Series:
    """Samples of one stream, (sent stamp in us, values), in time order and unbroken."""

    def __init__(self, cfg: PoseHistoryConfig) -> None:
        self.cfg = cfg
        self.samples: Deque[Tuple[int, Tuple[float, ...]]] = deque()

    def add(self, stamp_us: int, values: Tuple[float, ...]) -> None:
        if self.samples:
            last = self.samples[-1][0]
            if stamp_us == last:
                return
            if stamp_us < last or stamp_us - last > self.cfg.jump * 1e6:
                # The clock jumped (a new offset): what is kept is on another clock.
                # 时钟跳变（新的偏移）：已有的样本在另一个时钟上。
                self.samples.clear()
        self.samples.append((stamp_us, values))
        while self.samples and stamp_us - self.samples[0][0] > self.cfg.keep * 1e6:
            self.samples.popleft()

    def latest(self) -> Optional[int]:
        return self.samples[-1][0] if self.samples else None

    def at(self, t_us: float) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...], float]]:
        """Return the samples either side of t_us and how far between them it is, or None."""
        s = self.samples
        if not s or t_us < s[0][0] or t_us > s[-1][0]:
            return None
        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s[mid][0] <= t_us:
                lo = mid
            else:
                hi = mid
        (ta, a), (tb, b) = s[lo], s[hi]
        if tb - ta > self.cfg.max_gap * 1e6:
            return None
        f = 0.0 if tb == ta else (t_us - ta) / (tb - ta)
        return a, b, f


class PoseHistory:
    """
    The vehicle's positions and attitudes of the last few seconds, by time.

    Fed with PX4's samples as sent (timestamp_sample, on the companion's clock
    while time sync is on) and with the offsets PX4 reports; asked for the pose
    at a frame's stamp. Not thread-safe: feed and ask from one thread.
    输入 PX4 发出的样本和它报告的偏移，按帧的时间戳查询位姿。不是线程安全的，
    请在同一个线程里输入和查询。
    """

    def __init__(self, cfg: PoseHistoryConfig = PoseHistoryConfig()) -> None:
        self.cfg = cfg
        self.offset_us: int = 0
        # Sent stamp from which the offset applies. 偏移生效时刻（发出时钟）。
        self._since: Optional[int] = None
        self.why_not = ''        # why the last pose_at returned None
        self._positions = _Series(cfg)
        self._attitudes = _Series(cfg)

    def set_offset(self, offset_us: int, stamp_us: int) -> None:
        """
        Take PX4's timesync estimated_offset (us), sent with stamp_us.

        A change of more than 1 ms applies from stamp_us on. 变化超过 1 毫秒时，
        从 stamp_us 起生效。
        """
        if self._since is None or abs(offset_us - self.offset_us) > 1000:
            self._since = int(stamp_us)
        self.offset_us = int(offset_us)

    def add_position(self, stamp_us: int, x: float, y: float, z: float) -> None:
        self._positions.add(int(stamp_us), (float(x), float(y), float(z)))

    def add_attitude(self, stamp_us: int, q: Sequence[float]) -> None:
        self._attitudes.add(int(stamp_us), euler_from_quaternion(q))

    def to_sent_clock(self, stamp_us: float) -> Optional[float]:
        """
        Return a frame's stamp on the clock the samples were sent with, or None.

        None when it is near neither clock (no samples yet, or the clocks are
        not what this class expects). 两个时钟都不接近时返回 None。
        """
        latest = self._positions.latest()
        if latest is None:
            return None
        near = self.cfg.same_clock * 1e6
        if abs(stamp_us - latest) < near:           # the companion's clock (a real camera)
            return float(stamp_us)
        if abs(stamp_us - (latest + self.offset_us)) < near:   # PX4's clock (Gazebo)
            return float(stamp_us - self.offset_us)
        return None

    def pose_at(self, stamp_us: float) -> Optional[Pose6]:
        """
        Return the pose at a frame's stamp (us), or None if it cannot be told well.

        None (and why_not says why): the stamp is on neither clock, the offset
        changed less than cfg.settle before it, or there are no samples less
        than cfg.max_gap apart either side of it. 返回 None 的情况（why_not 说明
        原因）：时间戳不在任一时钟上、成像前不到 settle 秒偏移刚变过、或者前后没有间隔
        小于 max_gap 的样本。
        """
        t = self.to_sent_clock(stamp_us)
        if t is None:
            self.why_not = 'its stamp is on neither clock'
            return None
        if self._since is not None and t < self._since + self.cfg.settle * 1e6:
            self.why_not = 'imaged as the clock offset changed'
            return None
        p, a = self._positions.at(t), self._attitudes.at(t)
        if p is None or a is None:
            self.why_not = 'no pose samples close either side'
            return None
        (p0, p1, fp), (a0, a1, fa) = p, a
        x, y, z = (u + fp * (v - u) for u, v in zip(p0, p1))
        roll, pitch, yaw = (_angle_between(u, v, fa) for u, v in zip(a0, a1))
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        return x, y, z, yaw, roll, pitch
