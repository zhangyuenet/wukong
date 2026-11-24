#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Record the vehicle's true pose in PX4 SITL, for sim/evaluate.py --poses.

    python3 sim/truth_pose.py OUT.jsonl     # until Ctrl+C / SIGTERM, or PX4 goes away

Simulation only. PX4's Gazebo bridge publishes the true pose
(vehicle_local_position_groundtruth, vehicle_attitude_groundtruth). This
polls both with PX4's listener a few times a second and appends one JSON line
per sample: wall time, north, east, height above the ground, heading, roll,
pitch. The ground-truth position is Gazebo's world position turned to NED
(north = world y, east = world x, down = -z), the frame the scenario's truth
uses. Photos are taken while the vehicle holds still, so a few samples a second
are enough to know where each one was really taken from.
仅限仿真：轮询 PX4 的真值话题，记录飞机的真实位姿，供 evaluate.py 核对照片实际拍到的范围。
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX4CMD = os.path.join(ROOT, 'scripts', 'px4cmd.sh')
_FIELD = re.compile(r'^\s*(x|y|z|q):\s*(.+?)\s*$')


def _listen(topic: str) -> dict:
    out = subprocess.run([PX4CMD, 'listener', topic, '-n', '1'], capture_output=True,
                         text=True, timeout=5).stdout
    fields = {}
    for line in out.splitlines():
        m = _FIELD.match(line)
        if m and m.group(1) not in fields:
            fields[m.group(1)] = m.group(2)
    return fields


def sample() -> dict:
    """One true pose, or raise if PX4 does not answer."""
    pos = _listen('vehicle_local_position_groundtruth')
    att = _listen('vehicle_attitude_groundtruth')
    q = [float(v) for v in re.match(r'\[([^\]]+)\]', att['q']).group(1).split(',')]
    w, x, y, z = q                      # Hamilton, body FRD -> NED (as px4_link)
    return {'t': time.time(), 'n': float(pos['x']), 'e': float(pos['y']),
            'alt': -float(pos['z']),
            'heading': math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
            'roll': math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            'pitch': math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))}


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.split('\n\n')[1])
        return 2
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    failing_since = None
    n = 0
    with open(argv[0], 'a', encoding='utf-8') as f:
        while not stop:
            try:
                f.write(json.dumps(sample()) + '\n')
                f.flush()
                n += 1
                failing_since = None
            except KeyboardInterrupt:
                break
            except Exception:   # noqa: B902 - PX4 not up yet, or gone
                failing_since = failing_since or time.time()
                if n and time.time() - failing_since > 10.0:
                    break        # PX4 went away after we had samples: done
            time.sleep(0.2)
    print(f'[truth_pose] {n} samples -> {argv[0]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
