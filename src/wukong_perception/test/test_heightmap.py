# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Height-map perception: nearest building, eaves on several levels, any heading."""

import math

import numpy as np
import pytest

from test_geometry import synthetic_house  # noqa: I100 - sibling test module
from wukong_perception.heightmap import buildings, eave_lines, edge_cells, HeightMap


def _eaves(points, takeoff=(0.0, 0.0)):
    hm = HeightMap.from_points(points, res=0.2)
    blds = buildings(hm, takeoff)
    return blds, eave_lines(hm, edge_cells(hm, blds[0].cells))


def _drop_ground(p):
    return p[p[:, 2] > 0.05]


def test_single_gable_house_two_eaves():
    blds, eaves = _eaves(synthetic_house(step=0.1))
    assert len(blds) == 1
    names = sorted(e.name for e in eaves)
    assert names == ['north gutter', 'south gutter']
    south = next(e for e in eaves if e.name == 'south gutter')
    assert south.length == pytest.approx(11.0, abs=0.8)   # ends at the rakes
    assert south.height == pytest.approx(2.7, abs=0.25)


@pytest.mark.parametrize('bearing', [30.0, 125.0])
def test_rotated_house(bearing):
    pts = synthetic_house(center=(16, 6), bearing=bearing, length=13, depth=7, step=0.1)
    _, eaves = _eaves(pts)
    assert len(eaves) == 2
    for e in eaves:
        d = (e.side_bearing_deg - bearing) % 180
        assert min(d, 180 - d) == pytest.approx(90, abs=3)
        assert e.length == pytest.approx(14.0, abs=0.8)


def test_two_levels_and_the_nearest_building():
    main = synthetic_house(center=(15, 0), wall=5.8, step=0.1, seed=1)
    # A lower annex on the east side (its own gable roof, 3 m lower).
    annex = _drop_ground(synthetic_house(center=(15, 8.5), length=6, depth=6, wall=2.8,
                                         step=0.1, seed=2))
    far = _drop_ground(synthetic_house(center=(40, 30), step=0.2, seed=3))
    blds, eaves = _eaves(np.vstack([main, annex, far]))
    assert len(blds) == 2 and blds[0].distance < blds[1].distance
    upper = [e for e in eaves if e.name.startswith('upper')]
    lower = [e for e in eaves if e.name.startswith('lower')]
    assert len(upper) == 2 and len(lower) == 2
    assert all(e.height > 5 for e in upper) and all(e.height < 3 for e in lower)
    # Nothing from the far house. 不包括远处的房屋。
    assert all(math.dist(e.point(0.5), (40, 30)) > 15 for e in eaves)
