# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Height-map perception: nearest building, eaves on several levels, any heading."""

import math

import numpy as np
import pytest

from test_geometry import synthetic_house  # noqa: I100 - sibling test module
from wukong_perception.geometry import Eave
from wukong_perception.heightmap import (_extend, buildings, eave_lines, edge_cells,
                                         ground_height, HeightMap)


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


def test_an_eave_walks_out_to_its_corner_past_a_step_from_scans_not_lining_up():
    """
    Gazebo (clean): a gable end's last 0.9 m was 0.2 m further in; the eave stopped 1 m short.

    The west edge of a roof sloping up eastwards from 2.6 m: at east -8.0
    north of 8.8, one cell further in (-7.8) from 7.8 to 8.8, the corner.
    """
    res = 0.2
    north, east = np.arange(5.0, 22.0, res) + res / 2, np.arange(-10.0, -5.0, res) + res / 2
    nn, ee = np.meshgrid(north, east, indexing='ij')
    edge = np.where(nn >= 8.8, -8.1, -7.9)
    top = np.where((ee > edge) & (nn >= 7.8), 2.6 + 0.3 * (ee - edge), 0.0)
    hm = HeightMap((5.0, -10.0), res, top)
    eave = Eave('west gutter', (9.5, -8.05), (20.0, -8.05), 2.65, 270.0)
    out = _extend(hm, eave)
    assert min(out.start[0], out.end[0]) == pytest.approx(7.9, abs=0.25)
    assert max(out.start[0], out.end[0]) == pytest.approx(21.9, abs=0.25)   # the map's end


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


def test_ground_out_of_view_is_the_takeoff_ground():
    """Walls alone spread evenly over the heights: no spike, so the takeoff ground (0)."""
    walls = _drop_ground(synthetic_house(step=0.1))
    assert ground_height(walls[:, 2]) == 0.0
    assert ground_height(walls[:, 2], default=-0.2) == -0.2
    raised = synthetic_house(step=0.1) + np.array([0.0, 0.0, 0.4])   # ground in view, at 0.4
    assert ground_height(raised[:, 2]) == pytest.approx(0.4, abs=0.06)


def test_rays_below_the_roof_edge_stand_in_for_unseen_ground():
    """With no ground in view, eaves are found only where rays passed well below the roof."""
    # A few points far off: the grid reaches past the roof, as in flight, so the
    # cells outside the eaves are unseen rather than off the map.
    # 远处几个点，让栅格超出屋面（与飞行中一样），屋檐外的格子是“没看到”而不是“出界”。
    corners = [(0.0, -12.0, 0.0), (0.0, 12.0, 0.0), (30.0, -12.0, 0.0), (30.0, 12.0, 0.0)]
    pts = np.vstack([_drop_ground(synthetic_house(step=0.1)), corners])
    hm = HeightMap.from_points(pts, res=0.2)
    target = buildings(hm, (0.0, 0.0))[0]
    assert eave_lines(hm, edge_cells(hm, target.cells)) == []
    low = eave_lines(hm, edge_cells(hm, target.cells, clear_below=lambda n, e: 1.0))
    assert sorted(e.name for e in low) == ['north gutter', 'south gutter']
    # Rays only above the roof say nothing about where it ends. 只从屋面上方经过的射线不算。
    high = eave_lines(hm, edge_cells(hm, target.cells, clear_below=lambda n, e: 9.0))
    assert high == []
