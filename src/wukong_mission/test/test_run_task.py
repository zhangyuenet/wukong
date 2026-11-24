# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""What run_task writes into the report besides the photos."""

from types import SimpleNamespace

from wukong_mission.classify import Classification
from wukong_mission.run_task_node import understood_with_house
from wukong_mission.world import RoughHouse


def test_single_storey_map_reports_without_a_house_judgement():
    # GutterTask (a map with a roughly known house) keeps a RoughHouse in .house.
    # GutterTask（地图给出大致位置的房屋）的 .house 是 RoughHouse。
    task = SimpleNamespace(house=RoughHouse('house', (15.0, 0.0), 8.0, 7.0))
    assert understood_with_house('Check all gutters.', task) == 'Check all gutters.'
    assert understood_with_house('Check all gutters.', SimpleNamespace()) == 'Check all gutters.'


def test_house_judgement_is_added():
    c = Classification('detached', 2, 'attached', None, None, 'high', 'a garage on the left',
                       'model')
    text = understood_with_house('Check all gutters.', SimpleNamespace(house=c))
    assert text.startswith('Check all gutters. House: detached, 2 storey(s)')
    assert text.endswith('a garage on the left')
