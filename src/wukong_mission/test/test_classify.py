# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""House-kind judgement: the ruler, its readings, and which side faces the drone."""

import json

from PIL import Image
import pytest
from wukong_mission.classify import annotate, from_answer, front_of, HouseKind, TruthClassifier
from wukong_perception.heightmap import Rectangle


def test_front_is_the_side_facing_the_viewer_left_to_right():
    rect = Rectangle((16.0, 0.0), 30.0, 10.0, 90.0)          # a row east-west, 16 m north
    a, b = front_of(rect, (0.0, 0.0))
    assert a[0] == pytest.approx(11.0) and b[0] == pytest.approx(11.0)    # the south side
    assert a[1] < b[1]                       # looking north: west (left) first


def test_ruler_readings_become_ground_positions():
    front = ((11.0, -15.0), (11.0, 15.0))
    ans = HouseKind(kind='townhouse_row', storeys=2, garage='none', units=5, unit_from_m=12.0,
                    unit_to_m=18.5, end_unit=False, confidence='high', notes='')
    c = from_answer(ans, front)
    assert c.unit == ((11.0, -3.0), (11.0, 3.5))
    det = from_answer(HouseKind(kind='detached', storeys=2, garage='attached',
                                confidence='high', notes=''), front)
    assert det.unit is None and det.kind == 'detached'


def test_too_narrow_unit_is_ignored():
    ans = HouseKind(kind='townhouse_row', storeys=2, garage='none', unit_from_m=3.0,
                    unit_to_m=4.0, confidence='low', notes='')
    assert from_answer(ans, ((0, 0), (0, 20))).unit is None


def test_annotate_draws_on_the_photo(tmp_path):
    photo = tmp_path / 'p.jpg'
    Image.new('RGB', (1920, 1080), (128, 128, 128)).save(photo)
    out = tmp_path / 'ruler.jpg'
    # Drone at 6 m facing north; a 12 m front 11 m away, ruler at 2.5 m.
    pose = (0.0, 0.0, 6.0, 0.0, 0.0, 0.0)
    got = annotate(str(photo), ((11.0, -6.0), (11.0, 6.0)), 2.5, pose, str(out))
    assert got == str(out)
    img = Image.open(out)
    assert img.getpixel((960, 50))[0] > 200            # the red line of sight
    yellow = sum(1 for x in range(0, 1920, 4)
                 for y in range(540, 1080, 4) if img.getpixel((x, y))[2] < 80
                 and img.getpixel((x, y))[0] > 200 and img.getpixel((x, y))[1] > 180)
    assert yellow > 50                                  # the ruler, below the horizon


def test_offline_truth_of_an_m2_scenario_is_a_single_storey_house(tmp_path):
    """An M2 scenario's truth has one house, no buildings list: still judged, as detached."""
    (tmp_path / 'truth.json').write_text(json.dumps({'house': {'center': [15.0, 0.0]},
                                                     'gutters': []}))
    c = TruthClassifier.beside(str(tmp_path / 'map.yaml'))
    assert c is not None
    answer = c.classify_house('', (0.0, 0.0, 6.0, 0.0, 0.0, 0.0), ((0, 0), (1, 0)), 1.0, '')
    assert (answer.kind, answer.storeys, answer.confidence) == ('detached', 1, 'high')
    assert TruthClassifier.beside(str(tmp_path / 'nowhere' / 'map.yaml')) is None
