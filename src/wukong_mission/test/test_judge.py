# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Judge and report tests with a fake client."""

import base64
import io
import json
from types import SimpleNamespace

from PIL import Image

from wukong_mission.judge import ClaudeJudge, PhotoVerdict, Verdict
from wukong_mission.planner import Plan
from wukong_mission.report import summary_line, write_report


class FakeClient:
    def __init__(self, answer=None, stop_reason='end_turn'):
        self.requests = []
        self.answer, self.stop_reason = answer, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.requests.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               parsed_output=self.answer)


def photo(tmp_path, size=(2000, 1500)):
    p = tmp_path / 'p.jpg'
    Image.new('RGB', size, (120, 80, 40)).save(p)
    return str(p)


def test_judge_sends_a_downscaled_jpeg_and_the_question(tmp_path):
    client = FakeClient(PhotoVerdict(needs_attention='yes', confidence='high',
                                     findings='Leaves in the gutter.', where='left'))
    v = ClaudeJudge(client=client).judge('check gutters', 'south gutter', 'Blocked?',
                                         photo(tmp_path))
    assert (v.needs_attention, v.confidence, v.where) == ('yes', 'high', 'left')
    image, text = client.requests[0]['messages'][0]['content']
    data = base64.b64decode(image['source']['data'])
    assert max(Image.open(io.BytesIO(data)).size) <= 1568
    assert 'south gutter' in text['text'] and 'Blocked?' in text['text']


def test_judge_failure_becomes_a_verdict_not_an_exception(tmp_path):
    v = ClaudeJudge(client=FakeClient(stop_reason='refusal')).judge(
        'x', 'south gutter', '', photo(tmp_path))
    assert v.needs_attention == 'error' and 'no judgement' in v.findings


def test_report(tmp_path):
    verdicts = [Verdict('south gutter', str(tmp_path / 'a.jpg'), 'yes', 'high', 'Leaves.'),
                Verdict('north gutter', str(tmp_path / 'b.jpg'), 'no', 'high', 'Clean.'),
                Verdict('east side', str(tmp_path / 'c.jpg'), 'error', 'low', 'no judgement')]
    assert summary_line(verdicts) == ('Needs attention: south gutter. '
                                      'Could not tell: east side.')
    md = write_report(str(tmp_path), 'check gutters', Plan('check', []), True, '', verdicts)
    text = open(md, encoding='utf-8').read()
    assert '![south gutter](a.jpg)' in text and 'needs attention' in text
    data = json.load(open(tmp_path / 'report.json', encoding='utf-8'))
    assert data['flight_ok'] and len(data['photos']) == 3


def test_summary_when_nothing_could_be_judged():
    v = [Verdict('south gutter', 'a.jpg', 'unclear', 'low', '(offline)')]
    assert summary_line(v) == 'No photo could be judged: south gutter.'
    assert summary_line([]) == 'No photos were taken.'


def test_segment_summary():
    def seg(g, i, a):
        extra = {'gutter': g, 'segment': i, 'segments': 4}
        return Verdict(f'{g} {i}', 'x.jpg', a, 'high', '', extra=extra)
    v = [seg('south gutter', 2, 'yes'), seg('south gutter', 4, 'yes'),
         seg('south gutter', 1, 'no'), seg('north gutter', 3, 'unclear')]
    assert summary_line(v) == ('Needs cleaning: south gutter segments 2, 4. '
                               'Could not tell: north gutter segment 3.')
    assert summary_line([seg('north gutter', 1, 'no')]) == 'No gutter segment needs cleaning.'
    assert summary_line([seg('north gutter', 1, 'unclear')]).startswith('No segment could')


def test_crop_box_covers_the_segment_from_the_viewpoint():
    from wukong_mission.gutter_task import crop_box
    from wukong_mission.inspection import PhotoTarget
    # South gutter segment of the basic house, photographed from 4.5 m south, facing north.
    t = PhotoTarget('s', 'south gutter', 2, 4, (10.4, -2.75), (10.4, 0.0), 2.7,
                    (5.9, -1.375, 3.3))
    pose = (5.9, -1.375, 3.3, 0.0, 0.0, 0.0)
    (left, top, right, bottom), (u0, u1) = crop_box(t, pose)
    assert 0 <= left < right <= 1920 and 0 <= top < bottom <= 1080
    assert 0 < u0 < u1 < right - left        # neighbours' strips on both sides
    # Segment of 2.75 m at 4.5 m: ~0.6 rad wide, ~860 px with f ~ 1404 px.
    assert 700 < u1 - u0 < 1000


def test_crop_box_allows_for_the_altitude_estimate():
    # The vehicle 0.45 m lower than it believes sees the gutter as if it were
    # 0.45 m higher: still inside the crop. 飞机实际比估计低 0.45 米，看到的水槽
    # 相当于高了 0.45 米，仍在裁剪框内。
    from wukong_mission.gutter_task import crop_box
    from wukong_mission.inspection import PhotoTarget
    from wukong_perception.geometry import project, RGB_X500_DEPTH
    t = PhotoTarget('s', 'south gutter', 2, 4, (10.4, -2.75), (10.4, 0.0), 2.7,
                    (5.9, -1.375, 3.6))
    pose = (5.9, -1.375, 3.6, 0.0, 0.0, 0.0)
    (left, top, right, bottom), _ = crop_box(t, pose)
    for dz in (-0.45, 0.0, 0.45):
        _, v, _ = project([[10.4, -1.375, 2.7 + dz]], RGB_X500_DEPTH, pose)[0]
        assert top <= v <= bottom


def test_report_lists_segments_without_photo(tmp_path):
    extra = {'gutter': 'north gutter', 'segment': 2, 'segments': 3}
    v = [Verdict('north gutter segment 2 of 3', '', 'skipped', 'low',
                 'not inspected: the flight ended before this photo', extra=extra)]
    md = write_report(str(tmp_path), 'check gutters', Plan('check', []), False,
                      'skill 9 timed out', v)
    text = open(md, encoding='utf-8').read()
    assert 'not inspected' in text and '![' not in text
    assert 'did not complete (skill 9 timed out)' in text
