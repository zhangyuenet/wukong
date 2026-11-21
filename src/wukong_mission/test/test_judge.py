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
