"""Parser tests against real fixtures (first 90 cues of one file per convention era)."""

from pathlib import Path

import pytest

from survspk.subparse import detect_convention, parse_srt, parse_srt_text

FIX = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(not any(FIX.glob("*.srt")),
                                reason="real-subtitle fixtures stay on this machine (not in git)")


@pytest.fixture(scope="module")
def eras():
    return {name: parse_srt(FIX / f"{name}.srt") for name in
            ("s01_name_italic", "s20_gtgt", "s41_plain_dash_italic", "s47_name_dash")}


def test_all_fixtures_parse_90_cues(eras):
    for name, cues in eras.items():
        assert len(cues) == 90, name
        assert all(c.end_s > c.start_s for c in cues), name
        assert cues[0].idx == 0 and cues[-1].idx == 89


def test_s01_name_prefix_and_italics(eras):
    cues = eras["s01_name_italic"]
    conv = detect_convention(cues, name_min=5, marker_min=5)
    assert conv.n_name_prefix >= 5
    assert conv.has_italics
    # "RICHARD: <i>I've narrowed\nit down to four.</i>" -> name on line 0, both lines italic, tags stripped
    rich = next(c for c in cues if any(l.sdh_name == "RICHARD" for l in c.lines))
    assert rich.lines[0].sdh_name == "RICHARD"
    assert all(l.is_italic for l in rich.lines)
    assert "<" not in rich.text and rich.text.startswith("I've narrowed")


def test_s20_gtgt_markers_and_probst(eras):
    cues = eras["s20_gtgt"]
    conv = detect_convention(cues, name_min=1, marker_min=5)
    assert conv.has_gtgt and not conv.has_dash_turns
    probst = [c for c in cues if any(l.sdh_name == "PROBST" for l in c.lines)]
    assert probst, "PROBST: lines expected"
    line = next(l for l in probst[0].lines if l.sdh_name == "PROBST")
    assert line.is_turn  # ">> PROBST:" is both a turn and a name
    assert not line.text.startswith(">>")


def test_s41_plain_dashes_and_italics(eras):
    cues = eras["s41_plain_dash_italic"]
    conv = detect_convention(cues, name_min=5, marker_min=5)
    assert not conv.has_name_prefix
    assert conv.has_italics
    assert conv.n_dash > 0


def test_s47_two_speakers_in_one_cue(eras):
    cues = eras["s47_name_dash"]
    conv = detect_convention(cues, name_min=5, marker_min=5)
    assert conv.has_name_prefix and conv.has_dash_turns
    # cue 5: "-I went to Towson, bro.\n-Really?"
    c5 = cues[4]
    assert c5.n_turns == 2
    assert [l.text for l in c5.lines] == ["I went to Towson, bro.", "Really?"]
    # "(cheering)" is a sound cue, not speech
    c2 = cues[1]
    assert c2.lines[0].is_sound and not c2.is_speech
    # "TK:" alone on a line, text on the next line
    tk = next(c for c in cues if any(l.sdh_name == "TK" for l in c.lines))
    assert tk.lines[0].sdh_name == "TK" and tk.lines[0].text == ""
    assert tk.lines[1].text.startswith('"I caught you')
    assert conv.names["CAROLINE"] >= 1


def test_synthetic_edge_cases():
    txt = """1
00:00:01,000 --> 00:00:02,000
<i>>> JEFF PROBST: Come on in, guys!</i>

2
00:00:02,500 --> 00:00:03,000
-(gasps)
-MAN #2: What?

3
00:00:03.100 --> 00:00:04.000
♪ ♪

00:00:05,000 --> 00:00:06,000
No index line here.
"""
    cues = parse_srt_text(txt)
    assert len(cues) == 4
    l = cues[0].lines[0]
    assert l.sdh_name == "JEFF PROBST" and l.is_turn and l.is_italic and l.text == "Come on in, guys!"
    assert cues[1].lines[0].is_sound and cues[1].lines[0].is_turn
    assert cues[1].lines[1].sdh_name == "MAN #2" and cues[1].lines[1].text == "What?"
    assert not cues[2].is_speech
    assert cues[3].text == "No index line here." and cues[3].start_s == 5.0


def test_tolerates_crlf_and_bom():
    txt = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello.\r\n\r\n"
    cues = parse_srt_text(txt)
    assert len(cues) == 1 and cues[0].text == "Hello."
