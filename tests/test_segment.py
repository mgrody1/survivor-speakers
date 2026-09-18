"""Segmentation logic on the real S47 / S20 / S01 fixtures (no aligned words -> proportional timing)."""

from pathlib import Path

from survspk.config import SegmentCfg
from survspk.stage_segment import (Turn, Utt, build_utterances, can_merge, cue_to_turns, merge_turns,
                                   propagate_names)
from survspk.subparse import parse_srt

FIX = Path(__file__).parent / "fixtures"
CFG = SegmentCfg()
IDENT = lambda t: t  # noqa: E731


def _cues(name):
    out = []
    for c in parse_srt(FIX / f"{name}.srt"):
        d = c.to_row()
        d["cue_id"] = f"X_C{c.idx:04d}"
        out.append(d)
    return out


def test_shared_cue_is_split_into_two_turns():
    cues = _cues("s47_name_dash")
    c5 = cues[4]                       # "-I went to Towson, bro." / "-Really?"
    turns = cue_to_turns(c5, {}, IDENT)
    assert [t.text for t in turns] == ["I went to Towson, bro.", "Really?"]
    assert all(t.has_marker and t.shared_cue and not t.from_words for t in turns)
    assert turns[0].start == c5["start_s"] and abs(turns[1].end - c5["end_s"]) < 1e-6
    assert turns[0].end == turns[1].start
    # proportional split: the longer line gets more time
    assert (turns[0].end - turns[0].start) > (turns[1].end - turns[1].start)


def test_name_on_its_own_line_attaches_to_following_text():
    cues = _cues("s47_name_dash")
    tk = next(c for c in cues if any(l["sdh_name"] == "TK" for l in c["lines"]))
    turns = cue_to_turns(tk, {}, IDENT)
    assert len(turns) == 1 and turns[0].sdh_name == "TK" and turns[0].text.startswith('"I caught you')


def test_sound_only_cue_yields_no_turn():
    cues = _cues("s47_name_dash")
    assert cue_to_turns(cues[1], {}, IDENT) == []      # "(cheering)"


def test_word_timing_overrides_proportional():
    cues = _cues("s47_name_dash")
    c5 = cues[4]
    words = {0: [{"start_s": 13.10, "end_s": 13.90}], 1: [{"start_s": 14.00, "end_s": 14.20}]}
    turns = cue_to_turns(c5, words, IDENT)
    assert turns[0].from_words and (turns[0].start, turns[0].end) == (13.10, 13.90)
    assert (turns[1].start, turns[1].end) == (14.00, 14.20)


def _t(text, start, end, name=None, marker=False):
    return Turn("c", 0, [0], text, name, marker, False, start, end, len(text.split()), True, False)


def test_propagation_rules():
    ts = [_t("Hey.", 0, 1, name="TEENY"), _t("what's up", 1.2, 2), _t("nothing", 2.3, 3, marker=True),
          _t("still nothing", 3.1, 4), _t("later", 10, 11)]
    propagate_names(ts, CFG.sdh_propagate_gap_s)
    assert [t.name for t in ts] == ["TEENY", "TEENY", None, None, None]
    assert ts[1].name_inherited and not ts[0].name_inherited


def test_merge_rules():
    # sentence continues across cues -> merge even without names
    a = _t("Cassidy's", 0, 1); b = _t("gonna be first through.", 1.1, 2.5)
    assert can_merge(Utt(0, a.start, a.end, a.text, [a]), b, CFG)
    # sentence ended, no names -> do not merge (could be a new speaker in a marker-less file)
    c = _t("I know.", 2.6, 3.2)
    u = Utt(0, 0, 2.5, "Cassidy's gonna be first through.", [a, b])
    assert not can_merge(u, c, CFG)
    # sentence ended but both carry the same name -> merge
    u.name = "SAM"; c.name = "SAM"
    assert can_merge(u, c, CFG)
    # a marker always breaks
    d = _t("Yeah.", 3.3, 3.8, marker=True); d.name = "SAM"
    assert not can_merge(u, d, CFG)
    # gap too large breaks
    e = _t("and then", 5.0, 6.0); e.name = "SAM"
    assert not can_merge(u, e, CFG)
    # length cap
    f = _t("and then", 2.6, 14.0); f.name = "SAM"
    assert not can_merge(u, f, CFG)


def test_merge_turns_end_to_end():
    ts = [_t("So, I'm a hunter,", 0, 1), _t("and I know", 1.1, 2), _t("in hunting game animals,", 2.1, 3),
          _t("you wait.", 3.1, 4), _t("Me?", 4.2, 4.6, marker=True), _t("Like, I...", 4.7, 5.2)]
    propagate_names(ts, CFG.sdh_propagate_gap_s)
    utts = merge_turns(ts, CFG)
    assert [u.text for u in utts] == ["So, I'm a hunter, and I know in hunting game animals, you wait.", "Me? Like, I..."][:1] + \
           ["Me?", "Like, I..."]
    assert len(utts[0].turns) == 4


def test_build_on_s47_fixture_detects_recap_and_splits():
    cues = _cues("s47_name_dash")
    utts, stats = build_utterances(cues, {}, IDENT, CFG, None, "US47", 2, total_s=3866)
    assert stats["n_turns"] > stats["n_cues"]                  # shared cues were split
    assert stats["n_shared_cue_turns"] >= 10
    assert stats["n_utts"] < stats["n_turns"]                  # and continuations merged
    # recap ends with "The tribe has spoken." at ~78-82 s (verified against the file); the 5.5 s music beat
    # at 37 s must not end it
    assert stats["recap_end_s"] is not None and 75 < stats["recap_end_s"] < 90, stats["recap_end_s"]
    recap = [u for u in utts if u.segment == "recap"]
    assert recap and all(u.start < stats["recap_end_s"] for u in recap)
    assert any(u.name == "CAROLINE" for u in recap)
    # names propagate; the fixture is mostly dash-turn camp dialogue, so only a minority carry one here
    assert stats["n_utts_named"] >= 0.15 * stats["n_utts"]
    assert {"TK", "CAROLINE", "TEENY"} <= {u.name for u in utts}


def test_build_on_s20_fixture_gtgt_era():
    cues = _cues("s20_gtgt")
    utts, stats = build_utterances(cues, {}, IDENT, CFG, None, "US20", 2, total_s=2552)
    probst = [u for u in utts if u.name in ("PROBST", "JEFF PROBST")]
    assert probst
    # a '>>' turn without a name must not inherit PROBST
    assert all(u.name is None or u.name in ("PROBST", "JEFF PROBST") for u in utts)
    assert any(u.name is None and u.turns[0].has_marker for u in utts)
    assert stats["n_utts"] < stats["n_turns"]


def test_runs_and_domain_hint_on_s47_fixture():
    cues = _cues("s47_name_dash")
    utts, stats = build_utterances(cues, {}, IDENT, CFG, None, "US47", 2, total_s=3866)
    assert stats["n_runs"] > 0 and stats["n_confessional_runs"] >= 1
    conf = [u for u in utts if u.domain_hint == "confessional"]
    assert conf and all(u.run_dur >= CFG.confessional_run_s for u in conf)
    # a run is one speaker: members never carry two different names
    for rid in {u.run_id for u in utts}:
        assert len({u.name for u in utts if u.run_id == rid and u.name}) <= 1
    assert all(u.domain_hint in ("confessional", "field") for u in utts)


def test_host_question_hands_name_to_the_answer():
    hosts = frozenset({"PROBST", "JEFF", "JEFF PROBST"})
    cast = {"TIYANA", "GABE", "SUE"}
    addr = lambda tok: tok.upper() if tok.upper() in cast else None  # noqa: E731
    # question named PROBST, answer unnamed and unmarked (the US47E02 55:37 case), then Jeff again explicitly
    ts = [_t("Tiyana, do you agree with that?", 0, 2, name="PROBST"), _t("I honestly do agree.", 2.3, 4),
          _t("It seemed like they vibed really well.", 4.1, 6), _t("So, Gabe?", 6.5, 7, name="PROBST"),
          _t("Yeah.", 7.2, 7.6, marker=True)]
    propagate_names(ts, CFG.sdh_propagate_gap_s, hosts, addr)
    assert [t.name for t in ts] == ["PROBST", "TIYANA", "TIYANA", "PROBST", None]
    assert ts[1].name_inherited
    # and the answer no longer merges into the host's utterance
    utts = merge_turns(ts, CFG)
    assert [u.name for u in utts][:2] == ["PROBST", "TIYANA"]


def test_host_statement_without_question_does_not_hand_over():
    hosts = frozenset({"PROBST"})
    addr = lambda tok: tok.upper() if tok.upper() == "GABE" else None  # noqa: E731
    ts = [_t("Gabe, your enthusiasm for that is the hallmark of a new era player.", 0, 3, name="PROBST"),
          _t("You all tend to wear your hearts on your sleeves.", 3.1, 5)]
    propagate_names(ts, CFG.sdh_propagate_gap_s, hosts, addr)
    assert [t.name for t in ts] == ["PROBST", "PROBST"]
    # a castaway addressing another castaway is not a hand-over either
    ts = [_t("Sam, Sam, if I pull it any further?", 0, 2, name="ANIKA"), _t("Keep going.", 2.1, 3)]
    propagate_names(ts, CFG.sdh_propagate_gap_s, hosts, lambda tok: "SAM")
    assert [t.name for t in ts] == ["ANIKA", "ANIKA"]
    # unknown addressee (not in cast) -> no hand-over
    ts = [_t("Buddy, you ready?", 0, 2, name="PROBST"), _t("Yes.", 2.1, 3)]
    propagate_names(ts, CFG.sdh_propagate_gap_s, hosts, lambda tok: None)
    assert [t.name for t in ts] == ["PROBST", "PROBST"]


def test_propagation_without_host_info_is_unchanged():
    ts = [_t("Tiyana, do you agree with that?", 0, 2, name="PROBST"), _t("I honestly do agree.", 2.3, 4)]
    propagate_names(ts, CFG.sdh_propagate_gap_s)
    assert [t.name for t in ts] == ["PROBST", "PROBST"]
