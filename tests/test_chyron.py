"""Chyron OCR stage on the synthetic season with a fake OCR backend: matching, location cards, dedupe, anchoring,
label writing (human wins, explicit SDH disagreement queues), SDH agreement, and the bank's chyron_run labels."""

import json

import numpy as np
import pandas as pd

from survspk.chyron import (CastMatcher, ChyronCfg, Hit, OcrLine, anchor_utterance, chyron_episode, dedupe,
                            frame_hits, parse_location)
from survspk.stage_bank import collect_labelled
from tests.test_bank_assign import season  # noqa: F401
from tests.test_text_prior import R


def _frame(level: int) -> np.ndarray:
    return np.full((40, 200), level, dtype=np.uint8)


def _cfg():
    return ChyronCfg(backend="fake", dedupe_s=10.0)


def test_matcher_names_occupation_and_host(season):
    s, con, sea = season
    r = R(s)
    m = CastMatcher(r, "US99", 2, _cfg())
    assert m.match([OcrLine("ANDY  AI RESEARCH ASSISTANT", 0.9, 0.05, 0.6)])[0] == "S_A"
    assert m.match([OcrLine("ANDV", 0.7, 0.05, 0.6)])[0] == "S_A"                       # one OCR slip on a 4-letter name
    assert m.match([OcrLine("PROBST", 0.9, 0.05, 0.6)])[0] == "HOST_US"
    assert m.match([OcrLine("BUY MORE CEREAL", 0.9, 0.05, 0.6)]) is None
    # centred burned-in captions are ignored by position
    assert frame_hits([OcrLine("ANDY", 0.9, 0.7, 0.3)], m, _cfg()) == (None, None)
    hit, scene = frame_hits([OcrLine("MATSING TRIBE", 0.9, 0.05, 0.5), OcrLine("DAY 11", 0.9, 0.05, 0.8)], m, _cfg())
    assert hit is None and scene.tribe == "MATSING" and scene.day == 11 and scene.kind == "day"
    assert parse_location([OcrLine("VILLAINS / NIGHT 3", 0.9, 0.0, 0.0)]).kind == "night"


def test_dedupe_and_anchor():
    hits = [Hit(100.0, "S_A", "ALICE", 0.9, 90), Hit(103.0, "S_A", "ALICE", 0.95, 100), Hit(130.0, "S_A", "ALICE", 0.9, 90),
            Hit(101.0, "S_B", "BOB", 0.9, 95)]
    d = dedupe(hits, 10.0)
    assert [(h.t_s, h.castaway_id, h.match_score) for h in d] == [(100.0, "S_A", 100), (101.0, "S_B", 95), (130.0, "S_A", 90)]
    utts = [{"utt_id": "u1", "start_s": 90.0, "segment": "body"}, {"utt_id": "u2", "start_s": 97.5, "segment": "body"},
            {"utt_id": "u3", "start_s": 99.0, "segment": "body"}, {"utt_id": "u4", "start_s": 100.5, "segment": "recap"}]
    start = ChyronCfg(backend="fake", anchor="start")
    assert anchor_utterance(utts, 100.0, start)["utt_id"] == "u2"      # ~3 s before the card beats 1 s before
    assert anchor_utterance(utts, 120.0, start) is None
    # overlap (the default): the line most on air while the card is up, whenever it started
    on_air = [{"utt_id": "a", "start_s": 88.0, "end_s": 101.0, "segment": "body"},      # long line under the card
              {"utt_id": "b", "start_s": 101.2, "end_s": 104.0, "segment": "body"}]
    assert anchor_utterance(on_air, 99.0, _cfg(), t_end=102.0)["utt_id"] == "a"
    assert anchor_utterance(on_air, 101.5, _cfg(), t_end=104.0)["utt_id"] == "b"
    assert anchor_utterance(utts, 100.0, _cfg())["utt_id"] == "u2"     # no end times: falls back to the start rule


def test_chyron_episode_end_to_end(season):
    s, con, sea = season
    r = R(s)
    # E2: A (explicit name), D (unnamed, will get its chyron), C mislabelled as A (explicit) -> chyron C conflicts
    e2 = [
        {"spk": "S_A", "label": "S_A", "n": 2, "dur_each": 4.0},
        {"spk": "S_D", "label": None, "n": 3, "dur_each": 5.0, "text": "I'm 59 and I own a flight school"},
        {"spk": "S_C", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "WRONGNAME"},
        {"spk": "S_B", "label": "S_B", "n": 2, "dur_each": 4.0},
    ]
    ids = sea.episode(2, e2)
    st = {u["utt_id"]: u["start_s"] for u in con.execute("SELECT utt_id, start_s FROM utterances WHERE episode=2")}
    con.execute("UPDATE episodes SET recap_end_s=0 WHERE 1=0")
    con.execute("INSERT OR REPLACE INTO episodes (version_season, episode, duration_s, video_path) VALUES ('US99', 2, 400, '/nope.mkv')")
    # a human label on B's first line must survive
    con.execute("INSERT INTO labels (utt_id, speaker_id, source, confidence) VALUES (?,?,?,1.0)", (ids[3][0], "S_B", "human"))
    con.commit()
    # fake frames: the level is the key; each card shows ~3 s after the anchored line starts, held for two frames
    fake = {
        10: [OcrLine("ANDY  AI RESEARCH ASSISTANT", 0.9, 0.05, 0.6), OcrLine("KOTA TRIBE", 0.8, 0.05, 0.85)],
        20: [OcrLine("SUE", 0.9, 0.05, 0.6)],
        30: [OcrLine("SAM", 0.9, 0.05, 0.6)],
        40: [OcrLine("GABE", 0.9, 0.05, 0.6)],
        50: [OcrLine("KOTA TRIBE", 0.9, 0.05, 0.5), OcrLine("DAY 4", 0.9, 0.05, 0.8)],
        60: [OcrLine("that is a lot of rice", 0.9, 0.75, 0.5)],        # centred caption
    }
    frames = [(st[ids[0][0]] + 3.0, _frame(10)), (st[ids[0][0]] + 3.5, _frame(10)),
              (st[ids[1][0]] + 3.0, _frame(20)), (st[ids[2][0]] + 2.5, _frame(30)),
              (st[ids[3][0]] + 3.0, _frame(40)), (st[ids[1][0]] - 8.0, _frame(50)), (st[ids[2][0]] + 3.5, _frame(60))]
    from survspk.chyron import ocr_backend
    stats = chyron_episode(s, con, "US99", 2, r, frames=iter(frames), ocr=ocr_backend("fake", fake), cfg=_cfg())
    assert stats["n_hits"] == 4 and set(stats["castaways_hit"]) == {"S_A", "S_B", "S_C", "S_D"} and stats["n_scenes"] == 1
    hits = pd.read_sql_query("SELECT * FROM chyron_hits WHERE version_season='US99' AND episode=2 ORDER BY t_s", con)
    assert list(hits.castaway_id) == ["S_A", "S_D", "S_C", "S_B"] and hits.match_score.min() >= 85
    lab = pd.read_sql_query("SELECT * FROM labels", con).set_index("utt_id")
    assert lab.loc[ids[1][0], "source"] == "chyron" and lab.loc[ids[1][0], "speaker_id"] == "S_D"
    assert lab.loc[ids[0][0], "source"] == "chyron" and lab.loc[ids[0][0], "speaker_id"] == "S_A"   # agrees with SDH
    assert lab.loc[ids[3][0], "source"] == "human"                                               # untouched
    assert ids[2][0] not in lab.index                                                            # conflict: no label
    q = con.execute("SELECT reason, payload FROM review_queue WHERE utt_id=?", (ids[2][0],)).fetchone()
    assert q["reason"] == "chyron_conflict" and json.loads(q["payload"])["chyron"] == "S_C"
    assert stats["n_sdh_compared"] == 3 and abs(stats["sdh_agreement"] - 2 / 3) < 1e-3 and stats["n_conflicts"] == 1
    assert lab.loc[ids[1][0], "start_s"] == st[ids[1][0]]
    scenes = con.execute("SELECT tribe, day, kind FROM scenes WHERE version_season='US99'").fetchall()
    assert [tuple(x) for x in scenes] == [("KOTA", 4, "day")]
    # the bank sees D's whole run through the chyron anchor
    df = collect_labelled(s, con, "US99", [2], "vocals")
    d = df[df.speaker_id == "S_D"]
    assert len(d) == 3 and set(d.label_source) == {"chyron", "chyron_run"}
    # re-running replaces chyron labels and the open conflict without duplicating
    chyron_episode(s, con, "US99", 2, r, frames=iter(frames), ocr=ocr_backend("fake", fake), cfg=_cfg())
    assert con.execute("SELECT COUNT(*) FROM labels WHERE source='chyron'").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM review_queue WHERE reason='chyron_conflict'").fetchone()[0] == 1
    # the review queue shows the chyron candidate on the conflict
    from fastapi.testclient import TestClient
    from survspk.review_app import create_app
    c = TestClient(create_app(s, resolver_obj=r))
    g = next(g for g in c.get("/api/queue/US99/2").json()["groups"] if "chyron_conflict" in g["reasons"])
    assert g["chyron"]["speaker_id"] == "S_C"


def test_captions_bursts_and_ocr_cache(season, tmp_path):
    """Mixed-case lines are the show's captions, not cards; a burst of names is the opening credits; a run writes the
    raw OCR and --from-cache replays it to the same hits."""
    from survspk.chyron import drop_bursts, ocr_backend, read_ocr_cache, upper_share

    s, con, sea = season
    r = R(s)
    m = CastMatcher(r, "US99", 2, _cfg())
    assert upper_share("SUE FLIGHT SCHOOL OWNER") == 1.0 and upper_share("and I'm gonna say") < 0.1
    assert frame_hits([OcrLine("and I'm gonna say", 0.9, 0.05, 0.6)], m, _cfg()) == (None, None)       # "and" ~ ANDY
    assert frame_hits([OcrLine("Sue, you got that started?", 0.9, 0.05, 0.6)], m, _cfg()) == (None, None)
    assert frame_hits([OcrLine("SUE", 0.9, 0.05, 0.6)], m, _cfg())[0][0] == "S_D"
    credits = [Hit(300.0 + 3 * k, cid, cid, 0.9, 100) for k, cid in enumerate(["S_A", "S_B", "S_C", "S_D"])]
    kept, dropped = drop_bursts(credits + [Hit(500.0, "S_A", "ANDY", 0.9, 100)], 4, 40.0)
    assert [h.t_s for h in kept] == [500.0] and len(dropped) == 4
    kept, dropped = drop_bursts(credits[:3], 4, 40.0)                 # three names in a scene is ordinary
    assert len(kept) == 3 and not dropped
    # dedupe keeps how long a card stayed up
    d = dedupe([Hit(100.0, "S_A", "ANDY", 0.9, 100), Hit(100.5, "S_A", "ANDY", 0.9, 100), Hit(102.0, "S_A", "ANDY", 0.9, 100)], 10.0)
    assert len(d) == 1 and d[0].t_s == 100.0 and d[0].t_end == 102.0
    # cache round trip
    ids = sea.episode(2, [{"spk": "S_D", "label": None, "n": 2, "dur_each": 5.0}])
    st = {u["utt_id"]: u["start_s"] for u in con.execute("SELECT utt_id, start_s FROM utterances WHERE episode=2")}
    con.execute("INSERT OR REPLACE INTO episodes (version_season, episode, duration_s, video_path) VALUES ('US99', 2, 400, '/nope.mkv')")
    con.commit()
    fake = {20: [OcrLine("SUE", 0.9, 0.05, 0.6)], 60: [OcrLine("Sue said it", 0.9, 0.05, 0.6)]}
    frames = [(st[ids[0][0]] + 3.0, _frame(20)), (st[ids[0][0]] + 3.5, _frame(20)), (st[ids[0][0]] + 6.0, _frame(60))]
    cache = tmp_path / "ocr.jsonl"
    a = chyron_episode(s, con, "US99", 2, r, frames=iter(frames), ocr=ocr_backend("fake", fake), cfg=_cfg(), cache_path=cache,
                       write_labels=False)
    assert cache.exists() and len(list(read_ocr_cache(cache))) == 3 and a["n_hits"] == 1 and not a["from_cache"]
    b = chyron_episode(s, con, "US99", 2, r, cfg=_cfg(), cache_path=cache, use_cache=True, write_labels=False)
    assert b["from_cache"] and b["n_hits"] == 1 and b["castaways_hit"] == a["castaways_hit"]


def test_card_check_api_and_rerun(season):
    """The card check: the review API lists each card with the lines around it and the automatic pick; a checked card
    labels the chosen line (confidence 1.0), 'none' labels nothing, a re-run of the stage keeps both answers, and
    unchecking hands the card back to the automatic rule."""
    from fastapi.testclient import TestClient

    from survspk.chyron import ocr_backend
    from survspk.review_app import create_app

    s, con, sea = season
    r = R(s)
    ids = sea.episode(2, [{"spk": "S_A", "label": None, "n": 2, "dur_each": 4.0},
                          {"spk": "S_D", "label": None, "n": 2, "dur_each": 5.0}])
    st = {u["utt_id"]: u["start_s"] for u in con.execute("SELECT utt_id, start_s FROM utterances WHERE episode=2")}
    con.execute("INSERT OR REPLACE INTO episodes (version_season, episode, duration_s, video_path) VALUES ('US99', 2, 400, '/nope.mkv')")
    con.commit()
    fake = {20: [OcrLine("SUE", 0.9, 0.05, 0.6)]}
    frames = [(st[ids[1][0]] + 3.0, _frame(20))]

    def rerun():
        return chyron_episode(s, con, "US99", 2, r, frames=iter(frames), ocr=ocr_backend("fake", fake), cfg=_cfg())

    rerun()
    chy = lambda: {x["utt_id"]: (x["speaker_id"], x["confidence"]) for x in con.execute("SELECT * FROM labels WHERE source='chyron'")}  # noqa: E731
    assert chy() == {ids[1][0]: ("S_D", 1.0)} or set(chy()) == {ids[1][0]}                  # the automatic pick
    c = TestClient(create_app(s, resolver_obj=r))
    d = c.get("/api/cards/US99/2").json()
    card = d["cards"][0]
    assert card["castaway_id"] == "S_D" and card["suggested"] == ids[1][0] and not card["checked"]
    assert len(card["lines"]) >= 2 and [l["start_s"] for l in card["lines"]] == sorted(l["start_s"] for l in card["lines"])
    assert any(p["id"] == "S_D" and p["n"] == 1 and p["unchecked"] == 1 for p in d["per_castaway"])
    ep = next(e for e in c.get("/api/episodes").json() if e["ep"] == 2)
    assert ep["n_cards"] == 1 and ep["n_checked"] == 0
    other = next(l["utt_id"] for l in card["lines"] if l["utt_id"] != card["suggested"])
    # the person hears that the carded castaway speaks another line
    assert c.post("/api/card_check", json={"vs": "US99", "ep": 2, "t_s": card["t_s"], "castaway_id": "S_D", "utt_id": other}).json()["labelled"]
    assert chy() == {other: ("S_D", 1.0)}
    d = c.get("/api/cards/US99/2").json()
    assert d["cards"][0]["checked"] and d["cards"][0]["answer"] == other and d["per_castaway"][0]["n"] >= 0
    assert next(e for e in c.get("/api/episodes").json() if e["ep"] == 2)["n_checked"] == 1
    st2 = rerun()                                                   # a re-run keeps the answer
    assert chy() == {other: ("S_D", 1.0)} and st2["n_checked"] == 1
    # "none of these lines": no label, and a re-run does not bring the automatic one back
    c.post("/api/card_check", json={"vs": "US99", "ep": 2, "t_s": card["t_s"], "castaway_id": "S_D", "utt_id": None})
    assert chy() == {}
    rerun()
    assert chy() == {}
    # a human label on the chosen line wins over the check
    con.execute("INSERT INTO labels (utt_id, speaker_id, source, confidence) VALUES (?,?,?,1.0)", (other, "S_A", "human"))
    con.commit()
    assert not c.post("/api/card_check", json={"vs": "US99", "ep": 2, "t_s": card["t_s"], "castaway_id": "S_D", "utt_id": other}).json()["labelled"]
    assert con.execute("SELECT source, speaker_id FROM labels WHERE utt_id=?", (other,)).fetchone()[:] == ("human", "S_A")
    # uncheck: the card goes back to the automatic rule on the next run
    c.post("/api/card_uncheck", json={"vs": "US99", "ep": 2, "t_s": card["t_s"], "castaway_id": "S_D"})
    assert not c.get("/api/cards/US99/2").json()["cards"][0]["checked"]
    rerun()
    assert set(chy()) == {ids[1][0]}
    # the page serves the card mode
    assert "check name cards" in c.get("/").text


def test_sample_band_keeps_time_on_a_real_video(tmp_path):
    """Frames from sample_band must carry their true time: a 1080p clip whose band ffmpeg rounds to a height the
    arithmetic would not guess (US47 bug: 150 vs 152 rows) still yields fps x duration frames, the last at ~duration."""
    import shutil
    import subprocess

    import pytest

    from survspk.chyron import sample_band
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    v = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=2:duration=300",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(v)], check=True)
    cfg = ChyronCfg(fps=2.0, diff_threshold=0.0, hwaccel=None)
    ts = [t for t, _ in sample_band(v, (0.72, 1.0), 0.0, 300.0, cfg)]
    # the old arithmetic read 150-row frames out of 152-row ones: 608 frames and a last time of ~303.5 s
    assert 598 <= len(ts) <= 601 and abs(ts[-1] - 299.5) <= 0.6


def test_card_check_several_lines(season):
    """A card whose castaway speaks two of the nearby lines: both get the checked label, a re-run keeps both, and a
    check stored before several-lines existed (utt_id only) still reads as one line."""
    from fastapi.testclient import TestClient

    from survspk.chyron import card_checks, ocr_backend
    from survspk.review_app import create_app

    s, con, sea = season
    r = R(s)
    ids = sea.episode(2, [{"spk": "S_A", "label": None, "n": 2, "dur_each": 4.0},
                          {"spk": "S_D", "label": None, "n": 3, "dur_each": 3.0}])
    st = {u["utt_id"]: u["start_s"] for u in con.execute("SELECT utt_id, start_s FROM utterances WHERE episode=2")}
    con.execute("INSERT OR REPLACE INTO episodes (version_season, episode, duration_s, video_path) VALUES ('US99', 2, 400, '/nope.mkv')")
    con.commit()
    fake = {20: [OcrLine("SUE", 0.9, 0.05, 0.6)]}
    frames = [(st[ids[1][0]] + 1.0, _frame(20))]
    run = lambda: chyron_episode(s, con, "US99", 2, r, frames=iter(frames), ocr=ocr_backend("fake", fake), cfg=_cfg())  # noqa: E731
    run()
    c = TestClient(create_app(s, resolver_obj=r))
    card = c.get("/api/cards/US99/2").json()["cards"][0]
    two = [ids[1][0], ids[1][1]]
    assert all(u in [l["utt_id"] for l in card["lines"]] for u in two)
    res = c.post("/api/card_check", json={"vs": "US99", "ep": 2, "t_s": card["t_s"], "castaway_id": "S_D", "utt_ids": two}).json()
    assert res["n_labelled"] == 2
    chy = lambda: {x["utt_id"]: x["speaker_id"] for x in con.execute("SELECT * FROM labels WHERE source='chyron'")}  # noqa: E731
    assert chy() == {two[0]: "S_D", two[1]: "S_D"}
    got = c.get("/api/cards/US99/2").json()["cards"][0]
    assert got["answers"] == two and got["answer"] == two[0]
    run()
    assert chy() == {two[0]: "S_D", two[1]: "S_D"}
    # an older check row (utt_id only, no utt_ids) is one line
    con.execute("UPDATE card_checks SET utt_ids=NULL, utt_id=?", (two[1],))
    con.commit()
    assert card_checks(con, "US99", 2)[0]["utt_ids"] == [two[1]]
    run()
    assert chy() == {two[1]: "S_D"}


def test_a_castaway_named_like_the_host_wins_the_card(season):
    """US31: castaway Jeff Varner's card reads JEFF; the host never has a card, so the castaway wins."""
    s, con, sea = season
    r = R(s)
    m = CastMatcher(r, "US99", 2, _cfg())
    m.names["JEFF"] = "S_A"                       # a castaway called Jeff in this season
    assert m.match([OcrLine("JEFF THE AUSTRALIAN OUTBACK", 0.9, 0.05, 0.6)])[0] == "S_A"
    del m.names["JEFF"]
    assert m.match([OcrLine("JEFF", 0.9, 0.05, 0.6)])[0] == "HOST_US"
