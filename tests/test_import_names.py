"""import-names: borrowing NAME: labels from another release of the same captions."""
import json

import pandas as pd

from survspk import db as dbm
from survspk import import_names as im


def test_split_turns_names_and_dashes():
    t = im.split_turns(1.0, 2.0, "JEFF: Come on in, guys! - Thanks, Jeff.")
    assert [(x.name, x.text, x.marker) for x in t] == [("JEFF", "Come on in, guys!", False), (None, "Thanks, Jeff.", True)]
    t = im.split_turns(0, 1, "- I'm going home. - No way.")
    assert [x.marker for x in t] == [True, True]
    assert im.split_turns(0, 1, "Plain line.")[0].name is None
    assert im.split_turns(0, 1, "I: think so")[0].name is None       # one-letter "names" are not speakers...
    assert im.split_turns(0, 1, "Q: I'm the swing vote.")[0].name == "Q"   # ...except Q Burdette


LINES = [f"this is line number {w} of the episode" for w in
         "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
         "eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive".split()]


def _ours(offset=0.0):
    return [{"cue_idx": i, "line_i": 0, "start": 10.0 * i + offset, "end": 10.0 * i + offset + 2, "text": t,
             "sdh_name": None, "is_turn": False, "norm": im.norm(t)} for i, t in enumerate(LINES)]


def test_fit_clock_recovers_offset_and_drift():
    src = [im.Turn(10.0 * i, 10.0 * i + 2, None, t) for i, t in enumerate(LINES)]
    ours = _ours()
    for o in ours:
        o["start"] = 1.001 * o["start"] + 3.5
    slope, off, n = im.fit_clock(src, ours)
    assert n == len(LINES) and abs(slope - 1.001) < 1e-6 and abs(off - 3.5) < 1e-6


def test_match_names_across_offset_and_run_together_words():
    src = [im.Turn(10.0 * i, 10.0 * i + 2, "JEFF" if i == 3 else None, t.replace("line number", "linenumber"),
                   marker=i == 4) for i, t in enumerate(LINES)]
    ours = _ours(offset=2.0)
    ours[5]["sdh_name"] = "PROBST"                       # lines that already carry a name are left alone
    src[5].name = "JEFF"
    slope, off, _ = im.fit_clock(src, ours)
    hits = {h["cue_idx"]: h for h in im.match_names(src, ours, slope, off)}
    assert hits[3]["name"] == "JEFF" and hits[3]["turn"]     # a named turn is also a speaker change
    assert hits[4]["name"] is None and hits[4]["turn"]       # dash turn: marker only
    assert 5 not in hits
    assert set(hits) == {3, 4}


def _db(tmp_path):
    con = dbm.init_db(tmp_path / "t.sqlite", "DELETE")
    rows = [(f"US41_E02_C{i:04d}", "US41", 2, i, 10.0 * i, 10.0 * i + 2, t,
             json.dumps([{"text": t, "sdh_name": None, "is_turn": False, "is_italic": i == 0, "is_sound": False, "raw": t}]))
            for i, t in enumerate(LINES)]
    con.executemany("INSERT INTO cues (cue_id, version_season, episode, idx, start_s, end_s, raw_text, lines) VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return con


def _parquet(tmp_path, names):
    df = pd.DataFrame({"episode": "S41E02", "subtitle_number": range(len(LINES)),
                       "start_time": [10.0 * i + 1.0 for i in range(len(LINES))],
                       "end_time": [10.0 * i + 3.0 for i in range(len(LINES))],
                       "text": [(f"{names[i]}: " if i in names else "") + t for i, t in enumerate(LINES)]})
    df["duration"] = 2.0
    p = tmp_path / "src.parquet"
    df.to_parquet(p)
    return p


def test_import_writes_and_survives_reingest(tmp_path):
    con = _db(tmp_path)
    src = _parquet(tmp_path, {2: "KAMERON", 7: "DANIEL (V.O.)"})
    assert im.candidates(con, src, None, min_our=1, min_src=1) == [("US41", 2)]
    dry = im.import_episode(con, src, "US41", 2, write=False)
    assert dry["matched"] == 2 and dry["applied"] == 0 and abs(dry["offset_s"] + 1.0) < 1e-6
    assert con.execute("SELECT count(*) FROM sqlite_master WHERE name='name_imports'").fetchone()[0] == 0
    rep = im.import_episode(con, src, "US41", 2)
    assert rep["applied"] == 2
    lines = {r[0]: json.loads(r[1])[0] for r in con.execute("SELECT idx, lines FROM cues WHERE episode=2")}
    assert lines[2]["sdh_name"] == "KAMERON" and lines[2]["name_source"] == "import" and lines[2]["is_turn"]
    assert lines[7]["sdh_name"] == "DANIEL" and lines[0]["is_italic"]          # our formatting is kept
    # re-ingest: cues rebuilt without names; recorded imports go back only onto lines with the same text
    for idx, lines_json in con.execute("SELECT idx, lines FROM cues WHERE episode=2").fetchall():
        ls = json.loads(lines_json)
        for ln in ls:
            ln.pop("name_source", None); ln.pop("turn_source", None); ln["sdh_name"] = None; ln["is_turn"] = False
        if idx == 7:
            ls[0]["text"] = "a different release of this line"
        con.execute("UPDATE cues SET lines=? WHERE episode=2 AND idx=?", (json.dumps(ls), idx))
    assert im.apply_imports(con, "US41", 2) == 1
    lines = {r[0]: json.loads(r[1])[0] for r in con.execute("SELECT idx, lines FROM cues WHERE episode=2")}
    assert lines[2]["sdh_name"] == "KAMERON" and lines[7]["sdh_name"] is None


def test_too_few_anchors_writes_nothing(tmp_path):
    con = _db(tmp_path)
    src = _parquet(tmp_path, {2: "KAMERON"})
    df = pd.read_parquet(src)
    df = df.iloc[:10]
    df.to_parquet(src)
    rep = im.import_episode(con, src, "US41", 2)
    assert rep["anchors"] < im.MIN_ANCHORS and rep["applied"] == 0


def test_few_names_landing_writes_nothing(tmp_path):
    """Plenty of identical lines, but most of the source's names are on lines we do not have: another cut."""
    con = _db(tmp_path)
    src = _parquet(tmp_path, {2: "KAMERON"})
    df = pd.read_parquet(src)
    extra = pd.DataFrame({"episode": "S41E02", "subtitle_number": range(100, 103), "start_time": [300.0, 310.0, 320.0],
                          "end_time": [302.0, 312.0, 322.0], "duration": 2.0,
                          "text": [f"DANIEL: a scene cut from our version number {k}" for k in range(3)]})
    pd.concat([df, extra]).to_parquet(src)
    rep = im.import_episode(con, src, "US41", 2)
    assert rep["anchors"] >= im.MIN_ANCHORS and rep["matched"] == 1 and rep["source_names"] == 4
    assert not rep["ok"] and rep["applied"] == 0
