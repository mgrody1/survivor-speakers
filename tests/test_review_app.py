"""Review API on the synthetic season: queue grouping, clip slicing, labelling / undo / bulk accept."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from survspk.review_app import create_app
from survspk.stage_assign import assign_episode
from survspk.stage_bank import build_bank
from tests.test_bank_assign import season  # noqa: F401
from tests.test_text_prior import R


@pytest.fixture
def client(season):
    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    e2 = [
        {"spk": "S_A", "label": "S_A", "n": 2, "dur_each": 4.0},
        {"spk": "S_D", "label": None, "n": 3, "dur_each": 5.0, "text": "I'm 59 and I own a flight school"},   # queued
        {"spk": "S_C", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "WRONGNAME"},                       # sdh_conflict
        {"spk": "S_B", "label": None, "n": 2, "dur_each": 4.0},                                             # auto
    ]
    ids = sea.episode(2, e2)
    r = R(s)
    assign_episode(s, con, "US99", 2, variant="vocals", resolver=r, write=True)
    # a fake audio file so /api/clip has something to slice: 1 kHz tone, 16 kHz, long enough
    p = s.audio_path("raw", "US99", 2)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, 400 * 16000) / 16000
    sf.write(str(p), (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), 16000, subtype="PCM_16")
    c = TestClient(create_app(s, resolver_obj=r), raise_server_exceptions=True)
    c._ids = ids
    c._s, c._con = s, con
    yield c, r


def test_episode_list_and_queue_groups(client, monkeypatch):
    c, r = client
    eps = c.get("/api/episodes").json()
    assert eps and eps[0]["vs"] == "US99" and eps[0]["ep"] == 2 and eps[0]["n_open"] >= 3
    q = c.get("/api/queue/US99/2").json()
    assert q["n_open"] >= 2 and {x["id"] for x in q["candidates"]} >= {"S_A", "S_B", "S_C", "S_D", "HOST_US"}
    assert [x["id"] for x in q["pseudo"]] == ["OTHER", "UNKNOWN", "NOSPEECH"]
    groups = q["groups"]
    # sdh_conflict first, then the long unbanked run
    assert "sdh_conflict" in groups[0]["reasons"] and groups[0]["sdh"]["speaker_id"] == "S_A" and groups[0]["sdh"]["explicit"]
    g = next(g for g in groups if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    assert len(g["utt_ids"]) == 3 and g["dur"] > 14 and g["audio_top"] and g["before"] and g["after"]
    assert g["before"][-1]["speaker"] and g["after"][0]["speaker"]          # neighbours carry their labels


def test_clip_is_wav_slice(client, monkeypatch):
    c, r = client
    q = c.get("/api/queue/US99/2").json()
    g = q["groups"][0]
    resp = c.get(f"/api/clip/US99/2?start={g['start_s']}&end={g['end_s']}&variant=raw")
    assert resp.status_code == 200 and resp.headers["content-type"] == "audio/wav"
    import io
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 16000 and abs(len(data) / sr - (g["end_s"] - g["start_s"] + 0.7)) < 0.05
    # missing variant falls back to vocals if present, else 404
    assert c.get("/api/clip/US99/2?start=0&end=1&variant=center").status_code in (200, 404)


def test_label_undo_bulk(client, monkeypatch):
    c, r = client
    s, con = c._s, c._con
    q = c.get("/api/queue/US99/2").json()
    g = next(g for g in q["groups"] if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    # label two of three utterances (a split), then the rest
    resp = c.post("/api/label", json={"utt_ids": g["utt_ids"][:2], "speaker_id": "S_D"})
    assert resp.status_code == 200, resp.text
    res = resp.json()
    assert res["ok"] and res["n"] == 2
    lab = pd.read_sql_query("SELECT * FROM labels WHERE source='human'", con)
    assert set(lab.utt_id) == set(g["utt_ids"][:2]) and (lab.speaker_id == "S_D").all()
    q2 = c.get("/api/queue/US99/2").json()
    g2 = next(x for x in q2["groups"] if x["run_id"] == g["run_id"])
    assert g2["utt_ids"] == g["utt_ids"][2:]
    # undo reopens them
    c.post("/api/unlabel", json={"utt_ids": g["utt_ids"][:2]})
    assert pd.read_sql_query("SELECT COUNT(*) n FROM labels WHERE source='human'", con).n[0] == 0
    q3 = c.get("/api/queue/US99/2").json()
    assert next(x for x in q3["groups"] if x["run_id"] == g["run_id"])["utt_ids"] == g["utt_ids"]
    # pseudo label + bulk accept by audio top-1 at a threshold nothing reaches -> 0
    c.post("/api/label", json={"utt_ids": [g["utt_ids"][0]], "speaker_id": "NOSPEECH"})
    assert c.post("/api/bulk_accept", json={"vs": "US99", "ep": 2, "source": "audio", "min_confidence": 0.999}).json()["n"] == 0
    st = c.get("/api/stats/US99/2").json()
    assert st["body_utts"] == 9 and any(x["source"] == "human" for x in st["by_source"])


def test_split_and_unsplit_at_word_boundary(client):
    c, r = client
    s, con = c._s, c._con
    q = c.get("/api/queue/US99/2").json()
    g = next(g for g in q["groups"] if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    uid = g["utt_ids"][0]
    u = con.execute("SELECT * FROM utterances WHERE utt_id=?", (uid,)).fetchone()
    # give the utterance a cue with aligned words: "I'm 59 and I own a flight school" -> 8 words over its span
    con.execute("INSERT OR REPLACE INTO cues (cue_id, version_season, episode, idx, start_s, end_s, lines) VALUES (?,?,?,?,?,?,?)",
                ("US99_E02_C0001", "US99", 2, 1, u["start_s"], u["end_s"], "[]"))
    con.execute("INSERT OR REPLACE INTO utterance_cues (utt_id, cue_id, line_indices) VALUES (?,?,?)", (uid, "US99_E02_C0001", "[0]"))
    toks = "I'm 59 and I own a flight school".split()
    step = (u["end_s"] - u["start_s"]) / len(toks)
    for k, t in enumerate(toks):
        st = u["start_s"] + k * step
        con.execute("INSERT OR REPLACE INTO words (cue_id, line_idx, word_idx, word, start_s, end_s, score) VALUES (?,?,?,?,?,?,?)",
                    ("US99_E02_C0001", 0, k, t, st if k != 3 else None, st + step * 0.9 if k != 3 else None, 0.9))
    con.commit()
    words = c.get(f"/api/words/{uid}").json()
    assert [w["word"] for w in words] == toks and words[3]["start_s"] is not None      # gap interpolated
    # the parquet is fresh before, stale after the split
    assert not c.get("/api/episodes").json()[0]["stale_embeddings"]
    res = c.post("/api/split", json={"utt_id": uid, "before_word": 2}).json()
    assert res["ok"] and res["parts"] == [uid + "a", uid + "b"]
    a = con.execute("SELECT * FROM utterances WHERE utt_id=?", (uid + "a",)).fetchone()
    b = con.execute("SELECT * FROM utterances WHERE utt_id=?", (uid + "b",)).fetchone()
    assert a["text"] == "I'm 59" and b["text"] == "and I own a flight school"
    assert abs(a["end_s"] - res["t_cut"]) < 1e-3 and abs(b["start_s"] - res["t_cut"]) < 1e-3 and a["end_s"] == b["start_s"]
    assert a["idx"] == u["idx"] and b["idx"] == u["idx"] + 0.5 and b["sdh_name"] is None
    assert con.execute("SELECT COUNT(*) FROM utterances WHERE utt_id=?", (uid,)).fetchone()[0] == 0
    assert c.get("/api/episodes").json()[0]["stale_embeddings"]
    q2 = c.get("/api/queue/US99/2").json()
    ids2 = {u for g in q2["groups"] for u in g["utt_ids"]}
    assert uid + "a" in ids2 and uid + "b" in ids2 and uid not in ids2
    # each part can be labelled on its own
    c.post("/api/label", json={"utt_ids": [uid + "a"], "speaker_id": "S_D"})
    c.post("/api/label", json={"utt_ids": [uid + "b"], "speaker_id": "S_A"})
    lab = pd.read_sql_query("SELECT * FROM labels WHERE source='human'", con).set_index("utt_id")
    assert lab.loc[uid + "a", "speaker_id"] == "S_D" and lab.loc[uid + "b", "speaker_id"] == "S_A"
    # bad boundaries are refused
    assert c.post("/api/split", json={"utt_id": uid + "b", "before_word": 0}).status_code == 400
    # undo restores the original row, cues and queue entry, removes the parts and their labels
    assert c.post("/api/unsplit", json={"base_utt_id": uid}).json()["ok"]
    back = con.execute("SELECT * FROM utterances WHERE utt_id=?", (uid,)).fetchone()
    assert back["text"] == u["text"] and back["start_s"] == u["start_s"] and back["idx"] == u["idx"]
    assert con.execute("SELECT COUNT(*) FROM utterances WHERE utt_id IN (?,?)", (uid + "a", uid + "b")).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM labels WHERE utt_id IN (?,?)", (uid + "a", uid + "b")).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM utterance_cues WHERE utt_id=?", (uid,)).fetchone()[0] == 1
    assert not c.get("/api/episodes").json()[0]["stale_embeddings"]


def test_coverage_and_refit_loop(client):
    """Label the unbanked speaker's queued run by hand, refit, and the queue for it clears; human labels survive."""
    c, r = client
    s, con = c._s, c._con
    cov = {x["id"]: x for x in c.get("/api/coverage/US99/2").json()}
    assert cov["S_D"]["n"] == 0 and cov["S_D"]["thin"] and cov["S_A"]["n"] >= 2
    # S_D's queued run is plausibly theirs? no bank entry -> not in audio top-3; but the host (never speaks here)
    # is 'quiet', and everyone reports the same targets
    assert all(x["min_utts"] == 5 and x["min_secs"] == 15.0 for x in cov.values())
    assert cov["HOST_US"]["status"] == "quiet" and cov["HOST_US"]["open_secs"] == 0
    assert all(x["status"] in ("banked", "thin", "quiet") for x in cov.values())
    # a queued run whose audio top-3 names S_A counts as open material for S_A
    assert cov["S_A"]["open_secs"] > 0 or cov["S_A"]["status"] == "banked"
    q = c.get("/api/queue/US99/2").json()
    g = next(g for g in q["groups"] if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    c.post("/api/label", json={"utt_ids": g["utt_ids"], "speaker_id": "S_D"})
    cov2 = {x["id"]: x for x in c.get("/api/coverage/US99/2").json()}
    assert cov2["S_D"]["n"] == 3 and cov2["S_D"]["n_human"] == 3
    # refit: bank from E1+E2 labels, reassign E2. S_D now has a bank entry; the human labels are untouched.
    res = c.post("/api/refit", json={"vs": "US99", "ep": 2}).json()
    assert res["ok"] and res["bank_speakers"] == 4 and "S_D" not in res["unbankable"]
    lab = pd.read_sql_query("SELECT * FROM labels WHERE source='human'", con)
    assert set(lab.utt_id) == set(g["utt_ids"]) and (lab.speaker_id == "S_D").all()
    assert pd.read_sql_query("SELECT resolved FROM review_queue WHERE utt_id=?", con, params=(g["utt_ids"][0],)).resolved.iloc[0] == 1
    # a re-run of assign through the CLI path also leaves them alone
    from survspk.stage_assign import assign_episode
    assign_episode(s, con, "US99", 2, variant="vocals", bank_as_of=2, resolver=r, write=True)
    lab2 = pd.read_sql_query("SELECT * FROM labels WHERE utt_id IN (%s)" % ",".join("?" * len(g["utt_ids"])), con, params=g["utt_ids"])
    assert (lab2.source == "human").all()
