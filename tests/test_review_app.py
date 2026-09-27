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
    # "maybe": who each group could be, by the coverage panel's rule, with the audio rank (0 = captions only)
    for x in groups:
        assert {t[0] for t in x["audio_top"]} <= set(x["maybe"])
        assert all(1 <= x["maybe"][sid] <= k + 1 for k, (sid, _, _) in enumerate(x["audio_top"]))   # best rank over the lines
    assert groups[0]["maybe"]["S_A"] in (0, 1, 2, 3, 4, 5)                  # the SDH name of the conflict is in it


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


def test_human_labels_carry_their_span(client):
    c, r = client
    con = c._con
    q = c.get("/api/queue/US99/2").json()
    g = next(g for g in q["groups"] if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    c.post("/api/label", json={"utt_ids": g["utt_ids"][:1], "speaker_id": "S_D"})
    lab = con.execute("SELECT * FROM labels WHERE utt_id=?", (g["utt_ids"][0],)).fetchone()
    u = con.execute("SELECT * FROM utterances WHERE utt_id=?", (g["utt_ids"][0],)).fetchone()
    assert lab["version_season"] == "US99" and lab["episode"] == 2
    assert lab["start_s"] == u["start_s"] and lab["end_s"] == u["end_s"] and lab["text"] == u["text"]
    # labels written by assign get their span on the next init_db / assign
    assert con.execute("SELECT COUNT(*) FROM labels WHERE start_s IS NULL").fetchone()[0] == 0


def test_audit_sample_and_verdicts(client):
    """The audit sample is stable, spread over predicted speakers, and shrinks as verdicts come in; a confirm
    keeps the auto speaker, a reject writes the corrected one; both become human labels; undo clears the verdict."""
    c, r = client
    con = c._con
    a = c.get("/api/audit/US99/2?n=50").json()
    assert a["groups"] and all(g["pred"] and g["reasons"] == ["audit"] for g in a["groups"])
    assert all(g["audio_top"][0][0] == g["pred"] for g in a["groups"])
    assert a["stats"]["n"] == 0 and a["stats"]["precision"] is None
    # every audited utterance is an auto label
    src = {r_["utt_id"]: r_["source"] for r_ in con.execute("SELECT utt_id, source FROM labels")}
    assert all(src[u] == "auto" for g in a["groups"] for u in g["utt_ids"])
    # stable across calls
    assert [g["utt_ids"] for g in c.get("/api/audit/US99/2?n=50").json()["groups"]] == [g["utt_ids"] for g in a["groups"]]
    g0 = a["groups"][0]
    res = c.post("/api/audit_verdict", json={"utt_ids": g0["utt_ids"], "speaker_id": g0["pred"],
                                             "pred_speaker": g0["pred"], "pred_score": g0["pred_score"]}).json()
    assert res["verdict"] == "confirm" and res["stats"]["n"] == 1 and res["stats"]["precision"] == 1.0
    lab = {r_["utt_id"]: dict(r_) for r_ in con.execute("SELECT * FROM labels")}
    assert all(lab[u]["source"] == "human" and lab[u]["speaker_id"] == g0["pred"] for u in g0["utt_ids"])
    assert c.get("/api/audit/US99/2?n=1").json()["groups"] == []          # a sample of 1 is done after 1 verdict
    a2 = c.get("/api/audit/US99/2?n=50").json()
    assert g0["utt_ids"] not in [g["utt_ids"] for g in a2["groups"]] and a2["stats"]["n"] == 1
    if a2["groups"]:
        g1 = a2["groups"][0]
        res = c.post("/api/audit_verdict", json={"utt_ids": g1["utt_ids"], "speaker_id": "S_C",
                                                 "pred_speaker": g1["pred"], "pred_score": g1["pred_score"]}).json()
        assert res["verdict"] == ("confirm" if g1["pred"] == "S_C" else "reject")
        st = c.get("/api/audit_stats/US99/2").json()
        assert st["n"] == 2 and 0 <= st["wilson_low"] <= st["precision"] <= 1
        assert g1["pred"] in st["per_speaker"]
    # undo removes the verdict and the human label
    c.post("/api/unlabel", json={"utt_ids": g0["utt_ids"]})
    st = c.get("/api/audit_stats/US99/2").json()
    assert st["n"] == (1 if a2["groups"] else 0)
    assert g0["utt_ids"] in [g["utt_ids"] for g in c.get("/api/audit/US99/2?n=50").json()["groups"]]
    # a refit re-runs assign; the confirmed/rejected labels are human now and survive it
    if a2["groups"]:
        c.post("/api/refit", json={"vs": "US99", "ep": 2})
        assert all(r_["source"] == "human" for r_ in con.execute(
            "SELECT source FROM labels WHERE utt_id IN (%s)" % ",".join("?" * len(g1["utt_ids"])), g1["utt_ids"]))


def test_voice_samples_context_times_and_refit_changes(client):
    """Voice samples come from trusted lines (a person's, a card's, an explicit caption name); neighbours carry their
    times for 'play with the line before'; a refit reports what it changed."""
    c, r = client
    q = c.get("/api/queue/US99/2").json()
    g = next(g for g in q["groups"] if "I'm 59" in " ".join(u["text"] for u in g["utts"]))
    assert all("start_s" in x and "end_s" in x for x in g["before"] + g["after"])
    assert c.get("/api/voice/US99/2/S_D").json()["samples"] == []           # nobody has vouched for S_D yet
    c.post("/api/label", json={"utt_ids": g["utt_ids"], "speaker_id": "S_D"})
    v = c.get("/api/voice/US99/2/S_D?n=2").json()
    assert len(v["samples"]) == 2 and {x["source"] for x in v["samples"]} == {"human"} and v["n_available"] == 3
    assert all(x["utt_id"] in g["utt_ids"] and x["ep"] == 2 for x in v["samples"])
    sa = c.get("/api/voice/US99/2/S_A").json()["samples"]                   # explicit caption names count too
    assert sa and all(x["source"] in ("caption name", "human", "chyron") for x in sa)
    res = c.post("/api/refit", json={"vs": "US99", "ep": 2}).json()
    ch = res["changes"]
    assert set(ch) == {"n_flipped", "n_new", "n_dropped", "flips"} and ch["n_flipped"] == len(ch["flips"])
    assert all(f["old"] != f["new"] and f["new_source"] != "human" for f in ch["flips"])


def test_diar_strip_renumbers_voices_by_presence(client):
    c, r = client
    s = c._s
    from survspk.stage_diarize import diar_path
    q = c.get("/api/queue/US99/2").json()
    g = q["groups"][0]
    a, b = g["start_s"], g["end_s"]
    assert c.get(f"/api/diar/US99/2?start={a}&end={b}").json()["ch"] == []  # not diarized: empty strip
    dom = np.full(40000, -1, np.int8)
    m = a + 0.7 * (b - a)
    dom[int(a * 100):int(m * 100)] = 3                                       # channel 3 most of the line ...
    dom[int(m * 100):int(b * 100)] = 1                                       # ... then channel 1
    p = diar_path(s, "US99", 2)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, starts=np.array([0.0]), lens=np.array([40000]), dom=dom, pmax=np.zeros(40000, np.uint8))
    d = c.get(f"/api/diar/US99/2?start={a}&end={b}").json()
    ch = d["ch"]
    assert d["step"] == pytest.approx(0.05) and set(ch) <= {0, 1, -1}
    assert ch[0] == 0 and ch[-1] == 1                                        # the most heard voice is 0
    j = next(k for k, x in enumerate(ch) if x == 1)
    assert d["t0"] + j * d["step"] == pytest.approx(m, abs=0.1)


def test_bulk_confirm_and_undo(client):
    c, r = client
    con = c._con
    spk = con.execute("""SELECT l.speaker_id, COUNT(*) FROM labels l JOIN utterances u USING (utt_id)
                         WHERE u.episode=2 AND u.segment='body' AND l.source='auto' GROUP BY 1 ORDER BY 2 DESC""").fetchone()[0]
    pv = c.get(f"/api/bulk_confirm/US99/2?speaker_id={spk}&min_confidence=0").json()
    assert pv["n"] == pv["n_auto"] > 0
    assert c.get(f"/api/bulk_confirm/US99/2?speaker_id={spk}&min_confidence=1.01").json()["n"] == 0
    before = {r_[0]: r_[1] for r_ in con.execute("SELECT utt_id, confidence FROM labels WHERE source='auto' AND speaker_id=?", (spk,))}
    res = c.post("/api/bulk_confirm", json={"vs": "US99", "ep": 2, "speaker_id": spk, "min_confidence": 0}).json()
    assert res["n"] == pv["n"] and set(res["utt_ids"]) <= set(before)
    rows = con.execute("SELECT source, top_candidates FROM labels WHERE utt_id IN (%s)" % ",".join("?" * res["n"]), res["utt_ids"]).fetchall()
    assert all(x[0] == "human" and '"bulk:confirm"' in x[1] for x in rows)
    assert c.post("/api/bulk_unconfirm", json={"utt_ids": res["utt_ids"]}).json()["n"] == res["n"]
    after = {r_[0]: r_[1] for r_ in con.execute("SELECT utt_id, confidence FROM labels WHERE source='auto' AND speaker_id=?", (spk,))}
    assert after == before


def test_line_frame_needs_the_video(client):
    c, r = client
    assert c.get("/api/line_frame/US99/2?t=12.3").status_code == 404


def test_two_voices_dismiss_takes_a_json_body(client):
    """Its body model used to live inside create_app, where FastAPI could not resolve it and asked for a query param."""
    c, r = client
    uid = c._ids[0][0]
    assert c.post("/api/two_voices/dismiss", json={"utt_id": uid}).status_code == 200


def test_music_scenes_label_and_unlabel(client):
    c, r = client
    eps = c.get("/api/music_episodes").json()
    assert {(e["vs"], e["ep"]) for e in eps} >= {("US99", 1), ("US99", 2)} and all(e["n_labeled"] == 0 for e in eps)
    m = c.get("/api/music/US99/2").json()
    assert m["model_file"] is None and {x["key"] for x in m["cues"]} >= {"dodo", "strategy", "none"}
    sc = m["scenes"]
    assert len(sc) >= 2 and not m["done"]
    assert all(x["t1"] - x["t0"] <= 45.0 + 5.0 and x["lines"] and x["guess"] is None for x in sc)
    assert len({x["t0"] for x in sc}) == len(sc)                            # the round-robin keeps each scene once
    x = sc[0]
    assert c.post("/api/music_label", json={"vs": "US99", "ep": 2, "t0": x["t0"], "t1": x["t1"], "cue": "bogus"}).status_code == 400
    res = c.post("/api/music_label", json={"vs": "US99", "ep": 2, "t0": x["t0"], "t1": x["t1"], "cue": "dodo",
                                           "subjects": ["S_A"], "model_cue": None}).json()
    assert res["n"] == 1
    m2 = c.get("/api/music/US99/2").json()
    assert len(m2["scenes"]) == len(sc) - 1 and m2["done"][0]["label"] == {"cue": "dodo", "subjects": ["S_A"]}
    assert m2["stats"] == {"dodo": 1}
    # relabelling the same scene replaces it
    c.post("/api/music_label", json={"vs": "US99", "ep": 2, "t0": x["t0"], "t1": x["t1"], "cue": "strategy", "subjects": []})
    assert c.get("/api/music/US99/2").json()["stats"] == {"strategy": 1}
    c.post("/api/music_unlabel", json={"vs": "US99", "ep": 2, "t0": x["t0"]})
    assert len(c.get("/api/music/US99/2").json()["scenes"]) == len(sc)


def test_music_model_guess_orders_the_queue(client, tmp_path):
    c, r = client
    sc = c.get("/api/music/US99/2").json()["scenes"]
    last = max(sc, key=lambda x: x["t0"])
    uids = [u for run in c._ids for u in run]
    rows = c._con.execute("SELECT utt_id, start_s FROM utterances WHERE version_season='US99' AND episode=2").fetchall()
    p = tmp_path / "cues.csv"
    with open(p, "w") as f:
        f.write("version_season,episode,utt_id,cue_goofy,cue_tense\n")
        for uid, t in rows:
            hot = last["t0"] - 0.01 <= t <= last["t1"]
            f.write(f"US99,2,{uid},{0.9 if hot else 0.1},{0.1 if hot else 0.9}\n")
    assert uids
    c._s.raw.setdefault("music", {})["cues_csv"] = str(p)
    try:
        m = c.get("/api/music/US99/2").json()
        assert m["model_file"] == "cues.csv"
        top = m["scenes"][0]
        assert top["t0"] == last["t0"] and top["guess"] == "dodo" and top["model"]["dodo"] == 0.9   # goofy -> dodo
    finally:
        c._s.raw["music"].pop("cues_csv", None)
