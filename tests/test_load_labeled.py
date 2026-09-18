"""load_labeled / pool_runs against a tiny real sqlite + parquet (regression: `lab.flags` hit pandas' own
DataFrame.flags attribute instead of the column)."""

import json

import numpy as np
import pandas as pd
import pytest

from survspk.config import load_settings
from survspk.db import init_db
from survspk.stage_embed import (embedding_path, load_labeled, loo_predict, loo_scores, parquet_is_fresh, pool_runs,
                                 run_errors)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("SURVSPK_PROFILE", "macos")
    load_settings.cache_clear()
    s = load_settings()
    s.paths.work_root = tmp_path
    con = init_db(s.db_path, "DELETE")
    rng = np.random.default_rng(0)
    rows, vecs = [], []
    centers = {f"S{i}": rng.normal(size=8) for i in range(3)}
    i = 0
    for run_id, (spk, n, dur) in enumerate([("S0", 3, 9.0), ("S1", 3, 7.5), ("S2", 2, 2.0), ("S0", 4, 12.0),
                                             ("S1", 1, 1.0), ("S2", 3, 8.0)] * 3):
        for _ in range(n):                      # first utterance of a run carries the explicit NAME: line
            utt = f"US99_E01_U{i:04d}"
            flags = {"run_id": run_id, "run_dur_s": dur, "name_inherited": _ > 0}
            rows.append((utt, "US99", 1, i, i * 10.0, i * 10.0 + dur / n, "x", "body", 1, spk, spk, 0,
                         "confessional" if dur >= 6 else "field", 1, json.dumps(flags)))
            v = centers[spk] + rng.normal(scale=0.2, size=8)
            vecs.append({"utt_id": utt, "vector": (v / np.linalg.norm(v)).astype(np.float32), "snr_proxy": 1.0,
                         "start_s": i * 10.0, "end_s": i * 10.0 + dur / n})
            i += 1
    # one utterance with an empty flags string and an unresolved name -> must be tolerated / excluded
    rows.append((f"US99_E01_U{i:04d}", "US99", 1, i, 9999.0, 10003.0, "x", "body", 1, "ZED", None, 0, "field", 1, ""))
    vecs.append({"utt_id": f"US99_E01_U{i:04d}", "vector": np.ones(8, dtype=np.float32) / np.sqrt(8), "snr_proxy": 1.0,
                 "start_s": 9999.0, "end_s": 10003.0})
    con.executemany(
        """INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, is_speech,
           sdh_name, sdh_speaker_id, is_italic, domain_hint, align_ok, flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows)
    con.execute("UPDATE utterances SET sdh_resolution='cast' WHERE sdh_speaker_id IS NOT NULL")
    con.commit()
    p = embedding_path(s, "US99", 1, "vocals")
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vecs).to_parquet(p)
    yield s, con
    load_settings.cache_clear()


def test_load_labeled_reads_flags_column(tiny):
    s, con = tiny
    df = load_labeled(s, con, "US99", 1, "vocals", min_duration_s=0.0, min_per_speaker=1, all_durations=True)
    assert {"run_id", "run_dur", "domain_hint", "dur", "vector"} <= set(df.columns)
    assert df.speaker_id.notna().all() and set(df.speaker_id) == {"S0", "S1", "S2"}
    assert (df.run_id >= 0).all() and df.run_dur.max() == 12.0


def test_pool_runs_one_vector_per_run(tiny):
    s, con = tiny
    df = load_labeled(s, con, "US99", 1, "vocals", min_duration_s=0.0, min_per_speaker=1, all_durations=True)
    runs = pool_runs(df)
    assert len(runs) == df.run_id.nunique()
    assert runs.n_utts.sum() == len(df)
    assert np.allclose(np.linalg.norm(np.stack(runs.vector.to_numpy()), axis=1), 1.0, atol=1e-5)
    assert set(runs.domain) == {"confessional", "field"}


def test_loo_predict_matches_loo_scores():
    rng = np.random.default_rng(3)
    rows = []
    for s in range(3):
        c = rng.normal(size=8)
        for _ in range(6):
            v = c + rng.normal(scale=0.4, size=8)
            rows.append({"speaker_id": f"S{s}", "vector": (v / np.linalg.norm(v)).astype(np.float32)})
    df = pd.DataFrame(rows)
    p = loo_predict(df)
    sc = loo_scores(df)
    assert len(p) == len(df) and set(p.columns) >= {"pred", "second", "margin", "sim_true", "ex_pred"}
    assert abs((p.pred == df.speaker_id).mean() - sc["centroid_acc"]) < 1e-9
    assert (p.pred != p.second).all() and (p.margin >= 0).all()
    # a correct prediction means the true centroid was the best one
    ok = p.pred == df.speaker_id
    assert np.allclose(p.sim[ok], p.sim_true[ok])


def test_run_errors_reports_provenance_and_confusions(tiny):
    s, con = tiny
    res = run_errors(s, con, "US99", 1, "vocals", domain=None, min_per_speaker=1)
    runs, errors, per = res["runs"], res["errors"], res["per_speaker"]
    assert len(runs) == 18 and {"mmss", "n_explicit", "text", "pred", "margin"} <= set(runs.columns)
    assert (runs.n_explicit == 1).all()                   # one explicit NAME: line per run in the fixture
    assert set(per.index) == {"S0", "S1", "S2"} and {"n", "acc", "n_wrong_as"} <= set(per.columns)
    assert len(errors) == int((~runs.ok).sum())
    assert per.n_wrong_as.sum() == len(errors)
    conf = run_errors(s, con, "US99", 1, "vocals", domain="confessional", min_per_speaker=1)["runs"]
    assert (conf.domain == "confessional").all() and len(conf) < len(runs)


def test_stale_parquet_is_detected(tiny):
    s, con = tiny
    p = embedding_path(s, "US99", 1, "vocals")
    from survspk.stage_embed import _current_utts
    utts = _current_utts(con, "US99", 1, s.embed.min_duration_s)
    assert parquet_is_fresh(p, utts)
    # re-segmentation shifts one boundary -> stale; ablation refuses instead of scoring mis-paired vectors
    con.execute("UPDATE utterances SET end_s = end_s + 1.0 WHERE utt_id='US99_E01_U0003'")
    con.commit()
    assert not parquet_is_fresh(p, _current_utts(con, "US99", 1, s.embed.min_duration_s))
    with pytest.raises(RuntimeError, match="stale"):
        load_labeled(s, con, "US99", 1, "vocals", min_duration_s=0.0, min_per_speaker=1, all_durations=True)
    # an old parquet without spans is stale too
    pd.read_parquet(p).drop(columns=["start_s", "end_s"]).to_parquet(p)
    assert not parquet_is_fresh(p, utts)
