"""The auto-label calibrator: features, the logistic fit, the cut-off, and the runs it learns from."""

import numpy as np
import pandas as pd

from survspk import calibrate as cal
from survspk.stage_bank import build_bank
from tests.test_bank_assign import season  # noqa: F401
from tests.test_text_prior import R


def test_features_from_a_ranking():
    f = cal.features([["A", 0.8], ["B", 0.5], ["C", 0.4]], 6.0, 3, "confessional", "body", False, (40, 120.0), None, 4)
    assert set(f) == set(cal.FEATURES)
    assert abs(f["margin"] - 0.3) < 1e-9 and abs(f["gap23"] - 0.1) < 1e-9 and f["n_cands"] == 3 and f["confessional"] == 1
    assert f["log_bank_n"] > 0 and f["log_bank_n2"] == 0 and f["host_pred"] == 0
    g = cal.features([["HOST_US", 0.9]], 1.0, 1, "field", "recap", True, None, None, 1)
    assert g["top2"] == 0 and g["host_pred"] == 1 and g["recap"] == 1 and g["mention_pred"] == 1


def test_fit_learns_and_is_calibrated(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({f: rng.normal(size=n) for f in cal.FEATURES})
    z = 2.0 * df.top1 - 1.5 * df.mention_pred + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    m = cal.fit(df, y.to_numpy() if hasattr(y, "to_numpy") else y)
    p = m.prob(df)
    assert cal.auc(p, y) > 0.75 and abs(p.mean() - y.mean()) < 0.02
    c = dict(zip(cal.FEATURES, m.coef))
    assert c["top1"] > 1.0 and c["mention_pred"] < -0.8
    m.save(tmp_path / "c.json")
    m2 = cal.Calibrator.load(tmp_path / "c.json")
    assert np.allclose(m2.prob(df), p) and abs(m2.prob(df.iloc[0].to_dict()) - p[0]) < 1e-9


def test_threshold_matches_a_precision():
    p = np.array([0.95, 0.9, 0.8, 0.6, 0.4])
    y = np.array([1, 1, 1, 0, 1])
    d = np.ones(5)
    assert cal.threshold_for(p, y, d, 1.0) == 0.8
    assert cal.threshold_for(p, y, d, 0.8) == 0.4
    assert cal.threshold_for(p, np.zeros(5), d, 0.9) == 1.0


def test_replay_scores_runs_blind_from_earlier_episodes(season):
    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    sea.episode(2, [{"spk": "S_A", "label": "S_A", "n": 3, "dur_each": 4.0}, {"spk": "S_B", "label": "S_B", "n": 3, "dur_each": 4.0},
                    {"spk": "S_C", "label": None, "n": 2, "dur_each": 4.0}])
    before = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    df = cal.replay_frame(s, con, R(s), seasons=["US99"], log=lambda m: None)
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == before          # nothing written
    assert set(df.ep) == {2}                                                         # episode 1 has no earlier bank
    known = df[df.y.notna()]
    assert set(known.truth) >= {"S_A", "S_B"} and known.y.isin([0, 1]).all()
    assert (known.log_bank_n > 0).all() and set(cal.FEATURES) <= set(df.columns)
    # the unlabelled S_C run is scored too, with no truth
    assert df.truth.isna().any()


def test_context_features_see_other_runs_only(season):
    s, con, sea = season
    ids = sea.episode(2, [{"spk": "S_A", "label": "S_A", "n": 2, "dur_each": 4.0}, {"spk": "S_B", "label": "S_B", "n": 2, "dur_each": 4.0}])
    ctx = cal.EpisodeContext(con, "US99", 2)
    a0, a1 = ids[0]
    (t0, t1), = con.execute("SELECT MIN(start_s), MAX(end_s) FROM utterances WHERE utt_id IN (?,?)", (a0, a1)).fetchall()
    f = ctx.features([a0, a1], t0, t1, "S_A", "S_B")
    assert f["cap_pred_near"] == 0.0                   # its own caption name does not count
    assert f["cap_second_near"] == 1.0 and f["cap_next_pred"] == 0.0
    con.execute("INSERT INTO chyron_hits (version_season, episode, t_s, castaway_id) VALUES ('US99', 2, ?, 'S_A')", (t0 + 1,))
    assert cal.EpisodeContext(con, "US99", 2).features([a0, a1], t0, t1, "S_A", "S_B")["card_pred_near"] == 1.0


def _const(p_logit: float) -> cal.Calibrator:
    n = len(cal.FEATURES)
    return cal.Calibrator(list(cal.FEATURES), [0.0] * n, [1.0] * n, [0.0] * n, p_logit, threshold=0.5)


def test_assign_stores_p_and_can_let_the_calibrator_decide(season):
    from survspk.stage_assign import assign_episode

    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    sea.episode(2, [{"spk": "S_A", "label": None, "n": 3, "dur_each": 4.0}, {"spk": "S_B", "label": None, "n": 3, "dur_each": 4.0},
                    {"spk": "S_C", "label": None, "n": 2, "dur_each": 4.0}])
    df0, _ = assign_episode(s, con, "US99", 2, variant="vocals", resolver=R(s), write=True)
    assert con.execute("SELECT COUNT(*) FROM labels WHERE p_right IS NOT NULL").fetchone()[0] == 0     # no calibrator yet
    _const(4.0).save(cal.model_path(s))                        # p = 0.98 for everything
    assign_episode(s, con, "US99", 2, variant="vocals", resolver=R(s), write=True)
    ps = [r[0] for r in con.execute("SELECT p_right FROM labels WHERE source='auto'")]
    assert ps and all(abs(p - 0.982) < 0.01 for p in ps)
    s.raw.setdefault("thresholds", {})["calibrated_accept"] = True
    _const(-4.0).save(cal.model_path(s))                       # p = 0.02: nothing clears the cut-off
    df2, _ = assign_episode(s, con, "US99", 2, variant="vocals", resolver=R(s), write=True)
    assert (df2.decision == "auto").sum() == 0 and (df0.decision == "auto").sum() > 0
    s.raw["thresholds"]["calibrated_accept"] = False


def test_audit_ranker_uses_the_run_models_odds():
    m = _const(2.0)
    f = cal.features([["A", 0.8], ["B", 0.5]], 1.0, 1, "field", "body", False, None, None, 3)
    assert m.audit_prob(f) is None
    n = len(cal.AUDIT_FEATURES)
    coef = [0.0] * n
    coef[cal.AUDIT_FEATURES.index("lp")] = 1.0
    m.audit = cal.Calibrator(list(cal.AUDIT_FEATURES), [0.0] * n, [1.0] * n, coef, 0.0).__dict__
    assert abs(m.audit_prob(f) - m.prob(f)) < 1e-6              # lp in, same odds out


def test_top_reads_only_rankings():
    assert cal._top('[["A", 0.8], ["B", 0.5]]') == [["A", 0.8], ["B", 0.5]]
    assert cal._top('{"same_as_part": "US49_E13_U1600a"}') == [] and cal._top('{"note": "bulk:audio"}') == []
    assert cal._top("not json") == [] and cal._top(None) == []


def test_revisit_scores_an_episode_with_a_bank_from_the_others(season):
    from survspk.replay import revisit_bank
    from survspk.stage_assign import assign_episode

    s, con, sea = season
    sea.episode(2, [{"spk": s_, "label": s_, "n": 3, "dur_each": 4.0} for s_ in ("S_A", "S_B", "S_C")]
                + [{"spk": "S_B", "label": None, "n": 2, "dur_each": 4.0}])
    bank = revisit_bank(s, con, R(s), "US99", 1)
    assert {k[0] for k in bank} >= {"S_A", "S_B", "S_C"}
    n_banks = con.execute("SELECT COUNT(*) FROM speaker_bank").fetchone()[0]
    df, st = assign_episode(s, con, "US99", 1, variant="vocals", resolver=R(s), write=True, bank_override=bank)
    assert len(df) and con.execute("SELECT COUNT(*) FROM speaker_bank").fetchone()[0] == n_banks     # stored banks untouched
    assert st["sdh_agreement_runs"] is not None and st["sdh_agreement_runs"] > 0.8    # scored blind, still right


def test_queue_lines_carry_the_calibrators_chance(season):
    import json

    from survspk.stage_assign import assign_episode

    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    sea.episode(2, [{"spk": "S_D", "label": None, "n": 3, "dur_each": 5.0}, {"spk": "S_A", "label": None, "n": 2, "dur_each": 4.0}])
    assign_episode(s, con, "US99", 2, variant="vocals", resolver=R(s), write=True)       # no calibrator yet
    q = [json.loads(r[0]) for r in con.execute("SELECT payload FROM review_queue WHERE resolved=0 AND reason IN ('low_margin','no_candidate')")]
    assert q and all(x.get("p_right") is None for x in q)
    m = _const(1.0)
    m.save(cal.model_path(s))
    assert cal.score_queue(s, m, R(s)) == len(q)
    q = [json.loads(r[0]) for r in con.execute("SELECT payload FROM review_queue WHERE resolved=0 AND reason IN ('low_margin','no_candidate')")]
    assert all(abs(x["p_right"] - 0.731) < 0.01 and x["top"] for x in q)
