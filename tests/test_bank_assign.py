"""Bank + assign on a synthetic two-episode season: E1 labels -> bank -> E2 runs, incl. a mixed run to split,
a mislabelled utterance to drop, an unbanked speaker, and explicit-label monitoring."""

import json

import numpy as np
import pandas as pd
import pytest

from survspk.config import load_settings
from survspk.db import init_db
from survspk.stage_assign import Scorer, Thresholds, assign_episode, decide, split_run
from survspk.stage_bank import (BankEntry, build_bank, collect_labelled, farthest_point_sample, load_bank,
                                self_consistency_filter)
from survspk.stage_embed import embedding_path

D = 16
SPK = ["S_A", "S_B", "S_C", "S_D"]          # S_D speaks in E2 but never has a label -> unbankable


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class Season:
    """Builds utterances + parquet for episodes on the fly."""

    def __init__(self, s, con, seed=0):
        self.s, self.con = s, con
        self.rng = np.random.default_rng(seed)
        self.centers = {k: self.rng.normal(size=D) for k in SPK}
        self.i = {}

    def vec(self, spk, noise=0.25):
        return _unit(self.centers[spk] + self.rng.normal(scale=noise, size=D))

    def episode(self, ep, runs):
        """runs: list of dicts {spk (true voice) or spks (per utt), label (sdh name or None), explicit (bool),
        n, dur_each, domain, segment}. Writes utterances + parquet; returns utt ids per run."""
        rows, vecs, out_ids = [], [], []
        i, t, rid = 0, 100.0, 0
        for r in runs:
            spks = r.get("spks") or [r["spk"]] * r["n"]
            n = len(spks)
            ids = []
            for j, spk in enumerate(spks):
                uid = f"US99_E{ep:02d}_U{i:04d}"
                dur = r.get("dur_each", 3.0)
                flags = {"run_id": rid, "run_dur_s": n * dur, "name_inherited": not (r.get("explicit", True) and j == 0)}
                label = r.get("label")
                rows.append((uid, "US99", ep, i, t, t + dur, r.get("text", "blah blah"), r.get("segment", "body"), 1,
                             label, label, 0, r.get("domain", "confessional"), 1, json.dumps(flags),
                             "cast" if label else None))
                if dur >= self.s.embed.min_duration_s:      # embed skips utterances too short to embed
                    vecs.append({"utt_id": uid, "vector": self.vec(spk), "snr_proxy": 1.0, "start_s": t, "end_s": t + dur})
                ids.append(uid)
                t += dur + 0.3
                i += 1
            t += 5.0
            rid += 1
            out_ids.append(ids)
        self.con.executemany(
            """INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, is_speech,
               sdh_name, sdh_speaker_id, is_italic, domain_hint, align_ok, flags, sdh_resolution)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        self.con.commit()
        p = embedding_path(self.s, "US99", ep, "vocals")
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(vecs).to_parquet(p)
        return out_ids


@pytest.fixture
def season(tmp_path, monkeypatch):
    monkeypatch.setenv("SURVSPK_PROFILE", "macos")
    load_settings.cache_clear()
    s = load_settings()
    s.paths.work_root = tmp_path
    con = init_db(s.db_path, "DELETE")
    sea = Season(s, con)
    # E1: clean labelled confessionals for A, B, C (+ a field run each), one mislabelled run (voice B, label A)
    e1 = []
    for spk in ("S_A", "S_B", "S_C"):
        e1 += [{"spk": spk, "label": spk, "n": 3, "dur_each": 4.0} for _ in range(4)]
        e1 += [{"spk": spk, "label": spk, "n": 2, "dur_each": 2.0, "domain": "field"}]
    e1.append({"spk": "S_B", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "MISLABELLED"})
    sea.e1 = sea.episode(1, e1)
    yield s, con, sea
    load_settings.cache_clear()


def test_farthest_point_sample_spreads():
    X = np.stack([_unit(v) for v in np.eye(6)] + [_unit(np.eye(6)[0] + 0.01 * np.ones(6))])
    idx = farthest_point_sample(X, 6, seed_idx=0)
    assert len(idx) == 6 and len(set(idx)) == 6 and 6 not in idx          # the near-duplicate of row 0 is skipped


def test_consistency_filter_drops_the_mislabelled_run(season):
    s, con, sea = season
    df = collect_labelled(s, con, "US99", [1], "vocals")
    assert set(df.label_source) == {"sdh", "sdh_run"} and df.speaker_id.nunique() == 3
    assert len(df) == 3 * (4 * 3 + 2) + 2                 # every utterance of an anchored run counts
    kept, dropped = self_consistency_filter(df)
    assert set(dropped.text) == {"MISLABELLED"} and len(dropped) == 2
    assert (dropped.pred == "S_B").all() and (dropped.speaker_id == "S_A").all()
    assert len(kept) == len(df) - 2


def test_build_and_load_bank(season):
    s, con, sea = season
    st = build_bank(s, con, "US99", [1], variant="vocals")
    assert st["n_speakers"] == 3 and st["n_dropped"] == 2 and st["as_of"] == 1
    bank, used = load_bank(con, "US99", as_of=1)
    assert used == 1 and set(bank) == {(k, d) for k in ("S_A", "S_B", "S_C") for d in ("confessional", "field")}
    e = bank[("S_A", "confessional")]
    assert e.n_utts == 12 and e.exemplars.shape[1] == D and abs(np.linalg.norm(e.centroid) - 1) < 1e-5
    assert e.score(sea.centers["S_A"] / np.linalg.norm(sea.centers["S_A"])) > 0.9
    assert e.score(_unit(sea.centers["S_B"]), "exemplar") < 0.6
    # nothing stored for a later as_of; asking for as_of=0 finds nothing
    assert load_bank(con, "US99", as_of=0) == ({}, None)


def test_decide_thresholds():
    th = Thresholds(accept=0.55, margin=0.08, floor=0.35)
    assert decide([("A", 0.7), ("B", 0.4)], th).decision == "auto"
    assert decide([("A", 0.7), ("B", 0.66)], th).decision == "low_margin"
    assert decide([("A", 0.5), ("B", 0.1)], th).decision == "low_margin"
    assert decide([("A", 0.3), ("B", 0.1)], th).decision == "no_candidate"
    assert decide([], th).decision == "no_bank"


def test_split_run_cuts_at_speaker_change(season):
    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    bank, _ = load_bank(con, "US99", 1)
    sc = Scorer(bank, frozenset(SPK))
    th = Thresholds()
    vecs = [sea.vec("S_A") for _ in range(3)] + [sea.vec("S_C") for _ in range(2)]
    durs = [3.0] * 5
    assert split_run(vecs, durs, sc, "confessional", th) == [[0, 1, 2], [3, 4]]
    # a too-short or vector-less utterance never starts a group; it rides with its predecessor
    vecs2 = [sea.vec("S_A"), None, sea.vec("S_C", noise=0.2), sea.vec("S_C")]
    assert split_run(vecs2, [3.0, 0.4, 0.8, 3.0], sc, "confessional", th) == [[0, 1, 2], [3]]
    # pure run stays whole
    assert split_run([sea.vec("S_B") for _ in range(4)], [2.0] * 4, sc, "confessional", th) == [[0, 1, 2, 3]]


def test_assign_episode_end_to_end(season):
    s, con, sea = season
    build_bank(s, con, "US99", [1], variant="vocals")
    e2 = [
        {"spk": "S_A", "label": "S_A", "n": 3, "dur_each": 4.0},                       # explicit, agrees
        {"spk": "S_B", "label": None, "n": 3, "dur_each": 4.0},                        # unlabelled -> auto S_B
        {"spk": "S_C", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "WRONGNAME"},  # explicit but wrong -> conflict
        {"spks": ["S_A", "S_A", "S_C", "S_C"], "label": "S_A", "n": 4, "dur_each": 3.0, "text": "MIXED"},
        {"spk": "S_D", "label": None, "n": 2, "dur_each": 4.0, "text": "UNBANKED"},    # no bank entry -> low/none
        {"spk": "S_B", "label": None, "n": 1, "dur_each": 0.3, "text": "TINY"},        # no vector -> too_short
        {"spk": "S_B", "label": None, "n": 2, "dur_each": 4.0, "segment": "recap"},
    ]
    ids = sea.episode(2, e2)
    df, st = assign_episode(s, con, "US99", 2, variant="vocals", resolver=None, write=True)
    assert st["bank_as_of"] == 1 and st["n_bankable"] == 3
    by_run = {(r.run_id, r.sub): r for r in df.itertuples()}
    assert by_run[(0, 0)].pred == "S_A" and by_run[(0, 0)].decision == "auto" and by_run[(0, 0)].explicit == "S_A"
    assert by_run[(1, 0)].pred == "S_B" and by_run[(1, 0)].decision == "auto"
    assert by_run[(2, 0)].pred == "S_C" and by_run[(2, 0)].explicit == "S_A"        # monitored disagreement
    assert by_run[(3, 0)].split and by_run[(3, 1)].pred == "S_C" and by_run[(3, 0)].pred == "S_A"
    assert by_run[(4, 0)].decision in ("low_margin", "no_candidate")
    assert by_run[(5, 0)].decision == "too_short"
    assert by_run[(6, 0)].segment == "recap" and by_run[(6, 0)].pred == "S_B"
    # monitoring: 3 explicit runs in the body (runs 0, 2, 3-sub0), 2 agree
    assert st["n_sdh_runs"] == 3 and abs(st["sdh_agreement_runs"] - 2 / 3) < 1e-3
    # labels: explicit ones are 'sdh' (kept even when wrong), unlabelled confident ones 'auto'
    lab = pd.read_sql_query("SELECT * FROM labels", con).set_index("utt_id")
    assert lab.loc[ids[0][0], "source"] == "sdh" and lab.loc[ids[2][0], "speaker_id"] == "S_A"
    assert lab.loc[ids[1][0], "source"] == "auto" and lab.loc[ids[1][0], "speaker_id"] == "S_B"
    assert lab.loc[ids[3][3], "source"] == "auto" and lab.loc[ids[3][3], "speaker_id"] == "S_C"   # split half
    assert ids[4][0] not in lab.index and ids[5][0] not in lab.index
    q = pd.read_sql_query("SELECT * FROM review_queue", con)
    assert set(q[q.utt_id.isin(ids[2])].reason) == {"sdh_conflict"}
    assert set(q[q.utt_id.isin(ids[4])].reason) <= {"low_margin", "no_candidate"} and len(q[q.utt_id.isin(ids[4])]) == 2
    # re-running replaces rather than duplicates
    assign_episode(s, con, "US99", 2, variant="vocals", resolver=None, write=True)
    assert len(pd.read_sql_query("SELECT * FROM labels", con)) == len(lab)
    m = pd.read_sql_query("SELECT key, value FROM metrics WHERE version_season='US99' AND episode=2", con)
    assert "assign_sdh_agreement_runs" in set(m.key)
    # bank for E3 can now include E2's auto labels
    df2 = collect_labelled(s, con, "US99", [1, 2], "vocals")
    assert (df2.label_source == "auto").sum() >= 3


def test_mention_rule_splits_mixed_run_and_queues_self_naming(season):
    from tests.test_mentions import FakeResolver

    s, con, sea = season

    class R(FakeResolver):
        def __init__(self):
            super().__init__(s)
            self._rows = [
                {"castaway_id": "S_A", "castaway": "Andy", "full_name": "Andy Rueda", "full_name_detailed": None, "last_name": "Rueda"},
                {"castaway_id": "S_B", "castaway": "Gabe", "full_name": "Gabe Ortis", "full_name_detailed": None, "last_name": "Ortis"},
                {"castaway_id": "S_C", "castaway": "Sam", "full_name": "Sam Phalen", "full_name_detailed": None, "last_name": "Phalen"},
            ]
            self.season_aliases = {}

        def present(self, vs, ep):
            return frozenset({"S_A", "S_B", "S_C", "S_D"})

    build_bank(s, con, "US99", [1], variant="vocals")
    e2 = [
        # host-style question naming Gabe (voice: A, standing in for Jeff) glued to Gabe's answer (voice B)
        {"spks": ["S_A", "S_B", "S_B", "S_B"], "label": None, "n": 4, "dur_each": 4.0,
         "text": "So, Gabe, is this a surprise?"},
        # a single-speaker run that names its own predicted speaker -> queue whole
        {"spk": "S_C", "label": None, "n": 2, "dur_each": 4.0, "text": "With Sam, I am telling him everything."},
        # a clean run -> untouched
        {"spk": "S_A", "label": None, "n": 2, "dur_each": 4.0, "text": "blah"},
    ]
    ids = sea.episode(2, e2)
    # the fixture gives every utterance of a run the same text; make only the first utterance of run 0 the question
    con.execute("UPDATE utterances SET text='Look, the game says here is an island.' WHERE utt_id IN (?,?,?)", tuple(ids[0][1:]))
    con.commit()
    df, st = assign_episode(s, con, "US99", 2, variant="vocals", resolver=R(), write=True)
    r0 = df[df.run_id == 0].sort_values("sub")
    # the question is cut out and scored on its own: it is voice A, so it gets A's label rather than the queue
    assert len(r0) == 2 and r0.iloc[0].n_utts == 1 and r0.iloc[0].pred == "S_A" and r0.iloc[0].decision == "auto"
    assert r0.iloc[1].pred == "S_B" and r0.iloc[1].decision == "auto" and r0.iloc[1].n_utts == 3
    r1 = df[df.run_id == 1]
    assert len(r1) == 1 and r1.iloc[0].decision == "name_mentioned" and r1.iloc[0].pred == "S_C"
    assert df[df.run_id == 2].iloc[0].decision == "auto"
    assert st["n_name_mentioned"] == 1
    q = pd.read_sql_query("SELECT * FROM review_queue", con)
    assert set(q[q.utt_id.isin(ids[1])].reason) == {"name_mentioned"} and ids[0][0] not in set(q.utt_id)
    lab = pd.read_sql_query("SELECT * FROM labels", con).set_index("utt_id")
    assert lab.loc[ids[0][1], "speaker_id"] == "S_B" and lab.loc[ids[0][0], "speaker_id"] == "S_A"
