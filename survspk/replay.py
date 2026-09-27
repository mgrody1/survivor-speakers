"""Replay a season's assign in memory, episode by episode, to test changes to the bank and the scoring without
touching the database (it is opened read-only).

For each episode E >= 2 the bank is refit from episodes < E exactly as the rolling assign does (human, name-card and
caption-name labels), assign scores E's runs, and every run is written out with what a stacker could use:

  * the vocals scores (assign's own) and, for the other audio variants that exist (raw, center, vocals_center), the
    same run scored against a bank fit from that variant's embeddings
  * the true speaker where one is known (human or name-card label, else an explicit caption name)

With --self-train P, auto-labelled runs whose calibrated chance of being right is >= P (calibrator fit on the other
seasons) also join the bank for the next episodes, as if bank.use_auto_labels took only the surest ones.

`survspk calibrate` fits the calibrator on these runs; scripts/replay_assign.py writes them out for experiments.
Results (2026-09-27, 8 seasons): the other audio variants add little to a stacked model (AUC +0.006) and
self-training on p >= 0.93 / 0.97 auto labels leaves coverage and precision unchanged, so neither is used.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import stage_assign as sa
from .stage_bank import DOMAINS, borrow_host_entries, collect_labelled, fit_entry, self_consistency_filter
from .stage_embed import embedding_path


def vectors(settings, vs, ep, variant) -> dict[str, np.ndarray]:
    p = embedding_path(settings, vs, ep, variant)
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["utt_id", "vector"])
    return {u: np.asarray(v, dtype=np.float32) for u, v in zip(df.utt_id, df.vector)}


def bank_from(settings, con, vs, labelled: pd.DataFrame, as_of: int, variant: str, symmetric: bool = False) -> dict:
    b = settings.raw.get("bank", {})
    k, hl = int(b.get("exemplars_per_speaker", 20)), float(b.get("recency_half_life_episodes", 3))
    df = labelled[~labelled.self_mention]
    kept, _ = self_consistency_filter(df)
    bank = {}
    for (spk, dom), g in kept.groupby(["speaker_id", "domain_hint"]):
        if dom in DOMAINS:
            bank[(spk, dom)] = fit_entry(g, spk, dom, as_of, k, hl, symmetric=symmetric)
    host = settings.franchise_for(vs).host_id
    have = {d for (s_, d) in bank if s_ == host}
    for e, _ in borrow_host_entries(con, vs, host, variant, settings.embed.model, have):
        bank[(host, e.domain)] = e
    return bank


def revisit_bank(settings, con, res, vs: str, ep: int, variant: str = "vocals", cache: dict | None = None) -> dict:
    """A bank for re-scoring `ep` once the season is further along: every *other* episode's trusted labels (person,
    card, caption name), the nearest episodes weighted most. Built in memory; the stored rolling banks are untouched."""
    cache = {} if cache is None else cache
    eps = [r[0] for r in con.execute("SELECT DISTINCT episode FROM utterances WHERE version_season=? ORDER BY 1", (vs,))
           if r[0] != ep and embedding_path(settings, vs, r[0], variant).exists()]
    frames = []
    for e in eps:
        if (variant, e) not in cache:
            try:
                cache[(variant, e)] = collect_labelled(settings, con, vs, [e], variant, auto_min_conf=float("inf"), resolver=res)
            except Exception:  # noqa: BLE001 - a stale or missing episode is skipped
                cache[(variant, e)] = pd.DataFrame()
        if len(cache[(variant, e)]):
            frames.append(cache[(variant, e)])
    if not frames:
        return {}
    return bank_from(settings, con, vs, pd.concat(frames, ignore_index=True), ep, variant, symmetric=True)


def truth_map(con, vs, ep) -> dict[str, str]:
    t = {}
    for r in con.execute("""SELECT u.utt_id, u.sdh_speaker_id, u.flags, u.sdh_name, u.sdh_resolution FROM utterances u
                            WHERE u.version_season=? AND u.episode=?""", (vs, ep)):
        f = json.loads(r[2] or "{}")
        if r[1] and r[3] and not f.get("name_inherited", True) and r[4] in ("cast", "alias", "host"):
            t[r[0]] = r[1]
    for r in con.execute("""SELECT l.utt_id, l.speaker_id, l.source FROM labels l JOIN utterances u USING (utt_id)
                            WHERE u.version_season=? AND u.episode=? AND l.source IN ('chyron', 'human')
                            ORDER BY l.source='human'""", (vs, ep)):
        t[r[0]] = r[1]
    return t


def replay(settings, con, res, vs, variants=("vocals",), self_train=None, calib=None, log=print, bank_from_all: bool = False):
    eps = [r[0] for r in con.execute("SELECT DISTINCT episode FROM utterances WHERE version_season=? ORDER BY 1", (vs,))
           if embedding_path(settings, vs, r[0], "vocals").exists()]
    th = sa.Thresholds.from_settings(settings)
    host = settings.franchise_for(vs).host_id
    lab: dict = {}          # (variant, ep) -> labelled frame (trusted labels only)
    extra: dict = {}        # ep -> {utt_id: speaker} replay autos that joined the bank
    rows = []
    captured = {}
    import logging

    lg = logging.getLogger(sa.__name__)
    level = lg.level
    lg.setLevel(logging.WARNING)                             # one INFO summary per replayed episode is noise here
    orig_tp, orig_lb, orig_cal = sa._apply_text_prior, sa.load_bank, sa._apply_calibrator
    sa._apply_text_prior = lambda s_, c_, v_, e_, results: captured.__setitem__("r", results) or 0
    sa._apply_calibrator = lambda *a_, **k_: None           # the replay shows the rule's own decisions
    try:
        for ep in eps:
            if ep < 2 and not bank_from_all:
                continue
            # the rolling bank learns from the episodes before; bank_from_all (a second pass once the season is done)
            # from every other episode, nearest ones weighted most
            prev = [e for e in eps if e != ep] if bank_from_all else [e for e in eps if e < ep]
            banks = {}
            for v in variants:
                frames = []
                for e in prev:
                    if (v, e) not in lab:
                        try:
                            lab[(v, e)] = collect_labelled(settings, con, vs, [e], v, auto_min_conf=float("inf"), resolver=res)
                        except Exception:  # noqa: BLE001 - a variant missing or stale for one episode
                            lab[(v, e)] = pd.DataFrame()
                    f = lab[(v, e)]
                    if len(f) and extra.get(e):
                        vec = vectors(settings, vs, e, v)
                        add = [u for u in extra[e] if u in vec and u not in set(f.utt_id)]
                        if add:
                            base = pd.read_sql_query(
                                "SELECT utt_id, start_s, end_s, end_s - start_s AS dur, domain_hint, segment, text FROM utterances "
                                "WHERE utt_id IN (%s)" % ",".join("?" * len(add)), con, params=add)
                            base["vector"] = base.utt_id.map(vec)
                            base["speaker_id"] = base.utt_id.map(extra[e])
                            base["episode"], base["label_source"], base["self_mention"] = e, "auto", False
                            base = base[base.dur >= settings.segment.bank_min_duration_s]
                            f = pd.concat([f, base], ignore_index=True)
                    if len(f):
                        frames.append(f)
                if frames:
                    banks[v] = bank_from(settings, con, vs, pd.concat(frames, ignore_index=True), ep if bank_from_all else ep - 1, v,
                                         symmetric=bank_from_all)
            if "vocals" not in banks:
                continue
            sa.load_bank = lambda c_, v_, as_of=None, _b=banks["vocals"]: (_b, ep - 1)
            captured.pop("r", None)
            try:
                sa.assign_episode(settings, con, vs, ep, variant="vocals", bank_as_of=ep - 1, resolver=res, write=False)
            except Exception as ex:  # noqa: BLE001 - stale vocals parquet etc.
                log(f"{vs} E{ep:02d}: skipped ({type(ex).__name__}: {ex})")
                continue
            results = captured.get("r", [])
            body_c = frozenset(res.present(vs, ep)) | {host}
            recap_c = frozenset(res.present(vs, ep - 1) or res.present(vs, ep)) | {host}
            vv = {v: vectors(settings, vs, ep, v) for v in banks}
            ut = pd.read_sql_query("SELECT utt_id, start_s, end_s, end_s - start_s AS dur, text FROM utterances WHERE version_season=? AND episode=?",
                                   con, params=(vs, ep)).set_index("utt_id")
            truth = truth_map(con, vs, ep)
            extra[ep] = {}
            for r in results:
                sc = r.scored
                if not sc.top:
                    continue
                t = [truth[u] for u in r.utt_ids if u in truth]
                tr = max(set(t), key=t.count) if t else None
                cand = recap_c if r.segment == "recap" else body_c
                row = {"vs": vs, "ep": ep, "run_id": r.run_id, "sub": r.sub, "segment": r.segment, "domain": r.domain,
                       "start_s": r.start_s, "end_s": float(ut.loc[r.utt_ids].end_s.max()), "dur": float(ut.loc[r.utt_ids].dur.sum()),
                       "n_utts": len(r.utt_ids), "decision": sc.decision, "pred": sc.pred, "explicit": r.explicit,
                       "top": json.dumps([[s_, round(float(x), 4)] for s_, x in sc.top]), "truth": tr,
                       "truth_share": round(len(t) / len(r.utt_ids), 2),
                       "mention_pred": bool(res.mentions(" ".join(t_ or "" for t_ in ut.loc[r.utt_ids].text), sc.pred, vs)) if sc.pred else False,
                       "utt_ids": " ".join(r.utt_ids)}
                bp = banks["vocals"]
                ent = lambda b_, s_: (b_.get((s_, r.domain)) or b_.get((s_, "confessional")) or b_.get((s_, "field")))  # noqa: E731
                e1 = ent(bp, sc.pred)
                e2 = ent(bp, sc.top[1][0]) if len(sc.top) > 1 else None
                row["bank_n"], row["bank_s"] = (e1.n_utts, e1.total_dur_s) if e1 else (0, 0.0)
                row["bank_n2"] = e2.n_utts if e2 else 0
                for v in banks:
                    if v == "vocals":
                        continue
                    xs = [(vv[v][u], float(ut.loc[u].dur)) for u in r.utt_ids if u in vv[v]]
                    if not xs:
                        continue
                    rk = sa.Scorer(banks[v], cand, th.score_mode).rank(sa.pool([x for x, _ in xs], [d for _, d in xs]), r.domain)
                    if rk:
                        sp = dict(rk)
                        row[f"{v}_top1_spk"], row[f"{v}_top1"] = rk[0]
                        row[f"{v}_pred"] = sp.get(sc.pred)
                        row[f"{v}_second"] = max([x for s_, x in rk if s_ != sc.pred] or [np.nan])
                if calib is not None:
                    from .calibrate import EpisodeContext, features as _features
                    if "ctx" not in captured or captured.get("ctx_ep") != ep:
                        captured["ctx"], captured["ctx_ep"] = EpisodeContext(con, vs, ep), ep
                    tp = json.loads(row["top"])
                    cf = captured["ctx"].features(r.utt_ids, row["start_s"], row["end_s"], sc.pred, tp[1][0] if len(tp) > 1 else None)
                    f = _features(tp, row["dur"], row["n_utts"], r.domain, r.segment, row["mention_pred"],
                                  (row["bank_n"], row["bank_s"]), (row["bank_n2"], 0.0), ep, host, ctx=cf)
                    row["p"] = round(calib.prob(f), 4)
                    if self_train is not None and sc.decision == "auto" and row["p"] >= self_train and not r.explicit:
                        for u in r.utt_ids:
                            extra[ep][u] = sc.pred
                rows.append(row)
            log(f"{vs} E{ep:02d}: {len(results)} runs, {len(extra[ep])} self-trained lines")
    finally:
        sa._apply_text_prior, sa.load_bank, sa._apply_calibrator = orig_tp, orig_lb, orig_cal
        lg.setLevel(level)
    return pd.DataFrame(rows)


