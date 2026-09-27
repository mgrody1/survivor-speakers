"""Calibrated auto-label confidence: the chance that a run's top voice match is the right speaker.

`assign` accepts a run when its best bank score is >= accept and beats the runner-up by >= margin. That rule sees two
numbers. This model sees what else is known at the same moment: the third score, how long the run is, how much speech
the bank has for the top two, confessional or field, whether the run says the predicted speaker's name, how far into
the season, and the context (explicit caption names on other runs nearby, name cards for the top two within 90 s,
whether the caption-named line just before or after is the predicted speaker).

It learns from a replay (survspk/replay.py): each season's assign re-run in memory, episode by episode, from the bank
of the episodes before, so every run is scored the way assign scored it, blind to its own labels. The runs whose
speaker is known (a person's label, a name card, an explicit caption name) are the examples. Checked by leaving one
season out at a time.

The model is a small L2-regularised logistic regression (numpy), stored as JSON with its features, scaling,
coefficients and the probability cut-off that matches the current rule's precision.

    survspk calibrate                 # replay, fit, cross-check by season, print the comparison (read-only)
    survspk calibrate --write         # ... and save work_root/models/calibrator.json
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["top1", "top2", "top3", "margin", "gap23", "n_cands", "log_dur", "log_n_utts", "confessional", "recap",
        "mention_pred", "host_pred", "log_bank_n", "log_bank_s", "log_bank_n2", "log_ep", "top1_x_margin"]
# what else is known when assign runs: explicit caption names on *other* runs nearby, and name cards
CONTEXT = ["cap_pred_near", "cap_second_near", "cap_prev_pred", "cap_next_pred", "card_pred_near", "card_second_near",
           # whether the episode has caption names / name cards at all: without them the context above is always 0, and
           # that must not read as evidence against the match (captionless episodes lost ~6 points of speech otherwise)
           "ep_has_caps", "ep_has_cards"]
FEATURES = BASE + CONTEXT
NEAR_S, TURN_S, CARD_S = 60.0, 30.0, 90.0


class EpisodeContext:
    """An episode's explicit caption names (NAME: lines) and name-card hits, for the context features."""

    def __init__(self, con: sqlite3.Connection, vs: str, ep: int):
        cap = [(r[0], r[1], r[2], r[3]) for r in con.execute(
            """SELECT utt_id, start_s, end_s, sdh_speaker_id, flags, sdh_resolution FROM utterances
               WHERE version_season=? AND episode=? AND sdh_speaker_id IS NOT NULL""", (vs, ep))
            if r[5] in ("cast", "alias", "host") and not json.loads(r[4] or "{}").get("name_inherited", True)]
        self.cap_id = np.array([c[0] for c in cap], dtype=object)
        self.cap_t0 = np.array([c[1] for c in cap], dtype=float)
        self.cap_t1 = np.array([c[2] for c in cap], dtype=float)
        self.cap_spk = np.array([c[3] for c in cap], dtype=object)
        try:
            cards = con.execute("""SELECT t_s, castaway_id FROM chyron_hits WHERE version_season=? AND episode=?
                                   AND castaway_id IS NOT NULL""", (vs, ep)).fetchall()
        except sqlite3.OperationalError:
            cards = []
        self.card_t = np.array([c[0] for c in cards], dtype=float)
        self.card_spk = np.array([c[1] for c in cards], dtype=object)
        self.has_caps = float(len(cap) >= 20)
        self.has_cards = float(len(cards) > 0)

    def features(self, utt_ids, start: float, end: float, pred: str | None, second: str | None) -> dict:
        own = np.isin(self.cap_id, list(utt_ids)) if len(self.cap_id) else np.zeros(0, bool)
        t0, t1, spk = self.cap_t0[~own], self.cap_t1[~own], self.cap_spk[~own]
        near = (t1 >= start - NEAR_S) & (t0 <= end + NEAR_S)
        before = (t1 <= start) & (t1 >= start - TURN_S)
        after = (t0 >= end) & (t0 <= end + TURN_S)
        cn = (self.card_t >= start - CARD_S) & (self.card_t <= end + CARD_S)
        return {
            "cap_pred_near": float(bool(pred) and (spk[near] == pred).any()),
            "cap_second_near": float(bool(second) and (spk[near] == second).any()),
            "cap_prev_pred": float(bool(before.any()) and spk[before][np.argmax(t1[before])] == pred),
            "cap_next_pred": float(bool(after.any()) and spk[after][np.argmin(t0[after])] == pred),
            "card_pred_near": float(bool(pred) and (self.card_spk[cn] == pred).any()),
            "card_second_near": float(bool(second) and (self.card_spk[cn] == second).any()),
            "ep_has_caps": self.has_caps, "ep_has_cards": self.has_cards,
        }


def features(top: list, dur: float, n_utts: int, domain: str | None, segment: str | None, mention_pred: bool,
             bank_pred: tuple[int, float] | None, bank_2: tuple[int, float] | None, ep: int, host_id: str = "HOST_US",
             ctx: dict | None = None) -> dict:
    """One run's features. `top` is assign's [[speaker, score], ...] (up to 3), best first; `ctx` the
    EpisodeContext features (zeros when not given)."""
    s = [float(x[1]) for x in top] + [0.0, 0.0, 0.0]
    pred = top[0][0] if top else None
    bp, b2 = bank_pred or (0, 0.0), bank_2 or (0, 0.0)
    return {
        "top1": s[0], "top2": s[1], "top3": s[2], "margin": s[0] - s[1], "gap23": s[1] - s[2], "n_cands": min(len(top), 3),
        "log_dur": math.log1p(max(dur, 0.0)), "log_n_utts": math.log1p(n_utts),
        "confessional": float(domain == "confessional"), "recap": float(segment == "recap"),
        "mention_pred": float(bool(mention_pred)), "host_pred": float(bool(pred) and pred.startswith("HOST")),
        "log_bank_n": math.log1p(bp[0]), "log_bank_s": math.log1p(bp[1]), "log_bank_n2": math.log1p(b2[0]),
        "log_ep": math.log(max(ep, 1)), "top1_x_margin": s[0] * (s[0] - s[1]),
        **{k: float((ctx or {}).get(k, 0.0)) for k in CONTEXT},
    }


# ---------------------------------------------------------------------------------------------- the model
@dataclass
class Calibrator:
    features: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    threshold: float | None = None          # accept at p >= threshold (matches the rule's precision when fit)
    meta: dict = field(default_factory=dict)
    audit: dict | None = None               # the audit ranker (AUDIT_FEATURES), a Calibrator as a dict

    def audit_prob(self, feats: dict) -> float | None:
        """Chance an audited group's auto label is right, from this model's p and the group's own length and scores."""
        if not self.audit:
            return None
        p = min(max(self.prob(feats), 1e-4), 1 - 1e-4)
        return Calibrator(**self.audit).prob({**feats, "lp": math.log(p / (1 - p))})

    def prob(self, feats: dict | pd.DataFrame) -> np.ndarray | float:
        if isinstance(feats, dict):
            x = np.array([feats[f] for f in self.features], dtype=float)
            z = ((x - np.array(self.mean)) / np.array(self.std)) @ np.array(self.coef) + self.intercept
            return float(1.0 / (1.0 + math.exp(-z)))
        X = (feats[self.features].to_numpy(float) - np.array(self.mean)) / np.array(self.std)
        return 1.0 / (1.0 + np.exp(-(X @ np.array(self.coef) + self.intercept)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls(**json.loads(Path(path).read_text()))


def fit(df: pd.DataFrame, y: np.ndarray, l2: float = 1.0, w: np.ndarray | None = None, iters: int = 50,
        feats: list[str] | None = None) -> Calibrator:
    """L2-regularised logistic regression by Newton steps on standardised features."""
    feats = list(feats or FEATURES)
    X0 = df[feats].to_numpy(float)
    mu, sd = X0.mean(0), X0.std(0)
    sd[sd < 1e-9] = 1.0
    X = np.hstack([(X0 - mu) / sd, np.ones((len(X0), 1))])
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        g = X.T @ (w * (p - y)) + R @ b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + R
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-7:
            break
    return Calibrator(feats, mu.tolist(), sd.tolist(), b[:-1].tolist(), float(b[-1]))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rule_accepts(df: pd.DataFrame, accept: float, margin: float, auto_min_s: float) -> np.ndarray:
    return ((df.top1 >= accept) & (df.margin >= margin) & (df.mention_pred == 0) & (np.expm1(df.log_dur) >= auto_min_s)).to_numpy()


def threshold_for(p: np.ndarray, y: np.ndarray, dur: np.ndarray, precision: float) -> float:
    """The lowest cut-off whose accepted runs are right at least `precision` of the time (by speech duration)."""
    o = np.argsort(-p)
    ok = np.cumsum((y[o] == 1) * dur[o]) / np.maximum(np.cumsum(dur[o]), 1e-9)
    good = np.where(ok >= precision)[0]
    if not len(good):
        return 1.0
    return float(p[o][good.max()])


# ---------------------------------------------------------------------------------------------- the data
def model_path(settings) -> Path:
    return Path(settings.paths.work_root) / "models/calibrator.json"


def load_model(settings) -> Calibrator | None:
    p = model_path(settings)
    return Calibrator.load(p) if p.exists() else None


def bank_support(con: sqlite3.Connection, vs: str, ep: int) -> dict:
    """(speaker, domain) -> (n_utts, seconds) in the bank assign used for episode `ep` (latest as_of <= ep - 1)."""
    row = con.execute("SELECT MAX(as_of_episode) FROM speaker_bank WHERE version_season=? AND as_of_episode<=?", (vs, ep - 1)).fetchone()
    if not row or row[0] is None:
        return {}
    return {(r[0], r[1]): (int(r[2] or 0), float(r[3] or 0.0)) for r in con.execute(
        "SELECT speaker_id, domain, n_utts, total_dur_s FROM speaker_bank WHERE version_season=? AND as_of_episode=?", (vs, row[0]))}


def _support(b: dict, spk: str | None, domain: str | None):
    if not spk:
        return None
    return b.get((spk, domain)) or b.get((spk, "confessional")) or b.get((spk, "field"))


def group_features(con: sqlite3.Connection, resolver, vs: str, ep: int, utt_ids: list[str], top: list,
                   domain: str | None, bank: dict | None = None, ctx: EpisodeContext | None = None) -> dict | None:
    """The features of one auto-labelled group, from the database, as `collect` builds them."""
    top = _top(top)
    if not top or not utt_ids:
        return None
    rows = con.execute("SELECT start_s, end_s, text, segment FROM utterances WHERE utt_id IN (%s)" % ",".join("?" * len(utt_ids)),
                       list(utt_ids)).fetchall()
    if not rows:
        return None
    bank = bank_support(con, vs, ep) if bank is None else bank
    text = " ".join(r[2] or "" for r in rows)
    men = bool(resolver is not None and resolver.mentions(text, top[0][0], vs))
    ctx = ctx or EpisodeContext(con, vs, ep)
    second = top[1][0] if len(top) > 1 else None
    cf = ctx.features(utt_ids, min(r[0] for r in rows), max(r[1] for r in rows), top[0][0], second)
    return features(top, float(sum(r[1] - r[0] for r in rows)), len(rows), domain, rows[0][3], men,
                    _support(bank, top[0][0], domain), _support(bank, second, domain), ep, ctx=cf)


def _top(s) -> list:
    """assign's [[speaker, score], ...] from a label's top_candidates (JSON or list). Anything else (a split part's
    {"same_as_part": ...}, a note) is not a ranking: []."""
    if not s:
        return []
    try:
        v = json.loads(s) if isinstance(s, str) else s
    except ValueError:
        return []
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] and isinstance(x[1], (int, float)):
            out.append([x[0], float(x[1])])
    return out


# The groups audit mode shows are not assign's runs: they are what is left of a run once a person has labelled some of
# it, so their own length says a lot (short leftovers are often wrong) that the run-level model cannot see. A second,
# small model on the audit verdicts ranks the "likely errors": the run model's log-odds plus the group's length and scores.
AUDIT_FEATURES = ["lp", "log_dur", "top1", "margin"]


def audit_frame(con: sqlite3.Connection, resolver) -> pd.DataFrame:
    """One row per audited group (episode >= 2) with the run-model features and y = the verdict was confirm."""
    try:
        a = pd.read_sql_query("""SELECT utt_id, version_season AS vs, episode AS ep, group_key, pred_speaker, verdict, prev_label
                                 FROM audit_verdicts""", con)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    rows, cache = [], {}
    for _, g in a.groupby("group_key"):
        prev = [json.loads(p) for p in g.prev_label if p]
        prev = [p for p in prev if p.get("speaker_id") == g.pred_speaker.iloc[0] and p.get("top_candidates")]
        vs, ep = g.vs.iloc[0], int(g.ep.iloc[0])
        if not prev or ep < 2:
            continue
        if (vs, ep) not in cache:
            cache[(vs, ep)] = (bank_support(con, vs, ep), EpisodeContext(con, vs, ep))
        b, ctx = cache[(vs, ep)]
        f = group_features(con, resolver, vs, ep, g.utt_id.tolist(), prev[0]["top_candidates"], prev[0].get("domain"), b, ctx)
        if f:
            rows.append({"vs": vs, "ep": ep, "y": float(g.verdict.iloc[0] == "confirm"), **f})
    return pd.DataFrame(rows)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def evaluate_audit(T: pd.DataFrame, A: pd.DataFrame, l2: float = 1.0) -> dict:
    """Leave one season out: the run model fit without the season gives lp; the ranker fit on the other seasons'
    audit groups scores the season's. Reports the ranking and how many wrong labels the bottom 10% / 20% hold."""
    if A.empty or A.y.nunique() < 2:
        return {}
    A = A.reset_index(drop=True).copy()
    A["lp"] = np.nan
    for vs in A.vs.unique():
        tr = T[T.vs != vs]
        idx = A.index[A.vs == vs]
        A.loc[idx, "lp"] = _logit(fit(tr, tr.y.to_numpy(float), l2=l2).prob(A.loc[idx]))
    P = np.full(len(A), np.nan)
    for vs in A.vs.unique():
        tr, te = A[A.vs != vs], A[A.vs == vs]
        if tr.y.nunique() < 2:
            continue
        P[te.index] = fit(tr, tr.y.to_numpy(float), l2=3.0, feats=AUDIT_FEATURES).prob(te)
    ok = ~np.isnan(P)
    y, p = A.y.to_numpy()[ok], P[ok]
    o = np.argsort(p)
    return {"n": int(ok.sum()), "wrong": int((y == 0).sum()), "auc_top1": round(auc(A.top1.to_numpy()[ok], y), 3),
            "auc": round(auc(p, y), 3), "bottom10_wrong": int((y[o[:int(len(p) * .1)]] == 0).sum()),
            "bottom20_wrong": int((y[o[:int(len(p) * .2)]] == 0).sum())}


def replay_frame(settings, con: sqlite3.Connection, resolver, seasons: list[str] | None = None, log=print) -> pd.DataFrame:
    """Every run of every season with at least two embedded episodes, replayed by assign from the bank of the
    episodes before it (survspk/replay.py), with this module's features and, where the speaker is known (a person's
    or name card's label, else an explicit caption name), `y` = the top match was right."""
    from .replay import replay

    if seasons is None:
        seasons = [r[0] for r in con.execute("""SELECT version_season FROM utterances GROUP BY 1
                                                HAVING COUNT(DISTINCT episode) >= 2 ORDER BY 1""")]
    frames = [replay(settings, con, resolver, vs, log=log) for vs in seasons]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if df.empty:
        return df
    ctxs: dict = {}
    rows = []
    for r in df.itertuples():
        top = json.loads(r.top)
        second = top[1][0] if len(top) > 1 else None
        if (r.vs, r.ep) not in ctxs:
            ctxs[(r.vs, r.ep)] = EpisodeContext(con, r.vs, r.ep)
        cf = ctxs[(r.vs, r.ep)].features(r.utt_ids.split(), r.start_s, r.end_s, r.pred, second)
        rows.append(features(top, r.dur, r.n_utts, r.domain, r.segment, bool(r.mention_pred), (r.bank_n, r.bank_s),
                             (r.bank_n2, 0.0), r.ep, ctx=cf))
    fx = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df.drop(columns=[c for c in fx.columns if c in df.columns]), fx], axis=1)
    df["y"] = np.where(df.truth.notna(), (df.pred == df.truth).astype(float), np.nan)
    return df


QUEUED = ("low_margin", "no_candidate", "name_mentioned")


def evaluate(df: pd.DataFrame, accept: float = 0.55, margin: float = 0.08, auto_min_s: float = 2.0, l2: float = 1.0) -> dict:
    """Leave one season out on the replayed runs whose speaker is known. Reports: ranking (AUC) against the top
    score alone; among runs the rule accepts, how many of the wrong ones the calibrator's bottom tenth holds; the
    share of speech accepted at the rule's own precision (cut-off picked on the training seasons); and, on the runs
    assign queues, how well the probability matches reality (what the queue's suggestions show)."""
    T = df[df.y.notna()].reset_index(drop=True)
    P = np.full(len(T), np.nan)
    per, cov = [], []
    for vs in sorted(T.vs.unique()):
        tr, te = T[T.vs != vs], T[T.vs == vs]
        if tr.empty or tr.y.nunique() < 2:
            continue
        m = fit(tr, tr.y.to_numpy(float), l2=l2)
        P[te.index] = m.prob(te)
        ra = rule_accepts(tr, accept, margin, auto_min_s)
        prec = float((tr.y * tr.dur)[ra].sum() / max(tr.dur[ra].sum(), 1e-9))
        t = threshold_for(m.prob(tr), tr.y.to_numpy(), tr.dur.to_numpy(), prec)
        rr, cc = rule_accepts(te, accept, margin, auto_min_s), P[te.index] >= t
        c = (te.dur.sum(), te.dur[rr].sum(), (te.y * te.dur)[rr].sum(), te.dur[cc].sum(), (te.y * te.dur)[cc].sum())
        cov.append(c)
        per.append({"vs": vs, "runs": len(te), "threshold": round(t, 3),
                    "rule_cov": round(c[1] / c[0], 3), "rule_prec": round(c[2] / max(c[1], 1e-9), 3),
                    "cal_cov": round(c[3] / c[0], 3), "cal_prec": round(c[4] / max(c[3], 1e-9), 3),
                    "auc_top1": round(auc(te.top1.to_numpy(), te.y.to_numpy()), 3),
                    "auc_cal": round(auc(P[te.index], te.y.to_numpy()), 3)})
    ok = ~np.isnan(P)
    y = T.y.to_numpy()
    out: dict = {"per_season": per, "n_known": int(ok.sum()), "right": round(float(y[ok].mean()), 3) if ok.any() else None,
                 "auc_top1": round(auc(T.top1.to_numpy()[ok], y[ok]), 3), "auc_cal": round(auc(P[ok], y[ok]), 3)}
    acc = rule_accepts(T, accept, margin, auto_min_s) & ok
    if acc.any():
        k = max(1, int(acc.sum() * 0.1))
        low = np.argsort(P[acc])[:k]
        out["accepted"] = {"n": int(acc.sum()), "wrong": int((y[acc] == 0).sum()),
                           "auc_top1": round(auc(T.top1.to_numpy()[acc], y[acc]), 3), "auc_cal": round(auc(P[acc], y[acc]), 3),
                           "bottom10_wrong": int((y[acc][low] == 0).sum())}
    if cov:
        tot, rd, ry, cd, cy = np.sum(cov, axis=0)
        out["pooled"] = {"rule_cov": round(rd / tot, 3), "rule_prec": round(ry / max(rd, 1e-9), 3),
                         "cal_cov": round(cd / tot, 3), "cal_prec": round(cy / max(cd, 1e-9), 3)}
    q = T.decision.isin(QUEUED).to_numpy() & ok
    if q.any():
        pq, yq = P[q], y[q]
        bins = [(0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, 1.01)]
        out["queued"] = {"n": int(q.sum()), "right": round(float(yq.mean()), 3),
                         "auc_top1": round(auc(T.top1.to_numpy()[q], yq), 3), "auc_cal": round(auc(pq, yq), 3),
                         "reliability": [[lo, int(((pq >= lo) & (pq < hi)).sum()), round(float(pq[(pq >= lo) & (pq < hi)].mean()), 3),
                                          round(float(yq[(pq >= lo) & (pq < hi)].mean()), 3)]
                                         for lo, hi in bins if ((pq >= lo) & (pq < hi)).any()]}
    return out


def score_labels(settings, model: Calibrator, resolver=None) -> int:
    """Fill labels.p_right for the auto labels already in the database (assign fills it for new ones). The only
    write this module makes; `survspk calibrate --write` calls it after saving the model."""
    from . import db as dbm

    con = dbm.init_db(settings.db_path, settings.sqlite_journal)        # applies the labels.p_right migration
    rows = pd.read_sql_query("""SELECT l.utt_id, l.top_candidates, l.domain, l.run_id, u.version_season AS vs, u.episode AS ep
                                FROM labels l JOIN utterances u USING (utt_id) WHERE l.source='auto'""", con)
    n = 0
    for (vs, ep), e in rows.groupby(["vs", "ep"]):
        bank, ctx = bank_support(con, vs, int(ep)), EpisodeContext(con, vs, int(ep))
        for _, g in e.groupby(["run_id", "top_candidates"], dropna=False):
            f = group_features(con, resolver, vs, int(ep), g.utt_id.tolist(), g.top_candidates.iloc[0], g.domain.iloc[0], bank, ctx)
            if f is None:
                continue
            p = round(model.prob(f), 4)
            con.executemany("UPDATE labels SET p_right=? WHERE utt_id=?", [(p, u) for u in g.utt_id])
            n += len(g)
    con.commit()
    return n


def run(settings, write: bool = False, l2: float = 1.0, seasons: list[str] | None = None, log=print) -> dict:
    from .aliases import Resolver

    th = settings.raw.get("thresholds", {})
    accept, margin, auto_min_s = float(th.get("accept", 0.55)), float(th.get("margin", 0.08)), float(th.get("auto_min_s", 2.0))
    con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    res = Resolver(settings) if settings.survivor_db_path.exists() else None
    df = replay_frame(settings, con, res, seasons, log=log)
    T = df[df.y.notna()].reset_index(drop=True)
    rep = {"n_runs": int(len(df)), "seasons": sorted(df.vs.unique().tolist())}
    rep.update(evaluate(df, accept, margin, auto_min_s, l2))
    m = fit(T, T.y.to_numpy(float), l2=l2)
    ra = rule_accepts(T, accept, margin, auto_min_s)
    prec = float((T.y * T.dur)[ra].sum() / max(T.dur[ra].sum(), 1e-9))
    m.threshold = threshold_for(m.prob(T), T.y.to_numpy(), T.dur.to_numpy(), prec)
    m.meta = {"n_known": int(len(T)), "rule_precision": round(prec, 4), "accept": accept, "margin": margin,
              "auto_min_s": auto_min_s, "seasons": rep["seasons"], "l2": l2}
    A = audit_frame(con, res)
    rep["audit"] = evaluate_audit(T, A, l2)
    if len(A) >= 50 and A.y.nunique() == 2 and (A.y == 0).sum() >= 10:
        A = A.assign(lp=_logit(m.prob(A)))
        m.audit = fit(A, A.y.to_numpy(float), l2=3.0, feats=AUDIT_FEATURES).__dict__
    rep["model"] = {"threshold": round(m.threshold, 4), "coef": dict(zip(FEATURES, [round(c, 3) for c in m.coef]))}
    if write:
        path = model_path(settings)
        m.save(path)
        rep["written"] = str(path)
        rep["scored_labels"] = score_labels(settings, m, res)
    return rep
