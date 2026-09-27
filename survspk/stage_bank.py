"""Speaker bank (spec §7.9, §8.8) — one entry per (season, speaker, domain, as_of_episode).

What goes in (M1 lessons, docs/M1_REPORT.md §4b):
  * only *explicitly* named SDH utterances (`NAME:` on the line, not inherited), plus human / chyron labels and
    confident `auto` labels from earlier episodes;
  * body segment, duration >= bank_min_duration_s;
  * a self-consistency filter: an utterance whose vector sits confidently in another speaker's region is dropped
    (that is label noise — a name that propagated across an unmarked speaker change — not a voice).
Each entry stores a recency-weighted centroid and K farthest-point exemplars.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings
from .stage_embed import _current_utts, embedding_path, loo_predict, parquet_is_fresh, staleness_reason

log = logging.getLogger(__name__)

DOMAINS = ("confessional", "field")


PSEUDO_SPEAKERS = frozenset({"UNKNOWN", "NOSPEECH", "OTHER"})


class StaleEmbeddings(RuntimeError):
    """The episode's embedding parquet does not match its utterances table (re-segmented or split since).
    Carries which episode, so a caller can re-embed that one (it need not be the episode being assigned)."""

    def __init__(self, msg: str, vs: str | None = None, ep: int | None = None, variant: str | None = None,
                 model: str | None = None):
        super().__init__(msg)
        self.vs, self.ep, self.variant, self.model = vs, ep, variant, model


@dataclass
class BankEntry:
    speaker_id: str
    domain: str
    centroid: np.ndarray            # L2-normalised [d]
    exemplars: np.ndarray           # L2-normalised [k, d]
    n_utts: int
    total_dur_s: float

    def score(self, v: np.ndarray, mode: str = "centroid") -> float:
        """Cosine of a unit vector against this entry. mode: centroid | exemplar | blend."""
        c = float(v @ self.centroid)
        if mode == "centroid" or len(self.exemplars) == 0:
            return c
        e = float((self.exemplars @ v).max())
        return e if mode == "exemplar" else 0.5 * (c + e)


# ----------------------------------------------------------------------------- collecting labelled vectors


def load_vectors(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str,
                 model: str | None = None) -> pd.DataFrame:
    """utt_id, vector (unit float32), snr_proxy for one episode; refuses stale parquets."""
    path = embedding_path(settings, vs, ep, variant, model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `survspk embed {vs} {ep} --variant {variant}`")
    why = staleness_reason(path, _current_utts(con, vs, ep, settings.embed.min_duration_s))
    if why:
        raise StaleEmbeddings(f"{vs} {path.name} is stale ({why}); re-run `survspk embed {vs} {ep} --variant {variant}`",
                              vs=vs, ep=ep, variant=variant, model=model)
    df = pd.read_parquet(path, columns=["utt_id", "vector", "snr_proxy"])
    df["vector"] = df.vector.apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


def load_utterances(con: sqlite3.Connection, vs: str, ep: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT utt_id, idx, start_s, end_s, end_s - start_s AS dur, text, segment, sdh_name, sdh_speaker_id,
                  sdh_resolution, is_italic, domain_hint, flags
           FROM utterances WHERE version_season=? AND episode=? ORDER BY idx""", con, params=(vs, ep))
    flags = df["flags"].apply(lambda f: json.loads(f) if f else {})
    df["run_id"] = flags.apply(lambda d: d.get("run_id", -1))
    df["run_dur"] = flags.apply(lambda d: d.get("run_dur_s", 0.0))
    df["name_explicit"] = flags.apply(lambda d: not d.get("name_inherited", True)) & df.sdh_name.notna()
    df["episode"] = ep
    return df.drop(columns=["flags"])


def collect_labelled(settings: Settings, con: sqlite3.Connection, vs: str, episodes: list[int], variant: str,
                     model: str | None = None, auto_min_conf: float | None = None, resolver=None) -> pd.DataFrame:
    """Labelled body utterances from the given episodes, with their vectors.

    label sources, in priority order: human > chyron > sdh (explicit or same-run) > auto (confidence >= auto_min_conf).
    With a resolver, sdh-labelled utterances whose text names their own speaker are excluded (mention rule; column
    `self_mention` marks them and they are returned separately by build_bank for inspection)."""
    thr = settings.raw.get("thresholds", {})
    if auto_min_conf is None:
        auto_min_conf = float(thr.get("accept", 0.55)) + 0.05
    frames = []
    for ep in episodes:
        u = load_utterances(con, vs, ep)
        v = load_vectors(settings, con, vs, ep, variant, model)
        df = u.merge(v, on="utt_id")
        lab = pd.read_sql_query("SELECT utt_id, speaker_id AS lab_speaker, source, confidence FROM labels", con)
        df = df.merge(lab, on="utt_id", how="left")
        df["speaker_id"], df["label_source"] = None, None
        # sdh: explicit NAME: lines, plus inherited names in the *same run* as an explicit line with that name
        # (the rest of a confessional after its first cue). Runs with no explicit name at all are not labels —
        # that is where names propagated across an unmarked speaker change (M1 §4b). The self-consistency filter
        # then catches a speaker change inside a run.
        ok = df.sdh_resolution.isin(["cast", "alias", "host"]) & df.sdh_speaker_id.notna()
        anchored = set(zip(df[ok & df.name_explicit].run_id, df[ok & df.name_explicit].sdh_speaker_id))
        m = ok & np.array([(r, sp) in anchored for r, sp in zip(df.run_id, df.sdh_speaker_id)], dtype=bool)
        df.loc[m, "speaker_id"], df.loc[m, "label_source"] = df.loc[m, "sdh_speaker_id"], "sdh"
        df.loc[m, "label_source"] = np.where(df.loc[m, "name_explicit"], "sdh", "sdh_run")
        # chyron: the anchored utterance (its label) plus the rest of its run, where nothing names someone else --
        # the only run-level label the `>>`-era seasons have. A run with two chyron names is left alone.
        chy: dict[int, str] = {}
        for r_, sp in zip(df[df.source == "chyron"].run_id, df[df.source == "chyron"].lab_speaker):
            chy[r_] = None if chy.get(r_, sp) != sp else sp
        run_spk = df.run_id.map({r_: sp for r_, sp in chy.items() if sp and r_ >= 0})
        m = df.speaker_id.isna() & run_spk.notna() & (df.sdh_speaker_id.isna() | (df.sdh_speaker_id == run_spk))
        df.loc[m, "speaker_id"], df.loc[m, "label_source"] = run_spk[m], "chyron_run"
        # auto (confident) fills gaps
        m = df.speaker_id.isna() & (df.source == "auto") & (df.confidence >= auto_min_conf)
        df.loc[m, "speaker_id"], df.loc[m, "label_source"] = df.loc[m, "lab_speaker"], "auto"
        # human / chyron override everything
        for src in ("chyron", "human"):
            m = df.source == src
            df.loc[m, "speaker_id"], df.loc[m, "label_source"] = df.loc[m, "lab_speaker"], src
        # "unknown", "no speech" and "other voice" are answers, not speakers: a bank entry for them pools unrelated
        # voices, and the consistency filter would drop real lines that happen to sit near that pool
        df = df[df.speaker_id.notna() & ~df.speaker_id.isin(PSEUDO_SPEAKERS) & (df.segment == "body")].copy()
        df["self_mention"] = False
        sdh = df.label_source.isin(["sdh", "sdh_run"])
        if resolver is not None and sdh.any():        # seasons without caption names (S21-39) have none to check
            df.loc[sdh, "self_mention"] = [resolver.mentions(t or "", spk, vs)
                                           for t, spk in zip(df.loc[sdh, "text"], df.loc[sdh, "speaker_id"])]
        frames.append(df.drop(columns=["lab_speaker", "source", "confidence"]))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out[out.dur >= settings.segment.bank_min_duration_s].reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- fitting


def self_consistency_filter(df: pd.DataFrame, tol: float = 0.10, min_per_speaker: int = 5,
                            min_pool_s: float = 15.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop labelled utterances whose vector is confidently closer to another speaker's leave-one-out centroid
    than to their own (sim_pred - sim_true > tol). Returns (kept, dropped).

    Two passes: the first pass finds misfits; centroids are then recomputed from what survived and every dropped
    utterance is re-tested against the cleaned centroids, because a speaker whose own pool contained a few wrong
    labels (Jeff's challenge calls under a castaway's name) has a polluted centroid in pass 1 and can lose genuine
    utterances to it. Speakers with a pool below min_per_speaker utterances or min_pool_s seconds are not judged
    at all — there is nothing reliable to judge them against — and are kept whole (US47E01: Sue had 3 utterances)."""
    if df.empty:
        return df, df.iloc[0:0]
    pools = df.groupby("speaker_id").agg(n=("utt_id", "size"), dur=("dur", "sum"))
    judged = pools[(pools.n >= min_per_speaker) & (pools.dur >= min_pool_s)].index
    testable = df[df.speaker_id.isin(judged)]
    if testable.speaker_id.nunique() < 2:
        return df, df.iloc[0:0]

    def misfits(frame: pd.DataFrame) -> pd.DataFrame:
        p = loo_predict(frame)
        bad = ((p.pred != frame.speaker_id) & ((p.sim - p.sim_true) > tol)).to_numpy()
        out = frame[bad].copy()
        out["pred"], out["sim_pred"], out["sim_true"] = p.pred[bad].to_numpy(), p.sim[bad].to_numpy(), p.sim_true[bad].to_numpy()
        return out

    d1 = misfits(testable)
    if d1.empty:
        return df, d1
    # pass 2: cleaned centroids; re-admit anything that fits its own speaker once the noise is gone
    clean = testable[~testable.utt_id.isin(d1.utt_id)]
    cents = {s: _unit(np.stack(g.vector.to_numpy()).mean(0)) for s, g in clean.groupby("speaker_id")}
    readmit = []
    for _, r in d1.iterrows():
        own = cents.get(r.speaker_id)
        if own is None:
            continue
        sims = {s: float(r.vector @ c) for s, c in cents.items()}
        best = max(sims, key=sims.get)
        if best == r.speaker_id or sims[best] - sims[r.speaker_id] <= tol:
            readmit.append(r.utt_id)
    dropped = d1[~d1.utt_id.isin(readmit)]
    kept = df[~df.utt_id.isin(dropped.utt_id)]
    return kept, dropped


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def farthest_point_sample(X: np.ndarray, k: int, seed_idx: int = 0) -> list[int]:
    """Greedy farthest-point sampling on unit vectors (cosine distance). Returns row indices."""
    n = len(X)
    if n <= k:
        return list(range(n))
    chosen = [seed_idx]
    dmin = 1 - X @ X[seed_idx]
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, 1 - X @ X[j])
    return chosen


def fit_entry(g: pd.DataFrame, speaker_id: str, domain: str, as_of: int, k: int, half_life: float,
              symmetric: bool = False) -> BankEntry:
    """`symmetric`: episodes after as_of age too (a bank for revisiting an episode once the season is done)."""
    X = np.stack(g.vector.to_numpy()).astype(np.float32)
    d = as_of - g.episode.to_numpy()
    age = np.abs(d) if symmetric else d.clip(min=0)
    w = (0.5 ** (age / half_life)) * g.dur.to_numpy()          # recency x duration
    c = (X * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    c = c / (np.linalg.norm(c) + 1e-9)
    seed = int(np.argmax(g.dur.to_numpy()))
    ex = X[farthest_point_sample(X, k, seed)]
    return BankEntry(speaker_id, domain, c.astype(np.float32), ex, len(g), float(g.dur.sum()))


def build_bank(settings: Settings, con: sqlite3.Connection, vs: str, episodes: list[int], variant: str | None = None,
               model: str | None = None, as_of: int | None = None, write: bool = True, resolver=None,
               use_auto: bool | None = None) -> dict:
    """Fit the bank for `vs` from the labelled utterances of `episodes`, stored as as_of_episode = as_of
    (default: max(episodes)). Returns stats incl. the dropped (inconsistent) utterances for inspection.

    `use_auto` (default: config bank.use_auto_labels, false): also learn from confident auto labels. Off by default
    because the bank is refit as a season goes on: an auto label the bank got wrong would become part of that
    player's voice and make the next wrong call likelier. Human, name-card and caption-name labels are always used."""
    t0 = time.time()
    variant = variant or settings.audio.variant
    model = model or settings.embed.model
    as_of = as_of if as_of is not None else max(episodes)
    bcfg = settings.raw.get("bank", {})
    k = int(bcfg.get("exemplars_per_speaker", 20))
    half_life = float(bcfg.get("recency_half_life_episodes", 3))
    if use_auto is None:
        use_auto = bool(bcfg.get("use_auto_labels", False))
    df = collect_labelled(settings, con, vs, episodes, variant, model, resolver=resolver,
                          auto_min_conf=None if use_auto else float("inf"))
    if df.empty:
        raise LookupError(f"no labelled utterances in {vs} episodes {episodes}")
    mentioned = df[df.self_mention].copy()
    df = df[~df.self_mention]
    kept, dropped = self_consistency_filter(df)
    if len(mentioned):
        mentioned["pred"], mentioned["sim_pred"], mentioned["sim_true"] = "(names self)", np.nan, np.nan
        dropped = pd.concat([dropped, mentioned], ignore_index=True)
    entries: list[BankEntry] = []
    for (spk, dom), g in kept.groupby(["speaker_id", "domain_hint"]):
        if dom not in DOMAINS:
            continue
        entries.append(fit_entry(g, spk, dom, as_of, k, half_life))
    borrowed: dict[str, str] = {}
    if bool(bcfg.get("borrow_host", True)):
        host_id = settings.franchise_for(vs).host_id
        con.row_factory = sqlite3.Row
        have = {e.domain for e in entries if e.speaker_id == host_id}
        for e, src in borrow_host_entries(con, vs, host_id, variant, model, have):
            entries.append(e)
            borrowed[e.domain] = src
        if borrowed:
            log.info("%s: host voice borrowed from %s", vs, borrowed)
    if write:
        con.execute("DELETE FROM speaker_bank WHERE version_season=? AND as_of_episode=?", (vs, as_of))
        for e in entries:
            ids = kept[(kept.speaker_id == e.speaker_id) & (kept.domain_hint == e.domain)].utt_id.tolist()
            con.execute(
                """INSERT INTO speaker_bank (version_season, speaker_id, domain, as_of_episode, centroid, exemplars,
                   n_utts, dim, variant, model, total_dur_s, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (vs, e.speaker_id, e.domain, as_of, e.centroid.tobytes(), e.exemplars.tobytes(), e.n_utts,
                 int(e.centroid.shape[0]), variant, model, e.total_dur_s,
                 json.dumps({"episodes": episodes, "utt_ids": ids, "k": int(len(e.exemplars)),
                            **({"borrowed_from": borrowed[e.domain]} if e.domain in borrowed and not ids else {})})))
        con.commit()
    per = (kept.groupby(["speaker_id", "domain_hint"]).agg(n=("utt_id", "size"), dur=("dur", "sum"))
               .reset_index().pivot(index="speaker_id", columns="domain_hint", values=["n", "dur"]).fillna(0))
    stats = {
        "version_season": vs, "episodes": episodes, "as_of": as_of, "variant": variant, "model": model,
        "n_labelled": int(len(df) + len(mentioned)), "n_kept": int(len(kept)), "n_dropped": int(len(dropped)),
        "n_self_mention": int(len(mentioned)),
        "n_speakers": int(kept.speaker_id.nunique()), "n_entries": len(entries), "host_borrowed": borrowed,
        "use_auto": use_auto,
        "sources": df.label_source.value_counts().to_dict(), "per_speaker": per, "dropped": dropped,
        "seconds": round(time.time() - t0, 1),
    }
    log.info("%s bank as of E%02d: %d speakers, %d entries from %d labelled utts (%d dropped as inconsistent) in %.1fs",
             vs, as_of, stats["n_speakers"], len(entries), len(df), len(dropped), stats["seconds"])
    return stats


def borrow_host_entries(con: sqlite3.Connection, vs: str, host_id: str, variant: str, model: str,
                        have: set[str]) -> list[tuple[BankEntry, str]]:
    """The host's voice is the same in every season: for each domain this season's labels did not give him an entry
    (S21-39 captions never name him and he gets no name card), take the entry with the most utterances from another
    season of the franchise, same audio variant and embedding model. Returns (entry, season it came from)."""
    out = []
    for dom in DOMAINS:
        if dom in have:
            continue
        r = con.execute("""SELECT * FROM speaker_bank WHERE speaker_id=? AND domain=? AND version_season<>? AND variant=? AND model=?
                           ORDER BY n_utts DESC, as_of_episode DESC LIMIT 1""", (host_id, dom, vs, variant, model)).fetchone()
        if r is None:
            continue
        d = int(r["dim"])
        c = np.frombuffer(r["centroid"], dtype=np.float32)
        ex = np.frombuffer(r["exemplars"], dtype=np.float32).reshape(-1, d) if r["exemplars"] else np.zeros((0, d), np.float32)
        out.append((BankEntry(host_id, dom, c, ex, int(r["n_utts"] or 0), float(r["total_dur_s"] or 0.0)), r["version_season"]))
    return out


def load_bank(con: sqlite3.Connection, vs: str, as_of: int | None = None) -> tuple[dict[tuple[str, str], BankEntry], int | None]:
    """Bank entries for `vs` at the latest as_of_episode <= as_of (or the latest overall)."""
    q = "SELECT MAX(as_of_episode) FROM speaker_bank WHERE version_season=?"
    args: tuple = (vs,)
    if as_of is not None:
        q += " AND as_of_episode<=?"
        args = (vs, as_of)
    row = con.execute(q, args).fetchone()
    if row is None or row[0] is None:
        return {}, None
    used = int(row[0])
    out: dict[tuple[str, str], BankEntry] = {}
    for r in con.execute("SELECT * FROM speaker_bank WHERE version_season=? AND as_of_episode=?", (vs, used)):
        d = int(r["dim"])
        c = np.frombuffer(r["centroid"], dtype=np.float32)
        ex = np.frombuffer(r["exemplars"], dtype=np.float32).reshape(-1, d) if r["exemplars"] else np.zeros((0, d), np.float32)
        out[(r["speaker_id"], r["domain"])] = BankEntry(r["speaker_id"], r["domain"], c, ex, int(r["n_utts"] or 0),
                                                        float(r["total_dur_s"] or 0.0))
    return out, used
