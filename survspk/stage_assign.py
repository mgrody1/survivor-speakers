"""Assign (spec §7.7) — label the utterances of an episode from the speaker bank.

Units are *runs* (monologues, see stage_segment.assign_runs), not utterances: a run's utterance vectors are pooled
(duration-weighted) before scoring, because per-utterance accuracy is dominated by utterance length (M1 §4).
Runs are not trusted to be pure — dash-convention subtitles do not mark a speaker change between cues — so each run
is first walked for a change of confidently-identified speaker and split there (M1 §4b).

Free labels first: an utterance with an explicit `NAME:` line keeps that label (source `sdh`); the run's pooled
prediction is compared against it for monitoring and a confident disagreement is queued as `sdh_conflict`.
Inherited names are treated as unlabelled and only reported (`inherited_agreement`).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import db as dbm
from .config import Settings
from .stage_bank import DOMAINS, BankEntry, load_bank, load_utterances, load_vectors

log = logging.getLogger(__name__)


@dataclass
class Thresholds:
    accept: float = 0.55
    margin: float = 0.08
    floor: float = 0.35
    purity_min_s: float = 1.0          # utterances shorter than this do not vote on a run's purity
    score_mode: str = "centroid"       # centroid | exemplar | blend
    auto_min_s: float = 2.0            # a group with less speech than this gets no auto label (audits: US47 2/9 right
                                       # under 2 s vs 62/66 above; US45 E01-02 most misses were sub-2 s)
    voice_split: bool = True           # cut out lines whose own voice clearly prefers someone else (two people
                                       # alternating inside one run; US45 E07 audit)
    voice_split_margin: float = 0.08   # how much more the line's voice must score its own best speaker than the group's
    voice_split_confident: bool = False  # True: the line must also be confidently someone else on its own. False won
                                         # on unreviewed US45 E03-13 / US47 E04-14: caption-name agreement on confident
                                         # runs 86.9 -> 90.1% / 89.1 -> 91.0%, auto share -2.5 points

    @classmethod
    def from_settings(cls, s: Settings) -> "Thresholds":
        t = dict(s.raw.get("thresholds", {}))
        b = s.raw.get("bank", {})
        return cls(accept=float(t.get("accept", 0.55)), margin=float(t.get("margin", 0.08)),
                   floor=float(t.get("floor", 0.35)), purity_min_s=float(t.get("purity_min_s", 1.0)),
                   score_mode=str(b.get("score_mode", "centroid")), auto_min_s=float(t.get("auto_min_s", 2.0)),
                   voice_split=bool(t.get("voice_split", True)),
                   voice_split_margin=float(t.get("voice_split_margin", 0.08)),
                   voice_split_confident=bool(t.get("voice_split_confident", False)))


@dataclass
class Scorer:
    bank: dict[tuple[str, str], BankEntry]
    candidates: frozenset[str]
    mode: str = "centroid"

    def entry(self, spk: str, domain: str) -> BankEntry | None:
        e = self.bank.get((spk, domain))
        if e is None:                                   # fall back to the other domain (spec §7.7.4)
            for d in DOMAINS:
                e = self.bank.get((spk, d))
                if e is not None:
                    break
        return e

    def bankable(self) -> frozenset[str]:
        return frozenset(s for s in self.candidates if self.entry(s, "confessional") is not None)

    def rank(self, v: np.ndarray, domain: str) -> list[tuple[str, float]]:
        out = []
        for spk in self.candidates:
            e = self.entry(spk, domain)
            if e is not None:
                out.append((spk, e.score(v, self.mode)))
        out.sort(key=lambda x: -x[1])
        return out


@dataclass
class Scored:
    pred: str | None
    score: float
    margin: float
    top: list[tuple[str, float]]
    decision: str                       # auto | low_margin | no_candidate | no_bank

    @property
    def confident(self) -> bool:
        return self.decision in ("auto", "auto_text")


def decide(rank: list[tuple[str, float]], th: Thresholds) -> Scored:
    if not rank:
        return Scored(None, 0.0, 0.0, [], "no_bank")
    top1 = rank[0][1]
    top2 = rank[1][1] if len(rank) > 1 else -1.0
    margin = top1 - top2
    if top1 < th.floor:
        d = "no_candidate"
    elif top1 >= th.accept and margin >= th.margin:
        d = "auto"
    else:
        d = "low_margin"
    return Scored(rank[0][0], top1, margin, rank[:3], d)


def pool(vecs: list[np.ndarray], durs: list[float]) -> np.ndarray:
    V = np.stack(vecs)
    w = np.asarray(durs, dtype=np.float32)[:, None]
    v = (V * w).sum(0) / max(float(w.sum()), 1e-9)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def split_run(vecs: list[np.ndarray | None], durs: list[float], scorer: Scorer, domain: str,
              th: Thresholds) -> list[list[int]]:
    """Walk a run's utterances in order and cut it wherever a confidently identified speaker differs from the
    confidently identified speaker of the group so far. Short or unconfident utterances never start a new
    group; utterances without a vector (too short to embed) follow their predecessor."""
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_spk: str | None = None
    for i, (v, d) in enumerate(zip(vecs, durs)):
        if not cur:
            cur = [i]
            cur_spk = None
        else:
            cur.append(i)
        if v is None or d < th.purity_min_s:
            continue
        s = decide(scorer.rank(v, domain), th)
        if not s.confident:
            continue
        if cur_spk is None:
            cur_spk = s.pred
        elif s.pred != cur_spk:
            cur.pop()                                    # this utterance starts a new group
            groups.append(cur)
            cur, cur_spk = [i], s.pred
    if cur:
        groups.append(cur)
    return groups


def _split_on_voice(groups: list[list[int]], vecs, durs, scorer: Scorer, domain: str, th: Thresholds) -> list[list[int]]:
    """Voice purity split. Within a group whose pooled voice is confidently one speaker, a line of at least
    purity_min_s whose own voice confidently names someone else, and scores that someone at least `margin` above the
    group's speaker, is another person's turn (two people alternating inside one caption run). Cut such lines out
    into their own contiguous groups; each is then scored alone, and queued if it is not confident on its own."""
    out: list[list[int]] = []
    for idxs in groups:
        vv = [vecs[i] for i in idxs if vecs[i] is not None]
        dd = [durs[i] for i in idxs if vecs[i] is not None]
        if len(idxs) < 2 or len(vv) < 2:
            out.append(idxs)
            continue
        g = decide(scorer.rank(pool(vv, dd), domain), th)
        if not (g.confident and g.pred):
            out.append(idxs)
            continue
        flags = []
        for i in idxs:
            v, d = vecs[i], durs[i]
            if v is None or d < th.purity_min_s:
                flags.append(None)                        # too short to judge: follows its neighbour
                continue
            rank = scorer.rank(v, domain)
            s_i = decide(rank, th)
            own = dict(rank).get(g.pred, 0.0)
            ok = s_i.confident or not th.voice_split_confident
            flags.append(bool(ok and s_i.pred and s_i.pred != g.pred and s_i.score - own >= th.voice_split_margin))
        if not any(f for f in flags if f):
            out.append(idxs)
            continue
        cur, cur_flag = [], None
        for i, f in zip(idxs, flags):
            f = cur_flag if f is None else f
            if cur and f != cur_flag:
                out.append(cur)
                cur = []
            cur.append(i)
            cur_flag = f
        out.append(cur)
    return out


def _split_on_mentions(groups: list[list[int]], g: pd.DataFrame, vecs, durs, scorer: Scorer, domain: str,
                       th: Thresholds, resolver, vs: str) -> list[list[int]]:
    """Text-driven purity split. If a confidently predicted group contains utterances that name the predicted
    speaker *and* utterances that do not, the naming utterances are almost certainly someone else's turn inside a
    merged run ("So, Gabe, is this a surprise...?" + Gabe's 15-second answer). Cut them out into their own
    contiguous groups so the mention rule queues only those and the rest keeps its label."""
    texts = g.text.tolist()
    out: list[list[int]] = []
    for idxs in groups:
        vv = [vecs[i] for i in idxs if vecs[i] is not None]
        dd = [durs[i] for i in idxs if vecs[i] is not None]
        if len(idxs) < 2 or not vv:
            out.append(idxs)
            continue
        sc = decide(scorer.rank(pool(vv, dd), domain), th)
        if not (sc.confident and sc.pred):
            out.append(idxs)
            continue
        flags = [resolver.mentions(texts[i] or "", sc.pred, vs) for i in idxs]
        if all(flags) or not any(flags):
            out.append(idxs)
            continue
        cur, cur_flag = [idxs[0]], flags[0]
        for i, f in zip(idxs[1:], flags[1:]):
            if f == cur_flag:
                cur.append(i)
            else:
                out.append(cur)
                cur, cur_flag = [i], f
        out.append(cur)
    return out


def _apply_text_prior(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, results: list[RunResult]) -> int:
    """Combine the LLM text prior (text_prior.py) with the audio decision, if enabled and available:
      * audio undecided (low_margin / no_candidate / name_mentioned) + confident text prior that is compatible with
        the audio ranking (in its top-2, or audio had no candidate at all) -> decision auto_text, label source 'text'.
      * anything the text prior says with low confidence, or that contradicts a confident audio call, changes nothing
        (the review UI shows it as a suggestion)."""
    tp_cfg = settings.raw.get("text_prior", {})
    if not tp_cfg.get("use_in_assign", False):
        return 0
    try:
        rows = con.execute("SELECT first_utt_id, speaker_id, confidence FROM text_prior WHERE version_season=? AND episode=?",
                           (vs, ep)).fetchall()
    except sqlite3.OperationalError:
        return 0
    tp = {r[0]: (r[1], float(r[2] or 0)) for r in rows}
    min_conf = float(tp_cfg.get("min_confidence", 0.8))
    n = 0
    for r in results:
        hit = tp.get(r.utt_ids[0]) if r.utt_ids else None
        if not hit or not hit[0] or hit[1] < min_conf:
            continue
        spk, conf = hit
        sc = r.scored
        if sc.decision not in ("low_margin", "no_candidate", "name_mentioned"):
            continue
        top2 = [s for s, _ in sc.top[:2]]
        compatible = sc.decision == "no_candidate" or spk in top2 or (sc.decision == "name_mentioned" and spk != sc.pred)
        if not compatible:
            continue
        audio_score = dict(sc.top).get(spk, sc.score)
        r.scored = Scored(spk, float(audio_score), sc.margin, sc.top, "auto_text")
        r.extra["text_conf"] = conf
        n += 1
    return n


@dataclass
class RunResult:
    run_id: int
    sub: int
    segment: str
    domain: str
    start_s: float
    dur: float
    utt_ids: list[str]
    scored: Scored
    explicit: str | None               # the one explicit SDH speaker in the group, if exactly one
    inherited: str | None
    n_explicit: int
    split: bool
    text: str = ""
    extra: dict = field(default_factory=dict)


def _apply_calibrator(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, results: list[RunResult],
                      bank: dict, utts: pd.DataFrame, resolver, th: Thresholds) -> None:
    """With a calibrator (`survspk calibrate --write`), give every scored run its chance of being right
    (`extra["p_right"]`, stored on auto labels). With thresholds.calibrated_accept, that chance also makes the call
    between auto and low_margin (p >= the cut-off fit to the rule's precision), in place of the accept/margin rule;
    the floor, the mention rule and the minimum length still apply."""
    from .calibrate import EpisodeContext, features, load_model

    cal = load_model(settings)
    if cal is None:
        return
    use = bool(settings.raw.get("thresholds", {}).get("calibrated_accept", False)) and cal.threshold is not None
    host_id = settings.franchise_for(vs).host_id
    ctx = EpisodeContext(con, vs, ep)
    U = utts.set_index("utt_id")

    def sup(spk, dom):
        if not spk:
            return None
        e = bank.get((spk, dom)) or bank.get((spk, "confessional")) or bank.get((spk, "field"))
        return (e.n_utts, e.total_dur_s) if e is not None else None

    for r in results:
        sc = r.scored
        if not sc.top or not sc.pred:
            continue
        top = [[s_, float(x)] for s_, x in sc.top]
        second = top[1][0] if len(top) > 1 else None
        g = U.loc[[u for u in r.utt_ids if u in U.index]]
        text = " ".join(t or "" for t in g.text)
        men = bool(resolver is not None and resolver.mentions(text, sc.pred, vs))
        cf = ctx.features(r.utt_ids, r.start_s, r.start_s + r.dur, sc.pred, second)
        f = features(top, float(g.dur.sum()), len(g), r.domain, r.segment, men, sup(sc.pred, r.domain), sup(second, r.domain),
                     ep, host_id, ctx=cf)
        p = cal.prob(f)
        r.extra["p_right"] = round(p, 4)
        if use and sc.decision in ("auto", "low_margin") and sc.score >= th.floor:
            ok = p >= cal.threshold and not men and float(g.dur.sum()) >= th.auto_min_s
            sc.decision = "auto" if ok else "low_margin"


def rolling_episodes(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str,
                     model: str | None = None) -> list[int]:
    """Episodes before `ep` with lines and embeddings: what a rolling bank for `ep` learns from."""
    from .stage_embed import embedding_path
    rows = con.execute("""SELECT episode FROM episodes WHERE version_season=? AND episode<? AND COALESCE(n_utterances, 0) > 0
                          ORDER BY episode""", (vs, ep)).fetchall()
    return [r[0] for r in rows if embedding_path(settings, vs, r[0], variant, model or settings.embed.model).exists()]


def assign_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str | None = None,
                   model: str | None = None, bank_as_of: int | None = None, resolver=None,
                   write: bool = True, rolling: bool | None = None,
                   bank_override: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Label an episode's runs from the bank. With `rolling` (default: config bank.rolling, on unless a bank_as_of is
    given), the bank is first refit from every earlier episode's trusted labels, so a season's later episodes learn
    from the caption names, name cards and human checks of the episodes before them. `bank_override`: score with this
    bank instead (the revisit pass, from every other episode of the season)."""
    t0 = time.time()
    variant = variant or settings.audio.variant
    th = Thresholds.from_settings(settings)
    if rolling is None:
        rolling = bank_as_of is None and bank_override is None and bool(settings.raw.get("bank", {}).get("rolling", True))
    rolled = None
    if rolling:
        prev = rolling_episodes(settings, con, vs, ep, variant, model)
        if prev:
            from .stage_bank import build_bank
            rolled = build_bank(settings, con, vs, prev, variant=variant, model=model, as_of=ep - 1, resolver=resolver)
    if bank_override is not None:
        bank, used = bank_override, -1                    # fit without this episode: scores are blind to its labels
    else:
        bank, used = load_bank(con, vs, bank_as_of if bank_as_of is not None else ep - 1)
    if not bank:
        raise LookupError(f"no speaker bank for {vs} (as of <= {bank_as_of if bank_as_of is not None else ep - 1}); "
                          f"run `survspk bank {vs} --episodes ...` first")
    host_id = settings.franchise_for(vs).host_id
    if resolver is not None:
        body_c = frozenset(resolver.present(vs, ep)) | {host_id}
        recap_c = frozenset(resolver.present(vs, ep - 1) or resolver.present(vs, ep)) | {host_id}
    else:                                                # no survivoR: everyone in the bank is a candidate
        body_c = recap_c = frozenset(s for s, _ in bank)
    scorers = {"body": Scorer(bank, body_c, th.score_mode), "preview": Scorer(bank, body_c, th.score_mode),
               "recap": Scorer(bank, recap_c, th.score_mode)}

    utts = load_utterances(con, vs, ep)
    vec = load_vectors(settings, con, vs, ep, variant, model).set_index("utt_id").vector
    utts["vector"] = utts.utt_id.map(vec)
    utts["has_vec"] = utts.vector.notna()

    results: list[RunResult] = []
    for (seg, rid), g in utts.groupby(["segment", "run_id"], sort=False):
        g = g.sort_values("idx")
        scorer = scorers.get(seg, scorers["body"])
        domain = g.domain_hint.iloc[0] if g.domain_hint.iloc[0] in DOMAINS else "field"
        vecs = [v if isinstance(v, np.ndarray) else None for v in g.vector]
        durs = g.dur.tolist()
        groups = split_run(vecs, durs, scorer, domain, th) if len(g) > 1 else [list(range(len(g)))]
        if th.voice_split:
            groups = _split_on_voice(groups, vecs, durs, scorer, domain, th)
        if resolver is not None:
            groups = _split_on_mentions(groups, g, vecs, durs, scorer, domain, th, resolver, vs)
        for sub, idxs in enumerate(groups):
            gg = g.iloc[idxs]
            vv = [vecs[i] for i in idxs if vecs[i] is not None]
            dd = [durs[i] for i in idxs if vecs[i] is not None]
            full_text = " ".join(t or "" for t in gg.text)
            if vv:
                sc = decide(scorer.rank(pool(vv, dd), domain), th)
                # mention rule: a run that names its predicted speaker is almost never that speaker
                # (US47E02 23:13 "With Sam, I am telling him..." -> Sam at 0.79; it is Andy). Queue it.
                if resolver is not None and sc.confident and sc.pred and resolver.mentions(full_text, sc.pred, vs):
                    sc.decision = "name_mentioned"
                # too little speech to trust an auto label: leave it unlabelled (not queued; nobody can tell either)
                if sc.confident and float(gg.dur.sum()) < th.auto_min_s:
                    sc.decision = "too_short"
            else:
                sc = Scored(None, 0.0, 0.0, [], "no_bank")
                sc.decision = "too_short"
            expl = gg[gg.name_explicit & gg.sdh_resolution.isin(["cast", "alias", "host"])].sdh_speaker_id.dropna().unique()
            inh = gg[~gg.name_explicit & gg.sdh_resolution.isin(["cast", "alias", "host"])].sdh_speaker_id.dropna().unique()
            results.append(RunResult(
                run_id=int(rid), sub=sub, segment=seg, domain=domain, start_s=float(gg.start_s.min()),
                dur=float(gg.end_s.max() - gg.start_s.min()), utt_ids=gg.utt_id.tolist(), scored=sc,
                explicit=str(expl[0]) if len(expl) == 1 else None, inherited=str(inh[0]) if len(inh) == 1 else None,
                n_explicit=int(len(expl)), split=len(groups) > 1,
                text=full_text[:120],
                extra={"n_utts": int(len(gg)), "n_vec": len(vv), "mixed_explicit": len(expl) > 1}))

    n_text = _apply_text_prior(settings, con, vs, ep, results)
    if used is not None and used < ep:       # a bank fit on this very episode (E01) makes every score rosy: rule only
        _apply_calibrator(settings, con, vs, ep, results, bank, utts, resolver, th)

    df = pd.DataFrame([{
        "run_id": r.run_id, "sub": r.sub, "segment": r.segment, "domain": r.domain, "start_s": round(r.start_s, 2),
        "mmss": f"{int(r.start_s // 60):02d}:{int(r.start_s % 60):02d}", "dur": round(r.dur, 2),
        "n_utts": r.extra["n_utts"], "split": r.split, "pred": r.scored.pred, "score": round(r.scored.score, 3),
        "margin": round(r.scored.margin, 3), "decision": r.scored.decision, "explicit": r.explicit,
        "inherited": r.inherited, "mixed_explicit": r.extra["mixed_explicit"], "text": r.text,
    } for r in results])

    # ---- monitoring (spec §7.7.6): explicit SDH labels scored as if unlabelled
    body = df[df.segment == "body"]
    mon = body[body.explicit.notna() & body.pred.notna()]
    mon_conf = mon[mon.decision == "auto"]
    inh = body[body.inherited.notna() & body.explicit.isna() & body.pred.notna()]
    bankable = scorers["body"].bankable()
    stats = {
        "version_season": vs, "episode": ep, "bank_as_of": used, "variant": variant,
        "bank_rolled_from": rolled["episodes"] if rolled else None,
        "n_candidates": len(body_c), "n_bankable": len(bankable),
        "unbankable": sorted(body_c - bankable),
        "n_runs": int(len(df)), "n_split_runs": int(df.split.sum()),
        "decisions": df.decision.value_counts().to_dict(),
        "body_auto_rate": round(float(body.decision.isin(["auto", "auto_text"]).mean()), 3) if len(body) else None,
        "body_auto_dur_share": round(float(body[body.decision.isin(["auto", "auto_text"])].dur.sum() / max(body.dur.sum(), 1e-9)), 3) if len(body) else None,
        "sdh_agreement_runs": round(float((mon.pred == mon.explicit).mean()), 3) if len(mon) else None,
        "n_sdh_runs": int(len(mon)),
        "sdh_agreement_confident": round(float((mon_conf.pred == mon_conf.explicit).mean()), 3) if len(mon_conf) else None,
        "n_sdh_runs_confident": int(len(mon_conf)),
        "inherited_agreement": round(float((inh.pred == inh.inherited).mean()), 3) if len(inh) else None,
        "n_inherited_runs": int(len(inh)),
    }
    for dom in DOMAINS:
        m = mon[mon.domain == dom]
        mc = m[m.decision == "auto"]
        stats[f"sdh_agreement_{dom}"] = round(float((m.pred == m.explicit).mean()), 3) if len(m) else None
        stats[f"n_sdh_runs_{dom}"] = int(len(m))
        stats[f"sdh_agreement_{dom}_confident"] = round(float((mc.pred == mc.explicit).mean()), 3) if len(mc) else None
        stats[f"n_sdh_runs_{dom}_confident"] = int(len(mc))
        # auto share of this domain's body time — coverage, the other half of the story
        b = body[body.domain == dom]
        stats[f"auto_dur_share_{dom}"] = round(float(b[b.decision.isin(["auto", "auto_text"])].dur.sum() / max(b.dur.sum(), 1e-9)), 3) if len(b) else None
    # where the queue goes: seconds of queued body time by predicted speaker (thin bank entries show up here)
    stats["n_name_mentioned"] = int((df.decision == "name_mentioned").sum())
    stats["n_auto_text"] = n_text
    q = body[body.decision.isin(["low_margin", "no_candidate", "name_mentioned"])]
    stats["queued_dur_by_pred"] = q.groupby("pred").dur.sum().round(0).sort_values(ascending=False).astype(int).to_dict()

    if write:
        _write(con, vs, ep, results, utts, th, stats)
    stats["seconds"] = round(time.time() - t0, 1)
    log.info("%s E%02d assign (bank as of E%02d): %d runs, auto %s of body time; SDH agreement %s on %d runs "
             "(%s on %d confident); %d unbankable candidates %s",
             vs, ep, used or 0, len(df), stats["body_auto_dur_share"], stats["sdh_agreement_runs"], len(mon),
             stats["sdh_agreement_confident"], len(mon_conf), len(stats["unbankable"]), stats["unbankable"])
    return df, stats


def _write(con: sqlite3.Connection, vs: str, ep: int, results: list[RunResult], utts: pd.DataFrame,
           th: Thresholds, stats: dict) -> None:
    ids = tuple(utts.utt_id)
    ph = ",".join("?" * len(ids))
    # human labels (and chyron) are never touched by a re-run; their queue rows stay resolved
    protected = {r[0] for r in con.execute(
        f"SELECT utt_id FROM labels WHERE source IN ('human','chyron') AND utt_id IN ({ph})", ids)}
    con.execute(f"DELETE FROM labels WHERE source IN ('auto','sdh','text') AND utt_id IN ({ph})", ids)
    con.execute(f"DELETE FROM review_queue WHERE resolved=0 AND utt_id IN ({ph})", ids)
    now = "datetime('now')"
    by_id = utts.set_index("utt_id")
    n_labels = n_queue = 0
    for r in results:
        sc = r.scored
        top = json.dumps([[s, round(float(x), 4)] for s, x in sc.top])
        for uid in r.utt_ids:
            if uid in protected:
                continue
            u = by_id.loc[uid]
            explicit_ok = bool(u.name_explicit) and u.sdh_resolution in ("cast", "alias", "host") and pd.notna(u.sdh_speaker_id)
            if explicit_ok:
                con.execute(f"""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain,
                                labeled_at, run_id, margin) VALUES (?,?,?,?,?,?,{now},?,?)""",
                            (uid, u.sdh_speaker_id, "sdh", 1.0, top, r.domain, r.run_id, sc.margin))
                n_labels += 1
                if sc.confident and sc.pred != u.sdh_speaker_id and r.dur >= 3.0:
                    con.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                                (uid, "sdh_conflict", json.dumps({"sdh": u.sdh_speaker_id, "pred": sc.pred,
                                                                  "score": round(sc.score, 3), "margin": round(sc.margin, 3),
                                                                  "run_id": r.run_id, "start_s": r.start_s})))
                    n_queue += 1
                continue
            if sc.confident:
                src = "text" if sc.decision == "auto_text" else "auto"
                conf = r.extra.get("text_conf", sc.score) if src == "text" else sc.score
                con.execute(f"""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain,
                                labeled_at, run_id, margin, p_right) VALUES (?,?,?,?,?,?,{now},?,?,?)""",
                            (uid, sc.pred, src, conf, top, r.domain, r.run_id, sc.margin, r.extra.get("p_right")))
                n_labels += 1
            elif sc.decision in ("low_margin", "no_candidate", "name_mentioned"):
                con.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                            (uid, sc.decision, json.dumps({"top": sc.top[:3], "run_id": r.run_id, "start_s": r.start_s,
                                                           "inherited": r.inherited})))
                n_queue += 1
    for k, v in stats.items():
        if isinstance(v, (int, float)) and v is not None:
            con.execute("INSERT OR REPLACE INTO metrics (version_season, episode, key, value, payload, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
                        (vs, ep, f"assign_{k}", float(v), None))
    con.execute("INSERT OR REPLACE INTO metrics (version_season, episode, key, value, payload, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
                (vs, ep, "assign_summary", stats.get("sdh_agreement_runs"), json.dumps({k: v for k, v in stats.items() if not isinstance(v, pd.DataFrame)}, default=str)))
    con.commit()
    dbm.backfill_label_spans(con)
    stats["n_labels_written"], stats["n_queued"] = n_labels, n_queue
