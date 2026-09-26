"""Two voices inside one line: find where a line should be split.

Production path (`suggest_episode`): the diarizer's change track (survspk.stage_diarize) says where the voice changes
inside a line and how long the second voice lasts; the bank then embeds both sides and must name two different people.
Test on the 24 lines split by hand in US43-47 and 400 labelled one-speaker lines: diarizer >= 0.8 s and bank names
differ with contrast >= 0.2 finds 18/24 (cut within 1 s) and flags 6% of the one-speaker lines; the bank alone
(`propose`, every word gap tried) finds 16/24 and flags 3%.

Bank-only path (`propose`):

For a line long enough to hold two turns, every pause between words that leaves at least `min_side_s` on each side
is a candidate cut. Both sides are embedded and scored against the bank (cast present that episode + host). A cut is
proposed when the two sides prefer different speakers: the left side's speaker A over B plus the right side's B over
A (`contrast`) must clear the threshold, and each side's own speaker must reach `floor`. The best-contrast cut wins.

Named speakers come for free (the bank knows who is who), which an anonymous diarizer does not give us.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .stage_assign import Scorer
from .stage_bank import DOMAINS, load_bank


@dataclass
class SplitCfg:
    min_utt_s: float = 2.5        # shorter lines are not checked
    min_side_s: float = 1.0       # each side of a cut must hold this much audio
    floor: float = 0.35           # each side's own speaker must score at least this
    contrast: float = 0.15        # (sL[A] - sL[B]) + (sR[B] - sR[A]) needed to propose a cut
    pad_s: float = 0.1

    @classmethod
    def from_settings(cls, s: Settings) -> "SplitCfg":
        d = s.raw.get("split_detect", {}) or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Proposal:
    utt_id: str
    before_word: int             # the review app's split index: the right part starts at this word
    t_cut: float
    left: str
    right: str
    contrast: float
    s_left: float
    s_right: float


def line_words(con: sqlite3.Connection, utt: dict) -> list[dict]:
    """Aligned words of one line, in order, with times (words the aligner could not place are skipped)."""
    rows = con.execute(
        """SELECT w.word, w.start_s, w.end_s, w.line_idx, w.word_idx, c.idx AS cue_idx FROM words w
           JOIN utterance_cues uc ON uc.cue_id = w.cue_id JOIN cues c ON c.cue_id = w.cue_id
           WHERE uc.utt_id = ? ORDER BY c.idx, w.line_idx, w.word_idx""", (utt["utt_id"],)).fetchall()
    out = [dict(r) for r in rows if r["start_s"] is not None and utt["start_s"] - 0.3 <= r["start_s"] <= utt["end_s"] + 0.3]
    return out


def candidate_cuts(words: list[dict], start: float, end: float, min_side_s: float) -> list[tuple[int, float]]:
    """(k, t) for each pause between word k-1 and word k leaving min_side_s on both sides."""
    out = []
    for k in range(1, len(words)):
        t = 0.5 * (words[k - 1]["end_s"] + words[k]["start_s"])
        if t - start >= min_side_s and end - t >= min_side_s:
            out.append((k, t))
    return out


def propose(utts: list[dict], words_of, audio: np.ndarray, sr: int, encoder, scorer: Scorer,
            cfg: SplitCfg) -> dict[str, tuple[Proposal | None, float]]:
    """utt_id -> (best two-voice cut or None, its contrast; -1 when no cut had two different confident sides).
    A line is proposed for splitting when that contrast >= cfg.contrast. `words_of(utt)` gives the line's words."""
    jobs, slices = [], []
    for u in utts:
        if u["end_s"] - u["start_s"] < cfg.min_utt_s:
            continue
        ws = words_of(u)
        for k, t in candidate_cuts(ws, u["start_s"], u["end_s"], cfg.min_side_s):
            for a, b in ((u["start_s"], t), (t, u["end_s"])):
                i0, i1 = max(0, int((a - cfg.pad_s) * sr)), min(len(audio), int((b + cfg.pad_s) * sr))
                slices.append(audio[i0:i1].astype(np.float32))
            jobs.append((u, k, t))
    out: dict[str, tuple[Proposal | None, float]] = {}
    if not jobs:
        return out
    vecs = encoder.encode(slices).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    for j, (u, k, t) in enumerate(jobs):
        dom = u.get("domain_hint") if u.get("domain_hint") in DOMAINS else "field"
        rl = dict(scorer.rank(vecs[2 * j], dom))
        rr = dict(scorer.rank(vecs[2 * j + 1], dom))
        if not rl or not rr:
            continue
        a, b = max(rl, key=rl.get), max(rr, key=rr.get)
        if a == b or rl[a] < cfg.floor or rr[b] < cfg.floor:
            continue
        c = (rl[a] - rl.get(b, 0.0)) + (rr[b] - rr.get(a, 0.0))
        if c > out.get(u["utt_id"], (None, -1.0))[1]:
            out[u["utt_id"]] = (Proposal(u["utt_id"], k, round(t, 3), a, b, round(c, 3), round(rl[a], 3), round(rr[b], 3)), c)
    for u in utts:                           # checked but no two-voice cut: score -1
        if u["end_s"] - u["start_s"] >= cfg.min_utt_s:
            out.setdefault(u["utt_id"], (None, -1.0))
    return out


def scorer_for(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, resolver=None,
               as_of: int | None = None) -> Scorer:
    bank, _ = load_bank(con, vs, as_of if as_of is not None else ep)
    if not bank:                     # E01 of a season whose first bank was fit on E01-02 (stored as of E02)
        bank, _ = load_bank(con, vs)
    host = settings.franchise_for(vs).host_id
    cands = (frozenset(resolver.present(vs, ep)) | {host}) if resolver is not None else frozenset(s for s, _ in bank)
    return Scorer(bank, cands, settings.raw.get("bank", {}).get("score_mode", "centroid"))


@dataclass
class SuggestCfg:
    min_utt_s: float = 2.5        # lines shorter than this are not checked
    min_second_s: float = 0.8     # the diarizer must hear the second voice at least this long inside the line
    min_side_s: float = 0.5       # each side of the cut must hold this much audio for the bank to judge it
    min_contrast: float = 0.2     # bank: (sL[A] - sL[B]) + (sR[B] - sR[A]) with A != B
    floor: float = 0.35
    # automatic cut: a line with no human or name-card label where the diarizer hears a second voice this long
    auto: bool = True
    auto_second_s: float = 0.5
    auto_side_s: float = 0.5      # each part at least this long

    @classmethod
    def from_settings(cls, s: Settings) -> "SuggestCfg":
        d = s.raw.get("split_detect", {}) or {}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items() if hasattr(cls, k)})


def suggest_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, resolver=None, encoder=None,
                    track=None, audio: np.ndarray | None = None, write: bool = True, auto: bool | None = None,
                    reembed: bool = True, reassign: bool = True) -> list[dict]:
    """Two-voice lines in an episode's body, from the diarizer's change track.

    Automatic cut (split_detect.auto, on): every line where the diarizer hears a second voice for >= auto_second_s
    with >= auto_side_s on each side of the change, unless a person or a name card labelled the line. Cutting is
    cheap to get wrong: the parts keep their run, so assign pools them again and a one-speaker line cut in two
    normally gets one speaker on both parts (assign only separates a part whose own voice confidently names someone
    else). After cutting, the episode is re-embedded and re-assigned.
    Lines a person or a name card labelled are not cut; when the bank also names two different people there
    (contrast >= min_contrast) they are stored in split_suggestions for the review app. Returns every two-voice line
    found, with `auto` set on the ones cut."""
    from .stage_diarize import line_change, load_track

    cfg = SuggestCfg.from_settings(settings)
    if auto is not None:
        cfg.auto = auto
    track = track if track is not None else load_track(settings, vs, ep)
    if track is None:
        return []
    utts = [dict(r) for r in con.execute(
        """SELECT u.utt_id, u.start_s, u.end_s, u.domain_hint, l.source AS lab_source FROM utterances u
           LEFT JOIN labels l USING (utt_id)
           WHERE u.version_season=? AND u.episode=? AND u.segment='body' AND u.end_s - u.start_s >= ?
             AND json_extract(u.flags, '$.split_from') IS NULL""", (vs, ep, cfg.min_utt_s))]
    out = []
    for u in utts:
        sec2, cut = line_change(track, u["start_s"], u["end_s"])
        if cut is None:
            continue
        side = min(cut - u["start_s"], u["end_s"] - cut)
        d = {"utt_id": u["utt_id"], "t_cut": round(cut, 3), "second_s": round(sec2, 2), "side_s": side,
             "lab_source": u["lab_source"], "human": u["lab_source"] == "human",
             "domain": u["domain_hint"] if u["domain_hint"] in DOMAINS else "field",
             "left_spk": None, "right_spk": None, "contrast": None, "start_s": u["start_s"], "end_s": u["end_s"]}
        d["auto"] = bool(cfg.auto and u["lab_source"] in (None, "auto", "sdh") and sec2 >= cfg.auto_second_s
                         and side >= cfg.auto_side_s)
        if d["auto"] or (sec2 >= cfg.min_second_s and side >= cfg.min_side_s):
            out.append(d)
    review = [d for d in out if not d["auto"]]
    if review:                                             # bank names for the lines left to a person
        if audio is None:
            from .stage_align import load_audio
            audio = load_audio(settings.audio_path(settings.audio.variant, vs, ep), settings.audio.sample_rate)
        if encoder is None:
            from .stage_embed import Encoder
            encoder = Encoder(settings.embed.model, settings.embed.device)
        sr = settings.audio.sample_rate
        sl = []
        for d in review:
            for x0, x1 in ((d["start_s"], d["t_cut"]), (d["t_cut"], d["end_s"])):
                seg = audio[int(max(0.0, x0 - 0.1) * sr):int((x1 + 0.1) * sr)].astype(np.float32)
                sl.append(seg if len(seg) >= int(0.3 * sr) else np.pad(seg, (0, int(0.3 * sr) - len(seg))))
        V = encoder.encode(sl).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        scorer = scorer_for(settings, con, vs, ep, resolver)
        for j, d in enumerate(review):
            rl, rr = dict(scorer.rank(V[2 * j], d["domain"])), dict(scorer.rank(V[2 * j + 1], d["domain"]))
            if not rl or not rr:
                continue
            a, b = max(rl, key=rl.get), max(rr, key=rr.get)
            if a != b and rl[a] >= cfg.floor and rr[b] >= cfg.floor:
                d.update(left_spk=a, right_spk=b, contrast=round((rl[a] - rl.get(b, 0.0)) + (rr[b] - rr.get(a, 0.0)), 3))
    suggest = [d for d in review if d["contrast"] is not None and d["contrast"] >= cfg.min_contrast]
    out = [d for d in out if d["auto"]] + suggest
    if write:
        keep = {r[0]: r[1] for r in con.execute(
            "SELECT utt_id, status FROM split_suggestions WHERE version_season=? AND episode=? AND status<>'open'", (vs, ep))}
        con.execute("DELETE FROM split_suggestions WHERE version_season=? AND episode=? AND status='open'", (vs, ep))
        for d in out:
            if d["utt_id"] in keep:                        # a person accepted or dismissed this line, or undid its cut
                d["auto"] = False
                continue
            con.execute("""INSERT OR REPLACE INTO split_suggestions (utt_id, version_season, episode, t_cut, left_spk,
                           right_spk, second_s, contrast, status, created_at) VALUES (?,?,?,?,?,?,?,?,'open',datetime('now'))""",
                        (d["utt_id"], vs, ep, d["t_cut"], d["left_spk"], d["right_spk"], d["second_s"], d["contrast"]))
            if d["auto"]:
                _auto_split(con, d)
            elif not d["human"]:
                con.execute("""INSERT INTO review_queue (utt_id, reason, payload, resolved) VALUES (?, 'two_voices', ?, 0)
                               ON CONFLICT(utt_id) DO UPDATE SET resolved=0""",
                            (d["utt_id"], json.dumps({"two_voices": {k: d[k] for k in ("t_cut", "left_spk", "right_spk")}})))
        con.commit()
        cut_now = any(d["auto"] for d in out)
        if cut_now and reembed:                            # the parts need voice vectors before assign or any bank
            from .stage_embed import embed_episode, embedding_path
            for v in ("raw", "center", "vocals", "vocals_center"):
                if embedding_path(settings, vs, ep, v).exists():
                    embed_episode(settings, con, vs, ep, variant=v)
        # every automatic cut in the episode with a part still unlabelled, including cuts from an earlier run whose
        # re-assign did not finish
        bases = [r[0] for r in con.execute(
            """SELECT s.utt_id FROM split_suggestions s WHERE s.version_season=? AND s.episode=? AND s.status='auto'
               AND (NOT EXISTS (SELECT 1 FROM labels WHERE utt_id = s.utt_id || 'a')
                    OR NOT EXISTS (SELECT 1 FROM labels WHERE utt_id = s.utt_id || 'b'))""", (vs, ep))]
        if bases and reassign and (reembed or not cut_now):  # label the parts the normal way (runs pooled, voice split)
            from .stage_assign import assign_episode
            used = _bank_for(con, vs, ep)
            if used is not None:
                assign_episode(settings, con, vs, ep, bank_as_of=used, resolver=resolver, write=True, rolling=False)
                _fill_parts(settings, con, vs, ep, bases, used, resolver)
    return out


def _bank_for(con: sqlite3.Connection, vs: str, ep: int) -> int | None:
    """The bank a re-assign after cutting uses: the latest one fit before this episode, else the season's first
    (E01-02 of a new season share a bank stored as of E02)."""
    q = "SELECT MAX(as_of_episode) FROM speaker_bank WHERE version_season=?"
    return (con.execute(q + " AND as_of_episode<=?", (vs, max(ep - 1, 1))).fetchone()[0]
            or con.execute(q, (vs,)).fetchone()[0])


def _fill_parts(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, bases: list[str], as_of: int | None,
                resolver=None, margin: float | None = None) -> int:
    """A cut line whose second part came out unlabelled (too short to label alone, or undecided): if that part's own
    voice does not clearly name someone else, it takes its sibling's speaker. A one-speaker line cut by mistake so
    ends up with one speaker on both parts; a real second voice that the bank hears keeps it apart (and queued)."""
    from .stage_bank import StaleEmbeddings, load_vectors

    th = settings.raw.get("thresholds", {}) or {}
    margin = float(th.get("voice_split_margin", 0.08)) if margin is None else margin
    try:
        vec = load_vectors(settings, con, vs, ep, settings.audio.variant).set_index("utt_id").vector.to_dict()
    except (FileNotFoundError, StaleEmbeddings):          # no vectors for the parts: nothing contradicts the sibling
        vec = {}
    scorer = scorer_for(settings, con, vs, ep, resolver, as_of=as_of)
    n = 0
    for base in bases:
        parts = [base + "a", base + "b"]
        lab = {p: con.execute("SELECT speaker_id, confidence, source FROM labels WHERE utt_id=?", (p,)).fetchone() for p in parts}
        for p, sib in ((parts[0], parts[1]), (parts[1], parts[0])):
            if lab[p] is not None or lab[sib] is None or lab[sib][2] not in ("auto", "sdh", "human", "chyron"):
                continue
            x = lab[sib][0]
            v = vec.get(p)
            if v is not None:
                dom = con.execute("SELECT domain_hint FROM utterances WHERE utt_id=?", (p,)).fetchone()[0]
                rank = dict(scorer.rank(v, dom if dom in DOMAINS else "field"))
                if rank and max(rank.values()) - rank.get(x, 0.0) >= margin:
                    continue                               # its own voice leans elsewhere: leave it to review
            con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain,
                           labeled_at) VALUES (?,?,'auto',?,?,NULL,datetime('now'))""",
                        (p, x, round(float(lab[sib][1] or 0.6) * 0.9, 4), json.dumps({"same_as_part": sib})))
            con.execute("DELETE FROM review_queue WHERE utt_id=? AND resolved=0", (p,))
            n += 1
    con.commit()
    return n


def _auto_split(con: sqlite3.Connection, d: dict) -> None:
    """Split at the word boundary nearest the change point. The parts are labelled by the re-assign that follows."""
    from .splits import split_utterance, utt_words

    u = dict(con.execute("SELECT * FROM utterances WHERE utt_id=?", (d["utt_id"],)).fetchone())
    ws = utt_words(con, u)
    if len(ws) < 2:
        d["auto"] = False
        return
    k = min(range(1, len(ws)), key=lambda j: abs(0.5 * (ws[j - 1]["end_s"] + ws[j]["start_s"]) - d["t_cut"]))
    split_utterance(con, d["utt_id"], k, auto=True)
