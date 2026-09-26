"""Segmentation (spec §7.4): cues -> turns -> utterances, SDH name propagation, recap/preview.

    cue   : one subtitle block (may contain two speakers: dash / >> turns)
    turn  : the lines of one cue that belong to one speaker, timed from aligned words
    utt   : consecutive same-speaker turns merged into one unit for embedding (≤ merge_max_s)

Pure logic lives in build_utterances(); segment_episode() does the sqlite I/O.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from . import ids
from .config import SegmentCfg, Settings

log = logging.getLogger(__name__)

TERMINAL = re.compile(r"[.?!…]+[\"'”’)]*$")
PREVIOUSLY_RE = re.compile(r"\b(?:previously|last time),?\s+on\b", re.I)
# Every Survivor recap ends with the previous tribal council's verdict (or a medevac/quit) — M0 check: present
# in 461 of 474 recaps library-wide. This is far more reliable than looking for a gap: recap -> cold open ->
# title sequence means the biggest gap marks the title, not the recap's end.
VERDICT_RE = re.compile(r"tribe has spoken|voted out|time for you to go|eliminated|out of the game|grab your torch"
                        r"|medically|evacuat|quit", re.I)
NEXT_TIME_RE = re.compile(r"\bnext time,?\s+on\b", re.I)


@dataclass
class Turn:
    cue_id: str
    cue_idx: int
    line_indices: list[int]
    text: str
    sdh_name: str | None            # explicit NAME: on this turn
    has_marker: bool                # started with '-' or '>>'
    is_italic: bool
    start: float
    end: float
    n_words: int
    from_words: bool                # timing from aligned words (else proportional fallback)
    shared_cue: bool                # the cue held ≥2 turns
    name: str | None = None         # after propagation
    name_inherited: bool = False


@dataclass
class Utt:
    idx: int
    start: float
    end: float
    text: str
    turns: list[Turn] = field(default_factory=list)
    name: str | None = None
    name_inherited: bool = False
    is_italic: bool = False
    segment: str = "body"
    speaker_id: str | None = None
    resolution: str | None = None
    run_id: int = -1
    run_dur: float = 0.0
    domain_hint: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


# ----------------------------------------------------------------------------- turns


def _line_span(words: list[dict]) -> tuple[float, float] | None:
    placed = [(w["start_s"], w["end_s"]) for w in words if w.get("start_s") is not None and w.get("end_s") is not None]
    if not placed:
        return None
    return min(s for s, _ in placed), max(e for _, e in placed)


def cue_to_turns(cue: dict, words_by_line: dict[int, list[dict]], tmap: Callable[[float], float]) -> list[Turn]:
    """Split one cue's lines into speaker turns. A new turn starts at a line with a turn marker or a NAME:.
    Sound-only lines are dropped (a marked sound line still ends the previous turn)."""
    lines = cue["lines"]
    groups: list[list[int]] = []
    for i, l in enumerate(lines):
        starts_new = (l["is_turn"] or l["sdh_name"]) and (groups and groups[-1])
        if not groups or starts_new:
            groups.append([])
        if l["is_sound"] and not l["text"] and not l["sdh_name"]:
            continue
        groups[-1].append(i)
    groups = [g for g in groups if any(lines[i]["text"] for i in g)]
    if not groups:
        return []
    c_start, c_end = tmap(cue["start_s"]), tmap(cue["end_s"])
    total_chars = sum(len(lines[i]["text"]) for g in groups for i in g) or 1
    turns: list[Turn] = []
    cursor = c_start
    for g in groups:
        text = " ".join(lines[i]["text"] for i in g if lines[i]["text"]).strip()
        n_chars = len(text)
        spans = [sp for i in g if (sp := _line_span(words_by_line.get(i, [])))]
        if spans:
            start, end, from_words = min(s for s, _ in spans), max(e for _, e in spans), True
            cursor = end
        else:
            share = (c_end - c_start) * (n_chars / total_chars)
            start, end, from_words = cursor, cursor + share, False
            cursor = end
        n_words = sum(len(words_by_line.get(i, [])) for i in g) or len(text.split())
        turns.append(Turn(
            cue_id=cue["cue_id"], cue_idx=cue["idx"], line_indices=list(g), text=text,
            sdh_name=next((lines[i]["sdh_name"] for i in g if lines[i]["sdh_name"]), None),
            has_marker=any(lines[i]["is_turn"] for i in g[:1]),
            is_italic=sum(lines[i]["is_italic"] for i in g) * 2 > len(g),
            start=round(start, 3), end=round(max(end, start + 0.05), 3), n_words=n_words,
            from_words=from_words, shared_cue=len(groups) > 1,
        ))
    return turns


# ----------------------------------------------------------------------------- propagation & merge


# "Tiyana, do you agree with that?" / "So, Gabe, what happened out there?" — the host addressing a castaway by
# name and asking a question. The next unnamed turn is the castaway's answer, not the host's continuation.
# (M1 run-errors on US47E02: the answer inheriting PROBST was the single most confident 'error' in the LOO.)
ADDRESS_Q_RE = re.compile(
    r"^\W*(?:(?:so|and|okay|ok|well|now|all right|alright|but),?\s+)?([A-Z][a-z]+(?:[ -][A-Z][a-z]+)?),\s.*\?[\"'”’)]*$",
    re.I | re.S)


def propagate_names(turns: list[Turn], gap_s: float, host_names: frozenset[str] = frozenset(),
                    addressee: Callable[[str], str | None] | None = None) -> None:
    """Fill Turn.name from explicit NAME: lines. A name holds across following unnamed turns until a turn marker
    or a gap > gap_s. With `host_names` (normalised tokens) and `addressee` (text -> cast name token, or None), a
    host turn that addresses a castaway with a question hands the name to the next unnamed turn."""
    from .aliases import normalize_token

    current: str | None = None
    prev_end: float | None = None
    pending: str | None = None
    for t in turns:
        gap_break = prev_end is not None and t.start - prev_end > gap_s
        if t.sdh_name:
            current = t.sdh_name
            t.name, t.name_inherited = current, False
            pending = None
        else:
            if t.has_marker or gap_break:
                current = None
            if pending is not None and not gap_break:
                current = pending
            t.name, t.name_inherited = current, current is not None
            pending = None
        if addressee is not None and t.name and normalize_token(t.name) in host_names:
            m = ADDRESS_Q_RE.match(t.text or "")
            who = addressee(m.group(1)) if m else None
            pending = who if who and normalize_token(who) != normalize_token(t.name) else None
        prev_end = t.end


def can_merge(u: Utt, t: Turn, cfg: SegmentCfg) -> bool:
    if t.has_marker:
        return False
    gap = t.start - u.end
    if gap < -0.5 or gap > cfg.merge_gap_s:
        return False
    if (t.end - u.start) > cfg.merge_max_s:
        return False
    if t.sdh_name and t.sdh_name != u.name:
        return False
    if u.name != t.name:
        return False
    if TERMINAL.search(u.text):
        # sentence ended: only merge when both sides carry the same known name
        return u.name is not None and t.name == u.name
    return True   # sentence continues across the cue boundary: same speaker


def merge_turns(turns: list[Turn], cfg: SegmentCfg) -> list[Utt]:
    utts: list[Utt] = []
    for t in turns:
        if utts and can_merge(utts[-1], t, cfg):
            u = utts[-1]
            u.end = max(u.end, t.end)
            u.text = (u.text + " " + t.text).strip()
            u.turns.append(t)
            u.is_italic = sum(x.is_italic for x in u.turns) * 2 > len(u.turns)
        else:
            utts.append(Utt(idx=len(utts), start=t.start, end=t.end, text=t.text, turns=[t], name=t.name,
                            name_inherited=t.name_inherited, is_italic=t.is_italic))
    return utts


# ----------------------------------------------------------------------------- recap / preview


def detect_recap_end(utts: list[Utt], cfg: SegmentCfg) -> float | None:
    """End of the 'previously on' recap.

    1. find the recap phrase in the first recap_search_s (text joined across neighbouring utterances,
       because "Previously" / "on Survivor:" is often split across cues);
    2. recap ends at the last verdict phrase within recap_max_s of it;
    3. fallback: end of the utterance before the largest speech gap in the window (the title sequence —
       this swallows a cold open into the recap, which only costs a few bank utterances)."""
    k = None
    for i, u in enumerate(utts):
        if u.start > cfg.recap_search_s:
            break
        joined = u.text + " " + (utts[i + 1].text if i + 1 < len(utts) else "")
        if PREVIOUSLY_RE.search(joined):
            k = i
            break
    if k is None:
        return None
    t0 = utts[k].start
    window = [u for u in utts[k:] if u.start - t0 <= cfg.recap_max_s]
    verdicts = [u for u in window if VERDICT_RE.search(u.text)]
    if verdicts:
        return verdicts[-1].end
    best_gap, best_end = 0.0, None
    for a, b in zip(window, window[1:]):
        gap = b.start - a.end
        if a.end - t0 >= 20 and gap > best_gap:
            best_gap, best_end = gap, a.end
    return best_end if best_gap >= cfg.recap_gap_s else None


def detect_preview_start(utts: list[Utt], total_s: float | None) -> float | None:
    floor = 0.75 * total_s if total_s else (utts[-1].end * 0.75 if utts else 0)
    hits = [u for u in utts if u.start >= floor and NEXT_TIME_RE.search(u.text)]
    return hits[-1].start if hits else None


# ----------------------------------------------------------------------------- runs / domain


def assign_runs(utts: list[Utt], cfg: SegmentCfg) -> None:
    """Group consecutive utterances into monologue runs: no turn marker on the next utterance, gap below
    run_gap_s, and names compatible (equal, or one side unnamed). A run is one speaker by construction
    (speaker changes are marked in dash/`>>` files). Runs >= confessional_run_s are the confessional domain —
    on US47E02 this reproduced survivoR's hand-counted confessionals per player closely (spec §10)."""
    if not utts:
        return
    rid = 0
    groups: list[list[Utt]] = [[utts[0]]]
    for prev, u in zip(utts, utts[1:]):
        same_seg = u.segment == prev.segment
        marker = u.turns[0].has_marker if u.turns else False
        gap = u.start - prev.end
        names_ok = prev.name is None or u.name is None or prev.name == u.name
        if same_seg and not marker and gap < cfg.run_gap_s and names_ok:
            groups[-1].append(u)
        else:
            groups.append([u])
    for rid, g in enumerate(groups):
        dur = g[-1].end - g[0].start
        # a run inherits the one name its members carry (if any)
        names = {x.name for x in g if x.name}
        for x in g:
            x.run_id, x.run_dur = rid, dur
            if x.is_italic or dur >= cfg.confessional_run_s:
                x.domain_hint = "confessional"
            else:
                x.domain_hint = "field"
            if x.name is None and len(names) == 1:
                x.name, x.name_inherited = next(iter(names)), True


# ----------------------------------------------------------------------------- driver


def build_utterances(cues: list[dict], words: dict[str, dict[int, list[dict]]], tmap: Callable[[float], float],
                     cfg: SegmentCfg, resolver=None, vs: str = "", ep: int = 0,
                     total_s: float | None = None) -> tuple[list[Utt], dict]:
    turns: list[Turn] = []
    for c in cues:
        turns.extend(cue_to_turns(c, words.get(c["cue_id"], {}), tmap))
    turns.sort(key=lambda t: (t.start, t.cue_idx))
    host_names, addressee = frozenset(), None
    if resolver is not None and vs:
        from .aliases import normalize_token

        host_names, _ = resolver.host_names(vs)
        short, _, _ = resolver.maps(vs)
        present = resolver.present(vs, ep) if ep else frozenset()

        def addressee(token: str) -> str | None:
            cid = short.get(normalize_token(token))
            return token.upper() if cid and (not present or cid in present) else None
    propagate_names(turns, cfg.sdh_propagate_gap_s, host_names, addressee)
    utts = merge_turns(turns, cfg)
    recap_end = detect_recap_end(utts, cfg)
    preview_start = detect_preview_start(utts, total_s)
    for u in utts:
        if recap_end is not None and u.start < recap_end:
            u.segment = "recap"
        elif preview_start is not None and u.start >= preview_start:
            u.segment = "preview"
        else:
            u.segment = "body"
    assign_runs(utts, cfg)
    for u in utts:
        if u.name and resolver is not None:
            r = resolver.resolve(u.name, vs, ep)
            u.speaker_id, u.resolution = r.speaker_id, r.kind
    stats = {
        "n_cues": len(cues), "n_turns": len(turns), "n_utts": len(utts),
        "n_shared_cue_turns": sum(t.shared_cue for t in turns),
        "n_turns_from_words": sum(t.from_words for t in turns),
        "n_utts_named": sum(u.name is not None for u in utts),
        "n_utts_resolved": sum(u.speaker_id is not None for u in utts),
        "n_merged_into": sum(len(u.turns) > 1 for u in utts),
        "n_runs": len({u.run_id for u in utts}),
        "n_confessional_runs": len({u.run_id for u in utts if u.domain_hint == "confessional"}),
        "median_utt_s": sorted(u.duration for u in utts)[len(utts) // 2] if utts else None,
        "recap_end_s": recap_end, "preview_start_s": preview_start,
    }
    return utts, stats


def load_words(con: sqlite3.Connection, vs: str, ep: int) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for r in con.execute("""SELECT w.cue_id, w.line_idx, w.word_idx, w.word, w.start_s, w.end_s, w.score
                            FROM words w JOIN cues c USING (cue_id)
                            WHERE c.version_season=? AND c.episode=? ORDER BY w.cue_id, w.line_idx, w.word_idx""", (vs, ep)):
        out.setdefault(r["cue_id"], {}).setdefault(r["line_idx"], []).append(dict(r))
    return out


class HumanLabelsExist(RuntimeError):
    """Re-segmenting rebuilds every utterance; some human labels could not be re-anchored onto the new ones."""


REANCHOR_MIN_COVER = 0.6     # a new utterance inherits a human label when this share of it lies inside the label's span


def reanchor_labels(old: list[dict], new: list[tuple], min_cover: float = REANCHOR_MIN_COVER) -> tuple[dict[str, dict], list[dict]]:
    """Map human/chyron labels from the old utterances of an episode onto its new ones.

    old: [{utt_id, speaker_id, source, start_s, end_s, keys?, ...}]; new: [(utt_id, start_s, end_s[, keys])].
    `keys` are the (cue_id, line indices) pieces an utterance is made of. When both sides have them, matching goes by
    cue: a re-align moves every time (US31 E01 moved 2-4 s), so matching by time span would hand a label to the
    neighbouring line. A new utterance inherits a label when that label's pieces make up `min_cover` of it, or when
    every label with pieces inside it names the same speaker (cues regrouped: US31 E02 split one labelled line
    across two new ones).
    Without keys it falls back to time-span overlap.
    Returns (new_utt_id -> old label row, old labels no new utterance inherited)."""
    if not old or not new:
        return {}, list(old)
    out: dict[str, dict] = {}
    used: set[str] = set()
    by_key: dict = {}
    for r in old:
        for k in r.get("keys") or ():
            by_key.setdefault(k, []).append(r)
    spans = sorted(old, key=lambda r: r["start_s"])
    starts = [r["start_s"] for r in spans]
    for item in new:
        uid, a, b = item[:3]
        keys = set(item[3]) if len(item) > 3 and item[3] else None
        best = None
        if keys and by_key:
            hits: dict[str, list] = {}
            for k in keys:
                for r in by_key.get(k, ()):
                    hits.setdefault(r["utt_id"], [r, 0])[1] += 1
            if hits:
                speakers = {h[0]["speaker_id"] for h in hits.values()}
                r, n = max(hits.values(), key=lambda h: h[1])
                # the segmenter only joins cues it takes for one speaker, so a line holding pieces of labels that
                # all agree gets that speaker; labels that disagree inside one line are left for a person
                if n / len(keys) >= min_cover or len(speakers) == 1:
                    best = r
                    if len(speakers) == 1:
                        used.update(h[0]["utt_id"] for h in hits.values())
            if best is not None:
                out[uid] = best
                used.add(best["utt_id"])
            continue
        dur = b - a
        if dur <= 0:
            continue
        best_ov = 0.0
        j = bisect.bisect_right(starts, b)
        for r in spans[max(0, j - 64):j]:                 # labels are short; 64 covers any plausible overlap window
            if r.get("keys") and keys:
                continue
            ov = min(b, r["end_s"]) - max(a, r["start_s"])
            if ov > best_ov:
                best, best_ov = r, ov
        if best is not None and best_ov / dur >= min_cover:
            out[uid] = best
            used.add(best["utt_id"])
    lost = [r for r in old if r["utt_id"] not in used]
    return out, lost


def _utt_keys(con: sqlite3.Connection, vs: str, ep: int) -> dict[str, frozenset]:
    """utt_id -> its (cue_id, line indices) pieces, for every current utterance of the episode."""
    keys: dict[str, set] = {}
    for uid, cid, li in con.execute(
            """SELECT uc.utt_id, uc.cue_id, uc.line_indices FROM utterance_cues uc JOIN utterances u USING (utt_id)
               WHERE u.version_season=? AND u.episode=?""", (vs, ep)):
        keys.setdefault(uid, set()).add((cid, tuple(json.loads(li)) if li else ()))
    return {k: frozenset(v) for k, v in keys.items()}


def _protected_labels(con: sqlite3.Connection, vs: str, ep: int, keys: dict | None = None) -> list[dict]:
    """Human / chyron labels of the episode with their spans (taken from the live utterance rows) and cue pieces."""
    rows = [dict(r) for r in con.execute(
        """SELECT l.utt_id, l.speaker_id, l.source, l.top_candidates, u.start_s, u.end_s, u.text
           FROM labels l JOIN utterances u USING (utt_id)
           WHERE u.version_season=? AND u.episode=? AND l.source IN ('human', 'chyron') ORDER BY u.start_s""", (vs, ep))]
    if keys:
        for r in rows:
            r["keys"] = keys.get(r["utt_id"])
    return rows


def remap_utt_ids(old_keys: dict[str, frozenset], new_keys: dict[str, frozenset]) -> dict[str, str]:
    """old utt_id -> the new utterance holding most of its cue pieces (for card checks that name utterances)."""
    owner = {k: uid for uid, ks in new_keys.items() for k in ks}
    out = {}
    for uid, ks in old_keys.items():
        votes: dict[str, int] = {}
        for k in ks:
            if k in owner:
                votes[owner[k]] = votes.get(owner[k], 0) + 1
        if votes:
            out[uid] = max(votes, key=votes.get)
    return out


def _remap_card_checks(con: sqlite3.Connection, vs: str, ep: int, idmap: dict[str, str]) -> int:
    try:
        rows = con.execute("SELECT t_s, castaway_id, utt_id, utt_ids FROM card_checks WHERE version_season=? AND episode=?",
                           (vs, ep)).fetchall()
    except sqlite3.OperationalError:
        return 0
    n = 0
    for t_s, cid, uid, uids in rows:
        new_uid = idmap.get(uid, uid) if uid else None
        lst = json.loads(uids) if uids else ([uid] if uid else [])
        new_lst = list(dict.fromkeys(idmap.get(u, u) for u in lst))
        if new_uid != uid or new_lst != lst:
            con.execute("UPDATE card_checks SET utt_id=?, utt_ids=? WHERE version_season=? AND episode=? AND t_s=? AND castaway_id=?",
                        (new_uid, json.dumps(new_lst) if new_lst else None, vs, ep, t_s, cid))
            n += 1
    return n


def segment_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, resolver=None,
                    allow_relabel: bool = False) -> dict:
    epi = con.execute("SELECT align_offset_s, align_drift, align_stats, duration_s FROM episodes WHERE version_season=? AND episode=?",
                      (vs, ep)).fetchone()
    if not epi:
        raise LookupError(f"{vs} E{ep:02d} not in inventory")
    from .stage_align import time_map
    knots = (json.loads(epi["align_stats"]) or {}).get("knots") if epi["align_stats"] else None
    tmap = time_map(epi["align_offset_s"] or 0.0, epi["align_drift"] or 0.0, knots)
    cues = [dict(r) | {"lines": json.loads(r["lines"])} for r in
            con.execute("SELECT cue_id, idx, start_s, end_s, lines FROM cues WHERE version_season=? AND episode=? ORDER BY idx", (vs, ep))]
    if not cues:
        raise LookupError(f"no cues for {vs} E{ep:02d}")
    words = load_words(con, vs, ep)
    if not words:
        log.warning("%s E%02d has no aligned words; utterance timing falls back to proportional cue splits", vs, ep)
    utts, stats = build_utterances(cues, words, tmap, settings.segment, resolver, vs, ep, epi["duration_s"])

    # human / chyron labels survive a re-segment by time span; anything that cannot be re-anchored stops the
    # re-segment unless --force (word-level splits are rebuilt from the new utterances, so they are lost too)
    old_keys = _utt_keys(con, vs, ep)
    protected = _protected_labels(con, vs, ep, old_keys)
    new_keys = {ids.utt_id(vs, ep, u.idx): frozenset((t.cue_id, tuple(t.line_indices)) for t in u.turns) for u in utts}
    new_spans = [(ids.utt_id(vs, ep, u.idx), u.start, u.end, new_keys[ids.utt_id(vs, ep, u.idx)]) for u in utts]
    inherited, lost = reanchor_labels(protected, new_spans)
    if lost and not allow_relabel:
        shown = "; ".join(f"{r['utt_id']} {r['speaker_id']} {r['start_s']:.1f}-{r['end_s']:.1f}s" for r in lost[:8])
        raise HumanLabelsExist(f"{vs} E{ep:02d}: {len(lost)} of {len(protected)} human labels cannot be re-anchored "
                               f"onto the new utterances ({shown}{'; ...' if len(lost) > 8 else ''}). "
                               f"Pass --force to re-segment and drop them.")
    old_ids = "SELECT utt_id FROM utterances WHERE version_season=? AND episode=?"
    con.execute(f"DELETE FROM labels WHERE utt_id IN ({old_ids})", (vs, ep))
    con.execute(f"DELETE FROM review_queue WHERE utt_id IN ({old_ids})", (vs, ep))
    con.execute(f"DELETE FROM utterance_cues WHERE utt_id IN ({old_ids})", (vs, ep))
    con.execute("DELETE FROM utterances WHERE version_season=? AND episode=?", (vs, ep))
    try:
        con.execute("DELETE FROM utt_splits WHERE base_utt_id LIKE ?", (f"{vs}_E{ep:02d}_U%",))
    except sqlite3.OperationalError:
        pass
    urows, ucrows = [], []
    for u in utts:
        uid = ids.utt_id(vs, ep, u.idx)
        flags = {
            "merged_n": len(u.turns),
            "split_from_shared_cue": any(t.shared_cue for t in u.turns),
            "align_fallback": any(not t.from_words for t in u.turns),
            "name_inherited": u.name_inherited,
            "run_id": u.run_id, "run_dur_s": round(u.run_dur, 2),
        }
        urows.append((uid, vs, ep, u.idx, u.start, u.end, u.text, u.segment, 1, u.name, u.speaker_id, u.resolution,
                      int(u.is_italic), u.domain_hint,
                      int(all(t.from_words for t in u.turns)), json.dumps(flags), sum(t.n_words for t in u.turns)))
        for t in u.turns:
            ucrows.append((uid, t.cue_id, json.dumps(t.line_indices)))
    con.executemany("""INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, is_speech,
                       sdh_name, sdh_speaker_id, sdh_resolution, is_italic, domain_hint, align_ok, flags, n_words)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", urows)
    con.executemany("INSERT OR REPLACE INTO utterance_cues (utt_id, cue_id, line_indices) VALUES (?,?,?)", ucrows)
    span = {r[0]: r for r in urows}
    for uid, r in inherited.items():
        _, _, _, _, a, b, text = span[uid][:7]
        con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain,
                       labeled_at, version_season, episode, start_s, end_s, text)
                       VALUES (?,?,?,1.0,?,?,datetime('now'),?,?,?,?,?)""",
                    (uid, r["speaker_id"], r["source"], json.dumps({"reanchored_from": r["utt_id"]}), span[uid][13],
                     vs, ep, a, b, text))
    stats["n_labels_reanchored"], stats["n_labels_lost"] = len(inherited), len(lost)
    stats["n_card_checks_remapped"] = _remap_card_checks(con, vs, ep, remap_utt_ids(old_keys, new_keys))
    con.execute("""UPDATE episodes SET recap_end_s=?, preview_start_s=?, n_utterances=?, status='segmented'
                   WHERE version_season=? AND episode=?""", (stats["recap_end_s"], stats["preview_start_s"], len(utts), vs, ep))
    con.commit()
    try:
        cc = confessional_crosscheck(settings, con, utts, vs, ep)
        if cc:
            stats["confessional_crosscheck"] = {k: cc[k] for k in ("n_players", "spearman_count", "spearman_time", "mae_count")}
    except Exception as e:  # noqa: BLE001
        log.warning("confessional cross-check failed: %s", e)
    log.info("%s E%02d: %d cues -> %d turns -> %d utterances (median %.1fs); %d named, %d resolved; recap_end=%s preview=%s",
             vs, ep, stats["n_cues"], stats["n_turns"], stats["n_utts"], stats["median_utt_s"] or 0, stats["n_utts_named"],
             stats["n_utts_resolved"], stats["recap_end_s"], stats["preview_start_s"])
    return stats


def confessional_crosscheck(settings: Settings, con: sqlite3.Connection, utts: list[Utt], vs: str, ep: int) -> dict | None:
    """Compare our named confessional runs per player with survivoR's hand-counted confessionals (spec §10).
    Only runs whose members carry a resolved cast name count; unnamed runs are what the classifier is for."""
    if not settings.survivor_db_path.exists():
        return None
    sv = sqlite3.connect(f"file:{settings.survivor_db_path}?mode=ro", uri=True)
    try:
        ref = sv.execute("SELECT castaway_id, castaway, confessional_count, confessional_time FROM confessionals "
                         "WHERE version_season=? AND episode=?", (vs, ep)).fetchall()
    finally:
        sv.close()
    if not ref:
        return None
    runs: dict[int, list[Utt]] = {}
    for u in utts:
        if u.segment == "body" and u.domain_hint == "confessional":
            runs.setdefault(u.run_id, []).append(u)
    ours_n: dict[str, int] = {}
    ours_t: dict[str, float] = {}
    for g in runs.values():
        ids_ = {x.speaker_id for x in g if x.speaker_id and (x.resolution in ("cast", "alias"))}
        if len(ids_) == 1:
            cid = next(iter(ids_))
            ours_n[cid] = ours_n.get(cid, 0) + 1
            ours_t[cid] = ours_t.get(cid, 0.0) + (g[-1].end - g[0].start)
    rows = [{"castaway_id": r[0], "castaway": r[1], "ref_count": r[2] or 0, "ref_time": r[3] or 0,
             "our_count": ours_n.get(r[0], 0), "our_time": round(ours_t.get(r[0], 0.0), 1)} for r in ref]
    import pandas as pd

    df = pd.DataFrame(rows)

    def spearman(a: pd.Series, b: pd.Series) -> float | None:   # rank Pearson; avoids the scipy dependency
        if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
            return None
        return round(float(a.rank().corr(b.rank())), 3)

    out = {
        "n_players": len(df),
        "spearman_count": spearman(df.ref_count, df.our_count),
        "spearman_time": spearman(df.ref_time, df.our_time),
        "mae_count": round(float((df.ref_count - df.our_count).abs().mean()), 2),
        "table": rows,
    }
    for k in ("spearman_count", "spearman_time", "mae_count"):
        con.execute("INSERT OR REPLACE INTO metrics (version_season, episode, key, value, payload, computed_at) "
                    "VALUES (?,?,?,?,?,datetime('now'))", (vs, ep, f"confessional_{k}", out[k], json.dumps(rows)))
    con.commit()
    log.info("%s E%02d confessional cross-check vs survivoR: spearman count %s, time %s, MAE count %s (%d players)",
             vs, ep, out["spearman_count"], out["spearman_time"], out["mae_count"], out["n_players"])
    return out
