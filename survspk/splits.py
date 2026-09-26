"""Splitting one line into two (two speakers inside one caption with no marker): by hand in the review app, or
automatically when the diarizer and the bank agree strongly (survspk.split_detect)."""

from __future__ import annotations

import json
import sqlite3

SPLITS_SCHEMA = """CREATE TABLE IF NOT EXISTS utt_splits (
    base_utt_id TEXT PRIMARY KEY, original TEXT NOT NULL, cues TEXT NOT NULL, parts TEXT NOT NULL, created_at TEXT)"""


def utt_words(c: sqlite3.Connection, u: dict) -> list[dict]:
    """Ordered words of an utterance with times (audio seconds). Missing word times are interpolated; if no word
    was placed at all, times are spread proportionally to character length over the utterance span."""
    cues = c.execute("""SELECT uc.cue_id, uc.line_indices, cu.start_s FROM utterance_cues uc JOIN cues cu USING (cue_id)
                        WHERE uc.utt_id=? ORDER BY cu.start_s""", (u["utt_id"],)).fetchall()
    words: list[dict] = []
    for r in cues:
        lines = json.loads(r["line_indices"] or "[]")
        if not lines:
            continue
        ph = ",".join("?" * len(lines))
        for w in c.execute(f"""SELECT word, start_s, end_s FROM words WHERE cue_id=? AND line_idx IN ({ph})
                               ORDER BY line_idx, word_idx""", (r["cue_id"], *lines)):
            words.append({"word": w["word"], "start_s": w["start_s"], "end_s": w["end_s"]})
    if not words:                                  # no alignment rows at all: fall back to the text
        toks = (u["text"] or "").split()
        words = [{"word": t, "start_s": None, "end_s": None} for t in toks]
    n = len(words)
    if n == 0:
        return []
    if all(w["start_s"] is None for w in words):
        span = u["end_s"] - u["start_s"]
        total = sum(len(w["word"]) + 1 for w in words)
        t = u["start_s"]
        for w in words:
            d = span * (len(w["word"]) + 1) / total
            w["start_s"], w["end_s"] = t, t + d
            t += d
    else:                                          # interpolate gaps
        for i, w in enumerate(words):
            if w["start_s"] is None:
                prev = next((x["end_s"] for x in reversed(words[:i]) if x["end_s"] is not None), u["start_s"])
                nxt = next((x["start_s"] for x in words[i + 1:] if x["start_s"] is not None), u["end_s"])
                w["start_s"], w["end_s"] = prev, nxt
    for i, w in enumerate(words):
        w["i"] = i
    return words


def split_utterance(c: sqlite3.Connection, utt_id: str, before_word: int, auto: bool = False) -> dict:
    """Split one utterance at a word boundary. The parts get ids <utt_id>a / <utt_id>b and idx / idx+0.5, keep the
    run, and replace the original (its label, queue row and cue links; the original is kept in utt_splits for undo).
    By hand (`auto` False) both parts are queued; an automatic split queues nothing, the caller labels the parts.
    Embeddings for the episode are stale until it is re-embedded."""
    c.executescript(SPLITS_SCHEMA)
    row = c.execute("SELECT * FROM utterances WHERE utt_id=?", (utt_id,)).fetchone()
    if row is None:
        raise LookupError(utt_id)
    u = dict(row)
    words = utt_words(c, u)
    k = before_word
    if not (1 <= k < len(words)):
        raise ValueError(f"before_word must be in 1..{len(words) - 1}")
    t_cut = 0.5 * (words[k - 1]["end_s"] + words[k]["start_s"])
    t_cut = min(max(t_cut, u["start_s"] + 0.05), u["end_s"] - 0.05)
    flags = json.loads(u["flags"] or "{}")
    cues = [dict(r) for r in c.execute("SELECT cue_id, line_indices FROM utterance_cues WHERE utt_id=?", (utt_id,))]
    parts = []
    for suffix, ws, (a, b), idx_off, keep_name in (("a", words[:k], (u["start_s"], t_cut), 0, True),
                                                  ("b", words[k:], (t_cut, u["end_s"]), 0.5, False)):
        pid = f"{utt_id}{suffix}"
        f = dict(flags, split_from=utt_id, split_at_word=k, split_part=suffix, **({"split_auto": True} if auto else {}))
        c.execute("""INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, is_speech,
                     sdh_name, sdh_speaker_id, is_italic, domain_hint, align_ok, flags, n_words, sdh_resolution)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (pid, u["version_season"], u["episode"], u["idx"] + idx_off, a, b, " ".join(w["word"] for w in ws),
                   u["segment"], 1, u["sdh_name"] if keep_name else None, u["sdh_speaker_id"] if keep_name else None,
                   u["is_italic"], u["domain_hint"], u["align_ok"], json.dumps(f), len(ws),
                   u["sdh_resolution"] if keep_name else None))
        for cu in cues:
            c.execute("INSERT OR REPLACE INTO utterance_cues (utt_id, cue_id, line_indices) VALUES (?,?,?)",
                      (pid, cu["cue_id"], cu["line_indices"]))
        parts.append(pid)
    q = c.execute("SELECT reason, payload FROM review_queue WHERE utt_id=?", (utt_id,)).fetchone()
    for pid in ([] if auto else parts):
        c.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                  (pid, q["reason"] if q else "split", q["payload"] if q else json.dumps({"split_from": utt_id})))
    c.execute("INSERT OR REPLACE INTO utt_splits (base_utt_id, original, cues, parts, created_at) VALUES (?,?,?,?,datetime('now'))",
              (utt_id, json.dumps(u), json.dumps(cues), json.dumps(parts)))
    c.execute("DELETE FROM review_queue WHERE utt_id=?", (utt_id,))
    try:
        c.execute("UPDATE split_suggestions SET status=? WHERE utt_id=?", ("auto" if auto else "accepted", utt_id))
    except sqlite3.OperationalError:
        pass
    c.execute("DELETE FROM labels WHERE utt_id=?", (utt_id,))
    c.execute("DELETE FROM utterance_cues WHERE utt_id=?", (utt_id,))
    c.execute("DELETE FROM utterances WHERE utt_id=?", (utt_id,))
    c.commit()
    return {"parts": parts, "t_cut": round(t_cut, 3)}

