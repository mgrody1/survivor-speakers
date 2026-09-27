"""Borrow caption speaker names (`NAME:`) from another subtitle file of the same episode.

Some of our subtitle files never name speakers although another release of the same captions does (US41: 8 of 13
episodes; US46 E04). The names are the trusted labels a season's voice bank learns from, so this copies them onto our
own lines, keeping our timing and italics:

  1. The other file's lines are split into turns (dash turns, `NAME:` prefixes).
  2. Its clock is mapped onto ours: lines whose text matches exactly and uniquely in both give time pairs; a robust
     straight-line fit (offset and drift) maps the rest.
  3. Each named turn goes to the first of our lines near the mapped time whose text matches it (rapidfuzz >= 85 on
     letters and digits only, so run-together words from a cleaned release still match). Lines that already carry a
     name are left alone. A named or dashed source turn also marks our line as a new turn, so an imported name does
     not run on into the next speaker's lines.
  4. Nothing is written unless >= 20 lines match exactly and at least half of the source's names land; otherwise the
     two files are probably different cuts.

Every import is recorded in `name_imports` with the line's text, so re-ingesting the same subtitle file puts the
names back (and a different file, whose lines no longer match, does not).

Source used so far: hipml/survivor-subtitles-cleaned on Hugging Face (OpenSubtitles, S1-47; research use, credit CBS).
It stays on this machine: work_root/external/.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .subparse import NAME_RE, TAG_RE

HF_URL = "https://huggingface.co/datasets/hipml/survivor-subtitles-cleaned/resolve/main/data/train-00000-of-00001.parquet"
SPLIT_TURNS = re.compile(r"(?:^|\s)[-–—]\s*(?=\S)")
MIN_SCORE = 85.0
WINDOW_S = 4.0
MIN_ANCHOR_CHARS = 12
MIN_MATCH_SHARE = 0.5     # land fewer of the source's names than this and the releases differ too much (another cut)
MIN_ANCHORS = 20          # fewer exact matches than this: the two files are too different to trust the clock


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


@dataclass
class Turn:
    start: float
    end: float
    name: str | None
    text: str
    marker: bool = False      # the turn opened with a dash (a new speaker) in the source


def split_turns(start: float, end: float, text: str) -> list[Turn]:
    """One subtitle's text -> its turns, with the NAME: prefix (if any) pulled off each."""
    s = TAG_RE.sub("", text or "").strip()
    lead = bool(re.match(r"^[-–—]\s*\S", s))
    parts = [p for p in SPLIT_TURNS.split(s) if p and p.strip()]
    out = []
    for k, p in enumerate(parts):
        p = p.strip()
        m = NAME_RE.match(p)
        name, body = (re.sub(r"\s+", " ", m.group(1)).strip(" ."), m.group(2)) if m and len(m.group(1)) >= 2 else (None, p)
        out.append(Turn(start, end, name, body.strip(), marker=k > 0 or lead))
    return out


def load_source(path: Path, season: int, episode: int, cache: dict | None = None) -> list[Turn]:
    """Turns of one episode from the parquet (columns episode 'S46E04', start_time, end_time, text)."""
    cache = {} if cache is None else cache
    if "df" not in cache:
        cache["df"] = pd.read_parquet(path)
    df = cache["df"]
    d = df[df.episode == f"S{season:02d}E{episode:02d}"].sort_values("start_time")
    turns: list[Turn] = []
    for r in d.itertuples():
        turns.extend(split_turns(float(r.start_time), float(r.end_time), r.text))
    return turns


def ensure_source(settings, path: Path | None = None) -> Path:
    path = path or Path(settings.paths.work_root) / "external" / "hf_survivor_subtitles_cleaned.parquet"
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(HF_URL, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def our_lines(con: sqlite3.Connection, vs: str, ep: int) -> list[dict]:
    rows = con.execute("SELECT idx, start_s, end_s, lines FROM cues WHERE version_season=? AND episode=? ORDER BY idx",
                       (vs, ep)).fetchall()
    out = []
    for idx, a, b, lines in rows:
        for i, ln in enumerate(json.loads(lines or "[]")):
            out.append({"cue_idx": idx, "line_i": i, "start": a, "end": b, "text": ln.get("text") or "",
                        "sdh_name": ln.get("sdh_name"), "is_turn": bool(ln.get("is_turn")),
                        "norm": norm(ln.get("text") or "")})
    return out


def fit_clock(src: list[Turn], ours: list[dict]) -> tuple[float, float, int]:
    """(slope, offset, n anchors) mapping source time -> our time, from exact unique text matches."""
    def uniq(items, key):
        seen: dict = {}
        for it in items:
            seen.setdefault(key(it), []).append(it)
        return {k: v[0] for k, v in seen.items() if len(v) == 1 and len(k) >= MIN_ANCHOR_CHARS}
    a = uniq(src, lambda t: norm(t.text))
    b = uniq(ours, lambda x: x["norm"])
    pairs = [(a[k].start, b[k]["start"]) for k in a.keys() & b.keys()]
    if len(pairs) < 5:
        return 1.0, 0.0, len(pairs)
    x, y = np.array(pairs).T
    off = np.median(y - x)
    keep = np.abs(y - x - off) < 5.0
    if keep.sum() >= 5:
        slope, inter = np.polyfit(x[keep], y[keep], 1)
        if abs(slope - 1) < 0.01:
            return float(slope), float(inter), int(keep.sum())
    return 1.0, float(off), int(keep.sum())


def match_names(src: list[Turn], ours: list[dict], slope: float, offset: float) -> list[dict]:
    """[{cue_idx, line_i, name, turn, score, norm}]: what a source turn adds to the line of ours it lands on.

    `name` for an unnamed line under a named turn; `turn` when the source opens a new speaker there (a name or a dash)
    and our line has no marker, so an imported name does not run on into the next speaker's lines."""
    from rapidfuzz import fuzz

    starts = np.array([x["start"] for x in ours])
    taken: set = set()
    out = []
    for t in src:
        if not (t.name or t.marker) or not norm(t.text):
            continue
        tt = slope * t.start + offset
        lo, hi = np.searchsorted(starts, tt - WINDOW_S), np.searchsorted(starts, tt + WINDOW_S)
        best, score = None, 0.0
        nt = norm(t.text)
        for j in range(lo, hi):
            x = ours[j]
            if not x["norm"] or (x["cue_idx"], x["line_i"]) in taken:
                continue
            s = fuzz.ratio(nt, x["norm"])
            if len(x["norm"]) >= 8 and (nt.startswith(x["norm"][:8]) or x["norm"].startswith(nt[:8])):
                s = max(s, fuzz.partial_ratio(nt, x["norm"]))       # one release joined lines the other splits
            if s > score:
                best, score = x, s
        if best is not None and score >= MIN_SCORE:
            taken.add((best["cue_idx"], best["line_i"]))
            name = t.name.upper() if t.name and not best["sdh_name"] else None
            turn = not best["is_turn"] and not best["sdh_name"]
            if name or turn:
                out.append({"cue_idx": best["cue_idx"], "line_i": best["line_i"], "name": name, "turn": bool(turn),
                            "score": round(score, 1), "norm": best["norm"]})
    return out


SCHEMA = """CREATE TABLE IF NOT EXISTS name_imports (
    version_season TEXT NOT NULL, episode INTEGER NOT NULL, cue_idx INTEGER NOT NULL, line_i INTEGER NOT NULL,
    name TEXT, turn INTEGER NOT NULL DEFAULT 0, source TEXT, score REAL, norm_text TEXT, created_at TEXT,
    PRIMARY KEY (version_season, episode, cue_idx, line_i))"""


def apply_imports(con: sqlite3.Connection, vs: str, ep: int) -> int:
    """Write recorded imports into the episode's cues where the line is still the same text: the name if the line is
    still unnamed, the turn marker if it has none. Returns the number of names written."""
    con.execute(SCHEMA)
    imp = {(r[0], r[1]): (r[2], r[3], r[4]) for r in con.execute(
        "SELECT cue_idx, line_i, name, norm_text, turn FROM name_imports WHERE version_season=? AND episode=?", (vs, ep))}
    if not imp:
        return 0
    n = 0
    for idx, lines in con.execute("SELECT idx, lines FROM cues WHERE version_season=? AND episode=?", (vs, ep)).fetchall():
        ls = json.loads(lines or "[]")
        changed = False
        for i, ln in enumerate(ls):
            hit = imp.get((idx, i))
            if not hit or norm(ln.get("text") or "") != hit[1]:
                continue
            if hit[0] and not ln.get("sdh_name"):
                ln["sdh_name"], ln["name_source"] = hit[0], "import"
                changed, n = True, n + 1
            if hit[2] and not ln.get("is_turn"):
                ln["is_turn"], ln["turn_source"] = True, "import"
                changed = True
        if changed:
            con.execute("UPDATE cues SET lines=? WHERE version_season=? AND episode=? AND idx=?",
                        (json.dumps(ls, ensure_ascii=False), vs, ep, idx))
    con.commit()
    return n


def import_episode(con: sqlite3.Connection, src_path: Path, vs: str, ep: int, write: bool = True,
                   cache: dict | None = None, source: str = "hipml/survivor-subtitles-cleaned") -> dict:
    season = int(re.sub(r"\D", "", vs))
    src = load_source(src_path, season, ep, cache)
    ours = our_lines(con, vs, ep)
    have = sum(1 for x in ours if x["sdh_name"])
    rep = {"version_season": vs, "episode": ep, "our_names": have, "source_names": sum(1 for t in src if t.name)}
    if not src or not ours:
        return rep | {"anchors": 0, "matched": 0, "turns": 0, "ok": False, "applied": 0}
    slope, offset, n_anchor = fit_clock(src, ours)
    hits = match_names(src, ours, slope, offset)
    rep |= {"anchors": n_anchor, "offset_s": round(offset, 2), "drift": round(slope - 1, 5),
            "matched": sum(1 for h in hits if h["name"]), "turns": sum(1 for h in hits if h["turn"])}
    rep["ok"] = n_anchor >= MIN_ANCHORS and rep["matched"] >= MIN_MATCH_SHARE * max(rep["source_names"], 1)
    if write and hits and rep["ok"]:
        con.execute(SCHEMA)
        con.execute("DELETE FROM name_imports WHERE version_season=? AND episode=?", (vs, ep))
        con.executemany("""INSERT OR REPLACE INTO name_imports (version_season, episode, cue_idx, line_i, name, turn, source,
                           score, norm_text, created_at) VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
                        [(vs, ep, h["cue_idx"], h["line_i"], h["name"], int(h["turn"]), source, h["score"], h["norm"])
                         for h in hits])
        con.commit()
        rep["applied"] = apply_imports(con, vs, ep)
    else:
        rep["applied"] = 0
    return rep


def candidates(con: sqlite3.Connection, src_path: Path, seasons: list[str] | None, min_our: int = 20, min_src: int = 20) -> list[tuple[str, int]]:
    """Episodes whose own captions carry fewer than `min_our` names while the source has at least `min_src`."""
    df = pd.read_parquet(src_path, columns=["episode", "text"])
    df["n"] = df.text.fillna("").map(lambda t: sum(1 for x in split_turns(0, 0, t) if x.name))
    src_n = df.groupby("episode").n.sum()
    out = []
    for vs, ep, n in con.execute("""SELECT version_season, episode, SUM(lines LIKE '%"sdh_name": "%') FROM cues
                                    WHERE version_season LIKE 'US%' GROUP BY 1, 2 ORDER BY 1, 2"""):
        if seasons and vs not in seasons:
            continue
        key = f"S{int(vs[2:]):02d}E{ep:02d}"
        if (n or 0) < min_our and src_n.get(key, 0) >= min_src:
            out.append((vs, int(ep)))
    return out
