"""Subtitle ingest (the subtitle half of spec §7.1): parse the chosen subtitle file for each episode
into `cues`, record the detected convention on `episodes`, and tally SDH name tokens."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from . import ids
from .aliases import Resolver
from .config import Settings
from .db import localize, upsert
from .subparse import Cue, detect_convention, parse_srt, parse_srt_text

log = logging.getLogger(__name__)


def extract_embedded(video: Path, stream_index: int, timeout: int = 900) -> str:
    """Demux one subtitle stream to SRT text. Reads the whole file — only used when no sidecar exists."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{stream_index}", "-f", "srt", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return r.stdout


def load_cues_for_episode(row: sqlite3.Row, con: sqlite3.Connection | None = None, settings: Settings | None = None) -> list[Cue]:
    src = row["subtitle_path"]
    if not src:
        raise FileNotFoundError("episode has no subtitle")
    if con is not None and settings is not None:
        src = str(localize(con, settings, src))
    if "#s:" in src:
        video, idx = src.rsplit("#s:", 1)
        return parse_srt_text(extract_embedded(Path(video), int(idx)))
    return parse_srt(Path(src))


def ingest_episode(settings: Settings, con: sqlite3.Connection, row: sqlite3.Row, resolver: Resolver | None,
                   name_counter: dict[tuple[str, str], Counter] | None = None) -> dict:
    vs, ep = row["version_season"], row["episode"]
    cues = load_cues_for_episode(row, con, settings)
    conv = detect_convention(cues, settings.subparse.name_prefix_min_lines, settings.subparse.marker_min_lines)
    con.execute("DELETE FROM cues WHERE version_season=? AND episode=?", (vs, ep))
    con.executemany(
        "INSERT INTO cues (cue_id, version_season, episode, idx, start_s, end_s, raw_text, lines) VALUES (?,?,?,?,?,?,?,?)",
        [
            (ids.cue_id(vs, ep, c.idx), vs, ep, c.idx, c.start_s, c.end_s, c.raw_text,
             json.dumps([l.__dict__ for l in c.lines], ensure_ascii=False))
            for c in cues
        ],
    )
    con.execute(
        "UPDATE episodes SET subtitle_convention=?, status='subs_ingested' WHERE version_season=? AND episode=?",
        (json.dumps(conv.flags()), vs, ep),
    )
    from .import_names import apply_imports

    n_imported = apply_imports(con, vs, ep)        # names borrowed from another release of these captions (import-names)
    if name_counter is not None:
        for name, n in conv.names.items():
            name_counter.setdefault((vs, name), Counter())["lines"] += n
            name_counter[(vs, name)]["files"] += 1
    con.commit()
    return {"version_season": vs, "episode": ep, "n_cues": conv.n_cues, "n_name_prefix": conv.n_name_prefix,
            "n_gtgt": conv.n_gtgt, "n_dash": conv.n_dash, "n_italic": conv.n_italic,
            "n_multi_turn_cues": conv.n_multi_turn_cues, "n_names": len(conv.names), "n_imported": n_imported}


def ingest_all(settings: Settings, con: sqlite3.Connection, seasons: list[str] | None = None,
               include_reunions: bool = False, force: bool = False) -> pd.DataFrame:
    q = "SELECT * FROM episodes WHERE subtitle_path IS NOT NULL"
    if not include_reunions:
        q += " AND is_reunion = 0"
    if not force:
        q += " AND (status IS NULL OR status != 'subs_ingested')"
    q += " ORDER BY version_season, episode"
    rows = [r for r in con.execute(q) if not seasons or r["version_season"] in seasons]
    resolver = Resolver(settings) if settings.survivor_db_path.exists() else None
    counter: dict[tuple[str, str], Counter] = {}
    out = []
    for r in rows:
        try:
            out.append(ingest_episode(settings, con, r, resolver, counter))
        except Exception as e:  # noqa: BLE001
            log.error("%s E%02d: %s", r["version_season"], r["episode"], e)
            out.append({"version_season": r["version_season"], "episode": r["episode"], "error": str(e)})
    # name token table (merge with existing counts from earlier partial runs)
    for (vs, tok), c in counter.items():
        res = resolver.resolve(tok, vs) if resolver else None
        prev = con.execute("SELECT n_lines, n_files FROM name_tokens WHERE version_season=? AND token=?", (vs, tok)).fetchone()
        n_lines = c["lines"] + (prev["n_lines"] if prev and not force else 0)
        n_files = c["files"] + (prev["n_files"] if prev and not force else 0)
        upsert(con, "name_tokens", {
            "version_season": vs, "token": tok, "n_lines": n_lines, "n_files": n_files,
            "resolved_id": res.speaker_id if res else None, "resolution": res.kind if res else None,
        }, ("version_season", "token"))
    con.commit()
    if resolver:
        resolver.close()
    return pd.DataFrame(out)


def names_report(settings: Settings, con: sqlite3.Connection, min_lines: int = 3) -> dict[str, pd.DataFrame]:
    """Re-resolve every token (so alias edits take effect without re-ingesting) and summarise."""
    resolver = Resolver(settings)
    toks = pd.read_sql_query("SELECT * FROM name_tokens", con)
    res = [resolver.resolve(t, vs) for t, vs in zip(toks.token, toks.version_season)]
    toks["resolved_id"] = [r.speaker_id for r in res]
    toks["resolution"] = [r.kind for r in res]
    with con:
        con.executemany("UPDATE name_tokens SET resolved_id=?, resolution=? WHERE version_season=? AND token=?",
                        list(zip(toks.resolved_id, toks.resolution, toks.version_season, toks.token)))
    resolver.close()
    unresolved = toks[(toks.resolution.isin(["unresolved", "ambiguous"])) & (toks.n_lines >= min_lines)].sort_values(
        ["version_season", "n_lines"], ascending=[True, False])
    per_season = toks.groupby("version_season").apply(
        lambda g: pd.Series({
            "n_tokens": len(g),
            "lines_total": g.n_lines.sum(),
            "lines_cast": g.loc[g.resolution.isin(["cast", "alias"]), "n_lines"].sum(),
            "lines_host": g.loc[g.resolution == "host", "n_lines"].sum(),
            "lines_other": g.loc[g.resolution == "other", "n_lines"].sum(),
            "lines_stop": g.loc[g.resolution == "stop", "n_lines"].sum(),
            "lines_ambiguous": g.loc[g.resolution == "ambiguous", "n_lines"].sum(),
            "lines_unresolved": g.loc[g.resolution == "unresolved", "n_lines"].sum(),
            "n_cast_resolved": g.loc[g.resolution.isin(["cast", "alias"]), "resolved_id"].nunique(),
        }), include_groups=False).reset_index()
    per_season["pct_resolved"] = (100 * (per_season.lines_cast + per_season.lines_host + per_season.lines_other) /
                                  per_season.lines_total.replace(0, pd.NA)).round(1)
    return {"per_season": per_season, "unresolved": unresolved, "all_tokens": toks}
