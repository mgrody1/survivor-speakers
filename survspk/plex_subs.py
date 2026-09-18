"""Fetch subtitles for flagged episodes through Plex's subtitle agent (python-plexapi).

Replaces the owner's lost Plex script. Flow per episode:
  1. find the Plex episode by season/episode number in the configured library
  2. ep.searchSubtitles(language='en')  -> candidate SubtitleStreams from Plex's providers
  3. pick SDH if available (title/hearingImpaired flag), else the best-scored plain one
  4. ep.downloadSubtitles(candidate)    -> Plex stores it (metadata dir or sidecar, per library setting)
  5. read the stream back through the API and write it into subtitle_root as
     <video stem>.en.hi.srt (SDH) or <video stem>.en.srt, so the inventory picks it up
  6. re-inventory + re-ingest that episode

NOTE: the plexapi method names/kwargs below (`searchSubtitles`, `downloadSubtitles`, `hearingImpaired`)
are written from memory of python-plexapi 4.15; run with --dry-run first and check the printed
candidates. Needs PLEX_URL and PLEX_TOKEN in .env (see .env.example).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)


def flagged_episodes(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Episodes whose chosen subtitle is missing or truncated (not drift / cut differences)."""
    rows = con.execute(
        """SELECT e.version_season, e.episode, e.video_path, e.video_basename, e.subtitle_source,
                  s.duration_delta_s, s.last_cue_end_s, e.duration_s, e.survivor_length_min
           FROM episodes e LEFT JOIN subtitle_files s
             ON s.version_season=e.version_season AND s.episode=e.episode AND s.chosen=1
           WHERE e.is_reunion=0 AND (e.subtitle_source='none' OR e.subtitle_timing_ok=0)
           ORDER BY e.version_season, e.episode"""
    ).fetchall()
    out = []
    for r in rows:
        surv = r["survivor_length_min"]
        probable_double = bool(surv) and r["duration_s"] and r["duration_s"] > 1.7 * surv * 60 and r["duration_s"] > 70 * 60
        if probable_double:
            continue  # e.g. US19E12: the video holds two episodes; a fresh single-episode subtitle would not help
        if r["subtitle_source"] == "none":
            out.append(r)
        elif r["last_cue_end_s"] and r["duration_s"] and r["last_cue_end_s"] < 0.7 * r["duration_s"]:
            out.append(r)  # truncated
    return out


def _plex():
    from dotenv import load_dotenv
    from plexapi.server import PlexServer

    load_dotenv()
    url, token = os.environ.get("PLEX_URL"), os.environ.get("PLEX_TOKEN")
    if not url or not token:
        raise SystemExit("PLEX_URL / PLEX_TOKEN not set (see .env.example)")
    return PlexServer(url, token)


def _find_show(plex, show_title: str = "Survivor", tvdb_id: int | None = 76733, year: int | None = 2000):
    """Pick the right show. `section.get(title)` fuzzy-matches and returned *Australian Survivor* on the
    first run, so match on the TVDB id (from the folder name `{TvbId-76733}`), then year, then exact title."""
    shows, seen = [], set()
    for section in plex.library.sections():
        if section.type == "show":
            for sh in section.search(title=show_title):
                key = getattr(sh, "ratingKey", None) or (sh.title, getattr(sh, "year", None))
                if key not in seen:          # the same show can be listed by more than one library
                    seen.add(key)
                    shows.append(sh)
    if not shows:
        raise LookupError(f"no show matching {show_title!r} in any TV library")

    def guids(show) -> str:
        parts = [str(getattr(show, "guid", "") or "")]
        for g in getattr(show, "guids", []) or []:
            parts.append(str(getattr(g, "id", g)))
        return " ".join(parts)

    def unique(hits):
        titles = {(sh.title, getattr(sh, "year", None)) for sh in hits}
        return hits[0] if hits and len(titles) == 1 else None

    if tvdb_id and (hit := unique([sh for sh in shows if str(tvdb_id) in guids(sh)])):
        return hit
    if year and (hit := unique([sh for sh in shows if getattr(sh, "year", None) == year])):
        return hit
    if hit := unique([sh for sh in shows if sh.title.strip().lower() == show_title.strip().lower()]):
        return hit
    names = ", ".join(f"{sh.title} ({getattr(sh, 'year', '?')})" for sh in shows)
    raise LookupError(f"ambiguous show match for {show_title!r}: {names} — pass --tvdb-id or --year")


def _find_episode(show, season: int, episode: int):
    try:
        return show.episode(season=season, episode=episode)
    except Exception as e:  # noqa: BLE001
        raise LookupError(f"{show.title} S{season:02d}E{episode:02d} not in Plex ({e})") from e


def _is_sdh(stream) -> bool:
    title = (getattr(stream, "title", "") or "").lower()
    return bool(getattr(stream, "hearingImpaired", False)) or "sdh" in title or "hearing" in title or ".hi." in title


def fetch_for_episode(settings: Settings, plex, show, row: sqlite3.Row,
                      dry_run: bool = False, prefer_sdh: bool = True) -> Path | None:
    season = int(row["version_season"][2:])
    ep = _find_episode(show, season, row["episode"])
    log.info("%s E%02d -> Plex: %s (%s) S%02dE%02d %r", row["version_season"], row["episode"], show.title,
             getattr(show, "year", "?"), ep.seasonNumber, ep.index, ep.title)
    cands = ep.searchSubtitles(language="en")
    if not cands:
        log.warning("  no subtitle candidates from Plex providers")
        return None
    # sort: SDH first (if preferred), then provider score desc
    cands = sorted(cands, key=lambda s: (0 if (prefer_sdh and _is_sdh(s)) else 1,
                                         -float(getattr(s, "score", 0) or 0)))
    for s in cands[:8]:
        log.info("  candidate: sdh=%s score=%s title=%r provider=%s", _is_sdh(s), getattr(s, "score", None),
                 getattr(s, "title", None), getattr(s, "providerTitle", None))
    if dry_run:
        return None
    chosen = cands[0]
    before = {getattr(st, "id", None) for st in ep.subtitleStreams()}
    ep.downloadSubtitles(chosen)
    # Plex attaches the downloaded file asynchronously; poll until a new external stream appears.
    new: list = []
    for _ in range(20):
        time.sleep(2)
        ep.reload()
        new = [st for st in ep.subtitleStreams()
               if getattr(st, "id", None) not in before and getattr(st, "key", None)]
        if new:
            break
    if not new:
        new = [st for st in ep.subtitleStreams() if getattr(st, "key", None)]
        if not new:
            raise RuntimeError("download reported ok but no external subtitle stream appeared within 40 s")
        log.warning("  no *new* stream detected; using the newest existing external stream")
    stream = sorted(new, key=lambda st: (0 if _is_sdh(st) else 1, -(getattr(st, "id", 0) or 0)))[0]
    # Fetch the raw file through the API (plex.query() would try to parse it as XML).
    url = plex.url(stream.key, includeToken=True)
    resp = plex._session.get(url, timeout=120)
    resp.raise_for_status()
    raw = resp.content
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    if "-->" not in text[:20000]:
        raise RuntimeError(f"fetched stream {stream.key} does not look like SRT (starts {text[:80]!r})")
    is_sdh = _is_sdh(stream) or _is_sdh(chosen)
    stem = Path(row["video_basename"]).stem
    out = settings.paths.subtitle_root / f"{stem}.en{'.hi' if is_sdh else ''}.srt"
    if out.exists():
        out = settings.paths.subtitle_root / f"{stem}.plex.en{'.hi' if is_sdh else ''}.srt"
    out.write_text(text, encoding="utf-8")
    log.info("  wrote %s (%d bytes, %d cues)", out.name, len(text), text.count("-->"))
    return out


def sxxeyy_from_row(row: sqlite3.Row) -> str:
    return f"S{int(row['version_season'][2:]):02d}E{row['episode']:02d}"
