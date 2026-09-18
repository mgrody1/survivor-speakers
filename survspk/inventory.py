"""Inventory stage (spec §7.0): find videos and subtitles, probe media, choose a subtitle file,
and compare against survivoR episode metadata."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import ids
from .config import Settings
from .db import record_roots, upsert
from .subparse import is_sdh_filename, quick_stats

log = logging.getLogger(__name__)


@dataclass
class VideoFile:
    path: Path
    season: int
    episode: int
    episode2: int | None
    is_reunion: bool


@dataclass
class SubFile:
    path: Path
    season: int
    episode: int
    episode2: int | None
    is_sdh: bool


@dataclass
class Probe:
    duration_s: float | None = None
    audio_channels: int | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    subs: list[dict] = field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------------- scanning


def scan_videos(settings: Settings, franchise: str = "US") -> list[VideoFile]:
    out: list[VideoFile] = []
    exts = {e.lower() for e in settings.inventory.video_extensions}
    skip = [p.lower() for p in settings.inventory.skip_title_patterns]
    for season_dir in sorted(settings.paths.video_root.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in sorted(season_dir.iterdir()):
            if f.suffix.lower() not in exts or f.name.startswith("."):
                continue
            parsed = ids.parse_sxxeyy(f.name)
            if not parsed:
                log.warning("no SxxEyy in %s", f)
                continue
            s, e, e2 = parsed
            out.append(VideoFile(f, s, e, e2, any(p in f.name.lower() for p in skip)))
    return out


def scan_subtitles(settings: Settings) -> list[SubFile]:
    out: list[SubFile] = []
    for f in sorted(settings.paths.subtitle_root.rglob("*.srt")):
        if f.name.startswith("."):
            continue
        parsed = ids.parse_sxxeyy(f.name)
        if not parsed:
            log.warning("no SxxEyy in subtitle %s", f)
            continue
        s, e, e2 = parsed
        out.append(SubFile(f, s, e, e2, is_sdh_filename(f.name)))
    return out


# ----------------------------------------------------------------------------- ffprobe


def ffprobe(path: Path, timeout: int = 60) -> Probe:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,channels,width,height:stream_tags=language,title",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Probe(error="ffprobe timeout")
    if r.returncode != 0:
        return Probe(error=(r.stderr or "ffprobe failed").strip()[:500])
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        return Probe(error=f"bad ffprobe json: {e}")
    p = Probe()
    d = j.get("format", {}).get("duration")
    p.duration_s = float(d) if d else None
    for st in j.get("streams", []):
        t = st.get("codec_type")
        if t == "video" and p.video_codec is None:
            p.video_codec, p.width, p.height = st.get("codec_name"), st.get("width"), st.get("height")
        elif t == "audio" and p.audio_codec is None:
            p.audio_codec, p.audio_channels = st.get("codec_name"), st.get("channels")
        elif t == "subtitle":
            tags = st.get("tags", {}) or {}
            p.subs.append({"index": st.get("index"), "codec": st.get("codec_name"),
                           "language": tags.get("language"), "title": tags.get("title")})
    return p


# ----------------------------------------------------------------------------- survivoR lookups


def survivor_episode_meta(survivor_db: Path) -> dict[tuple[str, int], dict]:
    con = sqlite3.connect(survivor_db)
    try:
        df = pd.read_sql_query(
            "SELECT version_season, episode, episode_title, episode_length FROM episodes", con
        )
    finally:
        con.close()
    return {
        (r.version_season, int(r.episode)): {"title": r.episode_title, "length_min": r.episode_length}
        for r in df.itertuples()
        if pd.notna(r.episode)
    }


# ----------------------------------------------------------------------------- chooser

# A subtitle file's last cue normally ends 20-120 s before the video does (credits carry no dialogue).
# Ending *after* the video, or more than ~3 min before it, means a truncated file, a different cut,
# or a double episode.
TIMING_OK_MIN_S = -180.0
TIMING_OK_MAX_S = 15.0


def timing_ok(delta_s: float | None) -> bool | None:
    if delta_s is None:
        return None
    return TIMING_OK_MIN_S <= delta_s <= TIMING_OK_MAX_S


def choose_subtitle(cands: list[dict], prefer_sdh: bool, duration_s: float | None) -> dict | None:
    """Pick the best parsed candidate. Tiers: timing-ok files first, then unknown timing, then bad timing.
    Within a tier: SDH (if preferred), then the smaller |delta|, then more cues. Unparsed embedded streams
    (n_cues None) are only chosen when nothing parsed is available."""
    parsed = [c for c in cands if not c.get("parse_error") and (c.get("n_cues") or 0) > 50]

    def key(c: dict):
        ok = timing_ok(c.get("duration_delta_s"))
        tier = 0 if ok else (1 if ok is None else 2)
        delta = abs(c["duration_delta_s"]) if c.get("duration_delta_s") is not None else 1e6
        delta_bucket = 0 if delta <= 15 else round(delta / 30)   # 30 s buckets so cue count can break ties
        return (tier, 0 if (prefer_sdh and c["is_sdh"]) else 1, delta_bucket, -(c.get("n_cues") or 0))

    if parsed:
        return sorted(parsed, key=key)[0]
    emb = [c for c in cands if c["source"] == "embedded" and c.get("n_cues") is None]
    if emb:
        emb.sort(key=lambda c: (0 if c["is_sdh"] else 1, c["path"]))
        return emb[0]
    return None


def embedded_cache_path(settings: Settings, vs: str, episode: int, stream_index: int) -> Path:
    return settings.paths.work_root / "subs_embedded" / f"{vs}E{episode:02d}_s{stream_index}.srt"


def extract_embedded_to_cache(settings: Settings, video: Path, vs: str, episode: int, stream_index: int,
                              timeout: int = 900) -> Path:
    """Demux one text subtitle stream to a cached .srt (reads the whole video once)."""
    out = embedded_cache_path(settings, vs, episode, stream_index)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", f"0:{stream_index}", "-f", "srt", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[:500])
    return out


# ----------------------------------------------------------------------------- main


def build_inventory(
    settings: Settings,
    con: sqlite3.Connection,
    seasons: list[int] | None = None,
    probe: bool = True,
    force: bool = False,
    franchise: str = "US",
    extract_embedded: bool = True,
) -> pd.DataFrame:
    record_roots(con, settings)
    videos = scan_videos(settings, franchise)
    subs = scan_subtitles(settings)
    meta = survivor_episode_meta(settings.survivor_db_path) if settings.survivor_db_path.exists() else {}
    subs_by_ep: dict[tuple[int, int], list[SubFile]] = {}
    for sf in subs:
        subs_by_ep.setdefault((sf.season, sf.episode), []).append(sf)
        if sf.episode2:
            subs_by_ep.setdefault((sf.season, sf.episode2), []).append(sf)

    done = {
        (r["version_season"], r["episode"])
        for r in con.execute("SELECT version_season, episode FROM episodes WHERE probed_at IS NOT NULL")
    }
    summary_rows = []
    for v in videos:
        if seasons and v.season not in seasons:
            continue
        vs = ids.version_season(franchise, v.season)
        key = (vs, v.episode)
        if key in done and not force:
            continue
        m = meta.get(key, {})
        row: dict = {
            "version_season": vs, "episode": v.episode, "video_path": str(v.path), "video_basename": v.path.name,
            "is_reunion": v.is_reunion, "is_double": v.episode2 is not None, "episode2": v.episode2,
            "survivor_length_min": m.get("length_min"), "survivor_title": m.get("title"),
        }
        pr = Probe()
        if probe:
            pr = ffprobe(v.path, settings.inventory.ffprobe_timeout_s)
            if pr.error:
                log.warning("%s: %s", v.path.name, pr.error)
            row.update({
                "duration_s": pr.duration_s, "audio_channels": pr.audio_channels, "audio_codec": pr.audio_codec,
                "video_codec": pr.video_codec, "width": pr.width, "height": pr.height,
                "n_embedded_subs": len(pr.subs), "embedded_subs": pr.subs,
            })
            if pr.duration_s and m.get("length_min"):
                row["length_delta_min"] = round(pr.duration_s / 60 - float(m["length_min"]), 2)
        # subtitle candidates
        cands: list[dict] = []
        for sf in subs_by_ep.get((v.season, v.episode), []):
            st = quick_stats(sf.path)
            delta = (st["last_cue_end_s"] - pr.duration_s) if (st["last_cue_end_s"] and pr.duration_s) else None
            cands.append({
                "version_season": vs, "episode": v.episode, "path": str(sf.path), "source": "sidecar",
                "is_sdh": sf.is_sdh, "n_cues": st["n_cues"], "first_cue_s": st["first_cue_s"],
                "last_cue_end_s": st["last_cue_end_s"],
                "duration_delta_s": round(delta, 2) if delta is not None else None,
                "parse_error": st["parse_error"], "chosen": 0,
            })
        text_streams = [st for st in pr.subs if st.get("codec") in ("subrip", "ass", "ssa", "webvtt", "mov_text")]
        for st in text_streams:
            title = (st.get("title") or "").lower()
            cands.append({
                "version_season": vs, "episode": v.episode, "path": f"{v.path}#s:{st['index']}", "source": "embedded",
                "is_sdh": ("sdh" in title or title == "hi" or "cc" in title), "n_cues": None, "first_cue_s": None,
                "last_cue_end_s": None, "duration_delta_s": None, "parse_error": None, "chosen": 0,
            })
        # If no sidecar has acceptable timing, pay for demuxing the embedded streams so they compete fairly.
        sidecar_ok = any(c["source"] == "sidecar" and not c.get("parse_error") and (c.get("n_cues") or 0) > 50
                         and timing_ok(c.get("duration_delta_s")) for c in cands)
        if extract_embedded and text_streams and not sidecar_ok:
            for c in [c for c in cands if c["source"] == "embedded"]:
                idx = int(c["path"].rsplit("#s:", 1)[1])
                try:
                    srt = extract_embedded_to_cache(settings, v.path, vs, v.episode, idx)
                    st = quick_stats(srt)
                    delta = (st["last_cue_end_s"] - pr.duration_s) if (st["last_cue_end_s"] and pr.duration_s) else None
                    c.update({"n_cues": st["n_cues"], "first_cue_s": st["first_cue_s"], "last_cue_end_s": st["last_cue_end_s"],
                              "duration_delta_s": round(delta, 2) if delta is not None else None,
                              "parse_error": st["parse_error"], "extracted_path": str(srt)})
                    log.info("%s E%02d: extracted embedded stream %s -> %s cues, Δ=%s", vs, v.episode, idx,
                             st["n_cues"], c["duration_delta_s"])
                except Exception as e:  # noqa: BLE001
                    c["parse_error"] = f"extract failed: {e}"[:300]
        chosen = choose_subtitle(cands, settings.inventory.prefer_sdh, pr.duration_s)
        for c in cands:
            c["timing_ok"] = timing_ok(c.get("duration_delta_s"))
            c["chosen"] = int(chosen is not None and c["path"] == chosen["path"])
            upsert(con, "subtitle_files", {k: val for k, val in c.items() if k != "extracted_path"},
                   ("version_season", "episode", "path"))
        if chosen:
            row["subtitle_path"] = chosen.get("extracted_path") or chosen["path"]
            row["subtitle_source"] = f"{chosen['source']}_{'sdh' if chosen['is_sdh'] else 'plain'}"
            row["subtitle_timing_ok"] = timing_ok(chosen.get("duration_delta_s"))
        else:
            row["subtitle_path"] = None
            row["subtitle_source"] = "none"
            row["subtitle_timing_ok"] = None
        row["probed_at"] = datetime.now(timezone.utc).isoformat() if probe else None
        row["status"] = "inventoried"
        upsert(con, "episodes", row, ("version_season", "episode"))
        con.commit()
        summary_rows.append({k: row.get(k) for k in ("version_season", "episode", "subtitle_source", "duration_s",
                                                       "length_delta_min", "audio_channels", "n_embedded_subs")})
        log.info("%s E%02d  %s  dur=%s  Δmin=%s  ch=%s", vs, v.episode, row["subtitle_source"],
                 f"{pr.duration_s:.0f}" if pr.duration_s else "?", row.get("length_delta_min"), pr.audio_channels)
    return pd.DataFrame(summary_rows)


# ----------------------------------------------------------------------------- report


def _probable_cause(r: pd.Series) -> str:
    d = r.get("duration_delta_s")
    dur = r.get("duration_s") or 0
    last = r.get("last_cue_end_s") or 0
    surv = r.get("survivor_length_min")
    if d is None or pd.isna(d):
        return "unknown"
    if surv and not pd.isna(surv) and dur > 1.7 * float(surv) * 60 and dur > 70 * 60:
        return "video is probably a mis-named double episode (%.0f min vs survivoR %.0f)" % (dur / 60, surv)
    if d > TIMING_OK_MAX_S:
        if d < 120:
            return "subtitle runs %.0fs past file end: likely frame-rate drift; alignment stage should measure" % d
        return "subtitle covers %.0f min more than the file: finale+reunion cut or wrong episode" % (d / 60)
    if dur and last < 0.7 * dur:
        return "subtitle truncated (ends at %.0f%% of file): re-download" % (100 * last / dur)
    return "subtitle ends %.0f min early: different cut (file may include reunion/aftershow)" % (-d / 60)


def inventory_report(con: sqlite3.Connection, settings: Settings) -> dict[str, pd.DataFrame]:
    ep = pd.read_sql_query("SELECT * FROM episodes ORDER BY version_season, episode", con)
    sf = pd.read_sql_query("SELECT * FROM subtitle_files", con)
    body = ep[ep.is_reunion == 0].copy()
    per_season = body.groupby("version_season").agg(
        n_episodes=("episode", "count"),
        n_no_subs=("subtitle_source", lambda s: (s == "none").sum()),
        n_sdh=("subtitle_source", lambda s: s.str.endswith("sdh").sum()),
        n_embedded=("subtitle_source", lambda s: s.str.startswith("embedded").sum()),
        n_timing_bad=("subtitle_timing_ok", lambda s: (s == 0).sum()),
        n_doubles=("is_double", "sum"),
        n_51=("audio_channels", lambda s: (s == 6).sum()),
        n_length_mismatch=("length_delta_min", lambda s: (s.abs() > settings.inventory.max_length_delta_min).sum()),
    ).reset_index()
    missing = body[body.subtitle_source == "none"][["version_season", "episode", "video_basename", "n_embedded_subs"]]
    mism = body[body.length_delta_min.abs() > settings.inventory.max_length_delta_min][
        ["version_season", "episode", "video_basename", "duration_s", "survivor_length_min", "length_delta_min", "is_double"]]
    chosen = sf[sf.chosen == 1].merge(body[["version_season", "episode", "duration_s", "is_double", "survivor_length_min"]],
                                      on=["version_season", "episode"], how="inner")
    bad = chosen[chosen.timing_ok == 0].copy()
    bad["probable_cause"] = bad.apply(_probable_cause, axis=1)
    bad = bad[["version_season", "episode", "source", "is_sdh", "n_cues", "first_cue_s", "last_cue_end_s",
               "duration_s", "duration_delta_s", "is_double", "probable_cause"]]
    doubles = body[body.is_double == 1][["version_season", "episode", "video_basename", "duration_s", "survivor_length_min"]]
    return {"per_season": per_season, "missing_subtitles": missing, "subtitle_timing_bad": bad,
            "double_episode_files": doubles, "length_mismatch": mism}
