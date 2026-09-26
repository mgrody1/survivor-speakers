"""Audio extraction (spec §7.1, audio half).

raw    : ffmpeg -i video -vn -ac 1 -ar 16000        -> work_root/raw/<vs>/E<ep>.flac      (every episode)
center : ffmpeg -i video -vn -af "pan=mono|c0=FC" -ar 16000 -> work_root/center/...     (6-channel sources only)

Broadcast 5.1 mixes put dialogue almost entirely in the front-center channel and music/ambience in
L/R/surrounds, so the center channel is a nearly free "pre-separated" signal for the new era (spec §8.3).
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
from pathlib import Path

from .config import Settings
from .db import localize, record_roots

log = logging.getLogger(__name__)


def ffmpeg_extract(video: Path, out: Path, sample_rate: int, center: bool = False, timeout: int = 1800) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.flac")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vn", "-map", "0:a:0"]
    if center:
        cmd += ["-af", "pan=mono|c0=FC"]
    else:
        cmd += ["-ac", "1"]
    # -sample_fmt s16: the E-AC3/AAC decoders output float, which ffmpeg would otherwise store as 24-bit FLAC (2x size)
    cmd += ["-ar", str(sample_rate), "-sample_fmt", "s16", "-c:a", "flac", "-compression_level", "5", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[:800])
    tmp.replace(out)
    return out


def ffprobe_duration(path: Path) -> float | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=120)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def is_truncated(path: Path, want_s: float | None) -> bool:
    """Shorter than the video by more than max(5 s, 2%): a read that stopped early (US47 E08 on 2026-09-23 came out
    1,400 s of a 3,837 s episode after a NAS hiccup, and every later stage ran on the first 23 minutes)."""
    if not want_s or not path.exists():
        return False
    d = ffprobe_duration(path)
    return d is not None and d < want_s - max(5.0, 0.02 * want_s)


def extract_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, force: bool = False) -> dict:
    row = con.execute("SELECT video_path, audio_channels, duration_s FROM episodes WHERE version_season=? AND episode=?",
                      (vs, ep)).fetchone()
    if not row or not row["video_path"]:
        raise LookupError(f"{vs} E{ep:02d} not in inventory")
    record_roots(con, settings)
    video = localize(con, settings, row["video_path"])
    if not video.exists():
        raise FileNotFoundError(f"video not found on this machine: {video} (stored: {row['video_path']})")
    sr = settings.audio.sample_rate
    out: dict[str, str] = {}

    raw = settings.audio_path("raw", vs, ep)
    want = row["duration_s"]
    if raw.exists() and not force and is_truncated(raw, want):
        log.warning("%s E%02d raw is shorter than the video: extracting again", vs, ep)
        force = True
    if raw.exists() and not force:
        log.info("%s E%02d raw exists", vs, ep)
    else:
        log.info("%s E%02d extracting raw -> %s", vs, ep, raw)
        ffmpeg_extract(video, raw, sr, center=False)
    out["raw"] = str(raw)

    if (row["audio_channels"] or 0) >= 6 and settings.audio.make_center == "when_51":
        center = settings.audio_path("center", vs, ep)
        if center.exists() and not force:
            log.info("%s E%02d center exists", vs, ep)
        else:
            log.info("%s E%02d extracting center channel -> %s", vs, ep, center)
            ffmpeg_extract(video, center, sr, center=True)
        out["center"] = str(center)

    # sanity: extracted duration vs. probed duration; a truncated extraction stops the run instead of feeding it
    d = ffprobe_duration(raw)
    for name, path in out.items():
        if is_truncated(Path(path), want):
            raise RuntimeError(f"{vs} E{ep:02d} {name} audio is {ffprobe_duration(Path(path)):.0f} s of a {want:.0f} s video: "
                               f"the read stopped early (NAS?); run again with --force")
    if d and want and abs(d - want) > 2.0:
        log.warning("%s E%02d raw duration %.1f s differs from video %.1f s", vs, ep, d, want)

    con.execute("""UPDATE episodes SET audio_raw_path=?, audio_center_path=?, status='extracted'
                   WHERE version_season=? AND episode=?""", (out["raw"], out.get("center"), vs, ep))
    _merge_variants(con, vs, ep, out)
    con.commit()
    return out


def _merge_variants(con: sqlite3.Connection, vs: str, ep: int, new: dict[str, str]) -> None:
    import json

    r = con.execute("SELECT audio_variants FROM episodes WHERE version_season=? AND episode=?", (vs, ep)).fetchone()
    cur = json.loads(r["audio_variants"]) if r and r["audio_variants"] else {}
    cur.update(new)
    con.execute("UPDATE episodes SET audio_variants=? WHERE version_season=? AND episode=?", (json.dumps(cur), vs, ep))


def available_variants(settings: Settings, vs: str, ep: int) -> dict[str, Path]:
    return {v: p for v in ("raw", "center", "vocals", "vocals_center")
            if (p := settings.audio_path(v, vs, ep)).exists()}
