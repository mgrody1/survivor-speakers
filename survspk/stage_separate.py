"""Source separation (spec §7.2): Demucs vocals stem via demucs-mlx (Apple Silicon).

Input : work_root/<source>/<vs>/E<ep>.flac  where source = raw | center  (16 kHz mono FLAC from stage_extract)
Output: work_root/vocals[_center]/<vs>/E<ep>.flac  (16 kHz mono FLAC)

demucs-mlx resamples input to the model rate itself and accepts mono files. The stem comes back at the model
rate (44.1 kHz for htdemucs); we downmix to mono and let ffmpeg resample to 16 kHz FLAC.

API (README, verified 2026-09-07):
    from demucs_mlx import Separator
    separator = Separator(model="htdemucs")
    origin, stems = separator.separate_audio_file("song.wav")   # stems: {"drums","bass","other","vocals"} -> arrays
CLI fallback: demucs-mlx -n htdemucs -o <outdir> <input>  ->  <outdir>/htdemucs/<input stem>/vocals.wav
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


def _to_mono_float(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        return a
    # (channels, samples) or (samples, channels): channels is the small axis
    if a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:
        return a.mean(axis=0)
    return a.mean(axis=-1)


def _model_sample_rate(separator, default: int = 44100) -> int:
    for attr in ("samplerate", "sample_rate", "sr"):
        v = getattr(separator, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    model = getattr(separator, "model", None)
    for attr in ("samplerate", "sample_rate"):
        v = getattr(model, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return default


def separate_with_api(src: Path, model: str, tmp_dir: Path) -> Path:
    """Run demucs-mlx through its Python API; return path to a mono vocals WAV at the model rate."""
    import soundfile as sf
    from demucs_mlx import Separator

    separator = Separator(model=model)
    origin, stems = separator.separate_audio_file(str(src))
    if "vocals" not in stems:
        raise RuntimeError(f"demucs-mlx returned stems {list(stems)} without 'vocals'")
    vocals = _to_mono_float(stems["vocals"])
    sr = _model_sample_rate(separator)
    out = tmp_dir / "vocals_modelrate.wav"
    sf.write(out, vocals, sr, subtype="PCM_16")
    return out


def separate_with_cli(src: Path, model: str, tmp_dir: Path) -> Path:
    exe = shutil.which("demucs-mlx")
    if not exe:
        raise RuntimeError("demucs-mlx CLI not found and Python API import failed")
    r = subprocess.run([exe, "-n", model, "-o", str(tmp_dir), str(src)], capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:800])
    hits = list(tmp_dir.rglob("vocals.wav"))
    if not hits:
        raise RuntimeError(f"demucs-mlx CLI produced no vocals.wav under {tmp_dir}")
    return hits[0]


def to_flac_16k(src_wav: Path, out: Path, sample_rate: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.flac")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src_wav), "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16",
                        "-c:a", "flac", "-compression_level", "5", str(tmp)], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(r.stderr.strip()[:800])
    tmp.replace(out)
    return out


def separate_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, source: str | None = None,
                     force: bool = False) -> Path:
    source = source or settings.audio.separation_source
    src = settings.audio_path(source, vs, ep)
    if not src.exists():
        if source == "center":
            raise FileNotFoundError(f"{src} missing — center exists only for 5.1 sources; run extract first or use --source raw")
        raise FileNotFoundError(f"{src} missing — run extract first")
    variant = "vocals" if source == "raw" else "vocals_center"
    out = settings.audio_path(variant, vs, ep)
    if out.exists() and not force:
        log.info("%s E%02d %s exists", vs, ep, variant)
        return out
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="survspk_demucs_") as td:
        tmp_dir = Path(td)
        try:
            stem_wav = separate_with_api(src, settings.audio.demucs_model, tmp_dir)
        except ImportError as e:
            log.warning("demucs-mlx Python API unavailable (%s); trying CLI", e)
            stem_wav = separate_with_cli(src, settings.audio.demucs_model, tmp_dir)
        to_flac_16k(stem_wav, out, settings.audio.sample_rate)
    dt = time.time() - t0
    from .stage_extract import ffprobe_duration, _merge_variants

    dur = ffprobe_duration(out) or 0
    log.info("%s E%02d %s: %.0f s of audio separated in %.0f s (%.0fx realtime)", vs, ep, variant, dur, dt,
             (dur / dt) if dt else 0)
    col = "audio_vocals_path" if variant == "vocals" else "audio_vocals_center_path"
    con.execute(f"UPDATE episodes SET {col}=?, status='separated' WHERE version_season=? AND episode=?", (str(out), vs, ep))
    _merge_variants(con, vs, ep, {variant: str(out)})
    con.commit()
    return out
