"""Chyron frame grabbing (M0: fix crop regions). OCR + matching comes in M2."""

from __future__ import annotations

import subprocess
from pathlib import Path


def grab_frame(video: Path, t_s: float, out_png: Path, crop_frac: tuple[float, float] | None = None,
               scale_w: int = 960, timeout: int = 120) -> Path:
    """Grab one frame at t_s. If crop_frac=(y0,y1) fractions of height are given, crop that band."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"scale={scale_w}:-2"]
    if crop_frac:
        y0, y1 = crop_frac
        vf.append(f"crop=iw:ih*{y1 - y0:.3f}:0:ih*{y0:.3f}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t_s:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", ",".join(vf), str(out_png)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return out_png


def grab_contact_sheet(video: Path, times: list[float], out_png: Path, scale_w: int = 480, timeout: int = 300) -> Path:
    """Several frames tiled into one image (one row per time) — quick visual check of chyron timing."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = []
    for i, t in enumerate(times):
        p = out_png.with_suffix(f".{i}.png")
        grab_frame(video, t, p, scale_w=scale_w, timeout=timeout)
        tmp.append(p)
    inputs = sum([["-i", str(p)] for p in tmp], [])
    n = len(tmp)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"vstack=inputs={n}[v]"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[v]", str(out_png)],
                       capture_output=True, text=True, timeout=timeout)
    for p in tmp:
        p.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return out_png
