"""Speaker-change track per episode (Nemotron 3 Diarization, frame level) and what it says about one line.

The diarizer gives anonymous speakers, so it is used only for *where* the voice changes inside a line; the bank says
*who* each side is (survspk.split_detect). The model runs in a throwaway uv environment through
scripts/diarize_nemotron.py because its transformers support is not released yet.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)
FRAME_S = 0.01
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diarize_nemotron.py"


def diar_path(settings: Settings, vs: str, ep: int) -> Path:
    return settings.paths.work_root / "diar" / vs / f"E{ep:02d}.npz"


def diarize_episode(settings: Settings, vs: str, ep: int, variant: str | None = None, force: bool = False) -> Path | None:
    """Write the episode's change track; None (with a warning) when the diarizer cannot run."""
    cfg = settings.raw.get("diarize", {}) or {}
    out = diar_path(settings, vs, ep)
    if out.exists() and not force:
        return out
    audio = settings.audio_path(variant or cfg.get("variant") or settings.audio.variant, vs, ep)
    if not audio.exists():
        log.warning("%s E%02d: no %s audio to diarize", vs, ep, audio.parent.name)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "--no-project"]
    for dep in cfg.get("deps", ["git+https://github.com/huggingface/transformers", "torch", "numpy", "librosa"]):
        cmd += ["--with", dep]
    cmd += ["python", str(SCRIPT), str(audio), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        log.warning("%s E%02d: diarizer failed (%s); two-voice suggestions are skipped", vs, ep,
                    (r.stderr or r.stdout).strip().splitlines()[-1:] if (r.stderr or r.stdout) else r.returncode)
        return None
    log.info("%s E%02d diarized: %s", vs, ep, (r.stdout or "").strip().splitlines()[-1:])
    return out


@dataclass
class Track:
    starts: np.ndarray
    offs: np.ndarray        # frame offset of each window in dom
    lens: np.ndarray
    dom: np.ndarray

    def window_for(self, a: float, b: float) -> int:
        """The window that holds [a, b] with the most room on both sides."""
        best, k = -1e9, 0
        for i, (s, n) in enumerate(zip(self.starts, self.lens)):
            e = s + n * FRAME_S
            room = min(a - s, e - b)
            if room > best:
                best, k = room, i
        return k


def load_track(settings: Settings, vs: str, ep: int) -> Track | None:
    p = diar_path(settings, vs, ep)
    if not p.exists():
        return None
    z = np.load(p)
    lens = z["lens"].astype(np.int64)
    return Track(z["starts"].astype(np.float64), np.concatenate([[0], np.cumsum(lens)[:-1]]), lens, z["dom"])


def line_change(track: Track, a: float, b: float) -> tuple[float, float | None]:
    """(seconds held by the second-most-present voice inside [a, b], the time that best separates the two voices).
    0 and None when only one voice is heard."""
    k = track.window_for(a, b)
    s0, off, n = track.starts[k], track.offs[k], track.lens[k]
    i0, i1 = max(0, int((a - s0) / FRAME_S)), min(n, int((b - s0) / FRAME_S))
    dom = track.dom[off + i0: off + i1].astype(np.int64)
    if len(dom) == 0:
        return 0.0, None
    counts = np.bincount(dom[dom >= 0], minlength=8) if (dom >= 0).any() else np.zeros(8, int)
    order = np.argsort(-counts)
    s1, s2 = order[0], order[1]
    if counts[s2] == 0:
        return 0.0, None
    idx = np.where((dom == s1) | (dom == s2))[0]
    lab = (dom[idx] == s2).astype(np.int64)
    # cut j: frames before it are one voice, after it the other; errors = misplaced frames, either orientation
    c2 = np.concatenate([[0], np.cumsum(lab)])          # s2 frames among the first j
    c1 = np.arange(len(idx) + 1) - c2                    # s1 frames among the first j
    err = np.minimum(c2 + (c1[-1] - c1), c1 + (c2[-1] - c2))[1:-1]
    j = int(np.argmin(err)) + 1
    cut = s0 + (i0 + 0.5 * (idx[j - 1] + idx[j])) * FRAME_S
    return float(counts[s2] * FRAME_S), float(cut)
