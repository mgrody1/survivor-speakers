"""Alignment (spec §7.3).

1. Subtitle -> audio time map. First choice: ASR anchors. Transcribe sampled minutes of the audio with faster-whisper,
   match runs of ASR words to cue words, and take the median lag per 2-minute stretch; the map interpolates those
   knots (it follows slow timing wander and mid-episode cut differences). Fallback when too few anchors: a global
   offset + linear drift from cross-correlating a speech envelope with the "cue is on" signal. The envelope fit is
   weak on SDH files whose cues run back to back (US31 E01 took -4.6 s where the truth was about -0.5 s).
2. Word timestamps: WhisperX forced alignment of each cue's text against the (offset-corrected, padded)
   audio window. Words are what let stage_segment split shared cues and trim merged utterances.

Only `align_words_whisperx()` touches WhisperX; everything else is plain numpy/sqlite and unit-tested.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

HOP_S = 0.02  # 50 Hz envelope


# ----------------------------------------------------------------------------- audio helpers


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        # cheap linear resample (stages always write 16 kHz, so this is a safety net only)
        n = int(round(len(mono) * sample_rate / sr))
        mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
    return mono


def rms_envelope(audio: np.ndarray, sample_rate: int, hop_s: float = HOP_S) -> np.ndarray:
    hop = max(1, int(round(sample_rate * hop_s)))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    env = np.log(rms + 1e-6)
    return env.astype(np.float32)


def vad_envelope(audio: np.ndarray, sample_rate: int, hop_s: float = HOP_S, mode: int = 3) -> np.ndarray | None:
    """WebRTC VAD speech flags per 20 ms frame (1.0 = speech). Returns None if webrtcvad is not installed.
    Music fools it too, but far less than log-RMS: on US47E02 the global correlation confidence went from
    ~1 (RMS) to 7-9 (VAD)."""
    try:
        import webrtcvad
    except ImportError:
        return None
    hop = int(round(sample_rate * hop_s))
    if hop not in (int(sample_rate * 0.01), int(sample_rate * 0.02), int(sample_rate * 0.03)):
        return None
    v = webrtcvad.Vad(mode)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    n = len(pcm) // hop
    return np.fromiter((v.is_speech(pcm[i * hop:(i + 1) * hop].tobytes(), sample_rate) for i in range(n)),
                       dtype=np.float32, count=n)


def speech_envelope(audio: np.ndarray, sample_rate: int, hop_s: float = HOP_S) -> tuple[np.ndarray, str]:
    env = vad_envelope(audio, sample_rate, hop_s)
    if env is not None:
        return env, "webrtcvad"
    return rms_envelope(audio, sample_rate, hop_s), "logrms"


def cue_signal(cues: list[tuple[float, float]], n_frames: int, hop_s: float = HOP_S) -> np.ndarray:
    sig = np.zeros(n_frames, dtype=np.float32)
    for s, e in cues:
        a, b = int(s / hop_s), int(np.ceil(e / hop_s))
        if b > a and a < n_frames:
            sig[max(0, a): min(n_frames, b)] = 1.0
    return sig


# ----------------------------------------------------------------------------- offset / drift


@dataclass
class OffsetFit:
    offset_s: float          # b
    drift: float             # a - 1
    windows: list[dict]      # per-window {t_center, lag_s, confidence}
    applied: bool
    note: str
    knots: list | None = None   # [[t_sub, lag_s], ...] from ASR anchors; when set, map() interpolates these

    def map(self, t: float) -> float:
        return time_map(self.offset_s, self.drift, self.knots)(t)


def time_map(offset_s: float, drift: float, knots: list | None = None):
    """t_sub -> t_audio. Knots (sorted [t_sub, lag]) win over the linear fit; lag is held flat past the ends."""
    if knots:
        kt = np.array([k[0] for k in knots], float)
        kl = np.array([k[1] for k in knots], float)
        return lambda t: float(t + np.interp(t, kt, kl))
    a = 1.0 + (drift or 0.0)
    b = offset_s or 0.0
    return lambda t: a * t + b


def _global_lag(env: np.ndarray, cues: list[tuple[float, float]], a: float, max_lag: int, hop_s: float) -> tuple[int, float, float]:
    """Best lag (frames), confidence, and raw peak for cues rescaled by `a` against the whole episode."""
    sig = cue_signal([(s * a, e * a) for s, e in cues], len(env), hop_s)
    x = env - env.mean()
    y = sig - sig.mean()
    if not np.any(y):
        return 0, 0.0, 0.0
    nfft = 1 << (2 * len(x) - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft)), nfft)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]]) / (np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-9)
    k = int(np.argmax(cc))
    # Confidence = z-score of the peak against the far field (|lag - peak| > 5 s). Cues cover most of an
    # episode's timeline, so the true peak is several seconds wide; a narrow exclusion zone would compare the
    # peak with its own shoulder and report no confidence even when the fit is exact.
    far = int(5.0 / hop_s)
    mask = np.ones(len(cc), dtype=bool)
    mask[max(0, k - far): k + far + 1] = False
    rest = cc[mask]
    conf = (float(cc[k]) - float(rest.mean())) / (float(rest.std()) + 1e-9) if rest.size > 10 else 0.0
    return k - max_lag, conf, float(cc[k])


def estimate_offset_drift(env: np.ndarray, cues: list[tuple[float, float]], n_windows: int = 5,
                          window_s: float = 240.0, max_lag_s: float = 90.0, hop_s: float = HOP_S,
                          min_conf: float = 3.0, apply_min_offset_s: float = 0.25,
                          drift_range: float = 0.06) -> OffsetFit:
    """Fit t_audio = a * t_sub + b.

    Drift is found by scanning candidate rates `a` (±drift_range, e.g. 25 vs 23.976 fps = +4.3%), rescaling the
    cue signal and cross-correlating it with the audio envelope over the whole episode; the true rate gives the
    sharpest peak. The lag at that rate is the offset. Windowed lags are then computed at the chosen rate for
    the report (they should agree; disagreement means a cut differs mid-episode)."""
    n = len(env)
    max_lag = int(max_lag_s / hop_s)
    total_s = n * hop_s
    if n < 2 * max_lag + int(60 / hop_s) or not cues:
        return OffsetFit(0.0, 0.0, [], False, "audio too short or no cues")
    # coarse scan (0.2% steps) then fine scan (0.01% steps) around the best rate
    coarse = 1.0 + np.arange(-drift_range, drift_range + 1e-9, 0.002)
    best = max(((a, *_global_lag(env, cues, a, max_lag, hop_s)) for a in coarse), key=lambda r: r[3])
    fine = best[0] + np.arange(-0.002, 0.002 + 1e-9, 0.0001)
    a, lag, conf, peak = max(((a, *_global_lag(env, cues, a, max_lag, hop_s)) for a in fine), key=lambda r: r[3])
    offset = lag * hop_s
    drift = float(a - 1.0)
    # diagnostics: per-window lag at the chosen rate
    win = int(window_s / hop_s)
    margin = int(60 / hop_s)
    windows = []
    if n > win + 2 * margin:
        sig = cue_signal([(s * a, e * a) for s, e in cues], n, hop_s)
        for c in np.linspace(margin + win // 2, n - margin - win // 2, n_windows).astype(int):
            lo, hi = c - win // 2, c + win // 2
            y = sig[lo:hi]
            if y.sum() < 0.05 * len(y):
                continue
            # correlate the window directly (lag relative to the fitted rate; should be ~0)
            x = env[lo:hi] - env[lo:hi].mean()
            yy = y - y.mean()
            nfft = 1 << (2 * len(x) - 1).bit_length()
            cc = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(yy, nfft)), nfft)
            cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]]) / (np.sqrt((x ** 2).sum() * (yy ** 2).sum()) + 1e-9)
            k = int(np.argmax(cc))
            windows.append({"t_center": float(c * hop_s), "lag_s": float((k - max_lag) * hop_s), "peak": float(cc[k])})
    if conf < min_conf:
        return OffsetFit(0.0, 0.0, windows, False,
                         f"no confident correlation (conf {conf:.1f}); subtitle assumed in sync")
    note = f"rate {a:.4f}, offset {offset:+.2f}s, conf {conf:.1f}, peak {peak:.3f}"
    good = [w for w in windows if w["peak"] >= 0.5 * peak]   # ignore windows with no real correlation
    if len(good) >= 2:
        spread = max(w["lag_s"] for w in good) - min(w["lag_s"] for w in good)
        if spread > 1.5:
            note += f"; WARNING window lags spread {spread:.1f}s over {len(good)} good windows (mid-episode cut difference?)"
    # drift needs a bigger effect than offset to be applied: the fine scan resolves 1e-4 (≈0.4 s over an episode)
    applied = abs(offset) >= apply_min_offset_s or abs(drift) * total_s >= max(0.6, 2 * apply_min_offset_s)
    if not applied:
        return OffsetFit(0.0, 0.0, windows, False, note + "; below threshold, not applied")
    return OffsetFit(float(offset), drift, windows, True, note)


# ----------------------------------------------------------------------------- ASR anchors

_TOK = re.compile(r"[a-z0-9']+")


def _norm(text: str) -> list[str]:
    return _TOK.findall(text.lower().replace("\u2019", "'"))


@dataclass
class CueAnchors:
    starts: dict            # cue index -> audio time of its first word (when ASR heard it in a matched run)
    ends: dict              # cue index -> audio time where its last word ends
    knots: list             # [[t_sub, lag_s]] rolling median of the boundary lags, for everything unmatched


def cue_anchors(cues: list[tuple[float, float, str]], heard: list[tuple[str, float, float]], min_run: int = 3,
                half_window: int = 7, max_dev_s: float = 4.0) -> CueAnchors:
    """Match ASR words to cue words (runs of >= min_run identical tokens, in order). A cue whose first word is in a
    run gets that word's audio start; one whose last word is gets its audio end. Subtitle timing is loose per cue
    (US31: half the cues off by more than 0.65 s, a tenth by 2 s), so matched boundaries are used directly and the
    rest follow a rolling median of neighbouring lags. A boundary more than `max_dev_s` from that median is a stray
    match and is dropped."""
    import difflib

    sub = []   # (token, cue index, position, n tokens)
    for ci, (_, _, text) in enumerate(cues):
        toks = _norm(text)
        sub += [(t, ci, k, len(toks)) for k, t in enumerate(toks)]
    asr = [(t, ws, we) for w, ws, we in heard for t in _norm(w)]
    sm = difflib.SequenceMatcher(None, [x[0] for x in sub], [x[0] for x in asr], autojunk=False)
    bounds = []   # (t_sub, lag, kind, cue index, t_audio)
    for bl in sm.get_matching_blocks():
        if bl.size < min_run:
            continue
        for k in range(bl.size):
            _, ci, pos, n = sub[bl.a + k]
            _, ws, we = asr[bl.b + k]
            if pos == 0:
                bounds.append((cues[ci][0], ws - cues[ci][0], "s", ci, ws))
            if pos == n - 1:
                bounds.append((cues[ci][1], we - cues[ci][1], "e", ci, we))
    if not bounds:
        return CueAnchors({}, {}, [])
    bounds.sort()
    lags = np.array([x[1] for x in bounds])
    roll = np.array([np.median(lags[max(0, i - half_window):i + half_window + 1]) for i in range(len(lags))])
    starts, ends, knots = {}, {}, []
    for (t, lag, kind, ci, ta), r in zip(bounds, roll):
        if abs(lag - r) > max_dev_s:
            continue
        (starts if kind == "s" else ends)[ci] = ta
        if not knots or round(t, 2) > knots[-1][0]:
            knots.append([round(t, 2), round(float(r), 3)])
    return CueAnchors(starts, ends, knots)


def asr_words_faster_whisper(audio: np.ndarray, model: str = "small.en", sample_rate: int = 16000,
                             cache: Path | None = None) -> list[tuple[str, float, float]]:
    """(word, start, end) in audio time for the whole episode. Cached as JSON next to the other align reports
    (about 5 minutes for an hour of audio on the M3 Ultra CPU)."""
    import os

    total = round(len(audio) / sample_rate, 1)
    if cache is not None and cache.exists():
        got = json.loads(cache.read_text())
        if isinstance(got, list) or abs(got.get("audio_s", total) - total) < 1.0:
            return [tuple(w) for w in (got if isinstance(got, list) else got["words"])]
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 4))
    segs, _ = m.transcribe(audio.astype(np.float32), language="en", word_timestamps=True, vad_filter=True,
                           condition_on_previous_text=False)
    words = [(w.word, float(w.start), float(w.end)) for sg in segs for w in (sg.words or [])]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"audio_s": total, "model": model, "words": words}))
    return words


def asr_words_mlx(audio: np.ndarray, model: str = "mlx-community/whisper-small.en-mlx", sample_rate: int = 16000,
                  cache: Path | None = None) -> list[tuple[str, float, float]]:
    """Same as asr_words_faster_whisper, on the Apple GPU with mlx-whisper (scripts/asr_mlx.py, run in a throwaway
    `uv run --no-project` environment so the project venv never changes). Falls back to faster-whisper small.en on
    the CPU if the subprocess fails."""
    import subprocess
    import tempfile

    total = round(len(audio) / sample_rate, 1)
    if cache is not None and cache.exists():
        got = json.loads(cache.read_text())
        if abs(got.get("audio_s", total) - total) < 1.0:
            return [tuple(w) for w in got["words"]]
    if sample_rate != 16000:
        raise ValueError("mlx-whisper expects 16 kHz audio")
    script = Path(__file__).resolve().parents[1] / "scripts" / "asr_mlx.py"
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "a.npy", Path(td) / "w.json"
        np.save(src, audio.astype(np.float32))
        r = subprocess.run(["uv", "run", "--no-project", "--with", "mlx-whisper", "--with", "faster-whisper",  # (its VAD)
                            "python", str(script), str(src), str(out), model], capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
            log.warning("mlx-whisper failed (%s); falling back to faster-whisper small.en on the CPU", tail)
            fb = cache.with_name(cache.name.replace(_asr_tag("mlx", model), "small.en")) if cache is not None else None
            return asr_words_faster_whisper(audio, "small.en", sample_rate, fb)
        got = json.loads(out.read_text())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(got))
    return [tuple(w) for w in got["words"]]


def _asr_tag(backend: str, model: str) -> str:
    """The model's part of the cached transcript's file name."""
    return model if backend != "mlx" else "mlx-" + model.rstrip("/").split("/")[-1]


def fit_from_anchors(anc: CueAnchors, env_fit: OffsetFit, n_cues: int, min_share: float) -> OffsetFit | None:
    """The anchor map, or None when too few cue boundaries matched to trust it."""
    share = len(anc.starts) / max(1, n_cues)
    if share < min_share or len(anc.knots) < 10:
        return None
    lags = np.array([k[1] for k in anc.knots])
    note = (f"ASR anchors: {len(anc.starts)}/{n_cues} cue starts and {len(anc.ends)} ends heard, local lag median "
            f"{np.median(lags):+.2f}s range [{lags.min():+.2f}, {lags.max():+.2f}]; envelope said {env_fit.note}")
    return OffsetFit(float(np.median(lags)), 0.0, env_fit.windows, True, note, anc.knots)


# ----------------------------------------------------------------------------- WhisperX


def align_words_whisperx(audio: np.ndarray, segments: list[dict], device: str, sample_rate: int = 16000) -> list[dict]:
    """segments: [{"start","end","text"}] in audio time. Returns one {"words": [...]} per input segment, in order.

    WhisperX's align() splits each input segment into sentences before returning (1450 cues came back as 1799
    segments), so the only safe mapping is one align() call per cue and reading the flat `word_segments` list.
    Words WhisperX could not place lack start/end."""
    import whisperx

    try:
        model_a, meta = whisperx.load_align_model(language_code="en", device=device)
    except Exception as e:  # noqa: BLE001
        if device != "cpu":
            log.warning("align model on %s failed (%s); using cpu", device, e)
            device = "cpu"
            model_a, meta = whisperx.load_align_model(language_code="en", device="cpu")
        else:
            raise
    out: list[dict] = []
    n_fail = 0
    for seg in segments:
        try:
            res = whisperx.align([dict(seg)], model_a, meta, audio, device, return_char_alignments=False)
            words = [w for w in (res.get("word_segments") or []) if isinstance(w, dict) and w.get("word") is not None]
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 3:
                log.warning("align failed for segment at %.1fs (%s)", seg["start"], e)
            words = []
        out.append({"words": words})
    if n_fail:
        log.warning("%d/%d segments raised during alignment", n_fail, len(segments))
    return out


# ----------------------------------------------------------------------------- stage


def _cue_texts(lines: list[dict]) -> tuple[str, list[int]]:
    """Joined cue text and per-line whitespace token counts (WhisperX splits words on spaces)."""
    counts = [len(l["text"].split()) if l["text"] else 0 for l in lines]
    text = " ".join(l["text"] for l in lines if l["text"])
    return text, counts


def partition_words(words: list[dict], counts: list[int]) -> list[list[dict]]:
    """Split the flat WhisperX word list back into per-line lists using the token counts."""
    out, i = [], 0
    for c in counts:
        out.append(words[i:i + c])
        i += c
    if i < len(words):   # tokenizer disagreement: attach leftovers to the last non-empty line
        for grp in reversed(out):
            if grp or counts[out.index(grp)] > 0:
                grp.extend(words[i:])
                break
    return out


def align_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str | None = None,
                  force: bool = False, aligner=align_words_whisperx, asr=asr_words_faster_whisper) -> dict:
    variant = variant or settings.audio.variant
    audio_path = settings.audio_path(variant, vs, ep)
    if not audio_path.exists():
        alt = settings.audio_path("raw", vs, ep)
        if not alt.exists():
            raise FileNotFoundError(f"no audio for {vs} E{ep:02d}; run extract/separate first")
        log.warning("%s missing; aligning against raw", variant)
        audio_path = alt
    cfg = settings.align
    sr = settings.audio.sample_rate
    rows = con.execute("SELECT cue_id, idx, start_s, end_s, lines FROM cues WHERE version_season=? AND episode=? ORDER BY idx",
                       (vs, ep)).fetchall()
    if not rows:
        raise LookupError(f"no cues for {vs} E{ep:02d}; run ingest-subs")
    prev = con.execute("SELECT align_stats FROM episodes WHERE version_season=? AND episode=?", (vs, ep)).fetchone()[0]
    prev_stats = json.loads(prev) if prev else None
    # "already aligned" only for the same subtitle: a re-fetched file (fetch-subs) reuses cue ids, so words from the old
    # file would otherwise pass for this one (US47 E08: an ASR sidecar replaced by SDH kept the old offset and drift)
    if not force and prev_stats and prev_stats.get("n_cues") == len(rows) \
            and con.execute("SELECT 1 FROM words WHERE cue_id=? LIMIT 1", (rows[0]["cue_id"],)).fetchone():
        log.info("%s E%02d already aligned", vs, ep)
        return prev_stats

    t0 = time.time()
    audio = load_audio(audio_path, sr)
    env, env_kind = speech_envelope(audio, sr)
    fit = estimate_offset_drift(env, [(r["start_s"], r["end_s"]) for r in rows], cfg.offset_windows, cfg.offset_window_s,
                                cfg.max_lag_s, apply_min_offset_s=cfg.apply_min_offset_s)
    log.info("%s E%02d offset=%+.2fs drift=%+.4f via %s on %s (%s)", vs, ep, fit.offset_s, fit.drift, env_kind, variant, fit.note)
    anc = CueAnchors({}, {}, [])
    if cfg.anchor != "never" and asr is not None:
        cue_text = [(r["start_s"], r["end_s"], " ".join(l.get("text") or "" for l in json.loads(r["lines"]))) for r in rows]
        backend = cfg.anchor_backend if asr is asr_words_faster_whisper else "custom"
        model = cfg.anchor_mlx_model if backend == "mlx" else cfg.anchor_model
        tag = _asr_tag(backend, model)
        cache = settings.paths.work_root / "reports" / "align" / f"{vs}_E{ep:02d}_asr_{tag}_{variant}.json"
        try:
            anc = cue_anchors(cue_text, (asr_words_mlx if backend == "mlx" else asr)(audio, model, sr, cache))
            afit = fit_from_anchors(anc, fit, len(rows), cfg.anchor_min_share)
        except ImportError as e:
            log.warning("ASR anchors unavailable (%s); keeping the envelope fit", e)
            afit = None
        if afit is not None:
            fit = afit
            log.info("%s E%02d %s", vs, ep, fit.note)
        else:
            log.warning("%s E%02d too few ASR anchors (%d cue starts); keeping the envelope fit", vs, ep, len(anc.starts))
            anc = CueAnchors({}, {}, [])

    # build segments (speech cues only), in audio time, padded
    total = len(audio) / sr
    segs, seg_rows = [], []
    n_heard = 0
    for i, r in enumerate(rows):
        lines = json.loads(r["lines"])
        text, counts = _cue_texts(lines)
        if not text.strip():
            continue
        s0, e0 = anc.starts.get(i, fit.map(r["start_s"])), anc.ends.get(i, fit.map(r["end_s"]))
        if e0 - s0 < 0.2:   # a heard start with a mapped end (or the reverse) can cross; trust the map for both
            s0, e0 = fit.map(r["start_s"]), fit.map(r["end_s"])
        else:
            n_heard += int(i in anc.starts or i in anc.ends)
        s = max(0.0, s0 - cfg.pad_s)
        e = min(total, e0 + cfg.pad_s)
        if e - s < 0.2:
            continue
        segs.append({"start": s, "end": e, "text": text})
        seg_rows.append((r, lines, counts))

    aligned = aligner(audio, segs, cfg.device, sr) if segs else []

    con.execute("DELETE FROM words WHERE cue_id IN (SELECT cue_id FROM cues WHERE version_season=? AND episode=?)", (vs, ep))
    n_ok = n_words = n_placed = 0
    scores = []
    ins = []
    for (r, lines, counts), seg in zip(seg_rows, aligned):
        words = seg.get("words", []) or []
        per_line = partition_words(words, counts)
        placed = 0
        for li, wl in enumerate(per_line):
            for wi, w in enumerate(wl):
                st, en, sc = w.get("start"), w.get("end"), w.get("score")
                if st is not None and en is not None:
                    placed += 1
                    if sc is not None:
                        scores.append(float(sc))
                ins.append((r["cue_id"], li, wi, w.get("word"), st, en, sc))
        n_words += len(words)
        n_placed += placed
        n_ok += int(len(words) > 0 and placed >= 0.5 * len(words))
    con.executemany("INSERT OR REPLACE INTO words (cue_id, line_idx, word_idx, word, start_s, end_s, score) VALUES (?,?,?,?,?,?,?)", ins)
    stats = {
        "variant": variant, "n_cues": len(rows), "n_speech_cues": len(segs), "n_cues_aligned_ok": n_ok,
        "n_words": n_words, "n_words_placed": n_placed, "mean_word_score": float(np.mean(scores)) if scores else None,
        "offset_s": fit.offset_s, "drift": fit.drift, "offset_note": fit.note, "envelope": env_kind, "windows": fit.windows,
        "knots": fit.knots, "n_cue_starts_heard": len(anc.starts), "n_cue_ends_heard": len(anc.ends),
        "n_segments_from_asr": n_heard,
        "seconds": round(time.time() - t0, 1),
    }
    con.execute("""UPDATE episodes SET align_offset_s=?, align_drift=?, align_stats=?, status='aligned'
                   WHERE version_season=? AND episode=?""", (fit.offset_s, fit.drift, json.dumps(stats), vs, ep))
    con.commit()
    log.info("%s E%02d aligned: %d/%d speech cues ok, %d/%d words placed, mean score %s, %.0fs", vs, ep, n_ok, len(segs),
             n_placed, n_words, f"{stats['mean_word_score']:.2f}" if scores else "n/a", stats["seconds"])
    return stats
