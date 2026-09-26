"""Offset/drift estimation on synthetic speech envelopes, and word partitioning."""

import numpy as np
import pytest

from survspk.stage_align import HOP_S, estimate_offset_drift, partition_words, rms_envelope, cue_signal


def _synthetic(total_s=2400.0, seed=0, offset=0.0, drift=0.0, n_utts=700):
    """Speech intervals in *audio* time; cues are the same intervals expressed in subtitle time,
    where t_audio = (1 + drift) * t_sub + offset."""
    rng = np.random.default_rng(seed)
    t = 30.0
    speech = []
    while t < total_s - 30 and len(speech) < n_utts:
        dur = rng.uniform(0.8, 4.0)
        speech.append((t, t + dur))
        t += dur + rng.uniform(0.2, 3.0)
    n = int(total_s / HOP_S)
    env = np.full(n, -6.0, dtype=np.float32)          # log-RMS floor (silence / music bed)
    for s, e in speech:
        env[int(s / HOP_S): int(e / HOP_S)] = -2.0 + rng.normal(0, 0.3)
    env += rng.normal(0, 0.4, n).astype(np.float32)   # noise
    a = 1.0 + drift
    cues = [((s - offset) / a, (e - offset) / a) for s, e in speech]
    return env, cues


def test_in_sync_is_detected_and_not_applied():
    env, cues = _synthetic()
    fit = estimate_offset_drift(env, cues)
    assert not fit.applied
    assert abs(fit.offset_s) < 0.25 and abs(fit.drift) < 1e-4


def test_constant_offset_recovered():
    env, cues = _synthetic(offset=3.7)
    fit = estimate_offset_drift(env, cues)
    assert fit.applied
    assert abs(fit.offset_s - 3.7) < 0.15, fit
    assert abs(fit.drift) < 3e-4, fit
    assert abs(fit.map(1000.0) - 1003.7) < 0.4


def test_negative_offset_recovered():
    env, cues = _synthetic(offset=-12.0)
    fit = estimate_offset_drift(env, cues)
    assert fit.applied and abs(fit.offset_s + 12.0) < 0.15, fit


def test_frame_rate_drift_recovered():
    # 25 fps subs on a 23.976 fps file: t_audio = 1.0427 * t_sub  (a 42-min episode ends ~100 s late)
    env, cues = _synthetic(offset=0.5, drift=0.0427)
    fit = estimate_offset_drift(env, cues)
    assert fit.applied, fit
    assert abs(fit.drift - 0.0427) < 0.002, fit
    assert abs(fit.map(2000.0) - (1.0427 * 2000 + 0.5)) < 1.0, fit


def test_envelope_and_cue_signal_shapes():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    audio[sr * 2: sr * 3] = 0.5
    env = rms_envelope(audio, sr)
    assert len(env) == int(10 / HOP_S)
    assert env[int(2.5 / HOP_S)] > env[int(1.0 / HOP_S)] + 5
    sig = cue_signal([(2.0, 3.0)], len(env))
    assert sig.sum() == 50


def test_partition_words():
    words = [{"word": w} for w in "a b c d e".split()]
    assert [[w["word"] for w in g] for g in partition_words(words, [2, 0, 3])] == [["a", "b"], [], ["c", "d", "e"]]
    # tokenizer produced one extra token: it lands on the last non-empty line
    words6 = [{"word": w} for w in "a b c d e f".split()]
    out = partition_words(words6, [2, 3, 0])
    assert [len(g) for g in out] == [2, 4, 0]


# ----------------------------------------------------------------------------- ASR anchors

from survspk.stage_align import OffsetFit, cue_anchors, fit_from_anchors, time_map  # noqa: E402


def _script(n_cues=300, seed=1):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(300)]
    cues, t = [], 5.0
    for _ in range(n_cues):
        n = int(rng.integers(3, 9))
        cues.append((t, t + 0.4 * n, " ".join(rng.choice(vocab, n))))
        t += 0.4 * n + float(rng.uniform(0.2, 1.5))
    return cues, rng


def _heard(cues, rng, lag_of, jitter=0.0, drop_every=11):
    """What ASR would hear: each cue's words spread over the cue, shifted by lag_of(t) plus per-cue jitter."""
    out = []
    for s, e, text in cues:
        toks = text.split()
        j = float(rng.normal(0, jitter)) if jitter else 0.0
        step = (e - s) / len(toks)
        out += [(w, s + k * step + lag_of(s) + j, s + (k + 1) * step + lag_of(s) + j) for k, w in enumerate(toks)]
    return [h for i, h in enumerate(out) if i % drop_every]


def test_heard_cues_use_their_own_times_and_the_rest_follow_local_lag():
    """US31 E01: the envelope fit applied -4.6 s; the true lag wandered between 0 and -1.8 s and each cue was off by
    its own amount on top. Heard boundaries come straight from ASR; the map follows the local lag."""
    cues, rng = _script()
    lag_of = lambda t: -1.5 if 600 <= t < 900 else 0.3  # noqa: E731
    heard = _heard(cues, rng, lag_of, jitter=0.6)
    anc = cue_anchors(cues, heard)
    assert len(anc.starts) > 0.6 * len(cues)
    i = next(iter(anc.starts))
    first = [h for h in heard if h[0] == cues[i][2].split()[0] and abs(h[1] - cues[i][0]) < 4]
    assert any(abs(anc.starts[i] - h[1]) < 1e-9 for h in first)
    m = time_map(0.0, 0.0, anc.knots)
    # the map is a rolling median over a handful of jittered cues: right on average, within a cue's jitter anywhere
    assert abs(np.median([m(t) - t for t in range(100, 580, 10)]) - 0.3) < 0.25
    assert abs(np.median([m(t) - t for t in range(620, 880, 10)]) + 1.5) < 0.25
    assert max(abs(m(t) - t - (-1.5 if 600 <= t < 900 else 0.3)) for t in range(100, 580, 10)) < 1.0
    fit = fit_from_anchors(anc, OffsetFit(-4.6, 0.001, [], True, "env"), len(cues), 0.3)
    assert fit is not None and fit.knots and fit.map(750.0) == m(750.0)
    assert time_map(-4.6, 0.001, None)(1000.0) == pytest.approx(1000 * 1.001 - 4.6)


def test_stray_matches_dropped_and_too_few_keeps_the_envelope():
    cues, rng = _script()
    heard = _heard(cues, rng, lambda t: 0.2)
    # the words of cue 50 heard again 60 s later, out of place
    s, e, text = cues[50]
    stray = [(w, s + 60 + k * 0.3, s + 60.3 + k * 0.3) for k, w in enumerate(text.split())]
    anc = cue_anchors(cues, sorted(heard + stray, key=lambda h: h[1]))
    assert all(abs(t - cues[i][0] - 0.2) < 0.05 for i, t in anc.starts.items())
    few = cue_anchors(cues, heard[:40])
    env = OffsetFit(-1.0, 0.0, [], True, "env")
    assert fit_from_anchors(few, env, len(cues), 0.3) is None


def test_mlx_anchor_transcript_falls_back_to_cpu_and_caches_by_model(tmp_path, monkeypatch):
    import subprocess

    from survspk import stage_align as sa
    assert sa._asr_tag("mlx", "mlx-community/whisper-small.en-mlx") == "mlx-whisper-small.en-mlx"
    assert sa._asr_tag("faster-whisper", "small.en") == "small.en"
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom"))
    monkeypatch.setattr(sa, "asr_words_faster_whisper",
                        lambda audio, model, sr, cache: seen.update(model=model, cache=cache) or [("hi", 0.0, 0.2)])
    cache = tmp_path / "US99_E01_asr_mlx-whisper-small.en-mlx_vocals.json"
    got = sa.asr_words_mlx(np.zeros(16000, np.float32), "mlx-community/whisper-small.en-mlx", 16000, cache)
    assert got == [("hi", 0.0, 0.2)] and seen["model"] == "small.en"
    assert seen["cache"].name == "US99_E01_asr_small.en_vocals.json" and not cache.exists()
    cache.write_text('{"audio_s": 1.0, "words": [["yo", 0.1, 0.3]]}')            # a cached GPU transcript is reused
    assert sa.asr_words_mlx(np.zeros(16000, np.float32), "mlx-community/whisper-small.en-mlx", 16000, cache) == [("yo", 0.1, 0.3)]
