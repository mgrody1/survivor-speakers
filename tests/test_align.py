"""Offset/drift estimation on synthetic speech envelopes, and word partitioning."""

import numpy as np

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
