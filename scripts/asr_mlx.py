"""Word-timed transcript of one episode with mlx-whisper on the Apple GPU, for the ASR anchors in `survspk align`.

Runs in its own throwaway environment so the project venv never changes (called by stage_align.asr_words_mlx):

    uv run --no-project --with mlx-whisper --with faster-whisper python scripts/asr_mlx.py AUDIO.npy OUT.json [HF_REPO] [--no-vad]

AUDIO.npy is float32 mono at 16 kHz. OUT.json: {"audio_s", "model", "vad", "words": [[word, start, end], ...]}.
Like the CPU path (faster-whisper vad_filter=True), silence and music are cut out with Silero VAD before decoding
and the word times are mapped back; without it Whisper drifts or invents words over long music beds (US31 E01:
3.5% of matched cue starts more than 1 s from WhisperX without VAD, 0.7% on the CPU path with it).
"""
import json, sys, time

import numpy as np


def main(src: str, out: str, repo: str = "mlx-community/whisper-small.en-mlx", *flags: str) -> int:
    import mlx_whisper

    t0 = time.time()
    x = np.load(src).astype(np.float32)
    vad = "--no-vad" not in flags
    y, tmap = x, None
    if vad:
        from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps

        chunks = get_speech_timestamps(x, VadOptions())
        if chunks:
            y = np.concatenate([x[c["start"]:c["end"]] for c in chunks])
            tmap = SpeechTimestampsMap(chunks, 16000)
    r = mlx_whisper.transcribe(y, path_or_hf_repo=repo, language="en", word_timestamps=True,
                               condition_on_previous_text=False, verbose=None)
    words = []
    for s in r["segments"]:
        for w in s.get("words") or []:
            a, b = float(w["start"]), float(w["end"])
            if tmap is not None:
                k = tmap.get_chunk_index(a)
                a, b = tmap.get_original_time(a, k), tmap.get_original_time(b, k)
            words.append([w["word"], a, b])
    with open(out, "w") as f:
        json.dump({"audio_s": round(len(x) / 16000, 1), "model": repo, "vad": vad, "words": words}, f)
    print(f"{len(words)} words from {len(x) / 16000 / 60:.0f} min of audio ({len(y) / 16000 / 60:.0f} min after VAD) "
          f"in {time.time() - t0:.0f}s with {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
