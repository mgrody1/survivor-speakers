"""Diagnostic: where do a subtitle's cues really sit in the audio? Transcribe the episode with faster-whisper (word
timestamps), match ASR words to cue words, and print the subtitle->audio lag over the episode.

    .venv/bin/python scripts/asr_anchor.py US31 1 [--model small.en] [--variant vocals]

Writes reports/align/<vs>_E<ep>_anchors.csv (sub_t, audio_t per matched run) and caches the ASR words."""
import argparse, difflib, json, re, sqlite3, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402

TOK = re.compile(r"[a-z0-9']+")


def norm(s):
    return TOK.findall(s.lower().replace("’", "'"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vs"); ap.add_argument("ep", type=int)
    ap.add_argument("--model", default="small.en"); ap.add_argument("--variant", default="vocals")
    ap.add_argument("--min-run", type=int, default=4)
    a = ap.parse_args(argv)
    st = load_settings()
    out = st.paths.work_root / "reports" / "align"; out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{a.vs}_E{a.ep:02d}_asr_{a.model}_{a.variant}.json"
    if cache.exists():
        words = json.loads(cache.read_text()); words = words["words"] if isinstance(words, dict) else words
    else:
        from faster_whisper import WhisperModel
        t0 = time.time()
        m = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=16)
        segs, _ = m.transcribe(str(st.audio_path(a.variant, a.vs, a.ep)), language="en", word_timestamps=True,
                               vad_filter=True, condition_on_previous_text=False)
        words = [(w.word, w.start, w.end) for s in segs for w in (s.words or [])]
        cache.write_text(json.dumps(words))
        print(f"transcribed {len(words)} words in {time.time() - t0:.0f}s")
    con = sqlite3.connect(st.db_path)
    sub = []   # (token, sub_time)
    for s, e, lines in con.execute("SELECT start_s, end_s, lines FROM cues WHERE version_season=? AND episode=? ORDER BY idx",
                                   (a.vs, a.ep)):
        toks = norm(" ".join(l.get("text") or "" for l in json.loads(lines)))
        for i, t in enumerate(toks):
            sub.append((t, s + (e - s) * (i + 0.5) / len(toks)))
    asr = [(t, (ws + we) / 2) for w, ws, we in words for t in norm(w)]
    sm = difflib.SequenceMatcher(None, [t for t, _ in sub], [t for t, _ in asr], autojunk=False)
    rows = []
    for b in sm.get_matching_blocks():
        if b.size >= a.min_run:
            lags = [asr[b.b + k][1] - sub[b.a + k][1] for k in range(b.size)]
            rows.append((sub[b.a][1], float(np.median(lags)), b.size))
    csv = out / f"{a.vs}_E{a.ep:02d}_anchors.csv"
    csv.write_text("sub_t,lag_s,n\n" + "".join(f"{t:.2f},{l:.2f},{n}\n" for t, l, n in rows))
    print(f"{len(rows)} anchor runs covering {sum(r[2] for r in rows)}/{len(sub)} subtitle words")
    edges = np.arange(0, max(r[0] for r in rows) + 120, 120)
    for lo in edges[:-1]:
        L = [r[1] for r in rows if lo <= r[0] < lo + 120]
        if L:
            print(f"{lo:5.0f}-{lo + 120:5.0f}s  n={len(L):3d}  lag median {np.median(L):+6.2f}s  IQR [{np.percentile(L, 25):+.2f}, {np.percentile(L, 75):+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
