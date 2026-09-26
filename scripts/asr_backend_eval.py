"""Compare ASR transcripts as anchor sources for one aligned episode: how many cue starts/ends each one hears, and
how close its cue starts land to the WhisperX first-word times already in the DB (the reference).

    .venv/bin/python scripts/asr_backend_eval.py US46 3 A.json B.json ...

Reads the DB read-only."""
import json, sqlite3, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402
from survspk.stage_align import cue_anchors  # noqa: E402


def main(vs, ep, *paths):
    s = load_settings()
    con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT cue_id, start_s, end_s, lines FROM cues WHERE version_season=? AND episode=? ORDER BY idx",
                       (vs, int(ep))).fetchall()
    cues = [(a, b, " ".join(l.get("text") or "" for l in json.loads(t))) for _, a, b, t in rows]
    ref = {}
    for i, (cid, *_rest) in enumerate(rows):
        w = con.execute("SELECT start_s FROM words WHERE cue_id=? AND start_s IS NOT NULL ORDER BY line_idx, word_idx LIMIT 1",
                        (cid,)).fetchone()
        if w:
            ref[i] = w[0]
    print(f"{vs} E{int(ep):02d}: {len(cues)} cues, {len(ref)} with a WhisperX first word")
    for p in paths:
        g = json.loads(Path(p).read_text())
        words = [tuple(w) for w in (g["words"] if isinstance(g, dict) else g)]
        a = cue_anchors(cues, words)
        sd = np.array([a.starts[i] - ref[i] for i in a.starts if i in ref])
        d = np.abs(sd)
        print(f"  {Path(p).name:48s} words {len(words):6d}  starts {len(a.starts):5d} ({len(a.starts)/len(cues):.0%})  "
              f"ends {len(a.ends):5d}  |start - whisperx| median {np.median(d):.2f}s p90 {np.percentile(d, 90):.2f}s  "
              f">1s {np.mean(d > 1):.1%}  signed median {np.median(sd):+.2f}s  |start - whisperx - bias| median {np.median(np.abs(sd - np.median(sd))):.2f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
