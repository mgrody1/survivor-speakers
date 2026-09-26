"""If the diarizer cuts a one-speaker line (lenient rule), do both parts still end up with that speaker?
For each labelled one-speaker line the lenient rule would cut: embed both parts and apply assign's voice-split test
against the person's label (a part is cut out only if its own voice confidently names someone else by >= margin)."""
import json, sqlite3, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402
from survspk.cli import _resolver  # noqa: E402
from survspk.split_detect import scorer_for  # noqa: E402
from survspk.stage_align import load_audio  # noqa: E402
from survspk.stage_assign import Thresholds, decide  # noqa: E402
from survspk.stage_bank import DOMAINS  # noqa: E402
from survspk.stage_embed import Encoder  # noqa: E402

s = load_settings(); th = Thresholds.from_settings(s)
con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True); con.row_factory = sqlite3.Row
R = json.load(open(s.paths.work_root / "reports/tmp/split_auto_scores.json"))
split_keys = {r[0] for r in con.execute("SELECT base_utt_id FROM utt_splits")}
enc = Encoder(s.embed.model, s.embed.device); sr = s.audio.sample_rate; res = _resolver(s)
rows = []
for k, r in R.items():
    if k in split_keys or r["cut"] is None or r["sec2"] < 0.5:
        continue
    u = con.execute("SELECT u.version_season vs, u.episode ep, u.start_s a, u.end_s b, u.domain_hint dom, l.speaker_id lab FROM utterances u JOIN labels l USING (utt_id) WHERE u.utt_id=?", (k,)).fetchone()
    if u is None or min(r["cut"] - u["a"], u["b"] - r["cut"]) < 0.5:
        continue
    rows.append((k, dict(u), r["cut"]))
by = {}
for k, u, cut in rows:
    by.setdefault((u["vs"], u["ep"]), []).append((k, u, cut))
n_same = n_short = 0; detail = []
for (vs, ep), items in by.items():
    audio = load_audio(s.audio_path(s.audio.variant, vs, ep), sr)
    sc = scorer_for(s, con, vs, ep, res, as_of=max(ep - 1, 1))
    sl = []
    for k, u, cut in items:
        for x0, x1 in ((u["a"], cut), (cut, u["b"])):
            x = audio[int(max(0, x0 - 0.1) * sr):int((x1 + 0.1) * sr)].astype(np.float32)
            sl.append(x if len(x) >= 4800 else np.pad(x, (0, 4800 - len(x))))
    V = enc.encode(sl).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    for j, (k, u, cut) in enumerate(items):
        dom = u["dom"] if u["dom"] in DOMAINS else "field"
        bad = []
        for side, v, dur in ((0, V[2 * j], cut - u["a"]), (1, V[2 * j + 1], u["b"] - cut)):
            rank = sc.rank(v, dom); d = dict(rank)
            sd = decide(rank, th)
            if dur >= th.purity_min_s and sd.pred and sd.pred != u["lab"] and d[sd.pred] - d.get(u["lab"], 0) >= th.voice_split_margin:
                bad.append(side)
        n_same += not bad
        detail.append((k, round(cut - u["a"], 1), round(u["b"] - u["a"], 1), bad))
print(f"{len(detail)} labelled one-speaker lines the lenient rule would cut; both parts stay with the labelled speaker in {n_same}")
for d in detail:
    if d[3]:
        print("  part(s) would move to another voice:", d)
