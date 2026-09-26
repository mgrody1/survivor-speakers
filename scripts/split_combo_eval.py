"""Nemotron finds where the voice changes; the bank checks that the two sides are different named people.
Reads reports/tmp/nemo_scores_vocals.json (scripts/nemotron_split_eval.py) and scores the bank at those cuts."""
import collections, json, random, sqlite3, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402
from survspk.cli import _resolver  # noqa: E402
from survspk.split_detect import scorer_for  # noqa: E402
from survspk.stage_align import load_audio  # noqa: E402
from survspk.stage_bank import DOMAINS  # noqa: E402
from survspk.stage_embed import Encoder  # noqa: E402

s = load_settings()
con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True); con.row_factory = sqlite3.Row
res = _resolver(s)
N = json.load(open(s.paths.work_root / "reports/tmp/nemo_scores_vocals.json"))
pos = []
for r in con.execute("SELECT base_utt_id, original, parts FROM utt_splits"):
    o, parts = json.loads(r["original"]), json.loads(r["parts"])
    a = con.execute("SELECT end_s FROM utterances WHERE utt_id=?", (parts[0],)).fetchone()
    if a:
        pos.append(dict(key=r["base_utt_id"], vs=o["version_season"], ep=o["episode"], a=o["start_s"], b=o["end_s"], dom=o.get("domain_hint"), truth=a["end_s"]))
seasons = sorted({p["vs"] for p in pos})
neg = [dict(key=r[0], vs=r[1], ep=r[2], a=r[3], b=r[4], dom=r[5]) for r in con.execute(
    f"""SELECT u.utt_id, u.version_season, u.episode, u.start_s, u.end_s, u.domain_hint FROM labels l JOIN utterances u USING (utt_id)
        WHERE l.source='human' AND u.segment='body' AND u.end_s - u.start_s >= 2.5 AND u.version_season IN ({','.join('?' * len(seasons))})
          AND json_extract(u.flags, '$.split_from') IS NULL""", seasons)]
random.Random(0).shuffle(neg); neg = neg[:400]
enc = Encoder(s.embed.model, s.embed.device)
sr = s.audio.sample_rate
by_ep = collections.defaultdict(list)
for u in pos + neg:
    by_ep[(u["vs"], u["ep"])].append(u)
B = {}
for (vs, ep), us in sorted(by_ep.items()):
    audio = load_audio(s.audio_path(s.audio.variant, vs, ep), sr)
    sc = scorer_for(s, con, vs, ep, res, as_of=max(ep - 1, 1))
    todo = [u for u in us if N[u["key"]]["cut"] is not None]
    sl = []
    for u in todo:
        c = N[u["key"]]["cut"]
        for x0, x1 in ((u["a"], c), (c, u["b"])):
            sl.append(audio[int(max(0, x0 - 0.1) * sr):int((x1 + 0.1) * sr)].astype(np.float32))
    if not sl:
        continue
    V = enc.encode([x if len(x) >= 4800 else np.pad(x, (0, 4800 - len(x))) for x in sl]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    for j, u in enumerate(todo):
        dom = u["dom"] if u["dom"] in DOMAINS else "field"
        rl, rr = dict(sc.rank(V[2 * j], dom)), dict(sc.rank(V[2 * j + 1], dom))
        if not rl or not rr:
            continue
        a, b = max(rl, key=rl.get), max(rr, key=rr.get)
        B[u["key"]] = (a != b, (rl[a] - rl.get(b, 0)) + (rr[b] - rr.get(a, 0)) if a != b else -1.0)
def hit(p, rule):
    return rule(p) and abs(N[p["key"]]["cut"] - p["truth"]) <= 1.0
print("rule                                          hand splits found   one-speaker flagged")
for tn in (0.5, 0.8, 1.0):
    for c in (None, 0.0, 0.2, 0.4):
        rule = (lambda u, tn=tn, c=c: N[u["key"]]["secs2"] >= tn and (c is None or (u["key"] in B and B[u["key"]][0] and B[u["key"]][1] >= c)))
        h = sum(hit(p, rule) for p in pos); fa = sum(rule(u) for u in neg)
        name = f"nemotron >= {tn}s" + ("" if c is None else f" + bank names differ (contrast >= {c})")
        print(f"  {name:44s} {h:3d}/{len(pos)}            {fa:3d}/{len(neg)} ({fa / len(neg):.1%})")
