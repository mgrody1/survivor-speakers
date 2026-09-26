"""How well does survspk.split_detect find the splits made by hand in the review app?

Positives: every line split in the app (utt_splits), with the cut the person chose. Negatives: long lines a person
labelled and did not split (so heard as one speaker). Reports, per contrast threshold, how many hand splits are found
(with the cut within 1 s of the person's) and how many single-speaker lines would be flagged.

    .venv/bin/python scripts/split_detect_eval.py [--neg 400] [--pyannote]
"""
import collections, json, random, sqlite3, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402
from survspk.cli import _resolver  # noqa: E402
from survspk.split_detect import SplitCfg, line_words, propose, scorer_for  # noqa: E402
from survspk.stage_align import load_audio  # noqa: E402
from survspk.stage_embed import Encoder  # noqa: E402

args = sys.argv[1:]
n_neg = int(args[args.index("--neg") + 1]) if "--neg" in args else 400
s = load_settings()
con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
res = _resolver(s)
cfg = SplitCfg.from_settings(s)
cfg.contrast = -0.5                      # score everything; thresholds are swept below
if "--min-side" in args:
    cfg.min_side_s = float(args[args.index("--min-side") + 1])
print("min_side_s", cfg.min_side_s)

pos = []
for r in con.execute("SELECT base_utt_id, original, parts FROM utt_splits"):
    o, parts = json.loads(r["original"]), json.loads(r["parts"])
    a = con.execute("SELECT end_s FROM utterances WHERE utt_id=?", (parts[0],)).fetchone()
    if a is None:
        continue
    pos.append({"utt_id": parts[0], "key": r["base_utt_id"], "version_season": o["version_season"], "episode": o["episode"],
                "start_s": o["start_s"], "end_s": o["end_s"], "domain_hint": o.get("domain_hint"), "truth": a["end_s"]})
seasons = sorted({p["version_season"] for p in pos})
neg_all = [dict(r) for r in con.execute(
    f"""SELECT u.utt_id, u.utt_id AS key, u.version_season, u.episode, u.start_s, u.end_s, u.domain_hint
        FROM labels l JOIN utterances u USING (utt_id)
        WHERE l.source='human' AND u.segment='body' AND u.end_s - u.start_s >= {cfg.min_utt_s}
          AND u.version_season IN ({','.join('?' * len(seasons))}) AND json_extract(u.flags, '$.split_from') IS NULL""", seasons)]
random.Random(0).shuffle(neg_all)
neg = neg_all[:n_neg]
print(f"{len(pos)} hand splits in {seasons}; {len(neg)} of {len(neg_all)} labelled one-speaker lines >= {cfg.min_utt_s}s")

enc = Encoder(s.embed.model, s.embed.device)
by_ep = collections.defaultdict(list)
for u in pos + neg:
    by_ep[(u["version_season"], u["episode"])].append(u)
scores = {}
t0 = time.time()
for (vs, ep), us in sorted(by_ep.items()):
    audio = load_audio(s.audio_path(s.audio.variant, vs, ep), s.audio.sample_rate)
    sc = scorer_for(s, con, vs, ep, res, as_of=max(ep - 1, 1))
    got = propose(us, lambda u: line_words(con, u), audio, s.audio.sample_rate, enc, sc, cfg)
    for u in us:
        scores[u["key"]] = got.get(u["utt_id"], (None, -1.0))
print(f"scored in {time.time() - t0:.0f}s")

print("\ncontrast  hand splits found (cut within 1 s)  found, cut elsewhere  one-speaker lines flagged")
for thr in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
    hit = near = 0
    for p in pos:
        prop, c = scores[p["key"]]
        if prop is not None and c >= thr:
            if abs(prop.t_cut - p["truth"]) <= 1.0:
                hit += 1
            else:
                near += 1
    fa = sum(1 for u in neg if scores[u["key"]][0] is not None and scores[u["key"]][1] >= thr)
    print(f"  {thr:4.2f}    {hit:3d}/{len(pos)}                            {near:3d}                  {fa:3d}/{len(neg)} ({fa / max(1, len(neg)):.1%})")
print("\nper hand split (contrast, cut error s, left->right):")
for p in pos:
    prop, c = scores[p["key"]]
    print(f"  {p['key']:22s} {p['end_s'] - p['start_s']:5.1f}s  " + (f"c={c:+.2f}  err={prop.t_cut - p['truth']:+.2f}s  {prop.left}->{prop.right}" if prop else "no two-voice cut"))

print("\nstrongest flags on labelled one-speaker lines (listen to these: some may be two speakers labelled as one):")
fl = sorted(((scores[u["key"]][1], u) for u in neg if scores[u["key"]][0] is not None), key=lambda x: -x[0])[:12]
for c, u in fl:
    pr = scores[u["key"]][0]
    print(f"  {u['key']:22s} {u['end_s'] - u['start_s']:5.1f}s c={c:+.2f} cut at {pr.t_cut:.1f}s {pr.left}->{pr.right}")
