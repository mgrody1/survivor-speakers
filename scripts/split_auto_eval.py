"""Thresholds for automatic splits: the production path (episode change track from `survspk diarize` + bank names at
the cut) on the 24 hand splits and 400 human-labelled whole lines. Diarizes any episode that lacks a track."""
import collections, json, random, sqlite3, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survspk.config import load_settings  # noqa: E402
from survspk.cli import _resolver  # noqa: E402
from survspk.split_detect import scorer_for  # noqa: E402
from survspk.stage_align import load_audio  # noqa: E402
from survspk.stage_bank import DOMAINS  # noqa: E402
from survspk.stage_diarize import diarize_episode, line_change, load_track  # noqa: E402
from survspk.stage_embed import Encoder  # noqa: E402

s = load_settings()
con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True); con.row_factory = sqlite3.Row
res = _resolver(s)
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
R = {}
for (vs, ep), us in sorted(by_ep.items()):
    diarize_episode(s, vs, ep)
    tr = load_track(s, vs, ep)
    audio = load_audio(s.audio_path(s.audio.variant, vs, ep), sr)
    sc = scorer_for(s, con, vs, ep, res, as_of=max(ep - 1, 1))
    todo = []
    for u in us:
        sec2, cut = line_change(tr, u["a"], u["b"])
        R[u["key"]] = {"sec2": sec2, "cut": cut, "c": -1.0, "sl": 0, "sr": 0, "ml": 0, "mr": 0}
        if cut is not None and cut - u["a"] >= 0.5 and u["b"] - cut >= 0.5:
            todo.append((u, cut))
    if not todo:
        continue
    sl = []
    for u, cut in todo:
        for x0, x1 in ((u["a"], cut), (cut, u["b"])):
            x = audio[int(max(0, x0 - 0.1) * sr):int((x1 + 0.1) * sr)].astype(np.float32)
            sl.append(x if len(x) >= 4800 else np.pad(x, (0, 4800 - len(x))))
    V = enc.encode(sl).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    for j, (u, cut) in enumerate(todo):
        dom = u["dom"] if u["dom"] in DOMAINS else "field"
        rl, rr = sc.rank(V[2 * j], dom), sc.rank(V[2 * j + 1], dom)
        if not rl or not rr:
            continue
        a, b = rl[0][0], rr[0][0]
        dl, dr = dict(rl), dict(rr)
        if a != b:
            R[u["key"]].update(c=(dl[a] - dl.get(b, 0)) + (dr[b] - dr.get(a, 0)), sl=dl[a], sr=dr[b],
                               ml=dl[a] - (rl[1][1] if len(rl) > 1 else 0), mr=dr[b] - (rr[1][1] if len(rr) > 1 else 0),
                               side_min=min(cut - u["a"], u["b"] - cut))
json.dump(R, open(s.paths.work_root / "reports/tmp/split_auto_scores.json", "w"))
def rule(k, t2, tc, tside, tconf):
    r = R[k]
    return r["cut"] is not None and r["sec2"] >= t2 and r["c"] >= tc and r.get("side_min", 0) >= tside and min(r["sl"], r["sr"]) >= tconf
print(" 2nd voice  contrast  side>=  side score>=  | hand splits found  whole lines flagged")
for t2 in (0.8, 1.0, 1.5):
    for tc in (0.2, 0.4, 0.6, 0.8):
        for tside, tconf in ((0.5, 0.35), (1.0, 0.45), (1.0, 0.55)):
            h = sum(rule(p["key"], t2, tc, tside, tconf) and abs(R[p["key"]]["cut"] - p["truth"]) <= 1.0 for p in pos)
            f = [u["key"] for u in neg if rule(u["key"], t2, tc, tside, tconf)]
            print(f"   {t2:3.1f}s      {tc:3.1f}     {tside:3.1f}s     {tconf:4.2f}      |   {h:2d}/{len(pos)}           {len(f):3d}/{len(neg)} ({len(f)/len(neg):.1%})")
