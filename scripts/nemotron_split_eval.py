"""Same test as split_detect_eval.py, with NVIDIA Nemotron 3 Diarization (transformers port) as the change finder.
Each line is diarized in a window of up to 30 s around it; the score is how many seconds a second speaker holds inside
the line, and the cut is where the dominant speaker changes.

    uv run --no-project --with "git+https://github.com/huggingface/transformers" --with torch --with soundfile \
        --with numpy python scripts/nemotron_split_eval.py
"""
import json, random, sqlite3, subprocess, sys, time
import numpy as np, torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

W = "/Users/maxgrody/Documents/Claude/Projects/Stargazer/survivor_audio"
variant = sys.argv[1] if len(sys.argv) > 1 else "vocals"
con = sqlite3.connect(f"file:{W}/db/survspk.sqlite?mode=ro", uri=True); con.row_factory = sqlite3.Row
pos = []
for r in con.execute("SELECT base_utt_id, original, parts FROM utt_splits"):
    o, parts = json.loads(r["original"]), json.loads(r["parts"])
    a = con.execute("SELECT end_s FROM utterances WHERE utt_id=?", (parts[0],)).fetchone()
    if a:
        pos.append({"key": r["base_utt_id"], "vs": o["version_season"], "ep": o["episode"], "a": o["start_s"], "b": o["end_s"], "truth": a["end_s"]})
seasons = sorted({p["vs"] for p in pos})
neg = [dict(key=r[0], vs=r[1], ep=r[2], a=r[3], b=r[4]) for r in con.execute(
    f"""SELECT u.utt_id, u.version_season, u.episode, u.start_s, u.end_s FROM labels l JOIN utterances u USING (utt_id)
        WHERE l.source='human' AND u.segment='body' AND u.end_s - u.start_s >= 2.5 AND u.version_season IN ({','.join('?' * len(seasons))})
          AND json_extract(u.flags, '$.split_from') IS NULL""", seasons)]
random.Random(0).shuffle(neg); neg = neg[:400]      # same sample as split_detect_eval.py

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("nvidia/Nemotron-3-Diarization")
model = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization").to(dev).eval()
sr = proc.feature_extractor.sampling_rate
audio_cache = {}
def audio(vs, ep):
    k = (vs, ep)
    if k not in audio_cache:
        audio_cache.clear()
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{W}/{variant}/{vs}/E{ep:02d}.flac", "-ac", "1", "-ar", str(sr),
                              "-f", "f32le", "-"], capture_output=True).stdout
        audio_cache[k] = np.frombuffer(pcm, np.float32)
    return audio_cache[k]

full_cache = {}
def full(vs, ep):
    """Whole-episode pass (one forward, ~3 s on the M3 Ultra), as production would run it."""
    if (vs, ep) not in full_cache:
        full_cache.clear()
        x = audio(vs, ep)
        inp = proc(x, sampling_rate=sr, return_tensors="pt").to(dev)
        with torch.inference_mode():
            full_cache[(vs, ep)] = torch.sigmoid(model(**inp).logits)[0].float().cpu().numpy()
    return full_cache[(vs, ep)]

MODE = sys.argv[2] if len(sys.argv) > 2 else "window"
def score(u):
    if MODE == "full":
        P = full(u["vs"], u["ep"])
        w0 = max(0.0, u["a"] - 1.0)
        p = P[int(w0 * 100):int((u["b"] + 1.0) * 100)]
    else:
        x = audio(u["vs"], u["ep"])
        ctx = max(0.0, (28.0 - (u["b"] - u["a"])) / 2)
        w0 = max(0.0, u["a"] - min(ctx, 6.0)); w1 = min(len(x) / sr, u["b"] + min(ctx, 6.0))
        seg = x[int(w0 * sr):int(w1 * sr)]
        inp = proc(seg, sampling_rate=sr, return_tensors="pt").to(dev)
        with torch.inference_mode():
            p = torch.sigmoid(model(**inp).logits)[0].float().cpu().numpy()     # frames x 8, 10 ms
    t = w0 + np.arange(len(p)) * 0.01
    m = (t >= u["a"]) & (t < u["b"])
    q = p[m]; tt = t[m]
    if len(q) == 0:
        return 0.0, None
    act = q.max(1) >= 0.5
    dom = np.where(act, q.argmax(1), -1)
    secs = np.array([(dom == k).sum() * 0.01 for k in range(p.shape[1])])
    top = np.argsort(-secs)[:2]
    if secs[top[1]] <= 0:
        return 0.0, None
    s1, s2 = top
    # cut: the time that best separates s1 and s2 frames (fewest frames on the wrong side)
    idx = np.where((dom == s1) | (dom == s2))[0]
    best, cut = 1e9, None
    for j in range(1, len(idx)):
        left, right = dom[idx[:j]], dom[idx[j:]]
        e = min((left == s2).sum() + (right == s1).sum(), (left == s1).sum() + (right == s2).sum())
        if e < best:
            best, cut = e, 0.5 * (tt[idx[j - 1]] + tt[idx[j]])
    return float(secs[s2]), cut

t0 = time.time()
S = {u["key"]: score(u) for u in pos + neg}
print(f"{variant} {MODE}: scored {len(pos)} splits + {len(neg)} one-speaker lines in {time.time() - t0:.0f}s")
print("2nd speaker s  hand splits found (cut within 1 s)  found, cut elsewhere  one-speaker lines flagged")
for thr in (0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
    hit = sum(1 for p in pos if S[p["key"]][0] >= thr and S[p["key"]][1] is not None and abs(S[p["key"]][1] - p["truth"]) <= 1.0)
    near = sum(1 for p in pos if S[p["key"]][0] >= thr) - hit
    fa = sum(1 for u in neg if S[u["key"]][0] >= thr)
    print(f"  {thr:4.1f}        {hit:3d}/{len(pos)}                            {near:3d}                  {fa:3d}/{len(neg)} ({fa / len(neg):.1%})")
for p in pos:
    d, c = S[p["key"]]
    print(f"  {p['key']:22s} {p['b'] - p['a']:5.1f}s  2nd={d:4.1f}s  " + (f"err={c - p['truth']:+.2f}s" if c else "-"))
json.dump({k: {"secs2": v[0], "cut": v[1]} for k, v in S.items()},
          open(f"{W}/reports/tmp/nemo_scores_{variant}_{MODE}.json", "w"))
