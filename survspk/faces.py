"""Faces as speaker evidence (prototype): who is on screen while a line is spoken.

  extract   decode the episode at 2 frames a second (VideoToolbox), find faces and embed them with InsightFace
            buffalo_l (SCRFD detector + ArcFace, 512-d). Faces under MIN_FACE_H of the frame height are skipped.
            Saves work_root/faces/<vs>E<ep>.npz: frame time, box, detector score, unit embedding (float16).
            Numbers only; nothing leaves this machine.
  gallery   each castaway's reference faces, from
              name cards   the largest face on screen while their card shows (chyron_hits t_s..t_end_s)
              trusted lines  lines labelled by a caption NAME: or a person, in confessionals (one face fills the
                           screen), largest face in each frame; the host from his caption-named lines anywhere
            Line faces must match the castaway's name cards (VERIFY_SIM; voice often runs over B-roll of others);
            the host, with no card, keeps the faces around his most common one. A castaway's faces far from their
            own median (cos < OUTLIER_COS) are dropped.
  score     per line: the frames inside it; in each, the largest face; its similarity to each candidate (mean of the
            3 best gallery matches). The line's pick is the candidate most frames choose; `face_sim` its mean
            similarity on those frames, `face_share` the share of frames that chose it, `face_margin` the gap to the
            runner-up.

Evaluation (`survspk faces-eval`): leave one episode out. The gallery for episode e uses name cards from any episode
but trusted lines only from the other episodes, and is scored on e's trusted lines (caption names and people; name
card labels are left out of the truth because cards also build the gallery).

Needs InsightFace (research-use weights):
    uv run --with insightface --with onnxruntime --with opencv-python-headless survspk faces US47 --episodes 1-14
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FPS = 2.0
SCALE_W = 960
MIN_FACE_H = 0.08          # faces shorter than 8% of the frame are too small to name
OUTLIER_COS = 0.35
TOPK = 3
TRUSTED = ("sdh", "human")
CONF_MIN_AREA = 0.02       # a gallery frame from a confessional needs its largest face to fill 2% of the frame
VERIFY_SIM = 0.5           # a line's face joins the gallery only if it matches the castaway's name cards this well


# ----------------------------------------------------------------------------- extraction

def frames(video: Path, fps: float = FPS, scale_w: int = SCALE_W, hwaccel: str | None = "videotoolbox"):
    """Yield (t_s, BGR frame) at `fps`."""
    head = ["ffmpeg", "-v", "error", "-nostdin"] + (["-hwaccel", hwaccel] if hwaccel else [])
    vf = f"fps={fps},scale={scale_w}:-2"
    one = subprocess.run(head + ["-i", str(video), "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, timeout=180)
    if not one.stdout or len(one.stdout) % (scale_w * 3):
        raise RuntimeError(f"could not measure frames of {video.name}: {one.stderr.decode(errors='replace')[:300]}")
    h = len(one.stdout) // (scale_w * 3)
    n = scale_w * h * 3
    p = subprocess.Popen(head + ["-i", str(video), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(h, scale_w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def faces_path(settings, vs: str, ep: int) -> Path:
    return settings.paths.work_root / "faces" / f"{vs}E{ep:02d}.npz"


def extract_episode(settings, con: sqlite3.Connection, vs: str, ep: int, app=None, force: bool = False, log_=print) -> Path:
    from . import db as dbm

    out = faces_path(settings, vs, ep)
    if out.exists() and not force:
        return out
    row = con.execute("SELECT video_path FROM episodes WHERE version_season=? AND episode=?", (vs, ep)).fetchone()
    video = dbm.localize(con, settings, row[0]) if row and row[0] else None
    if video is None or not Path(video).exists():
        raise FileNotFoundError(f"video for {vs} E{ep:02d} not reachable: {video}")
    app = app or face_app()
    t0 = time.time()
    T, B, D, E = [], [], [], []
    fh = None
    n = 0
    for t, fr in frames(Path(video)):
        n += 1
        fh = fr.shape[0]
        for f in app.get(fr):
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            if (y2 - y1) < MIN_FACE_H * fh or f.normed_embedding is None:
                continue
            T.append(t)
            B.append((x1 / SCALE_W, y1 / fh, x2 / SCALE_W, y2 / fh))       # box as fractions of the frame
            D.append(float(f.det_score))
            E.append(f.normed_embedding.astype(np.float16))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=np.array(T, np.float32), box=np.array(B, np.float32).reshape(-1, 4),
                        det=np.array(D, np.float32), emb=np.array(E, np.float16).reshape(-1, 512), n_frames=n, fps=FPS)
    log_(f"{vs} E{ep:02d}: {n} frames, {len(T)} faces ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------- per-episode face table

@dataclass
class EpisodeFaces:
    t: np.ndarray          # (n,) frame time of each face
    area: np.ndarray       # (n,) face box area as a share of the frame
    emb: np.ndarray        # (n, 512) unit vectors
    frame_t: np.ndarray    # sorted unique frame times that have faces

    @classmethod
    def load(cls, path: Path) -> "EpisodeFaces":
        z = np.load(path)
        box = z["box"]
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        emb = z["emb"].astype(np.float32)
        return cls(z["t"], area, emb, np.unique(z["t"]))

    def largest_in(self, a: float, b: float) -> list[int]:
        """Index of the largest face in each frame between a and b."""
        lo, hi = np.searchsorted(self.t, a, "left"), np.searchsorted(self.t, b, "right")
        best: dict[float, int] = {}
        for i in range(lo, hi):
            if self.t[i] not in best or self.area[i] > self.area[best[self.t[i]]]:
                best[float(self.t[i])] = i
        return list(best.values())


# ----------------------------------------------------------------------------- gallery

def build_gallery(samples: dict[str, list[np.ndarray]], outlier_cos: float = OUTLIER_COS) -> dict[str, np.ndarray]:
    """castaway -> (k, 512) reference faces, with faces far from the castaway's own median dropped."""
    out = {}
    for cid, vs in samples.items():
        if not vs:
            continue
        x = np.vstack(vs).astype(np.float32)
        med = np.median(x, axis=0)
        med /= np.linalg.norm(med) or 1
        keep = x[x @ med >= outlier_cos]
        if len(keep):
            out[cid] = keep
    return out


def sims(gallery: dict[str, np.ndarray], e: np.ndarray, cands: list[str], topk: int = TOPK) -> dict[str, float]:
    out = {}
    for c in cands:
        g = gallery.get(c)
        if g is None:
            continue
        s = np.sort(g @ e)[::-1][:topk]
        out[c] = float(s.mean())
    return out


def score_line(ef: EpisodeFaces, a: float, b: float, gallery: dict, cands: list[str]) -> dict | None:
    """Face evidence for one line: the candidate most frames' largest face resembles."""
    idx = ef.largest_in(a, b)
    if not idx:
        return None
    votes: Counter = Counter()
    sim_sum: dict[str, float] = defaultdict(float)
    margins = []
    for i in idx:
        s = sims(gallery, ef.emb[i], cands)
        if not s:
            continue
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        top, sim = ranked[0]
        votes[top] += 1
        sim_sum[top] += sim
        margins.append(sim - (ranked[1][1] if len(ranked) > 1 else 0.0))
    if not votes:
        return None
    pick, n = votes.most_common(1)[0]
    return {"face_pred": pick, "face_sim": sim_sum[pick] / n, "face_share": n / len(idx), "face_frames": len(idx),
            "face_margin": float(np.median(margins)), "face_area": float(np.median(ef.area[idx]))}


# ----------------------------------------------------------------------------- evaluation

def _lines(con, vs):
    return con.execute("""
        SELECT u.utt_id, u.episode, u.start_s, u.end_s, u.domain_hint, u.segment, l.speaker_id, l.source, l.p_right
        FROM utterances u LEFT JOIN labels l USING (utt_id)
        WHERE u.version_season=? AND u.segment='body' ORDER BY u.episode, u.start_s""", (vs,)).fetchall()


def collect_samples(con, vs: str, eps: list[int], faces: dict[int, EpisodeFaces], host_id: str):
    """Enrollment samples: (source, episode, castaway, embedding)."""
    out = []
    for ep in eps:
        ef = faces.get(ep)
        if ef is None:
            continue
        for t0, t1, cid in con.execute("SELECT t_s, COALESCE(t_end_s, t_s), castaway_id FROM chyron_hits "
                                       "WHERE version_season=? AND episode=? AND castaway_id IS NOT NULL", (vs, ep)):
            for i in ef.largest_in(t0 - 0.25, t1 + 0.5):
                out.append(("card", ep, cid, ef.emb[i]))
    for r in _lines(con, vs):
        uid, ep, a, b, dom, seg, spk, src, p = r
        ef = faces.get(ep)
        if ef is None or src not in TRUSTED or not spk or spk in ("UNKNOWN", "NOSPEECH"):
            continue
        if spk == host_id or dom == "confessional":
            for i in ef.largest_in(a, b):
                if spk == host_id or ef.area[i] >= CONF_MIN_AREA:
                    out.append(("line", ep, spk, ef.emb[i]))
    return verify_line_samples(out)


def _anchor_keep(x: np.ndarray, thr: float) -> np.ndarray:
    """Mask of faces close to the face with the most close neighbours (the mode), for people without name cards."""
    sub = x[:: max(1, len(x) // 1500)]
    anchor = sub[int(((sub @ sub.T) >= thr).sum(1).argmax())]
    return x @ anchor >= thr


def verify_line_samples(samples: list, thr: float = VERIFY_SIM) -> list:
    """Drop line faces that are not the labelled speaker. Voice often runs over B-roll of other people (S47: only
    60% of confessional frames show the speaker), so a line's largest face is checked against the castaway's name
    cards; the host, who has no card, keeps the faces around his most common face."""
    cards: dict[str, list] = defaultdict(list)
    lines: dict[str, list] = defaultdict(list)
    for s in samples:
        (cards if s[0] == "card" else lines)[s[2]].append(s)
    gal = build_gallery({c: [s[3] for s in xs] for c, xs in cards.items()})
    out = [s for xs in cards.values() for s in xs]
    for c, xs in lines.items():
        x = np.vstack([s[3] for s in xs]).astype(np.float32)
        if c in gal:
            keep = np.sort(x @ gal[c].T, axis=1)[:, -TOPK:].mean(1) >= thr
        else:
            keep = _anchor_keep(x, thr)
        out += [s for s, k in zip(xs, keep) if k]
    return out


def evaluate(settings, con: sqlite3.Connection, resolver, vs: str, eps: list[int] | None = None, log_=print) -> dict:
    import pandas as pd

    _, host_id = resolver.host_names(vs)
    have = sorted(int(p.stem.split("E")[-1]) for p in (settings.paths.work_root / "faces").glob(f"{vs}E*.npz"))
    eps = [e for e in (eps or have) if e in have]
    faces = {e: EpisodeFaces.load(faces_path(settings, vs, e)) for e in eps}
    samples = collect_samples(con, vs, eps, faces, host_id)
    by_c: dict[str, list] = defaultdict(list)
    for src, e, c, v in samples:
        by_c[c].append((src, e, v))
    rows = []
    for ep in eps:
        gal = build_gallery({c: [v for src, e, v in xs if src == "card" or e != ep] for c, xs in by_c.items()})
        cands = sorted(set(resolver.present(vs, ep)) | {host_id})
        for r in _lines(con, vs):
            uid, e, a, b, dom, seg, spk, src, p = r
            if e != ep:
                continue
            sc = score_line(faces[ep], a, b, gal, cands)
            rows.append({"utt_id": uid, "episode": ep, "dur": b - a, "domain": dom, "label": spk, "source": src,
                         "p_right": p, **(sc or {})})
    df = pd.DataFrame(rows)
    df["has_face"] = df.get("face_pred").notna() if "face_pred" in df else False
    truth = df[df.source.isin(TRUSTED) & df.label.notna() & ~df.label.isin(["UNKNOWN", "NOSPEECH"])]
    rep = {"episodes": eps, "samples": dict(Counter(src for src, *_ in samples)),
           "gallery_sizes": {c: len(g) for c, g in build_gallery({c: [v for *_, v in xs] for c, xs in by_c.items()}).items()}}

    def acc(d):
        d = d[d.has_face]
        return {"lines": int(len(d)), "right": int((d.face_pred == d.label).sum()),
                "acc": round(float((d.face_pred == d.label).mean()), 3) if len(d) else None}

    rep["host_lines"] = acc(truth[truth.label == host_id])   # mostly voice-over: the host is rarely on screen
    truth = truth[truth.label != host_id]
    rep["truth_lines"] = int(len(truth))
    rep["truth_with_face"] = float(truth.has_face.mean()) if len(truth) else None
    for dom in ("confessional", "field"):
        d = truth[truth.domain == dom]
        rep[f"all_{dom}"] = acc(d)
        th = d[d.has_face & (d.face_sim >= 0.55) & (d.face_share >= 0.999) & (d.face_frames >= 6)]
        rep[f"{dom}_talking_head"] = acc(th) | {"coverage": round(len(th) / max(len(d), 1), 3)}
        for thr in (0.35, 0.45, 0.55):
            conf = d[d.has_face & (d.face_sim >= thr) & (d.face_share >= 0.6)]
            rep[f"{dom}_sim{thr}"] = acc(conf) | {"coverage": round(len(conf) / max(len(d), 1), 3)}
    unk = df[df.label.isna() | df.label.isin(["UNKNOWN"])]
    for dom in ("confessional", "field"):
        d = unk[(unk.domain == dom) & unk.has_face]
        c = d[(d.face_sim >= 0.45) & (d.face_share >= 0.6)]
        rep[f"unlabelled_{dom}"] = {"lines": int((unk.domain == dom).sum()), "with_face": int(len(d)),
                                    "confident": int(len(c)), "confident_seconds": round(float(c.dur.sum()), 0)}
    au = df[(df.source == "auto") & df.has_face & (df.face_sim >= 0.45) & (df.face_share >= 0.6)]
    rep["agree_with_voice_auto"] = {"lines": int(len(au)), "agree": round(float((au.face_pred == au.label).mean()), 3) if len(au) else None}
    out = settings.paths.work_root / "faces" / f"{vs}_eval.csv"
    df.to_csv(out, index=False)
    rep["lines_csv"] = str(out)
    return rep
