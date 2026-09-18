"""Speaker embeddings per utterance (spec §7.5) + the M1 audio-variant ablation (spec §8.3).

Embeddings: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), 192-d, L2-normalised, one Parquet per
(episode, audio variant):  work_root/embeddings/<vs>/E<ep>.<variant>.parquet
    utt_id, model, variant, vector (list<float32>), snr_proxy, duration_s

snr_proxy = RMS(variant slice) / RMS(raw slice): how much of the raw energy survives separation — high for
clean close-mic'd confessionals, low where music/wind dominated (spec §8.2 domain heuristic input).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .stage_align import load_audio

log = logging.getLogger(__name__)


DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model.split("/")[-1]).strip("-").lower()


def embedding_path(settings: Settings, vs: str, ep: int, variant: str, model: str | None = None) -> Path:
    model = model or settings.embed.model
    suffix = "" if model == DEFAULT_MODEL else f".{model_tag(model)}"
    return settings.work("embeddings") / vs / f"E{ep:02d}.{variant}{suffix}.parquet"


class Encoder:
    """Uniform wrapper: encode(list of float32 arrays) -> [n, d]. Backends: speechbrain (ECAPA, ResNet) and
    pyannote.audio (e.g. pyannote/wespeaker-voxceleb-resnet34-LM)."""

    def __init__(self, model: str, device: str, batch_size: int = 32):
        self.model, self.batch_size = model, batch_size
        self.backend = "pyannote" if model.startswith("pyannote/") else "speechbrain"
        self.device = device
        self._impl = None
        for dev in ([device, "cpu"] if device != "cpu" else ["cpu"]):
            try:
                self._impl = self._load(dev)
                self.device = dev
                self.encode([np.zeros(16000, dtype=np.float32)])   # smoke test: some ops are unsupported on MPS
                break
            except Exception as e:  # noqa: BLE001
                log.warning("encoder %s on %s failed (%s)", model, dev, e)
                self._impl = None
        if self._impl is None:
            raise RuntimeError(f"could not load {model} on any device")

    def _load(self, dev: str):
        if self.backend == "speechbrain":
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = Path.home() / ".cache" / "survspk" / self.model.replace("/", "__")
            return EncoderClassifier.from_hparams(source=self.model, savedir=str(savedir), run_opts={"device": dev})
        import os

        import torch
        from pyannote.audio import Inference, Model

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        m = Model.from_pretrained(self.model, token=os.environ.get("HF_TOKEN") or None)
        inf = Inference(m, window="whole")
        inf.to(torch.device(dev))
        return inf

    def encode(self, slices: list[np.ndarray]) -> np.ndarray:
        import torch

        if self.backend == "pyannote":
            out = []
            for seg in slices:
                wav = torch.from_numpy(seg[None, :].astype(np.float32))
                emb = self._impl({"waveform": wav, "sample_rate": 16000})
                out.append(np.asarray(emb, dtype=np.float32).reshape(-1))
            return np.stack(out)
        n = len(slices)
        order = np.argsort([len(x) for x in slices])
        res: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                maxlen = max(len(slices[i]) for i in idx)
                batch = np.zeros((len(idx), maxlen), dtype=np.float32)
                lens = np.zeros(len(idx), dtype=np.float32)
                for j, i in enumerate(idx):
                    batch[j, : len(slices[i])] = slices[i]
                    lens[j] = len(slices[i]) / maxlen
                emb = self._impl.encode_batch(torch.from_numpy(batch).to(self.device), torch.from_numpy(lens).to(self.device))
                emb = emb.squeeze(1).float().cpu().numpy()
                for j, i in enumerate(idx):
                    res[int(i)] = emb[j]
        return np.stack([res[i] for i in range(n)])


def _load_encoder(model: str, device: str, batch_size: int = 32) -> tuple[Encoder, str]:
    enc = Encoder(model, device, batch_size)
    return enc, enc.device


def embed_utterances(audio: np.ndarray, raw: np.ndarray | None, utts: list[dict], enc: Encoder, device: str,
                     sample_rate: int, pad_s: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (vectors [n,d] L2-normalised, snr_proxy [n])."""
    n = len(utts)
    slices, snr = [], np.ones(n, dtype=np.float32)
    for i, u in enumerate(utts):
        a = max(0, int((u["start_s"] - pad_s) * sample_rate))
        b = min(len(audio), int((u["end_s"] + pad_s) * sample_rate))
        seg = audio[a:b]
        if len(seg) < int(0.3 * sample_rate):
            seg = np.pad(seg, (0, int(0.3 * sample_rate) - len(seg)))
        slices.append(seg)
        if raw is not None:
            rs = raw[a:b]
            snr[i] = float(np.sqrt((seg ** 2).mean() + 1e-12) / np.sqrt((rs ** 2).mean() + 1e-12))
    vecs = enc.encode(slices).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs, snr


def _current_utts(con: sqlite3.Connection, vs: str, ep: int, min_duration_s: float) -> list[dict]:
    return [dict(r) for r in con.execute(
        """SELECT utt_id, start_s, end_s FROM utterances WHERE version_season=? AND episode=? AND is_speech=1
           AND end_s - start_s >= ? ORDER BY idx""", (vs, ep, min_duration_s))]


def staleness_reason(path: Path, utts: list[dict], tol_s: float = 0.02) -> str | None:
    """None if the parquet matches these utterances exactly; else a short explanation of what differs."""
    try:
        old = pd.read_parquet(path, columns=["utt_id", "start_s", "end_s"])
    except Exception as e:  # noqa: BLE001  (older parquet without spans -> treat as stale)
        return f"parquet has no start_s/end_s columns ({type(e).__name__}); embedded before spans were stored"
    cur = pd.DataFrame(utts)
    if len(old) != len(cur):
        only_old = set(old.utt_id) - set(cur.utt_id)
        only_new = set(cur.utt_id) - set(old.utt_id)
        return (f"{len(old)} embedded vs {len(cur)} current utterances; "
                f"{len(only_old)} ids only in parquet, {len(only_new)} only in db (e.g. {sorted(only_new | only_old)[:3]})")
    m = cur.merge(old, on="utt_id", suffixes=("", "_old"))
    if len(m) != len(cur):
        return f"{len(cur) - len(m)} utterance ids not in the parquet"
    moved = ((m.start_s - m.start_s_old).abs() > tol_s) | ((m.end_s - m.end_s_old).abs() > tol_s)
    if moved.any():
        ex = m[moved].iloc[0]
        return (f"{int(moved.sum())} utterances changed span (e.g. {ex.utt_id}: "
                f"{ex.start_s_old:.2f}-{ex.end_s_old:.2f} -> {ex.start_s:.2f}-{ex.end_s:.2f})")
    return None


def parquet_is_fresh(path: Path, utts: list[dict], tol_s: float = 0.02) -> bool:
    """True if the parquet was embedded from exactly these utterances (same ids, same spans). utt ids are
    positional (U0001...), so any re-segmentation shifts them and silently mis-pairs vectors with rows."""
    return staleness_reason(path, utts, tol_s) is None


def embed_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str | None = None,
                  force: bool = False, encoder=None, model: str | None = None) -> Path:
    variant = variant or settings.audio.variant
    model = model or settings.embed.model
    out = embedding_path(settings, vs, ep, variant, model)
    cfg = settings.embed
    utts = _current_utts(con, vs, ep, cfg.min_duration_s)
    if not utts:
        raise LookupError(f"no utterances for {vs} E{ep:02d}; run segment first")
    if out.exists() and not force:
        if parquet_is_fresh(out, utts):
            log.info("%s E%02d %s embeddings exist", vs, ep, variant)
            return out
        log.info("%s E%02d %s embeddings are stale (utterances re-segmented) -> re-embedding", vs, ep, variant)
    audio_path = settings.audio_path(variant, vs, ep)
    if not audio_path.exists():
        raise FileNotFoundError(f"{audio_path} missing")
    sr = settings.audio.sample_rate
    t0 = time.time()
    audio = load_audio(audio_path, sr)
    raw = None
    if variant != "raw":
        rp = settings.audio_path("raw", vs, ep)
        raw = load_audio(rp, sr) if rp.exists() else None
        if raw is not None and len(raw) != len(audio):
            m = min(len(raw), len(audio))
            raw, audio = raw[:m], audio[:m]
    enc, device = encoder or _load_encoder(model, cfg.device, cfg.batch_size)
    vecs, snr = embed_utterances(audio, raw, utts, enc, device, sr, cfg.pad_s, cfg.batch_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "utt_id": [u["utt_id"] for u in utts], "model": model, "variant": variant,
        "vector": [v.tolist() for v in vecs], "snr_proxy": snr,
        # start/end are stored so a later re-segmentation can be detected (utt ids are positional and shift)
        "start_s": [u["start_s"] for u in utts], "end_s": [u["end_s"] for u in utts],
        "duration_s": [u["end_s"] - u["start_s"] for u in utts],
    })
    df.to_parquet(out, index=False)
    dt = time.time() - t0
    log.info("%s E%02d %s [%s]: embedded %d utterances in %.0f s (%s)", vs, ep, variant, model_tag(model), len(utts), dt, device)
    con.execute("UPDATE episodes SET status='embedded' WHERE version_season=? AND episode=?", (vs, ep))
    con.commit()
    return out


# ----------------------------------------------------------------------------- ablation


def load_labeled(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str,
                 min_duration_s: float = 1.5, min_per_speaker: int = 5, model: str | None = None,
                 all_durations: bool = False) -> pd.DataFrame:
    path = embedding_path(settings, vs, ep, variant, model)
    if not parquet_is_fresh(path, _current_utts(con, vs, ep, settings.embed.min_duration_s)):
        raise RuntimeError(f"{path.name} is stale relative to the utterances table (re-segmented since it was "
                           f"embedded, or an old parquet without spans). Re-run: survspk embed {vs} {ep} "
                           f"--variant {variant}" + (f" --model {model}" if model else ""))
    df = pd.read_parquet(path)
    if "start_s" in df.columns:
        df = df.drop(columns=["start_s", "end_s"])
    lab = pd.read_sql_query(
        """SELECT utt_id, sdh_speaker_id AS speaker_id, sdh_name, sdh_resolution, segment, domain_hint, is_italic, flags,
                  start_s, text, end_s - start_s AS dur FROM utterances WHERE version_season=? AND episode=?""",
        con, params=(vs, ep))
    # NB: lab.flags is pandas' own DataFrame.flags attribute, not the column -> bracket access
    flags = lab["flags"].apply(lambda f: json.loads(f) if f else {})
    lab["run_id"] = flags.apply(lambda d: d.get("run_id", -1))
    lab["run_dur"] = flags.apply(lambda d: d.get("run_dur_s", 0.0))
    lab["name_explicit"] = flags.apply(lambda d: not d.get("name_inherited", True))
    df = df.merge(lab.drop(columns=["flags"]), on="utt_id")
    df = df[df.speaker_id.notna() & df.sdh_resolution.isin(["cast", "alias", "host"]) & (df.segment == "body")]
    if not all_durations:
        df = df[df.dur >= min_duration_s]
    counts = df.speaker_id.value_counts()
    return df[df.speaker_id.isin(counts[counts >= min_per_speaker].index)].reset_index(drop=True)


def pool_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One duration-weighted, renormalised vector per labelled run (a run is one speaker by construction)."""
    rows = []
    for rid, g in df.groupby("run_id"):
        if g.speaker_id.nunique() != 1:
            continue
        V = np.stack(g.vector.to_numpy())
        w = g.dur.to_numpy()[:, None]
        vec = (V * w).sum(0) / w.sum()
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        rows.append({"run_id": rid, "speaker_id": g.speaker_id.iloc[0], "vector": vec.astype(np.float32),
                     "run_dur": float(g.run_dur.iloc[0]), "domain": g.domain_hint.iloc[0], "n_utts": len(g)})
    return pd.DataFrame(rows)


def _score_or_none(df: pd.DataFrame, min_per_speaker: int = 3) -> dict | None:
    if df.empty:
        return None
    df = df[df.speaker_id.isin(df.speaker_id.value_counts()[lambda c: c >= min_per_speaker].index)]
    if len(df) < 15 or df.speaker_id.nunique() < 3:
        return None
    return loo_scores(df)


def loo_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out nearest-centroid prediction for every row. Returns a frame aligned with `df` with columns
    pred (best centroid), sim (cosine to it), second (runner-up speaker), margin (sim - runner-up sim),
    sim_true (cosine to the row's own leave-one-out centroid; NaN if it is the only sample), ex_pred (nearest exemplar)."""
    X = np.stack(df.vector.to_numpy())
    y = df.speaker_id.to_numpy()
    speakers = sorted(set(y))
    sums = {s: X[y == s].sum(axis=0) for s in speakers}
    cnts = {s: int((y == s).sum()) for s in speakers}
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    out = []
    for i in range(len(X)):
        best_c, best_s, second_c, second = None, -np.inf, None, -np.inf
        sim_true = np.nan
        for s in speakers:
            c = sums[s] - (X[i] if y[i] == s else 0)
            k = cnts[s] - (1 if y[i] == s else 0)
            if k == 0:
                continue
            c = c / k
            sim = float(X[i] @ c / (np.linalg.norm(c) + 1e-9))
            if s == y[i]:
                sim_true = sim
            if sim > best_s:
                second_c, second = best_c, best_s
                best_c, best_s = s, sim
            elif sim > second:
                second_c, second = s, sim
        j = int(np.argmax(S[i]))
        out.append({"pred": best_c, "sim": best_s, "second": second_c, "margin": best_s - second,
                    "sim_true": sim_true, "ex_pred": y[j]})
    return pd.DataFrame(out, index=df.index)


def loo_scores(df: pd.DataFrame) -> dict:
    """Leave-one-out speaker identification among SDH-labelled utterances: nearest centroid and nearest exemplar."""
    p = loo_predict(df)
    y = df.speaker_id.to_numpy()
    hit = (p.pred.to_numpy() == y)
    n = len(df)
    per = pd.Series(hit).groupby(y).agg(["sum", "count"])
    return {
        "n_utts": n, "n_speakers": int(len(set(y))),
        "centroid_acc": float(hit.mean()), "exemplar_acc": float((p.ex_pred.to_numpy() == y).mean()),
        "median_margin": float(p.margin.median()),
        "per_speaker_acc": {s: round(r["sum"] / r["count"], 3) for s, r in per.iterrows()},
    }


def run_errors(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str, model: str | None = None,
               domain: str | None = "confessional", min_run_s: float = 0.0, min_per_speaker: int = 3) -> dict:
    """Diagnose the run-level LOO: which runs are misclassified, how, and by whom.

    Returns {"runs": DataFrame of every scored run with prediction columns, "errors": the misclassified subset,
             "per_speaker": accuracy / count / n_wrong_as (how often others are predicted as this speaker),
             "confusions": pairs (true -> pred) with counts}."""
    df = load_labeled(settings, con, vs, ep, variant, model=model, all_durations=True, min_per_speaker=1)
    runs = pool_runs(df)
    if domain:
        runs = runs[runs.domain == domain]
    runs = runs[runs.run_dur >= min_run_s]
    keep = runs.speaker_id.value_counts()
    runs = runs[runs.speaker_id.isin(keep[keep >= min_per_speaker].index)].reset_index(drop=True)
    if runs.empty:
        return {"runs": runs, "errors": runs, "per_speaker": pd.DataFrame(), "confusions": pd.DataFrame()}
    p = loo_predict(runs)
    runs = pd.concat([runs, p], axis=1)
    meta = (df.sort_values("start_s").groupby("run_id")
              .agg(start_s=("start_s", "min"), sdh_name=("sdh_name", "first"),
                   n_explicit=("name_explicit", "sum"),      # explicit NAME: lines in the run (0 = label rests on inheritance)
                   text=("text", lambda t: " ".join(x or "" for x in t)[:110])))
    runs = runs.merge(meta, left_on="run_id", right_index=True, how="left")
    runs["ok"] = runs.pred == runs.speaker_id
    runs["mmss"] = runs.start_s.apply(lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}")
    errors = runs[~runs.ok].sort_values("margin", ascending=False)
    per = runs.groupby("speaker_id").agg(n=("ok", "size"), acc=("ok", "mean"), dur=("run_dur", "sum"))
    per["n_wrong_as"] = errors.pred.value_counts().reindex(per.index).fillna(0).astype(int)
    per["acc"] = per.acc.round(3)
    confusions = (errors.groupby(["speaker_id", "pred"]).size().rename("n").reset_index()
                  .sort_values("n", ascending=False))
    return {"runs": runs, "errors": errors, "per_speaker": per.sort_values("acc"), "confusions": confusions}


def ablation(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variants: list[str] | None = None,
             plot: bool = True, model: str | None = None) -> pd.DataFrame:
    model = model or settings.embed.model
    variants = variants or [v for v in ("raw", "center", "vocals", "vocals_center")
                            if embedding_path(settings, vs, ep, v, model).exists()]
    rows = []
    for v in variants:
        df = load_labeled(settings, con, vs, ep, v, model=model)
        if len(df) < 20:
            log.warning("%s: only %d labelled utterances; skipping", v, len(df))
            continue
        sc = loo_scores(df)
        runs = pool_runs(load_labeled(settings, con, vs, ep, v, model=model, all_durations=True, min_per_speaker=1))
        r_conf = _score_or_none(runs[runs.domain == "confessional"])
        r_long = _score_or_none(runs[runs.run_dur >= 10])
        r_field = _score_or_none(runs[(runs.domain == "field") & (runs.run_dur >= 1.5)])
        rows.append({
            "variant": v, "model": model_tag(model), "n_utts": sc["n_utts"], "n_speakers": sc["n_speakers"],
            "utt_acc": round(sc["centroid_acc"], 3), "utt_margin": round(sc["median_margin"], 3),
            "conf_run_acc": round(r_conf["centroid_acc"], 3) if r_conf else None,
            "n_conf_runs": int(r_conf["n_utts"]) if r_conf else 0,
            "run10_acc": round(r_long["centroid_acc"], 3) if r_long else None,
            "n_runs10": int(r_long["n_utts"]) if r_long else 0,
            "field_run_acc": round(r_field["centroid_acc"], 3) if r_field else None,
            "mean_snr_proxy": round(float(df.snr_proxy.mean()), 3),
        })
        if plot:
            try:
                _tsne_plot(settings, df, vs, ep, v)
            except Exception as e:  # noqa: BLE001
                log.warning("t-SNE plot failed for %s: %s", v, e)
    out = pd.DataFrame(rows)
    rep = settings.paths.work_root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    out.to_csv(rep / f"ablation_{vs}E{ep:02d}_{model_tag(model)}.csv", index=False)
    for r in rows:
        con.execute("INSERT OR REPLACE INTO metrics (version_season, episode, key, value, payload, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
                    (vs, ep, f"ablation_{r['model']}_{r['variant']}", r["utt_acc"], json.dumps(r)))
    con.commit()
    return out


def _tsne_plot(settings: Settings, df: pd.DataFrame, vs: str, ep: int, variant: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = np.stack(df.vector.to_numpy())
    Z = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 10)), init="pca", random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for s in sorted(df.speaker_id.unique()):
        m = (df.speaker_id == s).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=14, label=f"{s} ({m.sum()})", alpha=0.8)
    ax.set_title(f"{vs} E{ep:02d} — {variant} — ECAPA t-SNE of SDH-labelled body utterances")
    ax.legend(fontsize=7, ncol=2, markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    out = settings.paths.work_root / "reports" / f"tsne_{vs}E{ep:02d}_{variant}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out
