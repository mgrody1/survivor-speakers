"""Fine-tune the speaker-embedding model (SpeechBrain ECAPA-TDNN) on the exported corpus (export_corpus.py).

Why this and not the diarizer first: our own assign loop is bottlenecked by embedding quality on field speech
(music, wind, overlap), and ~900 castaways x an hour each of in-domain speech is a textbook speaker-verification
adaptation set. The fine-tuned model drops straight back into the pipeline (`embed --model <checkpoint dir>`).

Recipe: pretrained ECAPA (speechbrain/spkrec-ecapa-voxceleb) + a fresh AAM-softmax head over the training speakers;
3 s random crops, small LR on the encoder, larger LR on the head; the metric that matters is EER on the dev
verification trials (held-out episodes, seen speakers) - reported for the *untouched* model first (epoch 0), so the
number to beat is printed before any training happens.

Checkpoints are written in SpeechBrain's layout (hyperparams.yaml + embedding_model.ckpt + ...), so
`survspk embed US47 2 --model <out_dir>` and the ablation work unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRETRAINED = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class TrainCfg:
    corpus: Path
    out: Path
    epochs: int = 10
    batch_size: int = 32
    crop_s: float = 3.0
    lr_encoder: float = 1e-4
    lr_head: float = 1e-3
    margin: float = 0.2
    scale: float = 30.0
    freeze_epochs: int = 1            # train only the head for the first N epochs
    min_utts_per_speaker: int = 8     # speakers below this are dropped from the classification head (kept in trials)
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    smoke: bool = False               # random-init ECAPA, tiny data: exercises the loop without downloads
    eval_every: int = 1
    max_eval_utts: int = 4000


# ----------------------------------------------------------------------------- data


class CropDataset:
    """Random fixed-length crops of labelled utterances, read straight from the FLAC files."""

    def __init__(self, df: pd.DataFrame, spk2idx: dict[str, int], crop_s: float, sr: int = 16000, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.spk2idx, self.crop, self.sr, self.train = spk2idx, int(crop_s * sr), sr, train
        self._files: dict[str, object] = {}

    def __len__(self):
        return len(self.df)

    def _read(self, wav: str, start_s: float, end_s: float) -> np.ndarray:
        import soundfile as sf

        a, b = int(start_s * self.sr), int(end_s * self.sr)
        if self.train and b - a > self.crop:
            a = random.randint(a, b - self.crop)
            b = a + self.crop
        with sf.SoundFile(wav) as f:
            f.seek(a)
            x = f.read(b - a, dtype="float32", always_2d=True).mean(axis=1)
        if len(x) < self.crop:
            reps = int(math.ceil(self.crop / max(len(x), 1)))
            x = np.tile(x, reps)[: self.crop] if len(x) else np.zeros(self.crop, dtype=np.float32)
        return x.astype(np.float32)

    def __getitem__(self, i: int):
        import torch

        r = self.df.iloc[i]
        x = self._read(r.wav, r.start_s, r.end_s)
        return torch.from_numpy(x), self.spk2idx.get(r.speaker_id, -1), r.id


# ----------------------------------------------------------------------------- model


class AAMSoftmax:
    """Additive angular margin softmax head (ArcFace) as a plain torch module."""

    def __init__(self, dim: int, n_classes: int, margin: float, scale: float, device):
        import torch

        self.W = torch.nn.Parameter(torch.randn(n_classes, dim, device=device) * 0.01)
        self.m, self.s = margin, scale

    def parameters(self):
        return [self.W]

    def __call__(self, emb, target):
        import torch
        import torch.nn.functional as F

        cos = F.normalize(emb) @ F.normalize(self.W).T                     # [B, C]
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target_cos = torch.cos(theta + self.m)
        onehot = F.one_hot(target, cos.shape[1]).bool()
        logits = torch.where(onehot, target_cos, cos) * self.s
        return F.cross_entropy(logits, target)


def load_pretrained(smoke: bool, device):
    """Return (compute_features, mean_var_norm, embedding_model, source_dir or None)."""
    import torch

    if smoke:
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        emb = ECAPA_TDNN(input_size=80, lin_neurons=192, channels=[512, 512, 512, 512, 1536],
                         kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1], attention_channels=128)
        return Fbank(n_mels=80).to(device), InputNormalization(norm_type="sentence", std_norm=False).to(device), emb.to(device), None
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = Path.home() / ".cache" / "survspk" / PRETRAINED.replace("/", "__")
    enc = EncoderClassifier.from_hparams(source=PRETRAINED, savedir=str(savedir), run_opts={"device": str(device)})
    return enc.mods.compute_features, enc.mods.mean_var_norm, enc.mods.embedding_model, savedir


def embed_batch(feats, norm, model, wav, device):
    import torch

    wav = wav.to(device)
    lens = torch.ones(wav.shape[0], device=device)
    f = feats(wav)
    f = norm(f, lens)
    e = model(f, lens)
    return e.squeeze(1)


# ----------------------------------------------------------------------------- metrics


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold from cosine scores and 0/1 labels."""
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    frr = 1 - tp / n_pos                  # accepted below threshold -> rejected positives
    far = fp / n_neg
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(s[i])


def evaluate(feats, norm, model, dev: pd.DataFrame, trials: pd.DataFrame, cfg: TrainCfg, device) -> dict:
    import torch

    model.eval()
    ids = pd.unique(pd.concat([trials.enrol, trials.test]))
    sub = dev[dev.id.isin(ids)]
    if len(sub) > cfg.max_eval_utts:
        sub = sub.sample(cfg.max_eval_utts, random_state=0)
        keep = set(sub.id)
        trials = trials[trials.enrol.isin(keep) & trials.test.isin(keep)]
    ds = CropDataset(sub, {}, cfg.crop_s, train=False)
    vecs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(ds), cfg.batch_size):
            batch = [ds[j] for j in range(i, min(len(ds), i + cfg.batch_size))]
            # variable length at eval: embed one by one when lengths differ (utterances are short; fine)
            for wav, _, uid in batch:
                e = embed_batch(feats, norm, model, wav[None, :], device)
                vecs[uid] = torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()
    sc = np.array([float(vecs[a] @ vecs[b]) for a, b in zip(trials.enrol, trials.test) if a in vecs and b in vecs])
    lab = np.array([int(l) for a, b, l in zip(trials.enrol, trials.test, trials.label) if a in vecs and b in vecs])
    e, thr = eer(sc, lab)
    model.train()
    return {"eer": round(e, 4), "threshold": round(thr, 4), "n_trials": int(len(sc)),
            "mean_target": round(float(sc[lab == 1].mean()), 3) if (lab == 1).any() else None,
            "mean_nontarget": round(float(sc[lab == 0].mean()), 3) if (lab == 0).any() else None}


# ----------------------------------------------------------------------------- training


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(out: Path, model, source_dir: Path | None, history: list[dict], spk2idx: dict) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "embedding_model.ckpt")
    if source_dir is not None:
        for name in ("hyperparams.yaml", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.ckpt", "label_encoder.txt"):
            src = source_dir / name
            if src.exists() and not (out / name).exists():
                # resolve symlinks (the cache dir keeps HF symlinks) so the checkpoint dir is self-contained
                shutil.copyfile(src.resolve(), out / name)
    with open(out / "finetune.json", "w") as f:
        json.dump({"history": history, "n_speakers": len(spk2idx), "pretrained": PRETRAINED}, f, indent=1)


def train(cfg: TrainCfg) -> dict:
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    spk_dir = cfg.corpus / "spk"
    train_df = pd.read_csv(spk_dir / "train.csv")
    dev_df = pd.read_csv(spk_dir / "dev.csv") if (spk_dir / "dev.csv").exists() else pd.DataFrame(columns=train_df.columns)
    trials = (pd.read_csv(spk_dir / "trials_dev.txt", sep=" ", names=["label", "enrol", "test"])
              if (spk_dir / "trials_dev.txt").exists() and (spk_dir / "trials_dev.txt").stat().st_size else pd.DataFrame(columns=["label", "enrol", "test"]))
    counts = train_df.speaker_id.value_counts()
    speakers = sorted(counts[counts >= cfg.min_utts_per_speaker].index)
    spk2idx = {s: i for i, s in enumerate(speakers)}
    tr = train_df[train_df.speaker_id.isin(spk2idx)].reset_index(drop=True)
    log.info("train: %d utts, %d speakers (>= %d utts); dev: %d utts, %d trials; device %s",
             len(tr), len(speakers), cfg.min_utts_per_speaker, len(dev_df), len(trials), device)
    if len(speakers) < 2:
        raise LookupError("need at least 2 speakers with enough utterances to fine-tune")

    feats, norm, model, source_dir = load_pretrained(cfg.smoke, device)
    head = AAMSoftmax(192, len(speakers), cfg.margin, cfg.scale, device)
    ds = CropDataset(tr, spk2idx, cfg.crop_s, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                         drop_last=len(ds) > cfg.batch_size, collate_fn=_collate)
    opt_head = torch.optim.Adam(head.parameters(), lr=cfg.lr_head)
    opt_enc = torch.optim.Adam(model.parameters(), lr=cfg.lr_encoder)
    history: list[dict] = []
    if len(trials):
        base = evaluate(feats, norm, model, dev_df, trials, cfg, device)
        log.info("epoch 0 (pretrained, untouched): EER %.2f%% on %d trials  <- the number to beat", 100 * base["eer"], base["n_trials"])
        history.append({"epoch": 0, **base})
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_enc = epoch > cfg.freeze_epochs
        for p in model.parameters():
            p.requires_grad_(train_enc)
        tot, n = 0.0, 0
        for wav, y, _ in loader:
            y = y.to(device)
            emb = embed_batch(feats, norm, model, wav, device)
            loss = head(emb, y)
            opt_head.zero_grad()
            opt_enc.zero_grad()
            loss.backward()
            opt_head.step()
            if train_enc:
                opt_enc.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        rec = {"epoch": epoch, "loss": round(tot / max(n, 1), 4), "encoder_trained": train_enc, "seconds": round(time.time() - t0)}
        if len(trials) and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            rec.update(evaluate(feats, norm, model, dev_df, trials, cfg, device))
            log.info("epoch %d: loss %.3f  EER %.2f%%", epoch, rec["loss"], 100 * rec["eer"])
        else:
            log.info("epoch %d: loss %.3f", epoch, rec["loss"])
        history.append(rec)
        save_checkpoint(cfg.out, model, source_dir, history, spk2idx)
    return {"history": history, "out": str(cfg.out), "n_speakers": len(speakers), "device": str(device)}


def _collate(batch):
    import torch

    wav = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return wav, y, [b[2] for b in batch]
