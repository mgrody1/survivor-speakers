"""Fine-tune pyannote's segmentation model on the exported Survivor corpus (survspk export-corpus).

Run on the Mac, in the project venv (needs --extra diarize and HF_TOKEN with access to pyannote/segmentation-3.0):

    uv run python scripts/finetune_pyannote_seg.py --corpus /path/to/corpus --out models/seg-survivor \
        [--epochs 20] [--batch 32] [--duration 10] [--device mps]

What it does (pyannote.audio 3.x recipe):
  1. registry.load_database(<corpus>/database.yml)      -> protocol Survivor.SpeakerDiarization.All
  2. task = SpeakerDiarization(protocol, duration=10 s, max_speakers_per_chunk=4, ...)
  3. model = Model.from_pretrained("pyannote/segmentation-3.0"); model.task = task
  4. lightning Trainer.fit(model)  — checkpoints under <out>/, best by the task's own DER-ish metric on `development`
  5. prints DER on the development set before and after (the pretrained model first, so the gain is visible)

The UEM files make the trainer ignore every region we did not annotate (unsubtitled speech, UNKNOWN spans, gaps
with speech the VAD heard), so wrong "non-speech" targets are not taught. Overlap is under-annotated (captions do
not mark it) — expect the fine-tuned model to be better on turn-taking than on crosstalk, and say so when reporting.

The trained checkpoint plugs into pyannote's speaker-diarization pipeline as its `segmentation` component; the plan
is to use it for the `>>`-era seasons (S21-39) whose subtitles carry no names, where episode-1 bootstrap depends on
diarization quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--duration", type=float, default=10.0, help="training chunk length (s)")
    ap.add_argument("--max-speakers", type=int, default=4, help="max speakers per chunk (Survivor camp scenes: 3-4)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto", help="mps | cuda | cpu | auto")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import os

    import torch
    from pyannote.audio import Model
    from pyannote.audio.tasks import SpeakerDiarization
    from pyannote.database import FileFinder, registry
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    registry.load_database(str(args.corpus / "database.yml"))
    protocol = registry.get_protocol("Survivor.SpeakerDiarization.All", preprocessors={"audio": FileFinder()})

    task = SpeakerDiarization(protocol, duration=args.duration, max_speakers_per_chunk=args.max_speakers,
                              max_speakers_per_frame=2, batch_size=args.batch, num_workers=args.workers)
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=os.environ.get("HF_TOKEN"))
    model.task = task

    # a small LR for fine-tuning; pyannote's default configure_optimizers uses Adam(lr=1e-3)
    def configure_optimizers(self=model):
        return torch.optim.Adam(self.parameters(), lr=args.lr)

    model.configure_optimizers = configure_optimizers  # type: ignore[method-assign]

    accel = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else
                                                        "gpu" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    monitor, direction = task.val_monitor
    ckpt = ModelCheckpoint(dirpath=str(args.out), filename="{epoch}-{" + monitor + ":.3f}", monitor=monitor,
                           mode=direction, save_top_k=1, save_last=True)
    trainer = Trainer(accelerator=accel, devices=1, max_epochs=args.epochs, callbacks=[ckpt, EarlyStopping(monitor=monitor, mode=direction, patience=5)],
                      default_root_dir=str(args.out))
    trainer.fit(model)
    summary = {"best": ckpt.best_model_path, "best_score": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
               "monitor": monitor, "epochs": trainer.current_epoch, "corpus": str(args.corpus)}
    with open(args.out / "finetune.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
