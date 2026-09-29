"""Scene type from the picture (prototype): camp, challenge, tribal council, scenery, graphics.

  extract   decode the episode at 1 frame a second (VideoToolbox) and embed each frame with SigLIP 2
            (google/siglip2-base-patch16-224, Apache-2.0) on the Apple GPU. Saves work_root/scenes/<vs>E<ep>.npz:
            frame time, unit image embedding (float16, 768-d). Numbers only; nothing leaves this machine.
  zero-shot each frame against a few text descriptions per scene type (PROMPTS); the type is the best-matching
            description's, and `margin` its lead over the runner-up type.
  probe     a small logistic regression on the frozen embeddings, trained on frames whose type is known or checked
            by hand (the practical form of fine-tuning here).
  smooth    shots are seconds long and scenes minutes long: a running majority over SMOOTH_S removes flicker.

    uv run --with "transformers>=4.49" python -m survspk.scene_type US47 1-14
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

MODEL = "google/siglip2-base-patch16-224"
FPS = 1.0
SMOOTH_S = 15
# Where the scene is, not how it is shot: a confessional is filmed at camp and looks like any camp close-up in a
# single frame (S47 check: a "person talking to the camera" prompt took camp and tribal close-ups alike), so
# confessionals come from the line's domain plus a steady single face, not from this classifier.
PROMPTS = {
    # a shot, not a place: close-ups take the location of the wider shots around them (see `locate`)
    "closeup": ["a close-up of one person talking to the camera in an interview",
                "a single person sitting outdoors being interviewed, blurred background",
                "a close-up of a person's face"],
    "camp": ["a group of people at a beach camp with a shelter made of branches",
             "people sitting around a small campfire at a jungle camp in daylight",
             "people talking on a beach next to a palm-leaf shelter",
             "grainy green or blue night-vision footage of people in the jungle",
             "people searching through bushes in the dark with a flashlight"],
    "challenge": ["contestants competing in an obstacle course with flags and ropes",
                  "people solving a large puzzle at a competition on a beach",
                  "teams of contestants in matching colored clothes standing on mats at a competition arena"],
    "tribal": ["people sitting on benches at night lit by torches at a tribal council",
               "a man holding a torch in a dark room with fire and tiki torches",
               "a voting urn and parchment at night by firelight",
               "a close-up of a person's face at night, lit by firelight, dark background"],
    "scenery": ["an aerial shot of an island and the ocean", "a close-up of a wild animal or insect in the jungle",
                "waves, sunset, clouds or a landscape with no people"],
    "graphics": ["a title card with text on a black background", "a logo or text graphic on the screen"],
}
TYPES = list(PROMPTS)


def scenes_path(settings, vs: str, ep: int) -> Path:
    return settings.paths.work_root / "scenes" / f"{vs}E{ep:02d}.npz"


class Embedder:
    def __init__(self, model: str = MODEL):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model, torch_dtype=torch.float16 if self.dev == "mps" else torch.float32).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model)

    def images(self, frames: list[np.ndarray]) -> np.ndarray:
        x = self.proc(images=[f[:, :, ::-1] for f in frames], return_tensors="pt")["pixel_values"].to(self.dev, self.model.dtype)
        with self.torch.no_grad():
            e = self.model.get_image_features(pixel_values=x).float().cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def texts(self, texts: list[str]) -> np.ndarray:
        x = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            e = self.model.get_text_features(**x).float().cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def extract_episode(settings, con, vs: str, ep: int, emb: Embedder | None = None, force: bool = False,
                    batch: int = 64, log_=print) -> Path:
    from . import db as dbm
    from .faces import frames

    out = scenes_path(settings, vs, ep)
    if out.exists() and not force:
        return out
    row = con.execute("SELECT video_path FROM episodes WHERE version_season=? AND episode=?", (vs, ep)).fetchone()
    video = dbm.localize(con, settings, row[0]) if row and row[0] else None
    if video is None or not Path(video).exists():
        raise FileNotFoundError(f"video for {vs} E{ep:02d} not reachable: {video}")
    emb = emb or Embedder()
    t0 = time.time()
    T, E, buf = [], [], []
    for t, fr in frames(Path(video), fps=FPS, scale_w=448):
        T.append(t)
        buf.append(fr)
        if len(buf) == batch:
            E.append(emb.images(buf)); buf = []
    if buf:
        E.append(emb.images(buf))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=np.array(T, np.float32), emb=np.vstack(E).astype(np.float16), fps=FPS, model=MODEL)
    log_(f"{vs} E{ep:02d}: {len(T)} frames ({time.time() - t0:.0f}s)")
    return out


def prompt_matrix(emb: Embedder) -> tuple[np.ndarray, list[str]]:
    texts, owner = [], []
    for k, ps in PROMPTS.items():
        texts += ps
        owner += [k] * len(ps)
    return emb.texts(texts), owner


def zero_shot(X: np.ndarray, P: np.ndarray, owner: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: type index, margin over the runner-up type, and the per-type best similarity (n x types)."""
    s = X.astype(np.float32) @ P.T
    per = np.stack([s[:, [i for i, o in enumerate(owner) if o == k]].max(1) for k in TYPES], 1)
    srt = np.sort(per, 1)
    return per.argmax(1), srt[:, -1] - srt[:, -2], per


def smooth(labels: np.ndarray, k: int = SMOOTH_S, n_types: int = len(TYPES)) -> np.ndarray:
    """Running majority over k frames."""
    if len(labels) == 0:
        return labels
    oh = np.eye(n_types)[labels]
    c = np.cumsum(np.vstack([np.zeros((1, n_types)), oh]), 0)
    h = k // 2
    idx = np.arange(len(labels))
    lo, hi = np.clip(idx - h, 0, len(labels)), np.clip(idx + h + 1, 0, len(labels))
    return (c[hi] - c[lo]).argmax(1)


def locate(labels: np.ndarray, window: int = 30) -> np.ndarray:
    """Location per frame: close-ups take the most common place among the wider shots within +-window frames
    (graphics are not a place either, so they are left out of the vote); with no wider shot nearby they stay."""
    shot_only = {TYPES.index("closeup"), TYPES.index("graphics")}
    out = labels.copy()
    places = np.array([k not in shot_only for k in labels])
    for i in np.where(labels == TYPES.index("closeup"))[0]:
        lo, hi = max(0, i - window), min(len(labels), i + window + 1)
        near = labels[lo:hi][places[lo:hi]]
        if len(near):
            out[i] = np.bincount(near, minlength=len(TYPES)).argmax()
    return out


def main(argv=None):
    import sqlite3
    from .config import load_settings

    argv = argv or sys.argv[1:]
    vs, spec = argv[0], argv[1]
    eps = [int(x) for x in spec.split(",")] if "-" not in spec else list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1))
    s = load_settings(vs)
    con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    emb = Embedder()
    for ep in eps:
        extract_episode(s, con, vs, ep, emb, log_=lambda m: print(m, flush=True))
    print("done", flush=True)


if __name__ == "__main__":
    main()
