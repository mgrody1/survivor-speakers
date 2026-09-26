"""Frame-level speaker activity for one episode with NVIDIA Nemotron 3 Diarization (transformers port).

Runs in its own throwaway environment (transformers support is not in a release yet), called by `survspk diarize`:

    uv run --no-project --with "git+https://github.com/huggingface/transformers" --with torch --with numpy \
        --with librosa python scripts/diarize_nemotron.py AUDIO.flac OUT.npz

The model attends over its whole input, so an hour in one pass is slow and memory-hungry; it runs in 300 s windows
that overlap by 30 s (each about half a second on the M3 Ultra). Speaker channels are numbered per window, so a
consumer compares frames from one window only. Output (npz): starts [n] window start seconds, lens [n] frames per
window, dom [sum lens] int8 dominant channel per 10 ms frame (-1 = nobody above 0.5), pmax [sum lens] uint8
(max probability x 255).
"""
import subprocess, sys, time
import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL = "nvidia/Nemotron-3-Diarization"
WIN_S, HOP_S, SR = 300.0, 270.0, 16000


def main(src: str, out: str) -> int:
    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL).to(dev).eval()
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.float32)
    total = len(x) / SR
    starts, doms, pmaxs = [], [], []
    s = 0.0
    while True:
        seg = x[int(s * SR):int(min(total, s + WIN_S) * SR)]
        if len(seg) < SR:
            break
        inp = proc(seg, sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.inference_mode():
            p = torch.sigmoid(model(**inp).logits)[0].float().cpu().numpy()
        n = int(round(len(seg) / SR * 100))
        p = p[:n]
        pm = p.max(1)
        starts.append(s)
        doms.append(np.where(pm >= 0.5, p.argmax(1), -1).astype(np.int8))
        pmaxs.append(np.clip(pm * 255, 0, 255).astype(np.uint8))
        if s + WIN_S >= total:
            break
        s += HOP_S
    np.savez_compressed(out, starts=np.array(starts, np.float32), lens=np.array([len(d) for d in doms], np.int32),
                        dom=np.concatenate(doms), pmax=np.concatenate(pmaxs))
    print(f"{len(starts)} windows, {total / 60:.0f} min of audio in {time.time() - t0:.0f}s on {dev} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
