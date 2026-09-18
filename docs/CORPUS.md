# Training corpus and model fine-tuning

The labelled episodes are, once there are enough of them, a large in-domain speaker corpus: ~18-20 castaways plus
the host per season, 40+ seasons, an hour or so of speech per castaway, recorded outdoors with music, wind and
crosstalk — exactly the acoustic conditions the off-the-shelf models were not trained on. This document says what
we export, how trustworthy it is, and what to train on it, in the order that pays off.

## What is exported (`survspk export-corpus <dir>`)

```
<dir>/
  database.yml            pyannote.database file; protocol Survivor.SpeakerDiarization.All
  audio/<uri>.flac        symlinks into audio/<vs>/E<ep>.flac (flat, because pyannote resolves only {uri})
  lists/{train,dev,test}.lst
  rttm/<uri>.rttm         SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker_id> <NA> <NA>
  uem/<uri>.uem           <uri> 1 <start> <end>   -- the ANNOTATED regions (see below)
  spk/{train,dev,test}.csv   id, wav, start_s, end_s, duration, speaker_id, season, episode, source, domain, uri
  spk/trials_dev.txt      <0|1> <enrol_id> <test_id>   cosine-verification trials on the dev split
  stats.json
```

`<uri>` is `<version_season>_E<ep>`, e.g. `US47_E02`. Body segment only (no recap / next-time).

### Which labels go in

Exactly the labels the speaker bank itself trusts, and nothing else:

* human labels, always. Pseudo ids: `OTHER` is kept as one anonymous speaker per file (real speech, unknown
  identity — useful for the diarizer, useless for the embedder, so it is in the RTTM but not in `spk/`);
  `NOSPEECH` is kept as annotated non-speech (in the UEM, no SPEAKER line); `UNKNOWN` is cut out of the annotated
  regions entirely (we do not know what is there).
* explicit `NAME:` SDH utterances plus the rest of their run, and confident `auto` labels — after the bank's
  self-consistency filter (`stage_bank.self_consistency_filter`: an utterance whose voice confidently sits with a
  different speaker is dropped). Same code path as `bank`, so a fix there is a fix here.
* never the LLM text prior (retired; 26-37% on the A/B).

### Why a UEM at all

Captions cover most of the speech but not all of it, and they never mark overlap. If we handed the trainer the
whole episode with only the labelled spans as speech, every unsubtitled line and every UNKNOWN span would be taught
as *silence*, which is worse than teaching nothing. So the UEM says where we actually know the answer: every trusted
span with a 0.2 s margin, plus the gaps between consecutive spans in which the WebRTC VAD hears (almost) no
speech, as long as the gap is under 30 s. Those gaps become supervised non-speech. Everything else is excluded and
costs the trainer no loss. `--no-vad` turns gap bridging off (spans + margins only).

Two limits to state whenever a number is reported from this corpus. Overlap is under-annotated (a two-speaker span
usually carries one caption name), so a diarizer trained here will learn turn-taking well and crosstalk less well.
Timing is caption timing refined by the alignment stage (±0.1-0.3 s), not forced alignment; the 0.2 s margin
absorbs most of it, and the embedding recipe uses random crops from inside the span, which is robust to it.

### Splits

* `test`: whole seasons named with `--test-season` (unseen speakers — the honest generalisation number).
* `dev`: the last labelled episode of each season (held-out episodes of *seen* speakers — the number the assign loop
  cares about, since the bank always knows the cast).
* `train`: everything else.

Verification trials on dev: target pairs are two utterances of the same speaker; non-target pairs are two speakers
from the *same season* (hard negatives — same location, same mixing, same season of the show). Random cross-season
negatives would make the EER look better than it is.

### How much is enough

Per speaker the embedding recipe wants ≥ 8 utterances ≥ 1 s in train (below that a speaker is dropped from the
classification head but still counted in trials). US47 with E01-E02 labelled gives ~18 speakers and a few minutes
each — enough to run the tooling and get a *baseline* EER, not enough to fine-tune without over-fitting. A useful
first training run is on the order of 3+ seasons (~60 speakers, several hours); a serious one is the whole
SDH-named era (S40+, and S1-20 once their chyron-based bootstrap exists).

## What to train, in order

### 1. Speaker embedding (`survspk finetune-ecapa <corpus> <out>`)

The bottleneck in our own loop is embedding quality on field speech: the E02 held-out run already agrees with
explicit names on 96% of confident runs but confidently places only ~55-60% of body time, and the misses are
outdoors, windy, musical, or overlapped. Speaker-verification adaptation on in-domain data is the textbook fix and
it drops straight back into the pipeline.

Recipe: pretrained `speechbrain/spkrec-ecapa-voxceleb` (feature extractor + normaliser + ECAPA-TDNN) with a fresh
additive-angular-margin softmax head over the training speakers (m = 0.2, s = 30); 3 s random crops from inside
each utterance; head at lr 1e-3, encoder at 1e-4 and frozen for the first epoch; Adam; MPS on the Mac. Before the
first step it embeds the dev utterances with the *untouched* pretrained model and prints its EER on the trials —
that is epoch 0, the number to beat, and it appears in `finetune.json` and the console before any training.

```bash
uv run survspk export-corpus corpus/                      # stats + a baseline-sized dev set
uv run survspk finetune-ecapa corpus/ models/ecapa-survivor --epochs 10
uv run survspk embed US47 2 --model models/ecapa-survivor  # the checkpoint is a SpeechBrain directory
uv run survspk ablation US47 2                            # compare against speechbrain/spkrec-ecapa-voxceleb
```

The checkpoint directory contains `embedding_model.ckpt` (fine-tuned) plus the pretrained `hyperparams.yaml` and the
other files SpeechBrain's `EncoderClassifier.from_hparams` expects, so `--model <dir>` works exactly like the
hub name. `finetune.json` keeps the per-epoch loss / EER history. `--smoke` runs the loop on a random-initialised
ECAPA without downloads (what the test suite does).

Accept the fine-tuned model when, on the same held-out episode, both (a) dev EER drops against epoch 0 and (b)
`assign` places more body time at equal or better explicit-name agreement. (a) without (b) is over-fitting to the
seen speakers.

### 2. Segmentation / diarization (`scripts/finetune_pyannote_seg.py`)

Fine-tunes `pyannote/segmentation-3.0` with pyannote's own `SpeakerDiarization` task on the exported database
(10 s chunks, ≤ 4 speakers per chunk, ≤ 2 per frame, lr 1e-4, early stopping on the task's own validation metric).
The trained model replaces the segmentation component of the pyannote pipeline. Where it matters: the `>>`-era
seasons (S21-39) whose captions carry no names, where the episode-1 bootstrap relies on diarization quality, and
overlap handling everywhere — with the caveat above that the corpus under-annotates overlap, so measure DER on a
hand-checked few minutes before trusting a gain there.

```bash
uv run python scripts/finetune_pyannote_seg.py --corpus corpus/ --out models/seg-survivor --epochs 20
```

Needs the `diarize` extra and `HF_TOKEN` (gated model). Untested in the Cowork VM (no pyannote there); it follows
the pyannote 3.x fine-tuning recipe verbatim.

### 3. Not planned: an end-to-end text-conditioned model

Text carries little speaker identity here (the text-prior A/B), and the audio path is already at 96% on confident
runs; the remaining gains are acoustic.

## Refreshing the corpus

Labels change (review, refit, re-assign), so re-export before every training run — it is fast (seconds per
episode; the VAD pass on the vocals stem is the slow part). Episodes whose parquet is stale fall back to human
labels only and say so in the log; run `assign` (which re-embeds) first.
