# survivor-speakers (`survspk`)

Speaker-attributed Survivor transcripts. Spec: `../survivor_speaker_pipeline_spec.md` (v0.2).
Data findings: `../survivor_pipeline_data_findings.md`.

## Setup (macOS, M3 Ultra)

```bash
cd survivor-speakers
rm -rf .venv                 # only if an earlier sync picked the wrong Python
uv sync                      # M0 stack; uv installs CPython 3.12 itself (.python-version)
uv run survspk info          # should show profile: macos and the NAS paths as ok
uv run pytest

# everything (M1+): torch, speechbrain, demucs-mlx, whisperx, pyannote, plexapi, fastapi
uv sync --all-extras
# NB: `uv sync --extra X` installs ONLY extra X and removes the others — always use --all-extras here
cp .env.example .env                        # fill HF_TOKEN, PLEX_URL, PLEX_TOKEN, OMLX_BASE_URL/OMLX_API_KEY
```

Python: 3.12 (pinned in `.python-version`). Homebrew's 3.14 has no torch wheels.

Paths are chosen per *profile* in `config/default.yaml` (`macos` on the Mac, `vm` inside the Cowork
sandbox). Override with `SURVSPK_PROFILE=...`.

## M0 (no audio)

```bash
uv run survspk refresh-survivor        # survivoR tables -> survivor_audio/survivor_data/survivor.sqlite
uv run survspk inventory               # all seasons; ~1-2 s per file over the network; resumable
uv run survspk report                  # coverage, missing subtitles, duration mismatches
uv run survspk ingest-subs             # parse chosen subtitles -> cues; detect conventions
uv run survspk names-report            # SDH NAME: tokens vs survivoR; unresolved list
uv run survspk grab-frame US47 2 162   # chyron frame check
uv run pytest
```

Everything is idempotent and keyed on `(version_season, episode)`; `--force` re-does an episode.

## M1 (audio; run on the Mac)

```bash
uv run survspk run US47 2 --force            # extract -> separate -> align -> segment -> embed (all variants)
uv run survspk run US20 2 --force
uv run survspk ablation US47 2               # leave-one-out speaker-ID accuracy per audio variant + t-SNE plots
uv run survspk ablation US20 2
uv run survspk sync-check US11 -e 1 --variant vocals   # drift check on a clean stem (needs: run US11 1 --through separate)

# challenge ECAPA with another embedding model (speechbrain or pyannote ids; pyannote needs --extra diarize + HF_TOKEN)
uv run survspk embed US47 2 --variant vocals --model pyannote/wespeaker-voxceleb-resnet34-LM
uv run survspk ablation US47 2 --model pyannote/wespeaker-voxceleb-resnet34-LM
```

Embeddings for a non-default model are stored beside the default ones with a model tag suffix, so the two can be
compared without re-running anything else. The ablation reports per-utterance accuracy and, more usefully,
per-**run** accuracy (`conf_run_acc`, `run10_acc`, `field_run_acc`) — runs are what M3 classifies.

First run downloads models once: htdemucs (~80 MB), the WhisperX English alignment model (~360 MB, from
torch/HF), and ECAPA (~80 MB, from Hugging Face). Stage outputs land in `survivor_audio/`:
`raw/`, `center/` (5.1 sources), `vocals/`, `vocals_center/`, `embeddings/`, `reports/`.

Stages are individually runnable (`extract`, `separate --source raw|center`, `align --variant`, `segment`,
`embed --variant`) and idempotent; `--force` redoes an episode.

### What M1 verified without the Mac (Cowork VM, 2026-09-07)
* extract: US47E02 raw + center in 41 s over the NAS; 16-bit FLAC (24-bit was 2x the size).
* offset/drift estimator: recovers constant offsets to 0.15 s and a 4.3% frame-rate drift to 0.2% on synthetic
  data; on the real US47E02 it reports rate 1.0000 / offset 0.00 s with the center channel's per-window lags
  agreeing within 0.12 s. WebRTC VAD as the envelope raised confidence from ~1 to 7-9.
* **Known limitation:** on raw audio with a wall-to-wall music bed (US11E01, SDTV) the correlation is not
  trustworthy — that is what the vocals stem is for. Do not trust `sync-check --variant raw` on classic seasons.
* segment: turn splitting, name propagation, merge rules and recap detection are unit-tested on real fixtures
  (29 tests). Recap end = the previous tribal's verdict phrase (present in 461/474 recaps).

### First real results (US47E02, 2026-09-07, M3 Ultra)
* Whole pipeline: ~4.5 min per 64-min episode. Demucs 57-68x realtime; WhisperX alignment 59 s (1450/1450 cues,
  9,690 words, mean score 0.71); ECAPA 3 s per audio variant on MPS.
* Offset estimator on the vocals stem: rate 1.0000, offset +0.08 s, confidence 9.3 — trustworthy on separated audio.
* 1,485 cues -> 1,623 turns -> 1,035 utterances; 507 carry an SDH name (505 resolve to survivoR ids).
* Leave-one-out speaker ID over 14 speakers on SDH-labelled body utterances (>=1.5 s, n=247): **74% centroid**,
  identical across raw / center / vocals / vocals_center. Accuracy is driven by duration: 54% under 1.5 s,
  83-89% above 4 s. Inherited (propagated) names are as accurate as explicit ones once duration is controlled.
* **Confessional cross-check vs survivoR hand counts** (structural rule: unmarked same-name run >= 6 s):
  Spearman 0.85 on counts, 0.88 on time, MAE 1.4 confessionals/player, 17 players. This is the free
  end-to-end check from the spec (§10) working on the first episode.
* Run-level (pooled) accuracy: ~80% on confessional runs, ~86% on runs >= 10 s, 55-63% on field runs.
* US20E02 (`>>` era): offset +0.38 s, drift -0.0014 applied, 1374/1374 cues, 898 utterances; only PROBST is
  named, so the ablation is degenerate there — `>>`-era bootstrap relies on chyron OCR + diarization (M2).
* Implication for M3: classify **runs** (monologues) as a unit and build the bank from long utterances; the
  audio-variant choice matters less than utterance length at this stage.

Full write-up: `docs/M1_REPORT.md`.

## M2/M3 — bank + assign (SDH-named seasons need no bootstrap)

```bash
uv run survspk run US47 1                    # episode 1 through embed (~4.5 min)
uv run survspk bank US47 --episodes 1        # fit speaker_bank as of E01 from E01's explicit SDH labels
uv run survspk assign US47 1 --bank-as-of 1  # E01 touch-up: score E01's *unlabelled* runs against its own bank;
                                             #   confident ones -> auto, the rest -> review queue
uv run survspk review                        # label E01's queue by ear (this is the spec's episode-1 bootstrap)
uv run survspk bank US47 --episodes 1        # refit: human + explicit + confident auto -> every castaway banked
uv run survspk assign US47 2                 # now E02; agreement vs E02's explicit names is the held-out number
uv run survspk run-errors US47 2             # (LOO within one episode; the assign numbers above are the held-out ones)
```

Order matters: touching up episode 1 first is what fills the bank for the castaways who never got an explicit
`NAME:` line (US47E01: Sue, Sol, Caroline, Kyle); reviewing episode 2's queue first treats the symptoms instead.

You do not label the whole queue. The UI's *bank coverage* panel shows, per castaway, how much labelled speech the
bank has from this episode against the target (≥ 5 lines and ≥ 15 s — the consistency filter's minimum pool) and
splits the rest into *thin* (below target, and the queue still holds runs that could be theirs — label these) and
*quiet* (below target with nothing plausible left: they barely speak this episode; skip them, a later episode banks
them). "thin speakers first" orders the queue so the thin ones come up. When the panel says done, press **refit &
reassign** (bank from episodes 1..N with your labels, re-assign this episode; human labels are never overwritten;
re-embeds first if you split lines), and the runs the fuller bank can now place leave the queue. Two or three
rounds is typical; what remains is short fragments and overlapping chatter that neither the bank nor the features
need.

**Audit the auto labels.** The agreement numbers `assign` prints are measured on explicit `NAME:` lines, and those
belong to the castaways the bank already knows well, so they overstate the auto labels' precision. The UI's
*audit auto labels* mode draws a stable random sample of 20 auto-labelled runs per episode, spread across the
predicted speakers; `y` confirms, any other pick corrects. Every verdict is a human label, the header shows the
precision with a 95% lower bound, and `survspk audit-stats US47` prints it per episode with the confusions.
Run it once per episode after the review loop; it is the unbiased number to record.

**Re-segmenting a labelled episode.** Human labels are stored with their time span and carried over to the new
utterances by overlap (`segment` re-anchors them; `--force` drops the ones no new utterance covers). Word-level
splits do not survive a re-segment.

```bash
uv run survspk review                        # http://127.0.0.1:8765  (keys: 1-9, s, a, o, u, x, space, z/↩ undo; click a line to label
                                             #   just that line; ✂ split cuts a line at a word, then part 1 / part 2 are selected in turn)
# then grow the bank with the human labels and move on
uv run survspk bank US47 --episodes 1,2
uv run survspk assign US47 3
```

`bank` uses explicit `NAME:` utterances plus the rest of their run, drops utterances whose voice confidently sits with
another speaker (label noise), and stores a recency-weighted centroid + 20 farthest-point exemplars per speaker per
domain. `assign` pools each run, first splitting a run wherever a confidently identified speaker changes, then decides
`auto` / `low_margin` / `no_candidate` per `thresholds` and writes `labels` (source `sdh` for explicit names, `auto`
otherwise) and `review_queue`. Explicit SDH names are scored as if unlabelled and reported as
`sdh_agreement_runs` — the free per-episode accuracy monitor from the spec (§7.7.6). Candidates come from survivoR's
`boot_mapping` (recap uses the previous episode's cast). Two text rules sit on top of the audio: a run that names its
predicted speaker in the third person is queued (`name_mentioned`), and a run whose *some* utterances name the
predicted speaker is split there first (Jeff's question + the castaway's answer).

**Text prior (experiment, retired from the default flow).** `survspk text-prior` asks a local LLM (oMLX,
OpenAI-compatible) who speaks a run from the transcript, the survivoR cast sheet and the neighbouring runs.
US47E02 A/B on 60 explicitly-named runs: Qwen3.6-35B-A3B 8-bit 26%, the same with thinking 27%, Qwen3.6-27B bf16
37.5% (and it answered only 40 of 60). Confidence is uncalibrated (0.9-0.95 on nearly everything). Failure modes:
picks people the passage names, invents show knowledge, ignores tribe, over-attributes challenge cheers to the
host. Reliable only on host procedural lines and sentence continuations, which the audio already handles. Kept
as a research hook (`--eval-only`, `--thinking`, `--model`, `--clear`); `text_prior.use_in_assign` and
`show_in_review` stay off.

## Chyron OCR — free labels in every era

The on-screen name card (`TEENY / FREELANCE WRITER / LAVO TRIBE`, bottom-left, ~3 s into a castaway's first line)
is one speaker label per castaway per episode, and the only text label the S21–39 subtitles do not have.

```bash
uv sync --all-extras                          # adds the `ocr` extra: ocrmac (Apple Vision) + pillow
uv run survspk chyron US47 2                  # OCR the body, store chyron_hits + scenes, write `chyron` labels
uv run survspk chyron US47 2 --no-write-labels   # hits only
```

ffmpeg decodes the body on the video engine, keeps the bottom band, and only frames whose band changed are OCR'd.
OCR lines on the left are matched against the cast present (rapidfuzz ≥ 85, or one character off on a name of
four letters or more; occupation on the same line lifts a shaky match); lines in the centre are the show's own
captions and are ignored; `TRIBE / DAY n` cards go to `scenes`. Each hit labels the utterance that started ~3 s
before the card. A human label wins; an explicit SDH name that disagrees leaves the SDH label and queues the run as
`chyron_conflict` (key `c` in the UI takes the chyron's word). On named episodes the command prints agreement with
the SDH names, which is the OCR's precision check — read it on US47 E01/E02 before trusting it on a `>>` season.
The bank treats a chyron-anchored run like an SDH-anchored one (`chyron` + `chyron_run`).

**Check the cards (review UI, mode *check name cards*).** A card says who is on screen, not which line they speak;
the automatic anchor is right about three times in four, and one wrong card teaches the bank the wrong voice. For each
card the UI shows the frame just after it appears, the card text, and the lines from 8 s before to 4 s after it (the
automatic pick marked). Press the number of the line the carded castaway speaks, or `0` for none of them. The answer
is stored in `card_checks` and becomes a `chyron` label at confidence 1.0; re-runs of `survspk chyron` keep it; a
human label on the line still wins; `z` undoes. When every card is checked, *refit & reassign* builds the bank from
them and switches to the queue sorted thin-first, for castaways who never got a usable card.

For a season whose captions name nobody (S21-39):

```bash
uv run survspk run US25 1 && uv run --with ocrmac survspk chyron US25 1     # repeat for E02, E03
uv run survspk review                                                        # mode: check name cards, E01..E03
#   check every card -> refit & reassign -> label the castaways the coverage panel calls thin -> audit
```

`survspk chyron VS EP --from-cache` replays the saved raw OCR in seconds (after changing the matcher or filters).

## Corpus export and fine-tuning (see `docs/CORPUS.md`)

Every labelled episode can be exported as training data — a pyannote diarization database (RTTM + UEM of the
regions we actually know, lists, `database.yml`) and a speaker-embedding manifest with same-season verification
trials. Only the labels the bank trusts go in (human, explicit SDH + run, confident auto after the consistency
filter); UNKNOWN spans are excluded, unsubtitled speech is left unannotated rather than taught as silence.

```bash
uv run survspk export-corpus corpus/                          # rttm/ uem/ lists/ spk/ database.yml stats.json
uv run survspk finetune-ecapa corpus/ models/ecapa-survivor   # prints the pretrained model's dev EER first, then trains
uv run survspk embed US47 2 --model models/ecapa-survivor     # checkpoint dir is a SpeechBrain model dir
uv run python scripts/finetune_pyannote_seg.py --corpus corpus/ --out models/seg-survivor   # diarizer (needs HF_TOKEN)
```

Two labelled episodes are enough to run the tooling and read the baseline EER, not to train: plan the first real
embedding fine-tune at ~3 seasons and the diarizer at more. Accept a fine-tuned embedder only when dev EER drops
*and* `assign` places more body time at equal or better explicit-name agreement on the same held-out episode.
