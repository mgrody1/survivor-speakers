# M1 report — audio path on two episodes

Date: 2026-09-07. Machine: M3 Ultra (512 GB), macOS, `uv` env with extras `audio` + `align`.
Episodes: **US47E02** (new era, 5.1 source, `NAME:` + dash SDH) and **US20E02** (`>>` era, PROBST-only names).
Spec: `../../survivor_speaker_pipeline_spec.md` §7.2–7.5, §8.3, §10.

## 1. Throughput

| stage | US47E02 (64 min) | notes |
|---|---|---|
| extract | ~40 s | ffmpeg over the NAS; raw + center (5.1) as 16 kHz mono 16-bit FLAC |
| separate | ~60 s | demucs-mlx htdemucs, 57–68x realtime; `[convert]` extra required |
| align | ~60 s | WhisperX wav2vec2 per cue: 1450/1450 cues, 9,690 words, mean score 0.71 |
| segment | < 5 s | |
| embed | ~3 s per variant | ECAPA on MPS, batched |
| **total** | **~4.5 min** | one episode; four audio variants embedded |

At this rate the 686-episode library is ~2 days of unattended compute for M4–M6, not a bottleneck.

## 2. Offset / drift estimator (§7.3)

* US47E02, vocals stem: rate 1.0000, offset +0.08 s, confidence 9.3.
* US20E02, vocals stem: offset +0.38 s, drift −0.0014 (≈ −3.6 s over the episode, applied), 1374/1374 cues aligned.
* The per-window spread check fired a false alarm on US20E02 (52.5 s) because one window with a near-zero peak was counted. Now only windows whose peak is ≥ 50% of the best window contribute to the spread test.
* Raw classic-era audio with a continuous music bed (US11E01) still cannot be trusted for the estimator; always run it on the vocals stem. Documented in the spec.

## 3. Segmentation (§7.4)

US47E02: 1,485 cues → 1,623 turns → 1,035 utterances; 507 carry an SDH name, 505 resolve to survivoR ids.
US20E02: 898 utterances; only PROBST is ever named (as expected for the `>>` era).

Recap end is the previous tribal's verdict phrase (the earlier "first 4 s gap" rule fired on music beats).
Runs (unmarked, name-compatible consecutive utterances within 1.5 s) are recorded as `run_id`, `run_dur_s`; a run ≥ 6 s or an italic run is labelled `domain_hint = confessional`.

**Confessional cross-check against survivoR hand counts (§10, free end-to-end check):** on US47E02 the structural rule alone gives Spearman 0.85 on confessional count per player, 0.88 on confessional time, MAE 1.4 confessionals per player over 17 players. No audio model is involved in this number; it validates SDH name propagation + run detection.

## 4. Embedding ablation (§8.3)

Leave-one-out centroid classification over SDH-labelled body utterances (14 speakers, ≥ 1.5 s):

| unit | accuracy | n | comment |
|---|---|---|---|
| utterance | 67–74 % | 247–300 | identical across raw / center / vocals / vocals_center |
| utterance < 1.5 s | ~54 % | | |
| utterance > 4 s | 83–89 % | | |
| confessional run (pooled) | 79–82 % | 85 runs | `conf_run_acc`; see 4b for what the 20 % are |
| run ≥ 10 s | 86 % | 50 runs | `run10_acc` |
| field run | 55–63 % | | short, overlapping, music; many are two-speaker runs |

Findings:

1. **The audio variant does not matter at this stage.** Raw, center, Demucs vocals and Demucs-of-center give the same accuracy to within noise. Keep `vocals` as the default (it is what makes the offset estimator reliable), keep `raw` on disk until M4 for the `snr_proxy`, and stop producing `vocals_center`.
2. **Duration dominates.** Accuracy is first a function of utterance length. In aggregate, inherited names score like explicit ones once duration is controlled — but §4b shows that the *confident* errors are concentrated in inherited names that crossed an unmarked speaker change.
3. Therefore **M3 classifies runs, not utterances**, and builds the bank from long utterances / pooled runs. Utterances inherit their run's label.
4. 86 % on 10-second confessional runs looked low for ECAPA, so a second embedding model was compared (§4b): no difference. The remaining errors are in the labels, not the embeddings.
5. US20E02's ablation is degenerate (one labelled speaker). `>>`-era episodes get **no cast labels from subtitles**; their bootstrap (M2) depends on chyron OCR + diarization, exactly as the spec assumed. The t-SNE plot for US20E02 is still useful as a visual cluster check.

### 4b. Second model, and what the errors actually are

`pyannote/wespeaker-voxceleb-resnet34-LM` on the same vocals stem: utt 0.684, confessional runs 0.80 (85), runs ≥ 10 s
0.84 (50), field 0.61 — indistinguishable from ECAPA (0.667 / 0.80 / 0.86 / 0.55) and 36x slower unbatched.
**ECAPA stays.** Four audio variants and two models all land on the same ceiling, so the ceiling is not the embedding.

`survspk run-errors US47 2` lists the 17 wrong confessional runs with time, provenance and text. Read against the
cast list (US0702 Gabe, US0709 Sam, US0712 Sue, US0714 Tiyana, US0715 TK, HOST = Probst):

| cause | runs | examples |
|---|---|---|
| **two speakers merged into one run** (host Q + castaway answer at tribal; camp dialogue without a dash between cues; challenge chatter) | 11 | 55:37 "Tiyana, do you agree with that? I honestly do agree…" labelled PROBST, predicted Tiyana (margin 0.49); 41:01 castaway line + Jeff's challenge call labelled Rome, predicted HOST |
| **wrong label on a single-speaker run** (name propagated across a speaker change; the classifier is right) | 4 | 26:49 "I have Sue and Caroline, they're my little birds" labelled *Sue*, predicted Gabe, `sim_true` −0.08; 13:18 "I feel like Sam and the women will come back to camp" labelled *Sam*, predicted Andy |
| **plausible voice confusion** | 2 | 23:13 Andy → Sam (margin 0.30, both Gata men); 56:18 TK → Sue (margin 0.07, 5-utterance tribal answer, possibly also mixed) |

So roughly 15 of 17 "errors" are label or segmentation faults in the subtitle-derived truth, and the classifier's own
accuracy on clean single-speaker confessional runs is in the mid-90s. Three consequences:

1. Dash-convention files mark a speaker change only *inside* a shared cue; between cues an unnamed speaker change is
   invisible to text rules. Runs therefore cannot be assumed pure. **M3 must check run purity with the embeddings
   themselves** (utterance vectors within a run that split into two confident clusters → split the run) and/or with
   pyannote speaker-change points, before a run is scored or banked.
2. Inherited names are weak labels. The bank is built from *explicitly* named utterances (`n_explicit ≥ 1`) that are
   also self-consistent under leave-one-out (`sim_true` close to `sim`); inherited names are verified against the
   bank, not trusted.
3. One cheap text rule removes the most confident class of error: a host turn that addresses a castaway by name and
   asks a question hands the name to the next unnamed turn (`ADDRESS_Q_RE` in `stage_segment.propagate_names`).
   Implemented and unit-tested; statements without a question ("Gabe, your enthusiasm…") deliberately do not
   hand over, since the host usually keeps talking.

Also visible in the table: speakers with fewer than three labelled runs in the episode (Kyle, Caroline, Rachel,
Sierra, all of Lavo but Rome and Teeny) are not candidate classes at all, so anything they say is charged to someone
else. Within one episode that is unavoidable; across a season the bank fills in, and the §8 floor threshold
(`thresholds.floor`) is what must catch a speaker who has no bank entry yet.

## 5. Changes made during M1

* `roots` table + `localize()` so stored paths translate between the VM and the Mac.
* Per-cue WhisperX calls (the whole-episode call re-split segments on sentence boundaries: 1799 vs 1450).
* 16-bit FLAC (`-sample_fmt s16`); 24-bit doubled the size for nothing.
* Global rate-scan offset/drift estimator with far-field confidence; windowed correlation dropped.
* Run detection and `domain_hint`; confessional cross-check metrics (`confessional_spearman_count/time`, `confessional_mae_count`).
* `Encoder` wrapper with speechbrain and pyannote backends; `--model` on `embed`, `ablation`, `run`; run-level ablation columns (`conf_run_acc`, `run10_acc`, `field_run_acc`).
* Peak-weighted window-spread warning.
* Embedding parquets now store each utterance's span and are checked for freshness against the `utterances` table:
  utt ids are positional, so a re-segmentation that adds one utterance shifts every later id and silently mis-pairs
  vectors with rows (this produced a phantom "Jeff's question predicted as Tiyana, margin 0.54" after the first
  re-segment). `embed` re-runs automatically when stale; `ablation` / `run-errors` refuse stale parquets.
* Host-question hand-over rule in `propagate_names` (§4b).

## 6. Open before M2

* Re-segment US47E02 with the host-question rule and re-run `run-errors` to confirm the tribal Q&A cases clear.
* `sync-check --variant vocals` on the 17 drift-suspect episodes from M0 (needs `run <vs> <ep> --through separate` first).
* Scene boundaries (location cards) for the T10/T19 features: fold into the chyron OCR pass in M2.
