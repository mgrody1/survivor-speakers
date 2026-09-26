# Handoff — survivor-speakers (`survspk`)

Written 2026-09-10 for whoever picks this up next (another Claude session or a human). It says where things are,
how work gets done, what has been decided and why, the exact current state, and what comes next. Read this,
then `README.md`, then `docs/CORPUS.md`; the spec is `../survivor_speaker_pipeline_spec.md` (v0.2).

## 1. People, machines, paths

* User: **Max** (`mgrody1@umbc.edu`). Runs everything on a Mac called **m3-ultra** (M3 Ultra, 512 GB).
* Project root on the Mac: `/Users/maxgrody/Documents/Claude/Projects/Stargazer/survivor-speakers/`
  (package `survspk`, `uv` project, Python 3.12, tests under `tests/`).
* Data root (`paths.work_root`, macos profile): `/Users/maxgrody/Documents/Claude/Projects/Stargazer/survivor_audio/`
  — `db/survspk.sqlite` (the one database; ~300 MB), `raw/ vocals/ center/ embeddings/ subs_embedded/ frames/
  reports/ survivor_data/`. Audio per episode: `<variant>/<vs>/E<ep>.flac`, 16 kHz mono.
* survivoR snapshot (cast/boot data): `.../Stargazer/Gamebot/gamebot_lite/data/gamebot.sqlite`.
* Video/subtitle sources (read-only mounts): `/Volumes/Max2NUC Files/TV Shows/Survivor (2000) {TvbId-76733}`,
  `/Volumes/Max2NUC Files/Subtitles`. Plex serves them; `survspk fetch-subs` pulls SDH subtitles.
* Local LLM: oMLX on `http://127.0.0.1:8001/v1` (OpenAI-compatible; Qwen3.6-35B-A3B-8bit, Qwen3.6-27B). Only used
  by the retired `text-prior` experiment.
* Secrets live in `.env` (gitignored; not to be written by Claude): `HF_TOKEN`, `PLEX_URL`, `PLEX_TOKEN`,
  `OMLX_API_KEY`. Max adds lines himself; `.env.example` documents them.

## 2. How work gets done (the working agreement)

* Claude writes code in its own workspace and pushes files to the Mac with the device bridge
  (`device_commit_files` to the absolute Mac path, `force: true`), then verifies with `md5sum` on both sides via
  `device_bash` (`cd $HOME/mnt/Stargazer/survivor-speakers`). Every push so far has been md5-verified.
* Max runs all `uv run survspk …` / `uv run pytest` commands on the Mac and pastes output. Claude does not run the
  pipeline on real audio (no audio/models in the sandbox); Claude runs the test suite on synthetic data.
* `uv sync --all-extras` is the install command (a partial `--extra` removes the others). Dev group has pytest,
  ruff, httpx2 (starlette TestClient).
* The Mac's `uv run pytest -q` should report **65 passed** (~3.5 min; the ECAPA smoke test dominates).
* The DB on the Mac is the source of truth for labels. Never regenerate it; migrations in `survspk/db.py` are
  additive. Human labels are protected in every write path (`stage_assign._write`, refit).
* Reports go in `docs/` (M0_REPORT, M1_REPORT, CORPUS, this file). The spec gets status notes at milestones.

## 3. What the system does (one screen)

Speaker-attributed transcripts for Survivor (US seasons first). Per episode:
`fetch-subs → ingest-subs → extract (audio) → separate (vocals stem) → align (WhisperX word timing) → segment
(utterances, runs = monologues, SDH names, host rules) → embed (ECAPA vectors per utterance, parquet) → bank
(per-castaway voice profiles from trusted labels) → assign (score every unlabelled run, write labels or queue)
→ review (human UI) → refit`.

Key design points (all in code and README):
* Unit of classification is the **run** (a monologue), not the caption line; runs are split where a confidently
  identified speaker changes, and around lines that name the predicted speaker.
* **Bank** = per (speaker, domain ∈ {confessional, field}) recency-weighted centroid + 20 farthest-point exemplars.
  Fed by human labels, explicit `NAME:` SDH lines and the rest of their run, and confident auto labels; a
  two-pass self-consistency filter drops label noise (needs ≥ 5 utts / ≥ 15 s per speaker to judge).
* **Assign** thresholds: accept 0.55, margin 0.08, floor 0.35 (`config/default.yaml`). Decisions: auto |
  low_margin | no_candidate | name_mentioned | sdh_conflict; the last four go to `review_queue`.
* Label sources, by trust: human > chyron > sdh (explicit + same run) > auto (≥ accept + 0.05). Text prior: never.
* Parquet freshness: embeddings store start/end per utt; splits change utt ids → `StaleEmbeddings`; `assign` (CLI)
  and the UI's refit re-embed automatically.
* Candidates per episode come from survivoR `boot_mapping` (recap uses ep−1's cast). Host is `HOST_US`.

## 4. Decisions made, with the evidence

* **Episode 1 first.** The bank is fit from E1; castaways with no explicit `NAME:` line in E1 (US47: Sue, Sol,
  Caroline, Kyle) are missing from it, so E2's queue is large. Fix the cause: review E1 → refit → then E2. Max
  agreed; the README documents the sequence.
* **Held-out number** (E2 scored with the E1 bank): 95.9 % agreement with explicit SDH names on confident runs
  after the mention rule; but only ~55–60 % of body time placed confidently — the thin E1 bank is the limit.
* **LLM text prior retired.** oMLX A/B on 60 explicitly named E2 runs: Qwen3.6-35B-A3B 26 %, +thinking 27 %,
  Qwen3.6-27B bf16 37.5 % (40/60 answered). Max's bar was 80 %; we cut it (`text_prior.use_in_assign: false`,
  `show_in_review: false`). Code kept as a research hook (`survspk text-prior --eval-only`). Text carries little
  speaker identity here; the gains left are acoustic.
* **Review loop, not full labelling.** Label until every castaway who speaks enough reaches ≥ 5 lines / ≥ 15 s
  from this episode, press *refit & reassign*, repeat 2–3 times. Castaways with nothing plausible left in the
  queue show as *quiet* and are skipped (later episodes bank them).
* **Corpus for fine-tuning: infrastructure done, data not.** `export-corpus` (RTTM/UEM/database.yml + speaker
  manifest + same-season trials) and `finetune-ecapa` (ECAPA + AAM-softmax, baseline EER printed as epoch 0) are
  built and tested. First real training run at ~3 labelled seasons; diarizer (`scripts/finetune_pyannote_seg.py`)
  later and with the overlap caveat. Acceptance rule in `docs/CORPUS.md`.
* **WeSpeaker vs ECAPA**: ablation kept SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) as default.

## 5. Current state (Mac DB, 2026-09-09 22:00 UTC)

| | labels | open queue |
|---|---|---|
| US47 E01 | sdh 315 · auto 541 · human 3 | 492 (low_margin 251, no_candidate 226, sdh_conflict 12, name_mentioned 3) |
| US47 E02 | sdh 239 · auto 285 · human 25 | 414 (low_margin 189, no_candidate 225) |

* `speaker_bank`: US47 as_of 1, 34 entries (17 speakers × 2 domains).
* Audio present: raw US11/US20/US47; vocals + embeddings US20/US47. Only US47 E01–E02 are segmented/labelled.
* `corpus/` on the Mac was exported 2026-09-09 23:38: 2 episodes, 0.46 h speech, 18 speakers, 4000 dev trials.
  **Caveat:** E01 exported with human labels only (3) — its embeddings were stale at the time (a split), so
  `collect_labelled` fell back. Re-export after the E1 refit; expect ~0.7 h from E01 alone.
* The E1 review has begun (3 human labels). E2's 25 human labels predate the "E1 first" decision and are fine —
  they join the bank at `bank --episodes 1,2`.
* Review UI state (latest): candidates pane above coverage; per-line ▶; click a line to label only it (⇧ range,
  ⌘ add, esc all); ✂ split → part 1 auto-selected → label → part 2 auto-selected; coverage panel with explicit
  target, thin/quiet/banked states and a done message; refit auto re-embeds after splits; `↩ undo` button (and
  `z`) with per-episode history in localStorage.

## 6. The loop Max is running

```bash
cd ~/Documents/Claude/Projects/Stargazer/survivor-speakers
uv run survspk review                         # http://127.0.0.1:8765 → US47 E01, sort "thin speakers first"
#   label until the coverage panel says done → refit & reassign → repeat 2-3×
uv run survspk bank US47 --episodes 1
uv run survspk assign US47 2                  # held-out number: sdh_agreement_runs, body_auto_dur_share
uv run survspk review                         # E02, same loop (short)
uv run survspk bank US47 --episodes 1,2
uv run survspk run US47 3                     # next episode through embed (~4.5 min), then assign US47 3
uv run survspk export-corpus corpus/          # any time: corpus stats; finetune-ecapa corpus/ models/x --epochs 1 for baseline EER
```

Other commands: `run-errors US47 2` (LOO error list), `ablation US47 2` (embedding models), `segment --force`
(re-segment; refuses if human labels exist unless forced), `text-prior` (retired), `refresh-survivor`.

## 7. Next steps, in order

1. Finish the E1 review loop (Max); then `bank --episodes 1`, `assign US47 2`, record the held-out numbers
   (agreement on confident runs, share of body time placed) in `docs/M1_REPORT.md` §4b or a new M2 note.
2. E2 review (should be short), `bank --episodes 1,2`, then E3+ with `run` → `assign` → short review. Watch for
   the cast shrinking (boot_mapping handles it) and for merges/swaps (tribe field in coverage names).
3. Re-export the corpus after every few episodes; run `finetune-ecapa … --epochs 1` once for the baseline EER
   and keep the number in `docs/CORPUS.md`.
4. Update the spec (`survivor_speaker_pipeline_spec.md`) with the M2/M3 results and the retired text prior.
5. Later milestones from the spec: `>>`-era seasons (S21–39: no names in captions → chyron OCR bootstrap; `survspk/chyron.py` only grabs
   frames/contact sheets so far + diarization), cross-season host bank (§8.9), a `scene` column, AU seasons (deferred).
6. Possible UI wishes not yet built: keyboard shortcut for per-line play, batch-accept high-confidence audio
   (`/api/bulk_accept` exists; not surfaced), showing the previous/next group's speaker on the coverage row.

## 7b. Added 2026-09-17 (session with a fresh Claude)

* `labels` carries `version_season, episode, start_s, end_s, text` (additive migration, backfilled on `init_db`).
  `segment` re-anchors human/chyron labels onto the new utterances by span (≥ 60 % of the new utterance inside the
  old label's span); it refuses only when a label would be lost, `--force` drops those. Orphaned label / queue /
  split rows of the old utt ids are deleted.
* Review UI *audit* mode + `audit_verdicts` table + `survspk audit-stats`: a stable stratified sample of 50
  auto-labelled runs per episode; verdicts become human labels; undo restores the auto label from `prev_label`.
  This is the unbiased precision number; `sdh_agreement_*` is not (explicit names ⇒ well-banked speakers).
* Subtitle coverage, measured on the Mac DB: only 310/686 episodes have ≥ 20 `NAME:` lines. S21–S39 have
  essentially none; S40+ is mixed (US47: E01–E07, E13 named; E08–E12, E14 plain `.en.srt` on the NAS). The
  bank+assign path is therefore the general case, not a fallback.
* `survspk chyron` (`survspk/chyron.py`, `tests/test_chyron.py`): frame sampling (ffmpeg rawvideo pipe, band diff),
  OCR backends (apple-vision via ocrmac — API checked against ocrmac 1.0.1 source; tesseract; fake for tests),
  cast matching, location cards -> `scenes` table, anchoring, `chyron` labels + `chyron_conflict` queue rows,
  `chyron_*` metrics. Bank: `chyron_run` label source. UI: chyron candidate on key `c`. Not yet run on real video:
  first run should be `chyron US47 2` and compare `sdh_agreement` with the hits table by eye.
* The project folder is not a git repository; `Archive.zip` was the only backup. See the session notes for
  `git init` + `sqlite3 .backup`.

## 7c. Review UI changes (2026-09-23)

* A banner above the clip gives the episode's next step: when bank coverage is done it offers **refit & reassign**;
  after a refit (remembered in localStorage with the count of your labels, so new labels bring the prompt back) it
  offers the 50-clip audit; once the audit is done it shows the precision and a button to the next episode, or the
  `survspk run` / `assign` commands when the next episode has no queue yet.
* The stale-embeddings warning is now a `re-embed` tag on the refit button. The header wraps instead of squeezing.
* Counts name their units: the episode menu counts open lines, the header shows `groups · lines`.
* Queue groups carry `maybe` ({speaker: best audio rank, 0 = captions only}), the coverage panel's rule for
  "could be theirs". Thin rows in the coverage panel have a **show these** link that filters the queue to that
  castaway's likely clips (best rank first); clips that could be a thin castaway are tagged `could be X (audio #k)`.
  Thin-first sorting uses `maybe` too. Clicking a coverage name still labels the selection.
* An **autoplay** switch in the header (on by default, remembered per browser).

## 7d. First audit: US47 E01 (2026-09-23, after re-embed + refit)

* 64 of 75 audited auto runs confirmed: 85% by run (Wilson 95% lower bound 76%), 94% by speaking time (495 of 527 s).
* Duration decides it. Runs under 2 s: 2 of 9 right (mostly "Yeah.", "Oh, my God."; 5 went to UNKNOWN, 1 to
  NOSPEECH). Runs of 2 s or more: 62 of 66 (94%, lower bound about 85%). The assign score does not separate them:
  the 0.9+ bucket was the worst (10 of 15) because short clips score high.
* The real voice confusions: Anika and Rachel (both Gata) three times, including an 18.6 s confessional; Genevieve
  taken for Rome once (5.1 s). Watch that pair in the E02 audit.
* Short runs are 42 of 267 auto runs in E01-02 but only 43 s of 2,186 s, and the bank already ignores utterances
  under `bank_min_duration_s` (1.5 s), so they do not pollute the bank. A minimum run length for auto labels
  (about 2 s) would lift run precision to about 94% at a cost of 2% of auto-labelled time.

## 7e. E02-E03 audits and the audit fix (2026-09-23)

* Audits: E01 64/75, E02 52/59, E03 69/84; pooled 185/218 = 84.9% (lower bound 79.5%). Runs of 2 s or more:
  164/180 = 91%; by speaking time 93%. 14 wrong-person errors among the 180 long runs; the rest of the rejections
  are short clips marked UNKNOWN / NOSPEECH. About 10 s per verdict.
* Bug fixed: `/api/audit` refilled the sample to 50 open groups after every verdict, so the audit never ended (Max
  judged 75, 59 and 84). It now serves `AUDIT_SAMPLE - verdicts so far`. `AUDIT_SAMPLE` is 20 (about 3 minutes);
  precision is read pooled over the season in `survspk audit-stats` (20 x 14 episodes is about 280 verdicts).
* E03 coverage after assign: every castaway at target, nothing thin. From E03 on the bank needs no touch-up labels
  for US47; the plan is `run` + `assign` for E04-E14 unattended, a 20-clip audit per episode, and a bank rebuild
  mid-season.

## 7f. Chyron OCR on real video: US47 E01-E03 (2026-09-23)

* First run (E02, 4 min): 50 hits, SDH agreement 55%. Two clear faults, both fixed in `chyron.py`: mixed-case lines
  on the left are the show's dialogue captions ("and I'm gonna say" -> Andy, "Kishan, you got that started?"),
  so a card line now needs >= 80% capitals (`min_upper`); the opening credits flash every name within ~36 s, so 4+
  castaways inside 40 s are dropped (`burst_n`, `burst_s`). SDH agreement after: E01 70%, E02 71%, E03 67%.
* Raw OCR is cached per episode (`survivor_audio/reports/chyron/<vs>_E<ep>_ocr.jsonl`, written on every video run);
  `survspk chyron VS EP --from-cache` replays it in seconds. Hits carry `t_end` (last frame showing the card).
* The reading is good: 16-18 castaways carded per episode, no false name reads left after the filters. The weak
  step is anchoring a card to a line. `scripts/chyron_anchor_eval.py` scores rules against known speakers: the
  current rule (line starting ~3 s before the card) is right 74% of the time (46/62); overlap-with-card 68%, the line
  spoken at card+0.5 s 72%.
* `scripts/chyron_bootstrap_exp.py` (throwaway DB copy; SDH names erased except the host's, all labels deleted)
  bootstraps from name cards alone: auto-label precision E01 92%, E02 73%, E03 63%; E03 held out 59%. A few wrong
  anchors poison whole voices (Tiyana's two cards both landed on Sue/Caroline lines, so every E03 'Tiyana' was Kyle),
  and castaways with no good card (Caroline, Sol) have no bank entry.
* With every card moved to a line its castaway speaks (`--oracle`, what a person confirming cards would give):
  E01 95%, E02 86%, E03 80% (clips >= 2 s: 95 / 87 / 84%); E03 held out 78% (83% on >= 2 s). 16 of 18 castaways
  banked. So: card reading is ready; anchoring needs a human click per card (~50 cards for E01-E03) or a voice-
  agreement rule across a castaway's cards, plus a top-up for castaways who never get a usable card.

## 7g. Card check (2026-09-23)

* Review UI mode *check name cards*: one card at a time with the frame after it appears (`/api/card_frame`, ffmpeg
  from the NAS, cached as JPEG under `survivor_audio/frames/cards/`), the OCR text, and the lines within
  [t-8, t+4] s (`chyron.check_before_s/after_s`). Keys 1-9 pick the line, 0 = none, space plays, z undoes.
* Answers live in `card_checks` (new table, additive). `apply_labels` uses a check for a hit of the same castaway
  within 2 s instead of the time rule; a checked line gets a `chyron` label at confidence 1.0 with
  `top_candidates.checked = true`; `none` writes nothing and a re-run does not bring the automatic label back.
* API: `GET /api/cards/{vs}/{ep}`, `POST /api/card_check`, `POST /api/card_uncheck`, `GET /api/card_frame/{vs}/{ep}?t=`.
  `/api/episodes` now also lists episodes that have cards but no queue, with `n_cards` / `n_checked`.
* Tested on the throwaway DB copy (US47 E02, 29 cards, ~5 s each by keyboard): 16 of 17 carded castaways got a
  confirmed line. `tests/test_chyron.py::test_card_check_api_and_rerun` covers the API and re-runs.

## 7h. Chyron frame-clock bug, fixed (2026-09-23)

* `sample_band` computed the band height (150 rows for a 1080p video, crop 0.72-1.0, width 960) while ffmpeg makes
  152, so every frame read was 2 rows short and the frame clock ran 1.3% fast: a card at 7:42 was dated 7:48, one at
  ~30:45 was dated 31:08 (the card check showed Sam's card for an 'Anika' hit, and frames without any card). It now
  decodes one frame and measures it. `tests/test_chyron.py::test_sample_band_keeps_time_on_a_real_video` fails on
  the old code (608 frames, last at 303.5 s for a 300 s clip). Every chyron number in 7f was measured on the
  skewed clock.
* Re-read US47 E01-E03: SDH agreement 83 / 85 / 100% (was 70 / 71 / 67%). Anchoring on known lines: start rule
  88%, overlap (line most on air while the card is up) 93%; `chyron.anchor: overlap` is now the default, with
  `chyron_hits.t_end_s` stored. Chyron labels themselves: 97 / 93 / 96% right.
* Bootstrap from name cards alone (no captions, no human labels): auto labels E01 95%, E02 86%, E03 79%; E03 held out
  80% (84% on clips >= 2 s). The card check and a thin-speaker top-up are what is left to close the gap to the
  caption-based pipeline (91% on >= 2 s).
* Card checks made before the fix keep matching (2 s tolerance) where the shift was small; the one E01 'none' answer at
  31:08 no longer matches any card and that card shows as unchecked.

## 7i. US47 E08 file, US45 first pass (2026-09-24)

* US47 E08's WEBDL file was damaged at 23:20 (every ffmpeg read stops with "File ended prematurely"); Max replaced it
  with an HDTV-1080p copy (same 3,837 s). `stage_extract` now raises when the audio comes out shorter than the video
  (`is_truncated`: more than max(5 s, 2%) short) and re-extracts a short file on the next run; `stage_separate` redoes
  a stem shorter than its source. After `inventory -s 47 --force` + run --force: E08 64% of body time auto, 91% caption
  agreement on confident runs.
* US45: E01-E02 reviewed, E03-E13 assigned unattended (58-78% of body time auto; caption agreement on confident runs
  82-97%). Audit E01-E02: 23/40 right; 8 of the 17 misses are sub-2 s clips marked unknown; clips >= 2 s 18/25. Kendra's
  confessionals labelled Kellie 3 times at 0.93.
* After the E02 refit and a re-assign of E03-E13: audit of unreviewed E07 21/23 (91%, lower bound 73%). Misses: short
  clips marked unknown, and lines where two people interleave inside one caption run and the run got one of them.
* Bug fixed: human UNKNOWN / NOSPEECH / OTHER labels were banked as speakers (a pooled grab-bag entry that the
  consistency filter could lose real lines to). `stage_bank.PSEUDO_SPEAKERS` are now left out of the bank.

## 7j. Assign: 2 s minimum, voice split; host voice borrowed across seasons (2026-09-24)

* `thresholds.auto_min_s: 2.0`: a group with under 2 s of speech gets no auto label and is not queued (audits: most
  misses were sub-2 s). On unreviewed US45 E03-13 / US47 E04-14 it costs under 1 point of auto share and leaves
  caption-name agreement unchanged (86.9 / 89.1%).
* `thresholds.voice_split` (`stage_assign._split_on_voice`): inside a confidently labelled group, a line of >= 1 s
  whose own voice scores another speaker at least `voice_split_margin` (0.08) above the group's speaker is cut out
  and scored alone. With `voice_split_confident: false` (default) caption-name agreement on confident runs goes
  86.9 -> 90.1% (US45) and 89.1 -> 91.0% (US47) for about 2.5 points less auto share; requiring the line to be
  confident on its own changed almost nothing.
* `bank.borrow_host` (default on, `stage_bank.borrow_host_entries`): a season whose labels give the host no entry
  (S21-39: no caption names, no name card for Probst) takes his entry per domain from the fullest other-season bank
  with the same variant and model; the payload records `borrowed_from`.

## 7k. Align: cue times from ASR anchors (2026-09-24)

US31 E01 played early and cut off line ends. The envelope fit (webrtcvad vs "cue is on") had applied -4.62 s with
+0.11% drift; ASR says the true lag was -0.4 s on average. The fit is weak on SDH files whose cues run back to back
(the cue signal is almost always 1). E03 looked fine (-0.12 s) but its -0.12% drift was also spurious (2.5 s by the
end). Under the old maps 86-90% of US31 E01-03 cue windows started more than the 0.4 s pad away from the words.

Subtitle timing here is also loose per cue: half the cues are off by more than 0.65 s, a tenth by 2 s. So align now
transcribes the whole episode with faster-whisper small.en (CPU int8, ~5 min/hour; cached at
`reports/align/<vs>_E<ep>_asr_small.en_<variant>.json`), matches runs of >= 3 words to cue words, and gives WhisperX
each cue's heard start/end directly (75% of US31 cues). The rest follow a rolling median of neighbouring lags
(`align_stats.knots`, which stage_segment also uses). Fewer than 30% of cue starts heard keeps the envelope fit.
Config `align.anchor: always|never`. `scripts/asr_anchor.py` prints lag per 2-minute stretch for any episode.
US45/US47 were aligned with the envelope fit and have not been re-checked.

Why US31 is harder than US45/47: its subtitles are 2015 broadcast roll-up captions (ALL CAPS, 3-6 word fragments
shown back to back, `>>` for a new speaker, no names, words misheard). Measured against ASR, US47 E01 cues sit within
about 0.2 s of the speech (IQR per 2-minute stretch); US31 cues scatter by 1-1.5 s. US31 audio is also stereo AAC with
no centre channel (45/47 are 5.1).

Re-segment after re-align. Utterance times come from segment, so an align alone changes nothing you hear in the review
UI. Human/chyron labels and card checks are now re-anchored by cue (`utterance_cues`), not by time span: a re-align
moves every time, and by span a label would land on the neighbouring line. Dry run on a copy of the DB, US31 E01-03:
all 72/21/22 protected labels carried, 5 card checks remapped. Clip boundaries against ASR words, E01: speech starting
more than 0.3 s after the clip start 85% -> 11%, clip cutting off more than 0.3 s before the speech ends 83% -> 22%.
US47 E01 under the same measure: 27% and 18%.

## 7l. Rolling bank, trusted labels only (2026-09-24)

`assign` now refits the bank before each episode from every earlier episode (`bank.rolling: true`; off when
`--bank-as-of` is given, or with `--no-rolling`), stored as as_of = episode - 1. The bank learns only from caption
names, human labels and name cards (`bank.use_auto_labels: false`; `survspk bank --use-auto` to override): an auto
label the bank got wrong would otherwise join that player's voice and make the next wrong call likelier. The review
app's refit follows the same rule. Recency weighting (half-life 3 episodes) already favours recent episodes.

Dry run on a DB copy, frozen bank as of E02 vs rolling, caption-name agreement on confident runs / auto share of body
time: US44 E03-13 0.931 / 69.6% -> 0.926 / 71.7%; US45 E03-13 0.905 / 64.0% -> 0.907 / 67.6%; US47 E08, E11, E13
0.956 / 70.3% -> 0.963 / 78.0%. Same accuracy, 2-8 points more of each episode labelled. No season had unbankable
players at E02, so the case it helps most (someone silent in E01-02) is not in these numbers.

## 7m. Two voices in one line: diarizer + bank split suggestions (2026-09-25)

Lines that hold two speakers with no caption marker had to be split by hand. `run` now has a `diarize` stage (after
separate): NVIDIA Nemotron 3 Diarization, frame level, via `scripts/diarize_nemotron.py` in a throwaway uv env
(transformers support is not released yet), 300 s windows with 30 s overlap, ~13 s per episode on the M3 Ultra,
written to `diar/<vs>/E<ep>.npz`. One pass over a whole episode is quadratic and stalls; do not do that.
`assign` then calls `split_detect.suggest_episode`: for body lines >= 2.5 s the diarizer gives the second voice's
seconds and the change point; if the second voice holds >= 0.8 s, the bank embeds both sides and must name two
different people (contrast >= 0.2). Hits go to `split_suggestions` and the queue as `two_voices`; the review app
shows "sounds like two voices: A, then B from 'word'" with split there / one voice.

Test set: the 24 lines split by hand (US43-47) and 400 human-labelled lines >= 2.5 s left whole.
| method | hand splits found (cut within 1 s) | whole lines flagged |
|---|---|---|
| bank only, every word gap, contrast >= 0.6 | 16/24 | 3.0% |
| Nemotron only, 2nd voice >= 1.0 s | 19/24 | 7.8% |
| Nemotron >= 0.8 s + bank names differ (>= 0.2), shipped | 18/24 | 6.0% |
With the production track (300 s windows over the whole episode) the suggestion rule finds 14/24 and flags 2.8%.

Automatic cuts (changed the same day at Max's request: over-cutting is fine if a one-speaker line ends up with one
speaker on both parts). Any line with no human or name-card label where the diarizer hears a second voice >= 0.5 s
with >= 0.5 s either side is cut at the nearest word boundary (survspk.splits.split_utterance, shared with the review
app), the episode is re-embedded and re-assigned (parts keep their run, so assign pools them again), and an
unlabelled part whose own voice does not lean >= 0.08 towards someone else takes its sibling's speaker
(`_fill_parts`). Lines with a human or name-card label only get a suggestion. Sandbox on US44 E12 / E13 (temp work
root, real audio): 38 / 74 lines cut; both parts same speaker 10 / 17, different speakers 7 / 19 (the examples read
like a host question run into an answer), one part left for review 13 / 19, neither labelled 8 / 19; labelled
seconds on the cut lines 174 -> 137 / 277 -> 304; 29 / 59 parts queued. Parts show "auto-split ↩" in the review app;
undoing restores the line and marks it dismissed so it is not cut again. Test set: the cut rule finds 18/24 hand splits.
Some "whole" lines flagged are probably two voices labelled as one. pyannote 3.1 is gated on the Hub (terms not
accepted for this token), so it was not tested. `survspk splits VS EP` backfills an episode.

## 7n. ASR anchors on the GPU (2026-09-25)

`align.anchor_backend: mlx` (default): the anchor transcript comes from mlx-whisper small.en on the Apple GPU, with
the same Silero VAD the CPU path uses (scripts/asr_mlx.py, run by `stage_align.asr_words_mlx` in a throwaway
`uv run --no-project --with mlx-whisper --with faster-whisper` env, so the project venv never changes). About 75 s per
hour of audio vs about 7 min on the CPU; falls back to faster-whisper on the CPU if the subprocess fails. Cached as
reports/align/<vs>_E<ep>_asr_mlx-whisper-small.en-mlx_<variant>.json. Check (scripts/asr_backend_eval.py and
reports/tmp/consensus.py): the WhisperX words in the DB are not a fair reference, because they were aligned inside windows
built from the CPU anchors (several "errors" sat exactly 0.4 s, one pad, from the CPU time). Against the median of
the other transcripts, cue starts off by > 1 s: GPU small.en + VAD 0.8 / 0.8 / 1.1% vs CPU 1.0 / 1.8 / 3.1% (US31 E01,
US44 E09, US46 E03). large-v3-turbo hears 3-5% more cue starts but places them ~50 ms early with more spread; not used.
Batched faster-whisper on the CPU (batch 8) is ~3x faster with the same quality, the fallback if MLX breaks.

## 7o. Review app quality-of-life (2026-09-26)

* Voice samples: ♪ next to every candidate, name in the grid and coverage row, or ⇧+number. `/api/voice/VS/EP/SPK`
  picks up to 3 lines of 2-9 s from the season that a person, a checked card or an explicit caption name gave them;
  lines the latest bank kept first, confessionals first, one per episode first.
* Stills: `/api/line_frame` (cached in frames/lines), 1-3 per selection; header toggle "frames".
* Diarizer strip under each line (`/api/diar`, channels renumbered by presence over the group: 0 blue = main voice,
  1 orange = second, 2 purple = other). Click it to split the line at the nearest word boundary.
* `p` / "▶ with the line before": plays from the start of the previous line.
* Refit returns `changes` (labels flipped between speakers, newly labelled, sent back); the page lists the flips,
  longest first, with ▶ and "new ✓" / "old" buttons that write human labels.
* Audit mode, after the sample is complete: bulk-confirm one castaway's auto labels above a confidence
  (`/api/bulk_confirm`, source becomes human, note bulk:confirm with the previous label kept; `/api/bulk_unconfirm`
  and z undo it). These feed the bank as trusted labels.
* Header pace: lines per active minute this session (gaps over 2 min not counted) and an estimate for what is left.
* Fixed: `DismissIn` ("one voice") lived inside create_app, where FastAPI cannot resolve a body model under
  `from __future__ import annotations`; body models now live at module level. `.banner` had `display:flex`, which
  beat the `hidden` attribute and left an empty bar; `[hidden]` now wins.

## 7p. Backups (2026-09-26)

`uv run survspk backup [--keep 5] [--copy-to DIR] [--no-export]` (survspk/backup.py): an online snapshot of the live DB
(read-only connection + SQLite backup API, safe while the review app or a run is writing) into
db/backups/survspk_<YYYYMMDD_HHMMSS>.sqlite, quick_check'ed, newest `keep` kept; then a text-free export of what a
person decided (human + name-card labels, card checks, audit verdicts, splits, decided split suggestions) as CSVs +
manifest.json in work_root/exports/labels/ (overwritten each run; ~1.6 MB for 8 seasons). No dialogue or caption
text: the `text` column and the copies inside audit prev_label / split originals are dropped, so the export can go
to a private repo or cloud drive. `--copy-to` also copies both somewhere else (an external drive, iCloud Drive).
Takes ~2 s. Run it at the end of every labelling session.

## 8. Gotchas (each cost time once)

* Never open `survspk.sqlite` from the Cowork Linux VM (`~/mnt/Stargazer/...`). SQLite locks and the WAL index do not
  cross that mount, so a VM connection thinks it is alone and can checkpoint and delete `-wal`/`-shm` under the
  running review app (2026-09-24: card checks returned 500 "file is not a database"; the `.fuse_hidden*` files in
  `db/` are those deleted index files). Read the DB on the Mac, or copy it first with `.backup`.

* FUSE mount on the Mac: cannot delete files, tar overwrite fails → write files with `cat >` / commit_files;
  SQLite journal on the mount must be TRUNCATE (vm profile) — the macos profile uses the local disk and WAL.
* `md5` is not on the Mac's device shell; use `md5sum`. `sqlite3` CLI is absent; use `python3 -c` with sqlite3.
* pandas: `lab.flags` is a DataFrame attribute — always `lab["flags"]`.
* FastAPI body models must be module-level (PEP 563 + closure-local classes → params treated as query).
* sqlite connections are per-thread; anything threaded (text prior) precomputes DB reads on the main thread.
* The review UI serves sync endpoints from a thread pool, so `aliases.Resolver` keeps one read-only connection per
  thread (`threading.local`, fixed 2026-09-23). Before that, `/api/queue`, `/api/coverage` and `/api/audit` failed
  with a 500 whenever a request landed on a thread other than the one that built the resolver.
* Re-segmenting shifts positional utt ids; anything cached by utt id (parquet, labels) is checked for freshness.
* `uv sync --extra X` alone removes the other extras. Always `--all-extras`.
* starlette 1.6 TestClient needs `httpx2` (in the dev group now).
* oMLX: port 8001, key in `.env`, `response_format json_schema`, `chat_template_kwargs.enable_thinking=false`.
* A refit on episode N builds the bank from episodes 1..N, so a line split in an *earlier* episode makes that
  episode's parquet stale too. `StaleEmbeddings` now carries `vs/ep/variant`; the refit endpoint and `survspk assign`
  re-embed the episode the error names and retry (fixed 2026-09-23; before, the refit re-embedded only episode N and
  failed with "E01.vocals.parquet is stale"). The UI's re-embed tag also looks at episodes 1..N.
* Assign re-runs must never overwrite human labels — `_write` keeps a protected set; tests cover it.
* The review UI's queue groups utterances by run; labelling part of a group shrinks it client-side without a
  reload, so utt ids in `groups[i]` are the truth for that group until the next `loadQueue()`.

## 9. File map (what to read for what)

| area | files |
|---|---|
| CLI entry points | `survspk/cli.py` (typer): `info inventory refresh-survivor ingest-subs names-report report grab-frame missing-subs fetch-subs extract separate align sync-check segment embed ablation bank assign text-prior review export-corpus finetune-ecapa run-errors run` |
| config | `config/default.yaml`, `survspk/config.py` (profiles macos/vm, paths, thresholds, bank, text_prior) |
| DB | `survspk/db.py` (schema + migrations: utterances, cues, words, labels, review_queue, speaker_bank, utt_splits, metrics) |
| segmentation & names | `survspk/stage_segment.py`, `survspk/aliases.py` (mentions, bios), `survspk/ids.py`, `survspk/subparse.py` |
| embeddings | `survspk/stage_embed.py` (Encoder, parquet freshness, load_labeled, LOO scores, ablation) |
| bank / assign | `survspk/stage_bank.py`, `survspk/stage_assign.py` |
| review UI | `survspk/review_app.py` (FastAPI), `survspk/static/review.html` (single page, no build step) |
| corpus / training | `survspk/export_corpus.py`, `survspk/finetune_ecapa.py`, `scripts/finetune_pyannote_seg.py`, `docs/CORPUS.md` |
| retired | `survspk/text_prior.py` |
| tests (65) | `tests/test_bank_assign.py` (synthetic season fixture used by most suites), `test_review_app.py`, `test_export_corpus.py`, `test_load_labeled.py`, `test_mentions.py`, `test_text_prior.py`, `test_segment.py`, `test_align.py`, … |
| reports | `docs/M0_REPORT.md`, `docs/M1_REPORT.md` (§4b error analysis), `README.md` (setup, milestones, results) |
