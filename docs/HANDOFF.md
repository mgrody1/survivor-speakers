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

## 8. Gotchas (each cost time once)

* FUSE mount on the Mac: cannot delete files, tar overwrite fails → write files with `cat >` / commit_files;
  SQLite journal on the mount must be TRUNCATE (vm profile) — the macos profile uses the local disk and WAL.
* `md5` is not on the Mac's device shell; use `md5sum`. `sqlite3` CLI is absent; use `python3 -c` with sqlite3.
* pandas: `lab.flags` is a DataFrame attribute — always `lab["flags"]`.
* FastAPI body models must be module-level (PEP 563 + closure-local classes → params treated as query).
* sqlite connections are per-thread; anything threaded (text prior) precomputes DB reads on the main thread.
* Re-segmenting shifts positional utt ids; anything cached by utt id (parquet, labels) is checked for freshness.
* `uv sync --extra X` alone removes the other extras. Always `--all-extras`.
* starlette 1.6 TestClient needs `httpx2` (in the dev group now).
* oMLX: port 8001, key in `.env`, `response_format json_schema`, `chat_template_kwargs.enable_thinking=false`.
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
