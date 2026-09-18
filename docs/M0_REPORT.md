# M0 report — data, inventory, parser (2026-09-07)

All numbers come from `survivor_audio/db/survspk.sqlite` after `survspk refresh-survivor`, `inventory`,
`ingest-subs`, `names-report`. CSVs in `survivor_audio/reports/`. Everything ran inside the Cowork VM
(no audio touched; ffprobe reads headers only).

## survivoR

Refreshed from upstream GitHub (`refresh-survivor`): castaways 1,441 · castaway_details 1,200 ·
boot_mapping 15,864 · episodes 1,233 · confessionals 14,055. **US50 now has boot_mapping for all 13
episodes** (the Gamebot snapshot from 2025-11 had none). The Gamebot loader adapted cleanly; no
Airflow/Postgres needed.

## Inventory (US S1–S50)

| | |
|---|---|
| video files | 725 (39 reunions, skipped) → **686 body episodes** |
| containers | 553 mkv · 122 avi · 50 mp4 |
| audio | 491 stereo · **195 six-channel (5.1)** |
| subtitle chosen | 498 sidecar SDH · 172 sidecar plain · 7 embedded SDH · 8 embedded plain · **1 none (US50E07)** |
| subtitle timing ok | 656 ok · **29 suspect** · 1 n/a |
| double-episode files | 4 named `EyyEzz`/`Eyy-zz` (US30E04-05, US31E10-11, US36E01-02, US42E06-07) + US19E12 (81 min, mis-named) |

Embedded subtitle streams were demuxed only when no sidecar had acceptable timing (20 streams); that
rescued 15 episodes including US19E04, US21E06, US38E06, US45E01, US45E02 whose sidecars were truncated.

### The 29 timing-suspect episodes (chosen file still bad) — `reports/inventory_subtitle_timing_bad.csv`

* **Truncated (5) + missing (1) — RESOLVED 2026-09-07** via `survspk fetch-subs` (Plex subtitle agent,
  OpenSubtitles): US10E03, US17E06, US36E09, US49E09, US50E04, US50E07 now have full-length SDH sidecars
  (898–1,617 cues, all ending within 25 s of the video). **Every body episode in the library now has a
  usable subtitle.**
* **Frame-rate drift, subtitle runs 16–72 s past file end (17):** US11 E01/02/05/08/13, US12 E06/08/09/13,
  US13 E04/07, US14 E01, US16 E05, US48 E06/08/09 (SDTV/HDTV rips). Usable; the M1 alignment stage
  measures and corrects drift/offset. Verify on one of these in M1.
* **Different cut (7):** finale/reunion split differs between video and subtitle: US10E14 (+45 min),
  US12E15 (+4), US13E15 (+39), US14E11 (+3.5), US35E14 (+12), US46E13 (−29, file includes aftershow),
  US06E01 (−7, premiere). US19E12 is a double episode in one file with a single-episode subtitle.

### survivoR `episode_length` as a check

Useful only for gross errors. It mixes broadcast minutes with ads (e.g. 120 for two-hour premieres,
finale+reunion totals) and runtime without ads (e.g. 86 for US47E01), so −7 to −50 min deltas on
premieres/finales are normal. It did flag every double-episode file and the mis-named US19E12.

## Subtitles ingested

* **870,366 cues** across 685 episodes; 1,432 candidate subtitle files evaluated.
* Cue duration (US02/20/33/47 sample, n=69k): p10 0.9 s, median 1.8 s, p90 3.4 s; 13% under 1 s
  (the `>>` live-caption era is choppier than the new era). Utterance merging (spec §7.4) is essential.
* **Two speakers in one cue:** 33,402 cues (3.8%) library-wide; 10–13% in S40–S50 (dash turns) and
  also 2–3k per season in S3–S4 where `>>` appears mid-cue. Splitting (spec §7.4) applies to both.
* Convention flags per season are in `episodes.subtitle_convention`; they match the v0.2 §2.3 table.
  310 of 686 episodes have ≥20 `NAME:` lines.

## SDH speaker names (`names-report`)

**41,318 named lines. 98.2% resolve automatically:** 22,357 to a cast `castaway_id`, 18,187 to Jeff,
31 to non-cast `OTHER`. 338 lines are generic labels (MAN, WOMAN, ALL…). **140 lines are first-name
clashes** (ROB / JENNA in All-Stars) that resolve per episode from `boot_mapping` at the utterance stage.
**265 lines (0.6%) remain unresolved** — loved ones at family visits, medics, and caption artifacts;
they will be labelled OTHER in review.

Aliases needed so far (in `config/aliases.yaml`): SUSAN→Sue (US01), RICH (US01), MATT→Matthew (US06),
RYNO→Ryan O. and LILL→Lillian (US07), ROB M→Boston Rob (US08), JF→Jeff (US19), TED→announcer (US40),
WILL→other (US44). survivoR's `castaway` short name matched everything else, including B.B., TK, Teeny.

Named lines by era: S1–2 and S5–9 ≈1,000–1,900 per season (cast named); S3–4 and S10–20 ≈500–1,700
per season but **PROBST only**; S21–S39 ≈0; S40–S50 ≈800–3,900 per season (cast named).

## Chyron

Frame sheets in `survivor_audio/frames/`. Name chyron confirmed bottom-left, inside the bottom 25% band,
in S2 (`COLBY  Auto Customizer / OGAKOR TRIBE`), S25 (`MALCOLM  Bartender / MATSING TRIBE`) and S47
(`TEENY  FREELANCE WRITER / LAVO TRIBE`). Format is NAME / occupation / tribe in every era checked (no age).
Two facts for M2:

1. In classic seasons the chyron appears at the player's **first appearance of the episode**, confessional
   or not (Colby's shows over a camp shot). Chyron ≠ confessional; it is a speaker label only.
2. The same band carries **location cards** (`MATSING TRIBE / DAY 11`, `VILLAINS / NIGHT 3`) and the show's
   own **burned-in captions** for whispered speech (centered, white). OCR output must be fuzzy-matched to
   cast names; location cards are a useful by-product for scene segmentation.

## Environment notes

* Cowork VM shell is Linux aarch64 (4 cores, 3 GB): fine for M0; audio/ML stages run on macOS.
* The mounted folders are FUSE without delete permission: SQLite must use `journal_mode=TRUNCATE` there
  (`sqlite_journal` per profile in `config/default.yaml`); WAL on macOS.
* Background processes do not survive a VM shell call; long jobs run in ≤170 s chunks (all stages are
  resumable per episode).

## Deliverables checklist (spec §11 M0)

- [x] `episodes`, `subtitle_files`, `cues` rows for the whole library
- [x] duration-mismatch list (`inventory_length_mismatch.csv`, `inventory_subtitle_timing_bad.csv`)
- [x] unresolved-alias list (`names_unresolved.csv`), aliases applied
- [x] chyron crops per era (`frames/`)
- [x] tests: 14 passing (`uv run pytest`)

## Carry into M1

* Pick US47 and US20 as first seasons (both fully SDH sidecar, timing ok, US47 is 5.1).
* Alignment must (a) estimate global offset and linear drift per episode, (b) time the sub-cue splits.
* Handle double-episode files: split at the second episode's recap using survivoR boundaries (M1 or M4).
* ~~Re-download subtitles for the 5 truncated episodes + US50E07~~ done via Plex.
* `device_commit_files` from the Cowork side occasionally lands a stale copy; verify by checksum, or write
  through the VM shell (base64) for small files.
