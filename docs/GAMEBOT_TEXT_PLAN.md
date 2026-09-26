# Text data for Gamebot and the boot model: where it stands and what it needs

Written 2026-09-23. The question: can the subtitles and speaker labels here feed Gamebot (the warehouse behind
preferencespace.com's Survivor pages), improve the Who Goes Home boot model, and power a "Who said that?" trivia
question? Short answer: yes, in three stages, and the first stage needs no speaker attribution at all.

## 1. What exists (Mac DB, `survivor_audio/db/survspk.sqlite`)

* **Subtitles for 686 US episodes** (876,252 cues, S1 to S50; missing S29E15 and S30E15 reunions and S50E04/E07/E08).
  Each cue has start/end times and parsed lines with `sdh_name`, `is_turn`, `is_italic`, `is_sound`.
* **Speaker names in the subtitles depend on the era** (named lines per season, measured today):
  * S1-2, S5-9, S40-50 name the cast: roughly 500 to 4,000 named lines a season.
  * S3-4, S10-20 name mostly Probst (`>> PROBST:`); the cast is unnamed.
  * S21-39 name nobody (speaker changes only).
* **Italics** (off-camera voice-over, almost always confessional) are present in S1-2, S5-9, S40-42 and S46-50.
* **Speaker labels** exist only for US47 E01-E02 (sdh 554, auto 826, human 28). `segment` (recap end, runs,
  confessional hint) has run on those two episodes only.
* Audio: raw for US11, US20, US47; vocals and embeddings for US20, US47. About 4.5 minutes of compute per episode.
* **Data quality note:** 5,741 consecutive duplicate cues in US08, US42 and US45 alone (the same lines repeated
  cue after cue). Any count built on cues must drop consecutive duplicates first.

## 2. What was tried on 2026-09-23 (numbers are forward tests on US seasons 31-50, 292 councils)

Name mentions per castaway and episode, from the subtitle text with a crude first-name match (files in
`survivor_audio/reports/name_mentions*.csv`, scripts beside them):

| model | top pick right | log loss |
|---|---|---|
| blind guess | 16.8% | 1.836 |
| live boot model (gameplay stats only) | 22.8% | 1.817 |
| + mentions in the previous episode | 24.3% | 1.826 |
| mentions in the same episode before "time to vote", alone | 26.9% | 1.810 |
| gameplay stats + those pre-vote mentions | 28.1% | 1.803 |

The previous episode's text adds nothing; the current episode before the vote carries as much signal as every
gameplay stat combined, even with a crude name match and a tribal-start cut that only found 58% of councils.
That is a "tribal is tonight" product (update the odds as the episode plays), not next-week odds.

"Who said that?" prototype, from lines with an explicit subtitle name and the unnamed cues that follow them:
891 / 1,384 / 2,187 named utterances in US08 / US42 / US45, of which about a fifth are quotable (10 to 25
words, no cast name that gives the answer away). Some are wrong: the name carries across an unmarked speaker
change (the M1 report found the same), and field chatter mixes in. Quotes need the confessional filter and a
purity check before they are trivia.

## 3. The plan

### Stage 1: a text layer in Gamebot, no speaker IDs needed (days)
1. **bronze.subtitle_cues** from `survspk.sqlite` (read-only): episode, idx, start/end, text, `sdh_name`, italic,
   turn, sound. Drop consecutive duplicates. Local only (see section 5).
2. **silver.episode_segments**: recap end, each tribal council's start (Probst's lines: "time to vote", "go get
   your torches", "I'll go tally the votes", ...), the vote reading, the preview. Check the count of tribals
   against survivoR's councils per episode; target 95% of councils found (today's cut: 58%).
3. **silver.mentions**: castaway x episode x segment (before tribal, at tribal, after), resolved through
   `survspk/aliases.py` (nicknames, the Probst rules) instead of a first-name regex, plus a target-talk flag
   (vote, blindside, idol, send home...). Tests against a hand-checked episode.
4. **Boot model, "tonight" mode**: the pre-tribal features for the council being played, with a leakage test
   that fails if any cue after the tribal start reaches a feature. Evaluate forward with confidence intervals;
   292 councils give about +-5 points on top-1, so small gains are noise.

### Stage 2: speakers where the subtitles name them (weeks, mostly compute, light review)
1. `extract -> separate -> align -> segment -> embed` for the named eras (S1-2, S5-9, S40-50; about 230
   episodes, about 17 hours of compute).
2. Run purity (M1 report section 4b): split runs whose utterance embeddings form two confident clusters, before
   a name is trusted.
3. **silver.utterances** (speaker, confessional flag, confidence, provenance) and **silver.target_talk**
   (speaker -> castaway named, in confessional, with vote words). That gives "named as a target by N people"
   and "their own alliance is discussing them", the features most likely to move the model.
4. **"Who said that?"**: explicit subtitle name, confessional (italic or a long run), single-speaker after the
   purity check, 10 to 25 words, no name that gives it away; wrong choices from the same tribe at the time; an
   audit sample per season before a season's quotes go live.

### Stage 3: every season (the survspk roadmap)
S3-4 and S10-39 have no cast names in the subtitles: chyron OCR (`survspk chyron`, not yet run on real video),
per-season voice banks, `assign`, and review loops. Gate each season on an audit-mode precision number (for
example 95% on the 50-run audit sample) before its labels feed Gamebot. This is where most of Max's review time
would go.

## 4. Gamebot shape
* bronze: `subtitle_cues`, `survspk_labels` (as exported), `scenes` (from chyron location cards, later)
* silver: `episode_segments`, `mentions`, `utterances`, `target_talk`, `quotes`
* gold: per castaway-council text features joined to the existing boot features, under the same information rule
* Checks per load: coverage by season, label provenance mix, audit precision, tribal counts vs survivoR,
  and the leakage test.

## 5. Publishing rule
Subtitle text is the show's dialogue. Keep the text tables (bronze cues, silver utterances and quotes) local and
out of the site's Parquet export. Publish only aggregates (counts, shares, model outputs) and single short
attributed quotes as trivia answers.

## 6. Decisions for Max
1. Whether short attributed quotes as trivia are acceptable to you (fan use, one line at a time).
2. How much review time Stage 3 can have; Stages 1 and 2 need little.
3. Whether "tribal is tonight" is a product for the episode page, since it needs the episode to have aired.

## 7. Stage 1 status (2026-09-23, later the same day)

Built: `Gamebot/scripts/subtitle_mine.py` -> `Gamebot/data_cache/subtitles/subtitles.sqlite` (cues, recaps, tribals,
mentions, quotes) and `subtitle_mentions.csv`.

* Cues: 856,620 kept, 19,632 consecutive repeats dropped.
* Votes: 678 of 710 found by "go tally the votes"; 94% confirmed by the boot's name in the reading. The cut is
  "time to vote" (80%) or 60 s before the tally (the median gap). S21-39 captions are all capitals, so names match
  without case, and a name that is also a word (Will, Chase, Hope) needs punctuation after it in those captions.
* "Tonight" odds, 598 votes in US seasons 10-50, each season scored by a model trained on the seasons before it:

  | model | top pick right | log loss |
  |---|---|---|
  | blind guess | 16.7% | 1.848 |
  | live model (gameplay stats) | 25.3% | 1.818 |
  | the episode's text before the vote, alone | 32.2% | 1.673 |
  | both | 34.0% | 1.659 |

  The gain over the live model is +8.7 points (95% interval +4.8 to +12.6) and holds in every era (S10-20 34.8%,
  S21-39 33.1%, S40-50 34.9%). Features: each castaway's share of the name mentions before the vote, their share
  of mentions in lines about voting, idols or alliances, and mentions per 100 lines. Wired into the episode page the same day (see below).
* Episode page: the "Who goes home" tab opens on "This episode's vote" with a With the episode / Before the
  episode switch and a line saying whether the top pick went home. `who-goes-home/live.py` writes each council's
  `tonight` rows (tempered at 0.85, trained on earlier US seasons) when `subtitle_mentions.csv` exists; without it
  the page shows the gameplay odds alone. In the shipped files the episode's top pick went home 33.8% of the time
  over 568 votes, against 25.4% for the gameplay model on the same votes.
* Quotes: 20,006 attributed utterances; 180 kept as "Who said that?" questions in the quiz bank.

