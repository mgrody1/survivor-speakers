"""Derived speech tables for survivoR: who speaks, how much, who names whom, who answers whom.

`survspk export-stats [--season US47 ...] [--out DIR]` writes, per processed episode (body only: the "previously on"
recap and the scenes-from-next-week preview are left out of every count except `recap_seconds`):

  speech_stats.csv         castaway x episode   seconds, lines, turns, words, confessional vs field, mentions, host
                                                addresses, where the labels came from, expected precision
  speech_mentions.csv      speaker -> named     lines in which one speaker names another (and direct addresses)
  speech_interactions.csv  speaker -> next      turn transitions in conversation (not confessionals)
  speech_quality.csv       episode              coverage, label-source mix, audits, open queue, alignment
  data_dictionary.csv      every column: table, type, unit, meaning, what NA means
  manifest.json            export and pipeline versions, models, row counts, file checksums

Numbers and public castaway names only: dialogue text is read to find names and never written.

Every speaker label counts (person, name card, caption name, automatic voice match). An automatic label carries the
calibrator's chance of being right (`labels.p_right`); `expected_wrong_seconds` adds up (1 - p) over them, so every row
says how much of its speech is probably someone else's. Trusted labels count as right.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EXPORT_VERSION = "1.0.0"
SOURCE_NAMES = {"human": "human", "chyron": "namecard", "sdh": "caption", "auto": "auto"}
NOT_SPEAKERS = {"UNKNOWN", "NOSPEECH"}
UNKNOWN = "UNKNOWN"
TURN_GAP_S = 5.0            # a reply starts within this many seconds of the previous speaker's last line
LEAD = r"(?:(?:so|and|okay|ok|well|now|hey|all right|alright|but|yes|no|yeah|look|listen|come on|thank you|thanks)[,!]?\s+)?"


@dataclass
class EpisodeCtx:
    """Who can be named in an episode: castaway_id -> short name, name alternatives, who is still in the game."""
    cast: dict[str, str]
    alts: dict[str, list[str]]              # castaway_id (and host_id) -> lowercase names, longest first
    present: set[str]
    host_id: str
    _pats: dict = field(default_factory=dict, repr=False)

    def patterns(self) -> dict[str, tuple[re.Pattern, re.Pattern]]:
        """castaway_id -> (any mention, direct address). Direct address = the name said to someone: "Jeff, ...",
        "..., Jeff?" or the name alone."""
        if not self._pats:
            for cid, names in self.alts.items():
                a = "|".join(re.escape(n) for n in names)
                if not a:
                    continue
                any_ = re.compile(rf"(?<![A-Za-z'])(?:{a})(?![A-Za-z])", re.I)
                addr = re.compile(rf"^\W*{LEAD}(?:{a})\s*,|,\s*(?:{a})\s*[.?!]*[\"'”’)]*\s*$|^\W*(?:{a})\s*[.?!]+\W*$",
                                  re.I)
                self._pats[cid] = (any_, addr)
        return self._pats


def episode_ctx(resolver, vs: str, ep: int) -> EpisodeCtx:
    names = resolver.mention_names(vs)
    short = {r["castaway_id"]: (r["castaway"] or (r["full_name"] or "?").split()[0]) for r in resolver.cast(vs)}
    _, host_id = resolver.host_names(vs)
    return EpisodeCtx(cast=short, alts=names, present=set(resolver.present(vs, ep)), host_id=host_id)


def named_in(text: str, ctx: EpisodeCtx) -> tuple[set[str], set[str], int]:
    """(ids named, ids addressed, ambiguous names skipped). A name two castaways share goes to the one still in the
    game; if both are (or neither), it is skipped."""
    by_span: dict[tuple[int, int], set[str]] = defaultdict(set)
    addressed = set()
    for cid, (any_, addr) in ctx.patterns().items():
        hit = False
        for m in any_.finditer(text or ""):
            by_span[m.span()].add(cid)
            hit = True
        if hit and addr.search(text or ""):
            addressed.add(cid)
    named, amb = set(), 0
    for ids in by_span.values():
        if len(ids) > 1:
            live = {i for i in ids if i in ctx.present or i == ctx.host_id}
            ids = live if len(live) == 1 else set()
            amb += not ids
        named |= ids
    return named, addressed & named, amb


def load_episode(con: sqlite3.Connection, vs: str, ep: int) -> list[dict]:
    rows = con.execute("""
        SELECT u.utt_id, u.idx, u.start_s, u.end_s, u.segment, u.domain_hint, u.flags, u.text, u.align_ok,
               COALESCE(u.n_words, 0) AS n_words, l.speaker_id, l.source, l.p_right, l.confidence
        FROM utterances u LEFT JOIN labels l USING (utt_id)
        WHERE u.version_season=? AND u.episode=? ORDER BY u.start_s, u.idx""", (vs, ep)).fetchall()
    out = []
    for r in rows:
        d = dict(zip(("utt_id", "idx", "start", "end", "segment", "domain", "flags", "text", "align_ok", "n_words",
                      "speaker", "source", "p_right", "confidence"), r))
        if d["speaker"] in NOT_SPEAKERS:
            d["speaker"], d["source"] = None, None
        try:
            d["run_id"] = (json.loads(d["flags"] or "{}") or {}).get("run_id")
        except ValueError:
            d["run_id"] = None
        d["dur"] = max(0.0, (d["end"] or 0) - (d["start"] or 0))
        if d["source"] == "auto" and d["p_right"] is None:
            d["p_right"] = d["confidence"]           # labels from before the calibrator: its score is the best we have
        out.append(d)
    return out


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def episode_tables(utts: list[dict], ctx: EpisodeCtx, vs: str, ep: int, extra: dict | None = None) -> dict[str, list[dict]]:
    """speech_stats / speech_mentions / speech_interactions / speech_quality rows for one episode."""
    extra = extra or {}
    season = int(re.sub(r"\D", "", vs) or 0)
    key = {"version_season": vs, "season": season, "episode": ep}
    body = [u for u in utts if u["segment"] == "body"]
    host = ctx.host_id
    st: dict[str, Counter] = defaultdict(Counter)
    conf_runs: dict[str, set] = defaultdict(set)
    namers: dict[str, set] = defaultdict(set)
    edges: dict[tuple[str, str], Counter] = defaultdict(Counter)
    trans: Counter = Counter()
    q = Counter()

    for u in utts:
        if u["segment"] == "recap" and u["speaker"]:
            st[u["speaker"]]["recap_seconds"] += u["dur"]
        q[f"{u['segment'] or 'body'}_seconds"] += u["dur"]

    prev = None          # last body line: (speaker or None, end, domain)
    for u in body:
        spk, dur, conf = u["speaker"], u["dur"], u["domain"] == "confessional"
        q["body_lines"] += 1
        q["aligned_lines"] += bool(u["align_ok"])
        if spk:
            s = st[spk]
            s["speech_seconds"] += dur
            s["lines"] += 1
            s["words"] += u["n_words"]
            s["confessional_seconds" if conf else "field_seconds"] += dur
            s["confessional_lines"] += conf
            if conf and u["run_id"] is not None:
                conf_runs[spk].add(u["run_id"])
            src = SOURCE_NAMES.get(u["source"], "auto")
            s[f"seconds_{src}"] += dur
            q[f"seconds_{src}"] += dur
            if src == "auto":
                p = u["p_right"] if u["p_right"] is not None else 0.5
                s["expected_wrong_seconds"] += (1 - p) * dur
                q["auto_p_seconds"] += p * dur
            q["host_seconds" if spk == host else "castaway_seconds"] += dur
        else:
            q["unknown_seconds"] += dur
            q["unknown_lines"] += 1
        # turns: a new turn whenever the speaker changes (an unknown line ends a turn)
        new_turn = prev is None or spk is None or prev[0] != spk
        if spk and new_turn:
            st[spk]["turns"] += 1
        # conversation: one known speaker then another, both in the field, within TURN_GAP_S
        if prev is not None and new_turn and not conf and prev[2] != "confessional" and u["start"] - prev[1] <= TURN_GAP_S:
            if spk and prev[0]:
                trans[(prev[0], spk)] += 1
                q["transitions"] += 1
            elif spk or prev[0]:
                q["transitions_unknown"] += 1
        prev = (spk, u["end"], u["domain"])
        # names
        named, addressed, amb = named_in(u["text"] or "", ctx)
        q["ambiguous_names"] += amb
        src_id = spk or UNKNOWN
        for t in named:
            if t == spk:
                continue
            e = edges[(src_id, t)]
            e["lines_naming"] += 1
            e["direct_address"] += t in addressed
            e["in_confessional"] += conf
            q["mentions"] += 1
            q["mentions_unknown_speaker"] += spk is None
            st[t]["mentions_received"] += 1
            st[t]["mentions_received_confessional"] += conf
            st[t]["direct_addresses_received"] += t in addressed
            if spk == host:
                st[t]["host_mentions_received"] += 1
                st[t]["host_addresses_received"] += t in addressed
            if spk:
                st[spk]["mentions_given"] += 1
                namers[t].add(spk)

    # --- speech_stats: every castaway in the game this episode, plus anyone else who spoke or was named
    ids = (set(ctx.present) | {k for k, v in st.items() if v["speech_seconds"] or v["mentions_received"]}) - {host}
    ids = {i for i in ids if i in ctx.cast}
    cast_secs = sum(st[i]["speech_seconds"] for i in ids)
    sv = extra.get("survivor_confessionals", {})
    stats = []
    for cid in sorted(ids):
        s = st[cid]
        secs = s["speech_seconds"]
        stats.append(key | {
            "castaway_id": cid, "castaway": ctx.cast.get(cid), "in_game": int(cid in ctx.present),
            "speech_seconds": _r(secs, 1), "speech_share": _r(secs / cast_secs, 4) if cast_secs else None,
            "lines": s["lines"], "turns": s["turns"], "words": s["words"],
            "words_per_minute": _r(60 * s["words"] / secs, 1) if secs >= 10 else None,
            "confessional_seconds": _r(s["confessional_seconds"], 1), "confessional_lines": s["confessional_lines"],
            "confessional_runs": len(conf_runs[cid]), "field_seconds": _r(s["field_seconds"], 1),
            "recap_seconds": _r(s["recap_seconds"], 1),
            "mentions_given": s["mentions_given"], "mentions_received": s["mentions_received"],
            "mentions_received_confessional": s["mentions_received_confessional"], "namers": len(namers[cid]),
            "direct_addresses_received": s["direct_addresses_received"],
            "host_mentions_received": s["host_mentions_received"], "host_addresses_received": s["host_addresses_received"],
            "seconds_human": _r(s["seconds_human"], 1), "seconds_namecard": _r(s["seconds_namecard"], 1),
            "seconds_caption": _r(s["seconds_caption"], 1), "seconds_auto": _r(s["seconds_auto"], 1),
            "expected_wrong_seconds": _r(s["expected_wrong_seconds"], 1),
            "est_precision": _r(1 - s["expected_wrong_seconds"] / secs, 3) if secs else None,
            "survivor_confessional_count": sv.get(cid, (None, None))[0],
            "survivor_confessional_seconds": sv.get(cid, (None, None))[1],
        })

    mentions = [key | {"source_id": a, "target_id": b, "lines_naming": c["lines_naming"],
                       "direct_address": c["direct_address"], "in_confessional": c["in_confessional"]}
                for (a, b), c in sorted(edges.items())]
    out_n = Counter()
    for (a, _), n in trans.items():
        out_n[a] += n
    inter = [key | {"from_id": a, "to_id": b, "transitions": n, "share_of_from": _r(n / out_n[a], 3)}
             for (a, b), n in sorted(trans.items())]

    labelled = q["castaway_seconds"] + q["host_seconds"]
    body_s = labelled + q["unknown_seconds"]
    aud = extra.get("audit", (None, None))
    quality = [key | {
        "body_seconds": _r(body_s, 1), "body_lines": q["body_lines"], "castaway_seconds": _r(q["castaway_seconds"], 1),
        "host_seconds": _r(q["host_seconds"], 1), "unknown_seconds": _r(q["unknown_seconds"], 1),
        "unknown_lines": q["unknown_lines"], "speaker_coverage": _r(labelled / body_s, 4) if body_s else None,
        **{f"share_{k}": (_r(q[f"seconds_{k}"] / labelled, 4) if labelled else None)
           for k in ("human", "namecard", "caption", "auto")},
        "auto_expected_precision": _r(q["auto_p_seconds"] / q["seconds_auto"], 4) if q["seconds_auto"] else None,
        "audit_n": aud[0], "audit_right": aud[1],
        "queue_open_lines": extra.get("queue_open"), "aligned_share": _r(q["aligned_lines"] / q["body_lines"], 4) if q["body_lines"] else None,
        "recap_seconds": _r(q["recap_seconds"], 1), "preview_seconds": _r(q["preview_seconds"], 1),
        "castaways_in_game": len(ctx.present - {host}), "mentions": q["mentions"],
        "mentions_unknown_speaker": q["mentions_unknown_speaker"], "ambiguous_names": q["ambiguous_names"],
        "transitions": q["transitions"], "transitions_unknown": q["transitions_unknown"],
        "confessional_rank_agreement": extra.get("conf_spearman"),
    }]
    return {"speech_stats": stats, "speech_mentions": mentions, "speech_interactions": inter, "speech_quality": quality}


def episode_extra(con: sqlite3.Connection, sv: sqlite3.Connection | None, vs: str, ep: int) -> dict:
    ex: dict = {}
    groups = con.execute("""SELECT verdict, COALESCE(MAX(sample), 'random') FROM audit_verdicts
                            WHERE version_season=? AND episode=? GROUP BY group_key""", (vs, ep)).fetchall()
    rnd = [v for v, smp in groups if smp == "random"]           # as the review app: the random sample alone
    ex["audit"] = (len(rnd), sum(v == "confirm" for v in rnd)) if rnd else (None, None)
    ex["queue_open"] = con.execute("""SELECT COUNT(*) FROM review_queue q JOIN utterances u USING (utt_id)
                                      WHERE u.version_season=? AND u.episode=? AND q.resolved=0 AND u.segment='body'""",
                                   (vs, ep)).fetchone()[0]
    m = con.execute("SELECT value FROM metrics WHERE version_season=? AND episode=? AND key='confessional_spearman_count'",
                    (vs, ep)).fetchone()
    ex["conf_spearman"] = _r(m[0], 3) if m and m[0] is not None else None
    if sv is not None:
        ex["survivor_confessionals"] = {
            cid: (n, t) for cid, n, t in sv.execute(
                "SELECT castaway_id, confessional_count, confessional_time FROM confessionals WHERE version_season=? AND episode=?",
                (vs, ep))}
    return ex


# ---------------------------------------------------------------- data dictionary
KEYS = [("version_season", "text", "", "survivoR version_season, e.g. US47", "never NA"),
        ("season", "int", "", "season number", "never NA"),
        ("episode", "int", "", "episode number (survivoR)", "never NA")]
DICTIONARY: dict[str, list[tuple[str, str, str, str, str]]] = {
    "speech_stats": KEYS + [
        ("castaway_id", "text", "", "survivoR castaway_id", "never NA"),
        ("castaway", "text", "", "survivoR short name", "never NA"),
        ("in_game", "int", "0/1", "1 if survivoR boot_mapping lists them in this episode", "never NA"),
        ("speech_seconds", "real", "s", "time on lines attributed to them, episode body only", "0 = no attributed speech"),
        ("speech_share", "real", "0-1", "their speech_seconds / all castaways' speech_seconds this episode (host and unknown excluded)", "NA if no castaway speech"),
        ("lines", "int", "lines", "subtitle lines (utterances) attributed to them", ""),
        ("turns", "int", "turns", "stretches of consecutive lines by them; another speaker or an unattributed line ends a turn", ""),
        ("words", "int", "words", "Whisper-aligned words on their lines (caption words where alignment failed)", ""),
        ("words_per_minute", "real", "wpm", "words / speech_seconds * 60; lines are timed word to word, so pauses between lines are left out (runs faster than conversational wpm)", "NA under 10 s of speech"),
        ("confessional_seconds", "real", "s", "speech in confessional runs (italic captions or a run of 6 s or more)", ""),
        ("confessional_lines", "int", "lines", "lines in confessional runs", ""),
        ("confessional_runs", "int", "runs", "confessional runs; about 2x survivoR's hand count (a confessional cut by a scene counts twice) but ranks agree", ""),
        ("field_seconds", "real", "s", "speech outside confessionals (camp, challenges, Tribal Council)", ""),
        ("recap_seconds", "real", "s", "their lines replayed in the 'previously on' recap (not in speech_seconds)", ""),
        ("mentions_given", "int", "lines", "lines of theirs naming another castaway or the host", ""),
        ("mentions_received", "int", "lines", "lines by anyone else (including unattributed lines) naming them", ""),
        ("mentions_received_confessional", "int", "lines", "of mentions_received, lines in confessionals", ""),
        ("namers", "int", "people", "distinct attributed speakers who named them", ""),
        ("direct_addresses_received", "int", "lines", "lines naming them as the person spoken to ('Kenzie, ...', '..., Kenzie?')", ""),
        ("host_mentions_received", "int", "lines", "host lines naming them", ""),
        ("host_addresses_received", "int", "lines", "host lines addressing them directly", ""),
        ("seconds_human", "real", "s", "speech_seconds labelled by a person in the review app", ""),
        ("seconds_namecard", "real", "s", "speech_seconds labelled from an on-screen name card", ""),
        ("seconds_caption", "real", "s", "speech_seconds labelled by a caption NAME: tag", ""),
        ("seconds_auto", "real", "s", "speech_seconds labelled by automatic voice match", ""),
        ("expected_wrong_seconds", "real", "s", "sum over automatic labels of (1 - p_right) * duration: speech probably someone else's", ""),
        ("est_precision", "real", "0-1", "1 - expected_wrong_seconds / speech_seconds (trusted labels count as right)", "NA if no speech"),
        ("survivor_confessional_count", "int", "count", "survivoR confessionals.confessional_count, for comparison", "NA if survivoR has no row"),
        ("survivor_confessional_seconds", "real", "s", "survivoR confessionals.confessional_time", "NA if survivoR has no row"),
    ],
    "speech_mentions": KEYS + [
        ("source_id", "text", "", "who said the line: castaway_id, host id (HOST_US) or UNKNOWN (no speaker yet)", "never NA"),
        ("target_id", "text", "", "who is named: castaway_id or host id", "never NA"),
        ("lines_naming", "int", "lines", "lines by source naming target (a line naming them twice counts once)", ""),
        ("direct_address", "int", "lines", "of those, lines addressing target directly", ""),
        ("in_confessional", "int", "lines", "of those, lines in confessionals", ""),
    ],
    "speech_interactions": KEYS + [
        ("from_id", "text", "", "speaker of a turn in conversation (not a confessional)", "never NA"),
        ("to_id", "text", "", "the next speaker, starting within 5 s", "never NA"),
        ("transitions", "int", "count", "times to_id's turn directly followed from_id's", ""),
        ("share_of_from", "real", "0-1", "transitions / all of from_id's transitions this episode", ""),
    ],
    "speech_quality": KEYS + [
        ("body_seconds", "real", "s", "time on subtitle lines in the episode body (recap and preview excluded)", ""),
        ("body_lines", "int", "lines", "lines in the body", ""),
        ("castaway_seconds", "real", "s", "body speech attributed to castaways", ""),
        ("host_seconds", "real", "s", "body speech attributed to the host", ""),
        ("unknown_seconds", "real", "s", "body speech with no speaker yet (in the review queue or unmatched)", ""),
        ("unknown_lines", "int", "lines", "body lines with no speaker yet", ""),
        ("speaker_coverage", "real", "0-1", "(castaway_seconds + host_seconds) / body_seconds: the denominator every other count shares", ""),
        ("share_human", "real", "0-1", "share of attributed seconds labelled by a person", "NA if nothing attributed"),
        ("share_namecard", "real", "0-1", "share labelled from name cards", "NA if nothing attributed"),
        ("share_caption", "real", "0-1", "share labelled from caption NAME: tags", "NA if nothing attributed"),
        ("share_auto", "real", "0-1", "share labelled by automatic voice match", "NA if nothing attributed"),
        ("auto_expected_precision", "real", "0-1", "duration-weighted mean p_right of the automatic labels (calibrated on replays)", "NA if no automatic labels"),
        ("audit_n", "int", "groups", "automatic-label groups checked by ear in the random audit sample", "NA if not audited"),
        ("audit_right", "int", "groups", "of those, confirmed right", "NA if not audited"),
        ("queue_open_lines", "int", "lines", "body lines waiting in the review queue", ""),
        ("aligned_share", "real", "0-1", "share of body lines whose timing came from Whisper words (else caption timing)", ""),
        ("recap_seconds", "real", "s", "subtitle time in the recap", ""),
        ("preview_seconds", "real", "s", "subtitle time in the next-week preview", ""),
        ("castaways_in_game", "int", "people", "castaways in survivoR boot_mapping for the episode", ""),
        ("mentions", "int", "lines", "(line, named person) pairs in the body", ""),
        ("mentions_unknown_speaker", "int", "lines", "of those, on lines with no speaker yet", ""),
        ("ambiguous_names", "int", "count", "names shared by two castaways still in the game, not counted", ""),
        ("transitions", "int", "count", "conversation turn transitions between two attributed speakers", ""),
        ("transitions_unknown", "int", "count", "transitions with an unattributed side, not in speech_interactions", ""),
        ("confessional_rank_agreement", "real", "-1..1", "Spearman correlation of our caption-named confessional runs with survivoR's confessional counts", "NA if survivoR has none"),
    ],
}


def write_csv(path: Path, table: str, rows: list[dict]) -> None:
    cols = [f[0] for f in DICTIONARY[table]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def export(settings, con: sqlite3.Connection, resolver, seasons: list[str] | None, out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    have = con.execute("""SELECT version_season, episode FROM utterances WHERE version_season IN
                              (SELECT DISTINCT u.version_season FROM labels l JOIN utterances u USING (utt_id))
                          GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()           # seasons with any speaker labels
    eps = [(vs, ep) for vs, ep in have if not seasons or vs in seasons]
    sv = sqlite3.connect(f"file:{settings.survivor_db_path}?mode=ro", uri=True) if settings.survivor_db_path.exists() else None
    tables: dict[str, list[dict]] = defaultdict(list)
    for vs, ep in eps:
        ctx = episode_ctx(resolver, vs, ep)
        t = episode_tables(load_episode(con, vs, ep), ctx, vs, ep, episode_extra(con, sv, vs, ep))
        for k, rows in t.items():
            tables[k].extend(rows)
        log(f"{vs} E{ep:02d}: coverage {t['speech_quality'][0]['speaker_coverage']}")
    files = {}
    for name in DICTIONARY:
        p = out / f"{name}.csv"
        write_csv(p, name, tables[name])
        files[p.name] = {"rows": len(tables[name]), "sha256_16": _sha(p)}
    with open(out / "data_dictionary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "field", "type", "unit", "description", "na"])
        for tname, fields in DICTIONARY.items():
            for fld in fields:
                w.writerow([tname, *fld])
    cal = settings.paths.work_root / "models" / "calibrator.json"
    by_season: dict[str, list[int]] = defaultdict(list)
    for vs, ep in eps:
        by_season[vs].append(ep)
    manifest = {
        "export_version": EXPORT_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "survspk_commit": _git(Path(__file__).resolve().parents[1]),
        "episodes": dict(by_season),
        "counting": {"segment": "body only (recap and preview excluded)", "turn_gap_s": TURN_GAP_S,
                     "confessional": f"italic captions or runs >= {settings.segment.confessional_run_s} s",
                     "labels": "human, name card, caption NAME:, automatic voice match (p_right from the calibrator)"},
        "models": {"speaker_embedding": settings.embed.model, "asr_timing": settings.align.anchor_mlx_model,
                   "source_separation": settings.audio.demucs_model,
                   "calibrator": ({"sha256_16": _sha(cal), "modified": dt.datetime.fromtimestamp(cal.stat().st_mtime).isoformat(timespec="seconds")}
                                  if cal.exists() else None)},
        "survivoR_snapshot": ({"path": settings.survivor_db_path.name,
                               "modified": dt.datetime.fromtimestamp(settings.survivor_db_path.stat().st_mtime).isoformat(timespec="seconds")}
                              if settings.survivor_db_path.exists() else None),
        "files": files,
        "contains": "derived counts and public castaway names only; no audio, video or dialogue text",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if sv is not None:
        sv.close()
    return manifest
