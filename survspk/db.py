"""SQLite schema and helpers (spec §6)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    version_season      TEXT NOT NULL,
    episode             INTEGER NOT NULL,
    video_path          TEXT,
    video_basename      TEXT,
    subtitle_path       TEXT,
    subtitle_source     TEXT,          -- sidecar_sdh | sidecar_plain | embedded_sdh | embedded_plain | none
    subtitle_convention TEXT,          -- json flags from subparse
    subtitle_timing_ok  INTEGER,
    audio_channels      INTEGER,
    audio_codec         TEXT,
    video_codec         TEXT,
    width               INTEGER,
    height              INTEGER,
    n_embedded_subs     INTEGER,
    embedded_subs       TEXT,          -- json [{index, codec, language, title}]
    audio_raw_path      TEXT,
    audio_center_path   TEXT,
    audio_vocals_path   TEXT,
    duration_s          REAL,          -- ffprobe
    survivor_length_min REAL,          -- survivoR episodes.episode_length
    length_delta_min    REAL,          -- duration_s/60 - survivor_length_min
    survivor_title      TEXT,
    is_reunion          INTEGER DEFAULT 0,
    is_double           INTEGER DEFAULT 0,
    episode2            INTEGER,        -- second episode number for double-episode files
    subtitle_offset_ms  INTEGER DEFAULT 0,
    recap_end_s         REAL,
    preview_start_s     REAL,
    status              TEXT,
    probed_at           TEXT,
    PRIMARY KEY (version_season, episode)
);

CREATE TABLE IF NOT EXISTS subtitle_files (
    version_season   TEXT NOT NULL,
    episode          INTEGER NOT NULL,
    path             TEXT NOT NULL,     -- file path, or "<video>#s:<idx>" for embedded
    source           TEXT NOT NULL,     -- sidecar | embedded
    is_sdh           INTEGER,
    n_cues           INTEGER,
    first_cue_s      REAL,
    last_cue_end_s   REAL,
    duration_delta_s REAL,              -- last_cue_end_s - ffprobe duration
    parse_error      TEXT,
    timing_ok        INTEGER,           -- null unknown, 1 ok, 0 suspect (see inventory.timing_ok)
    chosen           INTEGER DEFAULT 0,
    PRIMARY KEY (version_season, episode, path)
);

CREATE TABLE IF NOT EXISTS cues (
    cue_id         TEXT PRIMARY KEY,
    version_season TEXT NOT NULL,
    episode        INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    raw_text       TEXT,
    lines          TEXT                 -- json [{text, sdh_name, is_turn, is_italic, is_sound}]
);
CREATE INDEX IF NOT EXISTS cues_ep ON cues(version_season, episode, idx);

CREATE TABLE IF NOT EXISTS utterances (
    utt_id         TEXT PRIMARY KEY,
    version_season TEXT NOT NULL,
    episode        INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    text           TEXT,
    segment        TEXT,
    is_speech      INTEGER,
    sdh_name       TEXT,
    sdh_speaker_id TEXT,
    is_italic      INTEGER,
    domain_hint    TEXT,
    align_ok       INTEGER,
    flags          TEXT
);
CREATE INDEX IF NOT EXISTS utt_ep ON utterances(version_season, episode, idx);

CREATE TABLE IF NOT EXISTS utterance_cues (
    utt_id       TEXT NOT NULL,
    cue_id       TEXT NOT NULL,
    line_indices TEXT,
    PRIMARY KEY (utt_id, cue_id)
);

CREATE TABLE IF NOT EXISTS speakers (
    version_season TEXT NOT NULL,
    speaker_id     TEXT NOT NULL,
    display_name   TEXT,
    short_name     TEXT,
    aliases        TEXT,
    active_from_ep INTEGER,
    active_to_ep   INTEGER,
    PRIMARY KEY (version_season, speaker_id)
);

CREATE TABLE IF NOT EXISTS speaker_bank (
    version_season TEXT NOT NULL,
    speaker_id     TEXT NOT NULL,
    domain         TEXT NOT NULL,
    as_of_episode  INTEGER NOT NULL,
    centroid       BLOB,
    exemplars      BLOB,
    n_utts         INTEGER,
    PRIMARY KEY (version_season, speaker_id, domain, as_of_episode)
);

CREATE TABLE IF NOT EXISTS labels (
    utt_id         TEXT PRIMARY KEY,
    speaker_id     TEXT NOT NULL,
    source         TEXT NOT NULL,       -- sdh | chyron | human | auto
    confidence     REAL,
    top_candidates TEXT,
    domain         TEXT,
    labeled_at     TEXT
);

CREATE TABLE IF NOT EXISTS review_queue (
    utt_id   TEXT PRIMARY KEY,
    reason   TEXT NOT NULL,
    payload  TEXT,
    resolved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chyron_hits (
    version_season TEXT NOT NULL,
    episode        INTEGER NOT NULL,
    t_s            REAL NOT NULL,
    ocr_text       TEXT,
    castaway_id    TEXT,
    ocr_conf       REAL,
    match_score    REAL,
    PRIMARY KEY (version_season, episode, t_s)
);

CREATE TABLE IF NOT EXISTS metrics (
    version_season TEXT NOT NULL,
    episode        INTEGER NOT NULL,
    key            TEXT NOT NULL,
    value          REAL,
    payload        TEXT,
    computed_at    TEXT,
    PRIMARY KEY (version_season, episode, key)
);

CREATE TABLE IF NOT EXISTS words (
    -- word-level alignment from WhisperX, per cue (spec §7.3); line_idx/word_idx index into cues.lines text
    cue_id   TEXT NOT NULL,
    line_idx INTEGER NOT NULL,
    word_idx INTEGER NOT NULL,
    word     TEXT,
    start_s  REAL,
    end_s    REAL,
    score    REAL,
    PRIMARY KEY (cue_id, line_idx, word_idx)
);

CREATE TABLE IF NOT EXISTS roots (
    -- the absolute roots each profile has used, so stored paths can be translated between the Mac and the VM
    profile TEXT NOT NULL,
    key     TEXT NOT NULL,          -- video_root | subtitle_root | work_root
    path    TEXT NOT NULL,
    PRIMARY KEY (profile, key)
);

CREATE TABLE IF NOT EXISTS name_tokens (
    -- every distinct SDH NAME: token seen, per season, with resolution result (M0 names-report)
    version_season TEXT NOT NULL,
    token          TEXT NOT NULL,
    n_lines        INTEGER,
    n_files        INTEGER,
    resolved_id    TEXT,
    resolution     TEXT,               -- cast | host | stop | alias | unresolved
    PRIMARY KEY (version_season, token)
);
"""


# Columns added after the first release; applied by migrate() so existing databases keep working.
MIGRATIONS: dict[str, dict[str, str]] = {
    "episodes": {
        "audio_vocals_center_path": "TEXT",
        "audio_variants": "TEXT",          # json {variant: path}
        "align_offset_s": "REAL",          # b in t_audio = a * t_sub + b
        "align_drift": "REAL",             # a - 1
        "align_stats": "TEXT",             # json
        "n_utterances": "INTEGER",
    },
    "utterances": {
        "n_words": "INTEGER",
        "sdh_resolution": "TEXT",          # cast | host | alias | other | ambiguous | unresolved
    },
    "speaker_bank": {
        "dim": "INTEGER",                  # embedding dimension of centroid / exemplars
        "variant": "TEXT",                 # audio variant the bank was fitted on
        "model": "TEXT",                   # embedding model
        "total_dur_s": "REAL",
        "payload": "TEXT",                 # json: exemplar utt_ids, episodes used, dropped-as-inconsistent count
    },
    "labels": {
        "run_id": "INTEGER",               # run the utterance was scored in (assign pools runs)
        "margin": "REAL",
    },
}


def migrate(con: sqlite3.Connection) -> list[str]:
    applied = []
    for table, cols in MIGRATIONS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                applied.append(f"{table}.{col}")
    con.commit()
    return applied


def connect(path: Path, journal: str = "WAL", row_factory: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    if row_factory:
        con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA journal_mode={journal}")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(path: Path, journal: str = "WAL") -> sqlite3.Connection:
    con = connect(path, journal)
    con.executescript(SCHEMA)
    con.commit()
    migrate(con)
    return con


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise


def upsert(con: sqlite3.Connection, table: str, row: dict[str, Any], keys: Iterable[str]) -> None:
    cols = list(row)
    placeholders = ",".join("?" for _ in cols)
    keys = list(keys)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in keys)
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({','.join(keys)}) DO UPDATE SET {updates}"
    )
    con.execute(sql, [_coerce(row[c]) for c in cols])


def _coerce(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, bool):
        return int(v)
    return v


def rows(con: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return con.execute(sql, params).fetchall()


# ----------------------------------------------------------------------------- path translation

ROOT_KEYS = ("video_root", "subtitle_root", "work_root")


def record_roots(con: sqlite3.Connection, settings) -> None:
    """Remember this profile's roots (called by every stage that writes paths)."""
    for k in ROOT_KEYS:
        con.execute("INSERT OR REPLACE INTO roots (profile, key, path) VALUES (?,?,?)",
                    (settings.profile, k, str(getattr(settings.paths, k))))
    con.commit()


def localize(con: sqlite3.Connection, settings, path: str | Path | None) -> Path | None:
    """Translate a path stored by another profile (e.g. the VM's /sessions/.../mnt/...) into this profile's
    equivalent. Paths under the current roots pass through unchanged. Works on '<video>#s:<idx>' too."""
    if path is None:
        return None
    s = str(path)
    suffix = ""
    if "#s:" in s:
        s, idx = s.rsplit("#s:", 1)
        suffix = f"#s:{idx}"
    current = {k: str(getattr(settings.paths, k)).rstrip("/") for k in ROOT_KEYS}
    if any(s == c or s.startswith(c + "/") for c in current.values()):
        return Path(s + suffix)
    try:
        stored = con.execute("SELECT key, path FROM roots WHERE profile != ?", (settings.profile,)).fetchall()
    except sqlite3.OperationalError:
        stored = []
    for r in sorted(stored, key=lambda r: -len(r["path"])):
        old = str(r["path"]).rstrip("/")
        if s == old or s.startswith(old + "/"):
            return Path(current[r["key"]] + s[len(old):] + suffix)
    return Path(s + suffix)
