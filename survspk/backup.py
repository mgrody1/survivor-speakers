"""`survspk backup`: a safe copy of the live database, and a small text-free export of the work a person did.

The DB (work_root/db/survspk.sqlite) holds everything, but most of it can be rebuilt by re-running the pipeline:
audio, alignment, utterances, embeddings, auto labels, caption names. What cannot be rebuilt is what a person
decided: human labels, checked name cards, audit verdicts, splits, and accepted or dismissed split suggestions.

1. Snapshot: SQLite's online backup API from a read-only connection, so it is consistent while the review app or a
   pipeline run is writing. Written to db/backups/survspk_<YYYYMMDD_HHMMSS>.sqlite (single file, no WAL),
   quick_check'ed, and only the newest `keep` snapshots are kept.
2. Export: CSVs + manifest.json under exports/labels/ (overwritten each run, so the folder diffs cleanly in git or a
   synced drive). Keyed by utt_id and times; no subtitle or dialogue text anywhere (the `text` column and the
   copies of it inside audit prev_label and split originals are dropped), so it can leave this machine.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

TEXT_KEYS = {"text", "sdh_name", "lines", "ocr_text"}      # never exported (dialogue / caption text)


@dataclass
class BackupResult:
    snapshot: Path | None = None
    snapshot_mb: float = 0.0
    check: str = ""
    removed: list[Path] = field(default_factory=list)
    export_dir: Path | None = None
    counts: dict = field(default_factory=dict)
    copied_to: list[Path] = field(default_factory=list)


def snapshot(db: Path, dest_dir: Path, keep: int = 5, stamp: str | None = None) -> tuple[Path, str, list[Path]]:
    """Online backup of `db` into dest_dir; returns (path, quick_check result, snapshots removed by rotation)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    out = dest_dir / f"survspk_{stamp}.sqlite"
    tmp = out.with_suffix(".sqlite.partial")
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst, pages=8192)
        dst.execute("PRAGMA journal_mode=DELETE")
        check = dst.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        dst.close()
        src.close()
    if check != "ok":
        raise RuntimeError(f"snapshot failed quick_check: {check} (left at {tmp})")
    tmp.replace(out)
    snaps = sorted(dest_dir.glob("survspk_*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for old in snaps[max(1, keep):]:
        for p in (old, Path(str(old) + "-wal"), Path(str(old) + "-shm")):
            if p.exists():
                p.unlink()
                removed.append(p)
    return out, check, removed


def _scrub(obj):
    """Drop dialogue/caption text from a JSON value (recursively)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in TEXT_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _json(s: str | None):
    try:
        return json.loads(s) if s else None
    except ValueError:
        return None


def _write(path: Path, cols: list[str], rows) -> int:
    n = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            keys = set(r.keys())
            w.writerow([r[c] if c in keys else None for c in cols])      # a column newer than this database: blank
            n += 1
    return n


def export_labels(con: sqlite3.Connection, out_dir: Path) -> dict:
    """The work a person did, as CSVs without any dialogue text. Returns row counts per file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con.row_factory = sqlite3.Row
    counts = {}

    def q(sql: str) -> list:
        try:
            return con.execute(sql).fetchall()
        except sqlite3.OperationalError:                    # a table made on first use (utt_splits, ...)
            return []

    def note(tc: str | None) -> str:
        return json.dumps(_scrub(_json(tc)), sort_keys=True) if tc else ""

    rows = [dict(r) | {"top_candidates": note(r["top_candidates"])} for r in con.execute(
        """SELECT l.utt_id, u.version_season, u.episode, u.start_s, u.end_s, u.segment, u.domain_hint AS domain,
                  l.speaker_id, l.source, l.confidence, l.top_candidates, l.labeled_at
           FROM labels l JOIN utterances u USING (utt_id)
           WHERE l.source IN ('human', 'chyron') ORDER BY u.version_season, u.episode, u.start_s""")]
    counts["labels.csv"] = _write(out_dir / "labels.csv", ["utt_id", "version_season", "episode", "start_s", "end_s",
                                                           "segment", "domain", "speaker_id", "source", "confidence",
                                                           "top_candidates", "labeled_at"], rows)
    counts["card_checks.csv"] = _write(out_dir / "card_checks.csv",
                                       ["version_season", "episode", "t_s", "castaway_id", "utt_id", "utt_ids", "created_at"],
                                       q("SELECT * FROM card_checks ORDER BY version_season, episode, t_s"))
    rows = [dict(r) | {"prev_label": json.dumps(_scrub(_json(r["prev_label"])), sort_keys=True) if r["prev_label"] else ""}
            for r in q("SELECT * FROM audit_verdicts ORDER BY version_season, episode, utt_id")]
    counts["audit_verdicts.csv"] = _write(out_dir / "audit_verdicts.csv",
                                          ["utt_id", "version_season", "episode", "run_id", "group_key", "pred_speaker",
                                           "pred_score", "verdict", "speaker_id", "prev_label", "created_at", "sample"], rows)
    rows = []
    for r in q("SELECT * FROM utt_splits ORDER BY base_utt_id"):
        orig = _json(r["original"]) or {}
        parts = _json(r["parts"]) or []
        spans = []
        for pid in parts:
            u = con.execute("SELECT start_s, end_s, flags FROM utterances WHERE utt_id=?", (pid,)).fetchone()
            f = (_json(u["flags"]) or {}) if u else {}
            spans.append({"utt_id": pid, "start_s": u["start_s"] if u else None, "end_s": u["end_s"] if u else None,
                          "split_at_word": f.get("split_at_word"), "auto": bool(f.get("split_auto"))})
        rows.append({"base_utt_id": r["base_utt_id"], "version_season": orig.get("version_season"),
                     "episode": orig.get("episode"), "start_s": orig.get("start_s"), "end_s": orig.get("end_s"),
                     "cues": json.dumps(_scrub(_json(r["cues"]))), "parts": json.dumps(spans), "created_at": r["created_at"]})
    counts["splits.csv"] = _write(out_dir / "splits.csv", ["base_utt_id", "version_season", "episode", "start_s", "end_s",
                                                           "cues", "parts", "created_at"], rows)
    counts["split_suggestions.csv"] = _write(
        out_dir / "split_suggestions.csv",
        ["utt_id", "version_season", "episode", "t_cut", "left_spk", "right_spk", "second_s", "contrast", "status", "created_at"],
        q("SELECT * FROM split_suggestions WHERE status IN ('accepted', 'dismissed', 'auto') ORDER BY version_season, episode, utt_id"))
    counts["music_labels.csv"] = _write(
        out_dir / "music_labels.csv",
        ["version_season", "episode", "t0", "t1", "cue", "subjects", "model_cue", "created_at"],
        q("SELECT * FROM music_labels ORDER BY version_season, episode, t0"))
    per = [dict(r) for r in con.execute(
        """SELECT u.version_season AS vs, COUNT(DISTINCT u.episode) AS episodes, SUM(l.source='human') AS human,
                  SUM(l.source='chyron') AS chyron FROM labels l JOIN utterances u USING (utt_id)
           WHERE l.source IN ('human', 'chyron') GROUP BY 1 ORDER BY 1""")]
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    (out_dir / "manifest.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "survspk_commit": commit, "files": counts, "per_season": per,
        "note": "No dialogue or caption text. Keys: utt_id (VS_E##_U####, split parts end in a/b) and times in audio "
                "seconds; restore by re-running the pipeline and re-applying these rows."}, indent=2))
    return counts


def run_backup(settings: Settings, keep: int = 5, export: bool = True, copy_to: Path | None = None,
               export_dir: Path | None = None) -> BackupResult:
    res = BackupResult()
    db = settings.db_path
    res.snapshot, res.check, res.removed = snapshot(db, db.parent / "backups", keep=keep)
    res.snapshot_mb = round(res.snapshot.stat().st_size / 1e6, 1)
    if export:
        res.export_dir = export_dir or (settings.paths.work_root / "exports" / "labels")
        con = sqlite3.connect(f"file:{res.snapshot}?mode=ro", uri=True)     # export from the snapshot just taken
        try:
            res.counts = export_labels(con, res.export_dir)
        finally:
            con.close()
    if copy_to is not None:
        copy_to = Path(copy_to).expanduser()
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(res.snapshot, copy_to / res.snapshot.name)
        res.copied_to.append(copy_to / res.snapshot.name)
        if res.export_dir is not None:
            shutil.copytree(res.export_dir, copy_to / "labels", dirs_exist_ok=True)
            res.copied_to.append(copy_to / "labels")
    return res
