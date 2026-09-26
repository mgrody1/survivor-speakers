"""survspk backup: a consistent snapshot with rotation, and an export of the human work with no dialogue text."""

import csv
import json
import sqlite3

from survspk.backup import export_labels, run_backup, snapshot
from survspk.splits import split_utterance
from tests.test_bank_assign import season  # noqa: F401


def test_snapshot_rotates_and_export_has_no_text(season, tmp_path):
    s, con, sea = season
    ids = sea.episode(2, [{"spk": "S_A", "label": None, "n": 2, "dur_each": 4.0, "text": "SECRET LINE ONE"},
                          {"spk": "S_B", "label": None, "n": 1, "dur_each": 6.0, "text": "SECRET WORDS TWO"}])
    a, b = ids[0][0], ids[1][0]
    con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, text)
                   VALUES (?, 'S_A', 'human', 1.0, '{"note": "audit:confirm"}', 'SECRET LINE ONE')""", (a,))
    con.execute("""INSERT INTO audit_verdicts (utt_id, version_season, episode, verdict, speaker_id, pred_speaker, prev_label)
                   VALUES (?, 'US99', 2, 'confirm', 'S_A', 'S_A', ?)""",
                (a, json.dumps({"utt_id": a, "speaker_id": "S_A", "source": "auto", "text": "SECRET LINE ONE"})))
    con.execute("""INSERT INTO card_checks (version_season, episode, t_s, castaway_id, utt_id, utt_ids)
                   VALUES ('US99', 2, 12.5, 'S_A', ?, ?)""", (a, json.dumps([a])))
    con.commit()
    split_utterance(con, b, 1)                                  # its original row (with text) goes into utt_splits
    con.commit()

    # snapshots: consistent, rotated to `keep`
    dest = tmp_path / "backups"
    for k in range(3):
        snapshot(s.db_path, dest, keep=2, stamp=f"2026010{k}_000000")
    left = sorted(p.name for p in dest.glob("survspk_*.sqlite"))
    assert len(left) == 2 and "survspk_20260100_000000.sqlite" not in left
    c2 = sqlite3.connect(dest / left[-1])
    assert c2.execute("SELECT speaker_id FROM labels WHERE utt_id=?", (a,)).fetchone()[0] == "S_A"
    c2.close()

    out = tmp_path / "export"
    counts = export_labels(sqlite3.connect(s.db_path), out)
    assert counts["labels.csv"] >= 1 and counts["audit_verdicts.csv"] == 1 and counts["card_checks.csv"] == 1
    assert counts["splits.csv"] == 1
    blob = "".join(p.read_text() for p in out.iterdir())
    assert "SECRET" not in blob                                 # no dialogue text anywhere
    labs = list(csv.DictReader((out / "labels.csv").open()))
    row = next(r for r in labs if r["utt_id"] == a)
    assert row["speaker_id"] == "S_A" and row["source"] == "human" and float(row["end_s"]) > float(row["start_s"])
    sp = list(csv.DictReader((out / "splits.csv").open()))[0]
    parts = json.loads(sp["parts"])
    assert sp["base_utt_id"] == b and [p_["utt_id"] for p_ in parts] == [b + "a", b + "b"]
    assert parts[0]["end_s"] <= parts[1]["start_s"] + 1e-6 and parts[1]["split_at_word"] == 1
    m = json.loads((out / "manifest.json").read_text())
    assert m["files"] == counts and m["per_season"][0]["vs"] == "US99"


def test_run_backup_copies(season, tmp_path):
    s, con, sea = season
    r = run_backup(s, keep=3, copy_to=tmp_path / "offsite")
    assert r.check == "ok" and r.snapshot.exists() and (tmp_path / "offsite" / r.snapshot.name).exists()
    assert (tmp_path / "offsite" / "labels" / "manifest.json").exists()
