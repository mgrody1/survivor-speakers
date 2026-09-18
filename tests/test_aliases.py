"""Resolver tests on a tiny synthetic survivoR database (All-Stars-style name clashes)."""

import sqlite3

import pytest

from survspk.aliases import Resolver, normalize_token
from survspk.config import load_settings


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s = s.model_copy(update={"paths": s.paths.model_copy(update={"work_root": tmp_path})})
    db = s.survivor_db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE castaways (version_season TEXT, castaway_id TEXT, castaway TEXT, full_name TEXT);
        CREATE TABLE castaway_details (castaway_id TEXT, full_name_detailed TEXT, last_name TEXT);
        CREATE TABLE boot_mapping (version_season TEXT, episode INTEGER, castaway_id TEXT);
        INSERT INTO castaways VALUES
          ('US08','US0094','Rob C.','Rob Cesternino'), ('US08','US0055','Boston Rob','Rob Mariano'),
          ('US08','US0096','Jenna M.','Jenna Morasca'), ('US08','US0009','Jenna L.','Jenna Lewis'),
          ('US08','US0060','Rupert','Rupert Boneham'),
          ('US01','US0016','Richard','Richard Hatch'), ('US01','US0013','Sue','Susan Hawk'),
          ('US47','US0713','Teeny','Teeny Chirichillo'), ('US47','US0715','TK','TK Foster');
        INSERT INTO castaway_details VALUES ('US0094','Rob Cesternino','Cesternino'),('US0055','Rob Mariano','Mariano');
        -- Rob C. and Jenna M. gone after episode 4
        INSERT INTO boot_mapping VALUES ('US08',2,'US0094'),('US08',2,'US0055'),('US08',2,'US0096'),('US08',2,'US0009'),
                                        ('US08',6,'US0055'),('US08',6,'US0009'),('US08',6,'US0060');
    """)
    con.commit(); con.close()
    return s


def test_normalize():
    assert normalize_token("JEFF PROBST (V.O.)") == "JEFF PROBST"
    assert normalize_token(" rob m. ") == "ROB M"
    assert normalize_token("B.B.") == "B.B"


def test_resolution_paths(settings):
    r = Resolver(settings)
    assert r.resolve("PROBST", "US47").kind == "host"
    assert r.resolve("Jeff Probst", "US01").speaker_id == "HOST_US"
    assert r.resolve("MAN", "US47").kind == "stop"
    assert r.resolve("MAN #2", "US47").kind == "stop"
    assert r.resolve("DR. BARRY", "US49").kind == "stop"
    assert r.resolve("TEENY", "US47") == r.resolve("teeny", "US47")
    assert r.resolve("TEENY", "US47").speaker_id == "US0713"
    assert r.resolve("TK", "US47").speaker_id == "US0715"
    assert r.resolve("RICHARD", "US01").speaker_id == "US0016"
    assert r.resolve("RICH", "US01").kind == "alias" and r.resolve("RICH", "US01").speaker_id == "US0016"
    assert r.resolve("SUSAN", "US01").speaker_id == "US0013"
    assert r.resolve("TED", "US40").speaker_id == "OTHER"
    assert r.resolve("JF", "US19").speaker_id == "HOST_US"
    assert r.resolve("ELAPSED TIME", "US30").kind == "stop"
    assert r.resolve("NOBODY", "US47").kind == "unresolved"


def test_first_name_clash_settled_per_episode(settings):
    r = Resolver(settings)
    amb = r.resolve("ROB", "US08")
    assert amb.kind == "ambiguous" and set(amb.candidates) == {"US0094", "US0055"}
    assert r.resolve("ROB", "US08", episode=2).kind == "ambiguous"        # both still in
    assert r.resolve("ROB", "US08", episode=6).speaker_id == "US0055"      # only Boston Rob left
    assert r.resolve("JENNA", "US08", episode=6).speaker_id == "US0009"
    assert r.resolve("ROB M", "US08").speaker_id == "US0055"               # alias
    assert r.resolve("ROB C.", "US08").speaker_id == "US0094"              # short name
    assert r.resolve("RUPERT", "US08").speaker_id == "US0060"
