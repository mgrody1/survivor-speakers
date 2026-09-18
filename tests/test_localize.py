import sqlite3
from pathlib import Path

from survspk import db as dbm
from survspk.config import load_settings


def test_localize_translates_other_profile_roots(tmp_path):
    s = load_settings()
    con = dbm.init_db(tmp_path / "t.sqlite", "TRUNCATE")
    con.execute("INSERT INTO roots VALUES ('other','video_root','/sessions/abc/mnt/Survivor (2000) {TvbId-76733}')")
    con.execute("INSERT INTO roots VALUES ('other','subtitle_root','/sessions/abc/mnt/Subtitles')")
    con.commit()
    vr = str(s.paths.video_root)
    stored = "/sessions/abc/mnt/Survivor (2000) {TvbId-76733}/Season 47/x.mkv"
    assert str(dbm.localize(con, s, stored)) == f"{vr}/Season 47/x.mkv"
    assert str(dbm.localize(con, s, stored + "#s:2")) == f"{vr}/Season 47/x.mkv#s:2"
    # paths already under the current roots pass through
    mine = f"{vr}/Season 1/y.avi"
    assert str(dbm.localize(con, s, mine)) == mine
    # unknown prefixes pass through unchanged
    assert str(dbm.localize(con, s, "/elsewhere/z.mkv")) == "/elsewhere/z.mkv"
    assert dbm.localize(con, s, None) is None
    dbm.record_roots(con, s)
    assert con.execute("SELECT COUNT(*) FROM roots WHERE profile=?", (s.profile,)).fetchone()[0] == 3
