"""Pull survivoR tables into a local SQLite (spec §2.5).

Adapted from Gamebot/gamebot_core/github_data_loader.py: download <table>.rda from the survivoR
GitHub `data/` directory, read with pyreadr, write with pandas. Falls back to the Gamebot snapshot
(gamebot_lite/data/gamebot.sqlite) per table when a download fails.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyreadr
import requests

from .config import Settings
from .db import connect as db_connect

log = logging.getLogger(__name__)

RDA_MAGIC = (b"RDX2", b"RDX3", b"RDA2", b"RDA3", b"\x1f\x8b", b"BZh", b"\xfd7zXZ")  # plain or compressed


def download_rda(table: str, raw_url: str, cache_dir: Path, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{table}.rda"
    if local.exists() and not force:
        return local
    url = f"{raw_url.rstrip('/')}/{table}.rda"
    log.info("downloading %s", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if not r.content.startswith(RDA_MAGIC):
        raise RuntimeError(f"{url}: payload does not look like an .rda file (starts {r.content[:8]!r})")
    local.write_bytes(r.content)
    return local


def read_rda(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    if not result:
        raise RuntimeError(f"{path}: no objects in rda")
    # survivoR .rda files contain a single object named after the table
    name, df = next(iter(result.items()))
    log.info("read %s: %s rows x %s cols", name, *df.shape)
    return df


def snapshot_table(snapshot: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(snapshot)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()
    # drop Gamebot's ingestion metadata columns so the schema matches upstream
    return df.drop(columns=[c for c in ("ingest_run_id", "ingested_at", "source_dataset") if c in df], errors="ignore")


def refresh(settings: Settings, force: bool = False, snapshot_only: bool = False) -> dict[str, dict]:
    out_db = settings.survivor_db_path
    out_db.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_db.parent / "rda_cache"
    con = db_connect(out_db, settings.sqlite_journal, row_factory=False)
    report: dict[str, dict] = {}
    try:
        for table in settings.survivor.tables:
            source = "upstream"
            df: pd.DataFrame | None = None
            if not snapshot_only:
                try:
                    df = read_rda(download_rda(table, settings.survivor.raw_url, cache_dir, force))
                except Exception as e:  # noqa: BLE001
                    log.warning("upstream failed for %s (%s); falling back to snapshot", table, e)
            if df is None:
                df = snapshot_table(settings.paths.survivor_snapshot, table)
                source = "snapshot"
            # pyreadr gives object dtype for factors/strings; normalise column names to lower snake
            df.columns = [str(c).strip().lower() for c in df.columns]
            df.to_sql(table, con, if_exists="replace", index=False)
            report[table] = {"rows": int(len(df)), "source": source}
        meta = pd.DataFrame(
            [{"table": t, **v, "refreshed_at": datetime.now(timezone.utc).isoformat()} for t, v in report.items()]
        )
        meta.to_sql("_refresh_log", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return report


def coverage_summary(settings: Settings) -> pd.DataFrame:
    """Per version_season: cast size, #episodes with boot_mapping, #episodes in episodes table."""
    con = sqlite3.connect(settings.survivor_db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT c.version_season,
                   COUNT(DISTINCT c.castaway_id)                       AS n_cast,
                   (SELECT COUNT(DISTINCT episode) FROM boot_mapping b
                     WHERE b.version_season = c.version_season)        AS n_ep_boot_mapping,
                   (SELECT COUNT(*) FROM episodes e
                     WHERE e.version_season = c.version_season)        AS n_ep_episodes
            FROM castaways c
            WHERE c.version_season LIKE 'US%' OR c.version_season LIKE 'AU%'
            GROUP BY c.version_season ORDER BY c.version_season
            """,
            con,
        )
    finally:
        con.close()
