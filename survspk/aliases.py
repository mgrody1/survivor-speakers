"""Resolve SDH `NAME:` tokens and chyron text to speaker ids (spec §2.5, §7.4).

Order: stoplist -> host names -> per-season aliases.yaml -> survivoR short name -> full name ->
first token of full name -> unresolved.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import Settings


@dataclass(frozen=True)
class Resolution:
    speaker_id: str | None
    kind: str  # cast | host | alias | other | stop | ambiguous | unresolved
    candidates: tuple[str, ...] = ()   # for 'ambiguous': the castaway_ids that matched


def normalize_token(tok: str) -> str:
    t = tok.upper()
    t = re.sub(r"\([^)]*\)", "", t)          # (V.O.), (OFF)
    t = re.sub(r"[^A-Z0-9.'\- #]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-")
    return t


class Resolver:
    def __init__(self, settings: Settings):
        self.settings = settings
        with open(settings.aliases_path) as f:
            cfg = yaml.safe_load(f) or {}
        self.stoplist = {normalize_token(x) for x in cfg.get("stoplist", [])}
        self.season_aliases: dict[str, dict[str, str]] = {
            vs: {normalize_token(k): v for k, v in (m or {}).items()} for vs, m in (cfg.get("seasons") or {}).items()
        }
        self._local = threading.local()     # one read-only connection per thread: the review UI serves requests from a
                                            # thread pool, and a sqlite connection refuses use outside its own thread

    @property
    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.settings.survivor_db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    # ---- per-season cast maps ----
    @lru_cache(maxsize=128)
    def cast(self, version_season: str) -> list[sqlite3.Row]:
        return self._con.execute(
            """SELECT DISTINCT c.castaway_id, c.castaway, c.full_name, d.full_name_detailed, d.last_name
               FROM castaways c LEFT JOIN castaway_details d USING (castaway_id)
               WHERE c.version_season = ?""",
            (version_season,),
        ).fetchall()

    @lru_cache(maxsize=128)
    def maps(self, version_season: str) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
        short: dict[str, str] = {}
        full: dict[str, str] = {}
        first: dict[str, list[str]] = {}
        for r in self.cast(version_season):
            cid = r["castaway_id"]
            if r["castaway"]:
                short[normalize_token(r["castaway"])] = cid
            for fn in (r["full_name"], r["full_name_detailed"]):
                if fn:
                    full[normalize_token(fn)] = cid
                    ft = normalize_token(fn).split(" ")[0]
                    first.setdefault(ft, [])
                    if cid not in first[ft]:
                        first[ft].append(cid)
        return short, full, first

    @lru_cache(maxsize=4096)
    def present(self, version_season: str, episode: int) -> frozenset[str]:
        """castaway_ids present in an episode per survivoR boot_mapping (any game_status)."""
        return frozenset(
            r[0] for r in self._con.execute(
                "SELECT DISTINCT castaway_id FROM boot_mapping WHERE version_season=? AND episode=?",
                (version_season, episode))
        )

    @lru_cache(maxsize=128)
    def mention_patterns(self, version_season: str) -> dict[str, "re.Pattern[str]"]:
        """speaker_id -> regex matching that person's name(s) as whole words in dialogue text. Used for the
        'mention rule': a run that names its predicted speaker in the third person is almost never that speaker
        (Survivor players say each other's names constantly and their own almost never). Nicknames from
        aliases.yaml are included; the host gets every configured host name."""
        names: dict[str, set[str]] = {}
        for r in self.cast(version_season):
            cid = r["castaway_id"]
            s = names.setdefault(cid, set())
            if r["castaway"]:
                s.add(r["castaway"])
            for fn in (r["full_name"], r["full_name_detailed"]):
                if fn:
                    s.add(fn.split()[0])
        for tok, cid in self.season_aliases.get(version_season, {}).items():
            if cid in names and len(tok) >= 3:
                names[cid].add(tok)
        hosts, host_id = self.host_names(version_season)
        names[host_id] = set(hosts) | {h.split()[0] for h in hosts}
        out = {}
        for cid, s in names.items():
            alts = sorted({re.escape(x.strip().lower()) for x in s if len(x.strip()) >= 2}, key=len, reverse=True)
            if alts:
                out[cid] = re.compile(r"(?<![A-Za-z'])(?:" + "|".join(alts) + r")(?![A-Za-z])", re.I)
        return out

    def mentions(self, text: str, speaker_id: str, version_season: str) -> bool:
        p = self.mention_patterns(version_season).get(speaker_id)
        return bool(p and text and p.search(text))

    def bios(self, version_season: str, episode: int) -> dict[str, dict]:
        """castaway_id -> {name, full_name, age, city, state, gender, occupation, tribe (as of episode)}."""
        rows = self._con.execute(
            """SELECT c.castaway_id, c.castaway, c.full_name, c.age, c.city, c.state, d.gender, d.occupation,
                      (SELECT tribe FROM boot_mapping b WHERE b.castaway_id=c.castaway_id AND b.version_season=c.version_season
                         AND b.episode=? ORDER BY b."order" DESC LIMIT 1) AS tribe
               FROM castaways c LEFT JOIN castaway_details d USING (castaway_id)
               WHERE c.version_season=? GROUP BY c.castaway_id""", (episode, version_season)).fetchall()
        return {r["castaway_id"]: {"name": r["castaway"] or (r["full_name"] or "?").split()[0], "full_name": r["full_name"],
                                   "age": r["age"], "city": r["city"], "state": r["state"], "gender": r["gender"],
                                   "occupation": r["occupation"], "tribe": r["tribe"]} for r in rows}

    def host_names(self, version_season: str) -> tuple[set[str], str]:
        fr = self.settings.franchise_for(version_season)
        return {normalize_token(x) for x in fr.host_names}, fr.host_id

    def resolve(self, token: str, version_season: str, episode: int | None = None) -> Resolution:
        """Resolve one SDH/chyron token. With `episode`, first-name clashes (ROB in All-Stars) are settled
        by who is still in the game; without it, such tokens come back as 'ambiguous'."""
        t = normalize_token(token)
        if not t:
            return Resolution(None, "unresolved")
        if t in self.stoplist or re.fullmatch(r"(MAN|WOMAN|GIRL|BOY|GUY|VOICE) ?#?\d+", t) or re.match(r"DR\.? ", t):
            return Resolution(None, "stop")
        hosts, host_id = self.host_names(version_season)
        if t in hosts:
            return Resolution(host_id, "host")
        al = self.season_aliases.get(version_season, {})
        if t in al:
            v = al[t]
            if v == "OTHER":
                return Resolution("OTHER", "other")
            if v.startswith("HOST_"):
                return Resolution(v, "host")
            return Resolution(v, "alias")
        short, full, first = self.maps(version_season)
        if t in short:
            return Resolution(short[t], "cast")
        if t in full:
            return Resolution(full[t], "cast")
        # "FIRST LAST" / "FIRST X." -> short-name match on first token
        head = t.split(" ")[0]
        if " " in t and head in short:
            return Resolution(short[head], "cast")
        cands = list(first.get(t, []))
        if " " in t and not cands:
            cands = list(first.get(head, []))
        if len(cands) == 1:
            return Resolution(cands[0], "cast")
        if len(cands) > 1:
            if episode is not None:
                alive = [c for c in cands if c in self.present(version_season, episode)]
                if len(alive) == 1:
                    return Resolution(alive[0], "cast")
            return Resolution(None, "ambiguous", tuple(sorted(cands)))
        return Resolution(None, "unresolved")

    def close(self) -> None:
        self._con.close()
