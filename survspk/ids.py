"""Identifier conventions (spec §5)."""

from __future__ import annotations

import re

# Doubles: S07E14E15, S30E04-05, S42E06-E07 are captured by the optional second episode group.
SXXEYY = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,2})(?:[-–]?[Ee]?(\d{1,2}))?(?![\dp])")


def version_season(franchise: str, season: int) -> str:
    return f"{franchise}{season:02d}"


def parse_sxxeyy(name: str) -> tuple[int, int, int | None] | None:
    """Return (season, episode, second_episode_or_None) from a filename, or None."""
    m = SXXEYY.search(name)
    if not m:
        return None
    s, e, e2 = int(m.group(1)), int(m.group(2)), m.group(3)
    return s, e, (int(e2) if e2 else None)


def cue_id(vs: str, episode: int, idx: int) -> str:
    return f"{vs}_E{episode:02d}_C{idx:04d}"


def utt_id(vs: str, episode: int, idx: int) -> str:
    return f"{vs}_E{episode:02d}_U{idx:04d}"


def episode_tag(vs: str, episode: int) -> str:
    return f"{vs}E{episode:02d}"
