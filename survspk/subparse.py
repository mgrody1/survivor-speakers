"""Era-aware SRT parser (spec §2.3, §7.1).

Conventions seen in the library:
  * `NAME:` speaker prefixes         (S1-2, S5-9, S40-50 SDH files; only PROBST in S3-4, S10-20)
  * `>>` speaker-change markers      (S3-4, S10-39)
  * leading `-`/`–` turn markers     (S5-6, S15-16, S19, S30, S32, S40-50) — two speakers in one cue
  * `<i>…</i>` italics               (off-camera speech / voice-over → confessional hint)
  * sound captions `(gasps)` `[music]` `♪`

We parse SRT ourselves rather than via pysubs2 because we need *per-line* italics and markers,
and pysubs2 normalises tags away.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")  # HTML-ish tags and ASS override blocks
ITALIC_OPEN_RE = re.compile(r"<i\b[^>]*>", re.I)
ITALIC_CLOSE_RE = re.compile(r"</i\s*>", re.I)
# Leading turn markers: ">>", "-", "–", "—", optionally repeated/with spaces.
TURN_RE = re.compile(r"^\s*(?:>>|[-–—])\s*")
# Speaker prefix: ALL-CAPS token(s) followed by a colon. Allows B.B., JEFF PROBST, TK, DR. WILL, MAN #2.
NAME_RE = re.compile(r"^\s*([A-Z][A-Z0-9.'\-#]*(?:\s+[A-Z0-9#][A-Z0-9.'\-#]*){0,2})\s*(?:\([^)]*\))?\s*:\s*(.*)$")
# Whole-line sound captions.
SOUND_RE = re.compile(r"^\s*(?:\([^)]*\)|\[[^\]]*\]|♪+.*♪*|\*[^*]*\*)\s*$")
MUSIC_CHARS = "♪♫"


@dataclass
class Line:
    text: str                  # cleaned speech text ("" for pure sound lines)
    sdh_name: str | None       # NAME token if this line carried a speaker prefix
    is_turn: bool              # line started with >> or a dash
    is_italic: bool
    is_sound: bool             # whole line is a sound caption
    raw: str


@dataclass
class Cue:
    idx: int
    start_s: float
    end_s: float
    raw_text: str
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines if l.text).strip()

    @property
    def is_speech(self) -> bool:
        return any(l.text for l in self.lines)

    @property
    def n_turns(self) -> int:
        return sum(l.is_turn for l in self.lines)

    def to_row(self) -> dict:
        return {
            "idx": self.idx,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "raw_text": self.raw_text,
            "lines": [asdict(l) for l in self.lines],
        }


@dataclass
class Convention:
    n_cues: int
    n_lines: int
    n_name_prefix: int
    n_gtgt: int
    n_dash: int
    n_italic: int
    n_sound: int
    n_multi_turn_cues: int
    names: dict[str, int]
    first_cue_s: float | None
    last_cue_end_s: float | None
    has_name_prefix: bool
    has_gtgt: bool
    has_dash_turns: bool
    has_italics: bool

    def flags(self) -> dict:
        return {
            "has_name_prefix": self.has_name_prefix,
            "has_gtgt": self.has_gtgt,
            "has_dash_turns": self.has_dash_turns,
            "has_italics": self.has_italics,
            "n_cues": self.n_cues,
            "n_name_prefix": self.n_name_prefix,
            "n_multi_turn_cues": self.n_multi_turn_cues,
        }


class SubtitleParseError(Exception):
    pass


def _ts(h: str, m: str, s: str, ms: str) -> float:
    ms = ms.ljust(3, "0")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_line(raw: str, italic_open: bool) -> tuple[Line, bool]:
    """Analyse one raw subtitle line. Returns (Line, italic_open_after)."""
    s = raw.strip()
    # italics: a line is italic if it opens/contains <i>, or we are inside an <i> from a prior line
    opens = bool(ITALIC_OPEN_RE.search(s))
    closes = bool(ITALIC_CLOSE_RE.search(s))
    is_italic = italic_open or opens
    if opens and closes:
        italic_after = italic_open  # self-contained span; state unchanged
    elif opens:
        italic_after = True
    elif closes:
        italic_after = False
    else:
        italic_after = italic_open
    # strip tags
    s = TAG_RE.sub("", s)
    s = s.replace("\u200b", "").replace("\ufeff", "").strip()
    # turn marker
    is_turn = False
    m = TURN_RE.match(s)
    if m and m.end() > 0:
        is_turn = True
        s = s[m.end():]
    # speaker prefix
    sdh_name = None
    m = NAME_RE.match(s)
    if m and len(m.group(1)) >= 2:
        sdh_name = re.sub(r"\s+", " ", m.group(1)).strip(" .")
        s = m.group(2)
    # sound caption
    is_sound = False
    if s and (SOUND_RE.match(s) or (any(c in s for c in MUSIC_CHARS) and len(re.sub(rf"[{MUSIC_CHARS}\s]", "", s)) < 3)):
        is_sound = True
        text = ""
    else:
        # remove inline parenthetical sound cues like "(laughs) I know" -> "I know"
        text = re.sub(r"\((?:[^()]*)\)", "", s)
        text = re.sub(r"\[(?:[^\[\]]*)\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return Line(text=text, sdh_name=sdh_name, is_turn=is_turn, is_italic=is_italic, is_sound=is_sound, raw=raw), italic_after


def parse_srt_text(text: str) -> list[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip("\n﻿ \t"))
    cues: list[Cue] = []
    idx = 0
    for block in blocks:
        lines = [l for l in block.split("\n")]
        # find the timing line (usually line 1; line 0 is the index, which may be missing)
        t_i = next((i for i, l in enumerate(lines[:3]) if TIME_RE.search(l)), None)
        if t_i is None:
            continue
        m = TIME_RE.search(lines[t_i])
        start = _ts(*m.groups()[:4])
        end = _ts(*m.groups()[4:])
        body = [l for l in lines[t_i + 1 :] if l.strip() != ""]
        if not body:
            continue
        italic_open = False
        parsed: list[Line] = []
        for raw in body:
            line, italic_open = _clean_line(raw, italic_open)
            parsed.append(line)
        cues.append(Cue(idx=idx, start_s=start, end_s=end, raw_text="\n".join(body), lines=parsed))
        idx += 1
    if not cues:
        raise SubtitleParseError("no cues parsed")
    return cues


def parse_srt(path: Path) -> list[Cue]:
    return parse_srt_text(read_text(path))


def detect_convention(cues: list[Cue], name_min: int = 20, marker_min: int = 20) -> Convention:
    n_lines = n_name = n_gtgt = n_dash = n_italic = n_sound = n_multi = 0
    names: Counter[str] = Counter()
    for c in cues:
        turns = 0
        for l in c.lines:
            n_lines += 1
            if l.sdh_name:
                n_name += 1
                names[l.sdh_name] += 1
            if l.is_turn:
                turns += 1
                if TAG_RE.sub("", l.raw).lstrip().startswith(">>"):
                    n_gtgt += 1
                else:
                    n_dash += 1
            n_italic += l.is_italic
            n_sound += l.is_sound
        n_multi += turns >= 2
    return Convention(
        n_cues=len(cues),
        n_lines=n_lines,
        n_name_prefix=n_name,
        n_gtgt=n_gtgt,
        n_dash=n_dash,
        n_italic=n_italic,
        n_sound=n_sound,
        n_multi_turn_cues=n_multi,
        names=dict(names.most_common()),
        first_cue_s=cues[0].start_s if cues else None,
        last_cue_end_s=max(c.end_s for c in cues) if cues else None,
        has_name_prefix=n_name >= name_min,
        has_gtgt=n_gtgt >= marker_min,
        has_dash_turns=n_dash >= marker_min,
        has_italics=n_italic >= marker_min,
    )


def quick_stats(path: Path) -> dict:
    """Cheap per-file stats for the inventory chooser."""
    try:
        cues = parse_srt(path)
    except Exception as e:  # noqa: BLE001
        return {"n_cues": 0, "first_cue_s": None, "last_cue_end_s": None, "parse_error": f"{type(e).__name__}: {e}"}
    return {
        "n_cues": len(cues),
        "first_cue_s": cues[0].start_s,
        "last_cue_end_s": max(c.end_s for c in cues),
        "parse_error": None,
    }


def is_sdh_filename(name: str) -> bool:
    n = name.lower()
    return ".hi." in n or "sdh" in n or ".cc." in n


if __name__ == "__main__":  # quick manual check:  python -m survspk.subparse file.srt
    import sys

    p = Path(sys.argv[1])
    cs = parse_srt(p)
    conv = detect_convention(cs)
    print({k: v for k, v in asdict(conv).items() if k != "names"})
    print("names:", list(conv.names.items())[:20])
    for c in cs[40:46]:
        print(c.idx, f"{c.start_s:.2f}-{c.end_s:.2f}", [(l.sdh_name, l.is_turn, l.is_italic, l.is_sound, l.text) for l in c.lines])
