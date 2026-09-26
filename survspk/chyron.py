"""Chyron OCR (spec §7.6 step 3, §8.6): the on-screen name card is one free speaker label per castaway per episode,
in every era, including the seasons whose subtitles carry no names.

Pipeline for one episode:
  sample  -> ffmpeg decodes the episode body, keeps only the chyron band (bottom of frame, bottom-left is the card),
             and hands back one grey frame per 1/fps second; frames that look like the previous kept frame are dropped
  ocr     -> Apple Vision (ocrmac) on the kept frames; a fake backend serves the tests
  match   -> OCR lines against the cast survivoR says is present (rapidfuzz); location cards (TRIBE / DAY n) are
             kept as scenes; the show's own centred captions are ignored by position
  label   -> each hit anchors the utterance that started 2-4 s before the card; that utterance gets a `chyron` label
             unless a human already labelled it or an explicit SDH name disagrees (-> review_queue chyron_conflict)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .aliases import normalize_token
from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class ChyronCfg:
    fps: float = 2.0
    scale_w: int = 960                  # width of the decoded band
    diff_threshold: float = 6.0         # mean |Δ| (0-255) against the last kept frame below which a frame is skipped
    min_match: int = 85                 # rapidfuzz ratio a name line must reach
    dedupe_s: float = 10.0              # one hit per castaway per this many seconds
    anchor_before_s: float = 6.0        # the anchored utterance starts within [t - anchor_before_s, t + anchor_after_s]
    anchor_after_s: float = 1.0
    anchor_ideal_s: float = 3.0         # ... preferring a start this long before the card
    anchor: str = "overlap"             # overlap: the line most on air while the card is up (US47 E01-03: 93% right on
                                        # 97 known lines); start: the line starting ~anchor_ideal_s before it (88%)
    overlap_pad_s: float = 0.5
    check_before_s: float = 8.0         # the card check offers the lines starting within [t - before, t + after]
    check_after_s: float = 4.0
    left_frac: float = 0.62             # OCR lines whose left edge is right of this fraction of the band are captions
    min_upper: float = 0.8              # share of capital letters a card line needs: cards are set in capitals, the
                                        # show's own dialogue captions are not ("Kishan, you got that started?")
    burst_n: int = 4                    # this many different castaways carded within burst_s seconds is the opening
    burst_s: float = 40.0               # credits (every name flashes by), not name cards for speech
    hwaccel: str | None = "videotoolbox"
    backend: str = "apple-vision"       # apple-vision | tesseract | fake

    @classmethod
    def from_settings(cls, s: Settings) -> "ChyronCfg":
        raw = dict(s.raw.get("chyron", {}))
        raw.setdefault("backend", str(s.raw.get("models", {}).get("ocr", "apple-vision")))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class OcrLine:
    text: str
    conf: float
    x0: float                            # left edge, fraction of band width
    y0: float                            # top edge, fraction of band height


@dataclass
class Hit:
    t_s: float
    castaway_id: str
    ocr_text: str
    ocr_conf: float
    match_score: float
    kind: str = "cast"                   # cast | host
    t_end: float | None = None           # last sampled frame that still showed the card


@dataclass
class Scene:
    t_s: float
    tribe: str
    day: int | None
    kind: str                            # day | night


# ----------------------------------------------------------------------------- frames


def sample_band(video: Path, crop_frac: tuple[float, float], t0: float, t1: float, cfg: ChyronCfg,
                timeout: int = 3600) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_s, grey band) for frames whose band changed since the last yielded one."""
    y0, y1 = crop_frac
    vf = f"fps={cfg.fps},crop=iw:ih*{y1 - y0:.3f}:0:ih*{y0:.3f},scale={cfg.scale_w}:-2"
    head = ["ffmpeg", "-v", "error", "-nostdin"]
    if cfg.hwaccel:
        head += ["-hwaccel", cfg.hwaccel]
    # the band's height as ffmpeg makes it: its crop and scale rounding can differ from arithmetic here by a row or two,
    # and reading frames of the wrong size slides every later frame (US47: 150 computed vs 152 made, so the frame
    # clock ran 1.3% fast and a card at 7:42 was dated 7:48). Decode one frame and measure it.
    one = subprocess.run(head + ["-ss", f"{t0:.3f}", "-i", str(video), "-frames:v", "1", "-vf", vf,
                                 "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=120)
    if not one.stdout or len(one.stdout) % cfg.scale_w:
        raise RuntimeError(f"could not measure the band of {video.name}: {one.stderr.decode(errors='replace')[:300]}")
    out_h = len(one.stdout) // cfg.scale_w
    cmd = head + ["-ss", f"{t0:.3f}", "-i", str(video), "-t", f"{max(t1 - t0, 0):.3f}", "-vf", vf,
                  "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    n = cfg.scale_w * out_h
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    last = None
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            fr = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, cfg.scale_w)
            t = t0 + i / cfg.fps
            i += 1
            if last is None or float(np.abs(fr.astype(np.int16) - last.astype(np.int16)).mean()) >= cfg.diff_threshold:
                last = fr
                yield t, fr
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace").strip()
        p.wait(timeout=timeout)
        if p.returncode not in (0, None) and err:
            log.warning("ffmpeg: %s", err[:300])


# ----------------------------------------------------------------------------- OCR backends


def ocr_backend(name: str, fake: dict | None = None) -> Callable[[np.ndarray], list[OcrLine]]:
    """A function grey band -> OCR lines. `fake` maps a frame's key (see fake_key) to lines, for tests."""
    if name == "fake":
        fake = fake or {}
        return lambda fr: list(fake.get(fake_key(fr), []))
    if name == "apple-vision":
        try:
            from ocrmac import ocrmac  # type: ignore
        except ImportError as e:
            raise ImportError("ocrmac is needed for Apple Vision OCR: uv sync --all-extras (macOS only)") from e
        from PIL import Image

        def run(fr: np.ndarray) -> list[OcrLine]:
            img = Image.fromarray(fr)
            out = []
            for text, conf, (x, y, w, h) in ocrmac.OCR(img, recognition_level="accurate").recognize():
                out.append(OcrLine(text, float(conf), float(x), float(1.0 - y - h)))   # Vision: origin bottom-left
            return out
        return run
    if name == "tesseract":
        import pytesseract  # type: ignore
        from PIL import Image

        def run_t(fr: np.ndarray) -> list[OcrLine]:
            d = pytesseract.image_to_data(Image.fromarray(fr), output_type=pytesseract.Output.DICT)
            H, W = fr.shape
            lines: dict[tuple, list] = {}
            for i, txt in enumerate(d["text"]):
                if not txt.strip():
                    continue
                key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                lines.setdefault(key, []).append(i)
            out = []
            for idxs in lines.values():
                words = [d["text"][i] for i in idxs]
                conf = float(np.mean([float(d["conf"][i]) for i in idxs])) / 100.0
                out.append(OcrLine(" ".join(words), conf, min(d["left"][i] for i in idxs) / W, min(d["top"][i] for i in idxs) / H))
            return out
        return run_t
    raise ValueError(f"unknown OCR backend {name!r}")


def fake_key(fr: np.ndarray) -> int:
    """Frames in the tests are flat grey images; their level is the key."""
    return int(fr.flat[0])


# ----------------------------------------------------------------------------- matching

LOCATION_RE = re.compile(r"^(?P<tribe>[A-Z][A-Z' .-]{1,30}?)\s*(?:TRIBE)?\s*[/|]?\s*(?P<kind>DAY|NIGHT)\s*(?P<n>\d{1,2})\b")


def parse_location(lines: list[OcrLine]) -> Scene | None:
    """`MATSING TRIBE / DAY 11` as one or two OCR lines -> Scene."""
    txt = " / ".join(normalize_token(ln.text) for ln in lines if ln.text.strip())
    m = LOCATION_RE.search(txt)
    if m:
        return Scene(0.0, m.group("tribe").replace(" TRIBE", "").strip(), int(m.group("n")), m.group("kind").lower())
    return None


class CastMatcher:
    """OCR text -> castaway_id for one episode's cast, with occupation as the tiebreaker."""

    def __init__(self, resolver, vs: str, ep: int, cfg: ChyronCfg):
        from rapidfuzz import fuzz, process

        self.fuzz, self.process = fuzz, process
        self.cfg = cfg
        self.vs, self.ep = vs, ep
        self.bios = resolver.bios(vs, ep)
        present = resolver.present(vs, ep)
        self.names: dict[str, str] = {}                 # normalized token -> castaway_id
        for cid, b in self.bios.items():
            if present and cid not in present:
                continue
            for nm in (b.get("name"), (b.get("full_name") or "").split(" ")[0]):
                t = normalize_token(nm or "")
                if len(t) >= 2:
                    self.names[t] = cid
        for tok, cid in getattr(resolver, "season_aliases", {}).get(vs, {}).items():
            if cid in self.bios and len(tok) >= 3 and (not present or cid in present):
                self.names[tok] = cid
        hosts, self.host_id = resolver.host_names(vs)
        self.hosts = hosts

    def _best(self, tok: str) -> tuple[str, float] | None:
        """Closest cast name: ratio >= min_match, or one character off on a name of four letters or more
        (short names never reach 85 with a single OCR slip)."""
        from rapidfuzz.distance import Levenshtein

        m = self.process.extractOne(tok, list(self.names), scorer=self.fuzz.ratio, score_cutoff=self.cfg.min_match)
        if m is not None:
            return m[0], float(m[1])
        if len(tok) >= 4:
            near = [(n, Levenshtein.distance(tok, n)) for n in self.names if abs(len(n) - len(tok)) <= 1]
            near = [(n, d) for n, d in near if d <= 1]
            if len(near) == 1:
                return near[0][0], float(self.cfg.min_match)
        return None

    def match(self, lines: list[OcrLine]) -> tuple[str, float, str] | None:
        """(castaway_id, score, matched text) for the best name line, or None."""
        best = None
        for ln in lines:
            t = normalize_token(ln.text)
            if not t:
                continue
            head = t.split(" ")[0]
            if (head in self.hosts or t in self.hosts) and head not in self.names:
                # the host never gets a name card; a castaway who shares his name does (US31 Jeff Varner:
                # "JEFF THE AUSTRALIAN OUTBACK")
                cand = (self.host_id, 100.0, ln.text)
            else:
                # the card's first line is NAME [occupation...]; try the first token, then the first two
                m = self._best(head)
                if m is None and " " in t:
                    m = self._best(" ".join(t.split(" ")[:2]))
                if m is None:
                    continue
                cid, score = self.names[m[0]], float(m[1])
                # occupation on the same line lifts a shaky name match
                occ = normalize_token(self.bios.get(cid, {}).get("occupation") or "")
                if occ and score < 100 and self.fuzz.partial_ratio(occ, t) >= 80:
                    score = min(100.0, score + 5)
                cand = (cid, score, ln.text)
            if best is None or cand[1] > best[1]:
                best = cand
        return best


def upper_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0


def frame_hits(lines: list[OcrLine], matcher: CastMatcher, cfg: ChyronCfg) -> tuple[tuple[str, float, str] | None, Scene | None]:
    """Split one frame's OCR into the card region (left) and captions (centre), match a name or a location card.
    Name cards are set in capitals; a mixed-case line on the left is the show's own caption, not a card."""
    card = [ln for ln in lines if ln.x0 <= cfg.left_frac]
    if not card:
        return None, None
    scene = parse_location(card)
    if scene is not None:
        return None, scene
    card = [ln for ln in card if upper_share(ln.text) >= cfg.min_upper]
    if not card:
        return None, None
    return matcher.match(card), None


def drop_bursts(hits: list[Hit], n: int, within_s: float) -> tuple[list[Hit], list[Hit]]:
    """(kept, dropped): hits inside a window of `within_s` that cards `n` or more different castaways are the opening
    credits or a cast montage, where names flash by with nobody speaking."""
    hs = sorted(hits, key=lambda h: h.t_s)
    bad: set[int] = set()
    for i, h in enumerate(hs):
        win = [j for j in range(len(hs)) if 0 <= hs[j].t_s - h.t_s <= within_s]
        if len({hs[j].castaway_id for j in win}) >= n:
            bad.update(win)
    return [h for i, h in enumerate(hs) if i not in bad], [h for i, h in enumerate(hs) if i in bad]


def dedupe(hits: list[Hit], within_s: float) -> list[Hit]:
    out: list[Hit] = []
    for h in sorted(hits, key=lambda h: h.t_s):
        prev = next((o for o in reversed(out) if o.castaway_id == h.castaway_id), None)
        if prev is not None and h.t_s - prev.t_s <= within_s:
            if h.match_score > prev.match_score:
                prev.match_score, prev.ocr_text, prev.ocr_conf = h.match_score, h.ocr_text, h.ocr_conf
            prev.t_end = max(prev.t_end or prev.t_s, h.t_end or h.t_s)
            continue
        out.append(Hit(h.t_s, h.castaway_id, h.ocr_text, h.ocr_conf, h.match_score, h.kind, h.t_end or h.t_s))
    return out


def dedupe_scenes(scenes: list[Scene], within_s: float = 30.0) -> list[Scene]:
    out: list[Scene] = []
    for sc in sorted(scenes, key=lambda x: x.t_s):
        if out and out[-1].tribe == sc.tribe and out[-1].day == sc.day and sc.t_s - out[-1].t_s <= within_s:
            continue
        out.append(sc)
    return out


# ----------------------------------------------------------------------------- anchoring + labels


def anchor_utterance(utts: list[dict], t: float, cfg: ChyronCfg, t_end: float | None = None) -> dict | None:
    """The body utterance the card belongs to. `overlap` (default): the line most on air while the card is up
    ([t, t_end] padded), falling back to the start rule when no line overlaps; `start`: the line that started within
    the window, closest to `anchor_ideal_s` before t."""
    if cfg.anchor == "overlap":
        a, b = t - cfg.overlap_pad_s, (t_end if t_end is not None else t) + cfg.overlap_pad_s
        on = [(min(b, u["end_s"]) - max(a, u["start_s"]), u) for u in utts
              if u.get("segment", "body") == "body" and "end_s" in u]
        on = [x for x in on if x[0] > 0]
        if on:
            return max(on, key=lambda x: (round(x[0], 3), -abs((t - x[1]["start_s"]) - cfg.anchor_ideal_s)))[1]
    lo, hi = t - cfg.anchor_before_s, t + cfg.anchor_after_s
    cands = [u for u in utts if lo <= u["start_s"] <= hi and u.get("segment", "body") == "body"]
    if not cands:
        return None
    return min(cands, key=lambda u: abs((t - u["start_s"]) - cfg.anchor_ideal_s))


@dataclass
class LabelOutcome:
    n_labels: int = 0
    n_conflicts: int = 0
    n_no_anchor: int = 0
    n_kept_human: int = 0
    n_checked: int = 0                   # hits settled by a card check
    agreement: list[tuple[str, str, str, float]] = field(default_factory=list)   # (utt_id, chyron, sdh, t)


def write_hits(con: sqlite3.Connection, vs: str, ep: int, hits: list[Hit], scenes: list[Scene]) -> None:
    con.executescript("""CREATE TABLE IF NOT EXISTS scenes (
        version_season TEXT NOT NULL, episode INTEGER NOT NULL, t_s REAL NOT NULL, tribe TEXT, day INTEGER, kind TEXT,
        PRIMARY KEY (version_season, episode, t_s))""")
    con.execute("DELETE FROM chyron_hits WHERE version_season=? AND episode=?", (vs, ep))
    con.execute("DELETE FROM scenes WHERE version_season=? AND episode=?", (vs, ep))
    con.executemany("INSERT OR REPLACE INTO chyron_hits (version_season, episode, t_s, ocr_text, castaway_id, ocr_conf, match_score, t_end_s) VALUES (?,?,?,?,?,?,?,?)",
                    [(vs, ep, round(h.t_s, 2), h.ocr_text, h.castaway_id, h.ocr_conf, h.match_score,
                      round(h.t_end, 2) if h.t_end is not None else None) for h in hits])
    con.executemany("INSERT OR REPLACE INTO scenes (version_season, episode, t_s, tribe, day, kind) VALUES (?,?,?,?,?,?)",
                    [(vs, ep, round(s.t_s, 2), s.tribe, s.day, s.kind) for s in scenes])
    con.commit()


CHECK_MATCH_S = 2.0                     # a card check applies to a hit of the same castaway within this many seconds


def card_checks(con: sqlite3.Connection, vs: str, ep: int) -> list[dict]:
    """The episode's card checks, each with `utt_ids`: the lines picked (empty = none of the lines)."""
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT t_s, castaway_id, utt_id, utt_ids FROM card_checks WHERE version_season=? AND episode=?", (vs, ep))]
    except sqlite3.OperationalError:          # a DB from before the card check, or before several lines per card
        try:
            rows = [dict(r) for r in con.execute(
                "SELECT t_s, castaway_id, utt_id, NULL AS utt_ids FROM card_checks WHERE version_season=? AND episode=?", (vs, ep))]
        except sqlite3.OperationalError:
            return []
    for r in rows:
        ids = json.loads(r["utt_ids"]) if r["utt_ids"] else None
        r["utt_ids"] = ids if ids is not None else ([r["utt_id"]] if r["utt_id"] else [])
    return rows


def check_for(checks: list[dict], t_s: float, castaway_id: str) -> dict | None:
    near = [c for c in checks if c["castaway_id"] == castaway_id and abs(c["t_s"] - t_s) <= CHECK_MATCH_S]
    return min(near, key=lambda c: abs(c["t_s"] - t_s)) if near else None


def card_lines(con: sqlite3.Connection, vs: str, ep: int, t_s: float, cfg: ChyronCfg) -> list[dict]:
    """The body lines a card could belong to, in time order (what the card check shows): every line that is on air
    at some point in [t - check_before_s, t + check_after_s]. A long confessional line that started earlier and is
    still going when the card appears counts (US47 E01's Aysha card: her line started 8.2 s before it)."""
    return [dict(r) for r in con.execute(
        """SELECT utt_id, start_s, end_s, text, sdh_speaker_id, sdh_resolution, domain_hint FROM utterances
           WHERE version_season=? AND episode=? AND segment='body' AND end_s >= ? AND start_s <= ? ORDER BY start_s""",
        (vs, ep, t_s - cfg.check_before_s, t_s + cfg.check_after_s))]


def apply_labels(con: sqlite3.Connection, vs: str, ep: int, hits: list[Hit], cfg: ChyronCfg, write: bool = True) -> LabelOutcome:
    """Chyron labels for the anchored utterances. Human labels win; an explicit SDH name that disagrees is queued as
    chyron_conflict and the SDH label stays. A card someone checked in the review UI (card_checks) labels the line
    they picked, or nothing when they said the castaway speaks none of the lines. Every hit that lands on an
    SDH-named utterance (explicit or inherited) is recorded in `agreement` — the free precision check on named
    episodes."""
    utts = [dict(r) for r in con.execute(
        """SELECT utt_id, start_s, end_s, text, segment, sdh_speaker_id, sdh_resolution, flags, domain_hint
           FROM utterances WHERE version_season=? AND episode=? ORDER BY idx""", (vs, ep))]
    for u in utts:
        u["name_explicit"] = not json.loads(u["flags"] or "{}").get("name_inherited", True)
    labels = {r["utt_id"]: dict(r) for r in con.execute(
        "SELECT l.utt_id, l.speaker_id, l.source FROM labels l JOIN utterances u USING (utt_id) WHERE u.version_season=? AND u.episode=?", (vs, ep))}
    checks = card_checks(con, vs, ep)
    by_id = {u["utt_id"]: u for u in utts}
    out = LabelOutcome()
    if write:
        con.execute("""DELETE FROM labels WHERE source='chyron' AND utt_id IN
                       (SELECT utt_id FROM utterances WHERE version_season=? AND episode=?)""", (vs, ep))
        con.execute("""DELETE FROM review_queue WHERE reason='chyron_conflict' AND resolved=0 AND utt_id IN
                       (SELECT utt_id FROM utterances WHERE version_season=? AND episode=?)""", (vs, ep))
    for h in hits:
        chk = check_for(checks, h.t_s, h.castaway_id)
        if chk is not None:
            out.n_checked += 1
            for uid in chk["utt_ids"]:
                u = by_id.get(uid)
                if u is None:
                    continue
                prev = labels.get(u["utt_id"])
                if prev and prev["source"] == "human":
                    out.n_kept_human += 1
                    continue
                out.n_labels += 1
                if write:
                    con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at,
                                                                  version_season, episode, start_s, end_s, text)
                                   VALUES (?,?,'chyron',1.0,?,?,datetime('now'),?,?,?,?,?)""",
                                (u["utt_id"], h.castaway_id, json.dumps({"t_s": round(h.t_s, 2), "ocr": h.ocr_text, "checked": True}),
                                 u["domain_hint"], vs, ep, u["start_s"], u["end_s"], u["text"]))
            continue
        u = anchor_utterance(utts, h.t_s, cfg, h.t_end)
        if u is None:
            out.n_no_anchor += 1
            continue
        sdh_ok = u["sdh_speaker_id"] and u["sdh_resolution"] in ("cast", "alias", "host")
        if sdh_ok:
            out.agreement.append((u["utt_id"], h.castaway_id, u["sdh_speaker_id"], h.t_s))
        prev = labels.get(u["utt_id"])
        if prev and prev["source"] == "human":
            out.n_kept_human += 1
            continue
        if sdh_ok and u["name_explicit"] and u["sdh_speaker_id"] != h.castaway_id:
            out.n_conflicts += 1
            if write:
                con.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                            (u["utt_id"], "chyron_conflict", json.dumps({"chyron": h.castaway_id, "sdh": u["sdh_speaker_id"],
                                                                           "t_s": round(h.t_s, 2), "ocr": h.ocr_text})))
            continue
        out.n_labels += 1
        if write:
            con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at,
                                                          version_season, episode, start_s, end_s, text)
                           VALUES (?,?,'chyron',?,?,?,datetime('now'),?,?,?,?,?)""",
                        (u["utt_id"], h.castaway_id, h.match_score / 100.0, json.dumps({"t_s": round(h.t_s, 2), "ocr": h.ocr_text}),
                         u["domain_hint"], vs, ep, u["start_s"], u["end_s"], u["text"]))
    if write:
        con.commit()
    return out


# ----------------------------------------------------------------------------- the stage


def ocr_cache_path(settings: Settings, vs: str, ep: int) -> Path:
    """The raw OCR of one episode's band, one JSON line per OCR'd frame: [t_s, [[text, conf, x0, y0], ...]].
    Short on-screen text only; it lives under work_root/reports and is never published."""
    return Path(settings.paths.work_root) / "reports" / "chyron" / f"{vs}_E{ep:02d}_ocr.jsonl"


def read_ocr_cache(path: Path) -> Iterator[tuple[float, list[OcrLine]]]:
    with open(path) as f:
        for line in f:
            t, lines = json.loads(line)
            yield float(t), [OcrLine(str(a), float(b), float(c), float(d)) for a, b, c, d in lines]


def chyron_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, resolver, video: Path | None = None,
                   frames: Iterator[tuple[float, np.ndarray]] | None = None, ocr: Callable | None = None,
                   write_labels: bool = True, cfg: ChyronCfg | None = None, use_cache: bool = False,
                   cache_path: Path | None = None) -> dict:
    """OCR the chyron band of an episode, store hits + scenes, label the anchored utterances. `frames` / `ocr`
    override sampling and the OCR backend (tests). Every run from video writes the raw OCR to the cache;
    `use_cache` re-reads it instead of decoding and OCR'ing the video again (seconds instead of minutes), for
    tuning the matching and anchoring."""
    t0 = time.time()
    cfg = cfg or ChyronCfg.from_settings(settings)
    epi = con.execute("SELECT video_path, duration_s, recap_end_s, preview_start_s FROM episodes WHERE version_season=? AND episode=?",
                      (vs, ep)).fetchone()
    if epi is None:
        raise LookupError(f"{vs} E{ep:02d} not in inventory")
    cache = cache_path or (ocr_cache_path(settings, vs, ep) if frames is None else None)
    cached = use_cache and cache is not None and cache.exists()
    if use_cache and not cached:
        raise FileNotFoundError(f"no OCR cache for {vs} E{ep:02d} at {cache}: run once without --from-cache")
    if frames is None and not cached:
        from . import db as dbm
        video = video or dbm.localize(con, settings, epi["video_path"])
        if video is None or not Path(video).exists():
            raise FileNotFoundError(f"video for {vs} E{ep:02d} not reachable: {video}")
        crop = tuple(settings.franchise_for(vs).chyron_crop)
        a = float(epi["recap_end_s"] or 0.0)
        b = float(epi["preview_start_s"] or epi["duration_s"] or 0.0)
        frames = sample_band(Path(video), crop, a, b, cfg)
    matcher = CastMatcher(resolver, vs, ep, cfg)
    raw_hits: list[Hit] = []
    scenes: list[Scene] = []
    n_frames = n_lines = 0
    if cached:
        frame_lines: Iterator[tuple[float, list[OcrLine]]] = read_ocr_cache(cache)
        sink = None
    else:
        ocr = ocr or ocr_backend(cfg.backend)
        frame_lines = ((t, ocr(fr)) for t, fr in frames)
        sink = None
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            sink = open(cache.with_suffix(".tmp"), "w")
    for t, lines in frame_lines:
        n_frames += 1
        n_lines += len(lines)
        if sink is not None and lines:
            sink.write(json.dumps([round(t, 3), [[ln.text, round(ln.conf, 3), round(ln.x0, 4), round(ln.y0, 4)] for ln in lines]]) + "\n")
        m, scene = frame_hits(lines, matcher, cfg)
        if scene is not None:
            scene.t_s = t
            scenes.append(scene)
        if m is not None:
            cid, score, text = m
            conf = max((ln.conf for ln in lines if ln.text == text), default=0.0)
            raw_hits.append(Hit(t, cid, text, conf, score, "host" if cid == matcher.host_id else "cast"))
    if sink is not None:
        sink.close()
        cache.with_suffix(".tmp").replace(cache)
    hits, burst = drop_bursts(dedupe(raw_hits, cfg.dedupe_s), cfg.burst_n, cfg.burst_s)
    scenes = dedupe_scenes(scenes)
    write_hits(con, vs, ep, hits, scenes)
    lab = apply_labels(con, vs, ep, hits, cfg, write=write_labels)
    agree = [a for a in lab.agreement]
    n_agree = sum(c == s for _, c, s, _ in agree)
    stats = {
        "version_season": vs, "episode": ep, "n_frames_ocr": n_frames, "n_ocr_lines": n_lines,
        "n_raw_hits": len(raw_hits), "n_hits": len(hits), "n_burst_dropped": len(burst), "n_scenes": len(scenes),
        "from_cache": cached,
        "castaways_hit": sorted({h.castaway_id for h in hits if h.kind == "cast"}),
        "n_labels": lab.n_labels, "n_conflicts": lab.n_conflicts, "n_no_anchor": lab.n_no_anchor, "n_kept_human": lab.n_kept_human,
        "n_checked": lab.n_checked,
        "n_sdh_compared": len(agree), "sdh_agreement": round(n_agree / len(agree), 3) if agree else None,
        "disagreements": [(uid, c, s, round(t, 1)) for uid, c, s, t in agree if c != s],
        "seconds": round(time.time() - t0, 1),
    }
    for k in ("n_hits", "n_scenes", "n_labels", "n_conflicts", "sdh_agreement", "n_sdh_compared"):
        if stats[k] is not None:
            con.execute("INSERT OR REPLACE INTO metrics (version_season, episode, key, value, payload, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
                        (vs, ep, f"chyron_{k}", float(stats[k]), None))
    con.commit()
    log.info("%s E%02d chyron: %d frames OCR'd, %d hits (%d castaways), %d labels, %d conflicts; SDH agreement %s on %d",
             vs, ep, n_frames, len(hits), len(stats["castaways_hit"]), lab.n_labels, lab.n_conflicts,
             stats["sdh_agreement"], len(agree))
    return stats


# ----------------------------------------------------------------------------- M0 helpers (frame checks)


def grab_frame(video: Path, t_s: float, out_png: Path, crop_frac: tuple[float, float] | None = None,
               scale_w: int = 960, timeout: int = 120) -> Path:
    """Grab one frame at t_s. If crop_frac=(y0,y1) fractions of height are given, crop that band."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"scale={scale_w}:-2"]
    if crop_frac:
        y0, y1 = crop_frac
        vf.append(f"crop=iw:ih*{y1 - y0:.3f}:0:ih*{y0:.3f}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t_s:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", ",".join(vf), str(out_png)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return out_png


def grab_contact_sheet(video: Path, times: list[float], out_png: Path, scale_w: int = 480, timeout: int = 300) -> Path:
    """Several frames tiled into one image (one row per time) — quick visual check of chyron timing."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = []
    for i, t in enumerate(times):
        p = out_png.with_suffix(f".{i}.png")
        grab_frame(video, t, p, scale_w=scale_w, timeout=timeout)
        tmp.append(p)
    inputs = sum([["-i", str(p)] for p in tmp], [])
    n = len(tmp)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"vstack=inputs={n}[v]"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[v]", str(out_png)],
                       capture_output=True, text=True, timeout=timeout)
    for p in tmp:
        p.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return out_png
