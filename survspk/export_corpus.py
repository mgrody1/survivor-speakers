"""Training-corpus export: the labelled episodes as (a) a pyannote diarization database and (b) a speaker-embedding
manifest with verification trials.

What counts as a trusted label (per utterance, body segment only):
  * human labels, always (pseudo ids: OTHER kept as one anonymous speaker per file; NOSPEECH kept as annotated
    non-speech; UNKNOWN excluded from the annotated regions altogether);
  * explicit SDH names and the rest of their run, and confident `auto` labels, *after* the bank's self-consistency
    filter — i.e. exactly the set the speaker bank itself trusts (stage_bank.collect_labelled + filter);
  * never the LLM text prior.

Formats:
  rttm/<uri>.rttm   SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker_id> <NA> <NA>
  uem/<uri>.uem     <uri> 1 <start> <end>   — the regions that are ANNOTATED. Only trusted spans (+margin) and
                    gaps between them in which the VAD hears no speech. Unsubtitled speech, overlaps we cannot
                    see, and UNKNOWN spans are simply absent, so the trainer takes no loss there.
  database.yml, lists/{train,dev,test}.lst  — pyannote.database protocol "Survivor.SpeakerDiarization.All"
  spk/{train,dev}.csv                        — id, wav, start_s, end_s, duration, speaker_id, season, episode
  spk/trials_dev.txt                         — "<0|1> <enrol_id> <test_id>" cosine-verification trials (dev only)
  stats.json
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)

PSEUDO_SPEECH = {"OTHER"}            # anonymous but real speech
PSEUDO_NONSPEECH = {"NOSPEECH"}
PSEUDO_EXCLUDE = {"UNKNOWN"}


@dataclass
class EpisodeExport:
    vs: str
    ep: int
    uri: str
    audio: Path
    rttm: list[tuple[float, float, str]] = field(default_factory=list)   # start, dur, speaker
    uem: list[tuple[float, float]] = field(default_factory=list)
    spk_rows: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- trusted labels


def trusted_labels(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str,
                   resolver=None) -> pd.DataFrame:
    """utt_id, start_s, end_s, dur, speaker_id, source, domain for every utterance we would train on."""
    from .stage_bank import collect_labelled, self_consistency_filter

    human = pd.read_sql_query(
        """SELECT l.utt_id, l.speaker_id, u.start_s, u.end_s, u.end_s - u.start_s AS dur, u.domain_hint AS domain
           FROM labels l JOIN utterances u USING (utt_id)
           WHERE l.source='human' AND u.version_season=? AND u.episode=? AND u.segment='body'""", con, params=(vs, ep))
    human["source"] = "human"
    try:
        auto = collect_labelled(settings, con, vs, [ep], variant, resolver=resolver)
    except (FileNotFoundError, RuntimeError) as e:          # no / stale embeddings: human labels only
        log.warning("%s E%02d: %s -> exporting human labels only", vs, ep, e)
        auto = pd.DataFrame()
    if not auto.empty:
        auto = auto[~auto.self_mention] if "self_mention" in auto else auto
        auto = auto[auto.label_source != "human"]
        kept, _ = self_consistency_filter(auto)
        kept = kept[~kept.utt_id.isin(human.utt_id)]
        kept = kept.rename(columns={"label_source": "source", "domain_hint": "domain"})[
            ["utt_id", "speaker_id", "start_s", "end_s", "dur", "domain", "source"]]
        out = pd.concat([human, kept], ignore_index=True)
    else:
        out = human
    return out.sort_values("start_s").reset_index(drop=True)


# ----------------------------------------------------------------------------- UEM construction


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def annotated_regions(spans: list[tuple[float, float]], total_s: float, speech_mask: np.ndarray | None,
                      hop_s: float, margin_s: float = 0.2, max_gap_s: float = 30.0,
                      max_gap_speech: float = 0.05) -> list[tuple[float, float]]:
    """Trusted spans (+margin), plus the silences between consecutive spans: a gap is annotated non-speech when the
    VAD hears (almost) no speech in it. Without a VAD mask only the spans themselves are annotated."""
    regs = [(max(0.0, a - margin_s), min(total_s, b + margin_s)) for a, b in spans]
    regs = _merge(regs)
    if speech_mask is None or len(regs) < 2:
        return regs
    filled: list[tuple[float, float]] = [regs[0]]
    for a, b in regs[1:]:
        pa, pb = filled[-1]
        gap = a - pb
        if 0 < gap <= max_gap_s:
            i0, i1 = int(pb / hop_s), int(a / hop_s)
            frac = float(speech_mask[i0:i1].mean()) if i1 > i0 else 0.0
            if frac <= max_gap_speech:
                filled[-1] = (pa, b)               # bridge the silence
                continue
        filled.append((a, b))
    return filled


# ----------------------------------------------------------------------------- per episode


def export_episode(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, variant: str = "raw",
                   resolver=None, use_vad: bool = True, min_spk_utt_s: float = 1.0) -> EpisodeExport:
    from .stage_align import HOP_S, load_audio, vad_envelope

    audio_path = settings.audio_path(variant, vs, ep)
    vocals = settings.audio_path("vocals", vs, ep)
    ex = EpisodeExport(vs, ep, f"{vs}_E{ep:02d}", audio_path)
    lab = trusted_labels(settings, con, vs, ep, settings.audio.variant, resolver)
    if lab.empty:
        ex.stats = {"n_labels": 0}
        return ex
    total = float(con.execute("SELECT MAX(end_s) FROM utterances WHERE version_season=? AND episode=?", (vs, ep)).fetchone()[0] or 0)
    mask = None
    if use_vad and vocals.exists():
        try:
            audio = load_audio(vocals, settings.audio.sample_rate)
            total = max(total, len(audio) / settings.audio.sample_rate)
            mask = vad_envelope(audio, settings.audio.sample_rate, HOP_S)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: VAD unavailable (%s); UEM = labelled spans only", ex.uri, e)
    speech_spans = []
    for r in lab.itertuples():
        sid = r.speaker_id
        if sid in PSEUDO_EXCLUDE:
            continue
        if sid in PSEUDO_NONSPEECH:
            speech_spans.append((r.start_s, r.end_s))          # annotated, but no SPEAKER line
            continue
        ex.rttm.append((r.start_s, r.end_s - r.start_s, sid))
        speech_spans.append((r.start_s, r.end_s))
        if sid not in PSEUDO_SPEECH and r.dur >= min_spk_utt_s:
            ex.spk_rows.append({"id": r.utt_id, "wav": str(audio_path), "start_s": round(r.start_s, 3),
                                "end_s": round(r.end_s, 3), "duration": round(r.dur, 3), "speaker_id": sid,
                                "season": vs, "episode": ep, "source": r.source, "domain": r.domain})
    ex.uem = annotated_regions(speech_spans, total, mask, HOP_S)
    spk = lab[~lab.speaker_id.isin(PSEUDO_EXCLUDE | PSEUDO_NONSPEECH | PSEUDO_SPEECH)]
    ex.stats = {
        "n_labels": int(len(lab)), "by_source": lab.source.value_counts().to_dict(),
        "speech_s": round(float(sum(d for _, d, _ in ex.rttm)), 1),
        "annotated_s": round(float(sum(b - a for a, b in ex.uem)), 1), "total_s": round(total, 1),
        "n_speakers": int(spk.speaker_id.nunique()), "spk_rows": len(ex.spk_rows),
        "per_speaker_s": spk.groupby("speaker_id").dur.sum().round(1).to_dict(),
    }
    return ex


def write_episode(ex: EpisodeExport, out: Path) -> None:
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    (out / "uem").mkdir(parents=True, exist_ok=True)
    with open(out / "rttm" / f"{ex.uri}.rttm", "w") as f:
        for s, d, spk in sorted(ex.rttm):
            f.write(f"SPEAKER {ex.uri} 1 {s:.3f} {d:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    with open(out / "uem" / f"{ex.uri}.uem", "w") as f:
        for a, b in ex.uem:
            f.write(f"{ex.uri} 1 {a:.3f} {b:.3f}\n")


# ----------------------------------------------------------------------------- corpus


def labelled_episodes(con: sqlite3.Connection, seasons: list[str] | None = None) -> list[tuple[str, int]]:
    q = """SELECT DISTINCT u.version_season, u.episode FROM labels l JOIN utterances u USING (utt_id)
           WHERE l.source IN ('human','auto','sdh')"""
    args: tuple = ()
    if seasons:
        q += f" AND u.version_season IN ({','.join('?' * len(seasons))})"
        args = tuple(seasons)
    return [(r[0], int(r[1])) for r in con.execute(q + " ORDER BY 1, 2", args)]


def make_trials(dev: pd.DataFrame, n_target: int = 2000, n_nontarget: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cosine-verification trials over dev utterances: target pairs = same speaker, different utterance;
    non-target pairs = different speakers from the SAME season (hard negatives: same acoustic conditions)."""
    rng = random.Random(seed)
    rows = []
    by_spk = {s: g.id.tolist() for s, g in dev.groupby("speaker_id") if len(g) >= 2}
    spks = list(by_spk)
    if len(spks) < 2:
        return pd.DataFrame(columns=["label", "enrol", "test"])
    season_of = dict(zip(dev.id, dev.season))
    for _ in range(n_target):
        s = rng.choice(spks)
        a, b = rng.sample(by_spk[s], 2)
        rows.append((1, a, b))
    by_season = {vs: g for vs, g in dev.groupby("season")}
    tries = 0
    while len(rows) < n_target + n_nontarget and tries < 20 * n_nontarget:
        tries += 1
        vs = rng.choice(list(by_season))
        g = by_season[vs]
        if g.speaker_id.nunique() < 2:
            continue
        a = g.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
        b = g[g.speaker_id != a.speaker_id].sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
        rows.append((0, a.id, b.id))
    return pd.DataFrame(rows, columns=["label", "enrol", "test"])


def export_corpus(settings: Settings, con: sqlite3.Connection, out: Path, seasons: list[str] | None = None,
                  variant: str = "raw", resolver=None, test_seasons: list[str] | None = None,
                  dev_last_episode: bool = True, use_vad: bool = True) -> dict:
    """Write the whole corpus. Splits: episodes of `test_seasons` -> test; otherwise the last labelled episode of
    each season -> dev (held-out episodes, seen speakers: what the assign loop faces); the rest -> train."""
    out = Path(out)
    eps = labelled_episodes(con, seasons)
    if not eps:
        raise LookupError("no labelled episodes")
    test_seasons = set(test_seasons or [])
    by_season: dict[str, list[int]] = {}
    for vs, ep in eps:
        by_season.setdefault(vs, []).append(ep)
    split: dict[str, str] = {}
    for vs, lst in by_season.items():
        for ep in lst:
            uri = f"{vs}_E{ep:02d}"
            if vs in test_seasons:
                split[uri] = "test"
            elif dev_last_episode and ep == max(lst) and len(lst) > 1:
                split[uri] = "dev"
            else:
                split[uri] = "train"
    exports: list[EpisodeExport] = []
    spk_rows: list[dict] = []
    for vs, ep in eps:
        ex = export_episode(settings, con, vs, ep, variant=variant, resolver=resolver, use_vad=use_vad)
        if not ex.rttm and not ex.uem:
            log.info("%s: nothing trusted; skipped", ex.uri)
            continue
        write_episode(ex, out)
        exports.append(ex)
        for r in ex.spk_rows:
            r["split"] = split[ex.uri]
            r["uri"] = ex.uri
        spk_rows.extend(ex.spk_rows)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        with open(out / "lists" / f"{name}.lst", "w") as f:
            for ex in exports:
                if split[ex.uri] == name:
                    f.write(ex.uri + "\n")
    _write_database_yml(out, exports, variant)
    spk = pd.DataFrame(spk_rows)
    (out / "spk").mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame()
    if not spk.empty:
        for name in ("train", "dev", "test"):
            spk[spk.split == name].drop(columns=["split"]).to_csv(out / "spk" / f"{name}.csv", index=False)
        dev = spk[spk.split == "dev"]
        trials = make_trials(dev) if len(dev) else pd.DataFrame(columns=["label", "enrol", "test"])
        trials.to_csv(out / "spk" / "trials_dev.txt", sep=" ", header=False, index=False)
    stats = {
        "episodes": {ex.uri: {**ex.stats, "split": split[ex.uri]} for ex in exports},
        "n_episodes": len(exports), "splits": {k: sum(1 for v in split.values() if v == k) for k in ("train", "dev", "test")},
        "speech_hours": round(sum(ex.stats.get("speech_s", 0) for ex in exports) / 3600, 2),
        "annotated_hours": round(sum(ex.stats.get("annotated_s", 0) for ex in exports) / 3600, 2),
        "n_speakers": int(spk.speaker_id.nunique()) if not spk.empty else 0,
        "spk_utts": {k: int((spk.split == k).sum()) for k in ("train", "dev", "test")} if not spk.empty else {},
        "spk_hours": round(float(spk.duration.sum()) / 3600, 2) if not spk.empty else 0.0,
        "n_trials_dev": int(len(trials)), "variant": variant,
        "per_speaker_s": spk.groupby("speaker_id").duration.sum().round(0).sort_values(ascending=False).to_dict() if not spk.empty else {},
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=1, default=str)
    log.info("corpus -> %s: %d episodes, %.2f h speech, %d speakers, %d spk utts",
             out, len(exports), stats["speech_hours"], stats["n_speakers"], len(spk))
    return stats


def _write_database_yml(out: Path, exports: list[EpisodeExport], variant: str) -> None:
    """pyannote.database resolves only {uri} in the audio template, and our uri is <vs>_E<ep> while the file is
    <root>/<vs>/E<ep>.flac — so a flat audio/ directory of symlinks keeps the template trivial."""
    import yaml

    link_dir = out / "audio"
    link_dir.mkdir(parents=True, exist_ok=True)
    for ex in exports:
        target = link_dir / f"{ex.uri}.flac"
        if not target.exists() and ex.audio.exists():
            try:
                target.symlink_to(ex.audio.resolve())
            except OSError:
                import shutil
                shutil.copy2(ex.audio, target)
    yml = {
        "Databases": {"Survivor": str(link_dir.resolve()) + "/{uri}.flac"},
        "Protocols": {"Survivor": {"SpeakerDiarization": {"All": {
            "train": {"uri": "lists/train.lst", "annotation": "rttm/{uri}.rttm", "annotated": "uem/{uri}.uem"},
            "development": {"uri": "lists/dev.lst", "annotation": "rttm/{uri}.rttm", "annotated": "uem/{uri}.uem"},
            "test": {"uri": "lists/test.lst", "annotation": "rttm/{uri}.rttm", "annotated": "uem/{uri}.uem"},
        }}}},
    }
    with open(out / "database.yml", "w") as f:
        yaml.safe_dump(yml, f, sort_keys=False)
