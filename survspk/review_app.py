"""Review UI (spec §7.8): FastAPI + one static page. `survspk review` then open http://127.0.0.1:8765.

Shows the review queue one *group* at a time (the utterances of one run that assign could not decide), with the
clip, the text, the audio model's top candidates, the LLM text prior's suggestion, the SDH name if any, and the
labelled neighbours. Keyboard-driven. Writes `human` labels and resolves queue rows.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import db as dbm
from .config import Settings, load_settings

STATIC = Path(__file__).parent / "static"
PSEUDO = {"OTHER": "Other (non-cast) voice", "UNKNOWN": "Unknown", "NOSPEECH": "No speech / music"}
# Bank-coverage targets shown in the review UI: the self-consistency filter's minimum pool per speaker
# (stage_bank.self_consistency_filter). Below it a bank entry exists but cannot be checked for label noise.
MIN_UTTS = 5
MIN_SECS = 15.0
QUIET_OPEN_S = 5.0     # thin speaker with < this many seconds of plausible open-queue material -> "quiet this episode"


class LabelIn(BaseModel):
    utt_ids: list[str]
    speaker_id: str                 # castaway id, HOST_*, or OTHER | UNKNOWN | NOSPEECH
    note: str | None = None


class UnlabelIn(BaseModel):
    utt_ids: list[str]


class SplitIn(BaseModel):
    utt_id: str
    before_word: int                # split so that words[before_word:] become the second utterance (1 <= k < n)


class UnsplitIn(BaseModel):
    base_utt_id: str


class RefitIn(BaseModel):
    vs: str
    ep: int


class BulkIn(BaseModel):
    vs: str
    ep: int
    source: str = "text_prior"      # text_prior | audio
    min_confidence: float = 0.9


def create_app(settings: Settings | None = None, resolver_obj=None) -> FastAPI:
    s = settings or load_settings()
    app = FastAPI(title="survspk review")
    state: dict = {"settings": s}
    if resolver_obj is not None:
        state["resolver"] = resolver_obj

    def con() -> sqlite3.Connection:
        return dbm.init_db(s.db_path, s.sqlite_journal)

    def resolver():
        if "resolver" not in state:
            from .aliases import Resolver
            state["resolver"] = Resolver(s) if s.survivor_db_path.exists() else None
        return state["resolver"]

    def names_for(vs: str, ep: int) -> tuple[dict[str, str], list[str]]:
        """id -> display name for candidates present in the episode (+ host), and the ordered candidate list."""
        r = resolver()
        fr = s.franchise_for(vs)
        if r is None:
            return {fr.host_id: "Jeff (host)"}, [fr.host_id]
        bios = r.bios(vs, ep)
        present = r.present(vs, ep)
        out = {cid: f"{b['name']} ({b.get('tribe') or '?'})" for cid, b in bios.items() if cid in present or not present}
        out[fr.host_id] = "Jeff Probst (host)"
        order = sorted((c for c in out if c != fr.host_id), key=lambda c: out[c]) + [fr.host_id]
        return out, order

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return FileResponse(STATIC / "review.html")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    # ------------------------------------------------------------------ api
    @app.get("/api/episodes")
    def episodes():
        c = con()
        rows = c.execute(
            """SELECT u.version_season AS vs, u.episode AS ep,
                      SUM(CASE WHEN q.resolved=0 THEN 1 ELSE 0 END) AS n_open,
                      SUM(CASE WHEN q.resolved=1 THEN 1 ELSE 0 END) AS n_done,
                      (SELECT COUNT(*) FROM labels l JOIN utterances u2 ON u2.utt_id=l.utt_id
                         WHERE u2.version_season=u.version_season AND u2.episode=u.episode AND l.source='human') AS n_human
               FROM review_queue q JOIN utterances u ON u.utt_id=q.utt_id
               GROUP BY u.version_season, u.episode ORDER BY u.version_season, u.episode""").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                from .stage_embed import _current_utts, embedding_path, parquet_is_fresh
                p = embedding_path(s, d["vs"], d["ep"], s.audio.variant)
                d["stale_embeddings"] = p.exists() and not parquet_is_fresh(p, _current_utts(c, d["vs"], d["ep"], s.embed.min_duration_s))
            except Exception:  # noqa: BLE001
                d["stale_embeddings"] = False
            out.append(d)
        return out

    @app.get("/api/queue/{vs}/{ep}")
    def queue(vs: str, ep: int, include_resolved: bool = False):
        c = con()
        names, order = names_for(vs, ep)
        utts = c.execute("""SELECT utt_id, idx, start_s, end_s, text, segment, sdh_name, sdh_speaker_id, sdh_resolution,
                                   domain_hint, flags FROM utterances WHERE version_season=? AND episode=? ORDER BY idx""",
                         (vs, ep)).fetchall()
        by_id = {r["utt_id"]: dict(r) for r in utts}
        for u in by_id.values():
            f = json.loads(u["flags"] or "{}")
            u["run_id"], u["name_inherited"] = f.get("run_id", -1), f.get("name_inherited", True)
            u["split_from"] = f.get("split_from")
        labels = {r["utt_id"]: dict(r) for r in c.execute(
            "SELECT l.* FROM labels l JOIN utterances u ON u.utt_id=l.utt_id WHERE u.version_season=? AND u.episode=?", (vs, ep))}
        q = c.execute("SELECT utt_id, reason, payload, resolved FROM review_queue WHERE utt_id IN (%s)%s" % (
            ",".join("?" * len(by_id)), "" if include_resolved else " AND resolved=0"), tuple(by_id)).fetchall()
        tp_by_utt = {}
        if s.raw.get("text_prior", {}).get("show_in_review", False):      # experiment only: 26-41% precision on US47E02
            try:
                for r in c.execute("SELECT * FROM text_prior WHERE version_season=? AND episode=?", (vs, ep)):
                    for uid in json.loads(r["utt_ids"] or "[]"):
                        tp_by_utt[uid] = dict(r)
            except sqlite3.OperationalError:
                pass

        # group queued utterances by (run_id, adjacent in episode order) — order position, not idx arithmetic,
        # so split parts (idx + 0.5) stay grouped with the rest of their run
        ordered_ids = [u["utt_id"] for u in sorted(by_id.values(), key=lambda x: (x["idx"], x["start_s"]))]
        pos = {uid: i for i, uid in enumerate(ordered_ids)}
        qrows = sorted((dict(r) for r in q), key=lambda r: pos[r["utt_id"]])
        groups: list[dict] = []
        for r in qrows:
            u = by_id[r["utt_id"]]
            pl = json.loads(r["payload"] or "{}")
            g = groups[-1] if groups else None
            if g and g["run_id"] == u["run_id"] and pos[g["utt_ids"][-1]] == pos[u["utt_id"]] - 1:
                g["utt_ids"].append(u["utt_id"])
                g["reasons"].add(r["reason"])
            else:
                groups.append({"run_id": u["run_id"], "utt_ids": [u["utt_id"]], "reasons": {r["reason"]}, "payload": pl,
                               "resolved": bool(r["resolved"])})

        def label_name(uid: str) -> str | None:
            l = labels.get(uid)
            return names.get(l["speaker_id"], l["speaker_id"]) if l else None

        def neighbours(first: str, last: str, k: int = 2) -> tuple[list, list]:
            i, j = pos[first], pos[last]
            before, after = [], []
            p = i - 1
            while p >= 0 and len(before) < k:
                u = by_id[ordered_ids[p]]
                before.insert(0, {"mmss": _mmss(u["start_s"]), "text": (u["text"] or "")[:140], "speaker": label_name(u["utt_id"])})
                p -= 1
            p = j + 1
            while p < len(ordered_ids) and len(after) < k:
                u = by_id[ordered_ids[p]]
                after.append({"mmss": _mmss(u["start_s"]), "text": (u["text"] or "")[:140], "speaker": label_name(u["utt_id"])})
                p += 1
            return before, after

        out = []
        for g in groups:
            us = [by_id[u] for u in g["utt_ids"]]
            pl = g["payload"]
            top = [[sid, float(x), names.get(sid, sid)] for sid, x in (pl.get("top") or [])[:3]]
            if pl.get("pred") and not top:                         # sdh_conflict payload
                top = [[pl["pred"], float(pl.get("score", 0)), names.get(pl["pred"], pl["pred"])]]
            sdh = next((u["sdh_speaker_id"] for u in us if u["sdh_speaker_id"] and u["sdh_resolution"] in ("cast", "alias", "host")), None)
            explicit = any(u["sdh_speaker_id"] and not u["name_inherited"] for u in us)
            t = tp_by_utt.get(g["utt_ids"][0]) or tp_by_utt.get(us[0].get("split_from") or "")
            before, after = neighbours(g["utt_ids"][0], g["utt_ids"][-1])
            out.append({
                "run_id": g["run_id"], "utt_ids": g["utt_ids"], "reasons": sorted(g["reasons"]), "resolved": g["resolved"],
                "start_s": us[0]["start_s"], "end_s": us[-1]["end_s"], "mmss": _mmss(us[0]["start_s"]),
                "dur": round(us[-1]["end_s"] - us[0]["start_s"], 1), "domain": us[0]["domain_hint"], "segment": us[0]["segment"],
                "utts": [{"utt_id": u["utt_id"], "start_s": u["start_s"], "end_s": u["end_s"], "text": u["text"],
                          "label": labels.get(u["utt_id"], {}).get("speaker_id"),
                          "label_source": labels.get(u["utt_id"], {}).get("source")} for u in us],
                "audio_top": top,
                "sdh": {"speaker_id": sdh, "name": names.get(sdh, sdh), "explicit": explicit} if sdh else None,
                "text_prior": {"speaker_id": t["speaker_id"], "name": names.get(t["speaker_id"], t["speaker_name"]),
                               "confidence": t["confidence"], "reason": t["reason"]} if t else None,
                "before": before, "after": after,
            })
        # order: conflicts and mention flags first (cheap, high value), then longest first
        prio = {"sdh_conflict": 0, "name_mentioned": 1, "low_margin": 2, "no_candidate": 3}
        out.sort(key=lambda g: (min(prio.get(r, 9) for r in g["reasons"]), -g["dur"]))
        return {"vs": vs, "ep": ep, "candidates": [{"id": c, "name": names[c]} for c in order],
                "pseudo": [{"id": k, "name": v} for k, v in PSEUDO.items()], "groups": out,
                "n_open": sum(not g["resolved"] for g in out)}

    @app.get("/api/clip/{vs}/{ep}")
    def clip(vs: str, ep: int, start: float = Query(...), end: float = Query(...), variant: str = "raw", pad: float = 0.35):
        import soundfile as sf

        path = s.audio_path(variant, vs, ep)
        if not path.exists():
            path = s.audio_path("vocals", vs, ep)
        if not path.exists():
            raise HTTPException(404, f"no audio for {vs} E{ep:02d}")
        info = sf.info(str(path))
        a = max(0, int((start - pad) * info.samplerate))
        b = min(info.frames, int((end + pad) * info.samplerate))
        data, sr = sf.read(str(path), start=a, stop=b, dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, np.asarray(data), sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/label")
    def label(inp: LabelIn):
        c = con()
        if not inp.utt_ids:
            raise HTTPException(400, "no utt_ids")
        ids = tuple(inp.utt_ids)
        ph = ",".join("?" * len(ids))
        for uid in ids:
            c.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at)
                         VALUES (?,?, 'human', 1.0, ?, (SELECT domain_hint FROM utterances WHERE utt_id=?), datetime('now'))""",
                      (uid, inp.speaker_id, json.dumps({"note": inp.note}) if inp.note else None, uid))
        c.execute(f"UPDATE review_queue SET resolved=1 WHERE utt_id IN ({ph})", ids)
        c.commit()
        return {"ok": True, "n": len(ids)}

    @app.post("/api/unlabel")
    def unlabel(inp: UnlabelIn):
        """Undo: remove human labels and reopen the queue rows."""
        c = con()
        ids = tuple(inp.utt_ids)
        ph = ",".join("?" * len(ids))
        c.execute(f"DELETE FROM labels WHERE source='human' AND utt_id IN ({ph})", ids)
        c.execute(f"UPDATE review_queue SET resolved=0 WHERE utt_id IN ({ph})", ids)
        c.commit()
        return {"ok": True}

    @app.post("/api/bulk_accept")
    def bulk_accept(inp: BulkIn):
        """Accept suggestions above a confidence for every open group: text_prior.confidence or audio top-1 score."""
        data = queue(inp.vs, inp.ep)
        n = 0
        for g in data["groups"]:
            if g["resolved"]:
                continue
            if inp.source == "text_prior" and g["text_prior"] and g["text_prior"]["speaker_id"] \
                    and g["text_prior"]["confidence"] >= inp.min_confidence:
                label(LabelIn(utt_ids=g["utt_ids"], speaker_id=g["text_prior"]["speaker_id"], note="bulk:text_prior"))
                n += 1
            elif inp.source == "audio" and g["audio_top"] and g["audio_top"][0][1] >= inp.min_confidence:
                label(LabelIn(utt_ids=g["utt_ids"], speaker_id=g["audio_top"][0][0], note="bulk:audio"))
                n += 1
        return {"ok": True, "n": n}


    SPLITS_SCHEMA = """CREATE TABLE IF NOT EXISTS utt_splits (
        base_utt_id TEXT PRIMARY KEY, original TEXT NOT NULL, cues TEXT NOT NULL, parts TEXT NOT NULL, created_at TEXT)"""

    def utt_words(c: sqlite3.Connection, u: dict) -> list[dict]:
        """Ordered words of an utterance with times (audio seconds). Missing word times are interpolated; if no word
        was placed at all, times are spread proportionally to character length over the utterance span."""
        cues = c.execute("""SELECT uc.cue_id, uc.line_indices, cu.start_s FROM utterance_cues uc JOIN cues cu USING (cue_id)
                            WHERE uc.utt_id=? ORDER BY cu.start_s""", (u["utt_id"],)).fetchall()
        words: list[dict] = []
        for r in cues:
            lines = json.loads(r["line_indices"] or "[]")
            if not lines:
                continue
            ph = ",".join("?" * len(lines))
            for w in c.execute(f"""SELECT word, start_s, end_s FROM words WHERE cue_id=? AND line_idx IN ({ph})
                                   ORDER BY line_idx, word_idx""", (r["cue_id"], *lines)):
                words.append({"word": w["word"], "start_s": w["start_s"], "end_s": w["end_s"]})
        if not words:                                  # no alignment rows at all: fall back to the text
            toks = (u["text"] or "").split()
            words = [{"word": t, "start_s": None, "end_s": None} for t in toks]
        n = len(words)
        if n == 0:
            return []
        if all(w["start_s"] is None for w in words):
            span = u["end_s"] - u["start_s"]
            total = sum(len(w["word"]) + 1 for w in words)
            t = u["start_s"]
            for w in words:
                d = span * (len(w["word"]) + 1) / total
                w["start_s"], w["end_s"] = t, t + d
                t += d
        else:                                          # interpolate gaps
            for i, w in enumerate(words):
                if w["start_s"] is None:
                    prev = next((x["end_s"] for x in reversed(words[:i]) if x["end_s"] is not None), u["start_s"])
                    nxt = next((x["start_s"] for x in words[i + 1:] if x["start_s"] is not None), u["end_s"])
                    w["start_s"], w["end_s"] = prev, nxt
        for i, w in enumerate(words):
            w["i"] = i
        return words

    @app.get("/api/words/{utt_id}")
    def words_of(utt_id: str):
        c = con()
        u = c.execute("SELECT utt_id, start_s, end_s, text FROM utterances WHERE utt_id=?", (utt_id,)).fetchone()
        if u is None:
            raise HTTPException(404, utt_id)
        return utt_words(c, dict(u))

    @app.post("/api/split")
    def split(inp: SplitIn):
        """Split one utterance into two at a word boundary (two speakers inside one cue with no dash). The parts
        get ids <utt_id>a / <utt_id>b and idx / idx+0.5, keep the run, and replace the original in the queue.
        Embeddings for the episode become stale until `survspk embed` runs again (it re-embeds automatically)."""
        c = con()
        c.executescript(SPLITS_SCHEMA)
        row = c.execute("SELECT * FROM utterances WHERE utt_id=?", (inp.utt_id,)).fetchone()
        if row is None:
            raise HTTPException(404, inp.utt_id)
        u = dict(row)
        words = utt_words(c, u)
        k = inp.before_word
        if not (1 <= k < len(words)):
            raise HTTPException(400, f"before_word must be in 1..{len(words) - 1}")
        t_cut = 0.5 * (words[k - 1]["end_s"] + words[k]["start_s"])
        t_cut = min(max(t_cut, u["start_s"] + 0.05), u["end_s"] - 0.05)
        flags = json.loads(u["flags"] or "{}")
        cues = [dict(r) for r in c.execute("SELECT cue_id, line_indices FROM utterance_cues WHERE utt_id=?", (inp.utt_id,))]
        parts = []
        for suffix, ws, (a, b), idx_off, keep_name in (("a", words[:k], (u["start_s"], t_cut), 0, True),
                                                      ("b", words[k:], (t_cut, u["end_s"]), 0.5, False)):
            pid = f"{inp.utt_id}{suffix}"
            f = dict(flags, split_from=inp.utt_id, split_at_word=k, split_part=suffix)
            c.execute("""INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, is_speech,
                         sdh_name, sdh_speaker_id, is_italic, domain_hint, align_ok, flags, n_words, sdh_resolution)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (pid, u["version_season"], u["episode"], u["idx"] + idx_off, a, b, " ".join(w["word"] for w in ws),
                       u["segment"], 1, u["sdh_name"] if keep_name else None, u["sdh_speaker_id"] if keep_name else None,
                       u["is_italic"], u["domain_hint"], u["align_ok"], json.dumps(f), len(ws),
                       u["sdh_resolution"] if keep_name else None))
            for cu in cues:
                c.execute("INSERT OR REPLACE INTO utterance_cues (utt_id, cue_id, line_indices) VALUES (?,?,?)",
                          (pid, cu["cue_id"], cu["line_indices"]))
            parts.append(pid)
        q = c.execute("SELECT reason, payload FROM review_queue WHERE utt_id=?", (inp.utt_id,)).fetchone()
        for pid in parts:
            c.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                      (pid, q["reason"] if q else "split", q["payload"] if q else json.dumps({"split_from": inp.utt_id})))
        c.execute("INSERT OR REPLACE INTO utt_splits (base_utt_id, original, cues, parts, created_at) VALUES (?,?,?,?,datetime('now'))",
                  (inp.utt_id, json.dumps(u), json.dumps(cues), json.dumps(parts)))
        c.execute("DELETE FROM review_queue WHERE utt_id=?", (inp.utt_id,))
        c.execute("DELETE FROM labels WHERE utt_id=?", (inp.utt_id,))
        c.execute("DELETE FROM utterance_cues WHERE utt_id=?", (inp.utt_id,))
        c.execute("DELETE FROM utterances WHERE utt_id=?", (inp.utt_id,))
        c.commit()
        return {"ok": True, "parts": parts, "t_cut": round(t_cut, 3)}

    @app.post("/api/unsplit")
    def unsplit(inp: UnsplitIn):
        c = con()
        c.executescript(SPLITS_SCHEMA)
        r = c.execute("SELECT * FROM utt_splits WHERE base_utt_id=?", (inp.base_utt_id,)).fetchone()
        if r is None:
            raise HTTPException(404, inp.base_utt_id)
        u, cues, parts = json.loads(r["original"]), json.loads(r["cues"]), json.loads(r["parts"])
        for pid in parts:
            for t in ("labels", "review_queue", "utterance_cues", "utterances"):
                c.execute(f"DELETE FROM {t} WHERE utt_id=?", (pid,))
        cols = [k for k in u if k != "utt_id"]
        c.execute(f"INSERT INTO utterances (utt_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                  (u["utt_id"], *[u[k] for k in cols]))
        for cu in cues:
            c.execute("INSERT OR REPLACE INTO utterance_cues (utt_id, cue_id, line_indices) VALUES (?,?,?)",
                      (u["utt_id"], cu["cue_id"], cu["line_indices"]))
        c.execute("INSERT OR REPLACE INTO review_queue (utt_id, reason, payload, resolved) VALUES (?,?,?,0)",
                  (u["utt_id"], "unsplit", None))
        c.execute("DELETE FROM utt_splits WHERE base_utt_id=?", (inp.base_utt_id,))
        c.commit()
        return {"ok": True, "utt_id": u["utt_id"]}


    @app.get("/api/coverage/{vs}/{ep}")
    def coverage(vs: str, ep: int):
        """Per candidate: how much labelled speech the bank would get from this episode (explicit SDH names, human
        labels, confident auto labels) against the consistency filter's minimum pool (MIN_UTTS / MIN_SECS), plus how
        much *open* queue material could still be theirs (they are in a queued run's audio top-3 / SDH name).
        status: banked (pool reached) | thin (below, and the queue still holds plausible material) | quiet (below,
        nothing plausible left: they barely speak in this episode -- a later episode will bank them)."""
        c = con()
        names, order = names_for(vs, ep)
        thr = s.raw.get("thresholds", {})
        auto_min = float(thr.get("accept", 0.55)) + 0.05
        rows = c.execute("""
            SELECT u.utt_id, u.end_s - u.start_s AS dur, u.sdh_speaker_id, u.sdh_resolution, u.flags,
                   l.speaker_id AS lab, l.source, l.confidence, q.payload AS qpayload
            FROM utterances u LEFT JOIN labels l ON l.utt_id = u.utt_id
                 LEFT JOIN review_queue q ON q.utt_id = u.utt_id AND q.resolved = 0
            WHERE u.version_season=? AND u.episode=? AND u.segment='body'""", (vs, ep)).fetchall()
        agg: dict[str, dict] = {cid: {"n": 0, "secs": 0.0, "n_human": 0, "open_n": 0, "open_secs": 0.0} for cid in order}
        for r in rows:
            spk = None
            if r["source"] in ("human", "chyron") and r["lab"] in agg:
                spk = r["lab"]
            elif r["source"] == "auto" and (r["confidence"] or 0) >= auto_min and r["lab"] in agg:
                spk = r["lab"]
            elif r["sdh_speaker_id"] in agg and r["sdh_resolution"] in ("cast", "alias", "host"):
                f = json.loads(r["flags"] or "{}")
                if not f.get("name_inherited", True):
                    spk = r["sdh_speaker_id"]
            if spk:
                agg[spk]["n"] += 1
                agg[spk]["secs"] += float(r["dur"])
                agg[spk]["n_human"] += int(r["source"] == "human")
            elif r["qpayload"] and r["source"] != "human":
                pl = json.loads(r["qpayload"])
                maybe = {sid for sid, _ in (pl.get("top") or [])} | {pl.get("pred"), pl.get("sdh")}
                if r["sdh_speaker_id"] in agg and r["sdh_resolution"] in ("cast", "alias", "host"):
                    maybe.add(r["sdh_speaker_id"])
                for sid in maybe & set(agg):
                    agg[sid]["open_n"] += 1
                    agg[sid]["open_secs"] += float(r["dur"])
        out = []
        for cid in order:
            a = agg[cid]
            thin = a["n"] < MIN_UTTS or a["secs"] < MIN_SECS
            status = "banked" if not thin else ("thin" if a["open_secs"] >= QUIET_OPEN_S else "quiet")
            out.append({"id": cid, "name": names[cid], "n": a["n"], "secs": round(a["secs"], 1), "n_human": a["n_human"],
                        "thin": thin, "status": status, "open_n": a["open_n"], "open_secs": round(a["open_secs"], 1),
                        "min_utts": MIN_UTTS, "min_secs": MIN_SECS})
        return out

    @app.post("/api/refit")
    def refit(inp: RefitIn):
        """Refit the season bank from episodes 1..ep (explicit + human + confident auto) and re-assign this episode
        against it. Human labels are untouched; undecided runs that the fuller bank can now place leave the queue."""
        from .stage_assign import assign_episode
        from .stage_bank import StaleEmbeddings, build_bank

        c = con()
        r = resolver()
        re_embedded = False
        try:
            try:
                st = build_bank(s, c, inp.vs, list(range(1, inp.ep + 1)), as_of=inp.ep, resolver=r)
                df, a = assign_episode(s, c, inp.vs, inp.ep, bank_as_of=inp.ep, resolver=r, write=True)
            except StaleEmbeddings:                     # split lines have no vectors yet: re-embed (minutes), then retry
                from .stage_embed import embed_episode

                embed_episode(s, c, inp.vs, inp.ep, force=True)
                re_embedded = True
                st = build_bank(s, c, inp.vs, list(range(1, inp.ep + 1)), as_of=inp.ep, resolver=r)
                df, a = assign_episode(s, c, inp.vs, inp.ep, bank_as_of=inp.ep, resolver=r, write=True)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except LookupError as e:
            raise HTTPException(400, str(e))
        n_open = c.execute("""SELECT COUNT(*) FROM review_queue q JOIN utterances u USING (utt_id)
                              WHERE u.version_season=? AND u.episode=? AND q.resolved=0""", (inp.vs, inp.ep)).fetchone()[0]
        return {"ok": True, "bank_speakers": st["n_speakers"], "bank_kept": st["n_kept"], "bank_dropped": st["n_dropped"],
                "unbankable": a.get("unbankable", []), "body_auto_dur_share": a.get("body_auto_dur_share"),
                "n_queued_utts": a.get("n_queued"), "n_open": n_open, "re_embedded": re_embedded}

    @app.get("/api/stats/{vs}/{ep}")
    def stats(vs: str, ep: int):
        c = con()
        rows = c.execute("""SELECT l.source, COUNT(*) AS n, SUM(u.end_s - u.start_s) AS secs FROM labels l
                            JOIN utterances u ON u.utt_id=l.utt_id WHERE u.version_season=? AND u.episode=? AND u.segment='body'
                            GROUP BY l.source""", (vs, ep)).fetchall()
        total = c.execute("SELECT COUNT(*) AS n, SUM(end_s - start_s) AS secs FROM utterances WHERE version_season=? AND episode=? AND segment='body'",
                          (vs, ep)).fetchone()
        return {"by_source": [dict(r) for r in rows], "body_utts": total["n"], "body_secs": total["secs"]}

    return app


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
