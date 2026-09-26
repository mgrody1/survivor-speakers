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

from . import splits as _splits

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


class VerdictIn(BaseModel):
    utt_ids: list[str]
    speaker_id: str                 # the human's answer; == pred_speaker means confirm
    pred_speaker: str
    pred_score: float | None = None


class CardCheckIn(BaseModel):
    vs: str
    ep: int
    t_s: float                      # the card (chyron_hits.t_s)
    castaway_id: str
    utt_id: str | None = None       # the line the carded castaway speaks; None = none of the lines
    utt_ids: list[str] | None = None  # several lines (they speak more than one); wins over utt_id


class CardUncheckIn(BaseModel):
    vs: str
    ep: int
    t_s: float
    castaway_id: str


class BulkConfirmIn(BaseModel):
    vs: str
    ep: int
    speaker_id: str
    min_confidence: float = 0.8


class DismissIn(BaseModel):
    utt_id: str


AUDIT_SAMPLE = 20                   # auto-labelled groups per episode in the audit sample (pooled over the season
                                    # in `audit-stats`; 20 an episode is ~3 min and ~280 verdicts a season)


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
        rows = [dict(r) for r in rows]
        cards = {(r["vs"], r["ep"]): r for r in card_counts(c)}
        have = {(r["vs"], r["ep"]) for r in rows}
        rows += [{"vs": k[0], "ep": k[1], "n_open": 0, "n_done": 0, "n_human": 0} for k in cards if k not in have]
        rows.sort(key=lambda d: (d["vs"], d["ep"]))
        out = []
        for d in rows:
            k = cards.get((d["vs"], d["ep"]))
            d["n_cards"], d["n_checked"] = (k["n_cards"], k["n_checked"]) if k else (0, 0)
            try:
                from .stage_embed import _current_utts, embedding_path, parquet_is_fresh
                p = embedding_path(s, d["vs"], d["ep"], s.audio.variant)
                d["stale_embeddings"] = p.exists() and not parquet_is_fresh(p, _current_utts(c, d["vs"], d["ep"], s.embed.min_duration_s))
            except Exception:  # noqa: BLE001
                d["stale_embeddings"] = False
            out.append(d)
        return out

    def card_counts(c: sqlite3.Connection) -> list[dict]:
        try:
            return [dict(r) for r in c.execute(
                """SELECT h.version_season AS vs, h.episode AS ep, COUNT(*) AS n_cards,
                          SUM(EXISTS (SELECT 1 FROM card_checks k WHERE k.version_season=h.version_season AND k.episode=h.episode
                                      AND k.castaway_id=h.castaway_id AND ABS(k.t_s - h.t_s) <= 2.0)) AS n_checked
                   FROM chyron_hits h WHERE h.castaway_id IS NOT NULL AND h.castaway_id NOT LIKE 'HOST%' GROUP BY 1, 2""")]
        except sqlite3.OperationalError:
            return []

    def build_groups(c: sqlite3.Connection, vs: str, ep: int, items: list[dict]) -> dict:
        """Group utterances of one run that sit next to each other in episode order, and dress each group with
        text, audio candidates, SDH name, text-prior vote and labelled neighbours (the shape the UI renders).
        items: [{utt_id, reason, payload (dict), resolved}] in any order."""
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
        try:
            tv = {r["utt_id"]: dict(r) for r in c.execute(
                "SELECT * FROM split_suggestions WHERE version_season=? AND episode=? AND status='open'", (vs, ep))}
        except sqlite3.OperationalError:
            tv = {}

        def two_voices(u: dict) -> dict | None:
            """The suggested cut as the split button needs it: before_word = the first word of the second voice."""
            d = tv.get(u["utt_id"])
            if not d:
                return None
            ws = utt_words(c, u)
            if len(ws) < 2:
                return None
            k = min(range(1, len(ws)), key=lambda j: abs(0.5 * (ws[j - 1]["end_s"] + ws[j]["start_s"]) - d["t_cut"]))
            return {"before_word": k, "word": ws[k]["word"], "t_cut": d["t_cut"],
                    "left": names.get(d["left_spk"], d["left_spk"]), "right": names.get(d["right_spk"], d["right_spk"]),
                    "second_s": d["second_s"]}

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
        qrows = sorted((r for r in items if r["utt_id"] in by_id), key=lambda r: pos[r["utt_id"]])
        def plausible(u: dict, pl: dict) -> dict[str, int]:
            """who this line could be, by the coverage panel's rule (audio top list, prediction, SDH name):
            speaker -> audio rank (1 = top), 0 when only the captions name them"""
            out: dict[str, int] = {}
            for k, t in enumerate(pl.get("top") or []):
                if isinstance(t, (list, tuple)) and t and t[0] not in out:
                    out[t[0]] = k + 1
            if pl.get("pred") and not out:                          # sdh_conflict payload: the audio's pick, no list
                out[pl["pred"]] = 1
            for sid in (pl.get("pred"), pl.get("sdh"),
                        u["sdh_speaker_id"] if u["sdh_resolution"] in ("cast", "alias", "host") else None):
                if sid and sid not in out:
                    out[sid] = 0
            return out

        groups: list[dict] = []
        for r in qrows:
            u = by_id[r["utt_id"]]
            pl = r["payload"] or {}
            g = groups[-1] if groups else None
            if g and g["run_id"] == u["run_id"] and pos[g["utt_ids"][-1]] == pos[u["utt_id"]] - 1:
                g["utt_ids"].append(u["utt_id"])
                g["reasons"].add(r["reason"])
            else:
                g = {"run_id": u["run_id"], "utt_ids": [u["utt_id"]], "reasons": {r["reason"]}, "payload": pl,
                     "resolved": bool(r["resolved"]), "maybe": {}}
                groups.append(g)
            for sid, k in plausible(u, pl).items():                # best rank over the group's lines
                if sid not in g["maybe"] or (k and (not g["maybe"][sid] or k < g["maybe"][sid])):
                    g["maybe"][sid] = k

        def label_name(uid: str) -> str | None:
            l = labels.get(uid)
            return names.get(l["speaker_id"], l["speaker_id"]) if l else None

        def neighbours(first: str, last: str, k: int = 2) -> tuple[list, list]:
            i, j = pos[first], pos[last]
            before, after = [], []
            p = i - 1
            while p >= 0 and len(before) < k:
                u = by_id[ordered_ids[p]]
                before.insert(0, {"mmss": _mmss(u["start_s"]), "text": (u["text"] or "")[:140], "speaker": label_name(u["utt_id"]),
                                  "start_s": u["start_s"], "end_s": u["end_s"]})
                p -= 1
            p = j + 1
            while p < len(ordered_ids) and len(after) < k:
                u = by_id[ordered_ids[p]]
                after.append({"mmss": _mmss(u["start_s"]), "text": (u["text"] or "")[:140], "speaker": label_name(u["utt_id"]),
                              "start_s": u["start_s"], "end_s": u["end_s"]})
                p += 1
            return before, after

        out = []
        for g in groups:
            us = [by_id[u] for u in g["utt_ids"]]
            pl = g["payload"]
            top = [[sid, float(x), names.get(sid, sid)] for sid, x in (pl.get("top") or [])[:3]]
            if pl.get("pred") and not top:                         # sdh_conflict payload
                top = [[pl["pred"], float(pl.get("score", 0)), names.get(pl["pred"], pl["pred"])]]
            chy = {"speaker_id": pl["chyron"], "name": names.get(pl["chyron"], pl["chyron"]), "ocr": pl.get("ocr")} if pl.get("chyron") else None
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
                          "label_source": labels.get(u["utt_id"], {}).get("source"),
                          "two_voices": two_voices(u),
                          "split_auto": u["split_from"] if json.loads(u["flags"] or "{}").get("split_auto") else None} for u in us],
                "audio_top": top,
                "chyron": chy,
                "sdh": {"speaker_id": sdh, "name": names.get(sdh, sdh), "explicit": explicit} if sdh else None,
                "text_prior": {"speaker_id": t["speaker_id"], "name": names.get(t["speaker_id"], t["speaker_name"]),
                               "confidence": t["confidence"], "reason": t["reason"]} if t else None,
                "before": before, "after": after,
                "maybe": g["maybe"],
            })
        return {"vs": vs, "ep": ep, "candidates": [{"id": c, "name": names[c]} for c in order],
                "pseudo": [{"id": k, "name": v} for k, v in PSEUDO.items()], "groups": out}

    @app.get("/api/queue/{vs}/{ep}")
    def queue(vs: str, ep: int, include_resolved: bool = False):
        c = con()
        q = c.execute("""SELECT q.utt_id, q.reason, q.payload, q.resolved FROM review_queue q JOIN utterances u USING (utt_id)
                         WHERE u.version_season=? AND u.episode=?%s""" % ("" if include_resolved else " AND q.resolved=0"),
                      (vs, ep)).fetchall()
        items = [{"utt_id": r["utt_id"], "reason": r["reason"], "payload": json.loads(r["payload"] or "{}"),
                  "resolved": bool(r["resolved"])} for r in q]
        data = build_groups(c, vs, ep, items)
        # order: conflicts and mention flags first (cheap, high value), then longest first
        prio = {"sdh_conflict": 0, "chyron_conflict": 0, "name_mentioned": 1, "two_voices": 1, "low_margin": 2, "no_candidate": 3}
        data["groups"].sort(key=lambda g: (min(prio.get(r, 9) for r in g["reasons"]), -g["dur"]))
        data["n_open"] = sum(not g["resolved"] for g in data["groups"])
        return data

    # ------------------------------------------------------------------ audit: precision of the auto labels
    # The assign monitor compares predictions with explicit NAME: lines, and those belong to the castaways the bank
    # already knows well. A random sample of auto-labelled runs, judged by ear, is the unbiased number.
    def audit_stats_for(c: sqlite3.Connection, vs: str, ep: int) -> dict:
        names, _ = names_for(vs, ep)
        rows = c.execute("""SELECT group_key, pred_speaker, verdict, speaker_id, MIN(pred_score) AS score
                            FROM audit_verdicts WHERE version_season=? AND episode=? GROUP BY group_key""", (vs, ep)).fetchall()
        n = len(rows)
        ok = sum(r["verdict"] == "confirm" for r in rows)
        per: dict[str, dict] = {}
        for r in rows:
            d = per.setdefault(r["pred_speaker"], {"name": names.get(r["pred_speaker"], r["pred_speaker"]), "n": 0, "confirmed": 0, "rejected_as": {}})
            d["n"] += 1
            if r["verdict"] == "confirm":
                d["confirmed"] += 1
            else:
                d["rejected_as"][names.get(r["speaker_id"], r["speaker_id"])] = d["rejected_as"].get(names.get(r["speaker_id"], r["speaker_id"]), 0) + 1
        return {"n": n, "n_confirmed": ok, "precision": round(ok / n, 3) if n else None,
                "wilson_low": round(_wilson_low(ok, n), 3) if n else None, "per_speaker": per}

    @app.get("/api/audit/{vs}/{ep}")
    def audit(vs: str, ep: int, n: int = AUDIT_SAMPLE):
        """A stable random sample of auto-labelled body runs, spread across predicted speakers, minus the groups
        already judged; `n` is the sample size per episode, so once `n` verdicts exist the sample is empty (it used
        to refill to `n` open groups after every verdict, so the audit never ended). Same group shape as the queue;
        `pred` / `pred_score` carry the auto label."""
        c = con()
        rows = c.execute("""SELECT l.utt_id, l.speaker_id, l.confidence, l.top_candidates, l.run_id FROM labels l
                            JOIN utterances u USING (utt_id)
                            WHERE u.version_season=? AND u.episode=? AND u.segment='body' AND l.source='auto'""", (vs, ep)).fetchall()
        judged = {r[0] for r in c.execute("SELECT utt_id FROM audit_verdicts WHERE version_season=? AND episode=?", (vs, ep))}
        items = []
        for r in rows:
            if r["utt_id"] in judged:
                continue
            try:
                top = json.loads(r["top_candidates"] or "[]")
            except ValueError:
                top = []
            if not top or not isinstance(top[0], list):
                top = [[r["speaker_id"], r["confidence"] or 0.0]]
            items.append({"utt_id": r["utt_id"], "reason": "audit", "resolved": False,
                          "payload": {"top": top, "pred": r["speaker_id"], "score": r["confidence"], "run_id": r["run_id"]}})
        data = build_groups(c, vs, ep, items)
        groups = data["groups"]
        # any group with a judged utterance is out (a partial verdict counts as judged)
        groups = [g for g in groups if not (set(g["utt_ids"]) & judged)]
        for g in groups:
            pl = next(it["payload"] for it in items if it["utt_id"] == g["utt_ids"][0])
            g["pred"], g["pred_score"] = pl["pred"], pl["score"]
            g["reasons"] = ["audit"]
        # stratified, deterministic: shuffle within each predicted speaker by a hash of the group, then round-robin
        import hashlib
        key = lambda g: hashlib.sha1(f"{vs}|{ep}|{g['utt_ids'][0]}".encode()).hexdigest()  # noqa: E731
        by_spk: dict[str, list] = {}
        for g in sorted(groups, key=key):
            by_spk.setdefault(g["pred"], []).append(g)
        picked: list[dict] = []
        stats = audit_stats_for(c, vs, ep)
        n = max(0, n - stats["n"])                       # what is left of this episode's sample
        spks = sorted(by_spk, key=lambda s_: hashlib.sha1(f"{vs}|{ep}|{s_}".encode()).hexdigest())
        while len(picked) < n and any(by_spk.values()):
            for s_ in spks:
                if by_spk[s_] and len(picked) < n:
                    picked.append(by_spk[s_].pop(0))
        data["groups"] = picked
        data["n_open"] = len(picked)
        data["n_pool"] = len(groups)
        data["stats"] = stats
        return data

    @app.post("/api/audit_verdict")
    def audit_verdict(inp: VerdictIn):
        """Record the verdict on an audited group and write the human label (confirm keeps the auto speaker)."""
        c = con()
        if not inp.utt_ids:
            raise HTTPException(400, "no utt_ids")
        verdict = "confirm" if inp.speaker_id == inp.pred_speaker else "reject"
        prev = {r["utt_id"]: dict(r) for r in c.execute(
            "SELECT * FROM labels WHERE utt_id IN (%s)" % ",".join("?" * len(inp.utt_ids)), inp.utt_ids)}
        label(LabelIn(utt_ids=inp.utt_ids, speaker_id=inp.speaker_id, note=f"audit:{verdict}"))
        first = inp.utt_ids[0]
        for uid in inp.utt_ids:
            u = c.execute("SELECT version_season, episode, flags FROM utterances WHERE utt_id=?", (uid,)).fetchone()
            if u is None:
                continue
            run_id = json.loads(u["flags"] or "{}").get("run_id")
            c.execute("""INSERT OR REPLACE INTO audit_verdicts (utt_id, version_season, episode, run_id, group_key, pred_speaker,
                         pred_score, verdict, speaker_id, prev_label, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                      (uid, u["version_season"], u["episode"], run_id, first, inp.pred_speaker, inp.pred_score, verdict,
                       inp.speaker_id, json.dumps(prev.get(uid)) if prev.get(uid) else None))
        c.commit()
        vs, ep = c.execute("SELECT version_season, episode FROM utterances WHERE utt_id=?", (first,)).fetchone()
        return {"ok": True, "verdict": verdict, "stats": audit_stats_for(c, vs, ep)}

    @app.get("/api/audit_stats/{vs}/{ep}")
    def audit_stats(vs: str, ep: int):
        return audit_stats_for(con(), vs, ep)

    # ------------------------------------------------------------------ card check: which line is the carded castaway
    # A name card says who is on screen, not which line they speak; the automatic rule (the line starting ~3 s
    # before the card) is right about three times in four, and one wrong card poisons that castaway's bank entry.
    # One click per card fixes it. The checked line gets a `chyron` label at confidence 1.0; re-runs keep it.
    def card_hits(c: sqlite3.Connection, vs: str, ep: int) -> list[dict]:
        return [dict(r) for r in c.execute("""SELECT t_s, t_end_s, ocr_text, castaway_id, match_score FROM chyron_hits
                                              WHERE version_season=? AND episode=? AND castaway_id IS NOT NULL ORDER BY t_s""", (vs, ep))]

    def drop_card_label(c: sqlite3.Connection, vs: str, ep: int, t_s: float, castaway_id: str) -> None:
        """Remove the chyron label this card wrote (automatic or checked) so a new answer replaces it."""
        for r in c.execute("""SELECT l.utt_id, l.top_candidates FROM labels l JOIN utterances u USING (utt_id)
                              WHERE u.version_season=? AND u.episode=? AND l.source='chyron' AND l.speaker_id=?""",
                           (vs, ep, castaway_id)).fetchall():
            try:
                t = float(json.loads(r["top_candidates"] or "{}").get("t_s"))
            except (TypeError, ValueError):
                continue
            if abs(t - t_s) <= 2.0:
                c.execute("DELETE FROM labels WHERE utt_id=? AND source='chyron'", (r["utt_id"],))

    @app.get("/api/cards/{vs}/{ep}")
    def cards(vs: str, ep: int):
        from .chyron import ChyronCfg, anchor_utterance, card_checks, card_lines, check_for

        c = con()
        cfg = ChyronCfg.from_settings(s)
        names, order = names_for(vs, ep)
        checks = card_checks(c, vs, ep)
        body = [dict(r) for r in c.execute("SELECT utt_id, start_s, end_s, segment FROM utterances WHERE version_season=? AND episode=?", (vs, ep))]
        labels = {r["utt_id"]: dict(r) for r in c.execute(
            "SELECT l.utt_id, l.speaker_id, l.source FROM labels l JOIN utterances u USING (utt_id) WHERE u.version_season=? AND u.episode=?", (vs, ep))}
        out = []
        host = s.franchise_for(vs).host_id
        for h in card_hits(c, vs, ep):
            if h["castaway_id"] == host:                 # the host has no card worth checking; his bank comes from elsewhere
                continue
            lines = card_lines(c, vs, ep, h["t_s"], cfg)
            sug = anchor_utterance(body, h["t_s"], cfg, h["t_end_s"])
            chk = check_for(checks, h["t_s"], h["castaway_id"])
            out.append({
                "t_s": h["t_s"], "mmss": _mmss(h["t_s"]), "castaway_id": h["castaway_id"],
                "name": names.get(h["castaway_id"], h["castaway_id"]), "ocr": h["ocr_text"],
                "suggested": sug["utt_id"] if sug else None,
                "checked": chk is not None, "answer": chk["utt_ids"][0] if chk and chk["utt_ids"] else None,
                "answers": chk["utt_ids"] if chk else [],
                "lines": [{"utt_id": u["utt_id"], "start_s": u["start_s"], "end_s": u["end_s"], "text": u["text"],
                           "dt": round(u["start_s"] - h["t_s"], 1),
                           "label": names.get((labels.get(u["utt_id"]) or {}).get("speaker_id"), (labels.get(u["utt_id"]) or {}).get("speaker_id")),
                           "label_source": (labels.get(u["utt_id"]) or {}).get("source")} for u in lines],
            })
        per = {}
        for x in out:
            p = per.setdefault(x["castaway_id"], {"id": x["castaway_id"], "name": x["name"], "n": 0, "confirmed": 0, "unchecked": 0})
            p["n"] += 1
            p["confirmed"] += bool(x["checked"] and x["answer"])
            p["unchecked"] += not x["checked"]
        cast = [{"id": cid, "name": names[cid]} for cid in order if cid != s.franchise_for(vs).host_id]
        return {"vs": vs, "ep": ep, "cards": out, "per_castaway": [per.get(x["id"], {**x, "n": 0, "confirmed": 0, "unchecked": 0}) for x in cast],
                "window": [cfg.check_before_s, cfg.check_after_s]}

    @app.post("/api/card_check")
    def card_check(inp: CardCheckIn):
        c = con()
        ids = list(dict.fromkeys(inp.utt_ids if inp.utt_ids else ([inp.utt_id] if inp.utt_id else [])))
        lines = []
        for uid in ids:
            u = c.execute("SELECT * FROM utterances WHERE utt_id=? AND version_season=? AND episode=?", (uid, inp.vs, inp.ep)).fetchone()
            if u is None:
                raise HTTPException(404, f"no line {uid} in {inp.vs} E{inp.ep:02d}")
            lines.append(u)
        h = c.execute("SELECT * FROM chyron_hits WHERE version_season=? AND episode=? AND castaway_id=? AND ABS(t_s - ?) <= 0.01",
                      (inp.vs, inp.ep, inp.castaway_id, inp.t_s)).fetchone()
        if h is None:
            raise HTTPException(404, "no such card")
        c.execute("DELETE FROM card_checks WHERE version_season=? AND episode=? AND castaway_id=? AND ABS(t_s - ?) <= 0.01",
                  (inp.vs, inp.ep, inp.castaway_id, inp.t_s))
        c.execute("""INSERT INTO card_checks (version_season, episode, t_s, castaway_id, utt_id, utt_ids, created_at)
                     VALUES (?,?,?,?,?,?,datetime('now'))""",
                  (inp.vs, inp.ep, h["t_s"], inp.castaway_id, ids[0] if ids else None, json.dumps(ids)))
        drop_card_label(c, inp.vs, inp.ep, h["t_s"], inp.castaway_id)
        wrote = 0
        for u in lines:
            prev = c.execute("SELECT source FROM labels WHERE utt_id=?", (u["utt_id"],)).fetchone()
            if not (prev and prev["source"] == "human"):             # a human label on the line wins
                c.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at,
                                                              version_season, episode, start_s, end_s, text)
                             VALUES (?,?,'chyron',1.0,?,?,datetime('now'),?,?,?,?,?)""",
                          (u["utt_id"], inp.castaway_id, json.dumps({"t_s": round(h["t_s"], 2), "ocr": h["ocr_text"], "checked": True}),
                           u["domain_hint"], inp.vs, inp.ep, u["start_s"], u["end_s"], u["text"]))
                wrote += 1
            c.execute("UPDATE review_queue SET resolved=1 WHERE utt_id=? AND reason='chyron_conflict'", (u["utt_id"],))
        c.commit()
        return {"ok": True, "labelled": bool(wrote), "n_labelled": wrote}

    @app.post("/api/card_uncheck")
    def card_uncheck(inp: CardUncheckIn):
        """Undo a card check: forget the answer and its label (the next `survspk chyron` run re-anchors the card by time)."""
        c = con()
        c.execute("DELETE FROM card_checks WHERE version_season=? AND episode=? AND castaway_id=? AND ABS(t_s - ?) <= 0.01",
                  (inp.vs, inp.ep, inp.castaway_id, inp.t_s))
        drop_card_label(c, inp.vs, inp.ep, inp.t_s, inp.castaway_id)
        c.commit()
        return {"ok": True}

    def frame_at(vs: str, ep: int, t: float, sub: str, scale_w: int) -> Path:
        """One video frame at t, cached as a small JPEG under frames/<sub>."""
        from .chyron import grab_frame

        c = con()
        out = Path(s.paths.work_root) / "frames" / sub / vs / f"E{ep:02d}_{t:.1f}.jpg"
        if not out.exists():
            row = c.execute("SELECT video_path FROM episodes WHERE version_season=? AND episode=?", (vs, ep)).fetchone()
            video = dbm.localize(c, s, row["video_path"]) if row else None
            if video is None or not Path(video).exists():
                raise HTTPException(404, "video not reachable (is the NAS mounted?)")
            try:
                grab_frame(Path(video), t, out, scale_w=scale_w)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, f"frame grab failed: {str(e)[:200]}")
        return out

    @app.get("/api/card_frame/{vs}/{ep}")
    def card_frame(vs: str, ep: int, t: float = Query(...)):
        """The video frame just after the card appears (who is on screen)."""
        out = frame_at(vs, ep, t + 0.3, "cards", 640)
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/line_frame/{vs}/{ep}")
    def line_frame(vs: str, ep: int, t: float = Query(...)):
        """A still from inside a line: confessionals show who is talking, camp scenes who is in the shot."""
        out = frame_at(vs, ep, round(t, 1), "lines", 320)
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

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
            c.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at,
                                                        version_season, episode, start_s, end_s, text)
                         SELECT utt_id, ?, 'human', 1.0, ?, domain_hint, datetime('now'), version_season, episode, start_s, end_s, text
                         FROM utterances WHERE utt_id=?""",
                      (inp.speaker_id, json.dumps({"note": inp.note}) if inp.note else None, uid))
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
        # an undone audit verdict puts the auto label back, so the group returns to the sample
        for r in c.execute(f"SELECT utt_id, prev_label FROM audit_verdicts WHERE utt_id IN ({ph}) AND prev_label IS NOT NULL", ids).fetchall():
            p = json.loads(r["prev_label"])
            cols = [k for k in p if k != "utt_id"]
            c.execute(f"INSERT OR REPLACE INTO labels (utt_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                      (r["utt_id"], *[p[k] for k in cols]))
        c.execute(f"DELETE FROM audit_verdicts WHERE utt_id IN ({ph})", ids)
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

    utt_words = _splits.utt_words

    @app.get("/api/words/{utt_id}")
    def words_of(utt_id: str):
        c = con()
        u = c.execute("SELECT utt_id, start_s, end_s, text FROM utterances WHERE utt_id=?", (utt_id,)).fetchone()
        if u is None:
            raise HTTPException(404, utt_id)
        return utt_words(c, dict(u))

    @app.post("/api/split")
    def split(inp: SplitIn):
        """Split one utterance into two at a word boundary (two speakers inside one cue with no dash). See
        survspk.splits.split_utterance; the parts replace the original in the queue."""
        c = con()
        try:
            r = _splits.split_utterance(c, inp.utt_id, inp.before_word)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **r}

    @app.post("/api/two_voices/dismiss")
    def two_voices_dismiss(inp: DismissIn):
        """One voice after all: drop the suggestion; the line stays in the queue only for its other reasons."""
        c = con()
        c.execute("UPDATE split_suggestions SET status='dismissed' WHERE utt_id=?", (inp.utt_id,))
        c.execute("DELETE FROM review_queue WHERE utt_id=? AND reason='two_voices'", (inp.utt_id,))
        c.commit()
        return {"ok": True}

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
        try:                                   # undoing an automatic split: do not split this line again
            c.execute("UPDATE split_suggestions SET status='dismissed' WHERE utt_id=?", (inp.base_utt_id,))
        except sqlite3.OperationalError:
            pass
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
        """Refit the season bank from episodes 1..ep (caption names + human + name cards) and re-assign this episode
        against it. Human labels are untouched; undecided runs that the fuller bank can now place leave the queue."""
        from .stage_assign import assign_episode
        from .stage_bank import StaleEmbeddings, build_bank

        c = con()
        r = resolver()
        before = episode_labels(c, inp.vs, inp.ep)
        re_embedded: list[int] = []
        try:
            # split lines have no vectors yet: re-embed the episode the error names (any of 1..ep, minutes each), retry
            for _ in range(inp.ep + 2):
                try:
                    st = build_bank(s, c, inp.vs, list(range(1, inp.ep + 1)), as_of=inp.ep, resolver=r)
                    df, a = assign_episode(s, c, inp.vs, inp.ep, bank_as_of=inp.ep, resolver=r, write=True)
                    break
                except StaleEmbeddings as e:
                    ep_stale = e.ep if e.ep is not None else inp.ep
                    if ep_stale in re_embedded:
                        raise
                    from .stage_embed import embed_episode

                    embed_episode(s, c, e.vs or inp.vs, ep_stale, variant=e.variant, model=e.model, force=True)
                    re_embedded.append(ep_stale)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except LookupError as e:
            raise HTTPException(400, str(e))
        n_open = c.execute("""SELECT COUNT(*) FROM review_queue q JOIN utterances u USING (utt_id)
                              WHERE u.version_season=? AND u.episode=? AND q.resolved=0""", (inp.vs, inp.ep)).fetchone()[0]
        return {"ok": True, "bank_speakers": st["n_speakers"], "bank_kept": st["n_kept"], "bank_dropped": st["n_dropped"],
                "unbankable": a.get("unbankable", []), "body_auto_dur_share": a.get("body_auto_dur_share"),
                "n_queued_utts": a.get("n_queued"), "n_open": n_open, "re_embedded": re_embedded,
                "changes": label_changes(c, inp.vs, inp.ep, before, episode_labels(c, inp.vs, inp.ep))}

    @app.get("/api/stats/{vs}/{ep}")
    def stats(vs: str, ep: int):
        c = con()
        rows = c.execute("""SELECT l.source, COUNT(*) AS n, SUM(u.end_s - u.start_s) AS secs FROM labels l
                            JOIN utterances u ON u.utt_id=l.utt_id WHERE u.version_season=? AND u.episode=? AND u.segment='body'
                            GROUP BY l.source""", (vs, ep)).fetchall()
        total = c.execute("SELECT COUNT(*) AS n, SUM(end_s - start_s) AS secs FROM utterances WHERE version_season=? AND episode=? AND segment='body'",
                          (vs, ep)).fetchone()
        return {"by_source": [dict(r) for r in rows], "body_utts": total["n"], "body_secs": total["secs"]}

    # ------------------------------------------------------------------ what a refit changed
    def episode_labels(c: sqlite3.Connection, vs: str, ep: int) -> dict[str, tuple]:
        return {r[0]: (r[1], r[2], r[3]) for r in c.execute(
            """SELECT l.utt_id, l.speaker_id, l.source, l.confidence FROM labels l JOIN utterances u USING (utt_id)
               WHERE u.version_season=? AND u.episode=?""", (vs, ep))}

    def label_changes(c: sqlite3.Connection, vs: str, ep: int, before: dict, after: dict, limit: int = 60) -> dict:
        """Lines whose speaker a refit changed (flipped from one person to another), plus how many it newly labelled
        and how many it took back to the queue. Human labels never change, so these are all machine labels."""
        names, _ = names_for(vs, ep)
        flips = [u for u, a in after.items() if u in before and before[u][0] != a[0]]
        new = sum(1 for u in after if u not in before)
        dropped = sum(1 for u in before if u not in after)
        rows = {r["utt_id"]: dict(r) for r in c.execute(
            "SELECT utt_id, start_s, end_s, text FROM utterances WHERE utt_id IN (%s)" % ",".join("?" * len(flips)), flips)} if flips else {}
        out = []
        for u in flips:
            if u not in rows:
                continue
            x = rows[u]
            out.append({"utt_id": u, "start_s": x["start_s"], "end_s": x["end_s"], "mmss": _mmss(x["start_s"]),
                        "dur": round(x["end_s"] - x["start_s"], 1), "text": (x["text"] or "")[:140],
                        "old": before[u][0], "old_name": names.get(before[u][0], before[u][0]), "old_source": before[u][1],
                        "new": after[u][0], "new_name": names.get(after[u][0], after[u][0]), "new_source": after[u][1],
                        "new_conf": after[u][2]})
        out.sort(key=lambda d: -d["dur"])
        return {"n_flipped": len(out), "n_new": new, "n_dropped": dropped, "flips": out[:limit]}

    # ------------------------------------------------------------------ what a castaway sounds like
    @app.get("/api/voice/{vs}/{ep}/{speaker_id}")
    def voice_samples(vs: str, ep: int, speaker_id: str, n: int = 3, exclude: str = ""):
        """A few clean lines of one speaker from this season: lines a person, a name card or an explicit caption
        name gave them, 2-9 s long, confessionals first, lines the bank kept first, spread over episodes."""
        c = con()
        skip = set(filter(None, exclude.split(",")))
        kept: set[str] = set()
        row = c.execute("SELECT MAX(as_of_episode) FROM speaker_bank WHERE version_season=?", (vs,)).fetchone()
        if row and row[0] is not None:
            for r in c.execute("SELECT payload FROM speaker_bank WHERE version_season=? AND as_of_episode=? AND speaker_id=?",
                               (vs, row[0], speaker_id)):
                try:
                    kept |= set(json.loads(r[0] or "{}").get("utt_ids") or [])
                except ValueError:
                    pass
        rows = [dict(r) for r in c.execute(
            """SELECT u.utt_id, u.episode, u.start_s, u.end_s, u.domain_hint, l.source, l.speaker_id AS lab,
                      u.sdh_speaker_id, u.sdh_resolution, u.flags
               FROM utterances u LEFT JOIN labels l USING (utt_id)
               WHERE u.version_season=? AND u.segment='body' AND u.end_s - u.start_s BETWEEN 2.0 AND 9.0
                 AND ((l.speaker_id=? AND l.source IN ('human', 'chyron'))
                      OR (l.utt_id IS NULL OR l.source NOT IN ('human', 'chyron')) AND u.sdh_speaker_id=?
                         AND u.sdh_resolution IN ('cast', 'alias', 'host'))""", (vs, speaker_id, speaker_id))]
        cand = []
        for r in rows:
            if r["utt_id"] in skip:
                continue
            if r["lab"] == speaker_id and r["source"] in ("human", "chyron"):
                src = r["source"]
            elif not json.loads(r["flags"] or "{}").get("name_inherited", True):
                src = "caption name"
            else:
                continue
            dur = r["end_s"] - r["start_s"]
            rank = (r["utt_id"] not in kept, {"human": 0, "chyron": 1}.get(src, 2), r["domain_hint"] != "confessional",
                    abs(dur - 4.5), r["utt_id"])
            cand.append((rank, {"utt_id": r["utt_id"], "vs": vs, "ep": r["episode"], "start_s": r["start_s"],
                                "end_s": r["end_s"], "source": src, "domain": r["domain_hint"]}))
        cand.sort(key=lambda x: x[0])
        out, eps = [], set()
        for _, d in cand:                                  # one per episode first, then the rest
            if d["ep"] not in eps and len(out) < n:
                out.append(d); eps.add(d["ep"])
        for _, d in cand:
            if len(out) >= n:
                break
            if d not in out:
                out.append(d)
        return {"speaker_id": speaker_id, "samples": out, "n_available": len(cand)}

    # ------------------------------------------------------------------ who the diarizer hears, 50 ms at a time
    @app.get("/api/diar/{vs}/{ep}")
    def diar(vs: str, ep: int, start: float = Query(...), end: float = Query(...), step: float = 0.05):
        """The diarizer's dominant voice over [start, end] from one of its windows (channels are only comparable
        within a window), renumbered by presence: 0 = the most heard voice, 1 = the next, 2 = any other, -1 = none."""
        from .stage_diarize import FRAME_S, load_track

        tr = load_track(s, vs, ep)
        if tr is None:
            return {"t0": start, "step": step, "ch": []}
        k = tr.window_for(start, end)
        s0, off, n = tr.starts[k], tr.offs[k], tr.lens[k]
        i0, i1 = max(0, int((start - s0) / FRAME_S)), min(int(n), int((end - s0) / FRAME_S))
        dom = tr.dom[off + i0: off + i1].astype(np.int64)
        if len(dom) == 0:
            return {"t0": start, "step": step, "ch": []}
        counts = np.bincount(dom[dom >= 0], minlength=8) if (dom >= 0).any() else np.zeros(8, int)
        order = [int(x) for x in np.argsort(-counts) if counts[x] > 0]
        remap = np.full(9, -1, np.int64)
        for j, x in enumerate(order):
            remap[x] = min(j, 2)
        per = max(1, int(round(step / FRAME_S)))
        ch = [int(remap[v]) if v >= 0 else -1 for v in dom[per // 2::per]]
        return {"t0": round(float(s0 + (i0 + per // 2) * FRAME_S - step / 2), 3), "step": per * FRAME_S, "ch": ch}

    # ------------------------------------------------------------------ bulk confirm (audit mode)
    def bulk_rows(c: sqlite3.Connection, inp) -> list:
        return c.execute("""SELECT l.utt_id, l.confidence, l.top_candidates, u.end_s - u.start_s AS dur FROM labels l
                            JOIN utterances u USING (utt_id) WHERE u.version_season=? AND u.episode=? AND u.segment='body'
                            AND l.source='auto' AND l.speaker_id=? AND l.confidence >= ?""",
                         (inp.vs, inp.ep, inp.speaker_id, inp.min_confidence)).fetchall()

    @app.get("/api/bulk_confirm/{vs}/{ep}")
    def bulk_confirm_preview(vs: str, ep: int, speaker_id: str, min_confidence: float = 0.8):
        c = con()
        rows = bulk_rows(c, BulkConfirmIn(vs=vs, ep=ep, speaker_id=speaker_id, min_confidence=min_confidence))
        n_auto = c.execute("""SELECT COUNT(*) FROM labels l JOIN utterances u USING (utt_id) WHERE u.version_season=? AND u.episode=?
                              AND u.segment='body' AND l.source='auto' AND l.speaker_id=?""", (vs, ep, speaker_id)).fetchone()[0]
        return {"n": len(rows), "secs": round(sum(r["dur"] for r in rows), 1), "n_auto": n_auto}

    @app.post("/api/bulk_confirm")
    def bulk_confirm(inp: BulkConfirmIn):
        """Turn one castaway's auto labels above a confidence into human labels without listening to each. The old
        label is kept in top_candidates so /api/bulk_unconfirm can put it back. Do it after the episode's audit:
        confirmed lines leave the audit pool."""
        c = con()
        ids = []
        for r in bulk_rows(c, inp):
            c.execute("UPDATE labels SET source='human', confidence=1.0, top_candidates=?, labeled_at=datetime('now') WHERE utt_id=?",
                      (json.dumps({"note": "bulk:confirm", "prev": {"confidence": r["confidence"], "top_candidates": r["top_candidates"]}}),
                       r["utt_id"]))
            ids.append(r["utt_id"])
        if ids:
            c.execute("UPDATE review_queue SET resolved=1 WHERE utt_id IN (%s)" % ",".join("?" * len(ids)), ids)
        c.commit()
        return {"ok": True, "n": len(ids), "utt_ids": ids}

    @app.post("/api/bulk_unconfirm")
    def bulk_unconfirm(inp: UnlabelIn):
        c = con()
        n = 0
        for uid in inp.utt_ids:
            r = c.execute("SELECT top_candidates FROM labels WHERE utt_id=? AND source='human'", (uid,)).fetchone()
            try:
                t = json.loads(r[0] or "{}") if r else {}
            except ValueError:
                t = {}
            if t.get("note") != "bulk:confirm":
                continue
            p = t.get("prev") or {}
            c.execute("UPDATE labels SET source='auto', confidence=?, top_candidates=? WHERE utt_id=?",
                      (p.get("confidence"), p.get("top_candidates"), uid))
            n += 1
        c.commit()
        return {"ok": True, "n": n}

    return app


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _wilson_low(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return max(0.0, (centre - half) / d)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
