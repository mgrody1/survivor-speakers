"""Text prior — a second, text-only vote on who is speaking, from a local LLM (oMLX, OpenAI-compatible).

Why: the audio bank is thin in early episodes and empty for castaways who never got an explicit SDH name, so the
review queue fills with long runs the bank cannot place ("I just graduated from business school", "us girls",
"Any votes cast against Gabe will not count"). A reader with the cast sheet resolves most of those from the words:
first-person facts against bios, exclusion by who is talked about or addressed, tribe / gender consistency, and
who spoke in the neighbouring runs. That is what this asks the model to do, one structured call per run.

It never writes labels itself. `text_prior` rows are stored per run and `assign` may combine them with the audio
decision (config `text_prior.use_in_assign`); the review UI shows them as a suggestion.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS text_prior (
    version_season TEXT NOT NULL,
    episode        INTEGER NOT NULL,
    run_id         INTEGER NOT NULL,
    sub            INTEGER NOT NULL,
    first_utt_id   TEXT,
    utt_ids        TEXT,               -- json list
    speaker_id     TEXT,               -- resolved castaway id / HOST_* / NULL for UNKNOWN
    speaker_name   TEXT,               -- what the model said
    confidence     REAL,
    reason         TEXT,
    excluded       TEXT,               -- json list of ids the model ruled out
    model          TEXT,
    prompt_hash    TEXT,
    latency_s      REAL,
    computed_at    TEXT,
    PRIMARY KEY (version_season, episode, run_id, sub)
);
"""


@dataclass
class LLMCfg:
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "Qwen3.6-35B-A3B-MLX-8bit"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 600
    timeout_s: float = 180.0
    concurrency: int = 4
    disable_thinking: bool = True          # Qwen3.x: pass chat_template_kwargs.enable_thinking=false
    context_runs: int = 3                  # neighbouring runs shown before and after
    use_in_assign: bool = False
    min_confidence: float = 0.8            # for use_in_assign

    @classmethod
    def from_settings(cls, s: Settings, base_url: str | None = None, model: str | None = None) -> "LLMCfg":
        """config text_prior.* <- .env / environment (OMLX_BASE_URL, OMLX_API_KEY, OMLX_MODEL) <- CLI overrides."""
        import os

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        d = dict(s.raw.get("text_prior", {}))
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.base_url = base_url or os.environ.get("OMLX_BASE_URL") or cfg.base_url
        cfg.model = model or os.environ.get("OMLX_MODEL") or cfg.model
        cfg.api_key = os.environ.get("OMLX_API_KEY") or cfg.api_key
        return cfg


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "description": "one name from the candidate list, HOST, or UNKNOWN"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "excluded": {"type": "array", "items": {"type": "string"}, "description": "candidates the text rules out"},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["speaker", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You identify who is speaking in a passage of a Survivor (US) episode transcript, using only the text and the
cast sheet. Rules of thumb that hold for this show:
- People say other players' names constantly and their own almost never; a passage that talks ABOUT someone in the
  third person, or addresses them by name, is almost never spoken by that person.
- First-person facts (age, job, hometown, tribe, gender words like "us girls", who they are aligned with) narrow the
  speaker; check them against the cast sheet.
- The host (Jeff Probst) runs challenges ("Survivors ready? Go!", "X has her piece", "Come on in, guys") and tribal
  council ("I'll read the votes", "grab your torch", "the tribe has spoken", asks players questions by name).
- Confessionals are one player speaking to camera about the game; camp dialogue is several players.
- Neighbouring passages and their known speakers are given for context; the previous speaker often continues, and
  a question addressed to X is usually answered by X.
Answer with JSON only: {"speaker": <one candidate short name, or HOST, or UNKNOWN>, "confidence": 0..1,
"excluded": [names the text rules out], "reason": <one sentence>}. Say UNKNOWN with low confidence when the text
does not decide it; do not guess from vibes."""


# ----------------------------------------------------------------------------- prompt


def cast_sheet(resolver, vs: str, ep: int, candidates: frozenset[str]) -> tuple[str, dict[str, str], dict[str, str]]:
    """Markdown cast sheet for the candidates + name<->id maps."""
    bios = resolver.bios(vs, ep)
    hosts, host_id = resolver.host_names(vs)
    lines = ["| name | tribe | gender | age | occupation | from |", "|---|---|---|---|---|---|"]
    name2id: dict[str, str] = {}
    id2name: dict[str, str] = {}
    for cid, b in sorted(bios.items(), key=lambda kv: (kv[1].get("tribe") or "", kv[1]["name"])):
        if cid not in candidates:
            continue
        nm = b["name"]
        name2id[nm.lower()] = cid
        id2name[cid] = nm
        age = f"{int(b['age'])}" if b.get("age") is not None else ""
        lines.append(f"| {nm} ({b.get('full_name') or nm}) | {b.get('tribe') or ''} | {b.get('gender') or ''} | {age} | "
                     f"{b.get('occupation') or ''} | {', '.join(x for x in (b.get('city'), b.get('state')) if x)} |")
    if host_id in candidates:
        for k in ("host", "jeff", "probst", "jeff probst"):
            name2id[k] = host_id
        id2name[host_id] = "HOST"
        lines.append("| HOST (Jeff Probst) | — | Male | | host: runs challenges and tribal council | |")
    return "\n".join(lines), name2id, id2name


def build_prompt(sheet: str, id2name: dict[str, str], target: dict, before: list[dict], after: list[dict],
                 vs: str, ep: int, mentioned: list[str] | None = None) -> str:
    def fmt(r: dict, mark: str) -> str:
        who = r.get("known")
        who = f"[{id2name.get(who, who)}]" if who else "[?]"
        return f"{mark} {r['mmss']} {who} {r['text']}"

    ctx_b = "\n".join(fmt(r, "  ") for r in before) or "  (start of episode)"
    ctx_a = "\n".join(fmt(r, "  ") for r in after) or "  (end of episode)"
    hints = ""
    if mentioned:
        hints += ("\nNames that appear in the PASSAGE: " + ", ".join(id2name.get(m, m) for m in mentioned)
                  + " — people are named by OTHERS; the speaker is almost never one of these. Exclude them unless the"
                    " passage is unmistakably someone speaking about themselves in the third person.")
    if target.get("audio_top"):
        hints += "\nAudio model's guesses (weak, may be wrong): " + ", ".join(
            f"{id2name.get(s, s)} {x:.2f}" for s, x in target["audio_top"])
    return (f"Season {vs}, episode {ep}. Cast present in this episode:\n{sheet}\n\n"
            f"Context before (with known speakers where available):\n{ctx_b}\n\n"
            f">>> PASSAGE TO IDENTIFY ({target['mmss']}, {target['dur']:.0f} s, {target['domain']}):\n{target['text']}\n\n"
            f"Context after:\n{ctx_a}{hints}\n\n"
            "Who speaks the PASSAGE? Use only the cast sheet and this transcript — do not use anything you believe you "
            "know about this season from elsewhere. JSON only.")


# ----------------------------------------------------------------------------- client


def chat_json(cfg: LLMCfg, system: str, user: str) -> tuple[dict, float]:
    body: dict = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "speaker", "schema": RESPONSE_SCHEMA, "strict": True}},
    }
    if cfg.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["max_tokens"] = max(cfg.max_tokens, 4000)          # room for the reasoning block
    req = urllib.request.Request(cfg.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=_headers(cfg))
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
        out = json.loads(resp.read().decode())
    content = out["choices"][0]["message"]["content"]
    # strip a leaked <think> block or code fence, then parse the first JSON object
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    content = content.strip().strip("`")
    if content.startswith("json"):
        content = content[4:]
    start, end = content.find("{"), content.rfind("}")
    parsed = json.loads(content[start:end + 1]) if start >= 0 else {"speaker": "UNKNOWN", "confidence": 0, "reason": "unparseable"}
    return parsed, time.time() - t0


def _headers(cfg: LLMCfg) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if cfg.api_key:
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def ping(cfg: LLMCfg) -> list[str]:
    req = urllib.request.Request(cfg.base_url.rstrip("/") + "/models", headers=_headers(cfg))
    with urllib.request.urlopen(req, timeout=10) as resp:
        return [m["id"] for m in json.loads(resp.read().decode()).get("data", [])]


# ----------------------------------------------------------------------------- driver


def run_text_prior(settings: Settings, con: sqlite3.Connection, vs: str, ep: int, resolver, only_queued: bool = True,
                   limit: int | None = None, force: bool = False, cfg: LLMCfg | None = None,
                   eval_only: bool = False, write: bool = True) -> pd.DataFrame:
    """Score runs of an episode with the LLM. Uses the run table `assign` wrote (reports/assign_<vs>E<ep>.csv is a
    copy; the DB `labels` + `review_queue` are the source). Groups = assign's (run_id, sub) groups, recovered from
    labels.run_id / queue payload; falls back to segment runs when assign has not run."""
    con.executescript(SCHEMA)
    cfg = cfg or LLMCfg.from_settings(settings)
    models = ping(cfg)
    if cfg.model not in models:
        log.warning("model %r not in %s; using it anyway (oMLX may alias)", cfg.model, models)
    from .stage_bank import load_utterances

    utts = load_utterances(con, vs, ep)
    utts = utts[utts.segment == "body"].sort_values("idx").reset_index(drop=True)
    host_id = settings.franchise_for(vs).host_id
    candidates = frozenset(resolver.present(vs, ep)) | {host_id}
    sheet, name2id, id2name = cast_sheet(resolver, vs, ep, candidates)

    lab = pd.read_sql_query("SELECT utt_id, speaker_id, source, confidence, run_id AS lab_run FROM labels", con).set_index("utt_id")
    queue = pd.read_sql_query("SELECT utt_id, reason, payload FROM review_queue WHERE resolved=0", con).set_index("utt_id")
    utts["known"] = utts.utt_id.map(lambda u: lab.speaker_id.get(u) if u in lab.index and lab.source.get(u) in ("sdh", "human", "chyron", "auto") else None)
    utts["queued"] = utts.utt_id.isin(queue.index)
    utts["explicit_id"] = utts.sdh_speaker_id.where(utts.name_explicit & utts.sdh_resolution.isin(["cast", "alias", "host"]))

    # groups: contiguous utterances with the same run_id, cut where assign split (label run_id + auto/queue pattern
    # is not enough to recover subs exactly, so approximate: split where `known` changes between two explicit ids)
    groups: list[list[int]] = []
    for rid, g in utts.groupby("run_id", sort=False):
        idx = list(g.index)
        cur = [idx[0]]
        for i in idx[1:]:
            a, b = utts.loc[cur[-1], "explicit_id"], utts.loc[i, "explicit_id"]
            if pd.notna(a) and pd.notna(b) and a != b:
                groups.append(cur)
                cur = [i]
            else:
                cur.append(i)
        groups.append(cur)

    def row(idxs: list[int]) -> dict:
        g = utts.loc[idxs]
        top = None
        first = g.utt_id.iloc[0]
        if first in queue.index:
            try:
                top = [(s, float(x)) for s, x in (json.loads(queue.payload[first]) or {}).get("top", [])][:3]
            except Exception:  # noqa: BLE001
                top = None
        elif first in lab.index and lab.source[first] == "auto":
            top = None
        known = g.known.dropna()
        return {"idxs": idxs, "run_id": int(g.run_id.iloc[0]), "utt_ids": g.utt_id.tolist(),
                "mmss": f"{int(g.start_s.min() // 60):02d}:{int(g.start_s.min() % 60):02d}",
                "dur": float(g.end_s.max() - g.start_s.min()), "domain": g.domain_hint.iloc[0],
                "text": " ".join(t or "" for t in g.text)[:1200],
                "known": known.iloc[0] if len(known) and known.nunique() == 1 else None,
                "explicit": g.explicit_id.dropna().iloc[0] if g.explicit_id.notna().any() else None,
                "queued": bool(g.queued.any()), "audio_top": top}

    rows = [row(idxs) for idxs in groups]
    # mentions are computed here, on the main thread: the resolver's sqlite connection is not shareable across
    # the worker threads below
    resolver.mention_patterns(vs)
    for r in rows:
        r["mentioned"] = [c for c in candidates if resolver.mentions(r["text"], c, vs)]
    if eval_only:                                        # only runs with an explicit name: a fast A/B of prompts / models
        targets = [i for i, r in enumerate(rows) if r["explicit"] is not None]
    else:
        targets = [i for i, r in enumerate(rows) if (r["queued"] or not only_queued or r["explicit"] is not None)]
    if limit:
        targets = targets[:limit]
    done = {(r["run_id"], r["sub"]) for r in con.execute(
        "SELECT run_id, sub FROM text_prior WHERE version_season=? AND episode=?", (vs, ep))} if not force else set()

    def work(i: int) -> dict | None:
        r = rows[i]
        sub = sum(1 for j in range(i) if rows[j]["run_id"] == r["run_id"])
        if (r["run_id"], sub) in done:
            return None
        before = [{**rows[j], "known": rows[j]["known"]} for j in range(max(0, i - cfg.context_runs), i)]
        after = [rows[j] for j in range(i + 1, min(len(rows), i + 1 + cfg.context_runs))]
        # never leak the target's own explicit label into the prompt; neighbours' labels are fair game
        mentioned = r["mentioned"]
        prompt = build_prompt(sheet, id2name, r, before, after, vs, ep, mentioned)
        try:
            ans, dt = chat_json(cfg, SYSTEM, prompt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            log.warning("run %d: LLM call failed: %s", r["run_id"], e)
            return None
        name = str(ans.get("speaker", "UNKNOWN")).strip()
        sid = name2id.get(name.lower().split(" (")[0])
        if name.upper() == "HOST":
            sid = host_id
        excluded = [name2id.get(str(x).lower()) for x in ans.get("excluded", []) or []]
        conf = float(ans.get("confidence", 0) or 0)
        reason = str(ans.get("reason", ""))[:400]
        # post-check: the model still picks people the passage names (US47E02: "me and Sue are locked in" -> Sue).
        # A mentioned person is not impossible (Rome talks about Rome) but it is the dominant error, so cap it.
        if sid and sid in mentioned:
            conf = min(conf, 0.3)
            reason = "[names a person mentioned in the passage] " + reason
        return {"version_season": vs, "episode": ep, "run_id": r["run_id"], "sub": sub, "first_utt_id": r["utt_ids"][0],
                "utt_ids": json.dumps(r["utt_ids"]), "speaker_id": sid, "speaker_name": name,
                "confidence": conf, "reason": reason,
                "excluded": json.dumps([x for x in excluded if x]), "model": cfg.model,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:12], "latency_s": round(dt, 2),
                "explicit": r["explicit"], "queued": r["queued"], "mmss": r["mmss"], "dur": r["dur"], "text": r["text"][:120]}

    t0 = time.time()
    out = []
    with cf.ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        for res in ex.map(work, targets):
            if res is not None:
                out.append(res)
                if not write:
                    continue
                con.execute("""INSERT OR REPLACE INTO text_prior (version_season, episode, run_id, sub, first_utt_id, utt_ids,
                               speaker_id, speaker_name, confidence, reason, excluded, model, prompt_hash, latency_s, computed_at)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                            (vs, ep, res["run_id"], res["sub"], res["first_utt_id"], res["utt_ids"], res["speaker_id"],
                             res["speaker_name"], res["confidence"], res["reason"], res["excluded"], res["model"],
                             res["prompt_hash"], res["latency_s"]))
                if len(out) % 20 == 0:
                    con.commit()
                    log.info("text prior %s E%02d: %d/%d done, %.1fs/call", vs, ep, len(out), len(targets),
                             (time.time() - t0) / len(out))
    con.commit()
    df = pd.DataFrame(out)
    log.info("text prior %s E%02d: %d runs in %.0fs", vs, ep, len(df), time.time() - t0)
    return df


def evaluate(df: pd.DataFrame) -> dict:
    """Precision of the text prior against explicit SDH names, by confidence bucket."""
    m = df[df.explicit.notna()].copy()
    if m.empty:
        return {}
    m["hit"] = m.speaker_id == m.explicit
    m["said"] = m.speaker_id.notna()
    out = {"n_explicit": int(len(m)), "n_answered": int(m.said.sum()),
           "precision_answered": round(float(m[m.said].hit.mean()), 3) if m.said.any() else None}
    for thr in (0.5, 0.7, 0.8, 0.9):
        mm = m[m.said & (m.confidence >= thr)]
        out[f"precision_conf>={thr}"] = round(float(mm.hit.mean()), 3) if len(mm) else None
        out[f"n_conf>={thr}"] = int(len(mm))
    q = m[m.queued]
    out["n_explicit_and_queued"] = int(len(q))
    if len(q):
        qq = q[q.said & (q.confidence >= 0.8)]
        out["precision_queued_conf>=0.8"] = round(float(qq.hit.mean()), 3) if len(qq) else None
        out["n_queued_conf>=0.8"] = int(len(qq))
    return out
