"""Score chyron anchoring rules against the speakers we already know (human labels, then explicit SDH names, then
inherited SDH names) on episodes whose raw OCR is cached (`survspk chyron VS EP` writes it).

    uv run python scripts/chyron_anchor_eval.py US47 1 2 3

Per rule: how many cards it anchors to a line whose speaker is known, and how often that speaker is the card's
castaway. Nothing is written."""

from __future__ import annotations

import json
import sqlite3
import sys

from survspk.aliases import Resolver
from survspk.chyron import (CastMatcher, ChyronCfg, Hit, dedupe, drop_bursts, frame_hits, ocr_cache_path,
                            read_ocr_cache)
from survspk.config import load_settings


def hits_for(s, con, r, vs, ep, cfg):
    m = CastMatcher(r, vs, ep, cfg)
    raw = []
    for t, lines in read_ocr_cache(ocr_cache_path(s, vs, ep)):
        hit, _ = frame_hits(lines, m, cfg)
        if hit:
            cid, score, text = hit
            raw.append(Hit(t, cid, text, 0.0, score, "host" if cid == m.host_id else "cast"))
    kept, burst = drop_bursts(dedupe(raw, cfg.dedupe_s), cfg.burst_n, cfg.burst_s)
    return kept, burst


def truth_rows(con, vs, ep):
    rows = con.execute("""SELECT u.utt_id, u.start_s, u.end_s, u.segment, u.sdh_speaker_id, u.sdh_resolution, u.flags,
                                 (SELECT speaker_id FROM labels l WHERE l.utt_id=u.utt_id AND l.source='human') AS human
                          FROM utterances u WHERE u.version_season=? AND u.episode=? ORDER BY u.start_s""", (vs, ep)).fetchall()
    out = []
    for r_ in rows:
        f = json.loads(r_["flags"] or "{}")
        sdh = r_["sdh_speaker_id"] if r_["sdh_resolution"] in ("cast", "alias", "host") else None
        if r_["human"] and r_["human"] not in ("UNKNOWN", "NOSPEECH", "OTHER"):
            who, kind = r_["human"], "human"
        elif sdh and not f.get("name_inherited", True):
            who, kind = sdh, "explicit"
        elif sdh:
            who, kind = sdh, "inherited"
        else:
            who, kind = None, None
        out.append({"utt_id": r_["utt_id"], "start_s": r_["start_s"], "end_s": r_["end_s"], "segment": r_["segment"],
                    "who": who, "kind": kind})
    return out


def rule_start(utts, h, lo=6.0, hi=1.0, ideal=3.0):          # the current rule
    c = [u for u in utts if h.t_s - lo <= u["start_s"] <= h.t_s + hi and u["segment"] == "body"]
    return min(c, key=lambda u: abs((h.t_s - u["start_s"]) - ideal)) if c else None


def rule_overlap(utts, h, pad=0.5):                          # the line most on air while the card is up
    a, b = h.t_s - pad, (h.t_end or h.t_s) + pad
    c = [(min(b, u["end_s"]) - max(a, u["start_s"]), u) for u in utts if u["segment"] == "body"]
    c = [x for x in c if x[0] > 0]
    return max(c, key=lambda x: x[0])[1] if c else None


def rule_at(utts, h, dt=0.5):                                # the line being spoken just after the card appears
    t = h.t_s + dt
    c = [u for u in utts if u["start_s"] <= t <= u["end_s"] and u["segment"] == "body"]
    return c[0] if c else None


RULES = {"start-3s (current)": rule_start, "overlap card on air": rule_overlap, "spoken at card+0.5s": rule_at}


def main(vs, eps):
    s = load_settings()
    con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    r = Resolver(s)
    cfg = ChyronCfg.from_settings(s)
    tot = {k: [0, 0, 0] for k in RULES}                      # anchored with known speaker, right, anchored at all
    for ep in eps:
        hits, burst = hits_for(s, con, r, vs, ep, cfg)
        utts = truth_rows(con, vs, ep)
        hosts = sum(h.kind == "host" for h in hits)
        print(f"\n{vs} E{ep:02d}: {len(hits)} cards kept ({hosts} host), {len(burst)} dropped as credits; "
              f"castaways carded {len({h.castaway_id for h in hits if h.kind == 'cast'})}")
        for name, fn in RULES.items():
            n_known = n_ok = n_any = 0
            for h in hits:
                u = fn(utts, h)
                if u is None:
                    continue
                n_any += 1
                if u["who"]:
                    n_known += 1
                    n_ok += u["who"] == h.castaway_id
            tot[name][0] += n_known; tot[name][1] += n_ok; tot[name][2] += n_any
            print(f"  {name:22} anchored {n_any:3}  known {n_known:3}  right {n_ok:3}  "
                  f"({100 * n_ok / n_known if n_known else 0:.0f}%)")
    print("\nall episodes")
    for name, (k, ok, a) in tot.items():
        print(f"  {name:22} anchored {a:3}  known {k:3}  right {ok:3}  ({100 * ok / k if k else 0:.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
