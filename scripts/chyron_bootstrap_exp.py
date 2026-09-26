"""Could name cards alone start a season? A dry run on a named season, entirely inside a throwaway copy of the DB.

On the copy, for the chosen episodes: every label, queue row and bank entry is deleted and every SDH name except the
host's is erased (the host bank would come from other seasons: Probst's voice is the same everywhere). Then the
cached name-card OCR writes `chyron` labels, the bank is fitted from them, the episodes are assigned, and the bank is
refitted from chyron + confident auto labels and assigned again (the refit loop, with no human in it).

The auto labels are scored against the real DB's known speakers: human labels (not UNKNOWN/NOSPEECH/OTHER), then
explicit SDH names. The real DB is only read.

    uv run python scripts/chyron_bootstrap_exp.py US47 1 2 3"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from survspk import db as dbm
from survspk.aliases import Resolver
from survspk.chyron import chyron_episode
from survspk.config import load_settings
from survspk.stage_assign import assign_episode
from survspk.stage_bank import build_bank


def truth(con, vs, eps):
    out = {}
    for r in con.execute(f"""SELECT u.utt_id, u.episode, u.end_s-u.start_s AS dur, u.segment, u.sdh_speaker_id, u.sdh_resolution, u.flags,
                                   (SELECT speaker_id FROM labels l WHERE l.utt_id=u.utt_id AND l.source='human') AS human
                            FROM utterances u WHERE u.version_season=? AND u.episode IN ({",".join("?" * len(eps))})""", (vs, *eps)):
        if r["segment"] != "body":
            continue
        f = json.loads(r["flags"] or "{}")
        if r["human"] and r["human"] not in ("UNKNOWN", "NOSPEECH", "OTHER"):
            out[r["utt_id"]] = (r["human"], r["episode"], r["dur"])
        elif r["sdh_resolution"] in ("cast", "alias", "host") and r["sdh_speaker_id"] and not f.get("name_inherited", True):
            out[r["utt_id"]] = (r["sdh_speaker_id"], r["episode"], r["dur"])
    return out


def score(con, vs, eps, tr, sources=("auto",)):
    pred = {r[0]: r[1] for r in con.execute(
        f"""SELECT l.utt_id, l.speaker_id FROM labels l JOIN utterances u USING (utt_id)
            WHERE u.version_season=? AND u.episode IN ({",".join("?" * len(eps))}) AND l.source IN ({",".join("?" * len(sources))})""",
        (vs, *eps, *sources))}
    rows = []
    for ep in eps:
        t = {u: v for u, v in tr.items() if v[1] == ep}
        hit = [(u, v) for u, v in t.items() if u in pred]
        ok = [(u, v) for u, v in hit if pred[u] == v[0]]
        long_ = [(u, v) for u, v in hit if v[2] >= 2.0]
        rows.append((ep, len(t), len(hit), len(ok), sum(v[2] for _, v in t.items()), sum(v[2] for _, v in hit),
                     sum(v[2] for _, v in ok), len(long_), sum(pred[u] == v[0] for u, v in long_)))
    return rows


def show(title, rows):
    print(f"\n{title}")
    print("  ep  known  labelled  right   precision  (>=2 s)   known time labelled")
    for ep, n, h, ok, T, HT, OT, nl, okl in rows:
        print(f"  E{ep:02d} {n:6} {h:9} {ok:6}   {100 * ok / h if h else 0:5.1f}%     {100 * okl / nl if nl else 0:5.1f}%   {100 * HT / T if T else 0:5.1f}%")


def main(vs, eps):
    s = load_settings()
    real = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    real.row_factory = sqlite3.Row
    tr = truth(real, vs, eps)
    show("real pipeline (bank from captions + your labels; its auto labels, scored on the same known lines)", score(real, vs, eps, tr))

    tmp = Path(tempfile.mkdtemp()) / "survspk_chyron_exp.sqlite"
    dst = sqlite3.connect(tmp)
    real.backup(dst)
    dst.close()
    s2 = s.model_copy(update={"layout": s.layout.model_copy(update={"db": str(tmp)})})
    assert s2.db_path == tmp and s.db_path != tmp
    con = dbm.init_db(s2.db_path, "WAL")
    q = ",".join("?" * len(eps))
    host = s.franchise_for(vs).host_id
    con.execute(f"DELETE FROM labels WHERE utt_id IN (SELECT utt_id FROM utterances WHERE version_season=? AND episode IN ({q}))", (vs, *eps))
    con.execute(f"DELETE FROM review_queue WHERE utt_id IN (SELECT utt_id FROM utterances WHERE version_season=? AND episode IN ({q}))", (vs, *eps))
    con.execute("DELETE FROM speaker_bank WHERE version_season=?", (vs,))
    con.execute(f"""UPDATE utterances SET sdh_speaker_id=NULL, sdh_resolution=NULL, sdh_name=NULL
                    WHERE version_season=? AND episode IN ({q}) AND COALESCE(sdh_speaker_id,'') != ?""", (vs, *eps, host))
    con.commit()
    r = Resolver(s2)
    for ep in eps:
        st = chyron_episode(s2, con, vs, ep, r, use_cache=True, write_labels=True, cache_path=None)
        print(f"E{ep:02d}: {st['n_hits']} cards, {st['n_labels']} chyron labels, castaways carded {len(st['castaways_hit'])}")
    if ORACLE:
        # what a person confirming each card would give: the card's label moves to a line within [t-8, t+4] s that the
        # carded castaway is known to speak; a card with no such line is dropped
        con.execute(f"DELETE FROM labels WHERE source='chyron' AND utt_id IN (SELECT utt_id FROM utterances WHERE version_season=? AND episode IN ({q}))", (vs, *eps))
        n = 0
        for h in con.execute(f"SELECT * FROM chyron_hits WHERE version_season=? AND episode IN ({q})", (vs, *eps)).fetchall():
            cands = con.execute("""SELECT utt_id, start_s, end_s, text, domain_hint FROM utterances WHERE version_season=? AND episode=?
                                   AND segment='body' AND start_s BETWEEN ? AND ? ORDER BY ABS(start_s - (? - 3))""",
                                (vs, h["episode"], h["t_s"] - 8, h["t_s"] + 4, h["t_s"])).fetchall()
            u = next((c for c in cands if c["utt_id"] in tr and tr[c["utt_id"]][0] == h["castaway_id"]), None)
            if u is None:
                continue
            con.execute("""INSERT OR REPLACE INTO labels (utt_id, speaker_id, source, confidence, top_candidates, domain, labeled_at,
                           version_season, episode, start_s, end_s, text) VALUES (?,?,'chyron',1.0,'{}',?,datetime('now'),?,?,?,?,?)""",
                        (u["utt_id"], h["castaway_id"], u["domain_hint"], vs, h["episode"], u["start_s"], u["end_s"], u["text"]))
            n += 1
        con.commit()
        print(f"oracle: {n} cards moved to a line their castaway speaks")
    show("chyron labels themselves", score(con, vs, eps, tr, ("chyron",)))
    for rnd in (1, 2, 3):
        b = build_bank(s2, con, vs, list(eps), as_of=max(eps), resolver=r)
        for ep in eps:
            assign_episode(s2, con, vs, ep, bank_as_of=max(eps), resolver=r, write=True)
        print(f"\nround {rnd}: bank {b.get('n_speakers')} speakers, kept {b.get('n_kept')}, dropped {b.get('n_dropped')}")
        show(f"round {rnd}: auto labels from a bank built on name cards" + (" + confident auto" if rnd > 1 else ""),
             score(con, vs, eps, tr))
    # held-out: bank from all but the last episode, assign the last
    con.execute("DELETE FROM speaker_bank WHERE version_season=?", (vs,))
    con.execute(f"DELETE FROM labels WHERE source='auto' AND utt_id IN (SELECT utt_id FROM utterances WHERE version_season=? AND episode IN ({q}))", (vs, *eps))
    con.commit()
    build_bank(s2, con, vs, list(eps[:-1]), as_of=eps[-2], resolver=r)
    for ep in eps[:-1]:
        assign_episode(s2, con, vs, ep, bank_as_of=eps[-2], resolver=r, write=True)
    build_bank(s2, con, vs, list(eps[:-1]), as_of=eps[-2], resolver=r)
    assign_episode(s2, con, vs, eps[-1], bank_as_of=eps[-2], resolver=r, write=True)
    show(f"held out: bank from E{eps[0]:02d}-E{eps[-2]:02d} name cards (+1 refit), assigning E{eps[-1]:02d}", score(con, vs, eps[-1:], tr))
    print(f"\n(copy at {tmp}; the real DB was only read)")


ORACLE = False

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--oracle"]
    ORACLE = "--oracle" in sys.argv
    main(args[0], [int(x) for x in args[1:]])
