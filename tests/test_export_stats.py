"""export-stats: derived speech tables from attributed lines (synthetic episode, no survivoR needed)."""
import csv
import json

from survspk import db as dbm
from survspk import export_stats as ex

CTX = dict(cast={"A1": "Kenzie", "B2": "Charlie", "C3": "Charlie", "D4": "Ben"},
           alts={"A1": ["kenzie"], "B2": ["charlie"], "C3": ["charlie"], "D4": ["benjamin", "ben"],
                 "HOST_US": ["jeff probst", "probst", "jeff"]},
           present={"A1", "B2", "D4"}, host_id="HOST_US")


def ctx(**kw):
    return ex.EpisodeCtx(**(CTX | kw))


def test_names_direct_address_and_shared_first_names():
    c = ctx()
    assert ex.named_in("Kenzie, what do you think?", c)[:2] == ({"A1"}, {"A1"})
    assert ex.named_in("Come on, Ben!", c)[:2] == ({"D4"}, {"D4"})
    assert ex.named_in("I think Kenzie is running this.", c)[:2] == ({"A1"}, set())
    assert ex.named_in("Kenzieland", c)[0] == set()                          # whole words only
    assert ex.named_in("Charlie is a threat.", c) == ({"B2"}, set(), 0)      # C3 is out of the game: B2
    both = ctx(present={"A1", "B2", "C3", "D4"})
    assert ex.named_in("Charlie is a threat.", both) == (set(), set(), 1)    # both in the game: skipped
    assert ex.named_in("Jeff, we're ready.", c)[:2] == ({"HOST_US"}, {"HOST_US"})


def U(i, t0, t1, spk, text="", seg="body", dom="field", src="sdh", p=None, run=None, words=3):
    return {"utt_id": f"u{i}", "idx": i, "start": t0, "end": t1, "segment": seg, "domain": dom, "text": text,
            "align_ok": 1, "n_words": words, "speaker": spk, "source": src if spk else None, "p_right": p,
            "run_id": run if run is not None else i, "dur": t1 - t0}


def test_episode_tables_counts_and_denominators():
    utts = [
        U(0, 0, 10, "A1", "Previously on Survivor", seg="recap"),
        U(1, 20, 24, "HOST_US", "Kenzie, you got it?"),
        U(2, 25, 27, "A1", "Yes, Jeff."),
        U(3, 28, 30, "D4", "Nice work, Kenzie."),
        U(4, 31, 33, None, "Ben needs to go."),                             # no speaker yet
        U(5, 40, 50, "A1", "Charlie and Ben are close.", dom="confessional", src="auto", p=0.9, run=7),
        U(6, 50, 56, "A1", "I have to move.", dom="confessional", src="auto", p=0.8, run=7),
        U(7, 70, 72, "D4", "Hey.", src="human"),
        U(8, 90, 92, "B2", "What?", src="chyron"),                          # 18 s later: not a reply
        U(9, 300, 310, "D4", "Next week", seg="preview"),
    ]
    t = ex.episode_tables(utts, ctx(), "US46", 4, {"survivor_confessionals": {"A1": (2, 40.0)}})
    st = {r["castaway_id"]: r for r in t["speech_stats"]}
    assert set(st) == {"A1", "B2", "D4"}                                   # C3 not in the game and never spoke
    a = st["A1"]
    assert a["speech_seconds"] == 18 and a["confessional_seconds"] == 16 and a["field_seconds"] == 2
    assert a["confessional_runs"] == 1 and a["turns"] == 2 and a["recap_seconds"] == 10
    assert a["seconds_auto"] == 16 and a["seconds_caption"] == 2
    assert a["expected_wrong_seconds"] == 2.2 and a["est_precision"] == round(1 - 2.2 / 18, 3)
    assert a["mentions_received"] == 2 and a["host_mentions_received"] == 1 and a["host_addresses_received"] == 1
    assert a["direct_addresses_received"] == 2 and a["namers"] == 2 and a["mentions_given"] == 3   # Jeff, Charlie, Ben
    assert a["survivor_confessional_count"] == 2 and st["B2"]["survivor_confessional_count"] is None
    assert st["D4"]["mentions_received"] == 2                              # the unattributed line counts
    assert abs(sum(r["speech_share"] for r in st.values()) - 1) < 1e-3
    m = {(r["source_id"], r["target_id"]): r for r in t["speech_mentions"]}
    assert m[("UNKNOWN", "D4")]["lines_naming"] == 1 and m[("A1", "B2")]["in_confessional"] == 1
    assert m[("A1", "HOST_US")]["direct_address"] == 1
    tr = {(r["from_id"], r["to_id"]): r["transitions"] for r in t["speech_interactions"]}
    assert tr == {("HOST_US", "A1"): 1, ("A1", "D4"): 1}                   # D4 -> unknown and unknown -> nothing
    q = t["speech_quality"][0]
    assert q["body_seconds"] == 30 and q["body_lines"] == 8
    assert q["unknown_seconds"] == 2 and q["host_seconds"] == 4 and q["recap_seconds"] == 10 and q["preview_seconds"] == 10
    assert q["speaker_coverage"] == round((q["castaway_seconds"] + 4) / q["body_seconds"], 4)
    assert abs(q["share_human"] + q["share_namecard"] + q["share_caption"] + q["share_auto"] - 1) < 1e-3
    assert q["auto_expected_precision"] == round((0.9 * 10 + 0.8 * 6) / 16, 4)
    assert q["transitions"] == 2 and q["transitions_unknown"] == 1


def test_load_episode_and_csv_columns_match_dictionary(tmp_path):
    con = dbm.init_db(tmp_path / "t.sqlite", "DELETE")
    con.execute("""INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, domain_hint,
                   align_ok, flags, n_words) VALUES ('US46_E04_U0000','US46',4,0,1.0,3.5,'Kenzie, go.','body','field',1,?,2)""",
                (json.dumps({"run_id": 3}),))
    con.execute("""INSERT INTO utterances (utt_id, version_season, episode, idx, start_s, end_s, text, segment, domain_hint,
                   align_ok, flags, n_words) VALUES ('US46_E04_U0001','US46',4,1,4.0,5.0,'Ok.','body','field',0,'{}',1)""")
    con.execute("INSERT INTO labels (utt_id, speaker_id, source, confidence) VALUES ('US46_E04_U0000','D4','auto',0.7)")
    con.execute("INSERT INTO labels (utt_id, speaker_id, source) VALUES ('US46_E04_U0001','UNKNOWN','human')")
    con.commit()
    utts = ex.load_episode(con, "US46", 4)
    assert utts[0]["run_id"] == 3 and utts[0]["p_right"] == 0.7 and utts[1]["speaker"] is None
    t = ex.episode_tables(utts, ctx(), "US46", 4, ex.episode_extra(con, None, "US46", 4))
    assert t["speech_quality"][0]["queue_open_lines"] == 0 and t["speech_quality"][0]["audit_n"] is None
    for name, rows in t.items():
        p = tmp_path / f"{name}.csv"
        ex.write_csv(p, name, rows)                        # raises if a row has a column the dictionary lacks
        header = next(csv.reader(open(p)))
        assert header == [f[0] for f in ex.DICTIONARY[name]]
    assert "Kenzie, go" not in json.dumps(t) and "Ok." not in json.dumps(t)   # never any dialogue in the output
