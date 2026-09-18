"""Text prior against a fake oMLX server: prompt content, response parsing (incl. leaked <think>), storage, eval,
and the assign combination behind text_prior.use_in_assign."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from survspk.stage_assign import assign_episode
from survspk.stage_bank import build_bank
from survspk.text_prior import LLMCfg, build_prompt, cast_sheet, chat_json, evaluate, run_text_prior
from tests.test_bank_assign import season  # noqa: F401  (fixture)
from tests.test_mentions import FakeResolver

BIOS = {
    "S_A": {"name": "Andy", "full_name": "Andy Rueda", "age": 31, "city": "Buffalo", "state": "New York", "gender": "Male", "occupation": "AI Research Assistant", "tribe": "Gata"},
    "S_B": {"name": "Gabe", "full_name": "Gabe Ortis", "age": 26, "city": "Baltimore", "state": "Maryland", "gender": "Male", "occupation": "Radio Show Host", "tribe": "Tuku"},
    "S_C": {"name": "Sam", "full_name": "Sam Phalen", "age": 24, "city": "Nashville", "state": "Tennessee", "gender": "Male", "occupation": "Sports Reporter", "tribe": "Gata"},
    "S_D": {"name": "Sue", "full_name": "Sue Smey", "age": 58, "city": "Putnam Valley", "state": "New York", "gender": "Female", "occupation": "Flight School Owner", "tribe": "Tuku"},
}


class R(FakeResolver):
    def __init__(self, s):
        super().__init__(s)
        self._rows = [{"castaway_id": k, "castaway": b["name"], "full_name": b["full_name"], "full_name_detailed": None,
                       "last_name": b["full_name"].split()[-1]} for k, b in BIOS.items()]
        self.season_aliases = {}

    def present(self, vs, ep):
        return frozenset(BIOS)

    def bios(self, vs, ep):
        return BIOS


class FakeLLM(BaseHTTPRequestHandler):
    """Answers by keyword: 'business school'/'59' -> Sue; a passage naming Sam -> Andy; otherwise UNKNOWN."""
    seen: list[dict] = []

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        self._send({"data": [{"id": "Qwen3.6-35B-A3B-MLX-8bit"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.seen.append(body)
        user = body["messages"][1]["content"]
        passage = user.split(">>> PASSAGE TO IDENTIFY")[1].split("Context after:")[0]
        if "59" in passage or "flight school" in passage.lower():
            ans = {"speaker": "Sue", "confidence": 0.95, "excluded": ["Andy", "Gabe", "Sam"], "reason": "says she is 59"}
        elif "Sam" in passage:
            ans = {"speaker": "Andy", "confidence": 0.85, "excluded": ["Sam"], "reason": "talks about Sam; Gata context"}
        else:
            ans = {"speaker": "UNKNOWN", "confidence": 0.2, "excluded": [], "reason": "nothing decisive"}
        content = "<think>\nlet me see\n</think>\n" + json.dumps(ans)
        self._send({"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def llm():
    srv = HTTPServer(("127.0.0.1", 0), FakeLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    FakeLLM.seen.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_cast_sheet_and_prompt(season):
    s, con, sea = season
    sheet, name2id, id2name = cast_sheet(R(s), "US99", 2, frozenset({"S_A", "S_D", "HOST_US"}))
    assert "Sue (Sue Smey)" in sheet and "Flight School Owner" in sheet and "HOST (Jeff Probst)" in sheet
    assert "Gabe" not in sheet                              # not a candidate this episode
    assert name2id["sue"] == "S_D" and name2id["jeff"] == "HOST_US" and id2name["S_A"] == "Andy"
    target = {"mmss": "12:00", "dur": 8.0, "domain": "confessional", "text": "When I win, my final words will be I'm 59",
              "audio_top": [("S_A", 0.41), ("S_D", 0.39)]}
    before = [{"mmss": "11:50", "text": "Good luck out there.", "known": "HOST_US"}]
    p = build_prompt(sheet, id2name, target, before, [], "US99", 2)
    assert "[HOST] Good luck" in p and ">>> PASSAGE TO IDENTIFY (12:00, 8 s, confessional)" in p
    assert "Andy 0.41, Sue 0.39" in p and "(end of episode)" in p


def test_chat_json_strips_think_block(llm):
    cfg = LLMCfg(base_url=llm, model="x")
    ans, dt = chat_json(cfg, "sys", ">>> PASSAGE TO IDENTIFY I'm 59 Context after: none")
    assert ans["speaker"] == "Sue" and ans["confidence"] == 0.95 and dt >= 0
    body = FakeLLM.seen[-1]
    assert body["response_format"]["type"] == "json_schema" and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_text_prior_end_to_end_and_assign_combination(season, llm):
    s, con, sea = season
    s.raw["text_prior"] = {"base_url": llm, "model": "Qwen3.6-35B-A3B-MLX-8bit", "concurrency": 2,
                           "use_in_assign": False, "min_confidence": 0.8}
    build_bank(s, con, "US99", [1], variant="vocals")
    e2 = [
        {"spk": "S_A", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "blah"},                          # auto, explicit
        {"spk": "S_D", "label": None, "n": 2, "dur_each": 5.0, "text": "When I win, I'm 59 and I own a flight school"},  # unbanked -> queue -> Sue
        {"spk": "S_A", "label": "S_A", "n": 2, "dur_each": 4.0, "text": "With Sam, I am telling him everything"},   # explicit Andy, mentions Sam
        {"spk": "S_C", "label": None, "n": 2, "dur_each": 4.0, "text": "nothing decisive here"},          # auto Sam
    ]
    ids = sea.episode(2, e2)
    r = R(s)
    df0, st0 = assign_episode(s, con, "US99", 2, variant="vocals", resolver=r, write=True)
    assert df0[df0.run_id == 1].iloc[0].decision in ("low_margin", "no_candidate")

    tp = run_text_prior(s, con, "US99", 2, r, only_queued=True)
    # queued run 1 and explicit runs 0, 2 are scored; auto-only run 3 is not
    assert set(tp.run_id) == {0, 1, 2}
    by = tp.set_index("run_id")
    assert by.loc[1, "speaker_id"] == "S_D" and by.loc[1, "confidence"] == 0.95 and by.loc[1, "queued"]
    assert by.loc[2, "speaker_id"] == "S_A" and by.loc[2, "explicit"] == "S_A"
    assert pd.isna(by.loc[0, "speaker_id"]) and by.loc[0, "speaker_name"] == "UNKNOWN"
    ev = evaluate(tp)
    assert ev["n_explicit"] == 2 and ev["n_answered"] == 1 and ev["precision_answered"] == 1.0
    # the target's own explicit label never appears in its prompt; neighbours' labels do
    p2 = next(b for b in FakeLLM.seen if "With Sam" in b["messages"][1]["content"].split(">>> PASSAGE")[1].split("Context after")[0])
    passage_block = p2["messages"][1]["content"].split(">>> PASSAGE")[1].split("Context after")[0]
    assert "[Andy]" not in passage_block
    stored = pd.read_sql_query("SELECT * FROM text_prior WHERE version_season='US99'", con)
    assert len(stored) == 3 and stored.model.iloc[0] == "Qwen3.6-35B-A3B-MLX-8bit"
    # idempotent: a second pass asks nothing new
    assert run_text_prior(s, con, "US99", 2, r, only_queued=True).empty

    # now let assign use it: the unbanked Sue run becomes auto_text with source 'text'
    s.raw["text_prior"]["use_in_assign"] = True
    df1, st1 = assign_episode(s, con, "US99", 2, variant="vocals", resolver=r, write=True)
    r1 = df1[df1.run_id == 1].iloc[0]
    assert r1.decision == "auto_text" and r1.pred == "S_D" and st1["n_auto_text"] == 1
    lab = pd.read_sql_query("SELECT * FROM labels", con).set_index("utt_id")
    assert lab.loc[ids[1][0], "source"] == "text" and lab.loc[ids[1][0], "speaker_id"] == "S_D" and lab.loc[ids[1][0], "confidence"] == 0.95
    assert lab.loc[ids[2][0], "source"] == "sdh"                                       # explicit labels untouched
    q = pd.read_sql_query("SELECT * FROM review_queue WHERE resolved=0", con)
    assert not q.utt_id.isin(ids[1]).any()
