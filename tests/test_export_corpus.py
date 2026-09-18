"""Corpus export on the synthetic season: trusted-label selection, RTTM/UEM formatting, splits, manifest + trials,
database.yml; plus an end-to-end smoke run of the ECAPA fine-tune loop (random-init model, CPU)."""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from survspk.export_corpus import _merge, annotated_regions, export_corpus, export_episode, make_trials, trusted_labels
from survspk.stage_align import HOP_S
from tests.test_bank_assign import season  # noqa: F401

SR = 16000


def _tone_flac(path, seconds, speech_spans):
    """Silence everywhere except a 220 Hz tone in `speech_spans` (so the VAD hears speech only there)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(len(x)) / SR
    for a, b in speech_spans:
        i, j = int(a * SR), int(b * SR)
        x[i:j] = 0.3 * np.sign(np.sin(2 * np.pi * 220 * t[i:j]))     # square wave: VAD flags it as speech reliably
    sf.write(str(path), x, SR, subtype="PCM_16")


@pytest.fixture
def corpus_season(season):
    s, con, sea = season
    # E2: labelled by hand (human labels), incl. the pseudo ids
    e2 = [{"spk": spk, "label": None, "n": 3, "dur_each": 4.0} for spk in ("S_A", "S_B", "S_C") for _ in range(3)]
    e2 += [{"spk": "S_A", "label": None, "n": 2, "dur_each": 3.0, "domain": "field"}]
    ids2 = sea.episode(2, e2)
    flat = [u for run in ids2 for u in run]
    utt = pd.read_sql_query("SELECT utt_id, start_s, end_s FROM utterances WHERE episode=2", con).set_index("utt_id")
    human = []
    for run, spk in zip(ids2[:9], ["S_A"] * 3 + ["S_B"] * 3 + ["S_C"] * 3):
        human += [(u, spk) for u in run]
    human += [(u, "S_A") for u in ids2[9]]
    # a few pseudo labels on E1 utterances that have no sdh label: OTHER, UNKNOWN, NOSPEECH
    e1_unlab = pd.read_sql_query("SELECT utt_id FROM utterances WHERE episode=1 ORDER BY idx", con).utt_id.tolist()
    human_e1 = [(e1_unlab[0], "OTHER"), (e1_unlab[1], "UNKNOWN"), (e1_unlab[2], "NOSPEECH")]
    con.executemany("INSERT OR REPLACE INTO labels (utt_id, speaker_id, source) VALUES (?,?,'human')", human + human_e1)
    con.commit()
    # audio: E1 and E2 raw + vocals, speech (tone) exactly where the utterances are
    for ep in (1, 2):
        spans = pd.read_sql_query("SELECT start_s, end_s FROM utterances WHERE episode=?", con, params=(ep,))
        total = float(spans.end_s.max()) + 10
        for variant in ("raw", "vocals"):
            _tone_flac(s.audio_path(variant, "US99", ep), total, list(zip(spans.start_s, spans.end_s)))
    yield s, con, sea, utt, flat


def test_merge_and_annotated_regions():
    assert _merge([(5, 7), (1, 3), (2, 4), (7, 8)]) == [(1, 4), (5, 8)]
    spans = [(10.0, 12.0), (13.0, 14.0), (60.0, 61.0)]
    # no VAD: spans + margin only
    regs = annotated_regions(spans, 100.0, None, HOP_S)
    assert regs == [(9.8, 12.2), (12.8, 14.2), (59.8, 61.2)]
    # VAD silent everywhere: the 0.6 s gap is bridged, the 45 s one is too (<= 30? no: 14.2 -> 59.8 is 45.6 s)
    mask = np.zeros(int(100 / HOP_S), dtype=np.float32)
    regs = annotated_regions(spans, 100.0, mask, HOP_S)
    assert regs == [(9.8, 14.2), (59.8, 61.2)]
    # speech in the gap -> not bridged
    mask[int(12.4 / HOP_S):int(12.7 / HOP_S)] = 1.0
    regs = annotated_regions(spans, 100.0, mask, HOP_S)
    assert regs == [(9.8, 12.2), (12.8, 14.2), (59.8, 61.2)]
    # a gap wider than max_gap_s is never bridged even when silent
    regs = annotated_regions([(0.0, 1.0), (40.0, 41.0)], 100.0, np.zeros(5000, dtype=np.float32), HOP_S)
    assert regs == [(0.0, 1.2), (39.8, 41.2)]


def test_trusted_labels_sources(corpus_season):
    s, con, sea, utt, flat = corpus_season
    e1 = trusted_labels(s, con, "US99", 1, "vocals")
    # human labels always; sdh + sdh_run after the consistency filter (MISLABELLED run dropped); no text prior
    assert set(e1.source) == {"human", "sdh", "sdh_run"}
    assert (e1[e1.source == "human"].speaker_id.isin(["OTHER", "UNKNOWN", "NOSPEECH"])).all()
    txt = pd.read_sql_query("SELECT utt_id, text FROM utterances WHERE episode=1", con).set_index("utt_id").text
    assert not (txt.loc[e1.utt_id] == "MISLABELLED").any()
    assert e1.start_s.is_monotonic_increasing
    e2 = trusted_labels(s, con, "US99", 2, "vocals")
    assert set(e2.source) == {"human"} and len(e2) == len(flat)


def test_export_episode_rttm_uem(corpus_season):
    s, con, sea, utt, flat = corpus_season
    ex = export_episode(s, con, "US99", 1, variant="raw")
    spk_lines = {spk for _, _, spk in ex.rttm}
    assert spk_lines >= {"S_A", "S_B", "S_C", "OTHER"} and "UNKNOWN" not in spk_lines and "NOSPEECH" not in spk_lines
    # every RTTM span lies inside some UEM region; the UNKNOWN utterance lies in none
    def covered(a, b):
        return any(ua <= a + 1e-6 and b - 1e-6 <= ub for ua, ub in ex.uem)
    assert all(covered(st, st + d) for st, d, _ in ex.rttm)
    unk = utt_row(con, "UNKNOWN")
    assert not covered(unk.start_s, unk.end_s)
    nos = utt_row(con, "NOSPEECH")
    assert covered(nos.start_s, nos.end_s)               # annotated as non-speech
    # gaps between consecutive utterances (silence in the synthetic audio) are bridged by the VAD
    assert len(ex.uem) < len(ex.rttm)
    # manifest rows: real speakers only, >= 1 s
    ids = {r["id"] for r in ex.spk_rows}
    assert ids and not any(sid in ("OTHER", "UNKNOWN", "NOSPEECH") for sid in {r["speaker_id"] for r in ex.spk_rows})
    assert ex.stats["n_speakers"] == 3 and ex.stats["by_source"]["human"] == 3
    assert ex.stats["annotated_s"] >= ex.stats["speech_s"] > 0
    # without VAD the annotated time shrinks to spans + margins
    ex2 = export_episode(s, con, "US99", 1, variant="raw", use_vad=False)
    assert ex2.stats["annotated_s"] < ex.stats["annotated_s"] and ex2.rttm == ex.rttm


def utt_row(con, speaker_id):
    return pd.read_sql_query("SELECT u.start_s, u.end_s FROM labels l JOIN utterances u USING (utt_id) WHERE l.speaker_id=?",
                             con, params=(speaker_id,)).iloc[0]


def test_export_corpus_layout(corpus_season, tmp_path):
    s, con, sea, utt, flat = corpus_season
    out = tmp_path / "corpus"
    stats = export_corpus(s, con, out, variant="raw")
    assert stats["n_episodes"] == 2 and stats["splits"] == {"train": 1, "dev": 1, "test": 0}
    assert stats["episodes"]["US99_E01"]["split"] == "train" and stats["episodes"]["US99_E02"]["split"] == "dev"
    assert (out / "lists" / "train.lst").read_text().split() == ["US99_E01"]
    assert (out / "lists" / "dev.lst").read_text().split() == ["US99_E02"]
    assert (out / "lists" / "test.lst").read_text() == ""
    # RTTM / UEM formats
    line = (out / "rttm" / "US99_E01.rttm").read_text().splitlines()[0].split()
    assert line[0] == "SPEAKER" and line[1] == "US99_E01" and line[2] == "1" and line[5:7] == ["<NA>", "<NA>"] and line[8:] == ["<NA>", "<NA>"]
    float(line[3]); float(line[4])
    uem = (out / "uem" / "US99_E02.uem").read_text().splitlines()
    assert all(len(x.split()) == 4 and x.split()[:2] == ["US99_E02", "1"] for x in uem)
    a, b = map(float, uem[0].split()[2:])
    assert b > a
    # database.yml points at the flat symlink dir; the symlinks resolve to the raw FLACs
    db = yaml.safe_load((out / "database.yml").read_text())
    assert db["Databases"]["Survivor"].endswith("/audio/{uri}.flac")
    prot = db["Protocols"]["Survivor"]["SpeakerDiarization"]["All"]
    assert set(prot) == {"train", "development", "test"} and prot["train"]["annotated"] == "uem/{uri}.uem"
    link = out / "audio" / "US99_E02.flac"
    assert link.exists() and sf.info(str(link)).samplerate == SR
    # speaker manifest + trials
    tr = pd.read_csv(out / "spk" / "train.csv")
    dev = pd.read_csv(out / "spk" / "dev.csv")
    assert set(tr.columns) >= {"id", "wav", "start_s", "end_s", "duration", "speaker_id", "season", "episode", "source", "uri"}
    assert set(tr.speaker_id) == {"S_A", "S_B", "S_C"} and set(dev.speaker_id) == {"S_A", "S_B", "S_C"}
    assert (tr.duration >= 1.0).all() and set(tr.uri) == {"US99_E01"} and set(dev.uri) == {"US99_E02"}
    trials = pd.read_csv(out / "spk" / "trials_dev.txt", sep=" ", names=["label", "enrol", "test"])
    assert set(trials.label) == {0, 1} and set(trials.enrol) <= set(dev.id) and set(trials.test) <= set(dev.id)
    spk_of = dict(zip(dev.id, dev.speaker_id))
    assert all(spk_of[a] == spk_of[b] for a, b, l in zip(trials.enrol, trials.test, trials.label) if l == 1)
    assert all(spk_of[a] != spk_of[b] for a, b, l in zip(trials.enrol, trials.test, trials.label) if l == 0)
    assert (trials[trials.label == 1].enrol != trials[trials.label == 1].test).all()
    js = json.loads((out / "stats.json").read_text())
    assert js["n_trials_dev"] == len(trials) and js["n_speakers"] == 3
    # test_seasons puts the whole season in test; then there is no dev and no trials
    out2 = tmp_path / "corpus2"
    st2 = export_corpus(s, con, out2, variant="raw", test_seasons=["US99"], use_vad=False)
    assert st2["splits"] == {"train": 0, "dev": 0, "test": 2} and st2["n_trials_dev"] == 0


def test_make_trials_same_season_negatives():
    rows = []
    for vs, spks in (("A", ["x", "y"]), ("B", ["p", "q"])):
        for spk in spks:
            rows += [{"id": f"{vs}_{spk}_{i}", "speaker_id": spk, "season": vs} for i in range(3)]
    dev = pd.DataFrame(rows)
    t = make_trials(dev, n_target=50, n_nontarget=50)
    assert len(t) == 100 and (t.label.value_counts() == 50).all()
    season_of = dict(zip(dev.id, dev.season))
    neg = t[t.label == 0]
    assert all(season_of[a] == season_of[b] for a, b in zip(neg.enrol, neg.test))
    # one speaker only -> no trials
    assert make_trials(dev[dev.speaker_id == "x"]).empty


def test_finetune_smoke(corpus_season, tmp_path):
    """Random-init ECAPA on the synthetic corpus: the loop runs, evaluates a baseline, saves a loadable checkpoint."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("speechbrain")
    from survspk.finetune_ecapa import TrainCfg, eer, train

    s, con, sea, utt, flat = corpus_season
    out = tmp_path / "corpus"
    export_corpus(s, con, out, variant="raw", use_vad=False)
    cfg = TrainCfg(corpus=out, out=tmp_path / "ckpt", epochs=2, batch_size=8, crop_s=1.0, freeze_epochs=1,
                   min_utts_per_speaker=2, device="cpu", smoke=True)
    res = train(cfg)
    h = res["history"]
    assert h[0]["epoch"] == 0 and 0.0 <= h[0]["eer"] <= 1.0 and h[0]["n_trials"] > 0     # baseline first
    assert [r["epoch"] for r in h[1:]] == [1, 2] and h[1]["encoder_trained"] is False and h[2]["encoder_trained"] is True
    assert all(np.isfinite(r["loss"]) for r in h[1:])
    ck = tmp_path / "ckpt"
    assert (ck / "embedding_model.ckpt").exists() and json.loads((ck / "finetune.json").read_text())["n_speakers"] == 3
    sd = torch.load(ck / "embedding_model.ckpt", map_location="cpu")
    assert any(k.startswith("blocks") for k in sd)
    # the EER helper on a toy separable case
    e, thr = eer(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert e == 0.0 and 0.2 <= thr <= 0.8
