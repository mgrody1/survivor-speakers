from survspk import ids
from survspk.inventory import choose_subtitle


def test_parse_sxxeyy_variants():
    assert ids.parse_sxxeyy("Survivor - S47E02 - Epic Boss Girl Move WEBDL-1080p.mkv") == (47, 2, None)
    assert ids.parse_sxxeyy("Survivor.S05E08.en.srt") == (5, 8, None)
    assert ids.parse_sxxeyy("Survivor.S07E14E15.Pearl.Islands.Reunion.srt") == (7, 14, 15)
    assert ids.parse_sxxeyy("Survivor (2000) S35E11 (1080p AMZN).en.srt") == (35, 11, None)
    assert ids.parse_sxxeyy("no episode here.mkv") is None
    # doubles written with a hyphen
    assert ids.parse_sxxeyy("Survivor - S30E04-05 - Winner Winner WEBDL-1080p.mkv") == (30, 4, 5)
    assert ids.parse_sxxeyy("Survivor - S42E06-E07 - x.mkv") == (42, 6, 7)
    # resolution tokens must not be mistaken for a second episode
    assert ids.parse_sxxeyy("Survivor S49E04 Go Kick Rocks Bro 1080p PMTP.srt") == (49, 4, None)


def test_ids():
    assert ids.version_season("US", 7) == "US07"
    assert ids.cue_id("US47", 2, 5) == "US47_E02_C0005"
    assert ids.utt_id("US47", 2, 5) == "US47_E02_U0005"


def _c(path, source="sidecar", is_sdh=False, n_cues=1000, delta=1.0, err=None):
    return {"path": path, "source": source, "is_sdh": is_sdh, "n_cues": n_cues, "duration_delta_s": delta,
            "parse_error": err}


def test_chooser_prefers_sdh_then_timing_then_cues():
    a = _c("plain.srt", is_sdh=False, n_cues=1400, delta=-2)
    b = _c("sdh.srt", is_sdh=True, n_cues=1300, delta=-3)
    assert choose_subtitle([a, b], True, 2600)["path"] == "sdh.srt"
    # SDH preference only applies within the timing-ok tier: a truncated SDH file loses to a good plain one
    trunc = _c("sdh_trunc.srt", is_sdh=True, n_cues=600, delta=-1300)
    assert choose_subtitle([a, trunc], True, 2600)["path"] == "plain.srt"
    # a subtitle that runs past the end of the file is also suspect
    over = _c("sdh_over.srt", is_sdh=True, n_cues=2500, delta=+2500)
    assert choose_subtitle([a, over], True, 2600)["path"] == "plain.srt"
    # normal "ends before the credits" deltas are fine
    credits = _c("sdh_credits.srt", is_sdh=True, n_cues=1300, delta=-95)
    assert choose_subtitle([a, credits], True, 2600)["path"] == "sdh_credits.srt"
    # a *broken* one (no cues) is excluded
    broken = _c("sdh_broken.srt", is_sdh=True, n_cues=0, delta=None, err="x")
    assert choose_subtitle([a, broken], True, 2600)["path"] == "plain.srt"
    # two plain files: small deltas are equivalent, more cues wins
    c = _c("plain2.srt", n_cues=900, delta=0.5)
    assert choose_subtitle([a, c], True, 2600)["path"] == "plain.srt"
    # big delta loses
    d = _c("plain_off.srt", n_cues=2000, delta=400)
    assert choose_subtitle([a, d], True, 2600)["path"] == "plain.srt"


def test_chooser_falls_back_to_embedded():
    e1 = _c("v.mkv#s:2", source="embedded", is_sdh=False, n_cues=None, delta=None)
    e2 = _c("v.mkv#s:3", source="embedded", is_sdh=True, n_cues=None, delta=None)
    assert choose_subtitle([e1, e2], True, None)["path"] == "v.mkv#s:3"
    assert choose_subtitle([], True, None) is None
    # an *extracted* embedded stream with good timing beats a truncated sidecar
    trunc = _c("sdh_trunc.srt", is_sdh=True, n_cues=600, delta=-1300)
    e3 = _c("v.mkv#s:3", source="embedded", is_sdh=True, n_cues=1400, delta=-40)
    assert choose_subtitle([trunc, e3], True, 2600)["path"] == "v.mkv#s:3"


def test_a_track_that_names_its_speakers_wins_within_a_timing_tier():
    base = {"source": "sidecar", "parse_error": None, "n_cues": 2100, "duration_delta_s": 3.0}
    downloaded = {**base, "path": "dl.en.hi.srt", "is_sdh": True, "n_names": 1}
    embedded = {**base, "path": "v.mkv#s:3", "source": "embedded", "is_sdh": True, "n_cues": 2000, "n_names": 241}
    assert choose_subtitle([downloaded, embedded], True, 3900)["path"] == "v.mkv#s:3"
    bad_timing = {**embedded, "duration_delta_s": 400.0}
    assert choose_subtitle([downloaded, bad_timing], True, 3900)["path"] == "dl.en.hi.srt"   # timing still comes first
