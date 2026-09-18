"""The mention rule: a run that names a person in the third person is almost never that person speaking."""

import re

from survspk.aliases import Resolver


class FakeResolver(Resolver):
    """Resolver over a hand-written cast (no survivoR sqlite needed)."""

    def __init__(self, settings):
        self.settings = settings
        self.stoplist = set()
        self.season_aliases = {"US47": {"T": "US0715", "ROB M": "US0055"}}   # 1-letter alias must be ignored
        self._rows = [
            {"castaway_id": "US0698", "castaway": "Andy", "full_name": "Andy Rueda", "full_name_detailed": None, "last_name": "Rueda"},
            {"castaway_id": "US0709", "castaway": "Sam", "full_name": "Sam Phalen", "full_name_detailed": None, "last_name": "Phalen"},
            {"castaway_id": "US0713", "castaway": "Teeny", "full_name": "Teeny Chirichillo", "full_name_detailed": "Teeny Chirichillo", "last_name": "Chirichillo"},
            {"castaway_id": "US0715", "castaway": "TK", "full_name": "TK Foster", "full_name_detailed": None, "last_name": "Foster"},
        ]

    def cast(self, version_season):
        return self._rows


def _resolver():
    from survspk.config import load_settings
    import os
    os.environ["SURVSPK_PROFILE"] = "macos"
    load_settings.cache_clear()
    return FakeResolver(load_settings())


def test_mentions_whole_word_case_insensitive_and_possessive():
    r = _resolver()
    assert r.mentions("With Sam, I am telling him this valuable information", "US0709", "US47")
    assert r.mentions("I feel like Sam and the women will come back", "US0709", "US47")
    assert r.mentions("that's sam's idol", "US0709", "US47")
    assert r.mentions("Andy was gone. It's like the babysitter...", "US0698", "US47")
    # not a mention: substring inside another word, or a different person
    assert not r.mentions("Samantha said no", "US0709", "US47")
    assert not r.mentions("With Sam, I am telling him", "US0698", "US47")
    assert not r.mentions("", "US0709", "US47")


def test_host_is_mentioned_by_any_host_name():
    r = _resolver()
    assert r.mentions("Thank you, Jeff.", "HOST_US", "US47")
    assert r.mentions("Probst is gonna love this", "HOST_US", "US47")
    assert not r.mentions("Survivors ready? Go!", "HOST_US", "US47")


def test_aliases_and_full_names_count_but_tiny_aliases_do_not():
    r = _resolver()
    assert r.mentions("TK didn't stop talking", "US0715", "US47")
    assert r.mentions("Teeny Chirichillo, come on down", "US0713", "US47")
    # the one-letter alias 'T' must not make every 't' a mention
    assert not r.mentions("I feel like T is almost more like bro", "US0715", "US47")   # 1-char alias is ignored
    assert not r.mentions("to the top", "US0715", "US47")
    pat = r.mention_patterns("US47")["US0715"]
    assert isinstance(pat, re.Pattern) and not pat.search("to the top")
