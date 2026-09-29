"""Face evidence (survspk/faces.py): the per-line scoring and gallery logic, on synthetic embeddings."""
import numpy as np

from survspk import faces as fx


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


A, B, H = unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])


def pad(v):
    out = np.zeros(512, np.float32)
    out[:4] = v
    return out


def ef(rows):
    """rows: (t, area, vector)"""
    t = np.array([r[0] for r in rows], np.float32)
    return fx.EpisodeFaces(t, np.array([r[1] for r in rows], np.float32), np.vstack([pad(r[2]) for r in rows]), np.unique(t))


def test_largest_face_per_frame():
    e = ef([(0.0, 0.05, A), (0.0, 0.01, B), (0.5, 0.02, B), (1.0, 0.03, A)])
    assert sorted(e.largest_in(0.0, 0.6)) == [0, 2]
    assert e.largest_in(2.0, 3.0) == []


def test_gallery_drops_outliers():
    g = fx.build_gallery({"A": [pad(A), pad(unit([0.9, 0.1, 0, 0])), pad(A), pad(B)], "Z": []})
    assert len(g["A"]) == 3 and "Z" not in g


def test_score_line_votes_and_shares():
    gal = fx.build_gallery({"A": [pad(A)] * 3, "B": [pad(B)] * 3, "H": [pad(H)] * 3})
    e = ef([(10.0, 0.06, A), (10.5, 0.06, unit([0.9, 0.2, 0, 0])), (11.0, 0.05, B), (11.0, 0.01, A)])
    s = fx.score_line(e, 10.0, 11.2, gal, ["A", "B", "H"])
    assert s["face_pred"] == "A" and s["face_frames"] == 3 and abs(s["face_share"] - 2 / 3) < 1e-6
    assert s["face_sim"] > 0.9 and s["face_margin"] > 0
    assert fx.score_line(e, 10.0, 11.2, gal, ["B"])["face_pred"] == "B"      # only candidates in the episode count
    assert fx.score_line(e, 20.0, 21.0, gal, ["A"]) is None


def test_line_samples_checked_against_cards():
    s = [("card", 1, "A", pad(A)), ("card", 2, "A", pad(A)),
         ("line", 3, "A", pad(A)), ("line", 3, "A", pad(B)),            # B-roll of someone else under A's voice
         ("line", 3, "H", pad(H)), ("line", 4, "H", pad(H)), ("line", 4, "H", pad(H)), ("line", 5, "H", pad(B))]
    out = fx.verify_line_samples(s)
    lines = [(c, tuple(v[:4].round(2))) for src, e, c, v in out if src == "line"]
    assert ("A", tuple(pad(A)[:4].round(2))) in lines and ("A", tuple(pad(B)[:4].round(2))) not in lines
    assert sum(c == "H" for c, _ in lines) == 3                         # no cards: keep the host's most common face
    assert sum(src == "card" for src, *_ in out) == 2
