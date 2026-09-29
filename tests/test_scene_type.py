import numpy as np

from survspk import scene_type as S


def test_zero_shot_picks_best_prompt_per_type():
    owner = ["camp", "tribal", "tribal"] + [t for t in S.TYPES if t not in ("camp", "tribal")]
    P = np.eye(len(owner), 8)
    X = np.array([[0.1, 0.2, 0.9, 0, 0, 0, 0, 0],      # second tribal prompt wins
                  [0.8, 0.7, 0.0, 0, 0, 0, 0, 0]])     # camp by 0.1
    lab, mar, per = S.zero_shot(X, P, owner)
    assert [S.TYPES[k] for k in lab] == ["tribal", "camp"]
    assert abs(mar[1] - 0.1) < 1e-6 and per.shape == (2, len(S.TYPES))


def test_smooth_removes_single_frame_flicker():
    lab = np.array([0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1])
    assert S.smooth(lab, k=5).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert S.smooth(np.array([], int)).tolist() == []


def test_closeups_take_the_surrounding_place():
    c, camp, tribal, g = (S.TYPES.index(x) for x in ("closeup", "camp", "tribal", "graphics"))
    lab = np.array([camp, camp, c, c, camp, tribal, tribal, c, tribal, g, c])
    out = S.locate(lab, window=2)
    assert out[2] == camp and out[3] == camp and out[7] == tribal
    assert out[9] == g and out[10] == tribal                           # graphics stay graphics and never vote
