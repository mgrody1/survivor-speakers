import numpy as np
import pandas as pd

from survspk.stage_embed import loo_scores


def test_loo_scores_separable_clusters():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(4):
        center = rng.normal(size=16)
        for _ in range(12):
            v = center + rng.normal(scale=0.3, size=16)
            rows.append({"speaker_id": f"S{s}", "vector": (v / np.linalg.norm(v)).astype(np.float32)})
    sc = loo_scores(pd.DataFrame(rows))
    assert sc["n_utts"] == 48 and sc["n_speakers"] == 4
    assert sc["centroid_acc"] > 0.95 and sc["exemplar_acc"] > 0.9
    assert sc["median_margin"] > 0


def test_loo_scores_random_is_chance():
    rng = np.random.default_rng(1)
    rows = [{"speaker_id": f"S{i % 5}", "vector": rng.normal(size=32).astype(np.float32)} for i in range(200)]
    sc = loo_scores(pd.DataFrame(rows))
    assert 0.05 < sc["centroid_acc"] < 0.4
