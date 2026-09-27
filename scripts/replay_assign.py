"""Write replayed runs to CSV for experiments (see survspk/replay.py). Opens the database read-only.

    uv run python scripts/replay_assign.py --seasons US45,US46 [--variants vocals,raw,center,vocals_center]
        [--self-train 0.97] --out ../survivor_audio/reports/tmp/replay/base
"""

import argparse
import sqlite3
import time
from pathlib import Path

from survspk import calibrate as cal
from survspk.aliases import Resolver
from survspk.config import load_settings
from survspk.replay import replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True)
    ap.add_argument("--variants", default="vocals,raw,center,vocals_center")
    ap.add_argument("--self-train", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = load_settings()
    con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    res = Resolver(s)
    calib = cal.load_model(s)
    for vs in a.seasons.split(","):
        t0 = time.time()
        df = replay(s, con, res, vs, a.variants.split(","), a.self_train, calib)
        out = Path(f"{a.out}_{vs}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"{vs}: {len(df)} runs -> {out} [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
