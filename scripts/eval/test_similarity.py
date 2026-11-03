r"""
scripts/eval/test_similarity.py

Regression test for similarity.py: the single metric module must reproduce every
stored, published number. Run it after any change to a metric or to the MIDI
reader. It exits non-zero on the first mismatch.

Checks
------
1. Compound similarity reproduces the `compound_ks` column of pairs.csv, which built
   the training buckets (3,304 pairs; the column is stored to 6 decimals).
2. The count-chroma variant of compound similarity reproduces outputs/eval/m1_triplet_ctx_sim.csv, the scores
   behind the headline H1_ctx result (M1, M2, M3; every row).

Usage:
    .venv\Scripts\python.exe scripts\eval\test_similarity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import (  # noqa: E402
    compound_similarity, compound_similarity_v1_1, midi_to_compound)

from paper_common import SECTIONS as SEC  # noqa: E402


def check(name: str, got: np.ndarray, want: np.ndarray, tol: float) -> None:
    diff = np.nanmax(np.abs(got - want))
    bad = int((np.abs(got - want) > tol).sum())
    status = "PASS" if bad == 0 else "FAIL"
    print(f"[{status}] {name}: {len(got)} values, max abs diff {diff:.1e} "
          f"(tolerance {tol:.0e}), {bad} over")
    if bad:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections_dir", default=str(SEC))
    ap.add_argument("--gen_dir", default="outputs/generated/m1_triplet")
    ap.add_argument("--ctx_sim_csv", default="outputs/eval/m1_triplet_ctx_sim.csv")
    args = ap.parse_args()
    sec = Path(args.sections_dir)

    # 1. training buckets
    pairs = pd.read_csv(sec / "pairs.csv")
    got = np.array([compound_similarity(np.load(str(sec / r.context_file)).astype(float),
                                np.load(str(sec / r.target_file)).astype(float))
                    for r in pairs.itertuples()])
    check("CS vs pairs.csv compound_ks", got, pairs.compound_ks.to_numpy(), 1e-6)

    # 2. headline eval -- read exactly as eval_triplet.py does
    ref = pd.read_csv(args.ctx_sim_csv)
    gen, got = Path(args.gen_dir), []
    for r in ref.itertuples():
        name = (f"{r.ctx_stem}_m2_generated.mid" if r.model == "m2"
                else f"{r.ctx_stem}_{r.model}_{r.bucket}_generated.mid")
        g = midi_to_compound(gen / name).copy()
        g[:, 0] -= g[0, 0]
        got.append(compound_similarity_v1_1(np.load(str(sec / f"{r.ctx_stem}.npy")), g))
    check("count-chroma variant vs m1_triplet_ctx_sim.csv", np.array(got),
          ref.ctx_sim.to_numpy(), 1e-12)
    print("All published metric values reproduced.")


if __name__ == "__main__":
    main()
