r"""
scripts/eval/eval_mid_label.py

Why does sim:mid not sit between sim:high and sim:low? Exploratory; run after
the main analysis.

  1. Generated, mid vs low tested in the opposite direction: is sim:mid
     further from the context than sim:low? One-sided Wilcoxon (less) on the
     per-context mean gap over draws, before and after the quality filter of
     eval_robustness_table.py.
  2. Training pairs: within one context that has both a mid and a low target
     in pairs.csv, how often is the mid target closer? Binomial test.
  3. Range: mean compound similarity per label, for real targets of the
     test contexts and for the generated outputs.

Inputs are the outputs of eval_robustness_table.py (robustness_scores.csv) and
eval_ab_labels.py (ab_label_measures.csv, all 3,304 pairs).

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_mid_label.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, PAIRS, load_generated, positive_share_boot  # noqa: E402

MEASURES = list(MAIN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--pairs", default=PAIRS)
    args = ap.parse_args()

    print("1. Generated: is sim:mid further from the context than sim:low?")
    for qf in (False, True):
        g = load_generated(args.scores, "draw" if qf else "none")
        print(f"  quality filter {qf}: {g.ctx_stem.nunique()} contexts")
        for m in MEASURES:
            w = g.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
            gap = (w["mid"] - w["low"]).rename("gap").reset_index()
            gap = gap.groupby("ctx_stem").gap.mean()
            nz = gap[gap.abs() > 1e-12]
            p = stats.wilcoxon(nz, alternative="less").pvalue
            print(f"    {m:16s} share mid closer {(nz > 0).mean():.4f}  p(mid < low) {p:.4f}")

    t = pd.read_csv(args.pairs_measures).merge(
        pd.read_csv(args.pairs)[["pair_id", "sim_bucket"]], on="pair_id")
    t["bucket"] = t.sim_bucket.str.replace("sim:", "")
    t["ctx_stem"] = t.context_file.str.replace(".npy", "", regex=False)

    print("\n2. Training pairs: within a context, is the mid target closer than the low one?")
    for m in MEASURES[:-1]:
        signs, pieces = [], []
        for stem, c in t.groupby("ctx_stem"):
            mid, low = c[c.bucket == "mid"][m], c[c.bucket == "low"][m]
            if len(mid) and len(low):
                signs.append(np.sign(mid.mean() - low.mean()))
                pieces.append(stem[:5])
        s, pc = np.array(signs), np.array(pieces)
        nz = s[s != 0]
        p = stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue
        # piece-level interval of the share: contexts of one piece resampled together
        rng = np.random.default_rng(0)
        groups = np.unique(pc)
        boot = positive_share_boot(s, pc, rng, keep=s != 0)
        print(f"    {m:16s} contexts {len(s)} ({len(groups)} pieces)  share mid closer {(nz > 0).mean():.4f} "
              f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]  binomial p {p:.4f}")

    print("\n3. Mean compound similarity (CS) per label, test contexts")
    g = load_generated(args.scores, "none")
    real = t[t.ctx_stem.isin(set(g.ctx_stem))].groupby("bucket").cs.mean()
    gen = g.groupby("bucket").cs.mean()
    for name, s in (("real targets", real), ("generated", gen)):
        print(f"    {name:13s} high {s['high']:.3f}  mid {s['mid']:.3f}  low {s['low']:.3f}"
              f"  range {s.max() - s.min():.3f}")


if __name__ == "__main__":
    main()
