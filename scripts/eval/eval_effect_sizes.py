r"""
scripts/eval/eval_effect_sizes.py

How large is the label effect, and does it survive when contexts from one piece
are not treated as independent?

For each of the 12 paper measures and each comparison (high-low, high-mid,
mid-low), on each context's mean gap over its valid draws:

  mean gap      in the measure's own units, with a 95% interval from resampling
                whole pieces (2,000 resamples; a piece's contexts move together)
  share         share of contexts with a positive gap (ties excluded), with the
                same piece-level interval
  p (piece)     one-sided: share of piece-level resamples whose mean gap is <= 0
  Holm          Holm correction of p (piece) over the 12 measures of one comparison;
                Holm (main), over the four main measures and CS (the paper's choice)
  real gap      the same gap between real targets of the test contexts
                (ab_label_measures.csv), and the generated gap as a share of it

Draw validity and the quality filter follow eval_robustness_table.py. The filter
runs three ways: none; "draw", the paper's (a draw goes if any output is broken);
and "pair" (a comparison drops a draw only if one of its own two outputs is broken).

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_effect_sizes.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_common import MAIN, NAMES, PAIRS, cluster_boot, holm, load_generated  # noqa: E402,F401

MEASURES = NAMES
MAIN_FAMILY = tuple(MAIN.values())
COMPARISONS = (("high", "low"), ("high", "mid"), ("mid", "low"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/effect_sizes")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    t = pd.read_csv(args.pairs_measures).merge(
        pd.read_csv(args.pairs)[["pair_id", "sim_bucket"]], on="pair_id")
    t["bucket"] = t.sim_bucket.str.replace("sim:", "")
    t["ctx_stem"] = t.context_file.str.replace(".npy", "", regex=False)

    rows = []
    for qf in ("none", "draw", "pair"):
        g = load_generated(args.scores, qf)
        real = t[t.ctx_stem.isin(set(g.ctx_stem))]
        for a, b in COMPARISONS:
            block = []
            for m, name in MEASURES.items():
                w = g.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
                bad = g.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values="bad")
                ok = ~(bad[a].astype(bool) | bad[b].astype(bool))
                gap = (w[a] - w[b])[ok].groupby("ctx_stem").mean()
                means, shares = cluster_boot(gap, rng, args.boot)
                nz = gap[gap.abs() > 1e-12]
                rb = real.groupby("bucket")[m].mean() if m in real else None
                real_gap = rb[a] - rb[b] if rb is not None else np.nan
                block.append({
                    "quality_filter": qf, "comparison": f"{a}-{b}", "measure": name,
                    "contexts": len(gap), "pieces": gap.index.str[:5].nunique(),
                    "mean_gap": gap.mean(), "gap_lo": np.percentile(means, 2.5),
                    "gap_hi": np.percentile(means, 97.5),
                    "share": (nz > 0).mean(), "share_lo": np.percentile(shares, 2.5),
                    "share_hi": np.percentile(shares, 97.5),
                    "p_piece": ((means <= 0).sum() + 1) / (args.boot + 1),
                    "p_context_wilcoxon": stats.wilcoxon(nz, alternative="greater").pvalue,
                    "real_gap": real_gap,
                    "share_of_real": gap.mean() / real_gap if real_gap else np.nan})
            ps = np.array([r["p_piece"] for r in block])
            for r, h in zip(block, holm(ps)):
                r["p_piece_holm"] = h
            # the main family: the four main measures and CS, as every later test
            main = [r for r in block if r["measure"] in MAIN_FAMILY]
            for r, h in zip(main, holm(np.array([r["p_piece"] for r in main]))):
                r["p_piece_holm_main"] = h
            rows += block

    out = pd.DataFrame(rows)
    out.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        for (qf, comp), d in out.groupby(["quality_filter", "comparison"], sort=False):
            f.write(f"\nquality filter {qf}, {comp}: {d.contexts.iloc[0]} contexts, {d.pieces.iloc[0]} pieces\n")
            f.write(f"  {'measure':18s} {'gap':>8s} {'95% (piece)':>20s} {'share':>6s} "
                    f"{'95% (piece)':>14s} {'p piece':>8s} {'Holm':>7s} {'p ctx':>8s} {'real gap':>9s} {'/real':>6s}\n")
            for _, r in d.iterrows():
                f.write(f"  {r.measure:18s} {r.mean_gap:8.4f} [{r.gap_lo:8.4f}, {r.gap_hi:8.4f}] "
                        f"{r.share:6.3f} [{r.share_lo:.3f}, {r.share_hi:.3f}] {r.p_piece:8.4f} "
                        f"{r.p_piece_holm:7.4f} {r.p_context_wilcoxon:8.4f} {r.real_gap:9.4f} {r.share_of_real:6.3f}\n")
    print(open(args.out + ".txt", encoding="utf-8").read())


if __name__ == "__main__":
    main()
