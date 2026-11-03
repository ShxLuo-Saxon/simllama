r"""
scripts/eval/eval_ablations_piece.py

Table 4 of the paper with the statistics of the main analysis: intervals and p
from resampling whole pieces, Holm over the five main measures of each comparison.

Reads outputs/eval/ablation_scores.csv (eval_ablations.py). One draw per model.

Comparisons (share of contexts with a positive gap; gap = first - second)
  main, 151 contexts
    M1 (mean of its three labels) above M2
    M1 high above M2
    M2 above M1 low
    M3 (mean of its three labels) above M2          prototype without a usable label
    M1 high above low
    M3 high above low, TWO-SIDED                    M3's labels are inverted in training,
                                                    so a reversal is possible
  no_prototype, the models retrained without the prototype (adapter rank 32), 127 contexts
    M1 above M2; M1 high above low
  main restricted to the 127 contexts of no_prototype
    M1 above M2; M1 high above low                  same contexts as the no_prototype rows

One-sided p is the share of resamples with mean gap <= 0; two-sided p is twice
the smaller tail.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_ablations_piece.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, cluster_boot, holm  # noqa: E402


def gaps(df: pd.DataFrame, m: str) -> dict[str, pd.Series]:
    w = df.pivot_table(index="ctx_stem", columns=["model", "label"], values=m)
    m1 = w["m1"].mean(axis=1)
    out = {"M1 above M2": m1 - w[("m2", "none")],
           "M1 high above low": w[("m1", "high")] - w[("m1", "low")]}
    if "m3" in w:
        out.update({"M1 high above M2": w[("m1", "high")] - w[("m2", "none")],
                    "M2 above M1 low": w[("m2", "none")] - w[("m1", "low")],
                    "M3 above M2": w["m3"].mean(axis=1) - w[("m2", "none")],
                    "M3 high above low (two-sided)": w[("m3", "high")] - w[("m3", "low")]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/ablation_scores.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/ablations_piece")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    s = pd.read_csv(args.scores)
    main, no_prototype = s[s.run == "main"], s[s.run == "no_prototype"]
    shared = set(no_prototype.ctx_stem)
    sets = {"main": main, "no_prototype": no_prototype, "main on no_prototype contexts": main[main.ctx_stem.isin(shared)]}

    rows = []
    for name, df in sets.items():
        block = {}
        for m, label in MAIN.items():
            for comp, g in gaps(df, m).items():
                if name != "main" and comp not in ("M1 above M2", "M1 high above low"):
                    continue
                g = g.dropna()
                means, shares = cluster_boot(g, rng, args.boot)
                nz = g[g.abs() > 1e-12]
                lo_tail = ((means <= 0).sum() + 1) / (args.boot + 1)
                hi_tail = ((means >= 0).sum() + 1) / (args.boot + 1)
                p = min(1.0, 2 * min(lo_tail, hi_tail)) if "two-sided" in comp else lo_tail
                block.setdefault(comp, []).append(dict(
                    run=name, comparison=comp, measure=label, contexts=len(g),
                    share=(nz > 0).mean(), share_lo=np.percentile(shares, 2.5),
                    share_hi=np.percentile(shares, 97.5), mean_gap=g.mean(),
                    gap_lo=np.percentile(means, 2.5), gap_hi=np.percentile(means, 97.5), p=p))
        for comp, b in block.items():
            for r, h in zip(b, holm(np.array([x["p"] for x in b]))):
                r["p_holm"] = h
            rows += b
    out = pd.DataFrame(rows)
    out.to_csv(args.out + ".csv", index=False)
    lines = []
    for (run, comp), d in out.groupby(["run", "comparison"], sort=False):
        lines.append(f"\n{run} | {comp} | {d.contexts.iloc[0]} contexts")
        for _, r in d.iterrows():
            lines.append(f"  {r.measure:10s} share {r.share:.3f} [{r.share_lo:.3f}, {r.share_hi:.3f}]  "
                         f"gap {r.mean_gap:+.4f} [{r.gap_lo:+.4f}, {r.gap_hi:+.4f}]  p {r.p:.4f}  Holm {r.p_holm:.4f}")
    text = "\n".join(lines) + "\n"
    Path(args.out + ".txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
