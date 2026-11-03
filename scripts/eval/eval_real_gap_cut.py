r"""
scripts/eval/eval_real_gap_cut.py

Table 2's real gap with the real targets cut at the context's length, as the
generated outputs are. The published real gap scores real targets at full length, and Jaccard and
histogram measures depend on length.

For every test pair of the 157 contexts, the target is cut at the context's span
(as in eval_copy_matched.py) and scored with the same measures as the outputs
(symbolic_measures.all_measures, compound_similarity). The script prints,
for the four main measures and CS:

  real gap, full length   pooled: mean of high targets minus mean of low targets
                          (eval_effect_sizes.py; it must reproduce Table 2)
  real gap, cut           the same with the cut targets
  generated gap           SimLlama high minus low, valid draws (Table 2)
  ratio                   generated over the cut real gap, with a 95% interval
                          from 2,000 resamples of whole pieces (both gaps
                          resampled together)
  paired, cut             within the contexts that have both a high and a low
                          target: real gap and ratio on those contexts only

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_real_gap_cut.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ab_labels import load  # noqa: E402
from paper_common import MAIN, PAIRS, load_generated  # noqa: E402
from similarity import compound_similarity  # noqa: E402
from symbolic_measures import all_measures  # noqa: E402

SEC = Path(PAIRS).parent


def score(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in pairs.itertuples():
        a, b = load(SEC / p.context_file), load(SEC / p.target_file)
        b = b[b[:, 0] - b[0, 0] <= a[:, 0].max() - a[:, 0].min()]
        m = all_measures(a, b)
        m["cs"] = compound_similarity(a.astype(float), b.astype(float))
        rows.append(dict(ctx_stem=p.context_file.removesuffix(".npy"), bucket=p.sim_bucket[4:], **m))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/real_gap_cut.txt")
    args = ap.parse_args()

    g = load_generated(args.scores)
    stems = set(g.ctx_stem)
    pairs = pd.read_csv(PAIRS)
    pairs = pairs[(pairs.split == "test") & pairs.context_file.str.removesuffix(".npy").isin(stems)
                  & pairs.sim_bucket.isin(["sim:high", "sim:low"])]
    full = pd.read_csv(args.pairs_measures).merge(pairs[["pair_id", "sim_bucket"]], on="pair_id")
    full = full.assign(bucket=full.sim_bucket.str[4:], ctx_stem=full.context_file.str.removesuffix(".npy"))
    cut = score(pairs)
    both = set(cut[cut.bucket == "high"].ctx_stem) & set(cut[cut.bucket == "low"].ctx_stem)

    pieces = sorted({s[:5] for s in stems})
    rng = np.random.default_rng(0)
    draws = [rng.choice(pieces, len(pieces)) for _ in range(args.boot)]
    lines = [f"{len(pairs)} high and low test pairs of {len(stems)} contexts; "
             f"{len(both)} contexts have both", "",
             f"{'measure':10s} {'real full':>9s} {'real cut':>9s} {'generated':>10s} {'ratio':>6s} "
             f"{'95% interval':>16s}   {'paired real':>11s} {'ratio':>6s}"]
    for m, name in MAIN.items():
        gen = g.pivot_table(index="ctx_stem", columns="bucket", values=m)
        gen = gen["high"] - gen["low"]
        rf = full.groupby("bucket")[m].mean()
        rc = cut.groupby("bucket")[m].mean()
        ratio = gen.mean() / (rc["high"] - rc["low"])
        boots = []
        for s in draws:
            gs = pd.concat([gen[gen.index.str[:5] == k] for k in s])
            cs = pd.concat([cut[cut.ctx_stem.str[:5] == k] for k in s])
            r = cs.groupby("bucket")[m].mean()
            if {"high", "low"} <= set(r.index):
                boots.append(gs.mean() / (r["high"] - r["low"]))
        w = cut[cut.ctx_stem.isin(both)].groupby(["ctx_stem", "bucket"])[m].mean().unstack()
        paired = (w["high"] - w["low"]).mean()
        lines.append(f"{name:10s} {rf['high'] - rf['low']:9.3f} {rc['high'] - rc['low']:9.3f} "
                     f"{gen.mean():+10.4f} {ratio:6.1%} [{np.percentile(boots, 2.5):5.1%}, "
                     f"{np.percentile(boots, 97.5):5.1%}]   {paired:11.3f} {gen[list(both)].mean() / paired:6.1%}")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
