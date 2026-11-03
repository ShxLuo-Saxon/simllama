r"""
scripts/eval/eval_real_gap_paired.py

Is the real gap of Table 2 comparable to the generated gap?
The real gap of eval_effect_sizes.py pools all real high-similarity targets of the
test contexts against all real low-similarity targets, and the two sets come from
different contexts. This script prints, for the four main measures and CS:

  1. the pooled real gap of eval_effect_sizes.py, for reference;
  2. the real gap within the test contexts that have both a high- and a
     low-similarity target: high minus low per context, then the mean over them;
  3. the generated gap on those same contexts and the ratio generated / real;
  4. mean CS of the real targets per label over the test contexts, and of the
     generated outputs (valid draws, no quality filter).

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_real_gap_paired.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, PAIRS, load_generated  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--controls", default="outputs/eval/controls_scores.csv")
    ap.add_argument("--out", default="outputs/eval/real_gap_paired.txt")
    args = ap.parse_args()

    t = pd.read_csv(args.pairs_measures).merge(
        pd.read_csv(args.pairs)[["pair_id", "sim_bucket"]], on="pair_id")
    t["bucket"] = t.sim_bucket.str.replace("sim:", "")
    t["ctx_stem"] = t.context_file.str.replace(".npy", "", regex=False)
    g = load_generated(args.scores, "none")
    real = t[t.ctx_stem.isin(set(g.ctx_stem))]

    per_ctx = real.groupby(["ctx_stem", "bucket"])[list(MAIN)].mean()
    both = sorted(set(per_ctx.xs("high", level="bucket").index)
                  & set(per_ctx.xs("low", level="bucket").index))
    lines = [f"Test contexts {g.ctx_stem.nunique()}; with both a high and a low real target: {len(both)}",
             "", f"{'measure':10s} {'pooled real':>12s} {'paired real':>12s} {'gen (paired)':>13s} {'ratio':>7s}"]
    for m, name in MAIN.items():
        pooled = real[real.bucket == "high"][m].mean() - real[real.bucket == "low"][m].mean()
        paired = (per_ctx.xs("high", level="bucket").loc[both, m]
                  - per_ctx.xs("low", level="bucket").loc[both, m]).mean()
        w = g.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
        gen = (w["high"] - w["low"]).groupby("ctx_stem").mean().loc[both].mean()
        lines.append(f"{name:10s} {pooled:12.4f} {paired:12.4f} {gen:13.4f} {gen / paired:7.3f}")

    lines += ["", "Mean CS per label: real targets of the test contexts; generated outputs (context means)"]
    for b in ("high", "mid", "low"):
        r = real[real.bucket == b].cs.mean()
        gm = g[g.bucket == b].groupby("ctx_stem").cs.mean().mean()
        go = g[g.bucket == b].cs.mean()
        lines.append(f"  {b:5s} real {r:.4f} ({(real.bucket == b).sum()} targets)  "
                     f"generated {gm:.4f} (over outputs {go:.4f})")
    # the no-label control, where an unlabelled model lands
    c = load_generated(args.controls)
    c = c[c.model == "m4"]
    lines.append(f"  no-label control                       generated {c.groupby('ctx_stem').cs.mean().mean():.4f} "
                 f"(over outputs {c.cs.mean():.4f})")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
