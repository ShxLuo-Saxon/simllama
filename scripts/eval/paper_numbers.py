r"""
scripts/eval/paper_numbers.py

Prints the numbers the paper quotes that no single eval output gives directly, from the
current outputs/eval files (all 157 test contexts):

  - per comparison and filter: how many of the 12 measures stay positive and how many are
    significant after Holm, and which;
  - the within-context standard deviation of single-draw Melody gaps, against the mean gap;
  - the single-draw Melody shares (min and max over the six draws);
  - the quality-filter counts: outputs, unstable outputs by label, draws dropped, output
    length by label after truncation.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\paper_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MIN_NOTES, unstable, valid_draws  # noqa: E402

EFF = "outputs/eval/effect_sizes.csv"
SCORES = "outputs/eval/robustness_scores.csv"
QF_FAIL = ("coverage", "pitch_ratio", "stuck")


def main() -> None:
    t = pd.read_csv(EFF)
    for (qf, comp), d in t.groupby(["quality_filter", "comparison"], sort=False):
        sig = d[d.p_piece_holm < 0.05].measure.tolist()
        print(f"{qf:5s} {comp:9s} contexts {d.contexts.iloc[0]}: positive {int((d.mean_gap > 0).sum())}/12, "
              f"Holm < 0.05: {len(sig)} {sig}")

    s = pd.read_csv(SCORES)
    s = valid_draws(s)
    w = s.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values="interval_3gram")
    gap = (w["high"] - w["low"]).rename("gap").reset_index()
    sd = gap.groupby("ctx_stem").gap.std().mean()
    print(f"\nMelody single-draw gap: mean {gap.gap.mean():.4f}; mean within-context SD {sd:.4f} "
          f"({sd / gap.groupby('ctx_stem').gap.mean().mean():.1f} times the mean gap)")
    shares = gap[gap.gap != 0].groupby("sample").gap.apply(lambda g: (g > 0).mean())
    print(f"Single-draw Melody shares: {shares.min():.1%} to {shares.max():.1%}")

    s["unstable"] = unstable(s)
    draws = s.groupby(["ctx_stem", "sample"]).unstable.any()
    print(f"\nOutputs {len(s)} in {len(draws)} valid draws; unstable {int(s.unstable.sum())}; "
          f"by label {s[s.unstable].bucket.value_counts().to_dict()}")
    print(f"Draws with an unstable output: {int(draws.sum())} of {len(draws)}; "
          f"contexts left {draws[~draws].reset_index().ctx_stem.nunique()}")
    print(f"Mean notes after truncation by label: {s.groupby('bucket').n_notes.mean().round(0).to_dict()}")

    # failure rate per model over all outputs, the ten-note failures included
    allm = pd.concat([pd.read_csv(SCORES).assign(model="m1"), pd.read_csv("outputs/eval/controls_scores.csv")])
    allm["fails"] = (allm.n_notes < MIN_NOTES) | unstable(allm)
    names = {"m1": "SimLlama", "m3": "wrong-label", "m4": "no-label", "m2": "unconditioned", "no_prototype": "label-only"}
    print("\nOutputs failing the quality filter, ten-note failures included:")
    for m, d in allm.groupby("model"):
        print(f"  {names[m]:13s} {int(d.fails.sum()):4d} of {len(d):4d} = {d.fails.mean():.1%}")
    r = allm[allm.model == "no_prototype"]
    print(f"  label-only: contexts with every output of ten notes or more: "
          f"{int((r.groupby('ctx_stem').n_notes.min() >= MIN_NOTES).sum())}")


if __name__ == "__main__":
    main()
