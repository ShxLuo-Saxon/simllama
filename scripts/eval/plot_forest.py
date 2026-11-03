r"""
scripts/eval/plot_forest.py

Forest plot of the label comparisons on the main measures, averaged over draws.

For each measure and comparison (high>low, high>mid, mid>low):

    gap(context)  = mean over kept draws of measure(ctx, a) - measure(ctx, b)
    share         = share of contexts with a positive gap (ties dropped)
    95% interval  = resampling whole pieces (eval_effect_sizes.py, 2,000 resamples)

0.5 means the label makes no difference. Draws are dropped exactly as in
eval_robustness_table.py (any output under 10 notes), and the shares are
checked against its robustness_table.csv. Intervals and significance come from
eval_effect_sizes.py (no quality filter): a filled marker has a piece-level
one-sided p < 0.05 after the Holm correction over the five measures shown of that
comparison. The piece-level resampling replaces a context-level bootstrap and
Wilcoxon test, which treated contexts of one piece as independent.

Usage:
    .venv\Scripts\python.exe scripts\eval\plot_forest.py `
        --scores outputs\eval\robustness_scores.csv `
        --table  outputs\eval\robustness_table.csv `
        --effects outputs\eval\effect_sizes.csv `
        --out    paper\figures\forest_labels.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, valid_draws  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"

# Main-text measures, one per dimension, then the training-label check.
# Display names follow the paper's naming table (the "Names" table).
MEASURES = list(MAIN.items())
COMPARISONS = [("high", "low"), ("high", "mid"), ("mid", "low")]


def share(gap: np.ndarray) -> float:
    # The CSV round trip can turn an exact tie into ~1e-17; count it as a tie.
    nz = gap[np.abs(gap) > 1e-12]
    return float((nz > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--table", default="outputs/eval/robustness_table.csv")
    ap.add_argument("--effects", default="outputs/eval/effect_sizes.csv")
    ap.add_argument("--out", default="paper/figures/forest_labels.pdf")
    args = ap.parse_args()
    eff = pd.read_csv(args.effects)
    eff = eff[eff.quality_filter == "none"].set_index(["measure", "comparison"])

    df = pd.read_csv(args.scores)
    df = valid_draws(df)
    table = pd.read_csv(args.table).set_index(["measure", "comparison"])

    rows = []
    for m, label in MEASURES:
        w = df.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
        for a, b in COMPARISONS:
            comp = f"{a}>{b}"
            gap = (w[a] - w[b]).rename("gap").reset_index()
            gap = gap.groupby("ctx_stem").gap.mean().dropna().to_numpy()
            est = share(gap)
            ref = table.loc[(m, comp)]
            assert abs(est - ref.fp_averaged) < 1e-9, (m, comp, est, ref.fp_averaged)
            e = eff.loc[(label, f"{a}-{b}")]
            assert abs(est - e.share) < 1e-9, (m, comp, est, e.share)
            rows.append(dict(measure=m, label=label, comparison=comp, share=est,
                             lo=e.share_lo, hi=e.share_hi, p=e.p_piece_holm_main, n=len(gap)))
    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2), sharey=True)
    ys = np.arange(len(MEASURES))[::-1]
    for ax, (a, b) in zip(axes, COMPARISONS):
        sub = res[res.comparison == f"{a}>{b}"].set_index("measure")
        ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
        for y, (m, _) in zip(ys, MEASURES):
            r = sub.loc[m]
            ax.plot([r.lo, r.hi], [y, y], color=INK, lw=1.5, zorder=2)
            ax.plot(r.share, y, "o", ms=6, mec=INK, mew=1.2,
                    mfc=INK if r.p < 0.05 else "white", zorder=3)
        ax.set_title(f"sim:{a} vs sim:{b}", fontsize=9, color=INK)
        ax.set_xlim(0.25, 0.80)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in MEASURES], fontsize=8)
    axes[1].set_xlabel("Share of contexts where the first label is closer to the context",
                       fontsize=8, color=INK)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})   # same bytes on every run
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(res.round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
