r"""
scripts/eval/plot_listening_trials.py

One point per listening-test trial: a measure's gap (x) against the listeners'
gap (y), on what listeners heard (the first --seconds of each clip).

    x = measure(ctx, sim:high) - measure(ctx, sim:low)
    y = mean over valid listeners of rating_high - rating_low, slot bias removed
    error bar = 95% CI of that mean across listeners

The slot bias (A rated above B whatever it holds, from
score_listening_ratings.py) is removed as in eval_listening_metrics.py: each
rating in slot A loses --slot_bias points. Points above y = 0 are trials where
listeners heard sim:high as more similar; points right of x = 0 are trials
where the measure agrees.

Valid listeners are those who pass every attention check, as in
score_listening_ratings.py. The measure is computed from the Test 1 MIDI the
clips were rendered from (eval_listening_metrics.clip_scores).

Usage:
    .venv\Scripts\python.exe scripts\eval\plot_listening_trials.py `
        --key       listening_test\stimuli\test2_one_set_10s\answer_key.csv `
        --responses listening_test\responses\test2\ `
        --out       paper\figures\listening_test_per_trial.pdf
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_listening_metrics import clip_scores, ratings  # noqa: E402
from paper_common import MAIN  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
POINT = "#2f6fb0"
# Axis names follow the paper's naming table (the "Names" table).
NAMES = {**MAIN, "cs": "CS"}    # the figure labels CS without "(ours)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    ap.add_argument("--test1", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--measure", default="interval_3gram")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--slot_bias", type=float, default=7.86)
    ap.add_argument("--out", default="paper/figures/listening_test_per_trial.pdf")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    r = ratings(key, Path(args.responses))
    r["r_adj"] = r.rating - args.slot_bias * r.isA
    w = r.pivot_table(index=["participant_id", "ctx_stem"], columns="cond",
                      values="r_adj").reset_index()
    w["gap"] = w["high"] - w["low"]
    g = w.groupby("ctx_stem").gap.agg(["mean", "std", "size"])
    g["ci"] = stats.t.ppf(0.975, g["size"] - 1) * g["std"] / np.sqrt(g["size"])

    sc = clip_scores(key, Path(args.test1), args.seconds)
    m = sc.pivot(index="ctx_stem", columns="cond", values=args.measure)
    g["x"] = (m["high"] - m["low"]).reindex(g.index)
    rho = stats.spearmanr(g.x, g["mean"])
    agree = int((np.sign(g.x) == np.sign(g["mean"])).sum())
    n_listeners = r.participant_id.nunique()
    name = NAMES.get(args.measure, args.measure)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.errorbar(g.x, g["mean"], yerr=g.ci, fmt="none", ecolor=GRID,
                elinewidth=1.0, zorder=2)
    ax.scatter(g.x, g["mean"], s=22, color=POINT, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.set_xlabel(f"{name} gap, high $-$ low", color=INK, fontsize=8)
    ax.set_ylabel("Listener gap, high $-$ low\n(rating points, mean $\\pm$ 95% CI)",
                  color=INK, fontsize=8)
    ax.tick_params(labelsize=7, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.set_title(f"{len(g)} trials, {n_listeners} listeners", fontsize=8, color=MUTED)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})   # same bytes on every run
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    g.round(4).to_csv(out.with_suffix(".csv"))
    print(f"Wrote {out} and {out.with_suffix('.png').name}  "
          f"({len(g)} trials, {n_listeners} listeners)")
    print(f"{args.measure}: Spearman rho {rho.statistic:.3f} (p {rho.pvalue:.4f}), "
          f"same sign in {agree} of {len(g)} trials; "
          f"listener gap > 0 in {(g['mean'] > 0).sum()} trials, "
          f"measure gap > 0 in {(g.x > 0).sum()}, measure gap = 0 in {(g.x == 0).sum()}")


if __name__ == "__main__":
    main()
