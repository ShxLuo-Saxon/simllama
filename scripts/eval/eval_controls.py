r"""
scripts/eval/eval_controls.py

The control models, sampled several times (train_controls.sh).

Scores every output of the retrained controls with the paper's measures, exactly
as eval_robustness_table.py scores M1 (context .npy as stored, output shifted to
start at 0, all 157 test contexts):

  M2   no context vector, no label       m2, 12 draws
  M3   context vector, permuted label    m3, 6 draws per label
  M4   context vector, no label          m4, 12 draws

M1 comes from outputs/eval/robustness_scores.csv (6 draws per label).

Validity: a draw is dropped if one of its outputs has fewer than 10 notes or fails
the quality filter (eval_effect_sizes.load_generated, "draw"); for M2 and M4 a draw
is one output. Each model is reduced to its mean per context and label over the
valid draws, then the gaps are taken.

Comparisons (gap = first - second; intervals and p from resampling whole pieces;
Holm over the five main measures of each comparison):

  the label, context vector fixed
    M1 high above M4                    a high request comes closer than no label
    M4 above M1 low                     a low request moves further away
    M4 above M1 mid (two-sided)         a mid request moves further away (two-sided: no direction was planned)
    M1 high above low                   for reference, on the same contexts
    M3 high above low, two-sided        permuted labels: no ordering expected
    M1 spread above M3 spread           (high - low) of M1 minus that of M3
  the context vector, no label
    M4 above M2
    Label only above M2         no-prototype M1 (label, no vector, rank 32, one
                                        draw; ablation_no_prototype), mean of its
                                        labels, same draw filter as M1
  generate-and-select (exploratory)
    from M4's 12 draws per context, keep the draw with the highest and the lowest
    compound similarity; M1 high above the highest, the lowest above M1 low

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_controls.py              # score all, then compare
    .venv\Scripts\python.exe scripts\eval\eval_controls.py --score no_prototype # rescore one model
    .venv\Scripts\python.exe scripts\eval\eval_controls.py --score      # compare from the CSV
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, MIN_NOTES, cluster_boot, holm, load_generated, unstable  # noqa: E402
from eval_robustness_table import SEC, score_dir  # noqa: E402

DIRS = {"m2": "outputs/generated/m2",
        "m3": "outputs/generated/m3",
        "m4": "outputs/generated/m4",
        "no_prototype": "outputs/generated/ablation_no_prototype"}
# score_dir reads group 1 as the context, group 2 as the label, group 3 as the draw
PATTERNS = {"m2": re.compile(r"^(.+)_(m2)_s(\d+)_generated\.mid$"),
            "m3": re.compile(r"^(.+)_m3_(high|mid|low)_s(\d+)_generated\.mid$"),
            "m4": re.compile(r"^(.+)_(m4)_s(\d+)_generated\.mid$"),
            "no_prototype": re.compile(r"^(.+)_m1_(high|mid|low)_generated\.mid$")}
LABELLED = ("m3", "no_prototype")


def score(models: list[str], stems: set[str], out: Path) -> None:
    """Score the given models; their old rows in `out` are replaced, others kept."""
    rows = [pd.read_csv(out)] if out.exists() else []
    if rows:
        rows[0] = rows[0][~rows[0].model.isin(models)]
    for model in models:
        r = pd.DataFrame(score_dir(Path(DIRS[model]), PATTERNS[model], stems, SEC,
                                   0 if model == "no_prototype" else None))
        r.insert(0, "model", model)
        if model not in LABELLED:
            r["bucket"] = "none"
        rows.append(r)
        print(f"{model}: {len(r)} outputs scored")
    pd.concat(rows).to_csv(out, index=False)


def per_context(scores: Path, controls: Path, qf: str) -> pd.DataFrame:
    """Mean per context, model and label over the valid draws, for every measure.
    qf "none": a draw goes only if an output has fewer than 10 notes (the main
    analysis); "draw": also if an output fails the quality filter (the check)."""
    m1 = load_generated(str(scores), qf).assign(model="m1")
    c = pd.read_csv(controls)
    labelled = [load_generated(c[c.model == m], qf) for m in LABELLED]
    solo = c[~c.model.isin(LABELLED)].copy()
    bad = solo.n_notes < MIN_NOTES
    if qf == "draw":
        bad |= unstable(solo)
    df = pd.concat([m1, *labelled, solo[~bad]], ignore_index=True)
    return df[df.ctx_stem.isin(set(m1.ctx_stem))]


def selected(df: pd.DataFrame, pick: str) -> pd.DataFrame:
    """Per context, the M4 draw with the highest (pick="max") or lowest CS."""
    m4 = df[df.model == "m4"]
    idx = m4.groupby("ctx_stem").cs.idxmax() if pick == "max" \
        else m4.groupby("ctx_stem").cs.idxmin()
    return m4.loc[idx.values].set_index("ctx_stem")


def gaps(df: pd.DataFrame, m: str) -> dict[str, pd.Series]:
    w = df.groupby(["ctx_stem", "model", "bucket"])[m].mean().unstack(["model", "bucket"])
    m4, m2 = w[("m4", "none")], w[("m2", "none")]
    hi, lo = selected(df, "max")[m], selected(df, "min")[m]
    m1s = w[("m1", "high")] - w[("m1", "low")]
    m3s = w[("m3", "high")] - w[("m3", "low")]
    return {"M1 high above M4": w[("m1", "high")] - m4,
            "M4 above M1 low": m4 - w[("m1", "low")],
            "M4 above M1 mid (two-sided)": m4 - w[("m1", "mid")],
            "M1 high above low": m1s,
            "M3 high above low (two-sided)": m3s,
            "M1 spread above M3 spread": m1s - m3s,
            "M4 above M2": m4 - m2,
            "Label only above M2": w["no_prototype"].mean(axis=1) - m2,
            "M1 high above M4 best-of-12": w[("m1", "high")] - hi,
            "M4 worst-of-12 above M1 low": lo - w[("m1", "low")]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--controls", default="outputs/eval/controls_scores.csv")
    ap.add_argument("--s0_csv", default="outputs/eval/contexts_all.csv")   # all 157 test contexts
    ap.add_argument("--score", nargs="*", default=list(DIRS), choices=list(DIRS),
                    help="models to (re)score; none to compare from the CSV")
    ap.add_argument("--quality_filter", choices=("none", "draw"), default="none")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/controls")
    args = ap.parse_args()
    if args.score:
        score(args.score, set(pd.read_csv(args.s0_csv).ctx_stem), Path(args.controls))

    df = per_context(Path(args.scores), Path(args.controls), args.quality_filter)
    args.out += "" if args.quality_filter == "none" else "_qf"
    draws = df.groupby(["model", "bucket"]).apply(
        lambda d: d.groupby("ctx_stem")["sample"].nunique().mean(), include_groups=False)
    rows, block = [], {}
    for m, label in MAIN.items():
        for comp, g in gaps(df, m).items():
            g = g.dropna()
            # one generator per cell, so adding a comparison leaves the others unchanged
            means, shares = cluster_boot(g, np.random.default_rng(0), args.boot)
            nz = g[g.abs() > 1e-12]
            lo_tail = ((means <= 0).sum() + 1) / (args.boot + 1)
            hi_tail = ((means >= 0).sum() + 1) / (args.boot + 1)
            p = min(1.0, 2 * min(lo_tail, hi_tail)) if "two-sided" in comp else lo_tail
            block.setdefault(comp, []).append(dict(
                comparison=comp, measure=label, contexts=len(g),
                pieces=g.index.str[:5].nunique(),
                share=(nz > 0).mean(), share_lo=np.percentile(shares, 2.5),
                share_hi=np.percentile(shares, 97.5), mean_gap=g.mean(),
                gap_lo=np.percentile(means, 2.5), gap_hi=np.percentile(means, 97.5), p=p))
    for comp, b in block.items():
        for r, h in zip(b, holm(np.array([x["p"] for x in b]))):
            r["p_holm"] = h
        rows += b
    out = pd.DataFrame(rows)
    out.to_csv(args.out + ".csv", index=False)

    lines = ["Mean valid draws per context: " +
             ", ".join(f"{k[0]} {k[1]} {v:.2f}" for k, v in draws.items())]
    for comp, d in out.groupby("comparison", sort=False):
        lines.append(f"\n{comp} | {d.contexts.iloc[0]} contexts, {d.pieces.iloc[0]} pieces")
        for _, r in d.iterrows():
            lines.append(f"  {r.measure:10s} share {r.share:.3f} [{r.share_lo:.3f}, {r.share_hi:.3f}]  "
                         f"gap {r.mean_gap:+.4f} [{r.gap_lo:+.4f}, {r.gap_hi:+.4f}]  "
                         f"p {r.p:.4f}  Holm {r.p_holm:.4f}")
    text = "\n".join(lines) + "\n"
    Path(args.out + ".txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
