r"""
scripts/eval/eval_ablations.py

The model ablations, scored with the paper's evaluation measures instead of
compound_sim.

Each output is scored as measure(context, output) with symbolic_measures and
compound similarity (CS), read exactly as eval_triplet.py reads it. Within one run folder, a
context is dropped if any of its 7 outputs has fewer than 10 notes (the
published rule), so every comparison is paired on the same contexts.

Runs (all single-draw, 157 test contexts, corrected by correct_generated_output)
  main    the main models: M1 (labels + prototype), M2 (no label, no prototype),
         M3 (trained on shuffled labels, given the correct label)
  no_prototype   the same three models retrained without the prototype at
         LoRA rank 32, to match capacity (the paper's prototype ablation)
  no_prototype_rank8    as no_prototype at LoRA rank 8 (unmatched capacity; reported for
         completeness)

Comparisons, per run (one-sided Wilcoxon over contexts, ties dropped)
  conditioning   M1 (mean of its three labels) > M2
  label meaning  M1 > M3 (each mean of three labels)
  gradient       high > low, within M1 and within M3
  gradient gap   M1's high - low gap > M3's high - low gap

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_ablations.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import compound_similarity, midi_to_compound  # noqa: E402
from symbolic_measures import all_measures  # noqa: E402
from paper_common import MAIN as MAIN_NAMES, MIN_NOTES  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
RUNS = {"main": "outputs/generated/m1_triplet",
        "no_prototype": "outputs/generated/ablation_no_prototype",
        "no_prototype_rank8": "outputs/generated/ablation_no_prototype_rank8"}
PAT = re.compile(r"^(.+)_(m[13])_(high|mid|low)_generated\.mid$|^(.+)_(m2)_generated\.mid$")
MAIN = list(MAIN_NAMES)


def score_run(d: Path, sec: Path) -> pd.DataFrame:
    rows, ctx = [], {}
    for f in sorted(d.glob("*_generated.mid")):
        m = PAT.match(f.name)
        if not m:
            continue
        stem = m.group(1) or m.group(4)
        model = m.group(2) or m.group(5)
        label = m.group(3) or "none"
        if stem not in ctx:
            ctx[stem] = np.load(str(sec / f"{stem}.npy"))
        g = midi_to_compound(f)
        if len(g):
            g = g.copy()
            g[:, 0] -= g[0, 0]
        row = dict(ctx_stem=stem, model=model, label=label, n_notes=len(g))
        if len(g) >= MIN_NOTES:
            row.update(all_measures(ctx[stem], g))
            row["cs"] = compound_similarity(ctx[stem].astype(float), g.astype(float))
        rows.append(row)
    return pd.DataFrame(rows)


def test(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    nz = x[x != 0]
    p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else np.nan
    return (nz > 0).mean(), p, len(x)


def comparisons(df: pd.DataFrame, m: str) -> list[dict]:
    w = df.pivot_table(index="ctx_stem", columns=["model", "label"], values=m)
    m1 = w["m1"].mean(axis=1)
    m3 = w["m3"].mean(axis=1) if "m3" in w else None
    out = []

    def add(name, x):
        fp, p, n = test(x)
        out.append(dict(measure=m, comparison=name, fp=fp, p=p, n=n,
                        mean_gap=x.mean()))
    add("M1 > M2 (conditioning)", m1 - w[("m2", "none")])
    if m3 is not None:
        add("M1 > M3 (label meaning)", m1 - m3)
        # M3 has the prototype and a label without meaning; M2 has neither, so this
        # isolates the prototype
        add("M3 > M2 (prototype)", m3 - w[("m2", "none")])
    add("M1 high > low", w[("m1", "high")] - w[("m1", "low")])
    if m3 is not None:
        add("M3 high > low", w[("m3", "high")] - w[("m3", "low")])
        add("M1 gap > M3 gap", (w[("m1", "high")] - w[("m1", "low")])
            - (w[("m3", "high")] - w[("m3", "low")]))
    add("M1 high > M2", w[("m1", "high")] - w[("m2", "none")])
    add("M2 > M1 low", w[("m2", "none")] - w[("m1", "low")])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections_dir", default=str(SEC))
    ap.add_argument("--out_csv", default="outputs/eval/ablations_table.csv")
    args = ap.parse_args()
    sec = Path(args.sections_dir)

    res, kept = [], []
    for run, d in RUNS.items():
        df = score_run(Path(d), sec)
        short = df.groupby("ctx_stem").n_notes.transform("min") < MIN_NOTES
        n_ctx = df.ctx_stem.nunique()
        df = df[~short]
        kept.append(df.assign(run=run))
        print(f"{run}: {n_ctx} contexts, {df.ctx_stem.nunique()} kept "
              f"(any of 7 outputs < 10 notes drops the context)")
        measures = [c for c in df.columns
                    if c not in ("ctx_stem", "model", "label", "n_notes")]
        for m in measures:
            for r in comparisons(df, m):
                res.append(dict(run=run, **r))
    res = pd.DataFrame(res)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    # per-output scores of the kept contexts, for eval_ablations_piece.py
    pd.concat(kept).to_csv(out.with_name("ablation_scores.csv"), index=False)

    for run in RUNS:
        t = res[res.run == run]
        print("\n" + "=" * 86)
        print(f"{run}: main measures   (fp = share of contexts with a positive gap; "
              "p one-sided)")
        print("=" * 86)
        main = t[t.measure.isin(MAIN)].pivot_table(
            index="comparison", columns="measure", values=["fp", "p"])
        for comp in main.index:
            cells = "  ".join(f"{m[:14]:>14s} {main.loc[comp, ('fp', m)]:.2f} "
                              f"p{main.loc[comp, ('p', m)]:.3f}" for m in MAIN)
            print(f"  {comp:24s} {cells}")
        n_all = t.groupby("comparison").apply(
            lambda g: f"{(g.p < 0.05).sum()}/{len(g)} p<.05, "
                      f"{(g.fp > 0.5).sum()}/{len(g)} positive", include_groups=False)
        print("  all measures:")
        for comp, s in n_all.items():
            print(f"    {comp:24s} {s}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
