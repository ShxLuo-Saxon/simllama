r"""
scripts/eval/eval_robustness_table.py

Is the high > low result robust to the choice of measure and to the luck of one
generation draw?

For every measure in symbolic_measures.all_measures plus compound similarity
(CS), scores
measure(context, output) for M1 high / mid / low, over:

  draw 0     the published outputs (m1_triplet), one draw per context
  draws 1-K  infer_resample.py outputs, corrected (m1_resample)

and reports, per measure and comparison:

  published   fp and one-sided Wilcoxon p on draw 0 alone (the paper's design)
  averaged    the same on each context's mean gap over all draws

Reading follows eval_triplet.py exactly: the context .npy as stored, the output
shifted to start at 0, and all 157 test contexts (eval_triplet.py keeps 151). A draw is dropped for a
context if any of its three outputs has fewer than 10 notes.

--quality_filter also drops a draw if any of its three outputs fails a
listening-stimulus quality filter (triage_listening_stimuli.py thresholds):
it stops early (spans < 55% of the context), is stuck on few pitches (< 15% of
the context's distinct pitches), or has a stuck melody (<= 3 distinct melody
pitches while the context has >= 8). Dropping the whole draw, never one output,
keeps the filter blind to the label.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_robustness_table.py
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
from symbolic_measures import all_measures, melody  # noqa: E402
from eval_resample_stability import icc_oneway  # noqa: E402
from paper_common import MIN_NOTES, output_flags, unstable, valid_draws  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
PUB = re.compile(r"^(.+)_m1_(high|mid|low)_generated\.mid$")
RES = re.compile(r"^(.+)_m1_(high|mid|low)_s(\d+)_generated\.mid$")
COMPARISONS = [("high", "low"), ("mid", "low"), ("high", "mid")]


def score_dir(d: Path, pattern: re.Pattern, stems: set[str], sec: Path,
              fixed_sample: int | None) -> list[dict]:
    rows, ctx = [], {}
    for f in sorted(d.glob("*_generated.mid")):
        m = pattern.match(f.name)
        if not m or m.group(1) not in stems:
            continue
        stem, bucket = m.group(1), m.group(2)
        sample = fixed_sample if fixed_sample is not None else int(m.group(3))
        if stem not in ctx:
            ctx[stem] = np.load(str(sec / f"{stem}.npy"))
        c = ctx[stem]
        g = midi_to_compound(f)
        if len(g):
            g = g.copy()
            g[:, 0] -= g[0, 0]
        row = dict(ctx_stem=stem, sample=sample, bucket=bucket, n_notes=len(g))
        if len(g):
            row.update(output_flags(c, g, melody))
        if len(g) >= MIN_NOTES:
            row.update(all_measures(ctx[stem], g))
            row["cs"] = compound_similarity(ctx[stem].astype(float), g.astype(float))
        rows.append(row)
    return rows


def test(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    nz = x[x != 0]
    p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 5 else np.nan
    return (nz > 0).mean(), p, len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published_dir", default="outputs/generated/m1_triplet")
    ap.add_argument("--resample_dir", default="outputs/generated/m1_resample")
    ap.add_argument("--s0_csv", default="outputs/eval/contexts_all.csv")   # all 157 test contexts
    ap.add_argument("--sections_dir", default=str(SEC))
    ap.add_argument("--out_csv", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--quality_filter", action="store_true")
    args = ap.parse_args()

    sec = Path(args.sections_dir)
    stems = set(pd.read_csv(args.s0_csv).ctx_stem)
    rows = score_dir(Path(args.published_dir), PUB, stems, sec, 0)
    rows += score_dir(Path(args.resample_dir), RES, stems, sec, None)
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # drop a draw for a context if any of its three outputs is short
    df = valid_draws(df)
    if args.quality_filter:
        bad = unstable(df)
        bad_draw = bad.groupby([df.ctx_stem, df["sample"]]).transform("any")
        print(f"Quality filter: {bad.sum()} outputs fail; dropping "
              f"{df[bad_draw].groupby(['ctx_stem', 'sample']).ngroups} draws; by label "
              f"{df[bad].bucket.value_counts().to_dict()}")
        df = df[~bad_draw]
    measures = [c for c in df.columns if c not in
                ("ctx_stem", "sample", "bucket", "n_notes", "coverage",
                 "pitch_ratio", "stuck")]
    print(f"Contexts {df.ctx_stem.nunique()}, draws {sorted(df['sample'].unique())}, "
          f"draws kept {df.groupby(['ctx_stem', 'sample']).ngroups}")

    res = []
    for m in measures:
        w = df.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
        for a, b in COMPARISONS:
            gap = (w[a] - w[b]).rename("gap").reset_index()
            fp0, p0, n0 = test(gap[gap["sample"] == 0].set_index("ctx_stem").gap)
            avg = gap.groupby("ctx_stem").gap.mean()
            fpa, pa, na = test(avg)
            # stability: each draw alone, and how much of the gap's variance is
            # between contexts rather than draw to draw (ICC, see
            # eval_resample_stability.py)
            per_draw = [test(g.set_index("ctx_stem").gap)[0]
                        for _, g in gap.groupby("sample")]
            icc = icc_oneway(gap.pivot(index="ctx_stem", columns="sample",
                                       values="gap"))[0]
            res.append(dict(measure=m, comparison=f"{a}>{b}", fp_published=fp0,
                            p_published=p0, fp_averaged=fpa, p_averaged=pa,
                            n=na, mean_gap=avg.mean(),
                            fp_draw_min=min(per_draw), fp_draw_max=max(per_draw),
                            icc=icc))
    res = pd.DataFrame(res)
    res.to_csv(out.with_name(out.stem.replace("scores", "table") + ".csv"), index=False)
    for comp in [f"{a}>{b}" for a, b in COMPARISONS]:
        print("\n" + "=" * 78)
        print(f"{comp}   fp = share of contexts with a positive gap (ties dropped); "
              "p one-sided")
        print("=" * 78)
        t = res[res.comparison == comp].set_index("measure").drop(columns="comparison")
        print(t.sort_values("p_averaged").round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
