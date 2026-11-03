r"""
scripts/eval/eval_copy_rate.py

Does `sim:high` sound similar because it copies the context?

A model asked for "high similarity" could reach it the cheap way: by repeating
the context note for note. That would score well on any similarity measure and
still not be what a composer means by a related section. This script measures
how much of each generated melody is lifted from its context, and compares it
with real game music: how much a composer's own related section lifts from its
context.

Melody is the skyline (highest note per onset), as in symbolic_measures.py.
Everything is transposition-invariant: melodies are compared as intervals.

Measures, per (context, output) pair
------------------------------------
  reuse4        share of the output's 4-interval melodic patterns (5 notes)
                that also occur in the context
  reuse4_rhythm the same, but a pattern must match in rhythm too
                (interval and inter-onset time, in sixteenths)
  longest_run   longest stretch of notes copied in a row, in notes
  copy_share    share of the output's melody inside copied stretches of 8 or
                more notes -- the part a listener would hear as a quotation

Compared
--------
  M1 sim:high / sim:mid / sim:low and M2 (no label), from the corrected
  published outputs (all 157 test contexts, as the paper); M4 (context vector, no
  label), first draw of the retrained control;
  real targets from pairs.csv, test split, by their similarity bucket.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_copy_rate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import midi_to_compound  # noqa: E402
from symbolic_measures import SIXTEENTH, melody  # noqa: E402
from paper_common import MIN_NOTES  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
N = 4               # intervals per pattern
MIN_COPY = 8        # notes in a row to count as a quotation


def seqs(t: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Interval sequence and (interval, ioi) sequence of the skyline melody."""
    o, p = melody(t)
    iv = np.diff(p).astype(int).tolist()
    ioi = np.round(np.diff(o) / SIXTEENTH).astype(int).tolist()
    return iv, list(zip(iv, ioi))


def reuse(ctx: list, out: list, n: int = N) -> float:
    grams = {tuple(ctx[i:i + n]) for i in range(len(ctx) - n + 1)}
    mine = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
    return float(np.mean([g in grams for g in mine])) if mine else np.nan


def runs(ctx: list, out: list) -> tuple[int, float]:
    """Longest copied run (notes) and share of `out` notes in runs >= MIN_COPY."""
    if not ctx or not out:
        return 0, np.nan
    # best[j] = longest run of matching intervals ending at out[j]
    best = np.zeros(len(out), dtype=int)
    prev = np.zeros(len(ctx) + 1, dtype=int)
    for j, b in enumerate(out):
        cur = np.zeros(len(ctx) + 1, dtype=int)
        for i, a in enumerate(ctx):
            if a == b:
                cur[i + 1] = prev[i] + 1
        best[j] = cur.max()
        prev = cur
    covered = np.zeros(len(out) + 1, dtype=bool)        # notes = intervals + 1
    for j, L in enumerate(best):
        if L + 1 >= MIN_COPY:
            covered[j + 1 - L: j + 2] = True
    return int(best.max()) + 1 if best.max() else 0, float(covered.mean())


def measure(ctx_t: np.ndarray, out_t: np.ndarray) -> dict:
    ci, cr = seqs(ctx_t)
    oi, orr = seqs(out_t)
    longest, share = runs(ci, oi)
    return dict(reuse4=reuse(ci, oi), reuse4_rhythm=reuse(cr, orr),
                longest_run=longest, copy_share=share)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="outputs/generated/m1_triplet")
    ap.add_argument("--m4_dir", default="outputs/generated/m4",
                    help="the no-label control (context vector, no label); first draw")
    ap.add_argument("--s0_csv", default="outputs/eval/contexts_all.csv")   # all 157 test contexts
    ap.add_argument("--sections_dir", default=str(SEC))
    ap.add_argument("--out_csv", default="outputs/eval/copy_rate.csv")
    args = ap.parse_args()

    sec, gen = Path(args.sections_dir), Path(args.gen_dir)
    stems = sorted(set(pd.read_csv(args.s0_csv).ctx_stem))
    pairs = pd.read_csv(sec / "pairs.csv")
    pairs = pairs[pairs.split == "test"].assign(
        ctx_stem=lambda d: d.context_file.str.removesuffix(".npy"))

    rows = []
    for s in stems:
        ctx = np.load(str(sec / f"{s}.npy"))
        ctx = ctx[np.argsort(ctx[:, 0], kind="stable")]
        for src, f in (("M1 high", f"{s}_m1_high_generated.mid"),
                       ("M1 mid", f"{s}_m1_mid_generated.mid"),
                       ("M1 low", f"{s}_m1_low_generated.mid"),
                       ("M2 no label", f"{s}_m2_generated.mid"),
                       ("M4 no label", Path(args.m4_dir).resolve() / f"{s}_m4_s0_generated.mid")):
            out = midi_to_compound(gen / f)
            if len(out) >= MIN_NOTES:
                rows.append(dict(ctx_stem=s, source=src, **measure(ctx, out)))
        for p in pairs[pairs.ctx_stem == s].itertuples():
            tgt = np.load(str(sec / p.target_file))
            tgt = tgt[np.argsort(tgt[:, 0], kind="stable")]
            rows.append(dict(ctx_stem=s, source=f"real {p.sim_bucket.split(':')[1]}",
                             **measure(ctx, tgt)))
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # one value per context and source (real targets can be several)
    per = df.groupby(["ctx_stem", "source"]).mean(numeric_only=True).reset_index()
    order = ["real high", "real mid", "real low", "M1 high", "M1 mid", "M1 low",
             "M2 no label", "M4 no label"]
    cols = ["reuse4", "reuse4_rhythm", "longest_run", "copy_share"]
    print(f"Contexts: {len(stems)}  (published set)\n")
    print("Mean over contexts (median in brackets):")
    summ = per.groupby("source")[cols].agg(["mean", "median"]).reindex(order)
    for src, r in summ.iterrows():
        print(f"  {src:12s} n {int((per.source == src).sum()):3d}   " + "   ".join(
            f"{c} {r[(c, 'mean')]:.3f} ({r[(c, 'median')]:.3f})" for c in cols))

    print("\nShare of outputs that quote the context (copy_share > 0) and that are")
    print("mostly quotation (copy_share >= 0.5):")
    for src in order:
        x = per[per.source == src].copy_share.dropna()
        print(f"  {src:12s} any {(x > 0).mean():.1%}   mostly {(x >= 0.5).mean():.1%}")

    print("\nPaired tests over contexts (Wilcoxon, two-sided):")
    w = per.pivot(index="ctx_stem", columns="source", values="reuse4_rhythm")
    for a, b in (("M1 high", "M1 low"), ("M1 high", "M1 mid"), ("M1 high", "M2 no label"),
                 ("M1 high", "real high"), ("M1 low", "real low")):
        x = w[[a, b]].dropna()
        if len(x) > 5:
            p = stats.wilcoxon(x[a], x[b]).pvalue
            print(f"  reuse4_rhythm  {a:10s} vs {b:12s} n {len(x):3d}   "
                  f"{x[a].mean():.3f} vs {x[b].mean():.3f}   p {p:.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
