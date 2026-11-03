r"""
scripts/eval/eval_gap_without_copies.py

Does the label effect survive without the outputs that copy their context?
12.4% of sim:high outputs copy a melodic run of 8 or more notes from the
context, 7.1% of sim:low ones; the excess could carry the Melody gap.

An output copies when its top voice shares such a run with the context
(eval_copy_rate.measure, copy_share > 0), as in eval_copy_matched.py. For SimLlama
(all valid draws) and the no-label control (all twelve draws of 10 notes or more)
the script drops every copying output, averages each context's remaining outputs
per label, and compares:

  high above low        SimLlama sim:high minus sim:low
  high above no label   SimLlama sim:high minus the no-label control

on the four main measures and CS, first with all outputs (it must reproduce the
published gaps), then without copying outputs. A context drops out of a
comparison if one side has no output left. Mean gaps come with 95% intervals and
one-sided p from 2,000 resamples of whole pieces, Holm over the five per row.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_gap_without_copies.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_copy_matched import m1_file  # noqa: E402
from eval_copy_rate import SEC, measure  # noqa: E402
from paper_common import MAIN, MIN_NOTES, cluster_boot, holm, load_generated  # noqa: E402
from similarity import midi_to_compound  # noqa: E402


def copies(df: pd.DataFrame, path) -> pd.Series:
    """True for each output (row) whose top voice copies a run of 8+ notes of its context."""
    out, ctx_cache = [], {}
    for r in df.itertuples():
        if r.ctx_stem not in ctx_cache:
            c = np.load(str(SEC / f"{r.ctx_stem}.npy"))
            ctx_cache[r.ctx_stem] = c[np.argsort(c[:, 0], kind="stable")]
        out.append(measure(ctx_cache[r.ctx_stem], midi_to_compound(path(r)))["copy_share"] > 0)
    return pd.Series(out, index=df.index)


def compare(a: pd.DataFrame, b: pd.DataFrame, rng, n: int) -> list[dict]:
    rows = []
    for m, name in MAIN.items():
        gap = (a.groupby("ctx_stem")[m].mean() - b.groupby("ctx_stem")[m].mean()).dropna()
        means, _ = cluster_boot(gap, rng, n)
        rows.append(dict(measure=name, n=len(gap), gap=gap.mean(), lo=np.percentile(means, 2.5),
                         hi=np.percentile(means, 97.5), p=(means <= 0).mean()))
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["holm"] = p
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--controls", default="outputs/eval/controls_scores.csv")
    ap.add_argument("--m4_dir", default="outputs/generated/m4")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/gap_without_copies")
    args = ap.parse_args()

    m1 = load_generated(args.scores)
    m1["copied"] = copies(m1, lambda r: m1_file(r.ctx_stem, r.bucket, r.sample))
    c = pd.read_csv(args.controls)
    m4 = c[(c.model == "m4") & (c.n_notes >= MIN_NOTES) & c.ctx_stem.isin(set(m1.ctx_stem))].copy()
    m4["copied"] = copies(m4, lambda r: Path(args.m4_dir) / f"{r.ctx_stem}_m4_s{r.sample}_generated.mid")

    lines, rows = [], []
    for label, drop in (("all outputs", False), ("copying outputs left out", True)):
        a1 = m1[~m1.copied] if drop else m1
        a4 = m4[~m4.copied] if drop else m4
        hi, lo = a1[a1.bucket == "high"], a1[a1.bucket == "low"]
        lines.append(f"{label}: SimLlama high {len(hi)}, low {len(lo)}, no label {len(a4)} outputs")
        for comp, (x, y) in {"high above low": (hi, lo), "high above no label": (hi, a4)}.items():
            for r in compare(x, y, np.random.default_rng(0), args.boot):
                rows.append(dict(outputs=label, comparison=comp, **r))
                lines.append(f"  {comp:20s} {r['measure']:10s} n {r['n']:3d}  {r['gap']:+.4f} "
                             f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  p {r['p']:.3f}  Holm {r['holm']:.3f}")
    share = m1.groupby("bucket").copied.mean()
    lines.append(f"copying share: high {share['high']:.1%}, mid {share['mid']:.1%}, "
                 f"low {share['low']:.1%}, no label {m4.copied.mean():.1%} (outputs)")
    pd.DataFrame(rows).to_csv(args.out + ".csv", index=False)
    text = "\n".join(lines) + "\n"
    Path(args.out + ".txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
