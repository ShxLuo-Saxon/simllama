r"""
scripts/eval/eval_copy_matched.py

Is the low quotation rate of sim:high outputs only a sign of their low similarity?
All draws are used.

Quotation follows eval_copy_rate.py: a melodic run of 8 or more notes, in
intervals, shared with the context. For each source it gives the share of
outputs (or real targets) that contain a quotation and that are mostly
quotation (copy_share >= 0.5):

  M1 high / mid / low: all six draws, valid draws only (paper_common.valid_draws);
  M4 no label: all twelve draws, outputs of 10 notes or more;
  real test targets by label (one row per pair);
  real test targets whose compound similarity (CS) lies within the middle
  50% of the M1 sim:high outputs' values (all valid draws): real sections as
  similar to their context as the model's sim:high outputs.

A context's value for a model is the share of its draws that quote, so each
context counts once. Generated outputs are cut at the context's length before
scoring (correct_generated_output.py); real targets are cut the same way here,
since a longer section has more chances to contain a run.
Shares come with 95% intervals from resampling whole pieces (2,000 resamples).

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_copy_matched.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_copy_rate import SEC, measure  # noqa: E402
from paper_common import MIN_NOTES, valid_draws  # noqa: E402
from similarity import midi_to_compound  # noqa: E402

GEN = Path("outputs/generated")


def m1_file(stem: str, bucket: str, draw: int) -> Path:
    """Draw 0 is the published triplet; draws 1-5 are the resample."""
    if draw == 0:
        return GEN / "m1_triplet" / f"{stem}_m1_{bucket}_generated.mid"
    return GEN / "m1_resample" / f"{stem}_m1_{bucket}_s{draw}_generated.mid"


def boot(groups: dict, rng, n: int = 2000) -> np.ndarray:
    """Mean of the values of resampled pieces."""
    keys = list(groups)
    return np.array([np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]).mean()
                     for _ in range(n)])


def share(d: pd.DataFrame, rng) -> str:
    """Share quoting (per context: share of its draws), interval over pieces; share mostly quotation."""
    q = d.assign(q=d.copy_share > 0, m=d.copy_share >= 0.5).groupby(["piece", "unit"])[["q", "m"]].mean()
    groups = {p: g.q.to_numpy() for p, g in q.groupby(level="piece")}
    b = boot(groups, rng)
    return (f"n {len(d):4d} outputs, {len(q):3d} units ({len(groups)} pieces)   contain {q.q.mean():6.1%} "
            f"[{np.percentile(b, 2.5):.1%}, {np.percentile(b, 97.5):.1%}]   mostly {q.m.mean():6.1%}")


def per_unit(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(q=d.copy_share > 0).groupby(["piece", "unit"]).q.mean().reset_index()


def diff(a: pd.DataFrame, b: pd.DataFrame, rng, n: int = 2000) -> str:
    """Share quoting in a minus that in b, both per unit, resampling the pieces of both together."""
    a, b = per_unit(a), per_unit(b)
    pieces = sorted(set(a.piece) | set(b.piece))
    ga = {p: a.q[a.piece == p].to_numpy() for p in pieces}
    gb = {p: b.q[b.piece == p].to_numpy() for p in pieces}
    out = []
    for _ in range(n):
        s = rng.choice(pieces, len(pieces))
        xa, xb = np.concatenate([ga[p] for p in s]), np.concatenate([gb[p] for p in s])
        if len(xa) and len(xb):
            out.append(xa.mean() - xb.mean())
    d = a.q.mean() - b.q.mean()
    return (f"{'M1 high minus matched real':28s} {d:+6.1%}  "
            f"[{np.percentile(out, 2.5):+.1%}, {np.percentile(out, 97.5):+.1%}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--m4_dir", default="outputs/generated/m4")
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--out", default="outputs/eval/copy_matched.txt")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    gen = valid_draws(pd.read_csv(args.scores))
    stems = sorted(set(pd.read_csv("outputs/eval/contexts_all.csv").ctx_stem))
    lo, hi = np.nanpercentile(gen[gen.bucket == "high"].cs, [25, 75])
    valid = set(zip(gen.ctx_stem, gen["sample"]))

    rows = []
    for s in stems:
        ctx = np.load(str(SEC / f"{s}.npy"))
        ctx = ctx[np.argsort(ctx[:, 0], kind="stable")]
        for k in range(6):
            if (s, k) not in valid:
                continue
            for b in ("high", "mid", "low"):
                rows.append(dict(source=f"M1 {b}", piece=s[:5], unit=s,
                                 **measure(ctx, midi_to_compound(m1_file(s, b, k)))))
        for k in range(12):
            f = Path(args.m4_dir) / f"{s}_m4_s{k}_generated.mid"
            out = midi_to_compound(f) if f.exists() else []
            if len(out) >= MIN_NOTES:
                rows.append(dict(source="M4 no label", piece=s[:5], unit=s, **measure(ctx, out)))
    gen_rows = pd.DataFrame(rows)

    pairs = pd.read_csv(SEC / "pairs.csv")
    pairs = pairs[(pairs.split == "test") & pairs.context_file.str.removesuffix(".npy").isin(set(stems))]
    cs = pd.read_csv(args.pairs_measures).set_index("pair_id").cs
    rows = []
    for p in pairs.itertuples():
        ctx = np.load(str(SEC / p.context_file))
        tgt = np.load(str(SEC / p.target_file))
        ctx = ctx[np.argsort(ctx[:, 0], kind="stable")]
        tgt = tgt[np.argsort(tgt[:, 0], kind="stable")]
        span = ctx[:, 0].max() - ctx[:, 0].min()
        tgt = tgt[tgt[:, 0] - tgt[0, 0] <= span]
        rows.append(dict(source="real " + p.sim_bucket.split(":")[1], cs=cs.get(p.pair_id, np.nan),
                         piece=p.context_file[:5], unit=p.pair_id, **measure(ctx, tgt)))
    real = pd.DataFrame(rows)
    matched = real[(real.cs >= lo) & (real.cs <= hi)]

    lines = [f"M1 sim:high CS, all valid draws: middle 50% {lo:.3f} to {hi:.3f}; "
             "real targets cut at the context's length", ""]
    for src in ("M1 high", "M1 mid", "M1 low", "M4 no label"):
        lines.append(f"  {src:28s} {share(gen_rows[gen_rows.source == src], rng)}")
    for b in ("high", "mid", "low"):
        lines.append(f"  {'real ' + b + ' (per pair)':28s} {share(real[real.source == 'real ' + b], rng)}")
    lines.append(f"  {'real, CS matched to M1 high':28s} {share(matched, rng)}")
    lines.append("  " + diff(gen_rows[gen_rows.source == "M1 high"], matched, np.random.default_rng(1)))
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
