r"""
scripts/eval/eval_mismatched_context.py

Does the label bring outputs closer to their own context, or only make them
generically more "typical"?
For each test context, K foreign contexts are drawn from other test pieces (fixed
seed). Every output of every draw is scored against its own context and against
each foreign context on the main measures. Per context and measure:

  own gap      mean over draws of m(own, high) - m(own, low)
  foreign gap  the same against the foreign contexts, averaged over them
  difference   own gap - foreign gap: the part of the label effect that is
               specific to the context the output was generated from

Intervals and one-sided p (difference <= 0) come from 2,000 resamples of whole
pieces; Holm over the five measures. Draw validity follows eval_robustness_table.py
(any output under 10 notes drops the draw); no quality filter.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_mismatched_context.py [--k 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import MAIN, cluster_boot, holm, valid_draws  # noqa: E402
from eval_robustness_table import PUB, RES, SEC  # noqa: E402
from similarity import compound_similarity, midi_to_compound  # noqa: E402
from symbolic_measures import MEASURES  # noqa: E402


def score(ctx: np.ndarray, g: np.ndarray) -> dict:
    out = {m: MEASURES[m](ctx, g) for m in MAIN if m != "cs"}
    out["cs"] = compound_similarity(ctx.astype(float), g.astype(float))
    return out


def outputs(dirs) -> list[tuple[str, int, str, Path]]:
    found = []
    for d, pat, fixed in dirs:
        for f in sorted(Path(d).glob("*_generated.mid")):
            m = pat.match(f.name)
            if m:
                found.append((m.group(1), fixed if fixed is not None else int(m.group(3)),
                              m.group(2), f))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published_dir", default="outputs/generated/m1_triplet")
    ap.add_argument("--resample_dir", default="outputs/generated/m1_resample")
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/mismatched_context")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    own = pd.read_csv(args.scores)
    own = valid_draws(own)
    valid = set(zip(own.ctx_stem, own["sample"]))
    stems = sorted(own.ctx_stem.unique())
    ctx = {s: np.load(str(SEC / f"{s}.npy")) for s in stems}
    foreign = {s: list(rng.choice([t for t in stems if t[:5] != s[:5]], args.k, replace=False))
               for s in stems}

    rows = []
    for stem, sample, bucket, f in outputs([(args.published_dir, PUB, 0),
                                            (args.resample_dir, RES, None)]):
        if (stem, sample) not in valid:
            continue
        g = midi_to_compound(f)
        g = g.copy()
        g[:, 0] -= g[0, 0]
        fs = [score(ctx[t], g) for t in foreign[stem]]
        rows.append(dict(ctx_stem=stem, sample=sample, bucket=bucket,
                         **{m: np.mean([x[m] for x in fs]) for m in MAIN}))
    fdf = pd.DataFrame(rows)
    fdf.to_csv(args.out + "_scores.csv", index=False)

    res = []
    for m, name in MAIN.items():
        def gap(df):
            w = df.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
            return (w["high"] - w["low"]).groupby("ctx_stem").mean()
        go, gf = gap(own), gap(fdf)
        d = (go - gf).dropna()
        means, _ = cluster_boot(d, rng, args.boot)
        res.append(dict(measure=name, contexts=len(d), own_gap=go.mean(), foreign_gap=gf.mean(),
                        difference=d.mean(), lo=np.percentile(means, 2.5),
                        hi=np.percentile(means, 97.5),
                        p_piece=((means <= 0).sum() + 1) / (args.boot + 1)))
    out = pd.DataFrame(res)
    out["p_holm"] = holm(out.p_piece.to_numpy())
    out.to_csv(args.out + ".csv", index=False)
    text = out.round(4).to_string(index=False)
    Path(args.out + ".txt").write_text(f"K = {args.k} foreign contexts per context\n{text}\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
