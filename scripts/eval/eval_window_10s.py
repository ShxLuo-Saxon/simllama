r"""
scripts/eval/eval_window_10s.py

Does the objective label effect hold on the first 10 seconds, the length the
listening test played?
Every context and every output of every valid draw is cut to its first 10 seconds
as in eval_listening_metrics.cut, scored on the main measures, and the high-low
gap is tested as in eval_effect_sizes.py (per-context mean over draws; piece-level
bootstrap; Holm over the five measures).

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_window_10s.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import cluster_boot, holm, valid_draws  # noqa: E402
from eval_listening_metrics import cut  # noqa: E402
from eval_mismatched_context import MAIN, outputs, score  # noqa: E402
from eval_robustness_table import PUB, RES, SEC  # noqa: E402
from similarity import midi_to_compound  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published_dir", default="outputs/generated/m1_triplet")
    ap.add_argument("--resample_dir", default="outputs/generated/m1_resample")
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/eval/window_10s")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    own = pd.read_csv(args.scores)
    own = valid_draws(own)
    valid = set(zip(own.ctx_stem, own["sample"]))
    ctx = {}
    rows = []
    for stem, sample, bucket, f in outputs([(args.published_dir, PUB, 0), (args.resample_dir, RES, None)]):
        if (stem, sample) not in valid:
            continue
        if stem not in ctx:
            c = np.load(str(SEC / f"{stem}.npy")).astype(float)
            c = c[np.argsort(c[:, 0], kind="stable")]
            c[:, 0] -= c[0, 0]
            ctx[stem] = cut(c, args.seconds)
        g = midi_to_compound(f).astype(float)
        g[:, 0] -= g[0, 0]
        g = cut(g, args.seconds)
        if len(g) < 2 or len(ctx[stem]) < 2:
            continue
        rows.append(dict(ctx_stem=stem, sample=sample, bucket=bucket, **score(ctx[stem], g)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out + "_scores.csv", index=False)

    res = []
    for m, name in MAIN.items():
        w = df.pivot_table(index=["ctx_stem", "sample"], columns="bucket", values=m)
        gap = (w["high"] - w["low"]).dropna().groupby("ctx_stem").mean()
        means, shares = cluster_boot(gap, rng, args.boot)
        nz = gap[gap.abs() > 1e-12]
        res.append(dict(measure=name, contexts=len(gap), share=(nz > 0).mean(),
                        mean_gap=gap.mean(), lo=np.percentile(means, 2.5), hi=np.percentile(means, 97.5),
                        p_piece=((means <= 0).sum() + 1) / (args.boot + 1)))
    out = pd.DataFrame(res)
    out["p_holm"] = holm(out.p_piece.to_numpy())
    out.to_csv(args.out + ".csv", index=False)
    text = f"First {args.seconds:g} s of context and outputs, high - low\n" + out.round(4).to_string(index=False) + "\n"
    Path(args.out + ".txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
