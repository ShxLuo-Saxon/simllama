r"""
scripts/eval/eval_ab_labels.py

How well does each similarity measure separate same-label section pairs (AA)
from different-label pairs (AB)?

The section letters (A, B, ...) come from the AI annotation, checked by a human;
a second annotator's labels are pending. Two sections of one piece with the same
letter are, by the annotation, versions of the same musical material. A measure
that tracks musical similarity should score AA pairs higher.

Two compound metrics exist in this repo, and both are scored:

  cs       compound similarity -- built the training buckets
                (pairs.csv column `compound_ks`) and the published AA/AB
                check. Duration-weighted chroma, per-section log duration bins.
                The paper's "compound similarity".
  cs_v1_1       a variant of compound similarity that selected the
                listening-test stimuli. Count chroma, fixed duration bins.
                Not used in the paper.

Reported per measure
--------------------
  AUC       chance that a random AA pair outscores a random AB pair (0.5 = no
            separation, 1 = perfect); 95% CI from resampling whole pieces
  d         Cohen's d, the gap between AA and AB means in pooled SDs
  paired    within one context with both an AA and an AB target, how often the
            AA target scores higher -- removes differences between pieces

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_ab_labels.py
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
from symbolic_measures import all_measures  # noqa: E402
from similarity import compound_similarity  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
LETTER = re.compile(r"_s\d+_([A-Za-z]+)_")


def load(path: Path) -> np.ndarray:
    t = np.load(str(path)).astype(np.int64)
    return t[np.argsort(t[:, 0], kind="stable")]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    u = stats.mannwhitneyu(pos, neg).statistic
    return u / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections_dir", default=str(SEC))
    ap.add_argument("--pairs_csv", default=None, help="default: <sections_dir>/pairs.csv")
    ap.add_argument("--out_csv", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    sec = Path(args.sections_dir)
    pairs = pd.read_csv(args.pairs_csv or sec / "pairs.csv")
    pairs["la"] = pairs.context_file.str.extract(LETTER)[0]
    pairs["lb"] = pairs.target_file.str.extract(LETTER)[0]
    pairs = pairs.dropna(subset=["la", "lb"])
    pairs["same"] = pairs.la == pairs.lb
    print(f"Pairs: {len(pairs)}  AA {pairs.same.sum()}  AB {(~pairs.same).sum()}  "
          f"pieces {pairs.piece_id.nunique()}")

    cache: dict[str, np.ndarray] = {}
    rows = []
    for i, p in enumerate(pairs.itertuples()):
        for f in (p.context_file, p.target_file):
            if f not in cache:
                cache[f] = load(sec / f)
        a, b = cache[p.context_file], cache[p.target_file]
        m = all_measures(a, b)
        m["cs"] = compound_similarity(a.astype(float), b.astype(float))
        rows.append(dict(pair_id=p.pair_id, piece_id=p.piece_id,
                         context_file=p.context_file, same=p.same, **m))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pairs)}")
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    check = pairs.set_index("pair_id").compound_ks
    diff = (df.set_index("pair_id").cs - check).abs().max()
    print(f"cs recomputed vs pairs file (compound_ks column): max abs difference {diff:.2e}")

    rng = np.random.default_rng(0)
    pieces = df.piece_id.unique()
    by_piece = {k: g for k, g in df.groupby("piece_id")}
    measures = [c for c in df.columns
                if c not in ("pair_id", "piece_id", "context_file", "same")]
    res = []
    for m in measures:
        x = df[[m, "same", "context_file"]].dropna()
        pos, neg = x[x.same][m].to_numpy(), x[~x.same][m].to_numpy()
        a = auc(pos, neg)
        boots = []
        for _ in range(args.boot):
            s = pd.concat([by_piece[k] for k in rng.choice(pieces, len(pieces))])
            s = s[[m, "same"]].dropna()
            sp, sn = s[s.same][m].to_numpy(), s[~s.same][m].to_numpy()
            if len(sp) and len(sn):
                boots.append(auc(sp, sn))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sd = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2)
        wins = []
        for _, g in x.groupby("context_file"):
            gp, gn = g[g.same][m].to_numpy(), g[~g.same][m].to_numpy()
            if len(gp) and len(gn):
                wins.append(np.mean([(u > v) + 0.5 * (u == v) for u in gp for v in gn]))
        res.append(dict(measure=m, auc=a, auc_lo=lo, auc_hi=hi,
                        d=(pos.mean() - neg.mean()) / sd,
                        aa_mean=pos.mean(), ab_mean=neg.mean(),
                        paired=np.mean(wins), n_contexts=len(wins)))
    res = pd.DataFrame(res).set_index("measure").sort_values("auc", ascending=False)
    print("\nAA vs AB separation (AUC 0.5 = none, 1 = perfect; CI resamples pieces)")
    print(res.round(3).to_string())
    res.to_csv(out.with_name(out.stem + "_summary.csv"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
