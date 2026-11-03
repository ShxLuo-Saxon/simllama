r"""
scripts/eval/eval_composite_vs_single.py

Is compound similarity (CS) better than a single measure? Exploratory.

CS is compared, on the two human standards of the paper, with
  - its own four components (compound_parts), and
  - the single paper measures (symbolic_measures.py), Melody included.

Standards and the paired comparison:
  annotation   AUC of same-letter (AA) against different-letter (AB) pairs,
               all 3,304 pairs of pairs.csv; difference AUC(CS) - AUC(m)
               with a 95% interval from resampling whole pieces.
  listeners    Pearson r with the 60 clip mean ratings (slot bias removed,
               first 10 s heard); difference r(CS) - r(m) with a 95% interval
               from resampling the 30 contexts (both clips of a context
               together).

A positive difference whose interval excludes 0 means CS is better.

It also prints how often Melody and CS score a pair exactly zero, and their
tertile cuts: a label metric must grade every pair into three levels.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_composite_vs_single.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_listening_metrics import SLOT_FILE, cut, ratings  # noqa: E402
from similarity import compound_parts, midi_to_compound  # noqa: E402
from paper_common import NAMES  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
# code name -> paper name (the paper's naming table)
SINGLE = {k: v for k, v in NAMES.items() if k != "cs"}
PARTS = {"part_chroma": "CS-Chroma", "part_duration": "CS-Duration",
         "part_register": "CS-Register", "part_density": "CS-Density"}
SLOT_BIAS = 7.86


def parts_frame(rows_ab) -> pd.DataFrame:
    return pd.DataFrame([{f"part_{k}": v for k, v in compound_parts(a, b).items()}
                         for a, b in rows_ab])


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def annotation(measures_csv: str, n_boot: int, rng) -> pd.DataFrame:
    df = pd.read_csv(measures_csv)
    cache: dict[str, np.ndarray] = {}
    pairs = pd.read_csv(SEC / "pairs.csv").set_index("pair_id")

    def load(f):
        if f not in cache:
            t = np.load(str(SEC / f)).astype(float)
            cache[f] = t[np.argsort(t[:, 0], kind="stable")]
        return cache[f]

    ab = [(load(pairs.at[i, "context_file"]), load(pairs.at[i, "target_file"]))
          for i in df.pair_id]
    df = pd.concat([df.reset_index(drop=True), parts_frame(ab)], axis=1)
    pieces = df.piece_id.unique()
    groups = {k: g.index.to_numpy() for k, g in df.groupby("piece_id")}
    boot_idx = [np.concatenate([groups[k] for k in rng.choice(pieces, len(pieces))])
                for _ in range(n_boot)]
    same = df.same.to_numpy()

    def auc_of(col, idx=None):
        x = df[col].to_numpy() if idx is None else df[col].to_numpy()[idx]
        s = same if idx is None else same[idx]
        ok = ~np.isnan(x)
        return auc(x[ok & s], x[ok & ~s])

    # can a measure grade all pairs? share of exact zeros and the tertile cuts
    for col, name in (("interval_3gram", "Melody"), ("cs", "CS")):
        x = df[col]
        q = x.quantile([1 / 3, 0.5, 2 / 3]).round(3).tolist()
        print(f"{name}: zero in {(x == 0).mean():.3f} of pairs "
              f"(AA {(x[df.same] == 0).mean():.3f}, AB {(x[~df.same] == 0).mean():.3f}); "
              f"33rd / 50th / 67th percentile {q}")

    cs = auc_of("cs")
    cs_b = np.array([auc_of("cs", i) for i in boot_idx])
    rows = [dict(standard="annotation", measure="CS (ours)", value=cs, diff=np.nan,
                 lo=np.nan, hi=np.nan)]
    for col, name in {**PARTS, **SINGLE}.items():
        v = auc_of(col)
        d = cs_b - np.array([auc_of(col, i) for i in boot_idx])
        rows.append(dict(standard="annotation", measure=name, value=v, diff=cs - v,
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def listeners(key_path: str, resp: str, test1: str, seconds: float, listening_csv: str,
              n_boot: int, rng) -> pd.DataFrame:
    key = pd.read_csv(key_path)
    r = ratings(key, Path(resp))
    r["r_adj"] = r.rating - SLOT_BIAS * r.isA
    clip = r.groupby(["ctx_stem", "cond"]).r_adj.mean().rename("rating").reset_index()

    rows = []
    for k in key[key.kind == "test"].itertuples():
        sdir = Path(test1) / k.source_set
        hs = pd.read_csv(sdir / "answer_key.csv").set_index("trial").at[k.source_trial, "high_slot"]
        tdir = sdir / "trials"
        ctx = cut(midi_to_compound(tdir / f"{k.source_trial}_1_Context.mid"), seconds)
        for cond, slot in (("high", hs), ("low", "B" if hs == "A" else "A")):
            gen = cut(midi_to_compound(tdir / f"{k.source_trial}_{SLOT_FILE[slot]}.mid"), seconds)
            p = compound_parts(ctx.astype(float), gen.astype(float))
            rows.append(dict(ctx_stem=k.ctx_stem, cond=cond, **{f"part_{a}": b for a, b in p.items()}))
    parts = pd.DataFrame(rows)

    # single measures and CS as scored by eval_listening_metrics.py are recomputed
    # here with the same functions, so the two scripts cannot drift apart
    from eval_listening_metrics import clip_scores
    sc = clip_scores(key, Path(test1), seconds)
    d = clip.merge(sc, on=["ctx_stem", "cond"]).merge(parts, on=["ctx_stem", "cond"])
    ctxs = d.ctx_stem.unique()
    groups = {c: g.index.to_numpy() for c, g in d.groupby("ctx_stem")}
    boot_idx = [np.concatenate([groups[c] for c in rng.choice(ctxs, len(ctxs))])
                for _ in range(n_boot)]

    def r_of(col, idx=None):
        x = d[col].to_numpy() if idx is None else d[col].to_numpy()[idx]
        y = d.rating.to_numpy() if idx is None else d.rating.to_numpy()[idx]
        ok = ~np.isnan(x)
        if np.std(x[ok]) == 0:
            return np.nan
        return stats.pearsonr(x[ok], y[ok]).statistic

    cs = r_of("cs")
    cs_b = np.array([r_of("cs", i) for i in boot_idx])
    out = [dict(standard="listeners", measure="CS (ours)", value=cs, diff=np.nan,
                lo=np.nan, hi=np.nan)]
    for col, name in {**PARTS, **SINGLE}.items():
        v = r_of(col)
        dd = cs_b - np.array([r_of(col, i) for i in boot_idx])
        out.append(dict(standard="listeners", measure=name, value=v, diff=cs - v,
                        lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5)))
    ref = pd.read_csv(listening_csv)
    ref = ref[ref.window.str.startswith("first")].set_index("measure").clip_r
    assert abs(ref["cs"] - cs) < 1e-9, (ref["cs"], cs)
    return pd.DataFrame(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_measures", default="outputs/eval/ab_label_measures.csv")
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    ap.add_argument("--test1", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--listening_csv", default="outputs/eval/listening_metrics.csv")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out_csv", default="outputs/eval/composite_vs_single.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    res = pd.concat([
        annotation(args.pairs_measures, args.boot, rng),
        listeners(args.key, args.responses, args.test1, args.seconds,
                  args.listening_csv, args.boot, rng),
    ])
    res.to_csv(args.out_csv, index=False)
    for std, label in (("annotation", "AUC, AA above AB (3,304 pairs)"),
                       ("listeners", "r with 60 clip mean ratings (first 10 s)")):
        t = res[res.standard == std].set_index("measure")
        print(f"\n=== {label}; diff = CS minus measure, 95% interval ===")
        print(t[["value", "diff", "lo", "hi"]].round(3).to_string())
        better = t[(t.lo > 0)].index.tolist()
        worse = t[(t.hi < 0)].index.tolist()
        print(f"CS better (interval > 0): {better}")
        print(f"CS worse  (interval < 0): {worse}")
    print(f"\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
