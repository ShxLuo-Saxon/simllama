r"""
scripts/eval/eval_listening_metrics.py

Which similarity measure best matches what listeners hear?

Uses the listening-test ratings as ground truth for perceived similarity, and
scores every measure in symbolic_measures.py against them:

  - the four compound similarity components on their own (count chroma, fixed
    duration bins);
  - compound similarity (both the duration-weighted and the count-chroma form),
    and variants of the count-chroma form: each component left out, equal weights,
    the duration bins rescaled to our tick rate;
  - order-sensitive measures (melodic contour, intervals, rhythm), which
    compound similarity lacks.

Each measure is computed on what listeners actually heard: the context and the
generated clip both cut to the first `--seconds` seconds (10 for Test 2). The
full-length version is also reported, since compound_sim was published on it.

For each measure it reports three things, all with the slot bias removed:

  model      rating ~ measure (z-scored) + slot, with participant, context and
             clip as random intercepts -- the Q2 model of evaluation_test2.md.
             Slope = rating points per 1 SD of the measure, with a Wald 95%
             interval (slope +- 1.96 SE).
  clips      correlation of the 60 clip mean ratings with the measure.
  trials     in how many of the 30 trials the measure and the listeners pick the
             same clip as more similar.

This is exploratory: many measures are compared on one listening test, so
p-values are also given after Holm correction.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_listening_metrics.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import midi_to_compound  # noqa: E402
from score_listening_ratings import is_survey, parse_survey, parse_wide  # noqa: E402
from similarity import compound_similarity  # noqa: E402
from symbolic_measures import all_measures  # noqa: E402
from paper_common import holm  # noqa: E402,F401

TICKS_PER_SEC = 100
SLOT_FILE = {"A": "2_A", "B": "3_B"}


def cut(t: np.ndarray, seconds: float | None, tail: float = 1.0) -> np.ndarray:
    """Keep notes starting before `seconds`; clamp their ends to seconds + tail,
    as make_short_stimuli.truncate_midi does for the audio."""
    if seconds is None or len(t) == 0:
        return t
    t = t[t[:, 0] < seconds * TICKS_PER_SEC].copy()
    end = (seconds + tail) * TICKS_PER_SEC
    t[:, 1] = np.minimum(t[:, 1], np.maximum(1, end - t[:, 0]))
    return t


def ratings(key: pd.DataFrame, resp: Path, keep_failed: bool = False) -> pd.DataFrame:
    """Long frame of valid ratings: participant, ctx_stem, cond, isA, rating.
    keep_failed keeps listeners who failed a check (sensitivity)."""
    files = sorted(resp.glob("*.csv")) if resp.is_dir() else [resp]
    long = pd.concat([parse_survey(f) if is_survey(f) else parse_wide(f)
                      for f in files], ignore_index=True)
    w = long.pivot_table(index=["participant_id", "trial"], columns="slot",
                         values="rating").reset_index()
    d = key.merge(w, on="trial")
    d["diff"] = np.where(d.high_slot == "A", d.A - d.B, d.B - d.A)
    bad = [] if keep_failed else d[(d.kind == "check") & (d["diff"] <= 0)].participant_id.unique()
    t = d[(d.kind == "test") & ~d.participant_id.isin(bad)]
    rows = []
    for r in t.itertuples():
        for cond, slot in (("high", r.high_slot), ("low", "B" if r.high_slot == "A" else "A")):
            rows.append(dict(participant_id=r.participant_id, ctx_stem=r.ctx_stem,
                             trial=r.trial, cond=cond, isA=float(slot == "A"),
                             rating=getattr(r, slot)))
    return pd.DataFrame(rows)


def clip_scores(key2: pd.DataFrame, test1: Path, seconds: float | None) -> pd.DataFrame:
    """Every measure for every clip, from the Test 1 MIDI it was rendered from."""
    rows = []
    for r in key2[key2.kind == "test"].itertuples():
        sdir = test1 / r.source_set
        k1 = pd.read_csv(sdir / "answer_key.csv").set_index("trial")
        hs = k1.at[r.source_trial, "high_slot"]
        tdir = sdir / "trials"
        ctx = cut(midi_to_compound(tdir / f"{r.source_trial}_1_Context.mid"), seconds)
        for cond, slot in (("high", hs), ("low", "B" if hs == "A" else "A")):
            gen = cut(midi_to_compound(tdir / f"{r.source_trial}_{SLOT_FILE[slot]}.mid"),
                      seconds)
            rows.append(dict(ctx_stem=r.ctx_stem, cond=cond, **all_measures(ctx, gen),
                             cs=compound_similarity(ctx.astype(float), gen.astype(float))))
    return pd.DataFrame(rows)


def crossed(df: pd.DataFrame, formula: str, term: str):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, df.assign(one=1), groups="one", re_formula="0",
                        vc_formula={"p": "0 + C(participant_id)",
                                    "x": "0 + C(ctx_stem)",
                                    "k": "0 + C(clip)"}).fit(reml=True)
    return m.params[term], m.bse[term], m.pvalues[term]


def evaluate(r: pd.DataFrame, sc: pd.DataFrame, slot_bias: float) -> pd.DataFrame:
    measures = [c for c in sc.columns if c not in ("ctx_stem", "cond")]
    df = r.merge(sc, on=["ctx_stem", "cond"])
    df["clip"] = df.ctx_stem + "|" + df.cond
    df["r_adj"] = df.rating - slot_bias * df.isA
    clip_mean = df.groupby("clip").agg(r=("r_adj", "mean"), ctx=("ctx_stem", "first"),
                                       cond=("cond", "first"))
    trial = df.groupby(["ctx_stem", "cond"]).r_adj.mean().unstack()
    listen_gap = trial["high"] - trial["low"]
    out = []
    for m in measures:
        sd = df[m].std()
        if sd == 0 or np.isnan(sd):
            continue
        df["z"] = (df[m] - df[m].mean()) / sd
        est, se, p = crossed(df, "rating ~ z + isA", "z")
        cm = clip_mean.join(sc.assign(clip=sc.ctx_stem + "|" + sc.cond)
                            .set_index("clip")[m])
        pr = stats.pearsonr(cm[m], cm.r)
        g = sc.pivot(index="ctx_stem", columns="cond", values=m)
        mg = (g["high"] - g["low"]).reindex(listen_gap.index)
        agree = int((np.sign(mg) == np.sign(listen_gap)).sum())
        rs = stats.spearmanr(mg, listen_gap)
        out.append(dict(measure=m, slope_per_sd=est, se=se,
                        ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se, p=p,
                        clip_r=pr.statistic, clip_p=pr.pvalue,
                        trials_agree=agree, trial_rho=rs.statistic))
    res = pd.DataFrame(out).set_index("measure")
    res["p_holm"] = holm(res.p)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    ap.add_argument("--test1", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="Clip length listeners heard")
    ap.add_argument("--slot_bias", type=float, default=7.86,
                    help="A-over-B rating bias from score_listening_ratings.py")
    ap.add_argument("--out_csv", default="outputs/eval/listening_metrics.csv")
    ap.add_argument("--keep_failed", action="store_true",
                    help="Keep listeners who failed a check (sensitivity, review round 8)")
    args = ap.parse_args()

    key2 = pd.read_csv(args.key)
    r = ratings(key2, Path(args.responses), args.keep_failed)
    print(f"Valid listeners: {r.participant_id.nunique()}  ratings: {len(r)}  "
          f"clips: {r.groupby(['ctx_stem', 'cond']).ngroups}")

    tables = []
    for secs, label in ((args.seconds, f"first {args.seconds:g} s (heard)"),
                        (None, "full length (as published)")):
        sc = clip_scores(key2, Path(args.test1), secs)
        res = evaluate(r, sc, args.slot_bias)
        tables.append(res.assign(window=label))
        print("\n" + "=" * 92)
        print(f"MEASURES vs LISTENER RATINGS -- {label}")
        print("=" * 92)
        print("  slope_per_sd: rating points per 1 SD of the measure (mixed model, "
              "clip counted once)")
        print("  clip_r: correlation over 60 clip means; trials_agree: same clip "
              "picked, of 30\n")
        cols = ["slope_per_sd", "ci_lo", "ci_hi", "p", "p_holm", "clip_r", "clip_p",
                "trials_agree", "trial_rho"]
        print(res.sort_values("slope_per_sd", ascending=False)[cols]
              .round(3).to_string())

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
