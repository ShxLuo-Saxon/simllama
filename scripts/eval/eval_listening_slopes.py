r"""
scripts/eval/eval_listening_slopes.py

Robustness check for the measure-agreement models of the listening test: does
a measure's effect on the ratings survive a by-listener random slope?

The model of eval_listening_metrics.py is

    rating ~ z(measure) + slot, random intercepts for listener, context and clip.

This script adds a random slope of z for each listener, as a separate variance
component (uncorrelated with the intercept), and reports the fixed slope with its
Wald interval, p and Holm p over the 12 measures, beside the intercept-only model.
The label-effect model needs no such check: it models the high-minus-low gap of
each trial, so its random listener intercept is already a by-listener label slope.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_listening_slopes.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_listening_metrics import clip_scores, crossed, holm, ratings  # noqa: E402
from paper_common import NAMES  # noqa: E402

KEY = "listening_test/answer_key.csv"
RESP = "listening_test/ratings.csv"
TEST1 = "listening_test/stimuli/test1_two_sets_full"
OUT = Path("outputs/eval/listening_slopes.txt")
MEASURES = NAMES


def with_slope(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = smf.mixedlm("rating ~ z + isA", df.assign(one=1), groups="one", re_formula="0",
                        vc_formula={"p": "0 + C(participant_id)",
                                    "ps": "0 + C(participant_id):z",
                                    "x": "0 + C(ctx_stem)",
                                    "k": "0 + C(clip)"}).fit(reml=True)
    conv = m.converged and not any("converge" in str(x.message).lower() for x in w)
    return m.params["z"], m.bse["z"], m.pvalues["z"], conv


def main() -> None:
    key2 = pd.read_csv(KEY)
    r = ratings(key2, Path(RESP))
    sc = clip_scores(key2, Path(TEST1), 10.0)
    df = r.merge(sc, on=["ctx_stem", "cond"])
    df["clip"] = df.ctx_stem + "|" + df.cond
    rows = []
    for col, name in MEASURES.items():
        df["z"] = (df[col] - df[col].mean()) / df[col].std()
        b0, s0, p0 = crossed(df, "rating ~ z + isA", "z")
        b1, s1, p1, conv = with_slope(df)
        rows.append(dict(measure=name, slope=b0, p=p0, slope_rs=b1, lo_rs=b1 - 1.96 * s1,
                         hi_rs=b1 + 1.96 * s1, p_rs=p1, converged=conv))
    res = pd.DataFrame(rows).set_index("measure")
    res["holm"] = holm(res.p)
    res["holm_rs"] = holm(res.p_rs)
    text = (f"{r.participant_id.nunique()} listeners, {df["clip"].nunique()} clips, first 10 s\n"
            "slope: intercepts only; slope_rs: with a by-listener random slope\n\n"
            + res.round(4).to_string() + "\n")
    print(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
