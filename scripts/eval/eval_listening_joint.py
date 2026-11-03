r"""
scripts/eval/eval_listening_joint.py

Does Melody predict listener ratings beyond the other measures that survive the
Holm correction? Exploratory.

On the 10 seconds heard, with the ratings and clip scores of
eval_listening_metrics.py:

  1. correlations of Melody, Density, Register and CS over the 60 clips;
  2. one mixed model with all four (each z-scored) plus the slot term, crossed
     random intercepts for listener, context and clip: each partial slope with
     a Wald 95% interval;
  3. likelihood-ratio test of adding Melody to the model with the other three
     (maximum likelihood fits, 1 degree of freedom);
  4. the contrast Melody minus CS in the model of step 2, with its Wald 95%
     interval;
  5. Melody and CS in one model without CS's own components, with all clips and
     without the clips that quote 8+ notes, and the per-trial Spearman of Figure 4
     without the trials that hold a quoting clip.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_listening_joint.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_listening_metrics import clip_scores, ratings  # noqa: E402
from eval_listening_round2 import copies  # noqa: E402

SLOT_BIAS = 7.86   # the slot term of the label-effect model, as in plot_listening_trials.py
TERMS = {"interval_3gram": "Melody", "density": "Density", "register": "Register", "cs": "CS"}


def fit(df: pd.DataFrame, formula: str, reml: bool):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, df.assign(one=1), groups="one", re_formula="0",
                           vc_formula={"p": "0 + C(participant_id)", "x": "0 + C(ctx_stem)",
                                       "k": "0 + C(clip)"}).fit(reml=reml)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    ap.add_argument("--test1", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/eval/listening_joint.txt")
    args = ap.parse_args()

    key2 = pd.read_csv(args.key)
    r = ratings(key2, Path(args.responses))
    sc = clip_scores(key2, Path(args.test1), args.seconds)
    df = r.merge(sc, on=["ctx_stem", "cond"])
    df["clip"] = df.ctx_stem + "|" + df.cond
    for m, name in TERMS.items():
        df[name] = (df[m] - df[m].mean()) / df[m].std()

    lines = [f"Listeners {df.participant_id.nunique()}, ratings {len(df)}, clips {df["clip"].nunique()}",
             "", "1. Correlations over the 60 clips"]
    lines.append(sc.rename(columns=TERMS)[list(TERMS.values())].corr().round(2).to_string())

    full = fit(df, "rating ~ Melody + Density + Register + CS + isA", reml=True)
    lines += ["", "2. Joint model (REML): rating points per SD, Wald 95% interval"]
    for t in TERMS.values():
        est, se, p = full.params[t], full.bse[t], full.pvalues[t]
        lines.append(f"  {t:9s} {est:+6.2f}  [{est - 1.96 * se:+6.2f}, {est + 1.96 * se:+6.2f}]  p {p:.4f}")

    without = fit(df, "rating ~ Density + Register + CS + isA", reml=False)
    with_m = fit(df, "rating ~ Melody + Density + Register + CS + isA", reml=False)
    lr = 2 * (with_m.llf - without.llf)
    lines += ["", "3. Adding Melody to Density + Register + CS (ML fits)",
              f"  LR {lr:.2f}, df 1, p {stats.chi2.sf(lr, 1):.5f}"]

    cov = full.cov_params()
    d = full.params["Melody"] - full.params["CS"]
    se = (cov.loc["Melody", "Melody"] + cov.loc["CS", "CS"] - 2 * cov.loc["Melody", "CS"]) ** 0.5
    lines += ["", "4. Melody minus CS in the joint model (REML), Wald 95% interval",
              f"  {d:+6.2f}  [{d - 1.96 * se:+6.2f}, {d + 1.96 * se:+6.2f}]  p {2 * stats.norm.sf(abs(d / se)):.4f}"]

    # 5. Melody and CS alone, all clips and without the quoting clips; the per-trial
    #    Spearman without trials holding a quoting clip
    q = copies(key2, Path(args.test1), args.seconds)
    quoting = set(q[q.longest >= 8].pipe(lambda x: x.ctx_stem + "|" + x.cond))
    lines += ["", f"5. Melody and CS alone (REML), Wald 95% interval; {len(quoting)} quoting clips"]
    for name, d in (("all clips", df), ("without quoting clips", df[~df["clip"].isin(quoting)])):
        two = fit(d, "rating ~ Melody + CS + isA", reml=True)
        cov = two.cov_params()
        diff = two.params["Melody"] - two.params["CS"]
        se = (cov.loc["Melody", "Melody"] + cov.loc["CS", "CS"] - 2 * cov.loc["Melody", "CS"]) ** 0.5
        lines.append(f"  {name:22s} " + "  ".join(
            f"{t} {two.params[t]:+.2f} [{two.params[t] - 1.96 * two.bse[t]:+.2f}, "
            f"{two.params[t] + 1.96 * two.bse[t]:+.2f}]" for t in ("Melody", "CS"))
            + f"  Melody - CS {diff:+.2f} [{diff - 1.96 * se:+.2f}, {diff + 1.96 * se:+.2f}]")
    w = (df.assign(r=df.rating - SLOT_BIAS * df.isA)
         .pivot_table(index=["participant_id", "ctx_stem"], columns="cond", values="r"))
    listen = (w["high"] - w["low"]).groupby("ctx_stem").mean()
    m = sc.pivot(index="ctx_stem", columns="cond", values="interval_3gram")
    gap = (m["high"] - m["low"]).reindex(listen.index)
    keep = ~listen.index.isin({c.split("|")[0] for c in quoting})
    for name, k in (("all trials", slice(None)), (f"without the {(~keep).sum()} quoting trials", keep)):
        rho = stats.spearmanr(gap[k], listen[k])
        lines.append(f"  per-trial Spearman, {name}: rho {rho.statistic:.2f}, p {rho.pvalue:.3f}")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
