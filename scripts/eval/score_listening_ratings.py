r"""
scripts/eval/score_listening_ratings.py

Score the **continuous-rating** listening test (listening_test/protocol.md).
Replaces an earlier script for the old binary schema (`answer_A_or_B` +
`similarity_1to3`).

Each trial yields TWO numbers: `rating_A` and `rating_B`, each 0-100. Everything
below is derived from those.

    diff        = rating_high - rating_low     signed, the Q1 outcome
    chose_high  = diff > 0                     the recovered 2AFC outcome
    rating      = each section's own rating    the Q2 outcome, 2 rows per trial
    cs          = compound_sim(ctx, section)   what the rating is tested against

Reported
--------
  attention checks  pass/fail per participant (checks NEVER enter the analysis)
  Q1  Wilcoxon signed-rank on `diff`, positive-delta trials  -- PRIMARY
  Q2  rating vs cs, linear mixed model where possible        -- PRIMARY
  ties                                                       -- protocol 7.0
  derived 2AFC rate vs 0.50                                  -- secondary
  negative-delta trials, reported separately                 -- probe

Input format
------------
webMUSHRA / Go Listen wide export: a header row of stimulus filenames
(`TrialNN_2_A.mp3`), one row per participant of ratings. Filenames carry the
trial and slot, so the answer key supplies everything else.

Test 2 uses a questionnaire platform instead. Its export has one header row of
question titles (`<N>.<question text>:<...>A` / `...B`) and no filenames, so
question N is read as `TrialNN` and the A/B suffix as the slot. The stimuli must
have been uploaded in that order. A row is one participant only if it carries a
answers every question; other rows (the researcher's own test runs) are skipped
and counted. A complete row without a usable ID is kept as `NOID_r<line>`.

Usage:
    .venv\Scripts\python.exe scripts\eval\score_listening_ratings.py `
        --key       listening_test\stimuli\test2_one_set_10s\answer_key.csv `
        --responses listening_test\responses\test2\
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FILE_RE = re.compile(r"(Trial\d+)_\d_([AB])\.(?:mp3|wav|ogg|flac)", re.I)
SURVEY_RE = re.compile(r"^(\d+)\..*([AB])$")      # "12.<question>:<...>A"
SURVEY_ID_RE = re.compile(r"^\d+\..*ID$", re.I)   # "33.<...>ID"


def is_survey(path: Path) -> bool:
    """True if the first row is question titles, not stimulus filenames."""
    head = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    return sum(bool(SURVEY_RE.match(str(c))) for c in head) >= 2


def parse_survey(path: Path) -> pd.DataFrame:
    """Questionnaire export -> long frame: participant, trial, slot, rating."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    cols = {}
    for c in raw.columns:
        m = SURVEY_RE.match(str(c))
        if m and not SURVEY_ID_RE.match(str(c)):
            cols[c] = (f"Trial{int(m.group(1)):02d}", m.group(2))
    id_cols = [c for c in raw.columns if SURVEY_ID_RE.match(str(c)) or c == "listener"]  # raw or anonymised
    if not cols or not id_cols:
        raise SystemExit(f"No rating or participant-ID columns in {path.name}")

    out, skipped = [], 0
    for i, row in raw.iterrows():
        vals = pd.to_numeric(row[list(cols)], errors="coerce")
        if vals.isna().any():
            skipped += 1                    # test run, or an unfinished session
            continue
        # Typed IDs vary: "p22", "p<invisible space>10", or a word for "none".
        # Keep letters and digits; a complete row with no usable ID still counts.
        pid = re.sub(r"[^A-Za-z0-9]", "", str(row[id_cols[0]])).upper()
        if pid in ("", "NAN"):
            pid = f"NOID_r{i + 2}"          # the row's line in the CSV
            print(f"  complete row without an ID: {pid}")
        for c, (trial, slot) in cols.items():
            out.append({"participant_id": pid, "trial": trial,
                        "slot": slot, "rating": float(vals[c])})
    print(f"{path.name}: {len(raw) - skipped} participant row(s), "
          f"{skipped} skipped (no ID or incomplete)")
    return pd.DataFrame(out, columns=["participant_id", "trial", "slot", "rating"])


def parse_wide(path: Path) -> pd.DataFrame:
    """Wide export -> long frame: participant, trial, slot, rating."""
    raw = pd.read_csv(path, header=None)

    # Locate the row carrying stimulus filenames, and the columns they occupy.
    file_row = None
    for i in range(min(6, len(raw))):
        if any(FILE_RE.search(str(v)) for v in raw.iloc[i]):
            file_row = i
            break
    if file_row is None:
        raise SystemExit(f"No stimulus filenames found in {path.name}")

    cols = {}
    for c in raw.columns:
        m = FILE_RE.search(str(raw.iloc[file_row, c]))
        if m:
            cols[c] = (m.group(1), m.group(2).upper())

    # Every later row that has numbers in those columns is one participant.
    out = []
    for i in range(file_row + 1, len(raw)):
        vals = pd.to_numeric(raw.iloc[i, list(cols)], errors="coerce")
        if vals.notna().sum() < len(cols) * 0.5:
            continue                                    # header or blank row
        pid = str(raw.iloc[i, 0]).strip()
        if pid in ("nan", "", "Name"):
            pid = f"{path.stem[:12]}_r{i}"
        for c, (trial, slot) in cols.items():
            v = pd.to_numeric(raw.iloc[i, c], errors="coerce")
            if pd.notna(v):
                out.append({"participant_id": pid, "trial": trial,
                            "slot": slot, "rating": float(v)})
    if not out:
        raise SystemExit(f"No participant rows parsed from {path.name}")
    return pd.DataFrame(out)


def crossed_lmm(formula: str, df: pd.DataFrame, term: str, clip: bool = False):
    """Fit `formula` + (1|participant) + (1|context) [+ (1|clip)]; return
    estimate, SE, p.

    statsmodels has no crossed-effects syntax, so the intercepts enter as
    variance components inside one all-covering group.

    `clip=True` adds one intercept per audio clip (context x condition). Use it
    for any model of single-clip ratings: every listener rates the same clip,
    and `cs` is a property of the clip, so without it the 20 ratings of one clip
    count as 20 independent pieces of evidence about `cs` and the p-value is
    far too small.    """
    import statsmodels.formula.api as smf
    vc = {"participant": "0 + C(participant_id)", "context": "0 + C(ctx_stem)"}
    if clip:
        vc["clip"] = "0 + C(clip)"
    m = smf.mixedlm(formula, df.assign(one=1), groups="one", re_formula="0",
                    vc_formula=vc).fit(reml=True)
    return m.params[term], m.bse[term], m.pvalues[term]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--responses", required=True,
                   help="Response CSV, a glob, or a directory")
    p.add_argument("--keep_failed", action="store_true",
                   help="Keep listeners who failed a check (sensitivity, review round 8)")
    args = p.parse_args()

    key = pd.read_csv(args.key)

    src = Path(args.responses)
    files = ([Path(f) for f in glob.glob(args.responses)] if "*" in args.responses
             else sorted(src.glob("*.csv")) if src.is_dir() else [src])
    files = [f for f in files if "answer_key" not in f.name
             and "response_sheet" not in f.name]
    if not files:
        raise SystemExit(f"No response files matched {args.responses}")
    long = pd.concat([parse_survey(f) if is_survey(f) else parse_wide(f)
                      for f in files], ignore_index=True)
    if long.empty:
        raise SystemExit("No participant rows parsed.")
    dup = long.groupby(["participant_id", "trial", "slot"]).size()
    if (dup > 1).any():
        ids = sorted(dup[dup > 1].index.get_level_values(0).unique())
        raise SystemExit(f"Participant ID used more than once: {ids}")

    # Wide per trial: rating_A / rating_B, then map slots onto conditions.
    w = long.pivot_table(index=["participant_id", "trial"], columns="slot",
                         values="rating").reset_index()
    w.columns.name = None
    w = w.rename(columns={"A": "rating_A", "B": "rating_B"})
    d = key.merge(w, on="trial", how="inner")

    d["rating_high"] = np.where(d.high_slot == "A", d.rating_A, d.rating_B)
    d["rating_low"] = np.where(d.high_slot == "A", d.rating_B, d.rating_A)
    d["diff"] = d.rating_high - d.rating_low
    d["chose_high"] = d["diff"] > 0

    participants = sorted(d.participant_id.unique())
    print(f"Participants: {len(participants)}  ({', '.join(participants)})")
    print(f"Trials per participant: {d.groupby('participant_id').size().unique()}")

    # ── Attention checks: screening only ──────────────────────────
    print("\n" + "=" * 62)
    print("ATTENTION CHECKS  (screening only -- excluded from all analysis)")
    print("=" * 62)
    chk = d[d.kind == "check"]
    excluded: list[str] = []
    for pid in participants:
        sub = chk[chk.participant_id == pid]
        npass = int(sub.chose_high.sum())
        test = d[(d.participant_id == pid) & (d.kind == "test")]
        flags = []
        if npass < len(sub):
            flags.append("FAILED CHECK")
        if test.rating_A.nunique() <= 1 and len(test) > 1:
            flags.append("ALL-SAME RESPONSES")
        if flags:
            excluded.append(pid)
        gaps = [f"{v:+.0f}" for v in sub["diff"]]
        print(f"  {pid:24s} {npass}/{len(sub)} passed   margins {gaps}"
              f"{'   <-- ' + ', '.join(flags) if flags else ''}")
    if excluded:
        print(f"\n  Excluded by protocol: {excluded}")
    if args.keep_failed:
        print("  --keep_failed: every listener is kept (sensitivity)")
        excluded = []

    t = d[(d.kind == "test") & (~d.participant_id.isin(excluded))].copy()
    if t.empty:
        raise SystemExit("No analysable trials remain.")

    # ── Ties (protocol 7.0) ───────────────────────────────────────
    n_tie = int((t["diff"] == 0).sum())
    print("\n" + "=" * 62)
    print(f"TIES  {n_tie}/{len(t)} = {n_tie/len(t):.1%}  "
          f"(equal ratings = no audible difference)")
    print("=" * 62)
    if n_tie / len(t) > 0.25:
        print("  >25% -- report as a primary result: the difference is largely")
        print("  imperceptible at this effect size (protocol 7.0).")
    else:
        print("  Dropped from the signed-rank and the derived 2AFC rate,")
        print("  per the standard treatment. Not split between conditions.")

    # ── Q1 PRIMARY: Wilcoxon signed-rank on diff, positive-delta ──
    print("\n" + "=" * 62)
    print("Q1  Does the MODEL do its job?   rating_high - rating_low  vs 0")
    print("    (positive-delta trials only -- negative reported separately)")
    print("=" * 62)
    tpos = t[t.delta > 0] if (t.delta > 0).any() else t
    nz = tpos[tpos["diff"] != 0]
    print(f"  n = {len(tpos)} trials ({len(nz)} non-tied)")
    print(f"  mean diff   : {tpos['diff'].mean():+.1f} rating points")
    print(f"  median diff : {tpos['diff'].median():+.1f}")
    if len(nz) >= 5:
        w_ = stats.wilcoxon(nz["diff"], alternative="greater")
        print(f"  Wilcoxon signed-rank : p = {w_.pvalue:.4f} (one-sided)"
              f"{'  *' if w_.pvalue < 0.05 else ''}")
        tt = stats.ttest_1samp(tpos["diff"], 0, alternative="greater")
        print(f"  paired t-test (robustness): p = {tt.pvalue:.4f}")
        # CI on the mean difference
        se = tpos["diff"].std(ddof=1) / np.sqrt(len(tpos))
        lo = tpos["diff"].mean() - 1.96 * se
        hi = tpos["diff"].mean() + 1.96 * se
        print(f"  95% CI on mean diff  : [{lo:+.1f}, {hi:+.1f}]")
        if lo <= 0:
            print("  NOTE: CI includes 0 -- cannot exclude no effect.")
    else:
        print("  Too few non-tied trials for a signed-rank test.")

    if len(participants) - len(excluded) > 1:
        per = tpos.groupby("participant_id")["diff"].mean()
        wp = stats.wilcoxon(per, alternative="greater")
        print(f"\n  Per-participant mean diff: {dict(per.round(1))}")
        print(f"  Participant-level Wilcoxon: p={wp.pvalue:.4f} "
              f"(avoids pseudo-replication)")
    else:
        print("\n  Single participant -- a design check, not evidence.")

    # ── Q1 secondary: the recovered 2AFC rate ─────────────────────
    print("\n  [secondary] derived 2AFC, ties dropped:")
    if len(nz):
        k, n = int(nz.chose_high.sum()), len(nz)
        bt = stats.binomtest(k, n, 0.5, alternative="greater")
        clo, chi = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
        print(f"    rated sim:high higher: {k}/{n} = {k/n:.3f}   "
              f"95% CI [{clo:.3f}, {chi:.3f}]   p = {bt.pvalue:.4f}")

    # ── Q2 PRIMARY: rating vs cs ──────────────────────────────────
    print("\n" + "=" * 62)
    print("Q2  Is the METRIC any good?   rating vs cs(ctx, section)")
    print("=" * 62)
    lg = pd.concat([
        t.assign(rating=t.rating_high, cs=t.cs_high, cond="sim:high"),
        t.assign(rating=t.rating_low, cs=t.cs_low, cond="sim:low"),
    ], ignore_index=True).dropna(subset=["rating", "cs"])
    lg["clip"] = lg.ctx_stem + "|" + lg.cond
    print(f"  n = {len(lg)} ratings ({len(t)} trials x 2 sections, "
          f"{lg['clip'].nunique()} distinct clips)")
    sp = stats.spearmanr(lg.cs, lg.rating)
    pe = stats.pearsonr(lg.cs, lg.rating)
    print(f"  Spearman(cs, rating) = {sp.statistic:+.3f}  p={sp.pvalue:.4f}")
    print(f"  Pearson (cs, rating) = {pe.statistic:+.3f}  p={pe.pvalue:.4f}")
    try:
        import statsmodels.formula.api as smf
        if lg.participant_id.nunique() > 1:
            m = smf.mixedlm("rating ~ cs", lg, groups=lg.participant_id).fit()
            print(f"  LMM slope on cs = {m.params['cs']:+.1f} rating points "
                  f"per unit cs, p={m.pvalues['cs']:.4f}")
        else:
            m = smf.ols("rating ~ cs", lg).fit()
            print(f"  OLS slope on cs = {m.params['cs']:+.1f}  "
                  f"p={m.pvalues['cs']:.4f}  (single participant: no random "
                  f"effect estimable)")
        print("  (positive slope => perceived similarity tracks the metric)")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [model unavailable: {e}]")

    # ── Crossed mixed models (protocol 7.1) ───────────────────────
    print("\n" + "=" * 62)
    print("LMM  participant AND context as crossed random intercepts (7.1)")
    print("=" * 62)
    if t.participant_id.nunique() > 1 and t.ctx_stem.nunique() > 1:
        lg = lg.assign(z=lg.groupby("participant_id").rating.transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)))
        fits = [("Q1  diff ~ 1", "diff ~ 1", tpos, "Intercept", False),
                ("Q2  rating ~ cs  [protocol, no clip]", "rating ~ cs", lg, "cs", False),
                ("Q2  rating ~ cs  + (1|clip)", "rating ~ cs", lg, "cs", True),
                ("Q2  z(rating) ~ cs + (1|clip)", "z ~ cs", lg, "cs", True)]
        for name, formula, df, term, clip in fits:
            est, se, pv = crossed_lmm(formula, df, term, clip=clip)
            print(f"  {name:36s} {term} = {est:+7.2f}  SE {se:5.2f}  "
                  f"p = {pv:.4f} (two-sided)")
        cm = lg.groupby("clip").agg(r=("rating", "mean"), cs=("cs", "first"))
        pr = stats.pearsonr(cm.cs, cm.r)
        print(f"  Q2  clip means vs cs ({len(cm)} clips): Pearson "
              f"{pr.statistic:+.3f}  p = {pr.pvalue:.4f}")
        tm = t.groupby("trial").agg(delta=("delta", "first"), m=("diff", "mean"))
        agree = int((np.sign(tm.delta) == np.sign(tm.m)).sum())
        print(f"  Trial-level agreement: metric and listeners pick the same "
              f"clip in {agree}/{len(tm)} trials")
        # Slot bias: does slot A get rated higher whatever it holds?
        # c = +1 when high is in A, -1 when in B; its coefficient is the A bias.
        ab = np.where(t.high_slot == "A", t["diff"], -t["diff"])
        per_ab = pd.Series(ab, index=t.index).groupby(t.participant_id).mean()
        print(f"  Slot bias: rating_A - rating_B over all test trials "
              f"{ab.mean():+.1f}, A higher in {(ab > 0).mean():.1%}, "
              f"participant-level Wilcoxon p = {stats.wilcoxon(per_ab).pvalue:.4f}")
        tpos_c = tpos.assign(c=np.where(tpos.high_slot == "A", 1.0, -1.0))
        lg_c = lg.assign(isA=np.where(
            (lg.cond == "sim:high") == (lg.high_slot == "A"), 1.0, 0.0))
        main_fit = {}
        for name, formula, df, term in (
                ("Q1  diff ~ 1 + slot", "diff ~ 1 + c", tpos_c, "Intercept"),
                ("    (slot term)", "diff ~ 1 + c", tpos_c, "c"),
                ("Q2  rating ~ cs + slot + (1|clip)", "rating ~ cs + isA", lg_c, "cs"),
                ("    (slot term)", "rating ~ cs + isA", lg_c, "isA")):
            est, se, pv = crossed_lmm(formula, df, term,
                                      clip=formula.startswith("rating"))
            main_fit[(formula, term)] = (est, se, pv)
            print(f"  {name:36s} {term} = {est:+7.2f}  SE {se:5.2f}  "
                  f"p = {pv:.4f} (two-sided)")
        print("  Context effect: the same test, one mean per context "
              "(generalises over pieces):")
        per_ctx = tpos.groupby("ctx_stem")["diff"].mean()
        wc = stats.wilcoxon(per_ctx, alternative="greater")
        print(f"    {len(per_ctx)} contexts, mean {per_ctx.mean():+.1f}, "
              f"Wilcoxon p = {wc.pvalue:.4f} (one-sided)")
    else:
        print("  Needs more than one participant and context.")

    # Fatigue: first half of the session vs second half. Every participant
    # heard one fixed order, so position and trial identity are confounded.
    pos = tpos.trial.str.extract(r"(\d+)")[0].astype(int)
    half = np.where(pos <= pos.max() / 2, "first", "second")
    by_half = tpos.groupby([tpos.participant_id, half])["diff"].mean().unstack()
    if by_half.shape[1] == 2 and len(by_half) > 1:
        print(f"\n  [sensitivity] first vs second half of the session: mean diff "
              f"{by_half['first'].mean():+.1f} vs {by_half['second'].mean():+.1f}, "
              f"paired Wilcoxon p = "
              f"{stats.wilcoxon(by_half['first'], by_half['second']).pvalue:.3f} "
              f"(two-sided; order is fixed, so position is confounded with trial)")

    if excluded:
        allp = d[(d.kind == "test") & (d.delta > 0)]
        per = allp.groupby("participant_id")["diff"].mean()
        print(f"\n  [sensitivity] Q1 with the {len(excluded)} excluded participants "
              f"put back: mean diff {allp['diff'].mean():+.1f}, participant-level "
              f"Wilcoxon p = {stats.wilcoxon(per, alternative='greater').pvalue:.4f}")

    # ── Negative-delta probe ──────────────────────────────────────
    tneg = t[t.delta < 0]
    if not tneg.empty:
        print("\n" + "=" * 62)
        print("NEGATIVE-DELTA  Metric-divergence probe  (NOT part of Q1)")
        print("=" * 62)
        nzn = tneg[tneg["diff"] != 0]
        k, n = int(nzn.chose_high.sum()), len(nzn)
        print(f"  n = {len(tneg)} trials, mean diff {tneg['diff'].mean():+.1f}")
        if n:
            lo_, hi_ = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
            print(f"  rated sim:high higher: {k}/{n} = {k/n:.3f}  "
                  f"95% CI [{lo_:.3f}, {hi_:.3f}]")
        print("  Here the metric ranks sim:low ABOVE sim:high, so:")
        print("    well above 0.50 => the METRIC is wrong on these contexts")
        print("    well below 0.50 => the MODEL genuinely failed on them")

    # ── Descriptive ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("DESCRIPTIVE")
    print("=" * 62)
    print("\nBy sign group:")
    print(t.groupby("stratum").agg(
        n=("diff", "size"), mean_diff=("diff", "mean"),
        chose_high=("chose_high", "sum"), med_delta=("delta", "median"),
    ).round(2).to_string())

    print("\nRating distribution:")
    allr = pd.concat([t.rating_high, t.rating_low])
    print(f"  min {allr.min():.0f}  q1 {allr.quantile(.25):.0f}  "
          f"median {allr.median():.0f}  q3 {allr.quantile(.75):.0f}  "
          f"max {allr.max():.0f}")
    print(f"  scale use: {allr.min():.0f}-{allr.max():.0f} of 0-100 "
          f"({(allr.max()-allr.min()):.0f} points)")

    print("\n|diff| vs |delta| (does metric magnitude predict perceived magnitude?):")
    sm = stats.spearmanr(t.delta.abs(), t["diff"].abs())
    print(f"  Spearman = {sm.statistic:+.3f}  p={sm.pvalue:.4f}")

    # ── Summary table for the paper ───────────────────────────────
    # Every p comes from one crossed mixed model with the slot term in it,
    # so the reader needs one method. CIs are Wald (estimate +- 1.96 SE).
    if t.participant_id.nunique() > 1 and t.ctx_stem.nunique() > 1:
        print("\n" + "=" * 62)
        print(f"SUMMARY TABLE  ({t.participant_id.nunique()} listeners, "
              f"{t.trial.nunique()} test trials each)")
        print("=" * 62)

        def row(label, est, se, p, scale=1.0, unit=" pts"):
            e, lo, hi = est * scale, (est - 1.96 * se) * scale, (est + 1.96 * se) * scale
            ptxt = "< 0.0001" if p < 1e-4 else f"{p:.4f}"
            print(f"  {label:44s} {e:+6.1f}{unit}  [{lo:+.1f}, {hi:+.1f}]  p {ptxt}")

        # Paper design: Q1 on ALL test trials, by label only.
        # The positive-delta subset depends on compound_sim, which the paper
        # no longer uses; it stays below for comparison with the protocol.
        t_c = t.assign(c=np.where(t.high_slot == "A", 1.0, -1.0))
        row("Q1 high - low, ALL trials (slot removed)",
            *crossed_lmm("diff ~ 1 + c", t_c, "Intercept"))
        nza = t[t["diff"] != 0]
        k, n = int(nza.chose_high.sum()), len(nza)
        clo, chi = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
        print(f"  {'   rated high more similar, ALL trials':44s} {k / n:6.1%}      "
              f"[{clo:.1%}, {chi:.1%}]  ({k}/{n}, ties dropped)")
        row("Q1 high - low (positive-delta, slot removed)",
            *main_fit[("diff ~ 1 + c", "Intercept")])
        if len(nz):
            k, n = int(nz.chose_high.sum()), len(nz)
            clo, chi = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
            print(f"  {'   rated high more similar':44s} {k / n:6.1%}      "
                  f"[{clo:.1%}, {chi:.1%}]  ({k}/{n}, ties dropped)")
        row("Q2 rating per 0.1 rise in cs (slot removed)",
            *main_fit[("rating ~ cs + isA", "cs")], scale=0.1)
        row("Slot bias, A rated above B",
            *main_fit[("rating ~ cs + isA", "isA")])

    print("\nPer trial (mean over valid participants):")
    per_trial = t.groupby("trial").agg(
        stratum=("stratum", "first"), delta=("delta", "first"),
        high_slot=("high_slot", "first"), rating_high=("rating_high", "mean"),
        rating_low=("rating_low", "mean"), diff=("diff", "mean"),
        chose_high=("chose_high", "mean"), n=("diff", "size"))
    print(per_trial.sort_values("delta").round(3).to_string())


if __name__ == "__main__":
    main()
