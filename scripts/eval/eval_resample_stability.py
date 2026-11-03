r"""
scripts/eval/eval_resample_stability.py

Generation-stability check: is M1's high-vs-low effect stable across repeated
draws, or does it depend on the luck of one draw?

Why
---
The published eval drew one output per context and bucket (sample 0). In the
listening test, listeners agreed closely within a trial but trials differed a
lot. Two readings predict different things here:

  unstable model         the same context flips between high > low and
                         high < low across draws; little between-context
                         variance beyond chance (low ICC)
  context-dependent      some contexts give high > low on most draws, others
                         on few; real between-context variance (high ICC)

Inputs
------
- `--resample_dir`: the CORRECTED output of infer_resample.py
  ({ctx}_m1_{bucket}_s{k}_generated.mid), produced by
  correct_generated_output.py like every other eval input.
- `--s0_csv`: the published ctx-sim scores (sample 0), from eval_triplet.py.
- Optional listening-test key and responses, to set the per-trial listener
  result beside the per-context stability.

Rules
-----
- Only the 151 contexts of the published eval are used (same exclusion rule).
- A draw (context, sample) is dropped if any of its three outputs has fewer than
  `--min_notes` notes, so every compared triple comes from one draw.
- compound_sim and MIDI reading are imported from similarity.py, so scores are
  computed exactly as published.

Reported
--------
1. Per-sample replication: fp and Wilcoxon p for high > low in each draw.
2. Averaged over draws: per-context mean delta, its fp and Wilcoxon p.
3. Variance split: between-context vs within-context (draw-to-draw) variance of
   delta, and the ICC. Also an overdispersion test on per-context win counts.
4. Does one draw predict the others? sample-0 delta vs mean of samples 1..K.
5. The same for mid vs low and high vs mid (the two-level claim).
6. With listening data: listener mean diff per trial vs that context's
   resampled mean delta and win rate.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_resample_stability.py `
        --resample_dir outputs\generated\m1_resample `
        --sections_dir "data\sections" `
        --listening_key       listening_test\stimuli\test2_one_set_10s\answer_key.csv `
        --listening_responses listening_test\responses\test2\
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from similarity import compound_sim, midi_to_compound  # noqa: E402

FILE_RE = re.compile(r"^(.+)_m1_(high|mid|low)_s(\d+)_generated\.mid$")
PAIRS = [("high", "low"), ("mid", "low"), ("high", "mid")]


def score_resample(res_dir: Path, sec_dir: Path, stems: set[str],
                   min_notes: int) -> pd.DataFrame:
    """One row per (ctx, sample, bucket): cs and note count."""
    rows, ctx_cache = [], {}
    for f in sorted(res_dir.glob("*_generated.mid")):
        m = FILE_RE.match(f.name)
        if not m or m.group(1) not in stems:
            continue
        stem, bucket, s = m.group(1), m.group(2), int(m.group(3))
        if stem not in ctx_cache:
            ctx_cache[stem] = np.load(str(sec_dir / f"{stem}.npy"))
        gen = midi_to_compound(f)
        if len(gen):
            gen = gen.copy()
            gen[:, 0] -= gen[0, 0]
        rows.append(dict(ctx_stem=stem, sample=s, bucket=bucket, n_notes=len(gen),
                         ctx_sim=compound_sim(ctx_cache[stem], gen)
                         if len(gen) >= min_notes else np.nan))
    return pd.DataFrame(rows)


def icc_oneway(wide: pd.DataFrame) -> tuple[float, float, float, float]:
    """ICC(1) for an unbalanced one-way layout; rows = contexts, cols = draws.

    Returns (icc, between-context variance, within-context variance, F-test p).
    """
    groups = [r.dropna().to_numpy() for _, r in wide.iterrows()]
    groups = [g for g in groups if len(g) >= 2]
    n_i = np.array([len(g) for g in groups])
    k, N = len(groups), n_i.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    var_b = max(0.0, (msb - msw) / n0)
    p = stats.f.sf(msb / msw, k - 1, N - k)
    return var_b / (var_b + msw), var_b, msw, p


def listener_trials(key_path: str, resp: str) -> pd.DataFrame:
    """Mean listener diff and chose-high rate per test trial, valid listeners."""
    from score_listening_ratings import is_survey, parse_survey, parse_wide
    src = Path(resp)
    files = ([Path(f) for f in glob.glob(resp)] if "*" in resp
             else sorted(src.glob("*.csv")) if src.is_dir() else [src])
    long = pd.concat([parse_survey(f) if is_survey(f) else parse_wide(f)
                      for f in files], ignore_index=True)
    w = long.pivot_table(index=["participant_id", "trial"], columns="slot",
                         values="rating").reset_index()
    d = pd.read_csv(key_path).merge(w, on="trial")
    d["diff"] = np.where(d.high_slot == "A", d.A - d.B, d.B - d.A)
    bad = d[(d.kind == "check") & (d["diff"] <= 0)].participant_id.unique()
    t = d[(d.kind == "test") & ~d.participant_id.isin(bad)]
    return t.groupby("ctx_stem").agg(
        trial=("trial", "first"), listen_diff=("diff", "mean"),
        listen_high=("diff", lambda x: (x > 0).mean()),
        n_listeners=("diff", "size")).reset_index()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample_dir", required=True)
    ap.add_argument("--sections_dir", required=True)
    ap.add_argument("--s0_csv", default="outputs/eval/m1_triplet_ctx_sim.csv")
    ap.add_argument("--min_notes", type=int, default=10)
    ap.add_argument("--out_csv", default="outputs/eval/m1_resample_ctx_sim.csv")
    ap.add_argument("--listening_key")
    ap.add_argument("--listening_responses")
    args = ap.parse_args()

    s0 = pd.read_csv(args.s0_csv)
    s0 = s0[s0.model == "m1"].assign(sample=0, n_notes=np.nan)
    stems = set(s0.ctx_stem)
    print(f"Published contexts (sample 0): {len(stems)}")

    res = score_resample(Path(args.resample_dir), Path(args.sections_dir),
                         stems, args.min_notes)
    if res.empty:
        raise SystemExit(f"No resampled files for those contexts in {args.resample_dir}")
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    samples = sorted(res["sample"].unique())
    print(f"Resampled files: {len(res)}  samples {samples}  -> {out}")

    allr = pd.concat([s0[["ctx_stem", "sample", "bucket", "ctx_sim"]],
                      res[["ctx_stem", "sample", "bucket", "ctx_sim"]]])
    w = allr.pivot_table(index=["ctx_stem", "sample"], columns="bucket",
                         values="ctx_sim").reset_index()
    n_draws = len(w)
    w = w.dropna(subset=["high", "mid", "low"])
    print(f"Draws kept: {len(w)} of {n_draws} "
          f"({n_draws - len(w)} dropped: an output under {args.min_notes} notes "
          f"or missing)")
    for a, b in PAIRS:
        w[f"{a}-{b}"] = w[a] - w[b]

    for a, b in PAIRS:
        col = f"{a}-{b}"
        print("\n" + "=" * 66)
        print(f"{a.upper()} vs {b.upper()}   delta = cs({a}) - cs({b})")
        print("=" * 66)

        # 1. each draw as a replication of the published experiment
        print("  per draw (each one a full replication):")
        fps = []
        for s, g in w.groupby("sample"):
            x = g[col]
            fp = (x > 0).mean()
            fps.append(fp)
            p = stats.wilcoxon(x, alternative="greater").pvalue
            tag = "  <- published" if s == 0 else ""
            print(f"    s{s}: n {len(x):3d}  fp {fp:.3f}  mean {x.mean():+.4f}  "
                  f"p {p:.4f} (one-sided){tag}")
        print(f"  fp across draws: mean {np.mean(fps):.3f}  "
              f"min {min(fps):.3f}  max {max(fps):.3f}")

        # 2. averaged over draws
        wide = w.pivot(index="ctx_stem", columns="sample", values=col)
        mean_d = wide.mean(axis=1)
        p = stats.wilcoxon(mean_d, alternative="greater").pvalue
        print(f"  averaged over draws, per context: n {len(mean_d)}  "
              f"fp {(mean_d > 0).mean():.3f}  mean {mean_d.mean():+.4f}  "
              f"p {p:.4f} (one-sided)")

        # 3. variance split
        icc, vb, vw, pf = icc_oneway(wide)
        print(f"  variance of delta: between contexts {vb:.6f}  "
              f"within context (draw to draw) {vw:.6f}")
        print(f"  ICC {icc:.3f}  (0 = all draw-to-draw noise, "
              f"1 = all context)   F-test p {pf:.4g}")
        wins = (wide > 0).sum(axis=1)
        tries = wide.notna().sum(axis=1)
        pbar = wins.sum() / tries.sum()
        exp = tries * pbar
        chi = (((wins - exp) ** 2) / (exp * (1 - pbar))).sum()
        dof = len(wins) - 1
        print(f"  win counts vs one shared rate {pbar:.3f}: chi2 {chi:.1f} on "
              f"{dof} df, p {stats.chi2.sf(chi, dof):.4g}  "
              f"(small p = contexts truly differ)")
        full = tries == tries.max()
        if full.any():
            dist = wins[full].value_counts().sort_index()
            print(f"  contexts with all {tries.max()} draws, by number of "
                  f"{a} > {b} wins: {dict(dist)}")

        # 4. does one draw predict the rest?
        if 0 in wide.columns and wide.shape[1] > 1:
            rest = wide.drop(columns=0).mean(axis=1)
            both = pd.concat([wide[0], rest], axis=1).dropna()
            r, pr = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])
            agree = (np.sign(both.iloc[:, 0]) == np.sign(both.iloc[:, 1])).mean()
            print(f"  sample 0 vs mean of the other draws: Spearman {r:+.3f} "
                  f"(p {pr:.4f}), same sign in {agree:.1%} of contexts")

    # 6. listening test
    if args.listening_key and args.listening_responses:
        print("\n" + "=" * 66)
        print("LISTENING TEST vs RESAMPLED STABILITY  (exploratory)")
        print("=" * 66)
        lt = listener_trials(args.listening_key, args.listening_responses)
        wide = w.pivot(index="ctx_stem", columns="sample", values="high-low")
        st = pd.DataFrame({
            "s0_delta": wide.get(0),
            "rest_mean": wide.drop(columns=0, errors="ignore").mean(axis=1),
            "rest_win": (wide.drop(columns=0, errors="ignore") > 0).mean(axis=1),
        }).reset_index()
        m = lt.merge(st, on="ctx_stem", how="left")
        print(f"  test trials matched: {m.rest_mean.notna().sum()} of {len(m)}")
        for col, what in (("s0_delta", "delta of the heard draw"),
                          ("rest_mean", "mean delta, other draws"),
                          ("rest_win", "win rate, other draws")):
            ok = m[[col, "listen_diff"]].dropna()
            if len(ok) > 3:
                r, p = stats.spearmanr(ok[col], ok.listen_diff)
                print(f"  listener diff vs {what:24s}: Spearman {r:+.3f}  "
                      f"p {p:.4f}  n {len(ok)}")
        print("\n  per trial:")
        print(m.sort_values("listen_diff")[
            ["trial", "ctx_stem", "listen_diff", "listen_high", "s0_delta",
             "rest_mean", "rest_win"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
