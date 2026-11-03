r"""
scripts/eval/eval_listening_round2.py

Further listening-test checks: stimuli, power, choice rate and robustness.

  1. Stimuli: the high-low gap of the 30 stimulus pairs on Melody and CS, on the
     10 seconds heard; mean with a 95% bootstrap interval over trials.
  2. Power: the minimum label effect the mixed model could detect with 80% power
     (two-sided alpha 0.05), from the standard error of its estimate.
  3. Choice rate: share of untied listener-trial pairs that favour the
     high-similarity output, with an interval from resampling listeners and
     trials together (two-way cluster bootstrap).
  4. Melody robustness: clips whose Melody score is zero on the 10 s heard, and
     clips that quote eight or more notes of the context; the Melody slope refitted
     without each group (mixed model of eval_listening_metrics.py).
  5. Every measure refitted without the quoting clips, Holm over 12.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_listening_round2.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_copy_rate import measure as copy_measure  # noqa: E402
from eval_listening_metrics import SLOT_FILE, clip_scores, crossed, cut, holm, ratings  # noqa: E402
from eval_listening_slopes import MEASURES  # noqa: E402
from similarity import midi_to_compound  # noqa: E402

SLOT_BIAS = 7.86
LABEL_EFFECT, LABEL_LO, LABEL_HI = 2.9, -2.0, 7.7      # Listening Test section, mixed model


def boot_mean(x: np.ndarray, rng, n=10000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def copies(key2: pd.DataFrame, test1: Path, seconds: float) -> pd.DataFrame:
    """Longest copied melodic run of each clip, on the 10 s heard."""
    rows = []
    for r in key2[key2.kind == "test"].itertuples():
        sdir = test1 / r.source_set
        k1 = pd.read_csv(sdir / "answer_key.csv").set_index("trial")
        hs = k1.at[r.source_trial, "high_slot"]
        tdir = sdir / "trials"
        ctx = cut(midi_to_compound(tdir / f"{r.source_trial}_1_Context.mid"), seconds)
        for cond, slot in (("high", hs), ("low", "B" if hs == "A" else "A")):
            gen = cut(midi_to_compound(tdir / f"{r.source_trial}_{SLOT_FILE[slot]}.mid"), seconds)
            rows.append(dict(ctx_stem=r.ctx_stem, cond=cond, longest=copy_measure(ctx, gen)["longest_run"]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    ap.add_argument("--test1", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/eval/listening_round2.txt")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    key2 = pd.read_csv(args.key)
    r = ratings(key2, Path(args.responses))
    sc = clip_scores(key2, Path(args.test1), args.seconds)
    lines = []

    g = sc.pivot(index="ctx_stem", columns="cond")
    lines.append("1. Stimulus gaps, high - low, 30 trials, 10 s heard")
    for m, name in (("interval_3gram", "Melody"), ("cs", "CS")):
        d = (g[(m, "high")] - g[(m, "low")]).to_numpy()
        mean, lo, hi = boot_mean(d, rng)
        lines.append(f"  {name:7s} mean {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  positive in {(d > 0).sum()} of {len(d)}")

    se = (LABEL_HI - LABEL_LO) / (2 * 1.96)
    lines += ["", "2. Power", f"  SE of the label effect {se:.2f} points; minimum detectable effect "
              f"(80% power, two-sided 0.05) = 2.80 x SE = {2.80 * se:.1f} points"]

    w = r.pivot_table(index=["participant_id", "ctx_stem"], columns="cond", values="rating").reset_index()
    isA = r[r.cond == "high"].set_index(["participant_id", "ctx_stem"]).isA
    w["gap"] = w["high"] - w["low"] - SLOT_BIAS * (2 * isA.reindex(
        pd.MultiIndex.from_frame(w[["participant_id", "ctx_stem"]])).to_numpy() - 1)
    raw = w.assign(raw=w["high"] - w["low"])
    raw = raw[raw.raw != 0]
    rate = (raw.raw > 0).mean()
    lis, tri = raw.participant_id.unique(), raw.ctx_stem.unique()
    bs = []
    for _ in range(4000):
        pl = rng.choice(lis, len(lis)); pt = rng.choice(tri, len(tri))
        sub = pd.concat([raw[raw.participant_id == p] for p in pl])
        sub = pd.concat([sub[sub.ctx_stem == t] for t in pt])
        bs.append((sub.raw > 0).mean())
    lines += ["", "3. Choice rate, untied pairs",
              f"  {rate:.3f} of {len(raw)}; two-way cluster 95% interval "
              f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"]

    cp = copies(key2, Path(args.test1), args.seconds)
    sc2 = sc.merge(cp, on=["ctx_stem", "cond"])
    df = r.merge(sc2, on=["ctx_stem", "cond"])
    df["clip"] = df.ctx_stem + "|" + df.cond
    zero = sc2.interval_3gram == 0
    quote = sc2.longest >= 8
    lines += ["", "4. Melody robustness",
              f"  clips with Melody = 0: {zero.sum()} of {len(sc2)}; clips quoting 8+ notes: {quote.sum()}",
              f"  quoting clips by label (review round 7, R18): {sc2[quote].cond.value_counts().to_dict()}"]
    for name, keep in (("all clips", None), ("without Melody = 0", ~zero), ("without quoting clips", ~quote)):
        clips = set((sc2[keep] if keep is not None else sc2).pipe(lambda d: d.ctx_stem + "|" + d.cond))
        d = df[df["clip"].isin(clips)].copy()
        d["z"] = (d.interval_3gram - d.interval_3gram.mean()) / d.interval_3gram.std()
        est, se_m, p = crossed(d, "rating ~ z + isA", "z")
        lines.append(f"  {name:22s} clips {d['clip'].nunique():2d}  Melody +{est:.2f} per SD "
                     f"[{est - 1.96 * se_m:+.2f}, {est + 1.96 * se_m:+.2f}]  p {p:.4f}")
    # 5. every measure without the quoting clips
    d = df[df["clip"].isin(set((sc2[~quote]).pipe(lambda x: x.ctx_stem + "|" + x.cond)))].copy()
    res = {}
    for col, name in MEASURES.items():
        d["z"] = (d[col] - d[col].mean()) / d[col].std()
        res[name] = crossed(d, "rating ~ z + isA", "z")
    ph = holm(pd.Series({k: v[2] for k, v in res.items()}))
    lines += ["", f"5. Every measure without the {quote.sum()} quoting clips (Holm over 12)"]
    for name, (est, se_m, p) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"  {name:17s} +{est:.2f} per SD [{est - 1.96 * se_m:+.2f}, {est + 1.96 * se_m:+.2f}]  "
                     f"p {p:.4f}  Holm {ph[name]:.4f}")
    text ="\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
