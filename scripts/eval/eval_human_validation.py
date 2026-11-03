r"""
scripts/eval/eval_human_validation.py

Validate the AI section letters (data/annotations/gemini_prompt_v1) against two independent human
annotations (data/annotations/human) of the same pieces.

The AI marks one level of sections; the humans mark nested levels (L1 finest). So no
single human level is chosen. An AI section is consistent with a human annotation at
level k when it equals one segment there (exact) or a run of consecutive segments
(run). Its human label at k is the set of letters in that run.

Reported
--------
  1. Boundaries: share of AI interior boundaries that are a human boundary at some level.
  2. Sections: share of AI sections that are one segment at some level (exact; and the
     lowest such level), or only a run of segments.
  3. Same / different: for each pair of AI sections of one piece, the human verdict is
     taken at the lowest level where both are one segment, else the lowest level where
     both match; "same" when their letter sets are equal. Agreement and Cohen's kappa for AI-A1, AI-A2 and A1-A2, on the pairs all
     three judge. A1-A2 is the human ceiling.
  4. The paper's AA/AB check (compound similarity, column compound_ks of
     pairs.csv) with the AI letters and with each annotator's verdict, on the same
     pairs. AUC with a 95% interval from resampling whole pieces.
  5. Diagnostic only, not in the paper: the humans mapped onto the AI's framework: the top human level is the section level
     (pr -> In, po -> Ou, tr -> Br, si -> St, it/mi/en -> main body; In/Ou/Br/St get the
     letter X, as the AI prompt asks). Lp and Ln are merged into the main body: the AI
     chose Lp exactly where the score has a loop, so they agree by construction. Per bar:
     function agreement and kappa, section boundary F1 (interior, exact) and letter
     pairwise F1, each X segment its own letter.

Placeholder letters mark no material: the AI's X (intro, outro, bridge, stinger) and the
humans' x (silence). Each such section is its own letter, so two of them never count as the
same material. --main_only drops from part 3 every pair
with an AI X section or a human x segment, since the AI and the humans label such
material by different schemes; the training pairs are main-body sections only.

Intervals: 95%, from resampling whole pieces.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\eval_human_validation.py [--boot 2000] [--exclude 00395 ...]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
AI_DIR = ROOT / "data/annotations/gemini_prompt_v1"
HUM = ROOT / "data/annotations/human"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import PAIRS  # noqa: E402
SECFILE = re.compile(r"_s\d+_([A-Za-z]+)_(\d+)-(\d+)\.npy$")
WHO = ("A1", "A2")


def human_levels(path: Path) -> dict:
    """{level: [(start, end, letter)]}, bars as offsets from the first bar, end exclusive."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["start_bar"].strip()]
    first = int(rows[0]["start_bar"])
    end = int(rows[-1]["start_bar"]) + int(rows[-1]["length"]) - first
    out = {}
    for k in range(1, 6):
        marks = [(int(r["start_bar"]) - first, r[f"L{k}_letter"].strip()) for r in rows
                 if r.get(f"L{k}_letter", "").strip()]
        marks = [(a, f"x{a}" if l == "x" else l) for a, l in marks]          # silence: its own letter
        if marks:
            out[k] = [(a, b, l) for (a, l), (b, _) in zip(marks, marks[1:] + [(end, "")])]
    return out


def match(a: int, b: int, levels: dict):
    """{level: (kind, letter set)} for every level where [a, b) is one segment or a run."""
    out = {}
    for k, segs in levels.items():
        starts = [i for i, s in enumerate(segs) if s[0] == a]
        ends = [i for i, s in enumerate(segs) if s[1] == b]
        if starts and ends and starts[0] <= ends[0]:
            i, j = starts[0], ends[0]
            out[k] = ("exact" if i == j else "run", frozenset(s[2] for s in segs[i:j + 1]))
    return out


def verdict(mi: dict, mj: dict):
    """Same material? At the lowest level where both are one segment; else the lowest where both match."""
    common = sorted(set(mi) & set(mj))
    exact = [k for k in common if mi[k][0] == mj[k][0] == "exact"]
    k = (exact or common or [None])[0]
    return None if k is None else mi[k][1] == mj[k][1]


def placeholder(ai_letter: str, mm: dict) -> bool:
    """An AI X section, or one that covers a human x (silence) segment at any level."""
    return ai_letter.startswith("X") or any(
        l.startswith("x") for w in mm.values() for _, ls in w.values() for l in ls)


def kappa(x, y) -> float:
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(groups: dict, stat, rng, n: int):
    """95% interval of stat(list of rows) when whole pieces are resampled."""
    keys, vals = list(groups), []
    for _ in range(n):
        v = stat([r for k in rng.choice(keys, len(keys)) for r in groups[k]])
        if v == v:
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def pairwise_f1(x, y) -> float:
    """F1 of the bar pairs that share a label in x and in y."""
    from collections import Counter
    c2 = lambda c: sum(v * (v - 1) / 2 for v in c.values())
    sx, sy, sxy = c2(Counter(x)), c2(Counter(y)), c2(Counter(zip(x, y)))
    return 1.0 if sx == sy == 0 else 2 * sxy / (sx + sy)


FN = {"pr": "In", "po": "Ou", "tr": "Br", "si": "St", "it": "main", "mi": "main", "en": "main"}
AI_FN = {"Lp": "main", "Ln": "main"}


def human_top(path: Path, edges: bool = False):
    """Per-bar (function, letter) of the top human level, mapped onto the AI's framework.
    edges: also map the opening run of pr (and the closing run of po) at any level to In (Ou)."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["start_bar"].strip()]
    first = int(rows[0]["start_bar"])
    k = max(k for k in range(1, 6) if any(r.get(f"L{k}_letter", "").strip() for r in rows))
    out = []
    for r in rows:
        if r[f"L{k}_letter"].strip():
            f = FN.get(r[f"L{k}_function"].strip(), "?")
            cur = (f, r[f"L{k}_letter"].strip() if f == "main" else f"X{int(r['start_bar']) - first}")
        out += [cur] * int(r["length"])
    if edges:
        for fn, tag, order in (("pr", "In", range(len(rows))), ("po", "Ou", range(len(rows) - 1, -1, -1))):
            for i in order:
                r = rows[i]
                if not any(r.get(f"L{j}_function", "").strip() == fn for j in range(1, 6)):
                    break
                a = int(r["start_bar"]) - first
                for b in range(a, a + int(r["length"])):
                    out[b] = (tag, f"X{tag}")
    return out


def kappa_multi(x, y) -> float:
    """Cohen's kappa for nominal labels."""
    from collections import Counter
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in cx) / n / n
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def auc(pos, neg) -> float:
    return stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--exclude", nargs="*", default=[], help="local ids to leave out (robustness check)")
    ap.add_argument("--main_only", action="store_true",
                    help="part 3 on main-body material only: drop every pair with an AI X section or a human x segment")
    ap.add_argument("--measure", default="compound_ks",
                    help="measure for part 4: compound_ks, or a column of ab_label_measures.csv such as interval_3gram")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pieces = [r for r in csv.DictReader(open(HUM / "pieces.csv", encoding="utf-8")) if r["comparable"] == "1" and r["local_id"] not in args.exclude]

    hit_b = {w: 0 for w in WHO}
    nb, nsec = 0, 0
    sec_kind = {w: {"exact": 0, "run": 0} for w in WHO}
    sec_lvl = {w: {} for w in WHO}
    pair_rows = []                                   # (piece, ai_same, A1, A2)
    ranges = {}                                      # (local id, start, end) -> {who: match}
    for p in pieces:
        lid = p["local_id"]
        ai = json.load(open(AI_DIR / f"{lid}.json", encoding="utf-8"))["Section"]
        a0 = min(s["BarRange"][0] for s in ai)
        lv = {w: human_levels(HUM / f"annotator{w[1]}" / f"{lid}.csv") for w in WHO}
        hend = max(segs[-1][1] for segs in lv["A1"].values())
        secs = [(s["BarRange"][0] - a0, min(s["BarRange"][1] - a0 + 1, hend),
                 s["Section"] if s["Section"] != "X" else f"X{s['BarRange'][0]}") for s in ai]
        nb += len(secs) - 1
        nsec += len(secs)
        m = []
        for a, b, _ in secs:
            mm = {w: match(a, b, lv[w]) for w in WHO}
            ranges[(lid, a + a0, b + a0 - 1)] = mm
            m.append(mm)
            for w in WHO:
                if mm[w]:
                    ex = [k for k, v in mm[w].items() if v[0] == "exact"]
                    sec_kind[w]["exact" if ex else "run"] += 1
                    if ex:
                        sec_lvl[w][f"L{min(ex)}"] = sec_lvl[w].get(f"L{min(ex)}", 0) + 1
        for w in WHO:
            allb = set().union(*[{s[0] for s in segs} for segs in lv[w].values()])
            hit_b[w] += sum(a in allb for a, _, _ in secs[1:])
        for i, j in itertools.combinations(range(len(secs)), 2):
            if args.main_only and any(placeholder(secs[k][2], m[k]) for k in (i, j)):
                continue
            pair_rows.append((lid, secs[i][2] == secs[j][2],
                              verdict(m[i]["A1"], m[j]["A1"]), verdict(m[i]["A2"], m[j]["A2"])))

    print(f"Pieces: {len(pieces)}; AI sections {nsec}; AI interior boundaries {nb}\n")
    print("1-2. Boundaries and sections against any human level")
    for w in WHO:
        e, r = sec_kind[w]["exact"], sec_kind[w]["run"]
        print(f"  {w}: boundaries {hit_b[w]}/{nb} = {hit_b[w] / nb:.2f}; sections exact {e}, run {r}, "
              f"consistent {(e + r)}/{nsec} = {(e + r) / nsec:.2f}; exact at (lowest) {dict(sorted(sec_lvl[w].items()))}")

    print("\n3. Same / different on AI section pairs (all pairs within a piece)")
    allp = [r for r in pair_rows if r[2] is not None and r[3] is not None]
    print(f"  pairs {len(pair_rows)}; judged by both humans {len(allp)}")
    groups = {}
    for r in allp:
        groups.setdefault(r[0], []).append(r)
    for name, x, y in (("AI-A1", 1, 2), ("AI-A2", 1, 3), ("A1-A2", 2, 3)):
        st = lambda rs: kappa([r[x] for r in rs], [r[y] for r in rs])
        agree = np.mean([r[x] == r[y] for r in allp])
        lo, hi = boot_ci(groups, st, rng, args.boot)
        print(f"  {name}: agreement {agree:.2f}, kappa {st(allp):.2f} ({lo:.2f} to {hi:.2f})")

    print("\n4. The paper's AA/AB check on its own pairs (compound_ks)")
    lids = {p["local_id"] for p in pieces}
    rows = []
    for r in csv.DictReader(open(PAIRS, encoding="utf-8")):
        if r["piece_id"] not in lids:
            continue
        (la, a1, a2), (lb, b1, b2) = (SECFILE.search(r[c]).groups() for c in ("context_file", "target_file"))
        ma, mb = ranges.get((r["piece_id"], int(a1), int(a2))), ranges.get((r["piece_id"], int(b1), int(b2)))
        if ma is None or mb is None:
            continue
        rows.append({"piece": r["piece_id"], "pair_id": r["pair_id"], "cs": float(r["compound_ks"]), "AI": la == lb,
                     "A1": verdict(ma["A1"], mb["A1"]), "A2": verdict(ma["A2"], mb["A2"])})
    both = [r for r in rows if r["A1"] is not None and r["A2"] is not None]
    print(f"  pairs in these pieces {len(rows)}; judged by both humans {len(both)} "
          f"({len({r['piece'] for r in both})} pieces)")
    by_piece = {}
    for r in both:
        by_piece.setdefault(r["piece"], []).append(r)
    keys = list(by_piece)
    if args.measure != "compound_ks":
        # another measure from eval_ab_labels.py output, joined by pair id
        alt = pd.read_csv("outputs/eval/ab_label_measures.csv").set_index("pair_id")[args.measure]
        for r in both:
            r["cs"] = float(alt[int(r["pair_id"])])
        print(f"  measure: {args.measure}")
    for lab in ("AI", "A1", "A2"):
        cs = np.array([r["cs"] for r in both]); same = np.array([r[lab] for r in both])
        boots = []
        for _ in range(args.boot):
            s = [r for k in rng.choice(keys, len(keys)) for r in by_piece[k]]
            c = np.array([r["cs"] for r in s]); y = np.array([r[lab] for r in s])
            if y.any() and (~y).any():
                boots.append(auc(c[y], c[~y]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  letters from {lab}: same {same.sum()}, different {(~same).sum()}; "
              f"AUC {auc(cs[same], cs[~same]):.3f} ({lo:.3f} to {hi:.3f})")


    for edges in (False, True):
        part5(pieces, edges, rng, args.boot)


def part5(pieces, edges, rng, nboot):
    print("\n5" + ("b" if edges else "a") + ". Humans mapped onto the AI's framework (top human level = sections"
          + ("; In / Ou = opening pr / closing po at any level)" if edges else "; In / Ou = top-level pr / po)"))
    per = {}                                         # piece -> {source: [(function, letter)] per bar}
    for p in pieces:
        lid = p["local_id"]
        hum = {w: human_top(HUM / f"annotator{w[1]}" / f"{lid}.csv", edges) for w in WHO}
        n = min(len(v) for v in hum.values())
        ai = json.load(open(AI_DIR / f"{lid}.json", encoding="utf-8"))["Section"]
        a0 = min(s["BarRange"][0] for s in ai)
        lab = [("-", "-")] * n
        for s in ai:
            f = AI_FN.get(s["Function"], s["Function"])
            for b in range(s["BarRange"][0] - a0, min(s["BarRange"][1] - a0 + 1, n)):
                lab[b] = (f, s["Section"] if s["Section"] != "X" else f"X{s['BarRange'][0] - a0}")
        per[lid] = {"AI": lab, **{w: v[:n] for w, v in hum.items()}}

    def starts(seq):
        return {i for i in range(1, len(seq)) if seq[i] != seq[i - 1]}

    def bf1(x, y):
        bx, by = starts(x), starts(y)
        return 1.0 if not bx and not by else 2 * len(bx & by) / (len(bx) + len(by))

    groups = {k: [v] for k, v in per.items()}
    for name, x, y in (("AI-A1", "AI", "A1"), ("AI-A2", "AI", "A2"), ("A1-A2", "A1", "A2")):
        fk = lambda rs: kappa_multi([f for r in rs for f, _ in r[x]], [f for r in rs for f, _ in r[y]])
        fa = lambda rs: np.mean([a[0] == b[0] for r in rs for a, b in zip(r[x], r[y])])
        sb = lambda rs: np.mean([bf1(r[x], r[y]) for r in rs])
        lf = lambda rs: np.mean([pairwise_f1([l for _, l in r[x]], [l for _, l in r[y]]) for r in rs])
        out = []
        for lab, st in (("function agreement", fa), ("function kappa", fk), ("section boundary F1", sb), ("letter pairwise F1", lf)):
            lo, hi = boot_ci(groups, st, rng, nboot)
            out.append(f"{lab} {st(list(per.values())):.2f} ({lo:.2f} to {hi:.2f})")
        print(f"  {name}: " + "; ".join(out))
    from collections import Counter
    for src in ("AI", "A1", "A2"):
        print(f"  bars by function, {src}: " + str(dict(Counter(f for v in per.values() for f, _ in v[src]))))


if __name__ == "__main__":
    main()
