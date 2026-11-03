r"""
scripts/data/extract_sections.py

Extract per-section compound-token arrays and build the section-pair dataset
used for section-level SimLlama training.

Pairing rules (X/s sections excluded throughout):
  Same-label : (first_occurrence, nth_occurrence) per label per piece.
               Encoded as (sec:A, sec:A) at training time.
  Diff-label : consecutive adjacent pairs only.
               Encoded as (sec:A, sec:B) at training time.

Outputs (written to --output_dir, default = <data_dir>/sections/):
  sections/<piece_id>_s<idx>_<label>_<bs>-<be>.npy
      Raw compound tokens (shape K x 6) for one section.
      No SOS/EOS -- added by the dataset loader.

  pairs.csv
      One row per training pair.  Columns:
        pair_id, piece_id, split, pair_type,
        context_file, target_file,
        context_label, target_label,
        context_bars, target_bars,
        context_ntokens, target_ntokens,
        tgt_key_shift, is_synthetic

      tgt_key_shift: semitone shift k in [0,11] that maximises
          cosine(ctx_chroma, roll(tgt_chroma, k)).  Applied to target before
          augmentation so ctx and tgt share the same tonal centre.
      is_synthetic: True for synthesised diff pairs (--synthesize only).

Optional flags:
  --matched_only : keep only pairs where the context_file appears in BOTH
                   same-label AND diff-label pairs (gate signal always present).
  --synthesize   : for each unmatched AA context (same pairs but no diff pair),
                   synthesise a diff pair using another section from the same piece.
                   Marks those rows is_synthetic=True.
  Use both flags together to train with matched and synthetic pairs.

Usage (from project root):
    python scripts/data/extract_sections.py \
        --data_dir  "data/npy" \
        --annot_dir "data/annotation/annotations"

    # Matched + synthetic pairs:
    python scripts/data/extract_sections.py \
        --data_dir  "..." --annot_dir "..." --matched_only --synthesize

    # Dry run (print stats only, no files written):
    python scripts/data/extract_sections.py ... --dry_run
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Key-shift helpers ─────────────────────────────────────────────────────────

def _chroma_weighted(arr: np.ndarray):
    """Duration-weighted pitch-class histogram (12,), normalised.
    Compound columns: onset(0) duration(1) octave(2) pitch_class(3) ...
    Returns None if no notes."""
    chroma = np.zeros(12, dtype=float)
    for row in arr:
        pc  = int(row[3]) % 12
        dur = max(float(row[1]), 0.0)
        chroma[pc] += dur
    total = chroma.sum()
    return chroma / total if total > 1e-9 else None


def _compute_key_shift(ctx_arr: np.ndarray, tgt_arr: np.ndarray,
                       min_gain: float = 0.1) -> int:
    """Return k in [0,11] that maximises cosine(ctx_chroma, roll(tgt_chroma, k)).

    Only returns a non-zero shift if it improves similarity over k=0 by at least
    min_gain (default 0.1).  This gates out spurious shifts on flat/noisy chromagrams
    where the argmax is arbitrary noise rather than a genuine key modulation.
    """
    ctx_ch = _chroma_weighted(ctx_arr)
    tgt_ch = _chroma_weighted(tgt_arr)
    if ctx_ch is None or tgt_ch is None:
        return 0
    sim_at_0 = float(np.dot(ctx_ch, tgt_ch))
    best_k, best_sim = 0, sim_at_0
    for k in range(1, 12):
        sim = float(np.dot(ctx_ch, np.roll(tgt_ch, k)))
        if sim > best_sim:
            best_sim = sim
            best_k = k
    if best_k != 0 and (best_sim - sim_at_0) < min_gain:
        return 0
    return best_k

# Minimum notes required for a section to be included in any pair.
MIN_TOKENS = 4

# Pieces excluded due to MIDI-MXL bar offset mismatch.
# Root cause: MuseScore exports MIDI with repeat signs expanded; music21 parses MXL
# bar offsets without expanding repeats.  Bar onsets in _condition.npy are derived
# from the written-score offsets, so sections later in the piece are mis-aligned in
# the expanded MIDI.  Confirmed by check_section_alignment.py (>200-tick offset).
EXCLUDE_PIECES: set[str] = set()  # no pieces excluded -- tempo-drift fix applied in mxl_to_mid.py


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_annotation(json_path):
    """Return (section_list, is_stinger).
    section_list: [(label, bar_start, bar_end), ...] with X/s removed, 1-indexed bars.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    is_stinger = data.get("IsStinger", False)
    sections = []
    for entry in data.get("Section", []):
        lbl = entry["Section"]
        if lbl in ("X", "s"):
            continue
        bs, be = entry["BarRange"]
        sections.append((lbl, bs, be))
    return sections, is_stinger


def extract_section_tokens(npy_tokens, bar_onsets, bar_start, bar_end):
    """
    Extract compound tokens for bars [bar_start, bar_end] (1-indexed, inclusive).

    Parameters
    ----------
    npy_tokens : np.ndarray, shape (N, 6)
        Full-piece compound tokens. Column 0 = onset tick.
    bar_onsets : np.ndarray, shape (num_bars,)
        Onset tick for each bar (0-indexed).
    bar_start, bar_end : int
        1-indexed bar range (inclusive).

    Returns
    -------
    np.ndarray, shape (K, 6) — tokens in the requested bar range.
    """
    b0 = bar_start - 1   # 0-indexed first bar
    b1 = bar_end   - 1   # 0-indexed last bar

    onset_lo = int(bar_onsets[b0])
    if b1 + 1 < len(bar_onsets):
        onset_hi = int(bar_onsets[b1 + 1])
    else:
        # Last section — extend to end of piece
        onset_hi = int(npy_tokens[:, 0].max()) + 1

    mask = (npy_tokens[:, 0] >= onset_lo) & (npy_tokens[:, 0] < onset_hi)
    return npy_tokens[mask]


def same_label_pairs(sections):
    """
    For each label, return (proto_idx, repeat_idx) pairs where proto_idx is
    always the first occurrence.
    """
    by_label = defaultdict(list)
    for i, (lbl, _, __) in enumerate(sections):
        by_label[lbl].append(i)
    pairs = []
    for lbl, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        proto = idxs[0]
        for repeat in idxs[1:]:
            pairs.append((proto, repeat))
    return pairs


def diff_label_consecutive_pairs(sections):
    """Adjacent (i, i+1) pairs where labels differ."""
    pairs = []
    for i in range(len(sections) - 1):
        if sections[i][0] != sections[i + 1][0]:
            pairs.append((i, i + 1))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,
                        help="Root of .npy dataset (contains condition.csv and processed/)")
    parser.add_argument("--annot_dir",  required=True,
                        help="Directory with per-piece annotation JSON files")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write sections/ and pairs.csv "
                             "(default: <data_dir>/sections)")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print statistics only; do not write any files")
    parser.add_argument("--matched_only", action="store_true",
                        help="Keep only pairs whose context_file appears in "
                             "both same-label AND diff-label pairs (after --synthesize).")
    parser.add_argument("--synthesize", action="store_true",
                        help="For each unmatched AA context (same pairs but no "
                             "diff pair), synthesise a diff pair from another section in "
                             "the same piece.  Marks those rows is_synthetic=True.")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    annot_dir  = Path(args.annot_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "sections"
    proc_dir   = data_dir / "processed"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load condition.csv for split labels
    splits = {}
    with open(data_dir / "condition.csv", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row["id"]).zfill(5)
            splits[pid] = row["split"]

    pair_rows   = []
    pair_id     = 0
    stats       = {"same": 0, "diff": 0, "skipped_stinger": 0,
                   "skipped_no_sections": 0, "skipped_short": 0,
                   "pieces_with_same": 0, "synthetic": 0}

    # piece_sec_files[pid] = {i: (fname, ntokens, arr)} for synthesis step
    piece_sec_files: dict = {}

    piece_ids = sorted(splits.keys())
    for pid in piece_ids:
        jf   = annot_dir  / f"{pid}.json"
        nf   = proc_dir   / f"{pid}.npy"
        cf   = proc_dir   / f"{pid}_condition.npy"

        if not jf.exists() or not nf.exists() or not cf.exists():
            continue

        if pid in EXCLUDE_PIECES:
            continue

        sections, is_stinger = load_annotation(jf)
        if is_stinger:
            stats["skipped_stinger"] += 1
            continue
        if len(sections) < 2:
            stats["skipped_no_sections"] += 1
            continue

        npy_tokens  = np.load(nf)
        cond        = np.load(cf)
        bar_onsets  = cond[0::2, 1].astype(np.int64)

        split = splits.get(pid, "train")

        # ── Extract and save per-section token arrays ────────────────────────
        sec_files = {}   # section index -> (fname, ntokens, arr)
        for i, (lbl, bs, be) in enumerate(sections):
            tokens = extract_section_tokens(npy_tokens, bar_onsets, bs, be)
            if len(tokens) < MIN_TOKENS:
                sec_files[i] = None
                continue
            fname = f"{pid}_s{i}_{lbl}_{bs}-{be}.npy"
            fpath = output_dir / fname
            if not args.dry_run:
                np.save(fpath, tokens.astype(np.int64))
            sec_files[i] = (fname, len(tokens), tokens)

        # Store for synthesis step (need arr later)
        piece_sec_files[pid] = (sections, sec_files, split)

        def valid(idx):
            return sec_files.get(idx) is not None

        # ── Same-label pairs ─────────────────────────────────────────────────
        sp = same_label_pairs(sections)
        piece_has_same = False
        for pi, ri in sp:
            if not valid(pi) or not valid(ri):
                stats["skipped_short"] += 1
                continue
            cf_name, cn, ctx_arr = sec_files[pi]
            rf_name, rn, tgt_arr = sec_files[ri]
            lbl = sections[pi][0]
            pair_rows.append({
                "pair_id":         pair_id,
                "piece_id":        pid,
                "split":           split,
                "pair_type":       "same",
                "context_file":    cf_name,
                "target_file":     rf_name,
                "context_label":   lbl,
                "target_label":    lbl,
                "context_bars":    f"{sections[pi][1]}-{sections[pi][2]}",
                "target_bars":     f"{sections[ri][1]}-{sections[ri][2]}",
                "context_ntokens": cn,
                "target_ntokens":  rn,
                "tgt_key_shift":   _compute_key_shift(ctx_arr, tgt_arr),
                "is_synthetic":    False,
            })
            pair_id += 1
            stats["same"] += 1
            piece_has_same = True
        if piece_has_same:
            stats["pieces_with_same"] += 1

        # ── Diff-label consecutive pairs ─────────────────────────────────────
        dp = diff_label_consecutive_pairs(sections)
        for ci, ti in dp:
            if not valid(ci) or not valid(ti):
                stats["skipped_short"] += 1
                continue
            cf_name, cn, ctx_arr = sec_files[ci]
            tf_name, tn, tgt_arr = sec_files[ti]
            pair_rows.append({
                "pair_id":         pair_id,
                "piece_id":        pid,
                "split":           split,
                "pair_type":       "diff",
                "context_file":    cf_name,
                "target_file":     tf_name,
                "context_label":   sections[ci][0],
                "target_label":    sections[ti][0],
                "context_bars":    f"{sections[ci][1]}-{sections[ci][2]}",
                "target_bars":     f"{sections[ti][1]}-{sections[ti][2]}",
                "context_ntokens": cn,
                "target_ntokens":  tn,
                "tgt_key_shift":   _compute_key_shift(ctx_arr, tgt_arr),
                "is_synthetic":    False,
            })
            pair_id += 1
            stats["diff"] += 1

    # ── Synthesize diff pairs for unmatched AA contexts ──────────────────────
    if args.synthesize:
        # Find context_files that already have at least one diff pair
        diff_ctx_files = {r["context_file"] for r in pair_rows if r["pair_type"] == "diff"}
        # Find context_files that have same pairs but no diff pair
        same_ctx_files = {r["context_file"] for r in pair_rows if r["pair_type"] == "same"}
        unmatched_ctx  = same_ctx_files - diff_ctx_files

        # Build index: context_file -> piece_id + section index
        ctx_to_piece = {}
        for pid, (sections, sec_files, split) in piece_sec_files.items():
            for i, entry in sec_files.items():
                if entry is not None:
                    fname, _, _ = entry
                    ctx_to_piece[fname] = (pid, i)

        # For each unmatched AA context, find a diff-label candidate in same piece
        for ctx_fname in sorted(unmatched_ctx):
            if ctx_fname not in ctx_to_piece:
                continue
            pid, ctx_idx = ctx_to_piece[ctx_fname]
            sections, sec_files, split = piece_sec_files[pid]
            ctx_lbl = sections[ctx_idx][0]

            # Find any valid section in same piece with a different label
            candidate = None
            for j, (lbl, bs, be) in enumerate(sections):
                if j == ctx_idx:
                    continue
                if lbl == ctx_lbl:
                    continue
                if sec_files.get(j) is None:
                    continue
                candidate = j
                break   # take the first available diff-label section

            if candidate is None:
                continue

            cf_name, cn, ctx_arr = sec_files[ctx_idx]
            tf_name, tn, tgt_arr = sec_files[candidate]
            pair_rows.append({
                "pair_id":         pair_id,
                "piece_id":        pid,
                "split":           split,
                "pair_type":       "diff",
                "context_file":    cf_name,
                "target_file":     tf_name,
                "context_label":   ctx_lbl,
                "target_label":    sections[candidate][0],
                "context_bars":    f"{sections[ctx_idx][1]}-{sections[ctx_idx][2]}",
                "target_bars":     f"{sections[candidate][1]}-{sections[candidate][2]}",
                "context_ntokens": cn,
                "target_ntokens":  tn,
                "tgt_key_shift":   _compute_key_shift(ctx_arr, tgt_arr),
                "is_synthetic":    True,
            })
            pair_id += 1
            stats["synthetic"] += 1

        print(f"Synthesised {stats['synthetic']} diff pairs for unmatched AA contexts.")

    # ── Filter to matched-only pairs ─────────────────────────────────────────
    if args.matched_only:
        diff_ctx = {r["context_file"] for r in pair_rows if r["pair_type"] == "diff"}
        same_ctx = {r["context_file"] for r in pair_rows if r["pair_type"] == "same"}
        matched_ctx = diff_ctx & same_ctx
        before = len(pair_rows)
        pair_rows = [r for r in pair_rows if r["context_file"] in matched_ctx]
        print(f"--matched_only: kept {len(pair_rows)}/{before} pairs "
              f"({len(matched_ctx)} unique matched contexts).")

        # Re-number pair_ids
        for i, row in enumerate(pair_rows):
            row["pair_id"] = i

    # ── Write pairs.csv ───────────────────────────────────────────────────────
    fieldnames = ["pair_id", "piece_id", "split", "pair_type",
                  "context_file", "target_file",
                  "context_label", "target_label",
                  "context_bars", "target_bars",
                  "context_ntokens", "target_ntokens",
                  "tgt_key_shift", "is_synthetic"]

    if not args.dry_run:
        csv_path = output_dir / "pairs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_rows)
        print(f"Wrote {len(pair_rows)} pairs to: {csv_path}")
        print(f"Section .npy files saved to: {output_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    train_same  = sum(1 for r in pair_rows if r["split"]=="train" and r["pair_type"]=="same")
    val_same    = sum(1 for r in pair_rows if r["split"]=="val"   and r["pair_type"]=="same")
    test_same   = sum(1 for r in pair_rows if r["split"]=="test"  and r["pair_type"]=="same")
    train_diff  = sum(1 for r in pair_rows if r["split"]=="train" and r["pair_type"]=="diff")
    val_diff    = sum(1 for r in pair_rows if r["split"]=="val"   and r["pair_type"]=="diff")
    test_diff   = sum(1 for r in pair_rows if r["split"]=="test"  and r["pair_type"]=="diff")
    n_synthetic = sum(1 for r in pair_rows if r.get("is_synthetic", False))

    print(f"\n{'='*55}")
    print("SECTION PAIR DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Total pairs      : {len(pair_rows)}  "
          f"(same={stats['same']}, diff={stats['diff']}, synthetic={stats['synthetic']})")
    print(f"  Pieces with same : {stats['pieces_with_same']}")
    print(f"  Skipped (stinger): {stats['skipped_stinger']}")
    print(f"  Skipped (<{MIN_TOKENS} tok) : {stats['skipped_short']}")
    if n_synthetic:
        print(f"  Synthetic diff   : {n_synthetic}")
    print()
    print(f"  {'Split':<8} {'same':>6} {'diff':>6} {'total':>7}")
    print(f"  {'-'*30}")
    for sp_name, s, d in [("train", train_same, train_diff),
                           ("val",   val_same,   val_diff),
                           ("test",  test_same,  test_diff)]:
        print(f"  {sp_name:<8} {s:>6} {d:>6} {s+d:>7}")

    if pair_rows:
        ctx_lens = [r["context_ntokens"] for r in pair_rows]
        tgt_lens = [r["target_ntokens"]  for r in pair_rows]
        print(f"\n  Context section tokens: "
              f"min={min(ctx_lens)}  max={max(ctx_lens)}  "
              f"mean={sum(ctx_lens)/len(ctx_lens):.0f}")
        print(f"  Target  section tokens: "
              f"min={min(tgt_lens)}  max={max(tgt_lens)}  "
              f"mean={sum(tgt_lens)/len(tgt_lens):.0f}")


if __name__ == "__main__":
    main()
