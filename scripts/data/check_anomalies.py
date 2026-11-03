"""
Anomaly check for processed .npy dataset files.

Scans all [id].npy files in the processed directory and reports:
  1. Long silence gaps  — gap between consecutive notes > SILENCE_THRESHOLD seconds
  2. Very short pieces  — fewer than MIN_TOKENS compound tokens
  3. Very long pieces   — more than MAX_TOKENS compound tokens
  4. Condition misalign — [id]_condition.npy row count not even (should be N_bars * 2)
  5. Token range issues — any col-0 value outside expected negative-ID or onset range

Outputs:
  - Console summary table
  - CSV report saved to OUTPUT_CSV

Usage:
    conda run -n moonbeam_test python scripts/data/check_anomalies.py
    conda run -n moonbeam_test python scripts/data/check_anomalies.py --data_dir <path>
"""

import os
import argparse
import pathlib
import numpy as np
import pandas as pd

DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[2]
                                 / "data" / "ninsheetmusic"))  # scores, metadata, processed/

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = pathlib.Path(
    DATA_ROOT / "processed" / "npy" / "processed"
)
OUTPUT_CSV = pathlib.Path(
    DATA_ROOT / "processed" / "npy" / "anomaly_report.csv"
)

TIME_RESOLUTION = 100          # ticks per second (matches MusicTokenizer)
SILENCE_THRESHOLD_SEC = 8.0    # flag if any inter-note silence gap > 8 seconds
SILENCE_THRESHOLD_TICKS = int(SILENCE_THRESHOLD_SEC * TIME_RESOLUTION)

MIN_TOKENS = 10    # flag if piece has fewer compound tokens than this
MAX_TOKENS = 5000  # flag if piece has more compound tokens than this
MAX_DUR_TICKS = 1023   # tokenizer vocab covers 0-1023; flag anything above this


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_music_npy(path: pathlib.Path) -> dict:
    """
    Load [id].npy (N, 6) and check for anomalies.

    Compound token columns:
        col 0 = onset (ticks at TIME_RESOLUTION ticks/sec)
        col 1 = duration (ticks)
        cols 2-5 = octave, pitch, instrument, velocity

    Returns a dict of findings (empty = no issues).
    """
    issues = {}
    try:
        tokens = np.load(str(path))  # (N, 6)
    except Exception as e:
        issues["load_error"] = str(e)
        return issues

    n = len(tokens)
    issues["n_tokens"] = n

    if n < MIN_TOKENS:
        issues["too_short"] = n
    if n > MAX_TOKENS:
        issues["too_long"] = n

    if n < 2:
        return issues

    onsets = tokens[:, 0].astype(np.int64)
    durs   = tokens[:, 1].astype(np.int64)

    # Duration out of tokenizer vocab range
    max_dur = int(durs.max())
    if max_dur > MAX_DUR_TICKS:
        worst_dur_idx = int(durs.argmax())
        issues["max_dur_ticks"]    = max_dur
        issues["max_dur_sec"]      = round(max_dur / TIME_RESOLUTION, 1)
        issues["max_dur_token_idx"] = worst_dur_idx
        issues["n_out_of_range_dur"] = int((durs > MAX_DUR_TICKS).sum())

    # Silence gap = next onset - (current onset + current duration)
    # Positive = silence; negative = overlap (normal for polyphony)
    note_ends   = onsets[:-1] + durs[:-1]
    next_onsets = onsets[1:]
    gaps        = next_onsets - note_ends

    max_silence_ticks = int(gaps.max())
    max_silence_sec   = max_silence_ticks / TIME_RESOLUTION

    if max_silence_ticks > SILENCE_THRESHOLD_TICKS:
        # Find position of worst gap
        worst_idx = int(gaps.argmax())
        issues["max_silence_sec"]   = round(max_silence_sec, 1)
        issues["max_silence_ticks"] = max_silence_ticks
        issues["gap_at_token_idx"]  = worst_idx
        issues["gap_at_onset_tick"] = int(onsets[worst_idx])
        issues["gap_at_onset_sec"]  = round(int(onsets[worst_idx]) / TIME_RESOLUTION, 1)

    return issues


def check_condition_npy(path: pathlib.Path) -> dict:
    """
    Load [id]_condition.npy (N_bars * 2, 6) and check for structural issues.
    """
    issues = {}
    if not path.exists():
        issues["condition_missing"] = True
        return issues

    try:
        cond = np.load(str(path))
    except Exception as e:
        issues["condition_load_error"] = str(e)
        return issues

    n_rows = len(cond)
    issues["condition_rows"] = n_rows

    if n_rows % 2 != 0:
        issues["condition_odd_rows"] = n_rows  # should always be even (2 per bar)

    # Check token IDs in col 0 are within expected negative ranges
    # func token IDs: -26 to -32
    # sec  token IDs: -33 to -59
    token_ids = cond[:, 0]
    func_rows = token_ids[0::2]  # even rows
    sec_rows  = token_ids[1::2]  # odd rows

    bad_func = func_rows[(func_rows < -32) | (func_rows > -26)]
    bad_sec  = sec_rows[ (sec_rows  < -59) | (sec_rows  > -33)]

    if len(bad_func) > 0:
        issues["bad_func_tokens"] = bad_func.tolist()
    if len(bad_sec) > 0:
        issues["bad_sec_tokens"] = bad_sec.tolist()

    return issues


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check .npy dataset files for anomalies.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_csv", type=str, default=str(OUTPUT_CSV))
    parser.add_argument(
        "--silence_threshold", type=float, default=SILENCE_THRESHOLD_SEC,
        help="Flag files with silence gap > this many seconds (default: 5.0)"
    )
    args = parser.parse_args()

    data_dir   = pathlib.Path(args.data_dir)
    output_csv = pathlib.Path(args.output_csv)
    silence_thr = int(args.silence_threshold * TIME_RESOLUTION)

    music_files = sorted(data_dir.glob("*.npy"))
    music_files = [f for f in music_files if not f.stem.endswith("_condition") and
                                             not f.stem.endswith("_condition_m3")]
    print(f"Found {len(music_files)} music .npy files in {data_dir}")

    rows = []
    n_flagged = 0

    for mf in music_files:
        file_id = mf.stem  # e.g. "00536"
        cond_path = mf.parent / f"{file_id}_condition.npy"

        music_issues = check_music_npy(mf)
        cond_issues  = check_condition_npy(cond_path)

        has_issue = (
            "load_error"         in music_issues or
            "too_short"          in music_issues or
            "too_long"           in music_issues or
            "max_dur_ticks"      in music_issues or
            "max_silence_sec"    in music_issues or
            "condition_missing"    in cond_issues or
            "condition_load_error" in cond_issues or
            "condition_odd_rows"   in cond_issues or
            "bad_func_tokens"      in cond_issues or
            "bad_sec_tokens"       in cond_issues
        )

        row = {"file_id": file_id, "flagged": has_issue}
        row.update({f"music_{k}": v for k, v in music_issues.items()})
        row.update({f"cond_{k}":  v for k, v in cond_issues.items()})
        rows.append(row)

        if has_issue:
            n_flagged += 1

    df = pd.DataFrame(rows)
    flagged_df = df[df["flagged"]].copy()

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANOMALY REPORT  --  {len(music_files)} files scanned")
    print(f"{'='*60}")
    print(f"Flagged: {n_flagged} / {len(music_files)}")

    if n_flagged > 0:
        print(f"\nFlagged files:")
        for _, r in flagged_df.iterrows():
            flags = []
            if "music_max_dur_ticks" in r and pd.notna(r.get("music_max_dur_ticks")):
                flags.append(f"long_note={r['music_max_dur_sec']}s ({int(r['music_max_dur_ticks'])} ticks, {int(r['music_n_out_of_range_dur'])} notes)")
            if "music_max_silence_sec" in r and pd.notna(r.get("music_max_silence_sec")):
                flags.append(f"silence={r['music_max_silence_sec']}s at token {r.get('music_gap_at_token_idx')} (onset {r.get('music_gap_at_onset_sec')}s)")
            if "music_too_short" in r and pd.notna(r.get("music_too_short")):
                flags.append(f"too_short={r['music_too_short']} tokens")
            if "music_too_long" in r and pd.notna(r.get("music_too_long")):
                flags.append(f"too_long={r['music_too_long']} tokens")
            if "cond_condition_missing" in r and r.get("cond_condition_missing"):
                flags.append("condition_missing")
            if "cond_condition_odd_rows" in r and pd.notna(r.get("cond_condition_odd_rows")):
                flags.append(f"odd_condition_rows={r['cond_condition_odd_rows']}")
            if "cond_bad_func_tokens" in r and r.get("cond_bad_func_tokens") is not None:
                flags.append(f"bad_func_tokens")
            if "cond_bad_sec_tokens" in r and r.get("cond_bad_sec_tokens") is not None:
                flags.append(f"bad_sec_tokens")
            print(f"  {r['file_id']}: {' | '.join(flags)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nFull report saved to: {output_csv}")
    print(f"(Filter column 'flagged=True' for anomalies only)")


if __name__ == "__main__":
    main()
