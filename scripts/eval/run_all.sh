#!/usr/bin/env bash
# scripts/eval/run_all.sh
#
# Regenerates every number of the paper from the corrected outputs, in dependency
# order: the scoring scripts (read MIDI) first, then the analyses (read CSV), then
# the table and figure makers. Every file these scripts produce in outputs/eval/ is
# written here, either by the script itself or from its printed output, so no saved
# printout goes stale.
#
# After a change to paper_common.py or to a shared measure: copy outputs/eval/ aside,
# run this, and compare each file with the copy (simllama-dev, Rule 6).
#
# --analyses skips the runs that need what the public release does not hold (the
# generated MIDI and the raw listening responses) and starts from the released score
# tables and ratings (docs/release_plan.md, decision 7).
#
# Usage (Git Bash, from the project root; about 15 minutes on CPU):
#     bash scripts/eval/run_all.sh
#     bash scripts/eval/run_all.sh --analyses
set -u
ANALYSES=0
[ "${1:-}" = "--analyses" ] && ANALYSES=1
if [ -n "${PY:-}" ]; then :                          # PY set by the caller
elif [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
else PY="python"; fi
E="scripts/eval"
O="outputs/eval"
LOG="$O/run_logs"
mkdir -p "$LOG"

# analyses that also read the section arrays or the test-1 MIDI, which the public release
# does not hold; --analyses skips them (docs/release_plan.md, decision 7, option B)
NEEDS_MIDI=" eval_copy_matched eval_train_overlap eval_real_gap_cut eval_gap_without_copies
  eval_chroma_overlap eval_composite_vs_single eval_listening_slopes eval_listening_joint
  eval_listening_round2 plot_listening_trials "

# run <script> <file for its printed output> [<args> ...]
run() {
  local s=$1 out=$2
  shift 2
  if [ $ANALYSES -eq 1 ] && [[ " $(echo $NEEDS_MIDI) " == *" $s "* ]]; then
    echo "$s skipped (needs MIDI or section arrays)"; return 0
  fi
  local t0=$(date +%s)
  "$PY" "$E/$s.py" "$@" > "$out" 2> "$LOG/$s.err"
  local rc=$?
  echo "$s -> $out  exit $rc  $(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { echo "stopped: see $LOG/$s.err"; exit $rc; }
}

# scoring (reads MIDI)
if [ $ANALYSES -eq 0 ]; then
run eval_robustness_table    "$O/robustness_table.txt"
run eval_controls            "$LOG/eval_controls.txt"
run eval_controls            "$LOG/eval_controls_qf.txt" --score --quality_filter draw
run eval_copy_rate           "$LOG/eval_copy_rate.txt"
run eval_ablations           "$O/ablations_table.txt"
run eval_mismatched_context  "$LOG/eval_mismatched_context.txt"
run eval_window_10s          "$LOG/eval_window_10s.txt"
run eval_listening_metrics   "$O/listening_metrics.txt"
fi

# analyses (read CSV)
run eval_effect_sizes        "$LOG/eval_effect_sizes.txt"
run eval_mid_label           "$O/mid_label.txt"
run paper_numbers            "$O/paper_numbers.txt"
run eval_copy_matched        "$LOG/eval_copy_matched.txt"
run eval_train_overlap       "$LOG/eval_train_overlap.txt"
run eval_ablations_piece     "$LOG/eval_ablations_piece.txt"
run eval_real_gap_paired     "$LOG/eval_real_gap_paired.txt"
run eval_real_gap_cut        "$LOG/eval_real_gap_cut.txt"
run eval_gap_without_copies  "$LOG/eval_gap_without_copies.txt"
run eval_chroma_overlap      "$LOG/eval_chroma_overlap.txt"
run eval_composite_vs_single "$O/composite_vs_single.txt"
run eval_listening_slopes    "$LOG/eval_listening_slopes.txt"
run eval_listening_joint     "$LOG/eval_listening_joint.txt"
run eval_listening_round2    "$LOG/eval_listening_round2.txt"
# all 26 listeners, the six who failed a check included (sensitivity, review round 8)
if [ $ANALYSES -eq 0 ]; then
run score_listening_ratings  "$LOG/listening_all26_full.txt" --key listening_test/answer_key.csv --responses listening_test/ratings.csv --keep_failed
run eval_listening_metrics   "$LOG/listening_metrics_all26.txt" --keep_failed --out_csv "$O/listening_metrics_all26.csv"
fi

# tables and figures of the paper
run make_label_comparison_table "$O/label_comparison_table.md"
run make_controls_table         "$O/controls_table.md"
run make_controls_table         "$O/controls_qf_table.md" "$O/controls_qf.txt"
run make_listening_tables       "$O/listening_tables.md"
run plot_forest                 "$LOG/plot_forest.txt"
run plot_listening_trials       "$LOG/plot_listening_trials.txt"
echo "done"
