#!/bin/bash
# ============================================================
# Control models, retrained and sampled several times
#
# One job does everything for one model, so it queues once:
#   0. pre-flight: files, Python imports, GPU, a 2-context generation test
#      with the Moonbeam base weights skipped (fails in minutes, not hours)
#   1. train with the training settings (scripts/slurm/train.sh) plus --patience 20
#   2. copy best.pt to $HOME (scratch expires after 60 days of inactivity)
#   3. generate the draws: M3 draws 0-5 (3 labels each); M2 and M4 draws 0-11
#
# Models: M2 no label, no context vector; M3 context vector + permuted labels;
#         M4 context vector, no label (differs from M1 only in the label)
#
# Outputs
#   $SCRATCH/runs/${CMODE}_rerun/best.pt         checkpoint (+ copy in $HOME/checkpoints/)
#   $SCRATCH/outputs/${CMODE}_rerun/*.mid         generated sections
#
# Pre-requisites (already on the cluster from the first training; the pre-flight checks them):
#   $SCRATCH/data/sections/*.npy and pairs.csv
#   $SCRATCH/models/moonbeam_839M.pt
#
# Usage (submit both; they run in parallel):
#   cd $SCRATCH/simllama && mkdir -p logs
#   git pull
#   CMODE=m2 sbatch --job-name=m2 scripts/slurm/train_controls.sh
#   CMODE=m3 sbatch --job-name=m3 scripts/slurm/train_controls.sh
#   CMODE=m4 sbatch --job-name=m4 scripts/slurm/train_controls.sh
#
# Resubmitting the same command resumes: training restarts from scratch only if
# best.pt is missing; generation skips files that exist.
# ============================================================

#SBATCH --job-name=controls
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

set -eo pipefail

SCRATCH=${SCRATCH:?set SCRATCH to the cluster scratch folder}
CMODE=${CMODE:?set CMODE=m2, m3 or m4}
case "$CMODE" in
  m2) N_DRAWS=12 ;;
  m3) N_DRAWS=6 ;;
  m4) N_DRAWS=12 ;;
  *)  echo "CMODE must be m2, m3 or m4, got $CMODE"; exit 1 ;;
esac

PROJECT=$SCRATCH/simllama
SEC=$SCRATCH/data/sections
MODEL=$SCRATCH/models/moonbeam_839M.pt
CONFIG=configs/simllama_839M.json
RUN=$SCRATCH/runs/${CMODE}_rerun
OUT=$SCRATCH/outputs/${CMODE}_rerun
KEEP=$HOME/checkpoints
mkdir -p $SCRATCH/logs $RUN $OUT $KEEP

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train   # before set -u: conda's scripts use unset variables
set -u
cd $PROJECT

echo "=== Control ${CMODE}: train + ${N_DRAWS} draws ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)  Commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- 0. pre-flight ------------------------------------------------------
echo "--- pre-flight ---"
for f in "$SEC/pairs.csv" "$MODEL" "$CONFIG"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
NPY=$(ls $SEC/*.npy | wc -l)
echo "npy files: $NPY"
[ "$NPY" -gt 1000 ] || { echo "too few .npy files in $SEC"; exit 1; }
python -c "import torch, scripts.training.train, scripts.inference.infer_resample; \
assert torch.cuda.is_available(), 'no GPU'; print('imports and GPU ok')"

(while true; do
    echo "-- GPU util $(date) --"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    sleep 600
done) &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null || true' EXIT

# ---- 1. train ------------------------------------------------------------
if [ -f "$RUN/best.pt" ]; then
  echo "--- best.pt exists, skipping training ---"
else
  echo "--- training: $(date) ---"
  python scripts/training/train.py \
      --config_path    $CONFIG \
      --sections_dir   $SEC \
      --pairs_csv      $SEC/pairs.csv \
      --checkpoint     $MODEL \
      --condition_mode $CMODE \
      --use_lora \
      --lora_rank      8 \
      --lora_alpha     16 \
      --context_length 1024 \
      --batch_size     2 \
      --num_epochs     200 \
      --patience       20 \
      --lr             3e-5 \
      --constant_lr \
      --seed           42 \
      --num_workers    2 \
      --save_dir       $RUN \
      --save_every     10
fi
[ -f "$RUN/best.pt" ] || { echo "training produced no best.pt"; exit 1; }

# ---- 2. keep the checkpoint ----------------------------------------------
cp -n "$RUN/best.pt" "$KEEP/${CMODE}_rerun_best.pt" || true
ls -la "$KEEP/${CMODE}_rerun_best.pt"

# ---- 3. generate ---------------------------------------------------------
echo "--- generation test, 2 contexts: $(date) ---"
python scripts/inference/infer_resample.py --model $CMODE \
    --ckpt $RUN/best.pt --config_path $CONFIG \
    --sections_dir $SEC --pairs_csv $SEC/pairs.csv \
    --output_dir $OUT --first_sample 0 --n_samples 1 --limit 2

echo "--- generation, ${N_DRAWS} draws: $(date) ---"
python scripts/inference/infer_resample.py --model $CMODE \
    --ckpt $RUN/best.pt --config_path $CONFIG \
    --sections_dir $SEC --pairs_csv $SEC/pairs.csv \
    --output_dir $OUT --first_sample 0 --n_samples $N_DRAWS \
    --len_multiplier 1.2 --temperature 0.8 --top_p 0.9 --base_seed 42

echo "=== done: $(date) ==="
echo "MIDI files: $(ls $OUT/*.mid 2>/dev/null | wc -l)  (expected about $([ $CMODE = m3 ] && echo $((157*3*N_DRAWS)) || echo $((157*N_DRAWS))))"
echo "Copy back: scp -r <user>@<cluster>:$OUT outputs/generated/"
