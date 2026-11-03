#!/bin/bash
# ============================================================
# Matched-triplet inference for ctx-sim evaluation
#
# For each unique context in test split (157 contexts):
#   - M1: generate with sim:high, sim:mid, sim:low (3 outputs)
#   - M2: generate unconditionally (1 output, confound check)
#   - M3: generate with sim:high, sim:mid, sim:low (3 outputs, wrong-label ablation)
# Total: 157 * 7 = 1099 MIDI files
#
# Prerequisite: best.pt for M1, M2, M3 must be in $SCRATCH/runs/
# Runtime estimate: ~90 min on A100 80GB
#
# Usage:
#   cd $SCRATCH/simllama && mkdir -p logs
#   git pull
#   sbatch scripts/slurm/infer_triplet.sh
# ============================================================

#SBATCH --job-name=m1_triplet_raw
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

SCRATCH=${SCRATCH:?set SCRATCH to the cluster scratch folder}
set -eo pipefail   # a failed step fails the job
# Checkpoints: override to use downloaded or differently placed files
CKPT_M1=${CKPT_M1:-$SCRATCH/runs/m1/best.pt}
CKPT_M2=${CKPT_M2:-$SCRATCH/runs/m2/best.pt}
CKPT_M3=${CKPT_M3:-$SCRATCH/runs/m3/best.pt}
PROJECT=$SCRATCH/simllama
SEC=$SCRATCH/data/sections
OUT=$SCRATCH/outputs/m1_triplet_raw

mkdir -p $SCRATCH/logs $OUT

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train

cd $PROJECT

echo "=== Matched-triplet inference ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/inference/infer_triplet.py \
    --ckpt_m1      $CKPT_M1 \
    --ckpt_m2      $CKPT_M2 \
    --ckpt_m3      $CKPT_M3 \
    --config_path  configs/simllama_839M.json \
    --sections_dir $SEC \
    --pairs_csv    $SEC/pairs.csv \
    --output_dir   $OUT \
    --split        test \
    --len_multiplier 1.2 \
    --temperature  0.8 \
    --top_p        0.9 \
    --seed         42

echo "Done: $(date)"
echo "Output dir: $OUT"
echo "File count: $(ls $OUT/*.mid 2>/dev/null | wc -l)"
