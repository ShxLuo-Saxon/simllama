#!/bin/bash
# ============================================================
# Generation-stability check -- repeated M1 sampling
#
# For each context in the test split (157 contexts):
#   M1 x sim:high / sim:mid / sim:low x 5 new samples
# Total: 157 * 3 * 5 = 2355 MIDI files
#
# Prerequisites (see docs/hpc_guide.md):
#   $SCRATCH/models/simllama_m1.pt   (M1 best.pt, epoch 11)
#   $SCRATCH/data/sections/                 (*.npy + pairs.csv)
# Runtime estimate: ~3.5 h on A100 (the 1099-file triplet run took ~90 min)
# Resumable: resubmit and existing files are skipped.
#
# Usage:
#   cd $SCRATCH/simllama && mkdir -p logs
#   git pull
#   sbatch scripts/slurm/infer_resample.sh
# ============================================================

#SBATCH --job-name=m1_resample
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

SCRATCH=${SCRATCH:?set SCRATCH to the cluster scratch folder}
set -eo pipefail   # a failed step fails the job
CKPT_M1=${CKPT_M1:-$SCRATCH/models/simllama_m1.pt}
PROJECT=$SCRATCH/simllama
SEC=$SCRATCH/data/sections
OUT=$SCRATCH/outputs/m1_resample

mkdir -p $SCRATCH/logs $OUT

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train

cd $PROJECT

echo "=== M1 resample (generation stability) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/inference/infer_resample.py --model m1 \
    --ckpt_m1      $CKPT_M1 \
    --config_path  configs/simllama_839M.json \
    --sections_dir $SEC \
    --pairs_csv    $SEC/pairs.csv \
    --output_dir   $OUT \
    --n_samples    5 \
    --len_multiplier 1.2 \
    --temperature  0.8 \
    --top_p        0.9 \
    --base_seed    42

echo "Done: $(date)"
echo "File count: $(ls $OUT/*.mid 2>/dev/null | wc -l)"
