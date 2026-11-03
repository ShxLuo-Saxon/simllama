#!/bin/bash
# ============================================================
# Training -- Similarity-Bucket Conditioning
# Sequence: [SOS][ctx][SOC][sim:high/mid/low][PROTO...][tgt][EOC]
# Backbone frozen; trains supplementary_embedding + proto_proj + LoRA Q/V
# 839M model, sae partition (A100 80GB), 200 epochs
#
# Pre-requisites (run once on HPC before sbatch):
#   1. scp sections/ dir to $SCRATCH/data/sections/
#      Must include pairs.csv (built by build_pairs.py locally)
#   2. scp moonbeam_839M.pt to $SCRATCH/models/  (if not already there)
#
# Build pairs.csv locally first:
#   .venv\Scripts\python.exe scripts\data\build_pairs.py \
#       --sections_dir "data/sections" \
#       --metadata     "data/metadata.xlsx"
#   scp sections/pairs.csv <user>@<cluster>:$SCRATCH/data/sections/
#
# Usage:
#   cd $SCRATCH/simllama && mkdir -p logs
#   git pull
#   sbatch scripts/slurm/train.sh             # M1 (default)
#   CMODE=m2 sbatch scripts/slurm/train.sh   # M2
#   CMODE=m3 sbatch scripts/slurm/train.sh   # M3
# ============================================================

#SBATCH --job-name=m1
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

SCRATCH=${SCRATCH:?set SCRATCH to the cluster scratch folder}
set -eo pipefail   # a failed step fails the job
CMODE=${CMODE:-m1}

mkdir -p $SCRATCH/logs $SCRATCH/runs/${CMODE}
PROJECT=$SCRATCH/simllama
SEC=$SCRATCH/data/sections
MODEL=$SCRATCH/models/moonbeam_839M.pt

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train

cd $PROJECT

echo "=== Training condition_mode=${CMODE} ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

(while true; do
    echo "-- GPU util $(date) --"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    sleep 300
done) &
GPU_LOG_PID=$!

python scripts/training/train.py \
    --config_path   configs/simllama_839M.json \
    --sections_dir  $SEC \
    --pairs_csv     $SEC/pairs.csv \
    --checkpoint    $MODEL \
    --condition_mode $CMODE \
    --use_lora \
    --lora_rank      8 \
    --lora_alpha     16 \
    --context_length 1024 \
    --batch_size     2 \
    --num_epochs     200 \
    --lr             3e-5 \
    --constant_lr \
    --num_workers    8 \
    --save_dir       $SCRATCH/runs/${CMODE} \
    --save_every     10

kill $GPU_LOG_PID 2>/dev/null

echo "Training ${CMODE} done: $(date)"
echo "Best checkpoint: $SCRATCH/runs/${CMODE}/best.pt"
