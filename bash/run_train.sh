#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --job-name=pre_paper_presplit
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

#SBATCH --mem=16000M
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Project root
PROJECT_DIR="$HOME/PMG"
cd "$PROJECT_DIR" || exit 1

# ── Env
export PATH="$HOME/.local/bin:$PATH"   # puts uv in PATH

echo "Job:  $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# ── Run
uv run crossval \
    -m model.name=resnet101,densenet201 \
    data_loader.train_raw=false \
    data_loader.pmg_negative_mode=paper \
    data_loader.balance_mode=pre_split \
