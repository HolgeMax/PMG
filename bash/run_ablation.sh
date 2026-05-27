#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --job-name=ablation_study_50_res_raw_corr_postsplit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=12:00:00
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
uv run ablation \
    ablation.device=cuda \
    ablation.checkpoint_dir=results/checkpoints/crossvalidation/resnet_preprocessed_paper_presplit \
    ablation.box_size_frac=0.5
