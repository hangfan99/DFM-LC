#!/bin/bash
# Minimal Slurm launcher for the paper DFM warm-start / LC / MC experiments.

#SBATCH -N 1
#SBATCH --gres=gpu:4

gpus=${GPUS:-4}
cpus=${CPUS_PER_TASK:-8}
cfg=${CFG:-configs/DFM-warm-start.yaml}
desc=${DESC:-DFM-warm-start}

torchrun \
  --nproc_per_node "${gpus}" \
  train.py \
  -c "${cfg}" \
  --world_size "${gpus}" \
  --per_cpus "${cpus}" \
  --desc "${desc}"
