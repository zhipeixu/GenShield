#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Training script for GenShield Stage 2 - Combined Tasks
# This stage combines four tasks:
#   1. SynthScars - Anomaly Repair with interleaved text+image output (CE + MSE loss)
#   2. SynthScars Normal - No anomaly, input == output image (CE + MSE loss)
#   3. AIGI-Holmes - AI-generated image detection (VLM, text-only, CE loss)
#   4. SynthScars 0.5to1 - Text output without loss, image output with loss (MSE only)
#
# Key features:
#   - Initialized from a Stage-1 checkpoint
#   - Task 4 emits text but does NOT apply CE loss on it; only MSE on the output image

set -e

# Resolve repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============== Configuration (edit these paths) ==============
RUN_NAME="genshield-stage2-$(date +%m%d)"
MODEL_PATH="/path/to/BAGEL-7B-MoT"                       # base BAGEL-7B-MoT weights
STAGE1_CKPT="/path/to/checkpoints/genshield-stage1/<step>"   # Stage-1 checkpoint dir
CHECKPOINT_DIR="/path/to/checkpoints/genshield-stage2"   # where to save Stage-2 checkpoints
WANDB_PROJECT="GenShield"

DATASET_CONFIG="${REPO_ROOT}/data/configs/stage2.yaml"
TRAIN_ENTRY="${REPO_ROOT}/train/pretrain_unified_navit.py"

# Enable wandb online mode
wandb online

# ============== Run Training ==============
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes="1" \
  --node_rank="0" \
  --nproc_per_node="8" \
  --master_addr="127.0.0.1" \
  --master_port="12345" \
  "${TRAIN_ENTRY}" \
  --dataset_config_file "${DATASET_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from "${STAGE1_CKPT}" \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema False \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 36864 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 10240 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${RUN_NAME}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --ema 0.99 \
  --save_every 100
