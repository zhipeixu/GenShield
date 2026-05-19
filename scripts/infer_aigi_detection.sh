#!/bin/bash
# AIGI Detection Inference Script
# Detect whether an image is AI-generated or real.

set -e

# Resolve repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============== Configuration (edit these paths) ==============
MODEL_PATH="/path/to/BAGEL-7B-MoT"                  # base BAGEL-7B-MoT weights
CHECKPOINT_PATH="/path/to/checkpoint/ema.safetensors"  # GenShield Stage-2 checkpoint
IMAGE_FOLDER="/path/to/test/images"                 # contains 'real' / 'fake' subfolders

# ============== Inference parameters ==============
PROMPT="Please evaluate whether this image is an AI creation or something real, and provide an explanation."
MAX_IMAGES=200
SEED=42

# ============== Run inference ==============
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/inference/infer_aigi_detection.py" \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --prompt "${PROMPT}" \
    --max_images ${MAX_IMAGES} \
    --seed ${SEED}
