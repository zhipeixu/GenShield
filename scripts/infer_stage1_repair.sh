#!/bin/bash
# Stage 1 Repair Inference Script
# Caption-guided anomaly repair: given a (caption, anomalous image) pair,
# the Stage 1 model outputs the repaired image.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============== Configuration (edit these paths) ==============
MODEL_PATH="/path/to/genshield_stage1_checkpoint"   # GenShield Stage-1 checkpoint dir
TEST_JSONL="/path/to/test_resize.jsonl"             # each line: image_path/before_edit + caption
OUTPUT_DIR="./results/stage1_repair"

# ============== Inference parameters ==============
CFG_TEXT_SCALE=3.0
CFG_IMG_SCALE=1.5
NUM_TIMESTEPS=50
TIMESTEP_SHIFT=3.0
CFG_INTERVAL_MIN=0.4
CFG_INTERVAL_MAX=1.0

# ============== Run inference ==============
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/inference/infer_stage1_repair.py" \
    --model_path "${MODEL_PATH}" \
    --test_jsonl "${TEST_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --cfg_text_scale ${CFG_TEXT_SCALE} \
    --cfg_img_scale ${CFG_IMG_SCALE} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --timestep_shift ${TIMESTEP_SHIFT} \
    --cfg_interval_min ${CFG_INTERVAL_MIN} \
    --cfg_interval_max ${CFG_INTERVAL_MAX} \
    --save_input
