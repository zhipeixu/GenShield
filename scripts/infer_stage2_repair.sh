#!/bin/bash
# Stage 2 Repair Inference Script
# "Diagnose-then-repair" inference: given an anomalous image, the Stage 2 model
# first generates an anomaly description and then outputs the repaired image.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============== Configuration (edit these paths) ==============
MODEL_PATH="/path/to/BAGEL-7B-MoT"                    # base BAGEL-7B-MoT directory
CHECKPOINT_PATH="/path/to/genshield_stage2/model.safetensors"  # Stage-2 checkpoint
JSONL_PATH="/path/to/test_resize.jsonl"               # each line: before_edit/image_path + caption
OUTPUT_DIR="./results/stage2_repair"
MAX_MEM_PER_GPU="80GiB"

# ============== Inference parameters ==============
MAX_TEXT_TOKENS=1000
TEXT_TEMPERATURE=0.3
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
CFG_INTERVAL_MIN=0.0
CFG_INTERVAL_MAX=1.0
TIMESTEP_SHIFT=3.0
NUM_TIMESTEPS=50
CFG_RENORM_MIN=0.0
CFG_RENORM_TYPE="text_channel"
SEED=42

# ============== Run inference ==============
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/inference/infer_stage2_repair.py" \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_mem_per_gpu "${MAX_MEM_PER_GPU}" \
    --jsonl_path "${JSONL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_text_tokens ${MAX_TEXT_TOKENS} \
    --text_temperature ${TEXT_TEMPERATURE} \
    --cfg_text_scale ${CFG_TEXT_SCALE} \
    --cfg_img_scale ${CFG_IMG_SCALE} \
    --cfg_interval_min ${CFG_INTERVAL_MIN} \
    --cfg_interval_max ${CFG_INTERVAL_MAX} \
    --timestep_shift ${TIMESTEP_SHIFT} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --cfg_renorm_min ${CFG_RENORM_MIN} \
    --cfg_renorm_type ${CFG_RENORM_TYPE} \
    --seed ${SEED}
