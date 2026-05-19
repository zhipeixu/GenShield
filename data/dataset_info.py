# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Dataset registry and metadata for GenShield training.

This file declares (1) the mapping from a logical dataset key to its
``IterableDataset`` implementation, and (2) the absolute paths and sample
counts that the training pipeline reads at launch time.

Before running ``scripts/train_stage1.sh`` / ``scripts/train_stage2.sh``,
edit the ``DATASET_INFO`` entries below so that:

* ``jsonl_path`` points to the JSONL annotation file on your machine.
* ``data_dir``  points to the image root used to resolve relative image
  paths inside the JSONL (only used by ``aigi_detection``; the other
  entries embed absolute image paths in the JSONL itself and keep
  ``data_dir`` as ``"/"``).
* ``num_total_samples`` matches the actual line count of the JSONL.

The keys at every level (e.g. ``correction_stage1`` / ``synthscars`` /
``aigi_detection`` / ``aigi_holmes`` / ...) are referenced by
``data/configs/stage1.yaml`` and ``data/configs/stage2.yaml`` and must
not be renamed.
"""

from .interleave_datasets import (
    AIGIDetectionIterableDataset,
    AnomalyRepairJSONLIterableDataset,
    AnomalyRepairStage2IterableDataset,
    SynthScarsHalfToOneIterableDataset,
)


DATASET_REGISTRY = {
    # Stage 1 / Stage 2 correction tasks.
    'correction_stage1':              AnomalyRepairJSONLIterableDataset,
    'correction_stage2_initial':      AnomalyRepairStage2IterableDataset,
    'correction_stage2_terminate':    AnomalyRepairStage2IterableDataset,
    'correction_stage2_intermediate': SynthScarsHalfToOneIterableDataset,
    # Detection task (shared by Stage 1 and Stage 2).
    'aigi_detection':                 AIGIDetectionIterableDataset,
}


DATASET_INFO = {
    # ---------------- Stage 1: instruction-guided correction ----------------
    # Anomaly image + caption (defect description)  ->  repaired image (MSE).
    'correction_stage1': {
        'synthscars': {
            'jsonl_path':        '/path/to/GenShield-Set-Correct/stage1_edit.jsonl',
            'num_total_samples': 11184,
        },
    },

    # ---------------- Stage 2: VCoT initial step (anomalous samples) --------
    # Image -> text (defect description, CE) + repaired image (MSE).
    'correction_stage2_initial': {
        'synthscars': {
            'data_dir':          '/',  # not used; jsonl carries absolute image paths
            'jsonl_path':        '/path/to/GenShield-Set-Correct/stage1_edit.jsonl',
            'num_total_samples': 11184,
        },
    },

    # ---------------- Stage 2: VCoT termination step (normal samples) -------
    # Normal image -> "no anomaly" text (CE) + same image (MSE).
    'correction_stage2_terminate': {
        'synthscars_normal': {
            'data_dir':          '/',  # not used; jsonl carries absolute image paths
            'jsonl_path':        '/path/to/GenShield-Set-Correct/stage2_normal.jsonl',
            'num_total_samples': 7837,
        },
    },

    # ---------------- Stage 2: VCoT intermediate step (curriculum) ----------
    # Half-repaired images produced by the Stage-1 model are used as the
    # starting state and pushed toward the fully-repaired target.
    'correction_stage2_intermediate': {
        'synthscars_half_to_one': {
            'data_dir':          '/',  # not used; jsonl carries absolute image paths
            'jsonl_path':        '/path/to/GenShield-Set-Correct/stage2_half_to_one.jsonl',
            'num_total_samples': 9355,
        },
    },

    # ---------------- Detection Expert (Stage 1 & 2) ------------------------
    # AIGI-Holmes structured detection annotations.
    'aigi_detection': {
        'aigi_holmes': {
            'data_dir':          '/path/to/GenShield-Set-Detect',                  # image root for relative paths
            'jsonl_path':        '/path/to/GenShield-Set-Detect/SFTDATA.jsonl',
            'num_total_samples': 64997,
        },
    },
}
