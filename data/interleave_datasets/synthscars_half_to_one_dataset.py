# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from ..data_utils import pil_img2rgb
from ..distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SynthScarsHalfToOneIterableDataset(DistributedIterableDataset):
    """
    Dataset for SynthScars 0.5 to 1 task - Text output without loss, Image output with loss.
    
    Training Flow:
    Input:
        1. Anomalous image (VIT + VAE encoding) - before_edit
        2. System prompt
    
    Output:
        1. Caption text (NO CE loss) - allowed but not supervised
        2. Repaired image (MSE loss) - after_edit
    
    Expected JSONL format:
    {
        "before_edit": "/path/to/anomaly_image.png",
        "after_edit": "/path/to/repaired_image.png",
        "caption": "anomaly description text"
    }
    """

    def __init__(
        self, 
        dataset_name, 
        transform, 
        tokenizer, 
        vit_transform,
        jsonl_path_list, 
        num_used_data,
        local_rank=0, 
        world_size=1, 
        num_workers=8, 
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
        system_prompt=None,
        repair_trigger=None,
        description_prefix=None,
        data_dir_list=None,  # Accept but ignore this parameter
        **kwargs  # Accept any additional unexpected parameters
    ):
        """
        Args:
            dataset_name: Name of the dataset
            transform: Image transform for VAE
            tokenizer: Text tokenizer
            vit_transform: Image transform for VIT
            jsonl_path_list: List of JSONL file paths
            num_used_data: List of number of samples to use from each JSONL
            local_rank: Local rank for distributed training
            world_size: World size for distributed training
            num_workers: Number of data loading workers
            data_status: Resume status
            shuffle_lines: Whether to shuffle the data
            shuffle_seed: Random seed for shuffling
            system_prompt: Custom system prompt (optional)
            repair_trigger: Custom repair trigger (optional)
            description_prefix: Prefix for the anomaly description output (optional)
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()
        
        # System prompt
        if system_prompt is None:
            self.system_prompt = (
                "You are an image anomaly repair assistant. This is an AI-generated image, but it has some defects and anomalies. Please fix these abnormal areas so that it looks as realistic and natural as possible."
            )
        else:
            self.system_prompt = system_prompt
        
        # Prefix for the model's output description
        if description_prefix is None:
            self.description_prefix = "\n\nThe following are some areas where problems exist:\n\n"
        else:
            self.description_prefix = description_prefix
        
        # Repair trigger that comes after the description
        if repair_trigger is None:
            self.repair_trigger = (
                "\n\n<repair>Check the abnormal area in the image and generate the image "
                "with the abnormality repaired.</repair>"
            )
        else:
            self.repair_trigger = repair_trigger

    def get_data_paths(
        self, 
        jsonl_path_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        """Load and optionally shuffle data paths from JSONL files."""
        data_paths = []
        for jsonl_path, num_data_point in zip(jsonl_path_list, num_used_data):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                raw_data = f.readlines()
            
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            
            raw_data = raw_data[:num_data_point]
            data_paths.extend(raw_data)
        
        return data_paths

    def _init_data(self):
        """Initialize data structure."""
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        """Add text to the sequence."""
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        """Add image to the sequence."""
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def parse_line(self, line):
        """
        Parse a single line from the JSONL file.
        
        Sequence:
        [Input - No Loss]
        1. Anomalous image (VIT + VAE) - before_edit
        2. System prompt
        
        [Output]
        3. Description text (NO CE loss) - caption with prefix and trigger
        4. Repaired image (MSE loss) - after_edit
        
        Returns:
            data: dict with sequence_plan, text_ids_list, image_tensor_list, num_tokens
        """
        data_item = json.loads(line)
        
        # Extract paths and caption
        before_edit_path = data_item['before_edit']
        after_edit_path = data_item['after_edit']
        caption = data_item.get('caption', '')
        
        # Load images
        try:
            anomaly_image = pil_img2rgb(Image.open(before_edit_path))
            repaired_image = pil_img2rgb(Image.open(after_edit_path))
        except Exception as e:
            print(f"Error loading images: {e}")
            print(f"  before_edit: {before_edit_path}")
            print(f"  after_edit: {after_edit_path}")
            raise
        
        # Initialize data structure
        data = self._init_data()
        
        # ============ INPUT SECTION (No Loss) ============
        
        # Step 1: Add the anomalous image (input, no loss)
        # This image needs both VAE encoding (for condition) and VIT encoding (for understanding)
        data = self._add_image(
            data, 
            anomaly_image,
            need_loss=False,  # No loss for input image
            need_vae=True,    # Encode with VAE for conditioning
            need_vit=True,    # Encode with VIT for visual understanding
        )
        
        # Step 2: Add the system prompt (no loss)
        data = self._add_text(data, self.system_prompt, need_loss=False)
        
        # ============ OUTPUT SECTION ============
        
        # Step 3: Add the text output (NO CE loss)
        # Text is allowed but NOT supervised
        if caption:
            output_text = self.description_prefix + caption + self.repair_trigger
        else:
            output_text = self.repair_trigger
        data = self._add_text(data, output_text, need_loss=False)  # NO CE loss on text
        
        # Step 4: Add the repaired image (MSE loss)
        # This is the target image that the model should generate
        data = self._add_image(
            data, 
            repaired_image,
            need_loss=True,   # Apply MSE loss on this image
            need_vae=False,   # Don't encode with VAE (it's the generation target)
            need_vit=False,   # Don't encode with VIT
        )
        
        return data

    def __iter__(self):
        """Iterate over the dataset."""
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            
            for row_idx, line in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data = self.parse_line(line)
                    
                    # Validate that we have loss (image loss)
                    has_loss = [item['loss'] for item in data['sequence_plan']]
                    if sum(has_loss) == 0:
                        print(f'No loss defined, skipped.')
                        continue
                    
                    yield dict(
                        image_tensor_list=data['image_tensor_list'],
                        text_ids_list=data['text_ids_list'],
                        sequence_plan=data['sequence_plan'],
                        num_tokens=data['num_tokens'],
                        data_indexes={
                            "data_indexes": row_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                
                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    traceback.print_exc()
                    continue

            # Repeat dataset
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
