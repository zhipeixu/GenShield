# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2 Inference Script for Anomaly Repair

This script performs batch inference for the Stage 2 model which:
1. Takes an anomalous image as input
2. Generates anomaly description text
3. Generates repaired image

Key points:
- Uses <repair> as the trigger for image generation
- Model outputs interleaved text and image
- Text saved to JSONL, images saved to folder
"""

import os
import json
from copy import deepcopy
from typing import Optional
from datetime import datetime

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
import random
import numpy as np


class Stage2Inferencer:
    """
    Inferencer for Stage 2 model that generates interleaved text and image output.
    
    The model first generates anomaly description text, then generates repaired image
    after seeing the <repair> trigger.
    """
    
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
        # System prompt matching training
        self.system_prompt = (
            "You are an image anomaly repair assistant. This is an AI-generated image, "
            "but it has some defects and anomalies. Please fix these abnormal areas so "
            "that it looks as realistic and natural as possible."
        )
        
        # Repair trigger - this is used during training to signal image generation
        self.repair_trigger = (
            "\n\n<repair>Check the abnormal area in the image and generate the image "
            "with the abnormality repaired.</repair>"
        )
    
    def init_gen_context(self):
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def inference_stage2(
        self,
        image: Image.Image,
        max_text_tokens: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: tuple = (0.0, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        enable_taylorseer: bool = False,
    ):
        """
        Stage 2 inference: Image -> Text Description + Repaired Image
        
        Flow:
        1. Input: anomalous image (VIT + VAE) + system prompt
        2. Model generates: anomaly description text
        3. Add repair trigger
        4. Model generates: repaired image
        
        Returns:
            dict: {'text': generated_description, 'image': repaired_image}
        """
        gen_context = self.init_gen_context()
        cfg_img_context = deepcopy(gen_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Step 1: Process input image (both VAE and VIT encoding, matching training)
            image_processed = self.vae_transform.resize_transform(pil_img2rgb(image))
            image_shape = image_processed.size[::-1]  # (H, W)
            
            # Update context with image (VAE + VIT)
            gen_context = self.update_context_image(image_processed, gen_context, vae=True, vit=True)
            cfg_img_context = self.update_context_image(image_processed, cfg_img_context, vae=True, vit=True)
            
            # Step 2: Add system prompt
            gen_context = self.update_context_text(self.system_prompt, gen_context)
            cfg_img_context = self.update_context_text(self.system_prompt, cfg_img_context)
            
            # Save context for text CFG (before generating text)
            cfg_text_context = deepcopy(gen_context)
            
            # Step 3: Generate anomaly description text
            generated_text = self.gen_text(
                gen_context, 
                max_length=max_text_tokens, 
                do_sample=do_sample, 
                temperature=text_temperature
            )
            
            # Step 4: Update context with generated text
            gen_context = self.update_context_text(generated_text, gen_context)
            cfg_img_context = self.update_context_text(generated_text, cfg_img_context)
            
            # Step 5: Add repair trigger and update context
            gen_context = self.update_context_text(self.repair_trigger, gen_context)
            cfg_img_context = self.update_context_text(self.repair_trigger, cfg_img_context)
            
            # Step 6: Generate repaired image
            repaired_image = self.gen_image(
                image_shape,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
            )
        
        return {
            'text': generated_text,
            'image': repaired_image
        }


def load_model(model_path: str, checkpoint_path: str, max_mem_per_gpu: str = "80GiB"):
    """Load model and create inferencer."""
    
    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Initialize model with empty weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device map
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print("Device map:", device_map)

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load checkpoint
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()
    print('Model loaded')

    # Create inferencer
    inferencer = Stage2Inferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    
    return inferencer


def process_jsonl(
    inferencer: Stage2Inferencer,
    jsonl_path: str,
    output_dir: str,
    output_jsonl_path: str,
    inference_hyper: dict,
    max_samples: Optional[int] = None,
):
    """
    Process JSONL file and run Stage 2 inference.
    
    Args:
        inferencer: Stage2Inferencer instance
        jsonl_path: Path to input JSONL file
        output_dir: Directory to save output images
        output_jsonl_path: Path to save output JSONL with text results
        inference_hyper: Inference hyperparameters
        max_samples: Maximum number of samples to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if max_samples is not None:
        lines = lines[:max_samples]
    
    total = len(lines)
    print(f"Processing {total} samples...")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            sample = json.loads(line)
        except Exception as e:
            print(f"[{idx}/{total}] JSON parse error: {e}")
            continue
        
        # Get image path (use before_edit as input)
        img_path = sample.get("before_edit") or sample.get("image_path") or sample.get("image")
        
        if not img_path:
            print(f"[{idx}/{total}] No image field found")
            continue
        
        if not os.path.exists(img_path):
            print(f"[{idx}/{total}] Image not found: {img_path}")
            continue
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}/{total}] Failed to open image: {img_path}, error={e}")
            continue
        
        print(f"[{idx}/{total}] Processing: {img_path}")
        
        # Run inference
        try:
            with torch.no_grad():
                output = inferencer.inference_stage2(image=image, **inference_hyper)
        except Exception as e:
            print(f"[{idx}/{total}] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        generated_text = output.get('text', '')
        repaired_image = output.get('image')
        
        # Save image
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save input image
        input_save_path = os.path.join(output_dir, f"{base_name}_input.png")
        image.save(input_save_path)
        
        # Save repaired image
        if repaired_image is not None:
            pred_save_path = os.path.join(output_dir, f"{base_name}_repaired.png")
            repaired_image.save(pred_save_path)
            print(f"[{idx}/{total}] Saved: {pred_save_path}")
        else:
            pred_save_path = None
            print(f"[{idx}/{total}] No image generated")
        
        # Record result
        result = {
            "input_image": img_path,
            "input_save_path": input_save_path,
            "repaired_save_path": pred_save_path,
            "generated_text": generated_text,
            "gt_caption": sample.get("caption", ""),
        }
        results.append(result)
        
        # Save JSONL incrementally
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Processing complete. Results saved to {output_jsonl_path}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 batch inference: anomaly description + repaired image."
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the BAGEL-7B-MoT base model directory (contains llm_config.json, "
             "vit_config.json, ae.safetensors, tokenizer files, ...).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained Stage-2 checkpoint (e.g., .../model.safetensors or "
             "the checkpoint directory accepted by accelerate.load_checkpoint_and_dispatch).",
    )
    parser.add_argument(
        "--max_mem_per_gpu",
        type=str,
        default="80GiB",
        help="Per-GPU max memory for accelerate device map.",
    )

    # Data paths
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the input JSONL file (each line: before_edit / image_path / image + caption).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save input/repaired images and results.jsonl.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all).",
    )

    # Inference hyperparameters
    parser.add_argument("--max_text_tokens", type=int, default=1000)
    parser.add_argument("--do_sample", action="store_true",
                        help="If set, sample text tokens instead of greedy decoding.")
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_min", type=float, default=0.0)
    parser.add_argument("--cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")
    parser.add_argument("--enable_taylorseer", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ======================
    # Set random seed
    # ======================
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ======================
    # Output paths
    # ======================
    os.makedirs(args.output_dir, exist_ok=True)
    image_output_dir = os.path.join(args.output_dir, "images")
    output_jsonl_path = os.path.join(args.output_dir, "results.jsonl")

    # Inference hyperparameters
    inference_hyper = dict(
        max_text_tokens=args.max_text_tokens,
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        enable_taylorseer=args.enable_taylorseer,
    )

    # ======================
    # Load model
    # ======================
    print("Loading model...")
    inferencer = load_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
    )

    # ======================
    # Run inference
    # ======================
    print(f"\n==== Stage 2 Batch Inference ====")
    print(f"Input JSONL : {args.jsonl_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"  - images  : {image_output_dir}")
    print(f"  - results : {output_jsonl_path}")

    process_jsonl(
        inferencer=inferencer,
        jsonl_path=args.jsonl_path,
        output_dir=image_output_dir,
        output_jsonl_path=output_jsonl_path,
        inference_hyper=inference_hyper,
        max_samples=args.max_samples,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()
