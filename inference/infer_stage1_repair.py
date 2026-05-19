#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 Inference Script for Anomaly Repair.

Given a (caption, anomalous image) pair, the Stage 1 model performs
instruction-guided correction and outputs the repaired image.

Usage:
    python inference/infer_stage1_repair.py \
        --model_path /path/to/checkpoint \
        --test_jsonl /path/to/test_resize.jsonl \
        --output_dir ./results/stage1_repair \
        --cfg_text_scale 3.0 \
        --cfg_img_scale 1.5 \
        --num_timesteps 50
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

from modeling.bagel import Bagel, BagelConfig
from modeling.autoencoder import load_ae
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens, pil_img2rgb
from inferencer import InterleaveInferencer


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型从: {model_path}")
    
    # 加载配置
    config = BagelConfig.from_pretrained(model_path)
    
    # 加载模型
    model = Bagel.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()
    
    # 加载VAE
    vae_model = load_ae(os.path.join(model_path, 'vae'))
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    # 加载tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    special_tokens = add_special_tokens(tokenizer)
    
    print("✓ 模型加载成功")
    return model, vae_model, tokenizer, special_tokens


def create_inferencer(model, vae_model, tokenizer, special_tokens):
    """创建推理器"""
    # VAE transform
    vae_transform = ImageTransform(
        image_stride=16,
        max_image_size=1024,
        min_image_size=512
    )
    
    # VIT transform
    vit_transform = ImageTransform(
        image_stride=14,
        max_image_size=518,
        min_image_size=224
    )
    
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=special_tokens
    )
    
    return inferencer


def build_repair_prompt(caption):
    """构建修复prompt"""
    system_prompt = (
        "You are an image anomaly repair assistant. "
        "This is an AI-generated image, but it has some defects and anomalies. "
        "Please fix these abnormal areas so that it looks as realistic and natural as possible. "
        "The following are some areas where problems exist:\n\n"
    )
    
    repair_trigger = (
        "\n\n<repair>Check the abnormal area in the image and generate the image "
        "with the abnormality repaired.</repair>"
    )
    
    full_prompt = system_prompt + caption + repair_trigger
    return full_prompt


def process_test_jsonl(
    inferencer,
    test_jsonl_path,
    output_dir,
    cfg_text_scale=3.0,
    cfg_img_scale=1.5,
    num_timesteps=50,
    timestep_shift=3.0,
    cfg_interval=(0.4, 1.0),
    save_input=True,
):
    """处理测试JSONL文件"""
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repaired_dir = output_dir / "repaired"
    repaired_dir.mkdir(exist_ok=True)
    
    if save_input:
        input_dir = output_dir / "input"
        input_dir.mkdir(exist_ok=True)
    
    # 读取测试数据
    print(f"\n读取测试数据: {test_jsonl_path}")
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"总测试样本数: {len(test_data)}")
    
    # 准备结果记录
    results = []
    
    # 处理每个测试样本
    for idx, item in enumerate(tqdm(test_data, desc="处理测试样本")):
        try:
            # 获取输入图片路径和描述
            # 注意: test_resize.jsonl 中使用 'before_edit' 或 'image_path' 作为输入
            input_image_path = item.get('before_edit') or item.get('image_path')
            caption = item['caption']
            
            if not os.path.exists(input_image_path):
                print(f"\n警告: 图片不存在: {input_image_path}")
                continue
            
            # 加载输入图片
            input_image = Image.open(input_image_path).convert('RGB')
            
            # 构建prompt
            prompt = build_repair_prompt(caption)
            
            # 推理生成修复后的图片
            with torch.no_grad():
                output = inferencer(
                    image=input_image,
                    text=prompt,
                    understanding_output=False,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    num_timesteps=num_timesteps,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                )
            
            repaired_image = output['image']
            
            # 保存结果
            base_name = Path(input_image_path).stem
            output_filename = f"{base_name}_repaired.png"
            output_path = repaired_dir / output_filename
            repaired_image.save(output_path)
            
            # 可选：保存输入图片
            if save_input:
                input_output_path = input_dir / f"{base_name}_input.png"
                input_image.save(input_output_path)
            
            # 记录结果
            result_item = {
                'index': idx,
                'input_path': input_image_path,
                'output_path': str(output_path),
                'caption': caption,
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"\n错误处理样本 {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果记录
    results_json_path = output_dir / "results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 测试完成!")
    print(f"  - 处理样本数: {len(results)}/{len(test_data)}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 修复图片: {repaired_dir}")
    print(f"  - 结果记录: {results_json_path}")


def main():
    parser = argparse.ArgumentParser(description='测试异常修复模型')
    
    # 模型参数
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='训练好的模型checkpoint路径'
    )
    
    # 数据参数
    parser.add_argument(
        '--test_jsonl',
        type=str,
        required=True,
        help='Path to the test JSONL file (each line: image_path / before_edit + caption).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/stage1_repair',
        help='Output directory for repaired images and results.json.'
    )
    
    # 推理参数
    parser.add_argument(
        '--cfg_text_scale',
        type=float,
        default=3.0,
        help='文本CFG scale'
    )
    parser.add_argument(
        '--cfg_img_scale',
        type=float,
        default=1.5,
        help='图像CFG scale'
    )
    parser.add_argument(
        '--num_timesteps',
        type=int,
        default=50,
        help='扩散步数'
    )
    parser.add_argument(
        '--timestep_shift',
        type=float,
        default=3.0,
        help='时间步偏移'
    )
    parser.add_argument(
        '--cfg_interval_min',
        type=float,
        default=0.4,
        help='CFG区间最小值'
    )
    parser.add_argument(
        '--cfg_interval_max',
        type=float,
        default=1.0,
        help='CFG区间最大值'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--save_input',
        action='store_true',
        help='是否保存输入图片'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("异常修复模型测试")
    print("="*60)
    
    # 加载模型
    model, vae_model, tokenizer, special_tokens = load_model(
        args.model_path, 
        device=args.device
    )
    
    # 创建推理器
    inferencer = create_inferencer(
        model, 
        vae_model, 
        tokenizer, 
        special_tokens
    )
    
    # 处理测试数据
    process_test_jsonl(
        inferencer=inferencer,
        test_jsonl_path=args.test_jsonl,
        output_dir=args.output_dir,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        num_timesteps=args.num_timesteps,
        timestep_shift=args.timestep_shift,
        cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
        save_input=args.save_input,
    )


if __name__ == '__main__':
    main()
