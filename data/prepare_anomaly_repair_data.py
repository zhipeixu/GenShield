#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本 - 将图像异常修复数据转换为Parquet格式

使用方法:
python prepare_anomaly_repair_data.py \
    --anomaly_dir /path/to/anomaly_images \
    --repaired_dir /path/to/repaired_images \
    --descriptions /path/to/descriptions.json \
    --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def load_descriptions(descriptions_path):
    """
    加载异常描述文件
    
    支持格式:
    1. JSON: {"image_name.jpg": "description text", ...}
    2. JSONL: {"image": "image_name.jpg", "description": "text"} per line
    3. CSV: columns: image_name, description
    """
    ext = os.path.splitext(descriptions_path)[1].lower()
    
    if ext == '.json':
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
    
    elif ext == '.jsonl':
        descriptions = {}
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                descriptions[data['image']] = data['description']
    
    elif ext == '.csv':
        df = pd.read_csv(descriptions_path)
        descriptions = {row['image_name']: row['description'] 
                       for _, row in df.iterrows()}
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return descriptions


def create_parquet_dataset(
    anomaly_dir,
    repaired_dir,
    descriptions,
    output_dir,
    samples_per_file=1000,
    image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')
):
    """
    创建Parquet数据集
    
    Args:
        anomaly_dir: 异常图片目录
        repaired_dir: 修复后图片目录
        descriptions: 字典 {image_name: description}
        output_dir: 输出目录
        samples_per_file: 每个parquet文件的样本数
        image_extensions: 支持的图片格式
    """
    os.makedirs(output_dir, exist_ok=True)
    parquet_dir = os.path.join(output_dir, 'parquet')
    os.makedirs(parquet_dir, exist_ok=True)
    
    # 收集所有有效的样本
    valid_samples = []
    skipped = 0
    
    print("扫描数据集...")
    for img_name in tqdm(descriptions.keys()):
        # 检查异常图片
        anomaly_path = os.path.join(anomaly_dir, img_name)
        if not os.path.exists(anomaly_path):
            # 尝试不同的扩展名
            found = False
            base_name = os.path.splitext(img_name)[0]
            for ext in image_extensions:
                test_path = os.path.join(anomaly_dir, base_name + ext)
                if os.path.exists(test_path):
                    anomaly_path = test_path
                    found = True
                    break
            if not found:
                print(f"Warning: 异常图片未找到 {img_name}")
                skipped += 1
                continue
        
        # 检查修复后图片
        repaired_path = os.path.join(repaired_dir, img_name)
        if not os.path.exists(repaired_path):
            # 尝试不同的扩展名
            found = False
            base_name = os.path.splitext(img_name)[0]
            for ext in image_extensions:
                test_path = os.path.join(repaired_dir, base_name + ext)
                if os.path.exists(test_path):
                    repaired_path = test_path
                    found = True
                    break
            if not found:
                print(f"Warning: 修复图片未找到 {img_name}")
                skipped += 1
                continue
        
        valid_samples.append({
            'image_name': img_name,
            'anomaly_path': anomaly_path,
            'repaired_path': repaired_path,
            'description': descriptions[img_name]
        })
    
    print(f"\n找到 {len(valid_samples)} 个有效样本，跳过 {skipped} 个")
    
    if len(valid_samples) == 0:
        print("错误: 没有找到有效样本!")
        return None
    
    # 分批写入parquet文件
    parquet_files = []
    parquet_info = {}
    
    num_files = (len(valid_samples) + samples_per_file - 1) // samples_per_file
    print(f"\n将创建 {num_files} 个parquet文件...")
    
    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min(start_idx + samples_per_file, len(valid_samples))
        batch_samples = valid_samples[start_idx:end_idx]
        
        # 读取图片并转换为bytes
        data = []
        print(f"\n处理文件 {file_idx + 1}/{num_files}...")
        for sample in tqdm(batch_samples):
            try:
                # 读取异常图片
                with open(sample['anomaly_path'], 'rb') as f:
                    anomaly_bytes = f.read()
                
                # 读取修复后图片
                with open(sample['repaired_path'], 'rb') as f:
                    repaired_bytes = f.read()
                
                # 验证图片可以打开
                Image.open(sample['anomaly_path']).convert('RGB')
                Image.open(sample['repaired_path']).convert('RGB')
                
                data.append({
                    'anomaly_image': anomaly_bytes,
                    'repaired_image': repaired_bytes,
                    'anomaly_description': sample['description']
                })
            except Exception as e:
                print(f"Error processing {sample['image_name']}: {e}")
                continue
        
        if len(data) == 0:
            print(f"Warning: 文件 {file_idx} 没有有效数据")
            continue
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        parquet_path = os.path.join(parquet_dir, f'anomaly_repair_{file_idx:04d}.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
        # 记录parquet信息
        parquet_table = pq.ParquetFile(parquet_path)
        num_row_groups = parquet_table.num_row_groups
        
        parquet_files.append(parquet_path)
        parquet_info[parquet_path] = {
            'num_row_groups': num_row_groups,
            'num_rows': len(df)
        }
        
        print(f"已保存: {parquet_path} ({len(df)} samples, {num_row_groups} row groups)")
    
    # 保存parquet信息
    info_dir = os.path.join(output_dir, 'parquet_info')
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, 'anomaly_repair.json')
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(parquet_info, f, indent=2, ensure_ascii=False)
    
    # 保存统计信息
    stats = {
        'total_samples': len(valid_samples),
        'num_parquet_files': len(parquet_files),
        'samples_per_file': samples_per_file,
        'skipped_samples': skipped,
        'total_row_groups': sum(info['num_row_groups'] for info in parquet_info.values())
    }
    
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("数据集创建完成!")
    print("="*60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"Parquet文件数: {stats['num_parquet_files']}")
    print(f"总Row Groups: {stats['total_row_groups']}")
    print(f"Parquet目录: {parquet_dir}")
    print(f"Info JSON: {info_path}")
    print(f"统计信息: {stats_path}")
    print("="*60)
    
    return parquet_info


def main():
    parser = argparse.ArgumentParser(
        description='将图像异常修复数据转换为Parquet格式'
    )
    parser.add_argument(
        '--anomaly_dir',
        type=str,
        required=True,
        help='异常图片目录路径'
    )
    parser.add_argument(
        '--repaired_dir',
        type=str,
        required=True,
        help='修复后图片目录路径'
    )
    parser.add_argument(
        '--descriptions',
        type=str,
        required=True,
        help='异常描述文件路径 (JSON/JSONL/CSV)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--samples_per_file',
        type=int,
        default=1000,
        help='每个parquet文件的样本数 (默认: 1000)'
    )
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.anomaly_dir):
        print(f"错误: 异常图片目录不存在: {args.anomaly_dir}")
        return
    
    if not os.path.exists(args.repaired_dir):
        print(f"错误: 修复图片目录不存在: {args.repaired_dir}")
        return
    
    if not os.path.exists(args.descriptions):
        print(f"错误: 描述文件不存在: {args.descriptions}")
        return
    
    # 加载描述
    print(f"加载描述文件: {args.descriptions}")
    descriptions = load_descriptions(args.descriptions)
    print(f"加载了 {len(descriptions)} 个描述")
    
    # 创建数据集
    create_parquet_dataset(
        anomaly_dir=args.anomaly_dir,
        repaired_dir=args.repaired_dir,
        descriptions=descriptions,
        output_dir=args.output_dir,
        samples_per_file=args.samples_per_file
    )


if __name__ == '__main__':
    main()
