import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from inferencer import InterleaveInferencer


def auto_output_path(checkpoint_path: str, image_folder: str) -> str:
    """
    自动生成输出路径：
    checkpoint_path = /a/b/checkpoints/0007200/ema.safetensors
    image_folder = /data/TestSet/Show-o
    输出：/a/b/checkpoints/0007200/result_Show-o.jsonl
    """
    ckpt_dir = os.path.dirname(checkpoint_path)
    folder_name = os.path.basename(os.path.normpath(image_folder))
    return os.path.join(ckpt_dir, f"tuzhan_result_{folder_name}.jsonl")


def main(args):
    # === LLM config ===
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # === ViT config ===
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # === VAE ===
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    # === Bagel config ===
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

    # === 初始化模型结构 (空权重) ===
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # === Tokenizer & Transform ===
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # === Device Map ===
    max_mem_per_gpu = args.max_mem_per_gpu
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device

    # === 加载模型权重 ===
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=args.checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()

    # === 创建推理器 ===
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )

    # === 固定随机种子 ===
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # === 自动生成输出路径 ===
    output_path = args.output_path or auto_output_path(args.checkpoint_path, args.image_folder)
    print(f"📝 Output will be saved to: {output_path}")

    # === 查找 real / fake 子文件夹 ===
    subfolders = [os.path.join(args.image_folder, d) for d in os.listdir(args.image_folder)
                  if os.path.isdir(os.path.join(args.image_folder, d)) and any(x in d.lower() for x in ["real"])]

    if not subfolders:
        raise ValueError("❌ image_folder 下必须包含名为包含 'real' 或 'fake' 的子文件夹")

    # === 收集图像文件 ===
    image_info = []
    for sub in subfolders:
        label = "real" if "real" in sub.lower() else "fake"
        files = sorted([
            os.path.join(sub, f)
            for f in os.listdir(sub)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])[:args.max_images]
        image_info.extend([(f, label) for f in files])

    print(f"📸 Found {len(image_info)} images (real+fake) to process")

    # === 清空旧输出文件 ===
    with open(output_path, "w") as f:
        pass

    correct_count, total_count = 0, 0

    for image_path, label in tqdm(image_info, desc="Processing images"):
        try:
            image = Image.open(image_path)
            output_dict = inferencer(image=image, text=args.prompt, understanding_output=True)
            response = output_dict["text"]

            result = {
                "image_path": image_path,
                "label": label,
                "output_text": response
            }

            total_count += 1
            if (label == "real" and "this is a real image" in response.lower()) or \
               (label == "fake" and "this is a fake image" in response.lower()):
                correct_count += 1

        except Exception as e:
            result = {"image_path": image_path, "label": label, "error": str(e)}

        with open(output_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    accuracy = correct_count / total_count if total_count else 0.0
    print(f"✅ Accuracy (processed {total_count} images): {accuracy:.4f}")

    # === 重命名输出文件，带上准确率 ===
    if not args.no_rename:
        base, ext = os.path.splitext(output_path)
        new_output_path = f"{base}_acc{accuracy:.4f}{ext}"
        os.rename(output_path, new_output_path)
        print(f"📁 Final output saved as: {new_output_path}")
    else:
        print(f"📁 Output saved as: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAGEL Image Real/Fake Evaluation")

    parser.add_argument("--model_path", type=str, required=True, help="Path to BAGEL model folder")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint .safetensors file")
    parser.add_argument("--image_folder", type=str, required=True, help="Parent folder containing 0_real and 1_fake")
    parser.add_argument("--output_path", type=str, default=None, help="Optional custom output JSONL path")
    parser.add_argument("--prompt", type=str, default="Please evaluate whether this image is an AI creation or something real, and provide an explanation.", help="Prompt text for evaluation")
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB", help="Max memory per GPU")
    parser.add_argument("--max_images", type=int, default=50, help="Number of images to take from each subfolder")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--no_rename", action="store_true", help="Do not rename output file with accuracy suffix")

    args = parser.parse_args()
    main(args)
