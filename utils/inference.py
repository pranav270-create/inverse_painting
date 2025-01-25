import argparse
import datetime
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.distributed as dist
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from utils.dist_tools import distributed_init
from utils.inference_helpers import *
import lpips
import sys

# Get the root directory of the inverse_painting project
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/renderer/ckpt/checkpoint-global_step-200000.ckpt', help="Path to renderer checkpoint.")
    parser.add_argument("--RP_path", type=str, default='./checkpoints/RP/checkpoint-global_step-80000.ckpt', help="Path to RP model checkpoint.")
    parser.add_argument("--output_dir", type=str, default='./results', help="Path to the output directory.")
    parser.add_argument("--llava_path", type=str, default='checkpoints/TP_llava', help="Path to LLaVA model checkpoint.")
    # Core inference parameters
    parser.add_argument("--steps", type=int, default=25, help="Number of steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale for inference.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for inference.")
    return parser.parse_args()


def main(args):
    # Load configurations and initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    config_path = os.path.join(os.path.dirname(args.ckpt_path), '..', 'config.yaml')
    config = OmegaConf.load(config_path)

    # Update config with arguments
    config.update({
        'pretrained_model_path': 'base_ckpt/realisticVisionV51_v51VAE',  # Using default value
        'split': 'test',
        'llava_path': args.llava_path,
        'binary': True,
        'binary_threshold': 0.2,
        'PE_sec': 20,
        'RP_path': args.RP_path,
    })

    # Set up output directory
    root_dst_dir = prepare_results_dir(config, args.ckpt_path, args.output_dir)
    full_state_dict = torch.load(args.ckpt_path, map_location='cpu')
    
    # Get time and random number for unique identification
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rand_num = random.randint(0, 100000)

    # Set default parameters
    total_step = 50
    guidance_scale = args.guidance_scale
    cur_guidance_scale = 1.0
    cur_alpha = 0.0
    PE_guidance_scale = 5.0
    TP_guidance_scale = 5.0
    RP_guidance_scale = 5.0
    dilate_RP = True
    combine_init = True
    combine_init_ratio = 0.2
    steps = args.steps
    num_actual_inference_steps = 50

    # Create temporary directory for intermediate results
    tmp_cur_img_folder = 'cache_cur_img'
    tmp_cur_img_path = f'{tmp_cur_img_folder}/{time_str}_{rand_num}.png'
    os.makedirs(tmp_cur_img_folder, exist_ok=True)

    # Initialize models
    TP = TP_wrapper(config, full_state_dict, device, dtype)
    RP = RP_wrapper(config, full_state_dict, device, dtype)
    PE = PE_wrapper(config, full_state_dict, device, dtype)
    
    # Prepare embeddings
    with torch.no_grad():
        PE_embeddings = PE.embed(config['PE_sec'])
    negative_next_TP_embeddings = TP.get_negative_embeddings()

    # Load pipeline and LPIPS
    pipeline, pipeline_kwargs = load_pipeline(config, config['pretrained_model_path'], "./base_ckpt/clip-vit-base-patch32", full_state_dict, dtype, device)
    lpips_fn_alex = lpips.LPIPS(net='alex', spatial=False).to(device)

    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Process input image
    ref_img_path = args.input_image
    dst_dir = os.path.join(root_dst_dir, f'single_image_{time_str}')
    os.makedirs(dst_dir, exist_ok=True)

    # target image
    ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
    ori_h, ori_w, c = ref_img.shape
    
    plt.imsave(f"{dst_dir}/ori_img.jpg", ref_img)
    ref_img = pad_to_16(ref_img)

    # current image: starting from white canvas
    cur_img = np.ones((ori_h, ori_w, 3)) * 255
    cur_img = pad_to_16(cur_img)
    cur_img = cur_img.astype(np.uint8)
    plt.imsave(f"{dst_dir}/sample_0.jpg", cur_img[:ori_h, :ori_w])

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    state = generator.get_state()

    next_RP_embeddings_prev = None
    next_RP_embeddings = None
    next_prompt = None

    cur_next_diffs = []
    next_ref_diffs = []        
    for idx in tqdm(range(total_step)):
        generator.set_state(state)

        generator_RP = torch.Generator(device=device)
        generator_RP.manual_seed(seed + idx)

        H, W, C = ref_img.shape

        # determine whether to stop, based on the last two differences
        if len(cur_next_diffs) > 3 and cur_next_diffs[-2] < 1e-3 and cur_next_diffs[-1] < 1e-3:
            break

        # determine whether to stop, based on difference between next and reference
        if len(next_ref_diffs) > 0 and next_ref_diffs[-1] < 1e-1:
            break

        kwargs = {}
        kwargs.update(pipeline_kwargs)              
        
        kwargs['use_PE'] = config['use_PE']
        kwargs['PE_guidance_scale'] = PE_guidance_scale
        kwargs['PE_embeddings'] = PE_embeddings
        kwargs['negative_PE_embeddings'] = torch.zeros_like(PE_embeddings)
    
        cur_img_path = tmp_cur_img_path 
        plt.imsave(cur_img_path, cur_img)
        
        cache_path = cur_img_path.replace('.png', '_.png')
        next_text_embeddings, next_prompt = TP(cur_img_path, ref_img_path, cache_path=cache_path)
    
        kwargs['TP_feature'] = next_text_embeddings
        kwargs['use_TP'] = config['use_TP']
        kwargs['TP_guidance_scale'] = TP_guidance_scale
        kwargs['negative_TP_feature'] = negative_next_TP_embeddings

        if idx == 0:
            cur_img_path = 'white'

        cur_img_path = tmp_cur_img_path 
        plt.imsave(cur_img_path, cur_img)

        # for mask generation
        if next_RP_embeddings is not None:
            next_RP_embeddings_prev = next_RP_embeddings.clone().to(torch.float32)

        if dilate_RP:
            # Try different thresholds to make RP more robust
            threshold_list = [0.5, 0.4, 0.3, 0.2, 0.1]
            for threshold in threshold_list:
                next_RP_embeddings, input_RP_embeddings_diff = RP(cur_img_path, ref_img_path, next_prompt=next_prompt, 
                                                                 next_RP_embeddings_prev=None, PE_sec=config['PE_sec'], 
                                                                 generator=generator_RP, threshold=threshold)

                if idx == 0:
                    break

                next_RP_embeddings_sum = next_RP_embeddings.sum()

                if next_RP_embeddings_sum < int(H * W * 0.05):
                    print(f'Warning: next_RP_embeddings is too small: {next_RP_embeddings_sum}, change to {threshold}')
                    continue 
                
                if next_RP_embeddings_prev is not None:
                    iou = (next_RP_embeddings * next_RP_embeddings_prev).sum() / ((next_RP_embeddings + next_RP_embeddings_prev) > 0).sum()
                    if iou < 0.8:
                        break
                    else:
                        sum_diff = next_RP_embeddings.float().sum() - next_RP_embeddings_prev.float().sum()
                        print(f'Warning: iou {iou} is too high, sum_diff {sum_diff}, change to {threshold}')
        else:
            next_RP_embeddings, input_RP_embeddings_diff = RP(cur_img_path, ref_img_path, next_prompt=next_prompt, 
                                                             next_RP_embeddings_prev=None, PE_sec=config['PE_sec'], 
                                                             generator=generator_RP, threshold=0.5)

        kwargs['RP_guidance_scale'] = RP_guidance_scale
        kwargs['RP_embeddings'] = next_RP_embeddings.to(dtype)
        kwargs['negative_RP_embeddings'] = torch.zeros_like(next_RP_embeddings)

        if combine_init and idx > 0:
            kwargs['combine_init'] = combine_init
            kwargs['combine_init_ratio'] = combine_init_ratio
            kwargs['img_init_latents'] = pred_next_latents

        generator = generator.set_state(state)
        outputs = pipeline(
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            cur_guidance_scale=cur_guidance_scale, 
            width=W,
            height=H,
            generator=generator,
            num_actual_inference_steps=num_actual_inference_steps,
            source_image=ref_img,
            cur_condition=cur_img,
            cur_alpha=cur_alpha,
            **kwargs,
        )

        pred_next_img = outputs.images
        pred_next_latents = outputs.latents
        
        # Process prediction
        pred_next_img = pred_next_img[0]
        pred_next_img = pred_next_img.cpu().numpy()
        pred_next_img = np.clip(pred_next_img * 255, 0, 255).astype(np.uint8)

        # Calculate differences
        cur_img_tensor = torch.tensor(cur_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]
        pred_next_img_tensor = torch.tensor(pred_next_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]
        ref_img_tensor = torch.tensor(ref_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]

        cur_img_tensor = (cur_img_tensor / 255.) * 2 - 1
        pred_next_img_tensor = (pred_next_img_tensor / 255.) * 2 - 1
        ref_img_tensor = (ref_img_tensor / 255.) * 2 - 1

        cur_next_diff = lpips_fn_alex(cur_img_tensor, pred_next_img_tensor).item()
        next_ref_diff = lpips_fn_alex(ref_img_tensor, pred_next_img_tensor).item()

        cur_next_diffs.append(cur_next_diff)
        next_ref_diffs.append(next_ref_diff)

        # Visualization
        next_RP_embeddings_vis = next_RP_embeddings.cpu().detach().numpy()
        next_RP_embeddings_vis = np.clip(next_RP_embeddings_vis * 255, 0, 255).astype(np.uint8)
        next_RP_embeddings_vis = next_RP_embeddings_vis[0,0]
        next_RP_embeddings_vis = next_RP_embeddings_vis[..., None]
        next_RP_embeddings_vis = np.concatenate([next_RP_embeddings_vis] * 3, axis=2)
        next_RP_embeddings_vis = next_RP_embeddings_vis[:ori_h, :ori_w, :]

        next_RP_embeddings_vis = Image.fromarray(next_RP_embeddings_vis)
        draw = ImageDraw.Draw(next_RP_embeddings_vis)
        font = ImageFont.load_default()
        draw.text((10, 10), next_prompt, (255, 0, 0), font=font)
        next_RP_embeddings_vis = np.array(next_RP_embeddings_vis)

        next_RP_embeddings_vis = np.concatenate([pred_next_img[:ori_h, :ori_w], next_RP_embeddings_vis], axis=1)
        plt.imsave(f"{dst_dir}/vis_sample_{idx+1}.jpg", next_RP_embeddings_vis)
        plt.imsave(f"{dst_dir}/sample_{idx+1}.jpg", pred_next_img[:ori_h, :ori_w, :])

        # Update current image
        cur_img = pred_next_img
        cur_img = cur_img[:ori_h, :ori_w, :]
        cur_img = pad_to_16(cur_img)
        cur_img = cur_img.astype(np.uint8)

        assert cur_img.shape[0] == ref_img.shape[0] and cur_img.shape[1] == ref_img.shape[1]

    print(f"Processing complete. Results saved in: {dst_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)