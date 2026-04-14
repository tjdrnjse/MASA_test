#!/usr/bin/env python
"""
MASA-SR Tiled Inference Engine
================================
Overlapping tiled super-resolution for high-resolution / low-VRAM environments.

Usage:
    python infer_tiled.py [--config config.yml]

Pipeline:
    1. Load config.yml
    2. Build MASA model and load checkpoint
    3. For each LR/Ref pair in test/input & test/ref:
        a. Resize Ref to LR*scale (guarantees 1:4 ratio)
        b. Split LR into overlapping tiles
        c. Extract corresponding Ref/Ref_down tiles
        d. Run MASA inference per tile
        e. Blend tiles with Gaussian or Linear weights
        f. Save result
"""

import os
import sys
import gc
import math
import argparse
import logging
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.modules import define_G
from dataloader.inference_dataset import InferenceDataset


# ---------------------------------------------------------------------------
# Blending kernels
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int) -> np.ndarray:
    """2-D Gaussian kernel normalised so that the peak equals 1.0."""
    sigma = size / 6.0
    ax = np.arange(size) - (size - 1) / 2.0
    yy, xx = np.meshgrid(ax, ax, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.max()).astype(np.float32)


def _linear_kernel(size: int) -> np.ndarray:
    """2-D tent (linear) kernel normalised so that the peak equals 1.0."""
    ax = np.arange(size, dtype=np.float32)
    dist = np.minimum(ax, size - 1 - ax) + 1.0
    kernel = np.outer(dist, dist)
    return (kernel / kernel.max()).astype(np.float32)


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 32):
    """Reflect-pad H and W to the next multiple of `multiple`.

    Returns:
        padded tensor, original H, original W
    """
    _, _, h, w = tensor.shape
    h_new = math.ceil(h / multiple) * multiple
    w_new = math.ceil(w / multiple) * multiple
    pad_b = h_new - h
    pad_r = w_new - w
    if pad_b > 0 or pad_r > 0:
        tensor = F.pad(tensor, (0, pad_r, 0, pad_b), mode='replicate')
    return tensor, h, w


# ---------------------------------------------------------------------------
# Core tiled inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tiled_inference(
    net,
    lr_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    ref_down_tensor: torch.Tensor,
    tile_size: int,
    overlap: int,
    scale: int,
    blending: str,
    device: torch.device,
) -> torch.Tensor:
    """Overlapping tiled SR inference.

    Args:
        net           : MASA model (eval mode).
        lr_tensor     : (1, C, H, W)  – LR image on CPU.
        ref_tensor    : (1, C, H*scale, W*scale) – Ref already scaled to 4x LR.
        ref_down_tensor: (1, C, H, W) – Ref downsampled back to LR resolution.
        tile_size     : Tile edge length in LR pixels.
        overlap       : Overlap between adjacent tiles in LR pixels.
        scale         : SR upscaling factor.
        blending      : "gaussian" | "linear".
        device        : Target compute device.

    Returns:
        (1, C, H*scale, W*scale) SR tensor on CPU, values in [0, 1].
    """
    _, C, H, W = lr_tensor.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")

    out_H, out_W = H * scale, W * scale
    output_sum = torch.zeros(1, C, out_H, out_W, dtype=torch.float32)
    weight_sum  = torch.zeros(1, 1, out_H, out_W, dtype=torch.float32)

    # Pre-compute blending kernel for a full-size tile
    tile_out = tile_size * scale
    if blending == 'gaussian':
        kernel_np = _gaussian_kernel(tile_out)
    else:
        kernel_np = _linear_kernel(tile_out)

    # Build tile start positions (ensure the last tile covers the image edge)
    def _tile_starts(length, ts, st):
        starts = list(range(0, length - ts, st))
        last = max(0, length - ts)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    y_starts = _tile_starts(H, tile_size, stride)
    x_starts = _tile_starts(W, tile_size, stride)
    total = len(y_starts) * len(x_starts)
    done  = 0

    for y in y_starts:
        y_end    = min(y + tile_size, H)
        th_lr    = y_end - y          # actual LR tile height (may be < tile_size at border)

        for x in x_starts:
            x_end    = min(x + tile_size, W)
            tw_lr    = x_end - x      # actual LR tile width

            # ---- Extract tiles (CPU) ----
            lr_tile       = lr_tensor      [:, :,  y:y_end,          x:x_end          ]
            ref_tile      = ref_tensor     [:, :,  y*scale:y_end*scale,  x*scale:x_end*scale  ]
            ref_down_tile = ref_down_tensor[:, :,  y:y_end,          x:x_end          ]

            # ---- Pad tiles to 32-multiple (model requirement) ----
            lr_tile_pad,       _, _  = _pad_to_multiple(lr_tile,       32)
            ref_tile_pad,      _, _  = _pad_to_multiple(ref_tile,      32)
            ref_down_tile_pad, _, _  = _pad_to_multiple(ref_down_tile, 32)

            # ---- Inference ----
            out_tile = net(
                lr_tile_pad.to(device),
                ref_tile_pad.to(device),
                ref_down_tile_pad.to(device),
            ).cpu()

            # ---- Remove padding from SR output ----
            out_tile = out_tile[:, :, :th_lr * scale, :tw_lr * scale]

            # ---- Blending kernel (cropped at border tiles) ----
            kh = th_lr * scale
            kw = tw_lr * scale
            if kh == tile_out and kw == tile_out:
                k = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)
            else:
                k = torch.from_numpy(kernel_np[:kh, :kw]).unsqueeze(0).unsqueeze(0)

            # ---- Accumulate ----
            oy, ox   = y * scale, x * scale
            oy_end   = oy + kh
            ox_end   = ox + kw
            output_sum[:, :, oy:oy_end, ox:ox_end] += out_tile * k
            weight_sum[:, :, oy:oy_end, ox:ox_end] += k

            # ---- Memory cleanup ----
            del lr_tile_pad, ref_tile_pad, ref_down_tile_pad, out_tile, k
            gc.collect()

            done += 1
            logging.debug(f'  Tile {done}/{total}: LR[{y}:{y_end}, {x}:{x_end}]')

    # ---- Normalise and clamp ----
    output = output_sum / weight_sum.clamp(min=1e-8)
    output = output.clamp(0.0, 1.0)
    return output


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: dict, device: torch.device):
    mc = cfg['model']
    model_args = SimpleNamespace(
        net_name = mc['net_name'],
        input_nc = mc['input_nc'],
        nf       = mc['nf'],
        num_nbr  = mc['num_nbr'],
        sr_scale = mc['sr_scale'],
        gpu_ids  = [],
        device   = device,
        dist     = False,
    )
    net = define_G(model_args)

    ckpt_path = cfg['test_setup']['checkpoint']
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please set test_setup.checkpoint in config.yml."
        )
    logging.info(f'Loading checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    clean = OrderedDict(
        (k[7:] if k.startswith('module.') else k, v)
        for k, v in state.items()
    )
    net.load_state_dict(clean, strict=True)
    net.to(device)
    net.eval()
    logging.info('Model ready.')
    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MASA-SR Tiled Inference')
    parser.add_argument('--config', default='config.yml',
                        help='Path to YAML config (default: config.yml)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup  = cfg['test_setup']
    paths  = cfg['paths']
    device = torch.device(setup['device'])
    scale      = setup['scale']
    tile_size  = setup['tile_size']
    overlap    = setup['overlap']
    blending   = setup.get('blending', 'gaussian')

    net = load_model(cfg, device)

    os.makedirs(paths['save_dir'], exist_ok=True)

    dataset = InferenceDataset(
        lr_dir  = paths['lr_dir'],
        ref_dir = paths['ref_dir'],
        scale   = scale,
    )
    logging.info(f'Found {len(dataset)} image pair(s).')

    for idx in range(len(dataset)):
        sample = dataset[idx]
        name   = sample['name']
        lr     = sample['LR'].unsqueeze(0)        # (1, C, H, W)
        ref    = sample['Ref'].unsqueeze(0)       # (1, C, H*4, W*4)
        ref_dn = sample['Ref_down'].unsqueeze(0)  # (1, C, H, W)

        h_lr, w_lr = lr.shape[2], lr.shape[3]
        logging.info(
            f'[{idx+1}/{len(dataset)}] {name}  '
            f'LR={w_lr}x{h_lr}  '
            f'tile={tile_size} overlap={overlap} blend={blending}'
        )

        output = run_tiled_inference(
            net            = net,
            lr_tensor      = lr,
            ref_tensor     = ref,
            ref_down_tensor= ref_dn,
            tile_size      = tile_size,
            overlap        = overlap,
            scale          = scale,
            blending       = blending,
            device         = device,
        )

        # Save as PNG (BGR, uint8)
        out_np = (output[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        stem, _ = os.path.splitext(name)
        out_path = os.path.join(paths['save_dir'], f'{stem}_sr.png')
        cv2.imwrite(out_path, out_np)
        logging.info(f'  Saved → {out_path}')

        del lr, ref, ref_dn, output
        gc.collect()

    logging.info('All done.')


if __name__ == '__main__':
    main()
