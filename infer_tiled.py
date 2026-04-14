#!/usr/bin/env python
"""
MASA-SR Tiled Inference Engine
================================
Overlapping tiled super-resolution for high-resolution / low-VRAM environments.

Usage:
    python infer_tiled.py [--config configs/test_tiled.yml]

Pipeline:
    1. Load YAML config
    2. Build MASA model and load checkpoint
    3. For each LR/Ref pair in test/input & test/ref:
        a. Resize Ref to LR*scale (guarantees 1:4 ratio)
        b. Pre-extract all overlapping tiles (padded to tile_size)
        c. Run batched inference (tile_batch_size tiles per forward pass)
        d. Blend SR tiles back with Gaussian or Linear weights
        f. Save result
"""

import os
import sys
import gc
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
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_to_size(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Replicate-pad tensor to exactly (target_h, target_w).
    Used to make border tiles the same shape as interior tiles for batching.
    """
    _, _, h, w = tensor.shape
    pad_b = target_h - h
    pad_r = target_w - w
    if pad_b > 0 or pad_r > 0:
        tensor = F.pad(tensor, (0, pad_r, 0, pad_b), mode='replicate')
    return tensor


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
    tile_batch_size: int = 1,
) -> torch.Tensor:
    """Overlapping tiled SR inference with GPU-efficient batching.

    All tiles are pre-extracted and padded to a uniform shape (tile_size),
    then processed in mini-batches of `tile_batch_size` in a single forward
    pass each — maximising GPU utilisation while keeping peak VRAM bounded.

    Args:
        net             : MASA model (eval mode).
        lr_tensor       : (1, C, H, W)            LR image on CPU.
        ref_tensor      : (1, C, H*scale, W*scale) Ref pre-scaled to 4x LR.
        ref_down_tensor : (1, C, H, W)            Ref downsampled to LR size.
        tile_size       : Tile edge in LR pixels (must be a multiple of 32).
        overlap         : Overlap between adjacent tiles in LR pixels.
        scale           : SR upscale factor.
        blending        : 'gaussian' | 'linear'.
        device          : Target compute device.
        tile_batch_size : Number of tiles per forward pass.

    Returns:
        (1, C, H*scale, W*scale) SR tensor on CPU, values in [0, 1].
    """
    if tile_size % 32 != 0:
        raise ValueError(f"tile_size ({tile_size}) must be a multiple of 32.")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size}).")

    _, C, H, W = lr_tensor.shape
    out_H, out_W = H * scale, W * scale
    tile_out = tile_size * scale

    output_sum = torch.zeros(1, C, out_H, out_W, dtype=torch.float32)
    weight_sum  = torch.zeros(1, 1, out_H, out_W, dtype=torch.float32)

    # Blending kernel for a full-size (interior) tile
    kernel_np = _gaussian_kernel(tile_out) if blending == 'gaussian' else _linear_kernel(tile_out)

    # -------------------------------------------------------------------------
    # Step 1: Compute tile grid positions
    # -------------------------------------------------------------------------
    def _tile_starts(length: int, ts: int, st: int):
        starts = list(range(0, length - ts, st))
        last = max(0, length - ts)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    y_starts = _tile_starts(H, tile_size, stride)
    x_starts = _tile_starts(W, tile_size, stride)

    # -------------------------------------------------------------------------
    # Step 2: Pre-extract all tiles, pad to uniform tile_size
    # -------------------------------------------------------------------------
    # Each entry: padded tiles (1, C, tile_size, tile_size) and metadata
    tiles = []
    for y in y_starts:
        y_end = min(y + tile_size, H)
        th = y_end - y          # actual LR tile height (<= tile_size at borders)
        for x in x_starts:
            x_end = min(x + tile_size, W)
            tw = x_end - x      # actual LR tile width

            # Extract (may be smaller than tile_size at image borders)
            lr_t  = lr_tensor      [:, :, y:y_end,          x:x_end         ]
            ref_t = ref_tensor     [:, :, y*scale:y_end*scale, x*scale:x_end*scale]
            rdn_t = ref_down_tensor[:, :, y:y_end,          x:x_end         ]

            # Pad border tiles to tile_size so every tile in a batch is the same shape
            lr_t  = _pad_to_size(lr_t,  tile_size,       tile_size      )
            ref_t = _pad_to_size(ref_t, tile_size*scale, tile_size*scale)
            rdn_t = _pad_to_size(rdn_t, tile_size,       tile_size      )

            tiles.append({
                'lr':  lr_t,   # (1, C, tile_size, tile_size)
                'ref': ref_t,  # (1, C, tile_size*scale, tile_size*scale)
                'rdn': rdn_t,  # (1, C, tile_size, tile_size)
                'th': th, 'tw': tw,
                'oy': y * scale, 'ox': x * scale,
            })

    total = len(tiles)
    logging.info(f'  Total tiles: {total}  (batch_size={tile_batch_size})')

    # -------------------------------------------------------------------------
    # Step 3: Process tiles in mini-batches
    # -------------------------------------------------------------------------
    for batch_start in range(0, total, tile_batch_size):
        batch = tiles[batch_start: batch_start + tile_batch_size]
        B = len(batch)

        # Stack into (B, C, tile_size, tile_size) — move to device
        lr_batch  = torch.cat([t['lr']  for t in batch], dim=0).to(device)
        ref_batch = torch.cat([t['ref'] for t in batch], dim=0).to(device)
        rdn_batch = torch.cat([t['rdn'] for t in batch], dim=0).to(device)

        # Single forward pass over B tiles simultaneously
        out_batch = net(lr_batch, ref_batch, rdn_batch).cpu()   # (B, C, tile_size*scale, tile_size*scale)

        # Distribute each tile result back into the accumulation buffers
        for j, tile_info in enumerate(batch):
            th, tw = tile_info['th'], tile_info['tw']
            oy, ox = tile_info['oy'], tile_info['ox']
            kh, kw = th * scale, tw * scale

            # Remove padding from SR output
            out_tile = out_batch[j:j+1, :, :kh, :kw]

            # Blending kernel (crop to actual size for border tiles)
            k = torch.from_numpy(
                kernel_np if (kh == tile_out and kw == tile_out)
                else kernel_np[:kh, :kw]
            ).unsqueeze(0).unsqueeze(0)

            output_sum[:, :, oy:oy+kh, ox:ox+kw] += out_tile * k
            weight_sum[:, :, oy:oy+kh, ox:ox+kw] += k

        del lr_batch, ref_batch, rdn_batch, out_batch
        gc.collect()

        done = min(batch_start + tile_batch_size, total)
        logging.debug(f'  Processed {done}/{total} tiles')

    # -------------------------------------------------------------------------
    # Step 4: Normalise and clamp
    # -------------------------------------------------------------------------
    output = (output_sum / weight_sum.clamp(min=1e-8)).clamp(0.0, 1.0)
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
            f"Please set test_setup.checkpoint in configs/test_tiled.yml."
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
    parser.add_argument('--config', default='configs/test_tiled.yml',
                        help='Path to YAML config (default: configs/test_tiled.yml)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    setup  = cfg['test_setup']
    paths  = cfg['paths']
    device = torch.device(setup['device'])
    scale           = setup['scale']
    tile_size       = setup['tile_size']
    overlap         = setup['overlap']
    blending        = setup.get('blending', 'gaussian')
    tile_batch_size = setup.get('tile_batch_size', 1)

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
        lr     = sample['LR'].unsqueeze(0)
        ref    = sample['Ref'].unsqueeze(0)
        ref_dn = sample['Ref_down'].unsqueeze(0)

        h_lr, w_lr = lr.shape[2], lr.shape[3]
        logging.info(
            f'[{idx+1}/{len(dataset)}] {name}  LR={w_lr}x{h_lr}  '
            f'tile={tile_size} overlap={overlap} batch={tile_batch_size} blend={blending}'
        )

        output = run_tiled_inference(
            net             = net,
            lr_tensor       = lr,
            ref_tensor      = ref,
            ref_down_tensor = ref_dn,
            tile_size       = tile_size,
            overlap         = overlap,
            scale           = scale,
            blending        = blending,
            device          = device,
            tile_batch_size = tile_batch_size,
        )

        out_np = (output[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        stem, _ = os.path.splitext(name)
        out_path = os.path.join(paths['save_dir'], f'{stem}_sr.png')
        cv2.imwrite(out_path, out_np)
        logging.info(f'  Saved -> {out_path}')

        del lr, ref, ref_dn, output
        gc.collect()

    logging.info('All done.')


if __name__ == '__main__':
    main()
