#!/usr/bin/env python
"""
Dummy pipeline verification for MASA-SR tiled inference.

Creates minimal 64x64 tensors on CPU and runs the full tiled inference
pipeline end-to-end.  On success every assertion is checked and this
script deletes itself automatically.

Run:
    python verify_dummy.py [--config config.yml]
"""

import os
import sys
import gc
import argparse
import logging
import yaml
import torch
from collections import OrderedDict
from types import SimpleNamespace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

DUMMY_LR_SIZE = 64   # LR tile size for verification (small = fast & low-memory)
SCALE         = 4    # must match model training scale


def _resolve_repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    repo = _resolve_repo_root()
    sys.path.insert(0, repo)

    config_path = os.path.join(repo, args.config)
    if not os.path.isfile(config_path):
        logger.error(f'Config not found: {config_path}')
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Force CPU for verification regardless of config setting
    device = torch.device('cpu')
    logger.info(f'Device: {device}')

    mc = cfg['model']

    # --- 1. Build model ---
    from models.modules import define_G

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

    logger.info('Building model...')
    net = define_G(model_args)
    net.to(device)
    net.eval()

    # Load checkpoint when available; fall back to random weights for
    # structural validation only.
    ckpt_path = cfg['test_setup'].get('checkpoint', '')
    if ckpt_path and os.path.isfile(ckpt_path):
        logger.info(f'Loading checkpoint: {ckpt_path}')
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        clean = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state.items()
        )
        net.load_state_dict(clean, strict=True)
        logger.info('Checkpoint loaded.')
    else:
        logger.warning(
            'Checkpoint not found — using random weights.  '
            'Output quality will be meaningless; only the pipeline shape is verified.'
        )

    # --- 2. Create dummy tensors (CPU, small) ---
    ref_size = DUMMY_LR_SIZE * SCALE
    logger.info(
        f'Creating dummy tensors: '
        f'LR=({DUMMY_LR_SIZE}x{DUMMY_LR_SIZE})  '
        f'Ref=({ref_size}x{ref_size})'
    )
    lr_dummy       = torch.rand(1, 3, DUMMY_LR_SIZE, DUMMY_LR_SIZE)
    ref_dummy      = torch.rand(1, 3, ref_size,      ref_size      )
    ref_down_dummy = torch.rand(1, 3, DUMMY_LR_SIZE, DUMMY_LR_SIZE)

    # --- 3. Run tiled inference ---
    from infer_tiled import run_tiled_inference

    tile_size = cfg['test_setup']['tile_size']
    overlap   = cfg['test_setup']['overlap']
    blending  = cfg['test_setup'].get('blending', 'gaussian')

    # Clamp tile_size so it does not exceed the dummy image
    effective_tile = min(tile_size, DUMMY_LR_SIZE)
    effective_ovlp = min(overlap, effective_tile // 4)

    logger.info(
        f'Running tiled inference  '
        f'tile={effective_tile}  overlap={effective_ovlp}  blend={blending}'
    )
    output = run_tiled_inference(
        net             = net,
        lr_tensor       = lr_dummy,
        ref_tensor      = ref_dummy,
        ref_down_tensor = ref_down_dummy,
        tile_size       = effective_tile,
        overlap         = effective_ovlp,
        scale           = SCALE,
        blending        = blending,
        device          = device,
    )

    # --- 4. Assertions ---
    exp_h = DUMMY_LR_SIZE * SCALE
    exp_w = DUMMY_LR_SIZE * SCALE

    assert output.shape == (1, 3, exp_h, exp_w), (
        f'Shape mismatch: expected (1,3,{exp_h},{exp_w}), got {tuple(output.shape)}'
    )
    assert not torch.isnan(output).any(),  'Output contains NaN!'
    assert not torch.isinf(output).any(),  'Output contains Inf!'
    assert output.min() >= -1e-5,          f'Output below 0: min={output.min():.6f}'
    assert output.max() <= 1.0 + 1e-5,    f'Output above 1: max={output.max():.6f}'

    logger.info(f'Output shape : {tuple(output.shape)}  ✓')
    logger.info(f'Output range : [{output.min():.4f}, {output.max():.4f}]  ✓')
    logger.info('All assertions PASSED.')

    # --- 5. Cleanup ---
    del lr_dummy, ref_dummy, ref_down_dummy, output, net
    gc.collect()

    # Self-delete on success
    script_path = os.path.abspath(__file__)
    logger.info(f'Auto-deleting verification script: {script_path}')
    try:
        os.remove(script_path)
        logger.info('Script deleted successfully.')
    except OSError as e:
        logger.warning(f'Could not delete script: {e}')

    logger.info('=== Verification complete ===')
    sys.exit(0)


if __name__ == '__main__':
    main()
