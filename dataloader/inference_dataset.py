"""
Custom inference dataset for MASA-SR tiled inference.

Loads matched LR / Ref image pairs from two directories by matching filenames.
Ref images are resized to exactly (LR_size * scale) so that the MASA model
always sees the trained 1:4 LR-to-Ref ratio, regardless of the original Ref
image resolution or aspect ratio.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


class InferenceDataset(Dataset):
    """Dataset that pairs LR images with Ref images by matching stem filenames.

    Args:
        lr_dir (str): Directory containing LR (low-resolution) images.
        ref_dir (str): Directory containing Ref (reference) images.
            Filenames must exactly match those in lr_dir (extension may differ).
        scale (int): SR scale factor. Ref will be resized to LR * scale.
    """

    def __init__(self, lr_dir: str, ref_dir: str, scale: int = 4):
        self.lr_dir = lr_dir
        self.ref_dir = ref_dir
        self.scale = scale

        lr_files = {
            os.path.splitext(f)[0]: f
            for f in sorted(os.listdir(lr_dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        }
        ref_files = {
            os.path.splitext(f)[0]: f
            for f in sorted(os.listdir(ref_dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        }

        matched_stems = sorted(set(lr_files.keys()) & set(ref_files.keys()))
        if not matched_stems:
            raise ValueError(
                f"No matching filename pairs found between:\n"
                f"  lr_dir : {lr_dir}\n"
                f"  ref_dir: {ref_dir}\n"
                f"Make sure both directories contain files with identical stems."
            )

        self.pairs = [
            (
                os.path.join(lr_dir, lr_files[stem]),
                os.path.join(ref_dir, ref_files[stem]),
                lr_files[stem],   # original filename (kept for saving)
            )
            for stem in matched_stems
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, ref_path, name = self.pairs[idx]

        lr = cv2.imread(lr_path)
        ref = cv2.imread(ref_path)

        if lr is None:
            raise IOError(f"Cannot read LR image: {lr_path}")
        if ref is None:
            raise IOError(f"Cannot read Ref image: {ref_path}")

        h_lr, w_lr = lr.shape[:2]

        # Resize Ref to exactly (h_lr * scale) x (w_lr * scale).
        # This guarantees the MASA 1:4 LR-to-Ref ratio regardless of the
        # original Ref size or aspect ratio.
        ref_resized = cv2.resize(
            ref,
            (w_lr * self.scale, h_lr * self.scale),
            interpolation=cv2.INTER_CUBIC,
        )
        # Ref_down: same spatial resolution as LR (used by MASA encoder)
        ref_down = cv2.resize(
            ref_resized,
            (w_lr, h_lr),
            interpolation=cv2.INTER_CUBIC,
        )

        def to_tensor(bgr_arr):
            arr = bgr_arr.astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).float()

        return {
            'LR':       to_tensor(lr),          # (3, H, W)
            'Ref':      to_tensor(ref_resized),  # (3, H*scale, W*scale)
            'Ref_down': to_tensor(ref_down),     # (3, H, W)
            'name':     name,
        }
