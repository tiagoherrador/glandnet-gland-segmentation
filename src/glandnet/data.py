"""Data loading for GlaS (Warwick-QU): images `<name>.bmp`, masks `<name>_anno.bmp` (instance labels)."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import Dataset

IMG_SIZE = 512


def list_pairs(root, split: str = "train", subset: str | None = None):
    """Sorted (image_path, mask_path) pairs. `split` in {"train","test"}; `subset` in {"A","B"} for test."""
    root = Path(root)
    pairs = []
    for img in sorted((root / split / "images").glob("*.bmp")):
        if subset and not img.stem.startswith(f"test{subset}"):
            continue
        mask = root / split / "masks" / f"{img.stem}_anno.bmp"
        if mask.exists():
            pairs.append((img, mask))
    return pairs


def split_train_val(n: int, val_frac: float = 0.2, seed: int = 42):
    """Reproducible shuffled split (same indices as the original notebook for seed=42)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int((1 - val_frac) * n)
    return idx[:cut], idx[cut:]


def load_image(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path) -> np.ndarray:
    m = np.array(Image.open(path))
    return (m[..., 0] if m.ndim == 3 else m) > 0


def build_border(mask_bin, iterations: int = 2) -> np.ndarray:
    """Thick gland contour: dilation XOR erosion."""
    return np.logical_xor(binary_dilation(mask_bin, iterations=iterations),
                          binary_erosion(mask_bin, iterations=iterations)).astype("float32")


def normalize(img_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float in [-1, 1]."""
    return torch.from_numpy(((img_uint8 / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)).float()


def resize_image(img, size: int = IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, size: int = IMG_SIZE):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def train_augmentations(size: int = IMG_SIZE):
    import albumentations as A  # optional dependency: pip install "glandnet[train]"
    return A.Compose([
        A.Resize(size, size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.HueSaturationValue(p=0.5), A.RandomBrightnessContrast(p=0.5),
    ])


class GlasTrainDataset(Dataset):
    """Returns (image [3,H,W] in [-1,1], target [2,H,W]: interior + border)."""

    def __init__(self, pairs, augment: bool = False, size: int = IMG_SIZE):
        self.pairs, self.size = list(pairs), size
        self.aug = train_augmentations(size) if augment else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = load_image(self.pairs[i][0]), load_mask(self.pairs[i][1]).astype("float32")
        target = np.stack([mask, build_border(mask > 0)], axis=-1)  # [H,W,2]
        if self.aug is not None:
            out = self.aug(image=img, mask=target)
            img, target = out["image"], out["mask"]
        else:
            img, target = resize_image(img, self.size), resize_mask(target, self.size)
        return normalize(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).float()


class GlasEvalDataset(Dataset):
    """Returns (image [3,H,W], binary gland mask [H,W] uint8 at model resolution, name)."""

    def __init__(self, pairs, size: int = IMG_SIZE):
        self.pairs, self.size = list(pairs), size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = load_image(self.pairs[i][0]), load_mask(self.pairs[i][1]).astype(np.uint8)
        return normalize(resize_image(img, self.size)), torch.from_numpy(resize_mask(mask, self.size)), self.pairs[i][0].stem
