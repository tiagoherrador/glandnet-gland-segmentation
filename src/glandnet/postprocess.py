"""From a probability map to labelled gland instances (threshold -> clean -> watershed)."""
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import binary_opening, disk, remove_small_objects
from skimage.segmentation import watershed


def clean_mask(pred_bin, min_size: int = 20, opening_radius: int = 0):
    mask = remove_small_objects(pred_bin.astype(bool), min_size=min_size)
    if opening_radius > 0:
        mask = binary_opening(mask, disk(opening_radius))
    return mask.astype(np.uint8)


def split_glands_watershed(pred_bin, footprint_size: int = 20):
    """Split touching glands. Smaller `footprint_size` -> more aggressive splitting."""
    distance = ndi.distance_transform_edt(pred_bin)
    coords = peak_local_max(distance, footprint=np.ones((footprint_size, footprint_size)), labels=pred_bin)
    markers_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        markers_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(markers_mask)
    return watershed(-distance, markers, mask=pred_bin)


def segment(prob_map, thr: float = 0.5, min_size: int = 20, footprint_size: int = 20):
    """prob_map [H,W] in [0,1] -> instance labels [H,W] (0 = background)."""
    raw = (prob_map > thr).astype(np.uint8)
    return split_glands_watershed(clean_mask(raw, min_size=min_size), footprint_size=footprint_size)


@torch.no_grad()
def predict_prob_map(model, images):
    """images [B,3,H,W] -> gland probability [B,H,W] (max over interior/border channels)."""
    return torch.sigmoid(model(images)).max(dim=1).values.cpu().numpy()
