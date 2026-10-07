"""Train the U-Net on GlaS train (80 %) with a held-out validation split (20 %).

    glandnet-train --data-root data --out-dir models
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GlasEvalDataset, GlasTrainDataset, list_pairs, split_train_val
from .losses import combined_loss, dice_loss
from .model import UNet4L, pick_device


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(data_root, out_dir, epochs=120, batch_size=4, lr=1e-4, patience=15, seed=42, num_workers=0):
    seed_all(seed)
    device = pick_device()
    pairs = list_pairs(data_root, "train")
    tr_idx, va_idx = split_train_val(len(pairs), 0.2, seed)
    tr_ds = GlasTrainDataset([pairs[i] for i in tr_idx], augment=True)
    va_ds = GlasTrainDataset([pairs[i] for i in va_idx], augment=False)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(va_ds, batch_size=2, num_workers=num_workers)
    print(f"device={device} train={len(tr_ds)} val={len(va_ds)}")

    model = UNet4L().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "unet4l.pth"
    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = combined_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        model.eval(); d = []
        with torch.no_grad():
            for x, y in va:
                d.append(1 - dice_loss(model(x.to(device)), y.to(device)).item())
        val = float(np.mean(d)); sched.step(val)
        print(f"epoch {ep:03d} train_loss={tot/len(tr_ds):.4f} val_dice={val:.4f}", flush=True)
        if val > best:
            best, bad = val, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {ep}")
                break
    return ckpt, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with train/ and test/ (GlaS)")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ckpt, best = train(a.data_root, a.out_dir, a.epochs, a.batch_size, a.lr, a.patience, a.seed)
    print(f"best val dice {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
