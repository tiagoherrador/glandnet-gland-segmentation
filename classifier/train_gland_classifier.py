"""Benign/malignant classifier for GlaS glands (corrected version).

Fixes the original notebook experiment, which (a) used testA = benign / testB = malignant
as labels and (b) trained and evaluated on the same test images.

Protocol
--------
* Labels: image-level grade from Grade.csv ("benign" / "malignant"), applied to every gland of
  the image (weak labels -- GlaS does not grade individual glands).
* Training data: gland crops from the **train** split only (ground-truth instance masks).
* Model selection: 5-fold StratifiedGroupKFold on the train split, grouped by patient ID.
  Epochs / threshold are fixed beforehand (no tuning on test).
* Final model: fit on the whole train split, evaluated once on test A, test B and A+B.
  - "gt"   : crops from ground-truth test masks (isolates the classifier)
  - "pred" : crops from the U-Net predicted gland instances (full pipeline; needs
             --pred-dir with <name>_labels_ws.npy files saved by the segmentation notebook)
* Image-level prediction = mean gland probability, threshold 0.5.

Usage
-----
python train_gland_classifier.py --data-root ../data --pred-dir ../predictions_unet4L --out ../results
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from skimage.measure import regionprops
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

MIN_AREA, PAD, SIZE = 80, 8, 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def read_grades(root):
    g = pd.read_csv(Path(root) / "metadata" / "Grade.csv")
    g.columns = [c.strip() for c in g.columns]
    g["label"] = g["grade (GlaS)"].str.strip().str.lower().map({"benign": 0, "malignant": 1})
    assert g["label"].notna().all()
    return g.rename(columns={"patient ID": "patient"})[["name", "patient", "label"]]


def crops_from_labels(img, lab, name, label, patient, source):
    """One 224x224 crop per gland instance (bbox + padding), kept in memory as uint8."""
    out = []
    H, W = lab.shape
    for r in regionprops(lab.astype(np.int32)):
        if r.area < MIN_AREA:
            continue
        a, b, c, d = r.bbox
        a, b, c, d = max(a - PAD, 0), max(b - PAD, 0), min(c + PAD, H), min(d + PAD, W)
        crop = Image.fromarray(img[a:c, b:d]).resize((SIZE, SIZE), Image.BILINEAR)
        out.append(dict(name=name, gland=r.label, patient=patient, label=label,
                        source=source, crop=np.asarray(crop)))
    return out


def build_samples(root, grades, split, pred_dir=None):
    root = Path(root)
    gt, pred = [], []
    for _, row in grades.iterrows():
        n = row["name"]
        if split == "train" and not n.startswith("train"):
            continue
        if split == "test" and not n.startswith("test"):
            continue
        d = "train" if n.startswith("train") else "test"
        img = np.array(Image.open(root / d / "images" / f"{n}.bmp").convert("RGB"))
        m = np.array(Image.open(root / d / "masks" / f"{n}_anno.bmp"))
        if m.ndim == 3:
            m = m[..., 0]
        gt += crops_from_labels(img, m, n, row["label"], row["patient"], "gt")
        if pred_dir is not None and n.startswith("test"):
            sub = "testA" if n.startswith("testA") else "testB"
            f = Path(pred_dir) / sub / "masks_label" / f"{n}_labels_ws.npy"
            if f.exists():
                pred += crops_from_labels(img, np.load(f), n, row["label"], row["patient"], "pred")
    return gt, pred


class CropDS(Dataset):
    def __init__(self, samples, train):
        self.s = samples
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        aug = [T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.ColorJitter(0.2, 0.2, 0.1, 0.02)]
        self.tf = T.Compose((aug if train else [T.ToPILImage()]) + norm)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.tf(self.s[i]["crop"]), self.s[i]["label"]


def fit(samples, device, epochs, seed=42):
    seed_all(seed)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(device)
    y = np.array([s["label"] for s in samples])
    w = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(CropDS(samples, True), batch_size=32, shuffle=True, num_workers=0)
    for ep in range(epochs):
        model.train(); tot = 0
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            opt.zero_grad(); l = loss_fn(model(x), t); l.backward(); opt.step()
            tot += l.item() * len(t)
        sched.step()
        print(f"  epoch {ep+1}/{epochs} loss={tot/len(samples):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, samples, device):
    model.eval()
    dl = DataLoader(CropDS(samples, False), batch_size=64, num_workers=0)
    p = []
    for x, _ in dl:
        p.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(p) if p else np.array([])


def metrics(samples, probs, boot=1000, seed=0):
    df = pd.DataFrame(dict(name=[s["name"] for s in samples], label=[s["label"] for s in samples], p=probs))
    out = {}
    if df["label"].nunique() == 2:
        out["gland_auc"] = float(roc_auc_score(df["label"], df["p"]))
    out["gland_acc"] = float(accuracy_score(df["label"], df["p"] >= .5))
    out["n_glands"] = int(len(df))
    img = df.groupby("name").agg(label=("label", "first"), p=("p", "mean")).reset_index()
    out["n_images"] = int(len(img))
    out["image_acc"] = float(accuracy_score(img["label"], img["p"] >= .5))
    out["image_f1"] = float(f1_score(img["label"], img["p"] >= .5, zero_division=0))
    if img["label"].nunique() == 2:
        out["image_auc"] = float(roc_auc_score(img["label"], img["p"]))
        rng = np.random.default_rng(seed); aucs = []
        for _ in range(boot):
            b = img.iloc[rng.integers(0, len(img), len(img))]
            if b["label"].nunique() == 2:
                aucs.append(roc_auc_score(b["label"], b["p"]))
        out["image_auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    return out, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ckpt", default=None, help="where to save the final model (.pth)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grades = read_grades(a.data_root)
    train, _ = build_samples(a.data_root, grades, "train")
    test_gt, test_pred = build_samples(a.data_root, grades, "test", a.pred_dir)
    print(f"device={device} train glands={len(train)} test(gt)={len(test_gt)} test(pred)={len(test_pred)}")
    tr_patients = set(s["patient"] for s in train); te_patients = set(g["patient"] for g in grades.to_dict("records") if g["name"].startswith("test"))
    print("patients shared between train and test:", sorted(tr_patients & te_patients))

    # ---- patient-grouped CV on train
    imgs = pd.DataFrame([{k: s[k] for k in ("name", "patient", "label")} for s in train]).drop_duplicates("name").reset_index(drop=True)
    oof_s, oof_p, cv = [], [], []
    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(skf.split(imgs, imgs["label"], imgs["patient"])):
        tr_names, va_names = set(imgs.loc[tr, "name"]), set(imgs.loc[va, "name"])
        print(f"fold {k+1}/{a.folds}: train imgs={len(tr_names)} val imgs={len(va_names)}", flush=True)
        m = fit([s for s in train if s["name"] in tr_names], device, a.epochs)
        vs = [s for s in train if s["name"] in va_names]
        oof_s += vs; oof_p.append(predict(m, vs, device))
        cv.append(dict(fold=k + 1, n_val_images=len(va_names)))
    # patient sizes are very uneven, so per-fold AUC is unreliable: pool out-of-fold predictions
    cv_pooled, _ = metrics(oof_s, np.concatenate(oof_p), boot=1000)
    print("CV (pooled out-of-fold):", cv_pooled, flush=True)
    pd.DataFrame(cv).to_csv(out / "cv_folds.csv", index=False)

    # ---- final model on the full train split, single evaluation on test
    print("final fit")
    model = fit(train, device, a.epochs)
    if a.ckpt:
        torch.save(model.state_dict(), a.ckpt)
    res = {"cv_pooled_out_of_fold": cv_pooled, "test": {}}
    for tag, S in (("gt", test_gt), ("pred", test_pred)):
        if not S:
            continue
        p = predict(model, S, device)
        pd.DataFrame(dict(name=[s["name"] for s in S], gland=[s["gland"] for s in S],
                          label=[s["label"] for s in S], prob_malignant=p)).to_csv(out / f"gland_predictions_{tag}.csv", index=False)
        for sub, pref in (("A", "testA"), ("B", "testB"), ("A+B", "test")):
            idx = [i for i, s in enumerate(S) if s["name"].startswith(pref)]
            r, img = metrics([S[i] for i in idx], p[idx])
            res["test"][f"{tag}_{sub}"] = r
            if sub == "A+B":
                img.to_csv(out / f"image_predictions_{tag}.csv", index=False)
            print(tag, sub, r)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
