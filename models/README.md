# Trained models

The trained models are **not stored in this repository**. They are shared on request — see *Get the trained models* in the main README.

| File | What it is | Size | SHA-256 |
|---|---|---|---|
| `unet4l.pth` | U-Net that finds the glands | 124 MB | `0b2941dde4f77ca1793fecaf341a8cff5c5827e438f1dd4ae35a118e49a23a52` |
| `gland_classifier_effb0.pth` | EfficientNet-B0 that judges each gland (benign / malignant) | 16 MB | `e6ecafcf03cf4eb3faca5064a83322773584ef169222efc2454e6cd302014f86` |

Put both files in this folder and check them with `shasum -a 256 -c SHA256SUMS.txt` (the checksum file comes with the package).
Then run `glandnet-predict --images your_image.png`.
