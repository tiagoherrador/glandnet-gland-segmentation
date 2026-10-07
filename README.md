# GlandNet – gland segmentation and grading in colon histology (GlaS)

[![tests](https://github.com/tiagoherrador/glandnet-gland-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagoherrador/glandnet-gland-segmentation/actions)

PyTorch package + CLIs for **gland instance segmentation** (4-level U-Net with an interior + border output, watershed to split touching glands)
and **benign/malignant classification** on the GlaS (Warwick-QU) colon-histology challenge dataset (85 train / 60 test-A / 20 test-B images).

![Pipeline example – malignant test image](assets/pipeline_testA_59.png)
*Test image `testA_59` (malignant): U-Net instances, then each gland classified — **green = predicted benign, red = predicted malignant** (number = p(malignant)). 37/42 glands called malignant → image called malignant. Fragments < 400 px² are hidden in the figure only.*

## Pipeline examples (held-out test images)

| Benign image, correctly called benign | Malignant image (test B), correctly called malignant |
|---|---|
| ![benign](assets/pipeline_testA_33.png) | ![malignant](assets/pipeline_testB_11.png) |

**A failure case** — `testA_24` is malignant but is called benign (mean p = 0.38): the segmentation fragments a long lumen into stripes and the classifier splits its vote between green and red boxes.
All 11 misclassified test images are malignant called benign, with mean p between 0.37 and 0.49 — i.e. the fixed 0.5 threshold is slightly too conservative for this model.

![failure](assets/pipeline_testA_24.png)

Generate these figures with `glandnet-visualize` (see Quick start).

## Segmentation results (leakage-free protocol)

Post-processing parameters (probability threshold, min object size, watershed footprint) are chosen by grid search **on the 20 % validation split only** and then
applied unchanged to test A and B (`glandnet-eval`; selected: threshold 0.65, min size 25, footprint 22).

| Split | Images | Object Dice | Object F1 | Pixel Dice | Hausdorff |
|---|---|---|---|---|---|
| Validation (selection) | 17 | 0.887 | 0.827 | 0.926 | 33.5 |
| **Test A** | 60 | 0.886 | 0.775 | 0.913 | 35.9 |
| **Test B** | 20 | 0.805 | 0.693 | 0.891 | 40.8 |
| Test A+B | 80 | 0.866 | 0.755 | 0.908 | 37.1 |

*Object F1*: a ground-truth gland is a hit if its best-matching predicted gland has Dice > 0.5; F1 = TP / (TP + (FP + FN)/2).
Test B (poorly differentiated, irregular glands) is clearly harder, and object-level F1 is well below pixel Dice — splitting touching glands remains the main failure mode.
The checkpoint was trained with the original notebook recipe (`notebooks/exploratory/`); `glandnet-train` is a refactor of that recipe and has **not been re-run end to end**.

## Benign / malignant classifier
EfficientNet-B0 on gland crops, **labels from `Grade.csv`** (image-level grade applied to each gland → weak labels), trained on the train split only,
5-fold patient-grouped CV, fixed 12 epochs and 0.5 threshold, test evaluated once (`glandnet-classify`).

| Setting | Images | Image AUC (95 % bootstrap CI) | Image acc. | Gland AUC |
|---|---|---|---|---|
| Cross-validation, train (out-of-fold) | 85 | 0.962 (0.916–0.994) | 0.92 | 0.876 |
| Test A+B · GT masks | 80 | 0.999 (0.996–1.0) | 0.975 | 0.968 |
| Test A+B · U-Net predicted masks (full pipeline) | 80 | 0.991 (0.973–1.0) | 0.863 | 0.882 |

**Read with care:** 11 patients appear in both train and test (how GlaS is split), so the model may partly recognise patient-specific tissue; test sets are tiny; gland-level labels are weak.
With predicted masks the image ranking stays excellent but accuracy drops at the fixed 0.5 threshold because the U-Net over-segments (2,183 predicted vs 753 true glands).
The predicted masks used here come from the original notebook run, whose post-processing parameters were tuned on test. Per-image/per-gland predictions are in `results/`.

## Quick start
```bash
pip install -e ".[train,dev]"
# put GlaS under data/{train,test}/{images,masks} and data/metadata/Grade.csv
glandnet-train    --data-root data --out-dir models          # U-Net, 80/20 split, seed 42
glandnet-eval     --data-root data --checkpoint models/unet4l.pth
glandnet-classify --data-root data --pred-dir predictions --out results
glandnet-visualize --data-root data --seg-checkpoint models/unet4l.pth \
    --cls-checkpoint models/gland_classifier_effb0.pth --names testA_59 testB_11 --out assets
pytest -q                                                    # 21 tests
```

## Code layout
`src/glandnet/` — `model.py` (U-Net), `losses.py` (BCE with border weight + Dice + Tversky), `postprocess.py` (threshold → clean → watershed),
`metrics.py` (pixel / object metrics), `visualize.py` (green/red gland-box figures), `data.py` (GlaS I/O, reproducible split), `train.py`, `evaluate.py`, `classifier.py`.
`tests/` covers the model shape, loss behaviour, watershed splitting of touching glands, metric edge cases, the data split and `Grade.csv` parsing.

## Decisions and lessons
- **Iteration path:** U-Net 3 levels + Tversky → + edge-weighted BCE → 4 levels with a border channel (the final model). The border channel is what lets watershed separate adjacent glands.
- **Tuning on test is a leak.** The first version swept thresholds on test A/B and reported the best. The package selects on validation instead; the numbers above are the honest ones.
- **A benign/malignant classifier needs real labels.** The first attempt used test-A = benign / test-B = malignant (wrong — both subsets contain both grades) and trained on test images. It was removed and replaced by the `Grade.csv`-based version above.
- **Patient-level structure matters:** splits are grouped by patient for CV, and the train/test patient overlap is reported rather than hidden.

## Data
Download GlaS from the Warwick-QU challenge page; it is not redistributed here. Model weights (`*.pth`) are not versioned.

## Author
Tiago Alejo Herrador — MIT licensed.
