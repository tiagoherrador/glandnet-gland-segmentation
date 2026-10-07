# GlandNet – gland segmentation and grading in colon histology (GlaS)

[![tests](https://github.com/tiagoherrador/glandnet-gland-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagoherrador/glandnet-gland-segmentation/actions)

PyTorch package + CLIs for **gland segmentation** (4-level U-Net with an interior + border output; each gland = one connected component of the predicted mask)
and **benign/malignant classification of every detected gland** on the GlaS (Warwick-QU) colon-histology challenge dataset (85 train / 60 test-A / 20 test-B images).

![Pipeline example – malignant test image](assets/pipeline_testA_59.png)
*Test image `testA_59` (malignant): U-Net glands, then each gland classified — **green = predicted benign, red = predicted malignant** (number = p(malignant)). 10/10 glands called malignant → image called malignant. Glands < 400 px² are hidden in the figure only.*

## Pipeline examples (held-out test images)

| Benign image, correctly called benign | Malignant image (test B), correctly called malignant |
|---|---|
| ![benign](assets/pipeline_testA_33.png) | ![malignant](assets/pipeline_testB_11.png) |

**A failure case** — `testA_44` is benign but is called malignant (mean p = 0.53). The segmentation is good (9 glands that match the image); the error is in the classifier, which marks two large elongated glands as malignant.
Overall, 5 of 80 test images are misclassified at the fixed 0.5 threshold: 4 malignant called benign (mean p 0.42–0.48) and 1 benign called malignant (p = 0.63).

![failure](assets/pipeline_testA_44.png)

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
Test B (poorly differentiated, irregular glands) is clearly harder, and object-level F1 is well below pixel Dice: adjacent glands that touch merge into a single predicted blob (the border channel is included in the gland probability), which costs matches at the object level.
The checkpoint was trained with the original notebook recipe (`notebooks/exploratory/`); `glandnet-train` is a refactor of that recipe and has **not been re-run end to end**.

## Benign / malignant classifier
EfficientNet-B0 on gland crops, **labels from `Grade.csv`** (image-level grade applied to each gland → weak labels), trained on the train split only,
5-fold patient-grouped CV, fixed 12 epochs and 0.5 threshold, test evaluated once (`glandnet-classify`).

| Setting | Images | Image AUC (95 % bootstrap CI) | Image acc. | Gland AUC |
|---|---|---|---|---|
| Cross-validation, train (out-of-fold) | 85 | 0.962 (0.916–0.994) | 0.92 | 0.876 |
| Test A+B · GT masks | 80 | 0.999 (0.996–1.0) | 0.975 | 0.968 |
| Test A+B · U-Net predicted glands (full pipeline, 730 glands) | 80 | 0.992 (0.974–1.0) | 0.9375 | 0.889 |

**Read with care:** 11 patients appear in both train and test (how GlaS is split), so the model may partly recognise patient-specific tissue; test sets are tiny; gland-level labels are weak.
With predicted glands the pipeline detects 730 glands vs 753 in the ground truth. The predicted masks used here come from the original notebook run, whose post-processing parameters were tuned on test. Per-image/per-gland predictions are in `results/`.

## Quick start
```bash
pip install -e ".[train,dev]"
# put GlaS under data/{train,test}/{images,masks} and data/metadata/Grade.csv
glandnet-train    --data-root data --out-dir models          # U-Net, 80/20 split, seed 42
glandnet-eval     --data-root data --checkpoint models/unet4l.pth
glandnet-classify --data-root data --pred-dir predictions --out results
glandnet-visualize --data-root data --seg-checkpoint models/unet4l.pth \
    --cls-checkpoint models/gland_classifier_effb0.pth --names testA_59 testB_11 --out assets
pytest -q                                                    # 23 tests
```

## Code layout
`src/glandnet/` — `model.py` (U-Net), `losses.py` (BCE with border weight + Dice + Tversky), `postprocess.py` (threshold → remove small objects → connected-component glands),
`metrics.py` (pixel / object metrics), `visualize.py` (green/red gland-box figures), `data.py` (GlaS I/O, reproducible split), `train.py`, `evaluate.py`, `classifier.py`.
`tests/` covers the model shape, loss behaviour, watershed splitting of touching glands, metric edge cases, the data split and `Grade.csv` parsing.

## Decisions and lessons
- **Iteration path:** U-Net 3 levels + Tversky → + edge-weighted BCE → 4 levels with a border channel (the final model).
- **Glands are connected components.** The watershed step floods the entire mask, so it does not change which pixels are glands; counting/cropping/evaluating uses connected components (`gland_instances`). Using the watershed ids as instances would cut large glands into many fragments (2,183 "glands" instead of 730). Making the watershed actually split touching glands is an open improvement.
- **Tuning on test is a leak.** The first version swept thresholds on test A/B and reported the best. The package selects on validation instead; the numbers above are the honest ones.
- **A benign/malignant classifier needs real labels.** The first attempt used test-A = benign / test-B = malignant (wrong — both subsets contain both grades) and trained on test images. It was removed and replaced by the `Grade.csv`-based version above.
- **Patient-level structure matters:** splits are grouped by patient for CV, and the train/test patient overlap is reported rather than hidden.

## Data
Download GlaS from the Warwick-QU challenge page; it is not redistributed here. Model weights (`*.pth`) are not versioned.

## Author
Tiago Alejo Herrador — MIT licensed.
