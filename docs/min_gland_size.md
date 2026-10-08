# Why very small detections are ignored (minimum gland size)

**Short version.** The gland finder sometimes marks tiny patches of tissue that are not glands at all. Each of those fragments then gets its own malignancy score, and the image verdict is the average of all its glands,
so a few meaningless fragments can drag a clearly malignant image below the cut-off. We now **discard detections smaller than 1,000 pixels** (about 0.4 % of the image) before classifying anything.
The value was chosen on a validation set, not copied from a paper — the literature supports the idea of filtering spurious predictions, but gives no number (see below).

## The case that revealed the problem: `testA_38`

The expert annotation of this image contains **one** gland. The model found it almost perfectly (overlap 0.96) and classified it as malignant with probability 1.00 — but it also marked **four small patches of connective tissue**
(472 to 1,853 px²) as glands. They scored 0.15, 0.03, 0.05 and 0.51, and the average of the five dropped to 0.35, below the cut-off: a malignant image called benign because of detections that were not glands.

![Before the size filter: 1 real gland, 5 detections](../assets/testA_38_before_size_filter.png)

With the minimum size in place, three of the four fragments disappear, the image keeps two detections and is correctly called malignant:

![After the size filter](../assets/pipeline_testA_38.png)

(The one remaining fragment, #2 in the table, is 1,853 px² and sits at the image border. The filter reduces spurious detections, it does not remove all of them.)

## What the literature says — and what it does not

Verified peer-reviewed sources are listed in [references.md](references.md). In short:

- **Removing spurious predictions is standard practice.** Reviews of pipelines for digital pathology describe classical post-processing as a routine part of deep-learning workflows that can improve results over the network alone,
  and explicitly include the "removal of errant predictions" (Salvi et al. 2021; Smith et al. 2021).
- **Glands are easy to confuse with other structures.** Gland-segmentation papers point out the difficulty of telling glandular from non-glandular structures ("strong mimics") and propose disregarding unreliable predictions (Chen et al. 2017; Graham et al. 2019).
- **Gland-level results are used to decide on the whole image.** Pipelines that segment glands first and classify afterwards (Awan et al. 2017; Ho et al. 2022) build their final decision on the detected glands, so (our reading) spurious detections feed directly into it.
- **No paper gives a minimum gland size.** None of the sources prescribes a size in pixels or µm², and none compares size-filtering rules for gland detection. The right value depends on the image resolution, the magnification and the model,
  so it has to be **chosen empirically** — which is what we did.

## How the value was chosen

We swept the minimum size on the **validation split only** (17 images, 186 annotated glands; probability cut-off 0.65), and kept the value with the best object-level F1.
Sizes are in pixels on the 512 × 512 working image (262,144 px): 1,000 px is 0.38 % of it.

| Minimum size (px) | Object F1 | Glands predicted | Spurious detections |
|---|---|---|---|
| 15 | 0.813 | 201 | 30 |
| 25 (previous setting) | 0.827 | 195 | 24 |
| 100 | 0.858 | 183 | 17 |
| 500 | 0.870 | 173 | 11 |
| **1,000 (chosen)** | **0.886** | **161** | **5** |
| 2,000 | 0.880 | 145 | 2 |
| 4,000 | 0.839 | 128 | 0 |

*Spurious detection* = a predicted gland with less than 10 % of its pixels inside the expert annotation. The full table (all cut-offs and sizes) is in `results/validation_sweep.csv`.
The F1 keeps improving up to 1,000 px, is flat at 2,000 px, and falls at 4,000 px because real small glands start to be discarded — that is the price of an overly aggressive filter.
The earlier setting (25 px) was simply the largest value in a very narrow search grid, so it was never able to pick a stricter filter.

## What changed on the test images

The same filter, applied unchanged to the held-out test sets (it was not tuned on them). For comparison, the last column re-runs the test with the previous 25 px setting (`test_reference_min_size_25` in `results/segmentation_metrics.json`; reference only, never used for selection):

| Test set | Object F1 (1,000 px) | Object F1 (25 px, before) | Spurious detections (1,000 px) | Spurious detections (25 px, before) |
|---|---|---|---|---|
| Set A (60 images) | 0.878 | 0.775 | 18 | 105 |
| Set B (20 images) | 0.796 | 0.693 | 6 | 28 |
| **A + B** | **0.858** | **0.755** | **24** | **133** |

The overlap of the outlines (Dice 0.866) does not change: the filter removes wrong detections, it does not redraw the correct ones.

**Effect on the benign / malignant call.** The filter was chosen using the *segmentation* score only, never the classification result. Even so, with the cautious 0.4 cut-off the test set now has
**no missed cancers (43 of 43 caught) and one false alarm (36 of 37 benign cleared)**; before the filter there was one missed cancer and two false alarms
(see [decision_threshold.md](decision_threshold.md)).

## Honest limits

- **Small validation set.** 17 images and 186 glands: the choice of 1,000 px is supported by the trend (F1 rises, then plateaus, then falls) rather than by a sharp optimum. Anything between roughly 1,000 and 2,000 px performs similarly.
- **It can discard real small glands.** The model now outputs fewer glands (646 on the test set) than the annotation contains; part of the gap is neighbouring glands being merged into one blob, and some small real glands are probably lost too (we did not measure how many).
- **Spurious detections are reduced, not eliminated** (24 remain on the test set, in 17 of the 80 images).
- **Pixels, not microns.** The value is expressed in pixels of the 512 × 512 working image. Converting it to µm² needs the scanner resolution of the slides, which this dataset's images do not carry in the files we use. On different data, re-select it.
- **A near-perfect test score deserves caution.** Some patients appear in both training and test, and labels are per image (see the main README), so the results above are best read as "the pipeline works well on this benchmark", not as clinical performance.

## How to change it

The selected parameters live in `results/segmentation_metrics.json` and are mirrored in `DEFAULT_PARAMS` in `src/glandnet/predict.py` (a test fails if they ever drift apart).
The search grid is `MIN_SIZES` in `src/glandnet/evaluate.py`. To re-select on your own validation data, run `glandnet-eval`.
