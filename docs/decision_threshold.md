# Why the model calls an image "malignant" at 40 % and not 50 %

**Short version.** GlandNet gives every gland a probability of being malignant. We call a gland — and an image — *malignant* when that probability is **0.4 or higher**
(not the usual 0.5). We did this on purpose, to **reduce missed cancers (false negatives)**, accepting a few more false alarms (false positives).

## Why missed cancers matter more than false alarms

The two possible mistakes do not cost the same:

| Mistake | What it means | What it costs |
|---|---|---|
| **False negative** | A malignant image is called benign | A tumour can go unnoticed; diagnosis and treatment are delayed. This is the dangerous error. |
| **False positive** | A benign image is called malignant | A pathologist takes a second look and clears it. Annoying and a bit of extra work, but safe. |

A tool that helps screen or prioritise slides should therefore lean towards *flagging too much* rather than *missing something*: a human reviews everything it flags,
but nobody reviews what it declares healthy. In medical terms, we favour **sensitivity** (share of malignant cases caught) over **specificity** (share of benign cases correctly cleared).

## What changing the cut-off does

Lowering the cut-off makes the model more suspicious: it catches more cancers but also flags more healthy images.

![Trade-off](../assets/threshold_tradeoff.png)

**Cross-validation on the 85 training images** (48 malignant, 37 benign; every prediction comes from a model that did not see that patient's images). This is the cleanest evidence we have:

| Cut-off | Malignant caught (sensitivity) | Benign cleared (specificity) | Of the images flagged, how many really are malignant (precision) |
|---|---|---|---|
| 0.5 (usual default) | 87.5 % (42/48) | 97.3 % (36/37) | 97.7 % |
| **0.4 (chosen)** | **91.7 % (44/48)** | **83.8 % (31/37)** | **88.0 %** |
| 0.3 | 97.9 % (47/48) | 70.3 % (26/37) | 81.0 % |
| 0.2 | 97.9 % (47/48) | 40.5 % (15/37) | 68.1 % |

Going from 0.5 to 0.4 catches **2 more cancers out of 48** and costs **5 more false alarms out of 37 benign images**. Going down to 0.3 would catch almost every cancer, but about 3 in 10 healthy
images would be flagged. Where exactly to stop is a **clinical and operational decision** (how many extra reviews can the lab absorb?), not a purely statistical one; 0.4 is a middle path.

**On the 80 held-out test images** (43 malignant, 37 benign), using the full pipeline (glands found by the model):

| Cut-off | Malignant caught | Benign cleared | Missed cancers | False alarms |
|---|---|---|---|---|
| 0.5 | 93.0 % (40/43) | 97.3 % (36/37) | 3 | 1 |
| **0.4** | **97.7 % (42/43)** | **94.6 % (35/37)** | **1** | **2** |
| 0.3 | 100 % (43/43) | 86.5 % (32/37) | 0 | 5 |

At 0.4, two of the three cancers missed at 0.5 are now caught (`testA_51`, `testB_14`), at the price of one extra false alarm (`testB_4`). One cancer is still missed (`testA_38`, probability 0.35).

## How honest is this number?

- **0.4 was a judgement call, not the output of an optimisation.** It was proposed after seeing that, in an earlier version of the evaluation, the missed cancers sat just below 0.5.
  Because we had already seen the test results when choosing it, the test-set figures above are **not a clean out-of-sample estimate**. The cross-validation table is the fairer reference:
  at 0.4 we should expect to catch roughly **9 in 10** cancers, not all of them.
- **0.4 reduces missed cancers; it does not eliminate them.** If the goal is ≥ 95 % sensitivity, the cross-validation says a cut-off near 0.3 is needed, with many more false alarms.
- **The image verdict is the average of its glands**, which can dilute a few very suspicious glands among many normal ones (see `testB_14`: glands #2, #3 and #4 score 1.00 but the average is 0.47).
  A rule like "malignant if *any* gland is clearly malignant" would be even more sensitive; it is not implemented yet.
- **Labels are weak and patients overlap** between training and test (see the main README), so absolute numbers are optimistic.
- A cut-off chosen on this dataset will not transfer unchanged to another hospital, scanner or staining protocol; it would need to be re-checked there.
- This is a research and portfolio project, **not a medical device**.

## How to change it

The cut-off is a single constant, `DECISION_THRESHOLD = 0.4` in `src/glandnet/classifier.py`, used by the figures, tables and metrics.
When running the trained models you can also override it per run: `glandnet-predict --images slide.png --threshold 0.5`.
The cross-validation and test sweeps behind the tables above are saved in `results/threshold_sweep_oof.csv` and `results/threshold_sweep_test_pred.csv`.
