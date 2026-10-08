# References

Peer-reviewed sources retrieved from PubMed (metadata verified there). What each one does and does **not** support is stated explicitly, so the list is not read as stronger evidence than it is.

| # | Reference | What it supports in this project |
|---|---|---|
| 1 | Sirinukunwattana K, et al. **Gland segmentation in colon histology images: The GlaS challenge contest.** *Med Image Anal* 2017;35:489–502. [doi:10.1016/j.media.2016.08.008](https://doi.org/10.1016/j.media.2016.08.008) | The dataset and the evaluation metrics (object-level F1, object Dice, Hausdorff). Gland morphology is what pathologists use for prognosis, and grading suffers from inter-observer variability. |
| 2 | Chen H, et al. **DCAN: Deep contour-aware networks for object instance segmentation from histology images.** *Med Image Anal* 2017;36:135–146. [doi:10.1016/j.media.2016.11.004](https://doi.org/10.1016/j.media.2016.11.004) | Detecting histological objects is hard because of "strong mimics" (structures that look like glands); boundary information helps to separate clustered instances. |
| 3 | Awan R, et al. **Glandular morphometrics for objective grading of colorectal adenocarcinoma histology images.** *Sci Rep* 2017;7:16852. [doi:10.1038/s41598-017-16516-w](https://doi.org/10.1038/s41598-017-16516-w) | Per-gland measurements are used to classify the whole image (normal vs cancer, grade) with a downstream classifier — the same gland-level → image-level design as here. Gland detection is "a prerequisite" of the analysis. |
| 4 | Graham S, et al. **MILD-Net: Minimal information loss dilated network for gland instance segmentation in colon histology images.** *Med Image Anal* 2019;52:199–211. [doi:10.1016/j.media.2018.12.001](https://doi.org/10.1016/j.media.2018.12.001) | It is non-trivial to tell glandular from non-glandular structures; unreliable predictions can be disregarded. |
| 5 | Salvi M, et al. **The impact of pre- and post-image processing techniques on deep learning frameworks: A comprehensive review for digital pathology image analysis.** *Comput Biol Med* 2021;128:104129. [doi:10.1016/j.compbiomed.2020.104129](https://doi.org/10.1016/j.compbiomed.2020.104129) | Most digital-pathology deep-learning methods include classical post-processing, which can improve performance over the network alone. |
| 6 | Smith B, et al. **Developing image analysis pipelines of whole-slide images: Pre- and post-processing.** *J Clin Transl Sci* 2021;5(1):e38. [doi:10.1017/cts.2020.531](https://doi.org/10.1017/cts.2020.531) | Practical pipelines include "removal of errant predictions" before interpreting results. |
| 7 | Ho C, et al. **A promising deep learning-assistive algorithm for histopathological screening of colorectal cancer.** *Sci Rep* 2022;12:2222. [doi:10.1038/s41598-022-06264-x](https://doi.org/10.1038/s41598-022-06264-x) | A two-stage composite model (gland instance segmentation + a classifier) for *screening*, evaluated with emphasis on sensitivity (97.4 %) so that pathologists focus on flagged cases. |

## What the literature does *not* say

None of these papers prescribes a minimum gland size in pixels or µm², and none compares size-filtering rules for gland detection. Removing spurious small predictions is standard practice
([5], [6]), but the right threshold depends on image resolution, magnification and the model. It is therefore **chosen empirically on a validation split** (see `docs/min_gland_size.md`),
not copied from a paper.
