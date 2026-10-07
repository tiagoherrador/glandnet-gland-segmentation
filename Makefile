DATA ?= data
.PHONY: install test train eval classify
install:
	pip install -e ".[train,dev]"
test:
	pytest -q
train:
	glandnet-train --data-root $(DATA) --out-dir models
eval:
	glandnet-eval --data-root $(DATA) --checkpoint models/unet4l.pth --out results/segmentation_metrics.json
classify:
	glandnet-classify --data-root $(DATA) --pred-dir predictions --out results --ckpt models/gland_classifier_effb0.pth
