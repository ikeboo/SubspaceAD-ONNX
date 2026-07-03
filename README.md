# SubspaceAD ONNX

A simple image anomaly detection implementation using a DINOv3 ONNX model and PCA.
It learns a normal subspace from normal images only and produces an anomaly heatmap for each input image.

## Setup

This project requires Python 3.12 or later and uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## 1. Export DINOv3 to ONNX

The following command exports a DINOv3 ViT-B/16 model using its middle seven layers:

```bash
uv run python subspaceadonnx/tools/export_onnx.py
```

It creates the following files:

```text
models/dinov3_vitb_middle7.onnx
models/dinov3_vitb_middle7.json
```

You can specify a different image size or set of intermediate layers:

```bash
uv run python subspaceadonnx/tools/export_onnx.py \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --height 448 \
  --width 448 \
  --layers 4,5,6,7,8,9,10 \
  --output models/dinov3_custom.onnx
```

## 2. Run anomaly detection

```python
from glob import glob

import cv2
import matplotlib.pyplot as plt

from subspacead_onnx import SubspaceAD


def read_rgb(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Fit the model using normal images only
normal_images = [read_rgb(path) for path in glob("data/train/good/*.png")]

model = SubspaceAD(
    "models/dinov3_vitb_middle7.onnx",
    pca_ev=0.99,
)
model.fit(normal_images)

# Run inference
target_image = read_rgb("data/test/sample.png")
anomaly_map = model(target_image)
image_score = model.image_score(target_image)

print(f"image score: {image_score:.4f}")

plt.imshow(target_image)
plt.imshow(anomaly_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
plt.colorbar(label="anomaly score")
plt.axis("off")
plt.show()
```

`anomaly_map` is a `float32` array with the same height and width as the input image. Higher values indicate more anomalous regions.

## Image mask prediction

You can also predict a binary anomaly mask directly from an image path:

```python
mask = model.predict_mask("data/test/sample.png", threshold=0.5)
```

## MVTec evaluation

Use `MVTecEvaluator` to evaluate MVTec-style datasets and append results to `results.csv` at the dataset root:

```python
from subspaceadonnx import MVTecEvaluator

evaluator = MVTecEvaluator(
    dataset_root="datasets",
    dataset_names=["leather"],
    onnx_path="models/dinov3_vitb_middle7.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
result = evaluator.evaluate()
print(result)
```

For each dataset category, the evaluator trains a new `SubspaceAD` model with
the images under `train/good`, runs inference on all images under `test`, and
then calculates the metrics. Training, inference (`n/N` images), and evaluation
status are displayed while it runs. To customize `SubspaceAD`, pass its keyword
arguments through `model_kwargs`, for example `model_kwargs={"pca_ev": 0.95}`.
The same `providers` argument can also be passed directly to `SubspaceAD` and
`DINOv3`. Providers are tried in the given priority order.

For evaluation, keep `normalize_map=False` (the default). AUROC and average
precision use score ordering, while AU-PRO sweeps thresholds over the shared
raw score range, so per-image min-max normalization is unnecessary and can
distort comparisons between images. `segmentation_pro` is normalized AU-PRO
up to FPR 0.3 by default.

See [inference.ipynb](inference.ipynb) for additional visualization and examples of saving and restoring the PCA parameters.

## Project files

- `export_onnx.py`: Exports intermediate DINOv3 features to ONNX
- `dinov3_onnx.py`: Handles preprocessing and feature extraction with ONNX Runtime
- `subspacead_onnx.py`: Learns the normal PCA subspace and calculates anomaly scores
- `inference.ipynb`: Provides inference and visualization examples
