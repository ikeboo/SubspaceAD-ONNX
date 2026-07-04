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

The exporter returns the CLS token from the last selected feature layer. This
allows ONNX to prune later transformer blocks that do not contribute to the
patch features used by SubspaceAD. Re-export an older model to get the smaller,
faster graph; patch-token values are unchanged.

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

from subspaceadonnx import SubspaceAD


def read_bgr(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image


# Fit the model using normal images only
normal_images = [read_bgr(path) for path in sorted(glob("data/train/good/*.png"))]

model = SubspaceAD(
    "models/dinov3_vitb_middle7.onnx",
    pca_ev=0.99,
)
model.fit(normal_images)

# A directory can be passed instead; supported images are loaded recursively
# model.fit("data/train/good")

# Save and restore the fitted PCA state (the ONNX model is not duplicated)
model.save_npz("models/pca_params.npz")

restored_model = SubspaceAD("models/dinov3_vitb_middle7.onnx")
restored_model.load_npz("models/pca_params.npz")

# Run inference
target_image = read_bgr("data/test/sample.png")
anomaly_map = model(target_image)
image_score = float(anomaly_map.max())

print(f"image score: {image_score:.4f}")
print(f"pixel threshold: {model.threshold_:.4f}")
print(f"image threshold: {model.image_threshold_:.4f}")

plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
plt.imshow(anomaly_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
plt.colorbar(label="anomaly score")
plt.axis("off")
plt.show()
```

`anomaly_map` is a `float32` array with the same height and width as the input image. Higher values indicate more anomalous regions.

`SubspaceAD` accepts OpenCV BGR arrays by default and forwards them unchanged to
`DINOv3`. `DINOv3.preprocess` performs the BGR-to-RGB conversion immediately
before resizing and normalization. Direct ndarray inputs must therefore also be
OpenCV-style BGR arrays.

By default, PCA is fitted after subtracting the normal mean at each patch
position (`spatial_centering=1.0`). Only one mean vector per position is saved;
training features are not retained as a memory bank. Inference uses the
orthogonal-projection identity for the squared residual and applies a logarithmic
transform relative to the learned median normal SPE. A lower normal-score offset
is subtracted before scaling, so normal maps retain useful contrast while their
training maximum remains 0.5. A normal holdout split learns separate pixel and
image thresholds. The legacy global-PCA behavior remains available:

```python
legacy_model = SubspaceAD(
    "models/dinov3_vitb_middle7.onnx",
    spatial_centering=0.0,
    score_transform="squared",
)
```

`score_transform` also accepts `"sqrt"`. Version-3 NPZ files persist the learned
reference, offset and thresholds. Version-1/2 files remain loadable
with their legacy scoring behavior.

## Image mask prediction

You can also predict a binary anomaly mask directly from an image path:

```python
mask = model.predict_mask("data/test/sample.png")
```

Omitting `threshold` uses `model.threshold_`, learned from holdout normal pixels.

## MVTec evaluation

Use `MVTecEvaluator` to evaluate MVTec-style datasets and append results to a CSV file. By default, it writes to `results.csv` at the dataset root:

```python
from subspaceadonnx import MVTecEvaluator

evaluator = MVTecEvaluator(
    dataset_root="datasets",
    dataset_names=["leather"],
    onnx_path="models/dinov3_vitb_middle7.onnx",
    result_path="outputs/mvtec_results.csv",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
result = evaluator.evaluate()
print(result)
```

Pass `save_hist=True` to `evaluate` to save each dataset's good/abnormal
image-score histogram as `{dataset_name}_hist.png` at the dataset root.

For each dataset category, the evaluator trains a new `SubspaceAD` model with
the images under `train/good`, runs inference on all images under `test`, and
then calculates the metrics. Each category is written as a separate CSV row,
followed by a macro-average row whose `datasets` value is `average`. Metric
values use five decimal places, and `description` is the last column. The
average row appends the evaluated categories to its description, for example
`dinov3_vitb_middle7;leather;screw`.
Training, inference (`n/N` images), and evaluation status are displayed while
it runs. To customize `SubspaceAD`, pass its keyword
arguments through `model_kwargs`, for example `model_kwargs={"pca_ev": 0.95}`.
The same `providers` argument can also be passed directly to `SubspaceAD` and
`DINOv3`. Providers are tried in the given priority order.

For evaluation, keep `normalize_map=False` (the default). AUROC and average
precision use score ordering, while AU-PRO sweeps thresholds over the shared
raw score range, so per-image min-max normalization is unnecessary and can
distort comparisons between images. `segmentation_pro` is normalized AU-PRO
up to FPR 0.3 by default.

On the 11 MVTec categories included in this repository, using the 224px
DINOv3-ViT-S+ Middle-7 model gave the following macro averages:

| Method | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO | Evaluation time |
|---|---:|---:|---:|---:|---:|
| Global PCA + squared SPE | 0.95421 | 0.97498 | 0.97650 | 0.91787 | 268.1s |
| Spatial PCA + legacy log-SPE | **0.95903** | **0.97952** | **0.98018** | 0.92446 | 263.7s |
| Spatial PCA + calibrated log-SPE (current) | 0.95875 | 0.97908 | 0.98015 | **0.92493** | **255.9s** |

The calibrated version fixes anomaly-map contrast and learns usable operating
thresholds; its four macro metrics remain above the original global-PCA
baseline. Per-category results are in
[`outputs/calibrated_v3_224.csv`](outputs/calibrated_v3_224.csv).

The pruned 224px export produced identical patch tokens in a direct comparison,
reduced the graph from 704 to 535 nodes and feature-extraction latency from
22.18ms to 17.97ms on CPU (100-image warm benchmark).

See [inference.ipynb](inference.ipynb) for additional visualization and examples of saving and restoring the PCA parameters.

## Project files

- `export_onnx.py`: Exports intermediate DINOv3 features to ONNX
- `dinov3_onnx.py`: Handles preprocessing and feature extraction with ONNX Runtime
- `subspacead_onnx.py`: Learns the normal PCA subspace and calculates anomaly scores
- `inference.ipynb`: Provides inference and visualization examples
