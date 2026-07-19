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

To keep shallow/local and deeper/structural cues separate, export multiple
layer groups with semicolons. Each group is averaged independently and becomes
one ONNX patch-token output:

```bash
uv run python subspaceadonnx/tools/export_onnx.py \
  --model-name facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --height 224 --width 224 \
  --layer-groups '2,3,4,5;6,7,8,9' \
  --output models/dinov3_vitsplus_224_dual.onnx
```

`SubspaceAD` detects these outputs automatically, fits an independent PCA to
each group, and averages their log-SPE patch scores. The final selected layer
is still block 9, so graph pruning and feature-extraction speed are retained.

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
anomaly_map, image_score = model(target_image)

# A string path is also loaded internally with cv2.imread
# anomaly_map, image_score = model("data/test/sample.png")

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
transform relative to the learned median normal SPE. By default, it combines
residuals from the 99% and 95% PCA subspaces (`multiband_pca_ev=0.95`) using the
same projection, recovering subtle deviations that the fine subspace can
reconstruct away. Scores above the learned normal 99th percentile receive a
small linear tail gain. These additions retain only PCA statistics and add no
feature search or memory bank. A lower normal-score offset is subtracted before
scaling, so normal maps retain useful contrast while their training maximum
remains 0.5. A normal holdout split learns separate pixel and image thresholds.
The legacy single-band behavior remains available:

```python
legacy_model = SubspaceAD(
    "models/dinov3_vitb_middle7.onnx",
    spatial_centering=0.0,
    score_transform="squared",
    multiband_pca_ev=None,
    tail_score_quantile=None,
)
```

`score_transform` also accepts `"sqrt"`. Version-7 NPZ files also persist the
adaptive local-tail and shared-PPCA-mixture statistics. Older NPZ versions
remain loadable with their historical scoring behavior.

The default `mixture_components=2` uses a gated two-mode normal model with a
shared PCA projection. Set `mixture_components=1` to disable it.

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
| Spatial PCA + calibrated log-SPE | 0.95875 | 0.97908 | 0.98015 | **0.92493** | **255.9s** |
| Dual 2–5 / 6–9 independent PCA + score mean | **0.96394** | **0.98326** | **0.98290** | **0.93334** | — |
| Dual + 95/99% multi-band SPE + normal-tail gain | **0.96485** | **0.98335** | **0.98330** | **0.93502** | — |
| Dual + adaptive local tail | **0.96687** | **0.98446** | **0.98338** | **0.93605** | — |
| Dual + adaptive shared-PPCA mixture (current) | **0.96809** | **0.98545** | **0.98388** | **0.93756** | — |

The calibrated version fixes anomaly-map contrast and learns usable operating
thresholds; its four macro metrics remain above the original global-PCA
baseline. Per-category results are in
[`outputs/calibrated_v3_224.csv`](outputs/calibrated_v3_224.csv).

The pruned 224px export produced identical patch tokens in a direct comparison,
reduced the graph from 704 to 535 nodes and feature-extraction latency from
22.18ms to 17.97ms on CPU (100-image warm benchmark).

The dual export has 543 nodes and measured 17.70ms versus 17.74ms for the
single-output pruned graph in a same-process 100-run CPU benchmark. Its full
11-category results are in
[`outputs/dual_branch_vitsplus_224.csv`](outputs/dual_branch_vitsplus_224.csv).
The multi-band/tail result is in
[`outputs/subspacead_multiband_v5_224.csv`](outputs/subspacead_multiband_v5_224.csv).
The current shared-PPCA-mixture result is in
[`outputs/shared_ppca_mixture_v7_224.csv`](outputs/shared_ppca_mixture_v7_224.csv).

See [inference.ipynb](inference.ipynb) for additional visualization and examples of saving and restoring the PCA parameters.

## Project files

- `export_onnx.py`: Exports intermediate DINOv3 features to ONNX
- `dinov3_onnx.py`: Handles preprocessing and feature extraction with ONNX Runtime
- `subspacead_onnx.py`: Learns the normal PCA subspace and calculates anomaly scores
- `inference.ipynb`: Provides inference and visualization examples
