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
uv run python export_dinov3_middle_onnx.py
```

It creates the following files:

```text
models/dinov3_vitb_middle7.onnx
models/dinov3_vitb_middle7.json
```

You can specify a different image size or set of intermediate layers:

```bash
uv run python export_dinov3_middle_onnx.py \
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

See [inference.ipynb](inference.ipynb) for additional visualization and examples of saving and restoring the PCA parameters.

## Project files

- `export_dinov3_middle_onnx.py`: Exports intermediate DINOv3 features to ONNX
- `dinov3_onnx.py`: Handles preprocessing and feature extraction with ONNX Runtime
- `subspacead_onnx.py`: Learns the normal PCA subspace and calculates anomaly scores
- `inference.ipynb`: Provides inference and visualization examples
