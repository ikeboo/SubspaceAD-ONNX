from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(
    target_img: np.ndarray | str | Path,
    anomaly_map,
    image_score,
    threshold=0.5,
    vmin=0.0,
    vmax=1.0,
    color_map="viridis",
):
    """
    target_img: BGR HWC、または画像ファイルのパス
    anomaly_map: HW float32
    image_score: float
    threshold: float, anomaly_mapの閾値
    """
    if isinstance(target_img, (str, Path)):
        image_path = Path(target_img)
        if not image_path.exists():
            raise FileNotFoundError(f"image_path not found: {image_path}")
        loaded_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded_img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        target_img = loaded_img

    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    axs[0].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Image")
    axs[1].imshow(anomaly_map, cmap=color_map, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Anomaly Map (score={image_score:.2f})")
    axs[2].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axs[2].imshow(anomaly_map, cmap=color_map, alpha=0.5, vmin=vmin, vmax=vmax)
    threshold_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_overlay = np.zeros((*anomaly_map.shape, 4), dtype=np.uint8)
    cv2.drawContours(contour_overlay, contours, -1, (255, 165, 0, 255), 5)
    axs[2].imshow(contour_overlay)
    axs[2].set_title(f"Overlay (Threshold > {threshold})")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
