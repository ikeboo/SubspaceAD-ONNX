import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from subspaceadonnx import MVTecEvaluator, SubspaceAD
from subspaceadonnx.core.dinov3_onnx import DINOv3


class _CaptureDino:
    def __init__(self) -> None:
        self.image = None

    def __call__(self, image):
        self.image = np.array(image, copy=True)
        return np.zeros(6, np.float32), np.ones((1, 1, 6), np.float32)


class InputColorTests(unittest.TestCase):
    def test_default_passes_opencv_bgr_to_dino_unchanged(self) -> None:
        dino = _CaptureDino()
        model = SubspaceAD("model.onnx", dino=dino, pca_dim=1)
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)

        model._extract_patch_features(bgr)

        np.testing.assert_array_equal(dino.image, bgr)

    def test_dino_preprocess_converts_bgr_to_rgb(self) -> None:
        dino = object.__new__(DINOv3)
        dino.height = 1
        dino.width = 1
        dino.mean = np.zeros((1, 1, 3), dtype=np.float32)
        dino.std = np.ones((1, 1, 3), dtype=np.float32)
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)

        pixel_values = dino.preprocess(bgr)

        self.assertEqual(pixel_values.shape, (1, 3, 1, 1))
        np.testing.assert_allclose(
            pixel_values[0, :, 0, 0],
            np.array([30, 20, 10], dtype=np.float32) / 255.0,
        )

    def test_mvtec_loader_keeps_opencv_bgr_order(self) -> None:
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            evaluator = object.__new__(MVTecEvaluator)

            loaded = evaluator._load_bgr_image(image_path)

        np.testing.assert_array_equal(loaded, bgr)


if __name__ == "__main__":
    unittest.main()
