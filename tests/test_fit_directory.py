import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from subspaceadonnx import SubspaceAD


class FitDirectoryTests(unittest.TestCase):
    def test_load_images_from_directory_recursively_loads_only_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            nested = directory / "nested"
            nested.mkdir()
            (directory / "notes.txt").write_text("not an image", encoding="utf-8")

            red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
            green_bgr = np.array([[[0, 255, 0]]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(directory / "a.png"), red_bgr))
            self.assertTrue(cv2.imwrite(str(nested / "b.PNG"), green_bgr))

            images = SubspaceAD.load_images_from_directory(directory)

        self.assertEqual(len(images), 2)
        np.testing.assert_array_equal(images[0][0, 0], [0, 0, 255])
        np.testing.assert_array_equal(images[1][0, 0], [0, 255, 0])

    def test_fit_accepts_directory_path(self) -> None:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_dim=1,
            blur=False,
        )
        images = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]
        extracted = [
            (np.array([[1.0, 2.0], [2.0, 1.0]]), (1, 2)),
            (np.array([[3.0, 4.0], [4.0, 3.0]]), (1, 2)),
        ]

        with patch.object(
            model,
            "load_images_from_directory",
            return_value=images,
        ) as load_images, patch.object(
            model,
            "_extract_patch_features",
            side_effect=extracted,
        ):
            result = model.fit("normal-images")

        self.assertIs(result, model)
        load_images.assert_called_once_with("normal-images")
        self.assertEqual(model.feature_dim_, 2)

    def test_load_images_from_directory_rejects_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No supported image files"):
                SubspaceAD.load_images_from_directory(temp_dir)

    def test_fit_learns_a_mean_for_each_patch_position(self) -> None:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_dim=1,
            spatial_centering=1.0,
            score_transform="squared",
            blur=False,
        )
        images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        extracted = [
            (np.array([[1.0, 2.0], [10.0, 20.0]], np.float32), (1, 2)),
            (np.array([[3.0, 4.0], [14.0, 24.0]], np.float32), (1, 2)),
        ]

        with patch.object(model, "_extract_patch_features", side_effect=extracted):
            model.fit(images)

        np.testing.assert_allclose(
            model.position_mean_,
            np.array([[2.0, 3.0], [12.0, 22.0]], np.float32),
        )

    def test_spatial_centering_rejects_inconsistent_patch_grids(self) -> None:
        model = SubspaceAD("model.onnx", dino=object(), pca_dim=1)
        images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        extracted = [
            (np.ones((2, 3), np.float32), (1, 2)),
            (np.ones((3, 3), np.float32), (1, 3)),
        ]

        with patch.object(model, "_extract_patch_features", side_effect=extracted):
            with self.assertRaisesRegex(ValueError, "same patch grid"):
                model.fit(images)

    def test_fit_learns_reference_offset_and_holdout_threshold(self) -> None:
        rng = np.random.default_rng(4)
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_ev=None,
            pca_dim=1,
            blur=False,
            random_state=2,
        )
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(10)]
        features = [
            rng.normal(size=(4, 3)).astype(np.float32)
            for _ in images
        ]
        extracted = [(value, (2, 2)) for value in features]

        with patch.object(
            model,
            "_extract_patch_features",
            side_effect=extracted,
        ):
            model.fit(images)

        normal_maxima = []
        for value in features:
            scores = model._score_features(value)
            anomaly_map = model._scores_to_map(scores, (2, 2), (8, 8))
            normal_maxima.append(float(np.max(model._calibrate_map(anomaly_map))))

        self.assertEqual(model.calibration_count_, 1)
        self.assertGreater(model.score_reference_, model.eps)
        self.assertGreaterEqual(model.score_offset_, 0.0)
        self.assertLessEqual(max(normal_maxima), 0.500001)
        self.assertGreaterEqual(model.threshold_, 0.0)
        self.assertLessEqual(model.threshold_, 0.5)


if __name__ == "__main__":
    unittest.main()
