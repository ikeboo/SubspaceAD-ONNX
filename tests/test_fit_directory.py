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
        np.testing.assert_array_equal(images[0][0, 0], [255, 0, 0])
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


if __name__ == "__main__":
    unittest.main()
