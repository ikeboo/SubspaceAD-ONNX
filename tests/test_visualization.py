import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from subspaceadonnx.tools.visualization import visualize


class VisualizationTests(unittest.TestCase):
    def test_visualize_loads_string_path_as_bgr_image(self) -> None:
        bgr = np.array(
            [[[10, 20, 30], [40, 50, 60]]],
            dtype=np.uint8,
        )
        anomaly_map = np.array([[0.1, 0.8]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "target.png"
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            with patch(
                "subspaceadonnx.tools.visualization.cv2.cvtColor",
                wraps=cv2.cvtColor,
            ) as cvt_color, patch(
                "subspaceadonnx.tools.visualization.plt.show",
            ):
                visualize(str(image_path), anomaly_map, image_score=0.8)

        np.testing.assert_array_equal(cvt_color.call_args_list[0].args[0], bgr)

    def test_visualize_raises_for_missing_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.png"
            with self.assertRaisesRegex(FileNotFoundError, "image_path not found"):
                visualize(
                    str(missing_path),
                    np.zeros((1, 1), dtype=np.float32),
                    image_score=0.0,
                )


if __name__ == "__main__":
    unittest.main()
