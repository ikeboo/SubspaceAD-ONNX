import tempfile
import unittest
from pathlib import Path

import numpy as np

from subspaceadonnx import SubspaceAD


class PcaPersistenceTests(unittest.TestCase):
    def _fitted_model(self) -> SubspaceAD:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_ev=None,
            pca_dim=2,
            feature_l2_normalize=True,
            normalize_map=True,
            calibration_target=0.25,
            blur=False,
            eps=1e-7,
        )
        features = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 4.0],
                [3.0, 4.0, 1.0],
                [4.0, 3.0, 2.0],
            ]
        )
        model._fit_pca(features)
        model.fit_max_score_ = 2.5
        model.score_scale_ = 0.1
        return model

    def test_save_and_load_npz_round_trip(self) -> None:
        original = self._fitted_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "pca.npz"
            original.save_npz(npz_path)

            restored = SubspaceAD("model.onnx", dino=object())
            result = restored.load_npz(npz_path)

        self.assertIs(result, restored)
        np.testing.assert_array_equal(restored.mean_, original.mean_)
        np.testing.assert_array_equal(restored.components_, original.components_)
        np.testing.assert_array_equal(restored.eigvals_, original.eigvals_)
        self.assertEqual(restored.pca_ev, original.pca_ev)
        self.assertEqual(restored.pca_dim, original.pca_dim)
        self.assertEqual(
            restored.feature_l2_normalize, original.feature_l2_normalize
        )
        self.assertEqual(restored.normalize_map, original.normalize_map)
        self.assertEqual(restored.calibration_target, original.calibration_target)
        self.assertEqual(restored.blur, original.blur)
        self.assertEqual(restored.eps, original.eps)
        self.assertEqual(restored.n_components_, original.n_components_)
        self.assertEqual(restored.feature_dim_, original.feature_dim_)
        self.assertEqual(restored.fit_max_score_, original.fit_max_score_)
        self.assertEqual(restored.score_scale_, original.score_scale_)

    def test_save_npz_requires_fitted_model(self) -> None:
        model = SubspaceAD("model.onnx", dino=object())

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "not fitted"):
                model.save_npz(Path(temp_dir) / "pca.npz")

    def test_load_npz_rejects_unrecognized_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "invalid.npz"
            np.savez(npz_path, mean=np.zeros((1, 2)))

            model = SubspaceAD("model.onnx", dino=object())
            with self.assertRaisesRegex(ValueError, "missing keys"):
                model.load_npz(npz_path)


if __name__ == "__main__":
    unittest.main()
