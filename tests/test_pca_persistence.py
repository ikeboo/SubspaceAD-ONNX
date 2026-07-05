import json
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
            spatial_centering=0.75,
            score_transform="sqrt",
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
        model.position_mean_ = np.arange(6, dtype=np.float32).reshape(2, 3)
        model.score_reference_ = 0.125
        model.score_offset_ = 0.25
        model.fit_max_score_ = 2.5
        model.score_scale_ = 0.1
        model.threshold_ = 0.35
        model.image_threshold_ = 0.45
        model.calibration_count_ = 3
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
        self.assertEqual(restored.spatial_centering, original.spatial_centering)
        self.assertEqual(restored.score_transform, original.score_transform)
        np.testing.assert_array_equal(
            restored.position_mean_, original.position_mean_
        )
        self.assertEqual(restored.normalize_map, original.normalize_map)
        self.assertEqual(restored.calibration_target, original.calibration_target)
        self.assertEqual(restored.blur, original.blur)
        self.assertEqual(restored.eps, original.eps)
        self.assertEqual(restored.n_components_, original.n_components_)
        self.assertEqual(restored.feature_dim_, original.feature_dim_)
        self.assertEqual(restored.score_reference_, original.score_reference_)
        self.assertEqual(restored.score_offset_, original.score_offset_)
        self.assertEqual(restored.fit_max_score_, original.fit_max_score_)
        self.assertEqual(restored.score_scale_, original.score_scale_)
        self.assertEqual(restored.threshold_, original.threshold_)
        self.assertEqual(restored.image_threshold_, original.image_threshold_)
        self.assertEqual(restored.calibration_count_, original.calibration_count_)

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

    def test_loads_version_one_file_with_legacy_scoring(self) -> None:
        original = self._fitted_model()
        metadata = {
            "format_version": 1,
            "model_name": "model.onnx",
            "pca_ev": None,
            "pca_dim": 2,
            "feature_l2_normalize": False,
            "normalize_map": False,
            "calibration_target": 0.5,
            "blur": True,
            "eps": 1e-8,
            "n_components": original.n_components_,
            "feature_dim": original.feature_dim_,
            "fit_max_score": 1.0,
            "score_scale": 0.5,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "v1.npz"
            np.savez(
                npz_path,
                metadata=np.asarray(json.dumps(metadata)),
                mean=original.mean_,
                components=original.components_,
                eigvals=original.eigvals_,
            )
            restored = SubspaceAD("model.onnx", dino=object()).load_npz(npz_path)

        self.assertEqual(restored.spatial_centering, 0.0)
        self.assertEqual(restored.score_transform, "squared")
        self.assertIsNone(restored.position_mean_)

    def test_loads_version_two_log_file_with_eps_reference(self) -> None:
        original = self._fitted_model()
        metadata = {
            "format_version": 2,
            "model_name": "model.onnx",
            "pca_ev": None,
            "pca_dim": 2,
            "feature_l2_normalize": False,
            "spatial_centering": 1.0,
            "score_transform": "log",
            "normalize_map": False,
            "calibration_target": 0.5,
            "blur": True,
            "eps": 1e-7,
            "n_components": original.n_components_,
            "feature_dim": original.feature_dim_,
            "fit_max_score": 2.0,
            "score_scale": 0.25,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "v2.npz"
            np.savez(
                npz_path,
                metadata=np.asarray(json.dumps(metadata)),
                mean=original.mean_,
                components=original.components_,
                eigvals=original.eigvals_,
                position_mean=original.position_mean_,
            )
            restored = SubspaceAD("model.onnx", dino=object()).load_npz(npz_path)

        self.assertEqual(restored.score_transform, "log")
        self.assertEqual(restored.score_reference_, metadata["eps"])
        self.assertEqual(restored.score_offset_, 0.0)
        self.assertEqual(restored.threshold_, 0.5)

    def test_multi_branch_save_and_load_round_trip(self) -> None:
        original = SubspaceAD(
            "dual.onnx",
            dino=object(),
            pca_ev=None,
            pca_dim=2,
            spatial_centering=0.0,
            score_transform="log",
        )
        rng = np.random.default_rng(4)
        extracted = []
        for _ in range(5):
            features = rng.normal(size=(2, 4, 6)).astype(np.float32)
            extracted.append((features, (2, 2), (8, 8)))
        original._fit_branches(extracted)
        original.score_offset_ = 0.2
        original.score_scale_ = 0.3

        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "dual.npz"
            original.save_npz(npz_path)
            restored = SubspaceAD("dual.onnx", dino=object()).load_npz(npz_path)

        self.assertEqual(len(restored.branch_models_), 2)
        test_features = rng.normal(size=(2, 4, 6)).astype(np.float32)
        np.testing.assert_allclose(
            restored._score_features(test_features),
            original._score_features(test_features),
        )
        self.assertEqual(restored.score_offset_, original.score_offset_)
        self.assertEqual(restored.score_scale_, original.score_scale_)


if __name__ == "__main__":
    unittest.main()
