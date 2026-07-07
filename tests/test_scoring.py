import unittest

import numpy as np

from subspaceadonnx import SubspaceAD


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(12)
        self.train = rng.normal(size=(80, 8))
        self.test = rng.normal(size=(12, 8))

    def _model(self, transform: str) -> SubspaceAD:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_ev=None,
            pca_dim=3,
            spatial_centering=0.0,
            score_transform=transform,
        )
        model._fit_pca(self.train)
        return model

    def test_projection_identity_matches_explicit_reconstruction(self) -> None:
        model = self._model("squared")
        centered = self.test - model.mean_
        projected = centered @ model.components_
        reconstructed = projected @ model.components_.T + model.mean_
        expected = np.sum((self.test - reconstructed) ** 2, axis=1)

        np.testing.assert_allclose(
            model._score_features(self.test),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_score_transforms_are_nonnegative_and_monotonic(self) -> None:
        squared = self._model("squared")._score_features(self.test)
        square_root = self._model("sqrt")._score_features(self.test)
        logarithmic = self._model("log")._score_features(self.test)

        np.testing.assert_allclose(square_root, np.sqrt(squared), rtol=1e-6)
        self.assertTrue(np.all(logarithmic >= 0.0))
        np.testing.assert_array_equal(
            np.argsort(logarithmic),
            np.argsort(squared),
        )

    def test_multi_branch_score_is_mean_of_independent_pcas(self) -> None:
        model = SubspaceAD(
            "dual.onnx",
            dino=object(),
            pca_ev=None,
            pca_dim=3,
            spatial_centering=0.0,
            score_transform="squared",
        )
        extracted = [
            (
                np.stack((self.train[index:index + 4], self.train[index + 4:index + 8])),
                (2, 2),
                (8, 8),
            )
            for index in range(0, 72, 8)
        ]
        model._fit_branches(extracted)
        features = np.stack((self.test[:4], self.test[4:8]))
        expected = np.mean(
            np.stack([
                branch._score_features(features[index])
                for index, branch in enumerate(model.branch_models_)
            ]),
            axis=0,
        )
        np.testing.assert_allclose(model._score_features(features), expected)

    def test_multiband_log_spe_and_tail_gain_match_explicit_formula(self) -> None:
        rng = np.random.default_rng(23)
        scales = np.geomspace(2.0, 0.05, num=12)
        train = rng.normal(size=(500, 12)) * scales
        test = rng.normal(size=(20, 12)) * scales
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_ev=0.95,
            multiband_pca_ev=0.75,
            multiband_score_weight=0.25,
            tail_score_quantile=0.9,
            tail_score_gain=0.5,
            spatial_centering=0.0,
            score_transform="log",
        )
        model._fit_pca(train)
        model._fit_score_reference(train)

        self.assertLess(model.multiband_components_, model.n_components_)
        centered = test - model.mean_
        projected = centered @ model.components_
        fine = np.maximum(
            np.sum(centered * centered, axis=1)
            - np.sum(projected * projected, axis=1),
            0.0,
        )
        coarse = fine + np.sum(
            projected[:, model.multiband_components_:] ** 2,
            axis=1,
        )
        base = (
            0.75 * np.log1p(fine / model.score_reference_)
            + 0.25 * np.log1p(
                coarse / model.multiband_score_reference_
            )
        )
        expected = base + 0.5 * np.maximum(
            base - model.tail_score_reference_,
            0.0,
        )

        np.testing.assert_allclose(
            model._score_features(test),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_branch_local_tail_adds_only_agreed_local_evidence(self) -> None:
        model = SubspaceAD(
            "dual.onnx",
            dino=object(),
            pca_dim=1,
            branch_local_tail_gain=1.0,
        )
        model.patch_grid_ = (2, 2)
        model.branch_local_tail_enabled_ = True
        model.branch_local_tail_thresholds_ = np.zeros(2, dtype=np.float32)
        branch_scores = [
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
        ]
        expected = np.mean(np.stack(branch_scores), axis=0)
        expected += np.minimum(
            model._local_score_residual(branch_scores[0]),
            model._local_score_residual(branch_scores[1]),
        )

        np.testing.assert_allclose(
            model._fuse_branch_scores(branch_scores),
            expected,
        )

    def test_branch_local_tail_uses_normal_position_variance_gate(self) -> None:
        class ScoreBranch:
            mean_ = np.ones((1, 1))
            components_ = np.ones((1, 1))

            @staticmethod
            def _score_features(features: np.ndarray) -> np.ndarray:
                return features[:, 0].astype(np.float32)

        model = SubspaceAD(
            "dual.onnx",
            dino=object(),
            pca_dim=1,
            branch_local_tail_min_position_variance=0.1,
            branch_local_tail_max_position_variance=0.5,
            position_local_tail_quantile=None,
        )
        model.branch_models_ = [ScoreBranch(), ScoreBranch()]
        model.patch_grid_ = (2, 2)
        position_pattern = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        extracted = []
        for image_offset in (-1.0, 1.0):
            scores = position_pattern + image_offset
            features = np.stack((scores, scores))[:, :, None]
            extracted.append((features, (2, 2), (8, 8)))

        model._fit_branch_local_tail(extracted)

        self.assertAlmostEqual(model.position_variance_ratio_, 0.2, places=6)
        self.assertTrue(model.branch_local_tail_enabled_)
        self.assertEqual(model.branch_local_tail_thresholds_.shape, (2,))

    def test_position_local_tail_uses_position_specific_thresholds(self) -> None:
        model = SubspaceAD(
            "dual.onnx",
            dino=object(),
            pca_dim=1,
            position_local_tail_gain=0.5,
        )
        model.patch_grid_ = (2, 2)
        model.position_local_tail_enabled_ = True
        model.position_local_tail_thresholds_ = np.array(
            [0.0, 0.0, 0.0, 0.25],
            dtype=np.float32,
        )
        branch_scores = [
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
        ]
        base = np.mean(np.stack(branch_scores), axis=0)
        expected = base + 0.5 * np.maximum(
            model._local_score_residual(base)
            - model.position_local_tail_thresholds_,
            0.0,
        )

        np.testing.assert_allclose(
            model._fuse_branch_scores(branch_scores),
            expected,
        )

    def test_shared_ppca_mixture_selects_position_mean_per_image(self) -> None:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_dim=1,
            mixture_components=2,
            mixture_descriptor_grid=1,
            mixture_min_separation=0.0,
            mixture_min_fraction=0.1,
            mixture_min_images=1,
        )
        model.patch_grid_ = (2, 2)
        low = np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1))
        high = np.tile(np.array([0.0, 1.0], dtype=np.float32), (4, 1))
        extracted = [
            (low + index * 0.001, (2, 2), (8, 8))
            for index in range(4)
        ] + [
            (high + index * 0.001, (2, 2), (8, 8))
            for index in range(4)
        ]

        model._fit_position_center(extracted)

        self.assertIsNotNone(model.mixture_position_means_)
        self.assertIsNotNone(model.mixture_descriptor_centers_)
        self.assertEqual(model.mixture_position_means_.shape, (2, 4, 2))
        self.assertGreater(model.mixture_separation_, 0.9)
        np.testing.assert_allclose(
            model._center_features(low + 0.001),
            np.full_like(low, -0.0005),
            atol=1e-6,
        )

    def test_shared_ppca_mixture_gate_rejects_weak_split(self) -> None:
        model = SubspaceAD(
            "model.onnx",
            dino=object(),
            pca_dim=1,
            mixture_components=2,
            mixture_min_separation=1.0,
        )
        model.patch_grid_ = (2, 2)
        rng = np.random.default_rng(9)
        extracted = [
            (rng.normal(size=(4, 3)).astype(np.float32), (2, 2), (8, 8))
            for _ in range(12)
        ]

        model._fit_position_center(extracted)

        self.assertIsNone(model.mixture_position_means_)
        self.assertIsNone(model.mixture_descriptor_centers_)
        self.assertIsNotNone(model.mixture_cluster_sizes_)


if __name__ == "__main__":
    unittest.main()
