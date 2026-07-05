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


if __name__ == "__main__":
    unittest.main()
