import unittest

import numpy as np

from mvtec_evaluation import MVTecEvaluator


class MVTecMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = MVTecEvaluator(
            dataset_root=".",
            dataset_names=[],
            description="test",
            model=object(),
        )

    def test_roc_auc_handles_tied_scores_without_order_dependence(self) -> None:
        scores = np.array([0.5, 0.5])
        self.assertEqual(self.evaluator._roc_auc(np.array([0, 1]), scores), 0.5)
        self.assertEqual(self.evaluator._roc_auc(np.array([1, 0]), scores), 0.5)

    def test_average_precision_handles_tied_scores_as_one_threshold(self) -> None:
        scores = np.array([0.5, 0.5])
        self.assertEqual(
            self.evaluator._average_precision(np.array([0, 1]), scores),
            0.5,
        )
        self.assertEqual(
            self.evaluator._average_precision(np.array([1, 0]), scores),
            0.5,
        )

    def test_metrics_accept_unnormalized_scores(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([100.0, 200.0, 300.0, 400.0])
        self.assertEqual(self.evaluator._roc_auc(labels, scores), 1.0)
        self.assertEqual(self.evaluator._average_precision(labels, scores), 1.0)

    def test_pro_uses_negative_pixels_from_good_images(self) -> None:
        examples = [
            {
                "gt_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
                "scores": np.zeros((2, 2), dtype=np.float32),
            },
            {
                "gt_mask": np.zeros((2, 2), dtype=np.uint8),
                "scores": np.zeros((2, 2), dtype=np.float32),
            },
        ]
        self.assertEqual(self.evaluator._pixel_negatives_count(examples), 7)

    def test_perfect_segmentation_has_perfect_aupro(self) -> None:
        examples = [
            {
                "gt_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
                "scores": np.array([[1.0, 0.0], [0.0, 0.0]]),
            },
            {
                "gt_mask": np.zeros((2, 2), dtype=np.uint8),
                "scores": np.zeros((2, 2), dtype=np.float32),
            },
        ]
        pixel_scores = np.concatenate([item["scores"].ravel() for item in examples])
        self.assertAlmostEqual(self.evaluator._pro_auc(examples, pixel_scores), 1.0)

    def test_constant_map_aupro_is_not_reported_as_perfect_or_zero(self) -> None:
        examples = [
            {
                "gt_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
                "scores": np.zeros((2, 2), dtype=np.float32),
            },
            {
                "gt_mask": np.zeros((2, 2), dtype=np.uint8),
                "scores": np.zeros((2, 2), dtype=np.float32),
            },
        ]
        pixel_scores = np.zeros(8, dtype=np.float32)
        self.assertAlmostEqual(self.evaluator._pro_auc(examples, pixel_scores), 0.15)


if __name__ == "__main__":
    unittest.main()
