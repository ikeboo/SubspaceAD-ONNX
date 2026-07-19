import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import cv2
import numpy as np

from subspaceadonnx.tools.mvtec_evaluation import MVTecEvaluator


class MVTecMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        onnx_path = root / "model.onnx"
        onnx_path.touch()
        self.evaluator = MVTecEvaluator(
            dataset_root=str(root),
            dataset_names=[],
            onnx_path=str(onnx_path),
            description="test",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

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

    def test_fast_curve_matches_stable_sort_reference_with_many_ties(self) -> None:
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=10_000, dtype=np.int32)
        scores = rng.integers(0, 20, size=labels.size).astype(np.float32)

        order = np.argsort(scores, kind="mergesort")[::-1]
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        threshold_indices = np.r_[
            np.flatnonzero(np.diff(sorted_scores)),
            sorted_scores.size - 1,
        ]
        expected_tps = np.cumsum(sorted_labels, dtype=np.int64)[threshold_indices]
        expected_fps = threshold_indices + 1 - expected_tps

        actual_fps, actual_tps = self.evaluator._binary_clf_curve(labels, scores)

        np.testing.assert_array_equal(actual_fps, expected_fps)
        np.testing.assert_array_equal(actual_tps, expected_tps)

    def test_metrics_accept_unnormalized_scores(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([100.0, 200.0, 300.0, 400.0])
        self.assertEqual(self.evaluator._roc_auc(labels, scores), 1.0)
        self.assertEqual(self.evaluator._average_precision(labels, scores), 1.0)

    def test_pixel_metric_inputs_keep_compact_dtypes(self) -> None:
        labels = np.array([0, 1], dtype=np.uint8)
        scores = np.array([0.0, 1.0], dtype=np.float32)

        actual_labels, actual_scores = (
            self.evaluator._validate_binary_metric_inputs(labels, scores)
        )

        self.assertEqual(actual_labels.dtype, np.uint8)
        self.assertEqual(actual_scores.dtype, np.float32)

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

    def test_vectorized_aupro_matches_scalar_reference(self) -> None:
        rng = np.random.default_rng(7)
        masks = [
            np.zeros((32, 32), dtype=np.uint8),
            np.pad(
                np.ones((6, 8), dtype=np.uint8),
                ((3, 23), (4, 20)),
            ),
            np.zeros((32, 32), dtype=np.uint8),
        ]
        masks[2][2:7, 3:9] = 1
        masks[2][20:28, 22:30] = 1
        examples = [
            {
                "gt_mask": mask,
                # Deliberate ties also exercise duplicate-FPR handling.
                "scores": rng.integers(0, 25, size=mask.shape).astype(np.float32),
            }
            for mask in masks
        ]
        pixel_scores = np.concatenate(
            [example["scores"].ravel() for example in examples]
        )

        expected = self._scalar_pro_auc_reference(examples, pixel_scores)
        actual = self.evaluator._pro_auc(examples, pixel_scores)

        self.assertAlmostEqual(actual, expected, places=15)

    def _scalar_pro_auc_reference(
        self,
        examples: list[dict[str, np.ndarray]],
        pixel_scores: np.ndarray,
    ) -> float:
        thresholds = np.concatenate((
            [np.nextafter(float(np.max(pixel_scores)), np.inf)],
            np.linspace(
                float(np.max(pixel_scores)),
                float(np.min(pixel_scores)),
                num=self.evaluator.pro_num_thresholds,
            ),
        ))
        negative_scores = []
        region_scores = []
        for example in examples:
            mask = example["gt_mask"]
            scores = example["scores"]
            negative_scores.append(scores[mask == 0])
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            for region_id in range(1, num_labels):
                region_scores.append(np.sort(scores[labels == region_id]))

        negative_scores_array = np.sort(np.concatenate(negative_scores))
        total_neg = negative_scores_array.size
        curve = []
        for threshold in thresholds:
            false_positives = total_neg - np.searchsorted(
                negative_scores_array,
                threshold,
                side="left",
            )
            overlaps = [
                (
                    scores.size
                    - np.searchsorted(scores, threshold, side="left")
                ) / scores.size
                for scores in region_scores
            ]
            curve.append((false_positives / total_neg, float(np.mean(overlaps))))

        curve.sort(key=lambda point: point[0])
        deduplicated = []
        for fpr, pro in curve:
            if deduplicated and fpr == deduplicated[-1][0]:
                deduplicated[-1] = (fpr, max(pro, deduplicated[-1][1]))
            else:
                deduplicated.append((fpr, pro))

        fprs = np.asarray([point[0] for point in deduplicated])
        pros = np.asarray([point[1] for point in deduplicated])
        limit = self.evaluator.pro_fpr_limit
        below = fprs < limit
        limited_fprs = np.concatenate((fprs[below], [limit]))
        limited_pros = np.concatenate((
            pros[below],
            [np.interp(limit, fprs, pros)],
        ))
        return float(np.trapezoid(limited_pros, limited_fprs) / limit)


class FakeModel:
    def __init__(self) -> None:
        self.fit_images: list[np.ndarray] | None = None
        self.score_methods: list[str] = []

    def fit(self, images: list[np.ndarray]) -> None:
        self.fit_images = images

    def __call__(
        self,
        image: np.ndarray,
        *,
        score_method: str = "max",
    ) -> tuple[np.ndarray, float]:
        self.score_methods.append(score_method)
        if np.max(image) == 0:
            anomaly_map = np.zeros((2, 2), dtype=np.float32)
        else:
            anomaly_map = np.array(
                [[1.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            )
        return anomaly_map, float(np.max(anomaly_map))


class MVTecEvaluationFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.onnx_path = self.root / "feature_extractor.onnx"
        self.onnx_path.touch()

        for dataset_name in ("leather", "tile"):
            self._write_image(
                self.root / dataset_name / "train" / "good" / "train.png",
                np.full((2, 2, 3), 127, dtype=np.uint8),
            )
            self._write_image(
                self.root / dataset_name / "test" / "good" / "good.png",
                np.zeros((2, 2, 3), dtype=np.uint8),
            )
            self._write_image(
                self.root / dataset_name / "test" / "defect" / "bad.png",
                np.full((2, 2, 3), 255, dtype=np.uint8),
            )
            self._write_image(
                self.root
                / dataset_name
                / "ground_truth"
                / "defect"
                / "bad_mask.png",
                np.array([[255, 0], [0, 0]], dtype=np.uint8),
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Unable to create test image: {path}")

    def test_trains_a_separate_model_and_reports_progress_per_dataset(self) -> None:
        result_path = self.root / "output" / "evaluation.csv"
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather", "tile"],
            onnx_path=str(self.onnx_path),
            result_path=result_path,
        )
        models = [FakeModel(), FakeModel()]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(evaluator, "_create_model", side_effect=models), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = evaluator.evaluate()

        self.assertEqual(result["description"], "feature_extractor")
        self.assertEqual(result["datasets"], "leather;tile")
        self.assertEqual(result["img_auroc"], 1.0)
        self.assertEqual(len(models), 2)
        self.assertTrue(all(model.fit_images is not None for model in models))
        self.assertTrue(all(len(model.fit_images) == 1 for model in models))
        self.assertTrue(
            all(model.score_methods == ["mtop1p", "mtop1p"] for model in models)
        )

        output = stdout.getvalue() + stderr.getvalue()
        self.assertIn("[MVTec 1/2][leather] Start fitting", output)
        self.assertIn("[MVTec 1/2][leather] Inference", output)
        self.assertIn("[MVTec 1/2][leather] Evaluation completed", output)
        self.assertIn("[MVTec 2/2][tile] Start fitting", output)
        self.assertTrue(result_path.exists())
        self.assertFalse((self.root / "results.csv").exists())
        with result_path.open(newline="", encoding="utf-8") as csvfile:
            rows = list(csv.DictReader(csvfile))

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["datasets"] for row in rows],
            ["leather", "tile", "average"],
        )
        self.assertEqual(
            list(rows[0]),
            [
                "date",
                "datasets",
                "img_auroc",
                "img_aupr",
                "seg_auroc",
                "seg_pro",
                "description",
            ],
        )
        self.assertEqual(rows[0]["img_auroc"], "1.00000")
        self.assertEqual(rows[0]["seg_pro"], "1.00000")
        self.assertEqual(rows[0]["description"], "feature_extractor")
        self.assertEqual(rows[2]["img_auroc"], "1.00000")
        self.assertEqual(
            rows[2]["description"],
            "feature_extractor;leather;tile",
        )

    def test_save_hist_writes_good_and_abnormal_image_score_histograms(self) -> None:
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather"],
            onnx_path=str(self.onnx_path),
        )

        with patch.object(evaluator, "_create_model", return_value=FakeModel()):
            evaluator.evaluate(save_hist=True)

        hist_path = self.root / "leather_hist.png"
        self.assertTrue(hist_path.exists())
        self.assertIsNotNone(cv2.imread(str(hist_path), cv2.IMREAD_COLOR))

    def test_histogram_clips_scores_to_range_and_uses_005_bins(self) -> None:
        import matplotlib.pyplot  # noqa: F401

        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather"],
            onnx_path=str(self.onnx_path),
        )

        with patch("matplotlib.axes.Axes.hist") as hist, \
                patch("matplotlib.axes.Axes.legend"):
            evaluator._save_image_score_histogram(
                "leather",
                image_labels=[0, 0, 1, 1],
                image_scores=[-0.2, 0.25, 0.75, 1.8],
            )

        plotted_scores = hist.call_args.args[0]
        np.testing.assert_array_equal(plotted_scores[0], [0.0, 0.25])
        np.testing.assert_array_equal(plotted_scores[1], [0.75, 1.0])
        np.testing.assert_allclose(
            hist.call_args.kwargs["bins"],
            np.arange(0.0, 1.05, 0.05),
        )
        self.assertEqual(hist.call_args.kwargs["label"], ("good", "abnormal"))
        self.assertEqual(hist.call_args.kwargs["histtype"], "stepfilled")

    def test_existing_csv_is_migrated_to_the_new_column_order(self) -> None:
        result_path = self.root / "results.csv"
        result_path.write_text(
            "date,description,datasets,img_auroc,img_aupr,seg_auroc,seg_pro\n"
            "2026-07-03 10:00:00,old model,leather,1.0,0.9,0.8,0.7\n",
            encoding="utf-8",
        )
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["tile"],
            onnx_path=str(self.onnx_path),
            result_path=result_path,
        )

        evaluator._append_results([{
            "date": "2026-07-04 10:00:00",
            "datasets": "tile",
            "img_auroc": 0.123456,
            "img_aupr": 0.234567,
            "seg_auroc": 0.345678,
            "seg_pro": 0.456789,
            "description": "new model",
        }])

        with result_path.open(newline="", encoding="utf-8") as csvfile:
            rows = list(csv.DictReader(csvfile))

        self.assertEqual(list(rows[0])[-1], "description")
        self.assertEqual(rows[0]["img_aupr"], "0.90000")
        self.assertEqual(rows[0]["description"], "old model")
        self.assertEqual(rows[1]["img_auroc"], "0.12346")

    def test_model_kwargs_are_forwarded_to_subspacead(self) -> None:
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather"],
            onnx_path=str(self.onnx_path),
            model_kwargs={"pca_ev": 0.95},
        )

        with patch("subspaceadonnx.tools.mvtec_evaluation.SubspaceAD") as model_class:
            evaluator._create_model()

        model_class.assert_called_once_with(
            str(self.onnx_path),
            providers=None,
            pca_ev=0.95,
        )

    def test_providers_are_forwarded_to_subspacead(self) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather"],
            onnx_path=str(self.onnx_path),
            providers=providers,
        )

        with patch("subspaceadonnx.tools.mvtec_evaluation.SubspaceAD") as model_class:
            evaluator._create_model()

        model_class.assert_called_once_with(
            str(self.onnx_path),
            providers=providers,
        )

    def test_categories_share_only_the_onnx_feature_extractor(self) -> None:
        providers = ["CPUExecutionProvider"]
        evaluator = MVTecEvaluator(
            dataset_root=str(self.root),
            dataset_names=["leather", "tile"],
            onnx_path=str(self.onnx_path),
            providers=providers,
        )
        shared_dino = object()
        models = [
            SimpleNamespace(dino=shared_dino),
            SimpleNamespace(dino=shared_dino),
        ]

        with patch(
            "subspaceadonnx.tools.mvtec_evaluation.SubspaceAD",
            side_effect=models,
        ) as model_class:
            first = evaluator._create_model()
            second = evaluator._create_model()

        self.assertIsNot(first, second)
        model_class.assert_has_calls([
            call(str(self.onnx_path), providers=providers),
            call(str(self.onnx_path), providers=None, dino=shared_dino),
        ])


if __name__ == "__main__":
    unittest.main()
