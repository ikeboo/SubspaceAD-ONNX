from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class MVTecEvaluator:
    """Evaluate MVTec-style datasets and append results to results.csv.

    The evaluator accepts a callable that computes an anomaly map for a given
    image path. It then computes image-level AUROC/AUPR, pixel-level AUROC,
    and normalized area under the PRO curve up to the configured FPR limit.
    Multiple dataset categories are evaluated separately and macro-averaged.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_names: list[str],
        description: str,
        model: object,
        *,
        image_score_method: str = "mtop1p",
        use_mask_output: bool = False,
        mask_threshold: float = 0.5,
        pro_fpr_limit: float = 0.3,
        pro_num_thresholds: int = 200,
        results_filename: str = "results.csv",
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_names = dataset_names
        self.description = description
        self.model = model
        self.image_score_method = image_score_method
        self.use_mask_output = use_mask_output
        self.mask_threshold = mask_threshold
        self.pro_fpr_limit = float(pro_fpr_limit)
        self.pro_num_thresholds = int(pro_num_thresholds)
        self.results_path = self.dataset_root / results_filename

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {self.dataset_root}")
        if not 0.0 < self.pro_fpr_limit <= 1.0:
            raise ValueError("pro_fpr_limit must be in (0, 1].")
        if self.pro_num_thresholds < 2:
            raise ValueError("pro_num_thresholds must be at least 2.")

    def evaluate(self) -> dict[str, float | str]:
        items = list(self._collect_test_items())
        if not items:
            raise ValueError("No test images found for evaluation.")

        per_dataset = {
            dataset_name: {
                "image_labels": [],
                "image_scores": [],
                "pixel_scores": [],
                "pixel_labels": [],
                "pro_examples": [],
            }
            for dataset_name in self.dataset_names
        }

        for item in items:
            anomaly_map = self._predict_map(item["image_path"])
            image_score = self._aggregate_image_score(anomaly_map)
            gt_mask = self._load_gt_mask(
                item["image_path"],
                item["category"],
                anomaly_map.shape,
                item["dataset_name"],
            )
            dataset_values = per_dataset[item["dataset_name"]]

            dataset_values["image_labels"].append(int(item["is_defect"]))
            dataset_values["image_scores"].append(image_score)
            dataset_values["pixel_scores"].append(anomaly_map.ravel())
            dataset_values["pixel_labels"].append(gt_mask.ravel())

            # PRO averages overlap over anomalous regions, but its false-positive
            # rate is measured over every negative pixel, including good images.
            dataset_values["pro_examples"].append({
                "gt_mask": gt_mask,
                "scores": anomaly_map,
            })

        # MVTec metrics are calculated per category and macro-averaged. Pooling
        # categories would overweight datasets with more images/pixels and would
        # require anomaly scores to be calibrated across categories.
        dataset_metrics = [
            self._compute_dataset_metrics(dataset_name, values)
            for dataset_name, values in per_dataset.items()
        ]
        image_auroc = float(np.mean([m["image_auroc"] for m in dataset_metrics]))
        image_aupr = float(np.mean([m["image_aupr"] for m in dataset_metrics]))
        segmentation_auroc = float(
            np.mean([m["segmentation_auroc"] for m in dataset_metrics])
        )
        segmentation_pro = float(
            np.mean([m["segmentation_pro"] for m in dataset_metrics])
        )

        result = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": self.description,
            "datasets": ";".join(self.dataset_names),
            "image_auroc": float(image_auroc),
            "image_aupr": float(image_aupr),
            "segmentation_auroc": float(segmentation_auroc),
            "segmentation_pro": float(segmentation_pro),
        }

        self._append_results(result)
        return result

    def _compute_dataset_metrics(
        self,
        dataset_name: str,
        values: dict[str, list],
    ) -> dict[str, float]:
        if not values["image_labels"]:
            raise ValueError(f"No test images found for dataset: {dataset_name}")

        image_labels = np.asarray(values["image_labels"], dtype=np.int32)
        image_scores = np.asarray(values["image_scores"], dtype=np.float64)
        pixel_scores = np.concatenate(values["pixel_scores"]).astype(np.float64)
        pixel_labels = np.concatenate(values["pixel_labels"]).astype(np.int32)
        return {
            "image_auroc": self._roc_auc(image_labels, image_scores),
            "image_aupr": self._average_precision(image_labels, image_scores),
            "segmentation_auroc": self._roc_auc(pixel_labels, pixel_scores),
            "segmentation_pro": self._pro_auc(
                values["pro_examples"],
                pixel_scores,
            ),
        }

    def _collect_test_items(self) -> Iterable[dict[str, str | bool]]:
        for dataset_name in self.dataset_names:
            dataset_dir = self.dataset_root / dataset_name
            test_dir = dataset_dir / "test"
            if not test_dir.exists():
                raise FileNotFoundError(f"test dir not found: {test_dir}")

            for category_dir in sorted(test_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                category = category_dir.name
                is_defect = category != "good"

                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    for image_path in sorted(category_dir.glob(ext)):
                        yield {
                            "dataset_name": dataset_name,
                            "category": category,
                            "image_path": image_path,
                            "is_defect": is_defect,
                        }

    def _predict_map(self, image_path: Path) -> np.ndarray:
        if self.use_mask_output:
            if not hasattr(self.model, "predict_mask"):
                raise AttributeError(
                    "model must implement predict_mask(image_path: str, threshold: float) "
                    "when use_mask_output=True"
                )
            anomaly_map = self.model.predict_mask(
                str(image_path),
                threshold=self.mask_threshold,
            )
        else:
            if not hasattr(self.model, "predict_anomaly_map"):
                raise AttributeError(
                    "model must implement predict_anomaly_map(image_path: str) -> np.ndarray"
                )
            anomaly_map = self.model.predict_anomaly_map(str(image_path))

        anomaly_map = np.asarray(anomaly_map)
        if anomaly_map.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got shape={anomaly_map.shape}")
        if anomaly_map.size == 0:
            raise ValueError("Anomaly map must not be empty.")

        if anomaly_map.dtype == np.bool_:
            anomaly_map = anomaly_map.astype(np.float32)
        elif anomaly_map.dtype == np.uint8:
            anomaly_map = anomaly_map.astype(np.float32) / 255.0

        anomaly_map = anomaly_map.astype(np.float32)
        if not np.all(np.isfinite(anomaly_map)):
            raise ValueError(f"Anomaly map contains NaN or infinity: {image_path}")
        return anomaly_map

    def _aggregate_image_score(self, anomaly_map: np.ndarray) -> float:
        flat = anomaly_map.ravel().astype(np.float64)
        if self.image_score_method == "max":
            return float(np.max(flat))
        if self.image_score_method == "mean":
            return float(np.mean(flat))
        if self.image_score_method == "p99":
            return float(np.percentile(flat, 99))
        if self.image_score_method == "mtop5":
            k = min(5, flat.size)
            return float(np.mean(np.partition(flat, -k)[-k:]))
        if self.image_score_method == "mtop1p":
            k = max(1, int(flat.size * 0.01))
            return float(np.mean(np.partition(flat, -k)[-k:]))
        raise ValueError(f"Unknown image_score_method: {self.image_score_method}")

    def _load_gt_mask(
        self,
        image_path: Path,
        category: str,
        shape: tuple[int, int],
        dataset_name: str,
    ) -> np.ndarray:
        if category == "good":
            return np.zeros(shape, dtype=np.uint8)

        candidate_name = f"{image_path.stem}_mask{image_path.suffix}"
        gt_path = self.dataset_root / dataset_name / "ground_truth" / category / candidate_name
        if not gt_path.exists():
            raise FileNotFoundError(f"ground truth mask not found: {gt_path}")

        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise ValueError(f"Unable to read ground truth mask: {gt_path}")
        if gt_mask.shape != shape:
            gt_mask = cv2.resize(gt_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        return (gt_mask > 0).astype(np.uint8)

    def _roc_auc(self, labels: np.ndarray, scores: np.ndarray) -> float:
        labels, scores = self._validate_binary_metric_inputs(labels, scores)
        fps, tps = self._binary_clf_curve(labels, scores)
        pos = int(np.sum(labels))
        neg = labels.size - pos
        if pos == 0 or neg == 0:
            raise ValueError("AUROC is undefined when labels contain only one class.")

        fpr = np.concatenate(([0.0], fps / neg))
        tpr = np.concatenate(([0.0], tps / pos))
        return float(np.trapezoid(tpr, fpr))

    def _average_precision(self, labels: np.ndarray, scores: np.ndarray) -> float:
        labels, scores = self._validate_binary_metric_inputs(labels, scores)
        fps, tps = self._binary_clf_curve(labels, scores)
        total_true = int(np.sum(labels))
        if total_true == 0:
            raise ValueError("Average precision is undefined without positive labels.")

        precision = tps / (tps + fps)
        recall = tps / total_true
        recall_step = np.diff(np.concatenate(([0.0], recall)))
        return float(np.sum(recall_step * precision))

    def _validate_binary_metric_inputs(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels, dtype=np.int32).ravel()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if labels.size == 0 or labels.size != scores.size:
            raise ValueError("labels and scores must be non-empty and have equal length.")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain only 0 and 1.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must not contain NaN or infinity.")
        return labels, scores

    def _binary_clf_curve(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return FP/TP counts at distinct score thresholds, handling ties."""
        order = np.argsort(scores, kind="mergesort")[::-1]
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        threshold_indices = np.r_[
            np.flatnonzero(np.diff(sorted_scores)),
            sorted_scores.size - 1,
        ]
        tps = np.cumsum(sorted_labels, dtype=np.int64)[threshold_indices]
        fps = threshold_indices + 1 - tps
        return fps.astype(np.float64), tps.astype(np.float64)

    def _pro_auc(self, pro_examples: list[dict[str, np.ndarray]], pixel_scores: np.ndarray) -> float:
        if len(pro_examples) == 0:
            return 0.0

        min_score = float(np.min(pixel_scores))
        max_score = float(np.max(pixel_scores))

        # Start above the maximum so the curve contains the no-prediction
        # endpoint (FPR=0, PRO=0), then lower the threshold monotonically.
        thresholds = np.concatenate((
            [np.nextafter(max_score, np.inf)],
            np.linspace(max_score, min_score, num=self.pro_num_thresholds),
        ))
        curve = []
        negative_scores, region_scores = self._prepare_pro_scores(pro_examples)
        total_neg = negative_scores.size
        if total_neg == 0:
            return 0.0
        if not region_scores:
            return 0.0
        for thr in thresholds:
            fp = total_neg - np.searchsorted(negative_scores, thr, side="left")
            fpr = float(fp) / total_neg
            overlaps = [
                (scores.size - np.searchsorted(scores, thr, side="left"))
                / scores.size
                for scores in region_scores
            ]
            pro_value = float(np.mean(overlaps))
            curve.append((fpr, pro_value))

        curve.sort(key=lambda x: x[0])
        # Several thresholds can have the same FPR. At a vertical segment, the
        # highest attainable PRO is the value relevant for subsequent area.
        deduplicated = []
        for fpr, pro in curve:
            if deduplicated and fpr == deduplicated[-1][0]:
                deduplicated[-1] = (fpr, max(pro, deduplicated[-1][1]))
            else:
                deduplicated.append((fpr, pro))

        fprs = np.asarray([x[0] for x in deduplicated], dtype=np.float64)
        pros = np.asarray([x[1] for x in deduplicated], dtype=np.float64)
        limit = self.pro_fpr_limit
        below = fprs < limit
        limited_fprs = np.concatenate((fprs[below], [limit]))
        limited_pros = np.concatenate((pros[below], [np.interp(limit, fprs, pros)]))
        return float(np.trapezoid(limited_pros, limited_fprs) / limit)

    def _prepare_pro_scores(
        self,
        pro_examples: list[dict[str, np.ndarray]],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Sort background and per-region scores for efficient thresholding."""
        negative_scores = []
        region_scores = []
        for example in pro_examples:
            gt_mask = example["gt_mask"].astype(np.uint8)
            scores = example["scores"]
            negative_scores.append(scores[gt_mask == 0])

            num_labels, labels = cv2.connectedComponents(gt_mask, connectivity=8)
            for region_id in range(1, num_labels):
                region_scores.append(np.sort(scores[labels == region_id]))

        return np.sort(np.concatenate(negative_scores)), region_scores

    def _pixel_negatives_count(self, pro_examples: list[dict[str, np.ndarray]]) -> int:
        total = 0
        for example in pro_examples:
            total += int(np.size(example["scores"])) - int(np.sum(example["gt_mask"]))
        return total

    def _false_positive_rate(self, pro_examples: list[dict[str, np.ndarray]], threshold: float, total_neg: int) -> float:
        fp = 0
        for example in pro_examples:
            pred = example["scores"] >= threshold
            fp += int(np.count_nonzero(pred & (example["gt_mask"] == 0)))
        return float(fp) / total_neg if total_neg > 0 else 0.0

    def _pro_at_threshold(self, pro_examples: list[dict[str, np.ndarray]], threshold: float) -> float:
        total_overlap = 0.0
        total_regions = 0
        for example in pro_examples:
            gt_mask = example["gt_mask"].astype(np.uint8)
            labels = cv2.connectedComponents(gt_mask, connectivity=8)[1]
            if labels.max() == 0:
                continue
            pred = example["scores"] >= threshold
            for region_id in range(1, int(labels.max()) + 1):
                region = labels == region_id
                region_size = int(np.count_nonzero(region))
                if region_size == 0:
                    continue
                overlap = int(np.count_nonzero(pred & region)) / region_size
                total_overlap += overlap
                total_regions += 1
        return float(total_overlap / total_regions) if total_regions > 0 else 0.0

    def _append_results(self, result: dict[str, float | str]) -> None:
        fieldnames = [
            "date",
            "description",
            "datasets",
            "image_auroc",
            "image_aupr",
            "segmentation_auroc",
            "segmentation_pro",
        ]
        write_header = not self.results_path.exists()
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)
