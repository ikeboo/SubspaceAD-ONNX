from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..core.subspacead import SubspaceAD


class MVTecEvaluator:
    """Evaluate MVTec-style datasets and append results to results.csv.

    A separate SubspaceAD model is trained from each category's ``train/good``
    images, then evaluated against that category's test images. It computes
    image-level AUROC/AUPR, pixel-level AUROC, and normalized area under the PRO
    curve up to the configured FPR limit. Multiple categories are evaluated
    separately and macro-averaged.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_names: list[str],
        onnx_path: str,
        *,
        description: str | None = None,
        providers: list[str] | None = None,
        model_kwargs: Mapping[str, object] | None = None,
        image_score_method: str = "mtop1p",
        use_mask_output: bool = False,
        mask_threshold: float = 0.5,
        pro_fpr_limit: float = 0.3,
        pro_num_thresholds: int = 200,
        result_path: str | Path | None = None,
        results_filename: str | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_names = dataset_names
        self.onnx_path = Path(onnx_path)
        self.description = description or self.onnx_path.stem
        self.providers = list(providers) if providers is not None else None
        self.model_kwargs = dict(model_kwargs or {})
        self.image_score_method = image_score_method
        self.use_mask_output = use_mask_output
        self.mask_threshold = mask_threshold
        self.pro_fpr_limit = float(pro_fpr_limit)
        self.pro_num_thresholds = int(pro_num_thresholds)
        if result_path is not None and results_filename is not None:
            raise ValueError("Pass either result_path or results_filename, not both.")
        if result_path is not None:
            self.result_path = Path(result_path)
        else:
            self.result_path = self.dataset_root / (results_filename or "results.csv")
        self._shared_dino: object | None = None

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {self.dataset_root}")
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"onnx_path not found: {self.onnx_path}")
        if len(set(self.dataset_names)) != len(self.dataset_names):
            raise ValueError("dataset_names must not contain duplicates.")
        if "providers" in self.model_kwargs:
            raise ValueError(
                "Pass providers through the providers argument, not model_kwargs."
            )
        if not 0.0 < self.pro_fpr_limit <= 1.0:
            raise ValueError("pro_fpr_limit must be in (0, 1].")
        if self.pro_num_thresholds < 2:
            raise ValueError("pro_num_thresholds must be at least 2.")

    def evaluate(self) -> dict[str, float | str]:
        if not self.dataset_names:
            raise ValueError("dataset_names must contain at least one dataset.")

        dataset_metrics = []
        dataset_count = len(self.dataset_names)
        for dataset_index, dataset_name in enumerate(self.dataset_names, start=1):
            progress_label = f"[MVTec {dataset_index}/{dataset_count}][{dataset_name}]"
            train_paths = list(self._collect_train_images(dataset_name))
            if not train_paths:
                raise ValueError(
                    f"No training images found in: "
                    f"{self.dataset_root / dataset_name / 'train' / 'good'}"
                )

            print(f"{progress_label} Start fitting ({len(train_paths)}imgs)", flush=True)
            model = self._create_model()
            normal_images = [self._load_rgb_image(path) for path in train_paths]
            model.fit(normal_images)
            del normal_images
            print(f"{progress_label} Training completed", flush=True)

            items = list(self._collect_test_items(dataset_name))
            if not items:
                raise ValueError(f"No test images found for dataset: {dataset_name}")

            values = {
                "image_labels": [],
                "image_scores": [],
                "pixel_scores": [],
                "pixel_labels": [],
                "pro_examples": [],
            }
            for item in tqdm(
                items,
                desc=f"{progress_label} Inference",
                unit="images",
                dynamic_ncols=True,
            ):
                anomaly_map = self._predict_map(model, item["image_path"])
                image_score = self._aggregate_image_score(anomaly_map)
                gt_mask = self._load_gt_mask(
                    item["image_path"],
                    item["category"],
                    anomaly_map.shape,
                    dataset_name,
                )

                values["image_labels"].append(int(item["is_defect"]))
                values["image_scores"].append(image_score)
                values["pixel_scores"].append(anomaly_map.ravel())
                values["pixel_labels"].append(gt_mask.ravel())

                # PRO averages overlap over anomalous regions, but its false-positive
                # rate is measured over every negative pixel, including good images.
                values["pro_examples"].append({
                    "gt_mask": gt_mask,
                    "scores": anomaly_map,
                })

            print(f"{progress_label} Start evaluation", flush=True)
            metrics = self._compute_dataset_metrics(dataset_name, values)
            dataset_metrics.append(metrics)
            print(
                f"{progress_label} Evaluation completed "
                f"(image AUROC={metrics['img_auroc']:.4f}, "
                f"pixel AUROC={metrics['seg_auroc']:.4f})",
                flush=True,
            )

        # MVTec metrics are macro-averaged. Pooling categories would overweight
        # datasets with more images/pixels and require cross-category calibration.
        img_auroc = float(np.mean([m["img_auroc"] for m in dataset_metrics]))
        img_aupr = float(np.mean([m["img_aupr"] for m in dataset_metrics]))
        seg_auroc = float(
            np.mean([m["seg_auroc"] for m in dataset_metrics])
        )
        seg_pro = float(
            np.mean([m["seg_pro"] for m in dataset_metrics])
        )

        evaluated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "date": evaluated_at,
            "description": self.description,
            "datasets": ";".join(self.dataset_names),
            "img_auroc": float(img_auroc),
            "img_aupr": float(img_aupr),
            "seg_auroc": float(seg_auroc),
            "seg_pro": float(seg_pro),
        }

        dataset_results = [
            {
                "date": evaluated_at,
                "datasets": dataset_name,
                **metrics,
                "description": self.description,
            }
            for dataset_name, metrics in zip(self.dataset_names, dataset_metrics)
        ]
        dataset_results.append({
            "date": evaluated_at,
            "datasets": "average",
            "img_auroc": img_auroc,
            "img_aupr": img_aupr,
            "seg_auroc": seg_auroc,
            "seg_pro": seg_pro,
            "description": ";".join([self.description, *self.dataset_names]),
        })
        self._append_results(dataset_results)
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
        # Anomaly maps and masks already use compact float32/uint8 dtypes.  Keep
        # them that way: metric calculation only depends on score ordering and
        # binary labels, and widening a full MVTec category needlessly allocates
        # another large pair of arrays.
        pixel_scores = np.concatenate(values["pixel_scores"])
        pixel_labels = np.concatenate(values["pixel_labels"])
        return {
            "img_auroc": self._roc_auc(image_labels, image_scores),
            "img_aupr": self._average_precision(image_labels, image_scores),
            "seg_auroc": self._roc_auc(pixel_labels, pixel_scores),
            "seg_pro": self._pro_auc(
                values["pro_examples"],
                pixel_scores,
            ),
        }

    def _create_model(self) -> SubspaceAD:
        model_kwargs = dict(self.model_kwargs)
        if "dino" not in model_kwargs and self._shared_dino is not None:
            model_kwargs["dino"] = self._shared_dino

        # PCA state remains independent because each category gets a fresh
        # SubspaceAD instance.  The immutable ONNX feature extractor/session is
        # shared, avoiding an expensive model reload for every category.
        providers = None if "dino" in model_kwargs else self.providers
        model = SubspaceAD(
            str(self.onnx_path),
            providers=providers,
            **model_kwargs,
        )
        if self._shared_dino is None:
            self._shared_dino = getattr(model, "dino", None)
        return model

    def _collect_train_images(self, dataset_name: str) -> Iterable[Path]:
        train_dir = self.dataset_root / dataset_name / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"train/good dir not found: {train_dir}")
        yield from self._iter_image_files(train_dir)

    def _collect_test_items(
        self,
        dataset_name: str,
    ) -> Iterable[dict[str, str | bool | Path]]:
        test_dir = self.dataset_root / dataset_name / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"test dir not found: {test_dir}")

        for category_dir in sorted(test_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            is_defect = category != "good"

            for image_path in self._iter_image_files(category_dir):
                yield {
                    "dataset_name": dataset_name,
                    "category": category,
                    "image_path": image_path,
                    "is_defect": is_defect,
                }

    def _iter_image_files(self, directory: Path) -> Iterable[Path]:
        supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in supported_suffixes:
                yield path

    def _load_rgb_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _predict_map(self, model: object, image_path: Path) -> np.ndarray:
        if self.use_mask_output:
            if not hasattr(model, "predict_mask"):
                raise AttributeError(
                    "model must implement predict_mask(image_path: str, threshold: float) "
                    "when use_mask_output=True"
                )
            anomaly_map = model.predict_mask(
                str(image_path),
                threshold=self.mask_threshold,
            )
        else:
            if not hasattr(model, "predict_anomaly_map"):
                raise AttributeError(
                    "model must implement predict_anomaly_map(image_path: str) -> np.ndarray"
                )
            anomaly_map = model.predict_anomaly_map(str(image_path))

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
        labels = np.asarray(labels)
        if labels.dtype != np.uint8:
            labels = labels.astype(np.int32)
        labels = labels.ravel()
        scores = np.asarray(scores)
        if scores.dtype != np.float32:
            scores = scores.astype(np.float64)
        scores = scores.ravel()
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
        # Tie ordering cannot affect counts sampled at the end of each tie, so
        # a stable sort is unnecessary.  NumPy's default quicksort is notably
        # faster for the millions of pixel scores in a MVTec category.
        order = np.argsort(scores)[::-1]
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
        negative_scores, region_scores = self._prepare_pro_scores(pro_examples)
        total_neg = negative_scores.size
        if total_neg == 0:
            return 0.0
        if not region_scores:
            return 0.0

        # searchsorted accepts every threshold at once.  This removes the hot
        # Python loop over thresholds while evaluating exactly the same points.
        false_positives = total_neg - np.searchsorted(
            negative_scores,
            thresholds,
            side="left",
        )
        fprs = false_positives.astype(np.float64) / total_neg
        overlap_sums = np.zeros(thresholds.size, dtype=np.float64)
        for scores in region_scores:
            overlap_sums += (
                scores.size
                - np.searchsorted(scores, thresholds, side="left")
            ) / scores.size
        pros = overlap_sums / len(region_scores)

        # Several thresholds can have the same FPR. At a vertical segment, the
        # highest attainable PRO is the value relevant for subsequent area.
        # Thresholds decrease monotonically, so FPR and PRO increase
        # monotonically as well; the final point of each equal-FPR run is its
        # maximum-PRO point.
        keep = np.concatenate((fprs[1:] != fprs[:-1], [True]))
        fprs = fprs[keep]
        pros = pros[keep]
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
            gt_mask = np.asarray(example["gt_mask"], dtype=np.uint8)
            scores = example["scores"]
            negative_scores.append(scores[gt_mask == 0])

            # Good images contribute negatives but contain no regions.  They
            # are common in MVTec, so avoid allocating a full label image.
            if not np.any(gt_mask):
                continue
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

    def _append_results(self, results: Iterable[dict[str, float | str]]) -> None:
        fieldnames = [
            "date",
            "datasets",
            "img_auroc",
            "img_aupr",
            "seg_auroc",
            "seg_pro",
            "description",
        ]
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.result_path.exists() or self.result_path.stat().st_size == 0
        if not write_header:
            with open(self.result_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_fieldnames = reader.fieldnames
                existing_results = list(reader)

            if existing_fieldnames != fieldnames:
                if set(existing_fieldnames or []) != set(fieldnames):
                    raise ValueError(
                        f"Unexpected CSV columns in {self.result_path}: "
                        f"{existing_fieldnames}"
                    )
                with open(
                    self.result_path,
                    mode="w",
                    newline="",
                    encoding="utf-8",
                ) as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(
                        self._format_result(result) for result in existing_results
                    )

        with open(self.result_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for result in results:
                writer.writerow(self._format_result(result))

    @staticmethod
    def _format_result(result: dict[str, float | str]) -> dict[str, float | str]:
        metric_names = {
            "img_auroc",
            "img_aupr",
            "seg_auroc",
            "seg_pro",
        }
        return {
            key: f"{float(value):.5f}" if key in metric_names else value
            for key, value in result.items()
        }
