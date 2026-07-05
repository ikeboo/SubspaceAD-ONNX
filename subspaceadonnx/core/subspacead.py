# subspacead_dinov3.py

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional, Tuple
from tqdm.auto import tqdm
import cv2
import numpy as np


# 既存のDINOv3 ONNXラッパーを使う想定
# 例:
from .dinov3_onnx import DINOv3


class SubspaceAD:
    """
    DINOv3 patch token + PCA subspace anomaly detection.

    Expected usage:
        model = SubspaceAD("dinov3_vitsplus")
        model.fit(normal_imgs)
        anomaly_map = model(target_img)

    Assumed DINOv3 interface:
        cls_token, patch_tokens = dinov3(img: np.ndarray)

    patch_tokens can be one of:
        [N, C]
        [1, N, C]
        [H_patch, W_patch, C]
        [1, H_patch, W_patch, C]
    """

    IMAGE_EXTENSIONS = frozenset(
        {
            ".bmp",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }
    )

    def __init__(
        self,
        model_name: str = "dinov3_vitsplus",
        *,
        dino=None,
        providers: list[str] | None = None,
        pca_ev: Optional[float] = 0.99,
        pca_dim: Optional[int] = None,
        max_fit_tokens: Optional[int] = None,
        feature_l2_normalize: bool = False,
        branch_fusion: str = "mean",
        spatial_centering: float = 1.0,
        score_transform: str = "log",
        normalize_map: bool = False,
        calibration_target: float = 0.5,
        calibration_fraction: float = 0.1,
        score_offset_quantile: float = 0.01,
        threshold_quantile: float = 0.995,
        image_threshold_quantile: float = 0.99,
        blur: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
    ):
        """
        Args:
            model_name:
                DINOv3ラッパーに渡すモデル名。
            dino:
                既に生成済みのDINOv3インスタンスを渡す場合に使用。
                Noneの場合は DINOv3(model_name) を呼ぶ。
            providers:
                DINOv3のONNX Runtimeに渡すExecution Providerの優先順。
                dinoを直接渡す場合は指定できません。
            pca_ev:
                PCAで保持する累積寄与率。pca_dimがNoneのとき有効。
                公式実装のデフォルトに近い 0.99 を既定値にしています。
            pca_dim:
                PCA主成分数を直接指定する場合に使用。
            max_fit_tokens:
                正常画像が多い場合、PCAに使うpatch token数をランダムに制限。
                few-shotならNoneで問題ありません。
            feature_l2_normalize:
                DINOv3ラッパーが未正規化特徴を返す場合にTrueを検討。
                既にx_norm_patchtokens相当ならFalse推奨。
            branch_fusion:
                ONNXが複数のpatch token列を返す場合のscore統合方法。
                各出力に独立PCAを学習し、現在は"mean"で平均します。
            spatial_centering:
                各patch位置の正常特徴平均をPCA前に差し引く強さ。0.0は従来の
                global PCA、1.0は完全な位置中心化です。特徴そのものは保持せず、
                位置ごとの平均だけを保存します。
            score_transform:
                PCA残差に適用する単調変換。"squared"、"sqrt"、"log"から選択。
                logは大きな正常残差の影響を抑え、局所異常を見やすくします。
            normalize_map:
                Trueの場合、共通スケーリング後の異常マップをさらに画像ごとに
                0-1正規化する。画像間で共通の閾値を使う場合はFalseにする。
            calibration_target:
                fit画像全体の異常マップ最大値を合わせる値。
            calibration_fraction:
                PCA学習から除外し、閾値校正に使う正常画像の割合。
            score_offset_quantile:
                正常スコアから差し引く下側分位点。
            threshold_quantile:
                holdout正常画素からpixel閾値を決める分位点。
            image_threshold_quantile:
                holdout正常画像の最大値からimage閾値を決める分位点。
            blur:
                異常マップをresize後に軽くGaussian blurする。
            eps:
                数値安定化用。
            random_state:
                max_fit_tokens使用時の乱数seed。
        """
        self.model_name = model_name
        self.providers = list(providers) if providers is not None else None
        if dino is not None and providers is not None:
            raise ValueError("providers cannot be specified when dino is provided.")
        self.dino = (
            dino
            if dino is not None
            else DINOv3(model_name, providers=self.providers)
        )

        if pca_ev is None and pca_dim is None:
            raise ValueError("pca_ev または pca_dim のどちらかは指定してください。")
        if calibration_target <= 0:
            raise ValueError("calibration_targetは0より大きい値にしてください。")
        if not 0.0 <= spatial_centering <= 1.0:
            raise ValueError("spatial_centeringは0.0から1.0の範囲にしてください。")
        if score_transform not in {"squared", "sqrt", "log"}:
            raise ValueError(
                "score_transformは'squared'、'sqrt'、'log'のいずれかにしてください。"
            )
        if branch_fusion != "mean":
            raise ValueError("branch_fusionは現在'mean'のみ対応しています。")
        if not 0.0 <= calibration_fraction < 1.0:
            raise ValueError("calibration_fractionは0.0以上1.0未満にしてください。")
        for name, value in (
            ("score_offset_quantile", score_offset_quantile),
            ("threshold_quantile", threshold_quantile),
            ("image_threshold_quantile", image_threshold_quantile),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}は0.0から1.0の範囲にしてください。")

        self.pca_ev = pca_ev
        self.pca_dim = pca_dim
        self.max_fit_tokens = max_fit_tokens
        self.feature_l2_normalize = feature_l2_normalize
        self.branch_fusion = branch_fusion
        self.spatial_centering = float(spatial_centering)
        self.score_transform = score_transform
        self.normalize_map = normalize_map
        self.calibration_target = float(calibration_target)
        self.calibration_fraction = float(calibration_fraction)
        self.score_offset_quantile = float(score_offset_quantile)
        self.threshold_quantile = float(threshold_quantile)
        self.image_threshold_quantile = float(image_threshold_quantile)
        self.blur = blur
        self.eps = eps
        self.random_state = random_state

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigvals_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.feature_dim_: Optional[int] = None
        self.position_mean_: Optional[np.ndarray] = None
        self.score_reference_: float = 1.0
        self.score_offset_: float = 0.0
        self.fit_max_score_: Optional[float] = None
        self.score_scale_: float = 1.0
        self.threshold_: float = self.calibration_target
        self.image_threshold_: float = self.calibration_target
        self.calibration_count_: int = 0
        self.branch_models_: list[SubspaceAD] = []

    def fit(self, imgs: list[np.ndarray] | str | Path) -> "SubspaceAD":
        """
        正常画像リストまたはディレクトリから正常部分空間を作成する。

        Args:
            imgs:
                正常画像のlist、または正常画像を含むディレクトリ。
                listの各要素は np.ndarray で、DINOv3ラッパーが受け付ける
                形式に合わせてください。ディレクトリの場合は配下を再帰的に
                探索し、対応する画像ファイルをOpenCVのBGRで読み込みます。

        Returns:
            self
        """
        if isinstance(imgs, (str, Path)):
            imgs = self.load_images_from_directory(imgs)

        if len(imgs) == 0:
            raise ValueError("fitには少なくとも1枚の正常画像が必要です。")

        extracted = []

        for img in tqdm(imgs, desc="Extracting features",unit="images"):
            feat, grid_size = self._extract_patch_features(img)
            extracted.append((feat, grid_size, img.shape[:2]))

        fit_extracted, calibration_extracted = self._split_fit_calibration(extracted)
        if extracted[0][0].ndim == 3:
            self._fit_branches(fit_extracted)
            self._fit_score_scale(extracted, calibration_extracted)
            return self

        self._fit_position_center(fit_extracted)
        centered_features = [
            self._center_features(feat) for feat, _, _ in fit_extracted
        ]
        X = np.concatenate(centered_features, axis=0).astype(np.float64)

        if self.max_fit_tokens is not None and X.shape[0] > self.max_fit_tokens:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.max_fit_tokens, replace=False)
            X = X[idx]
        print(f"Fitting PCA on {X.shape[0]} patch tokens with feature dim {X.shape[1]}.")
        self._fit_pca(X)
        self._fit_score_reference(X)
        self._fit_score_scale(extracted, calibration_extracted)
        return self

    def _fit_branches(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ) -> None:
        """Fit one compact PCA per ONNX patch-token output."""
        branch_count = extracted[0][0].shape[0]
        if any(item[0].ndim != 3 or item[0].shape[0] != branch_count for item in extracted):
            raise ValueError("All fit images must return the same patch-token branches.")

        self.branch_models_ = []
        for branch_index in range(branch_count):
            branch = self._new_branch_model()
            branch_extracted = [
                (features[branch_index], grid_size, output_size)
                for features, grid_size, output_size in extracted
            ]
            branch._fit_position_center(branch_extracted)
            x = np.concatenate(
                [branch._center_features(item[0]) for item in branch_extracted],
                axis=0,
            ).astype(np.float64)
            if self.max_fit_tokens is not None and x.shape[0] > self.max_fit_tokens:
                rng = np.random.default_rng(self.random_state)
                indices = rng.choice(x.shape[0], size=self.max_fit_tokens, replace=False)
                x = x[indices]
            print(
                f"Fitting branch {branch_index + 1}/{branch_count} PCA on "
                f"{x.shape[0]} patch tokens with feature dim {x.shape[1]}."
            )
            branch._fit_pca(x)
            branch._fit_score_reference(x)
            self.branch_models_.append(branch)

    def _new_branch_model(self) -> "SubspaceAD":
        return SubspaceAD(
            self.model_name,
            dino=object(),
            pca_ev=self.pca_ev,
            pca_dim=self.pca_dim,
            max_fit_tokens=self.max_fit_tokens,
            feature_l2_normalize=self.feature_l2_normalize,
            spatial_centering=self.spatial_centering,
            score_transform=self.score_transform,
            normalize_map=self.normalize_map,
            calibration_target=self.calibration_target,
            calibration_fraction=0.0,
            score_offset_quantile=self.score_offset_quantile,
            threshold_quantile=self.threshold_quantile,
            image_threshold_quantile=self.image_threshold_quantile,
            blur=self.blur,
            eps=self.eps,
            random_state=self.random_state,
        )

    def _split_fit_calibration(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ) -> tuple[
        list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
        list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ]:
        """Reserve normal images for threshold calibration when possible."""
        n_images = len(extracted)
        if self.calibration_fraction <= 0.0 or n_images < 10:
            self.calibration_count_ = n_images
            return extracted, extracted

        n_calibration = max(1, int(round(n_images * self.calibration_fraction)))
        n_calibration = min(n_calibration, n_images - 2)
        rng = np.random.default_rng(self.random_state)
        calibration_indices = set(
            rng.choice(n_images, size=n_calibration, replace=False).tolist()
        )
        fit_extracted = [
            item for index, item in enumerate(extracted)
            if index not in calibration_indices
        ]
        calibration_extracted = [
            item for index, item in enumerate(extracted)
            if index in calibration_indices
        ]
        self.calibration_count_ = len(calibration_extracted)
        return fit_extracted, calibration_extracted

    def _fit_position_center(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ) -> None:
        """Fit a compact position-conditioned normal mean.

        A single mean vector per patch position captures the expected object layout
        without retaining any training tokens.  Interpolating it with the global
        mean makes ``spatial_centering=0`` exactly equivalent to global centering.
        """
        first_features, first_grid, _ = extracted[0]
        n_patches, feature_dim = first_features.shape
        for features, grid_size, _ in extracted[1:]:
            if grid_size != first_grid or features.shape != (n_patches, feature_dim):
                raise ValueError(
                    "All fit images must produce the same patch grid and feature dim: "
                    f"expected grid={first_grid}, features={(n_patches, feature_dim)}, "
                    f"got grid={grid_size}, features={features.shape}"
                )

        position_mean = np.zeros((n_patches, feature_dim), dtype=np.float64)
        for features, _, _ in extracted:
            position_mean += features
        position_mean /= len(extracted)
        global_mean = np.mean(position_mean, axis=0, keepdims=True)
        self.position_mean_ = (
            global_mean
            + self.spatial_centering * (position_mean - global_mean)
        ).astype(np.float32)

    def _center_features(self, features: np.ndarray) -> np.ndarray:
        if self.position_mean_ is None:
            return features
        if features.shape != self.position_mean_.shape:
            raise ValueError(
                "patch feature shape mismatch for spatial centering: "
                f"got {features.shape}, expected {self.position_mean_.shape}"
            )
        return features - self.position_mean_

    @classmethod
    def load_images_from_directory(
        cls,
        directory_path: str | Path,
    ) -> list[np.ndarray]:
        """ディレクトリ配下の対応画像ファイルをOpenCVのBGRで読み込む。

        サブディレクトリも再帰的に探索し、パス順に読み込みます。画像以外の
        ファイルは無視します。

        Args:
            directory_path: 画像を含むディレクトリ。

        Returns:
            BGR画像のlist。
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"image directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"image directory is not a directory: {directory}"
            )

        image_paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in: {directory}")

        images = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            images.append(image)

        return images

    def __call__(
        self,
        target_img: np.ndarray,
        *,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_map: Optional[bool] = None,
        return_patch_map: bool = False,
    ) -> np.ndarray:
        """
        推論して異常度マップを返す。

        Args:
            target_img:
                推論対象画像。
            output_size:
                出力サイズ。Noneなら target_img.shape[:2] に合わせる。
                指定形式は (height, width)。
            normalize_map:
                共通スケーリング後に画像ごとの0-1正規化も行うか。
                Noneならself.normalize_map。共通閾値を使う場合はFalse。
            return_patch_map:
                Trueならpatch grid解像度の異常マップを返す。
                Falseなら画像サイズへresizeしたマップを返す。

        Returns:
            anomaly_map:
                shape = [H, W] の np.float32。
        """
        self._check_fitted()

        feat, grid_size = self._extract_patch_features(target_img)
        scores = self._score_features(feat)

        if output_size is None:
            output_size = target_img.shape[:2]

        out = self._scores_to_map(
            scores,
            grid_size,
            output_size,
            return_patch_map=return_patch_map,
        )
        out = self._calibrate_map(out)

        do_norm = self.normalize_map if normalize_map is None else normalize_map
        if do_norm:
            out = self._minmax_norm(out)

        return out.astype(np.float32)

    def _fit_score_scale(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
        calibration_extracted: Optional[
            list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]
        ] = None,
    ) -> None:
        """Fit affine map calibration and normal-only operating thresholds."""
        if calibration_extracted is None:
            calibration_extracted = extracted

        scored = []
        patch_scores = []
        for feat, grid_size, output_size in extracted:
            scores = self._score_features(feat)
            scored.append((scores, grid_size, output_size))
            patch_scores.append(scores)

        self.score_offset_ = float(
            np.quantile(
                np.concatenate(patch_scores),
                self.score_offset_quantile,
            )
        )
        fit_max = 0.0
        for scores, grid_size, output_size in tqdm(
            scored,
            desc="Calibrating anomaly-map scale",
            unit="images"
        ):
            anomaly_map = self._scores_to_map(scores, grid_size, output_size)
            fit_max = max(fit_max, float(np.max(anomaly_map)))

        self.fit_max_score_ = fit_max
        score_range = fit_max - self.score_offset_
        if score_range <= self.eps:
            self.score_scale_ = 1.0
            warnings.warn(
                "fit画像の異常スコア範囲がほぼ0のため、係数を1.0にしました。",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            self.score_scale_ = self.calibration_target / score_range

        calibration_pixels = []
        calibration_maxima = []
        for feat, grid_size, output_size in calibration_extracted:
            scores = self._score_features(feat)
            anomaly_map = self._scores_to_map(
                scores,
                grid_size,
                output_size,
            )
            anomaly_map = self._calibrate_map(anomaly_map)
            calibration_pixels.append(anomaly_map.ravel())
            calibration_maxima.append(float(np.max(anomaly_map)))

        self.threshold_ = float(
            np.quantile(
                np.concatenate(calibration_pixels),
                self.threshold_quantile,
            )
        )
        self.image_threshold_ = float(
            np.quantile(
                np.asarray(calibration_maxima),
                self.image_threshold_quantile,
            )
        )

        print(
            "Anomaly-map scale calibrated: "
            f"offset={self.score_offset_:.6g}, fit_max={self.fit_max_score_:.6g}, "
            f"scale={self.score_scale_:.6g}, "
            f"scaled_max={(self.fit_max_score_ - self.score_offset_) * self.score_scale_:.6g}, "
            f"pixel_threshold={self.threshold_:.6g}, "
            f"image_threshold={self.image_threshold_:.6g}"
        )

    def _calibrate_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        return (
            np.maximum(anomaly_map - self.score_offset_, 0.0)
            * self.score_scale_
        )

    def _scores_to_map(
        self,
        scores: np.ndarray,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
        *,
        return_patch_map: bool = False,
    ) -> np.ndarray:
        """patch scoreを推論時と同じ後処理で異常マップへ変換する。"""
        h_p, w_p = grid_size
        patch_map = scores.reshape(h_p, w_p).astype(np.float32)

        if return_patch_map:
            return patch_map

        out_h, out_w = output_size
        out = cv2.resize(
            patch_map,
            (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.blur:
            out = cv2.GaussianBlur(out, (3, 3), sigmaX=4.0)

        return out

    def image_score(
        self,
        target_img: np.ndarray,
        *,
        method: str = "max",
    ) -> float:
        """
        異常マップから画像単位の異常スコアを作る補助関数。

        Args:
            target_img:
                入力画像。
            method:
                画像スコアの集約方法。
                - "max": 最大異常値をそのまま用いる。学習済みの
                  image_threshold_に対応する既定値。
                - "mean": 異常マップ全体の平均値を用いる。
                - "p99": 99パーセンタイル値を用いる。
                - "mtop5": 最高異常値上位5個の平均を用いる。
                - "mtop1p": 最高異常値上位1%の平均を用いる。
        """
        anomaly_map = self(target_img, normalize_map=False)

        if method == "max":
            return float(np.max(anomaly_map))

        if method == "mean":
            return float(np.mean(anomaly_map))

        if method == "p99":
            return float(np.percentile(anomaly_map, 99))

        if method == "mtop5":
            flat = anomaly_map.ravel()
            k = min(5, flat.size)
            return float(np.mean(np.partition(flat, -k)[-k:]))

        if method == "mtop1p":
            flat = anomaly_map.ravel()
            k = max(1, int(flat.size * 0.01))
            return float(np.mean(np.partition(flat, -k)[-k:]))

        raise ValueError(f"Unknown image score method: {method}")

    def _extract_patch_features(
        self,
        img: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        DINOv3からpatch tokenを取り出し、[N, C]に変換する。
        """
        # SubspaceADの公開入力はOpenCV BGR。色変換せずDINOv3へ渡す。
        out = self.dino(img)

        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                "DINOv3(img) は (cls_token, patch_tokens) を返す必要があります。"
            )

        features = []
        grid_size = None
        for patch_tokens in out[1:]:
            patch_grid, current_grid = self._patch_tokens_to_grid(patch_tokens, img)
            if grid_size is not None and current_grid != grid_size:
                raise ValueError(
                    "All patch-token outputs must use the same grid: "
                    f"got {grid_size} and {current_grid}"
                )
            grid_size = current_grid
            feat = patch_grid.reshape(-1, patch_grid.shape[-1]).astype(np.float32)
            if self.feature_l2_normalize:
                norm = np.linalg.norm(feat, axis=1, keepdims=True)
                feat = feat / (norm + self.eps)
            features.append(feat)

        if len(features) == 1:
            return features[0], grid_size
        return np.stack(features, axis=0), grid_size

    def _patch_tokens_to_grid(
        self,
        patch_tokens: np.ndarray,
        img: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        patch tokenを [H_patch, W_patch, C] に揃える。
        """
        tokens = np.asarray(patch_tokens)

        if tokens.ndim == 4:
            if tokens.shape[0] != 1:
                raise ValueError(
                    f"batch付きpatch_tokensはbatch=1のみ対応です: {tokens.shape}"
                )
            tokens = tokens[0]

        if tokens.ndim == 3 and tokens.shape[0] == 1:
            tokens = tokens[0]

        if tokens.ndim == 3:
            h_p, w_p, c = tokens.shape
            if c <= 4:
                raise ValueError(
                    f"patch_tokensの最後の次元が特徴次元に見えません: {tokens.shape}"
                )
            return tokens.astype(np.float32), (h_p, w_p)

        if tokens.ndim == 2:
            n, c = tokens.shape
            h_p, w_p = self._infer_grid_size(n, img)
            if h_p * w_p != n:
                raise ValueError(
                    f"patch数 {n} を grid に変換できません: inferred={(h_p, w_p)}"
                )
            return tokens.reshape(h_p, w_p, c).astype(np.float32), (h_p, w_p)

        raise ValueError(f"Unsupported patch_tokens shape: {tokens.shape}")

    def _infer_grid_size(
        self,
        n_patches: int,
        img: np.ndarray,
    ) -> tuple[int, int]:
        """
        flattenされたpatch token数からgrid sizeを推定する。
        可能ならDINOv3ラッパー側の属性を優先する。
        """
        for attr_name in ("patch_grid", "grid_size", "last_grid_size"):
            grid = getattr(self.dino, attr_name, None)
            if callable(grid):
                grid = grid()

            if grid is not None and len(grid) == 2:
                h_p, w_p = int(grid[0]), int(grid[1])
                if h_p * w_p == n_patches:
                    return h_p, w_p

        s = int(math.sqrt(n_patches))
        if s * s == n_patches:
            return s, s

        # 入力画像のアスペクト比に近いfactor pairを探す
        img_h, img_w = img.shape[:2]
        target_ratio = img_h / max(img_w, 1)

        candidates = []
        for h in range(1, int(math.sqrt(n_patches)) + 1):
            if n_patches % h == 0:
                w = n_patches // h
                candidates.append((h, w))
                candidates.append((w, h))

        if not candidates:
            raise ValueError(f"patch数からgridを推定できません: {n_patches}")

        h_p, w_p = min(
            candidates,
            key=lambda hw: abs((hw[0] / max(hw[1], 1)) - target_ratio),
        )
        return h_p, w_p

    def _fit_pca(self, X: np.ndarray) -> None:
        """
        PCA正常部分空間を作る。
        X shape: [num_tokens, feature_dim]
        """
        if X.ndim != 2:
            raise ValueError(f"X must be [N, C], got {X.shape}")

        n, c = X.shape
        if n < 2:
            raise ValueError("PCAには少なくとも2個以上のpatch tokenが必要です。")

        self.feature_dim_ = c
        self.mean_ = X.mean(axis=0, keepdims=True)

        Xc = X - self.mean_

        cov = (Xc.T @ Xc) / max(n - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals = np.maximum(eigvals, 0.0)

        if self.pca_dim is not None:
            k = int(self.pca_dim)
        else:
            total = float(np.sum(eigvals))
            if total <= self.eps:
                k = 1
            else:
                cum = np.cumsum(eigvals) / total
                k = int(np.searchsorted(cum, self.pca_ev) + 1)

        k = max(1, min(k, c))

        self.n_components_ = k
        self.eigvals_ = eigvals[:k].astype(np.float64)
        self.components_ = eigvecs[:, :k].astype(np.float64)

    def _squared_spe_centered(self, X: np.ndarray) -> np.ndarray:
        """Return squared PCA residuals for position-centered features."""
        X = X.astype(np.float64)
        if X.shape[1] != self.feature_dim_:
            raise ValueError(
                f"feature dim mismatch: got {X.shape[1]}, expected {self.feature_dim_}"
            )

        Xc = X - self.mean_
        z = Xc @ self.components_
        # Components are orthonormal, so the squared reconstruction residual is
        # ||Xc||^2 - ||projection||^2. Avoid materializing the reconstruction.
        scores = np.sum(Xc * Xc, axis=1) - np.sum(z * z, axis=1)
        return np.maximum(scores, 0.0)

    def _fit_score_reference(self, centered_features: np.ndarray) -> None:
        """Fit a data-scale reference so log compression adds no huge offset."""
        raw_scores = self._squared_spe_centered(centered_features)
        positive_scores = raw_scores[raw_scores > self.eps]
        if positive_scores.size == 0:
            self.score_reference_ = 1.0
        else:
            self.score_reference_ = max(
                float(np.median(positive_scores)),
                self.eps,
            )

    def _score_features(self, X: np.ndarray) -> np.ndarray:
        """Use PCA reconstruction residuals as patch anomaly scores."""
        self._check_fitted()
        if self.branch_models_:
            if X.ndim != 3 or X.shape[0] != len(self.branch_models_):
                raise ValueError(
                    f"Expected {len(self.branch_models_)} feature branches, got {X.shape}"
                )
            branch_scores = [
                model._score_features(X[index])
                for index, model in enumerate(self.branch_models_)
            ]
            return np.mean(np.stack(branch_scores, axis=0), axis=0).astype(np.float32)

        centered = self._center_features(X)
        scores = self._squared_spe_centered(centered)

        if self.score_transform == "sqrt":
            scores = np.sqrt(scores)
        elif self.score_transform == "log":
            scores = np.log1p(scores / self.score_reference_)

        return scores.astype(np.float32)

    def _check_fitted(self) -> None:
        if self.branch_models_:
            if all(model.mean_ is not None and model.components_ is not None for model in self.branch_models_):
                return
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("SubspaceAD is not fitted. Call model.fit(normal_imgs) first.")

    def _minmax_norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn = float(np.min(x))
        mx = float(np.max(x))

        if mx - mn < self.eps:
            return np.zeros_like(x, dtype=np.float32)

        return (x - mn) / (mx - mn + self.eps)

    def state_dict(self) -> dict:
        """
        DINOv3本体を除くPCA状態だけ保存したい場合用。
        """
        self._check_fitted()
        return {
            "model_name": self.model_name,
            "pca_ev": self.pca_ev,
            "pca_dim": self.pca_dim,
            "feature_l2_normalize": self.feature_l2_normalize,
            "branch_fusion": self.branch_fusion,
            "branches": [model.state_dict() for model in self.branch_models_],
            "spatial_centering": self.spatial_centering,
            "score_transform": self.score_transform,
            "normalize_map": self.normalize_map,
            "calibration_target": self.calibration_target,
            "calibration_fraction": self.calibration_fraction,
            "score_offset_quantile": self.score_offset_quantile,
            "threshold_quantile": self.threshold_quantile,
            "image_threshold_quantile": self.image_threshold_quantile,
            "blur": self.blur,
            "eps": self.eps,
            "mean": self.mean_,
            "components": self.components_,
            "eigvals": self.eigvals_,
            "n_components": self.n_components_,
            "feature_dim": self.feature_dim_,
            "position_mean": self.position_mean_,
            "score_reference": self.score_reference_,
            "score_offset": self.score_offset_,
            "fit_max_score": self.fit_max_score_,
            "score_scale": self.score_scale_,
            "threshold": self.threshold_,
            "image_threshold": self.image_threshold_,
            "calibration_count": self.calibration_count_,
        }

    def save_npz(self, npz_path: str | Path) -> None:
        """Save the fitted PCA state and inference settings to an NPZ file.

        The DINOv3 model itself is not included.  Metadata is stored as JSON so
        that the resulting file can be loaded without enabling pickle support.

        Args:
            npz_path: Destination path. ``.npz`` is appended by NumPy when the
                supplied path does not already end with that suffix.
        """
        state = self.state_dict()
        is_multi_branch = bool(state["branches"])
        metadata = {
            "format_version": 4 if is_multi_branch else 3,
            "model_name": str(state["model_name"]),
            "pca_ev": (
                None if state["pca_ev"] is None else float(state["pca_ev"])
            ),
            "pca_dim": (
                None if state["pca_dim"] is None else int(state["pca_dim"])
            ),
            "feature_l2_normalize": bool(state["feature_l2_normalize"]),
            "branch_fusion": str(state["branch_fusion"]),
            "spatial_centering": float(state["spatial_centering"]),
            "score_transform": str(state["score_transform"]),
            "normalize_map": bool(state["normalize_map"]),
            "calibration_target": float(state["calibration_target"]),
            "calibration_fraction": float(state["calibration_fraction"]),
            "score_offset_quantile": float(state["score_offset_quantile"]),
            "threshold_quantile": float(state["threshold_quantile"]),
            "image_threshold_quantile": float(
                state["image_threshold_quantile"]
            ),
            "blur": bool(state["blur"]),
            "eps": float(state["eps"]),
            "score_offset": float(state["score_offset"]),
            "fit_max_score": float(state["fit_max_score"] or 0.0),
            "score_scale": float(state["score_scale"]),
            "threshold": float(state["threshold"]),
            "image_threshold": float(state["image_threshold"]),
            "calibration_count": int(state["calibration_count"]),
        }
        arrays = {"metadata": np.asarray(json.dumps(metadata))}
        if is_multi_branch:
            branch_metadata = []
            for index, branch in enumerate(state["branches"]):
                branch_metadata.append({
                    key: value
                    for key, value in branch.items()
                    if key not in {"mean", "components", "eigvals", "position_mean", "branches"}
                })
                arrays[f"branch_{index}_mean"] = branch["mean"]
                arrays[f"branch_{index}_components"] = branch["components"]
                arrays[f"branch_{index}_eigvals"] = branch["eigvals"]
                if branch["position_mean"] is not None:
                    arrays[f"branch_{index}_position_mean"] = branch["position_mean"]
            metadata["branch_metadata"] = branch_metadata
            arrays["metadata"] = np.asarray(json.dumps(metadata))
        else:
            metadata.update({
                "n_components": int(state["n_components"]),
                "feature_dim": int(state["feature_dim"]),
                "score_reference": float(state["score_reference"]),
            })
            arrays.update({
                "metadata": np.asarray(json.dumps(metadata)),
                "mean": state["mean"],
                "components": state["components"],
                "eigvals": state["eigvals"],
            })
            if state["position_mean"] is not None:
                arrays["position_mean"] = state["position_mean"]
        np.savez_compressed(npz_path, **arrays)

    def load_npz(self, npz_path: str | Path) -> "SubspaceAD":
        """Load PCA state and inference settings saved by :meth:`save_npz`.

        The current DINOv3 instance is retained, so it must be compatible with
        the feature dimension of the model used when fitting.

        Args:
            npz_path: Path to an NPZ file created by :meth:`save_npz`.

        Returns:
            self
        """
        with np.load(npz_path, allow_pickle=False) as saved:
            missing_keys = {"metadata"}.difference(saved.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"Invalid SubspaceAD NPZ: missing keys: {missing}")

            try:
                metadata = json.loads(str(saved["metadata"].item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("Invalid SubspaceAD NPZ metadata.") from exc

            if not isinstance(metadata, dict):
                raise ValueError("Invalid SubspaceAD NPZ metadata.")

            format_version = metadata.get("format_version")
            if format_version not in {1, 2, 3, 4}:
                raise ValueError(
                    "Unsupported SubspaceAD NPZ format version: "
                    f"{metadata.get('format_version')!r}"
                )

            if format_version == 4:
                branch_metadata = metadata.pop("branch_metadata", None)
                if not isinstance(branch_metadata, list) or not branch_metadata:
                    raise ValueError("Invalid multi-branch SubspaceAD NPZ metadata.")
                branches = []
                for index, branch_meta in enumerate(branch_metadata):
                    prefix = f"branch_{index}_"
                    required = {prefix + name for name in ("mean", "components", "eigvals")}
                    missing = required.difference(saved.files)
                    if missing:
                        raise ValueError(
                            "Invalid SubspaceAD NPZ: missing keys: "
                            + ", ".join(sorted(missing))
                        )
                    branch = {
                        **branch_meta,
                        "mean": np.array(saved[prefix + "mean"], copy=True),
                        "components": np.array(saved[prefix + "components"], copy=True),
                        "eigvals": np.array(saved[prefix + "eigvals"], copy=True),
                    }
                    position_key = prefix + "position_mean"
                    if position_key in saved.files:
                        branch["position_mean"] = np.array(saved[position_key], copy=True)
                    branches.append(branch)
                state = {**metadata, "branches": branches}
            else:
                required_keys = {"mean", "components", "eigvals"}
                missing_keys = required_keys.difference(saved.files)
                if missing_keys:
                    missing = ", ".join(sorted(missing_keys))
                    raise ValueError(f"Invalid SubspaceAD NPZ: missing keys: {missing}")
                state = {
                    **metadata,
                    "mean": np.array(saved["mean"], copy=True),
                    "components": np.array(saved["components"], copy=True),
                    "eigvals": np.array(saved["eigvals"], copy=True),
                }
                if "position_mean" in saved.files:
                    state["position_mean"] = np.array(
                        saved["position_mean"], copy=True
                    )

        self.load_state_dict(state)
        return self

    def predict_anomaly_map(
        self,
        image_path: str,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_map: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Reads an image from disk and returns the anomaly map.

        Args:
            image_path: Path to the image file.
            output_size: Size of the output anomaly map / mask (height, width).
            normalize_map: Whether to apply per-image min-max normalization.
                If None, self.normalize_map is used.

        Returns:
            anomaly_map: Floating point anomaly map of shape [H, W].
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image_path not found: {image_path}")

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        return self(
            img,
            output_size=output_size,
            normalize_map=normalize_map,
        )

    def predict_mask(
        self,
        image_path: str,
        threshold: Optional[float] = None,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_map: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Reads an image from disk, computes the anomaly map, and returns a binary mask.

        Args:
            image_path: Path to the image file.
            threshold: Threshold applied to the anomaly map. If None, use the
                normal holdout threshold learned during fit.
            output_size: Size of the output anomaly map / mask (height, width).
            normalize_map: Whether to apply per-image min-max normalization.
                If None, self.normalize_map is used.

        Returns:
            mask: Binary mask of shape [H, W], dtype uint8, values 0/255.
        """
        anomaly_map = self.predict_anomaly_map(
            image_path,
            output_size=output_size,
            normalize_map=normalize_map,
        )
        if threshold is None:
            threshold = self.threshold_
        return (anomaly_map >= threshold).astype(np.uint8) * 255

    def load_state_dict(self, state: dict) -> None:
        """
        DINOv3インスタンス作成後にPCA状態を復元する。
        """
        self.pca_ev = state["pca_ev"]
        self.pca_dim = state["pca_dim"]
        self.feature_l2_normalize = state["feature_l2_normalize"]
        self.branch_fusion = str(state.get("branch_fusion", "mean"))
        self.spatial_centering = float(state.get("spatial_centering", 0.0))
        self.score_transform = str(state.get("score_transform", "squared"))
        self.normalize_map = state["normalize_map"]
        self.calibration_target = float(
            state["calibration_target"] if "calibration_target" in state else 0.5
        )
        self.calibration_fraction = float(state.get("calibration_fraction", 0.0))
        self.score_offset_quantile = float(
            state.get("score_offset_quantile", 0.0)
        )
        self.threshold_quantile = float(state.get("threshold_quantile", 1.0))
        self.image_threshold_quantile = float(
            state.get("image_threshold_quantile", 1.0)
        )
        self.blur = state["blur"]
        self.eps = state["eps"]

        self.branch_models_ = []
        for branch_state in state.get("branches", []):
            branch = self._new_branch_model()
            branch.load_state_dict(branch_state)
            self.branch_models_.append(branch)

        self.mean_ = state.get("mean")
        self.components_ = state.get("components")
        self.eigvals_ = state.get("eigvals")
        self.n_components_ = state.get("n_components")
        self.feature_dim_ = state.get("feature_dim")
        self.position_mean_ = state.get("position_mean")
        self.score_reference_ = float(
            state.get(
                "score_reference",
                self.eps if self.score_transform == "log" else 1.0,
            )
        )
        self.score_offset_ = float(state.get("score_offset", 0.0))
        self.fit_max_score_ = float(
            (state["fit_max_score"] if "fit_max_score" in state else 0.0) or 0.0
        )
        self.score_scale_ = float(
            state["score_scale"] if "score_scale" in state else 1.0
        )
        self.threshold_ = float(state.get("threshold", 0.5))
        self.image_threshold_ = float(state.get("image_threshold", 0.5))
        self.calibration_count_ = int(state.get("calibration_count", 0))
