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
        mixture_components: int = 2,
        mixture_descriptor_grid: int = 4,
        mixture_min_separation: float = 0.08,
        mixture_min_fraction: float = 0.15,
        mixture_min_images: int = 8,
        score_transform: str = "log",
        multiband_pca_ev: Optional[float] = 0.95,
        multiband_score_weight: float = 0.25,
        tail_score_quantile: Optional[float] = 0.99,
        tail_score_gain: float = 0.25,
        branch_local_tail_quantile: Optional[float] = 0.99,
        branch_local_tail_gain: float = 1.0,
        branch_local_tail_min_position_variance: float = 0.1,
        branch_local_tail_max_position_variance: float = 0.5,
        position_local_tail_quantile: Optional[float] = 0.95,
        position_local_tail_gain: float = 0.5,
        position_local_tail_min_spatial_correlation: float = 0.8,
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
            mixture_components:
                正常画像の空間的な外観を最大いくつのモードで表すか。2の場合、
                低解像度化したpatch特徴を2-meansで分け、十分に分離したときだけ
                モード別の位置平均を使います。PCA部分空間は共有します。
            mixture_descriptor_grid:
                正常モードの選択に使うpatch特徴の空間poolingサイズ。
            mixture_min_separation:
                2モードを有効にするbetween/total分散比の下限。
            mixture_min_fraction:
                各モードに必要な正常画像割合の下限。
            mixture_min_images:
                各モードに必要な正常画像数の下限。few-shot時は、全画像を2群に
                分けられるよう自動的に緩和されます。
            score_transform:
                PCA残差に適用する単調変換。"squared"、"sqrt"、"log"から選択。
                logは大きな正常残差の影響を抑え、局所異常を見やすくします。
            multiband_pca_ev:
                log-SPEに併用する粗いPCA部分空間の累積寄与率。0.95なら、
                通常の0.99部分空間で再構成される中分散方向も25%だけ異常度へ
                戻します。Noneで無効化します。追加の行列積は発生しません。
            multiband_score_weight:
                粗い部分空間のlog-SPEを混合する重み。0.0で従来の単帯域SPE、
                1.0で粗い部分空間だけを使用します。
            tail_score_quantile:
                正常patch scoreの上側分位点。ここを超えたscoreだけを増幅し、
                小さな異常領域のimage scoreへの寄与を保ちます。Noneで無効化。
            tail_score_gain:
                tail_score_quantileを超えた分に追加する線形gain。
            branch_local_tail_quantile:
                二枝の局所score差分がともに超えたときだけ増幅する正常分位点。
                単一枝では自動的に無効です。Noneで無効化します。
            branch_local_tail_gain:
                二枝で合意した局所tail超過量に掛けるgain。
            branch_local_tail_min_position_variance:
                局所tailを有効化する正常score位置分散比の下限。均一textureでは
                局所強調を行わず、正常な細かい模様の過検出を避けます。
            branch_local_tail_max_position_variance:
                正常score位置分散比の上限。強く位置固定された物体では既存の
                位置PCAを優先し、正常edgeの過強調を避けます。
            position_local_tail_quantile:
                空間的に連続した正常構造を持つカテゴリで使う、位置別の局所
                score差分の正常分位点。Noneで位置別方式を無効化します。
            position_local_tail_gain:
                位置別の局所tail超過量に掛けるgain。
            position_local_tail_min_spatial_correlation:
                位置別方式へ切り替える正常scoreと3x3近傍scoreの相関下限。
                下回る場合は二枝の合意を優先します。
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
        if mixture_components not in {1, 2}:
            raise ValueError("mixture_componentsは1または2にしてください。")
        if mixture_descriptor_grid < 1:
            raise ValueError("mixture_descriptor_gridは1以上にしてください。")
        if not 0.0 <= mixture_min_separation <= 1.0:
            raise ValueError("mixture_min_separationは0.0から1.0の範囲にしてください。")
        if not 0.0 < mixture_min_fraction <= 0.5:
            raise ValueError("mixture_min_fractionは0.0より大きく0.5以下にしてください。")
        if mixture_min_images < 1:
            raise ValueError("mixture_min_imagesは1以上にしてください。")
        if score_transform not in {"squared", "sqrt", "log"}:
            raise ValueError(
                "score_transformは'squared'、'sqrt'、'log'のいずれかにしてください。"
            )
        if branch_fusion != "mean":
            raise ValueError("branch_fusionは現在'mean'のみ対応しています。")
        if multiband_pca_ev is not None and not 0.0 < multiband_pca_ev <= 1.0:
            raise ValueError("multiband_pca_evは0より大きく1.0以下にしてください。")
        if not 0.0 <= multiband_score_weight <= 1.0:
            raise ValueError(
                "multiband_score_weightは0.0から1.0の範囲にしてください。"
            )
        if tail_score_quantile is not None and not 0.0 <= tail_score_quantile <= 1.0:
            raise ValueError("tail_score_quantileは0.0から1.0の範囲にしてください。")
        if tail_score_gain < 0.0:
            raise ValueError("tail_score_gainは0.0以上にしてください。")
        if (
            branch_local_tail_quantile is not None
            and not 0.0 <= branch_local_tail_quantile <= 1.0
        ):
            raise ValueError(
                "branch_local_tail_quantileは0.0から1.0の範囲にしてください。"
            )
        if branch_local_tail_gain < 0.0:
            raise ValueError("branch_local_tail_gainは0.0以上にしてください。")
        if not (
            0.0 <= branch_local_tail_min_position_variance
            <= branch_local_tail_max_position_variance
            <= 1.0
        ):
            raise ValueError(
                "branch local tailの位置分散比範囲は0.0以上1.0以下で、"
                "min <= maxにしてください。"
            )
        if (
            position_local_tail_quantile is not None
            and not 0.0 <= position_local_tail_quantile <= 1.0
        ):
            raise ValueError(
                "position_local_tail_quantileは0.0から1.0の範囲にしてください。"
            )
        if position_local_tail_gain < 0.0:
            raise ValueError("position_local_tail_gainは0.0以上にしてください。")
        if not 0.0 <= position_local_tail_min_spatial_correlation <= 1.0:
            raise ValueError(
                "position_local_tail_min_spatial_correlationは"
                "0.0から1.0の範囲にしてください。"
            )
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
        self.mixture_components = int(mixture_components)
        self.mixture_descriptor_grid = int(mixture_descriptor_grid)
        self.mixture_min_separation = float(mixture_min_separation)
        self.mixture_min_fraction = float(mixture_min_fraction)
        self.mixture_min_images = int(mixture_min_images)
        self.score_transform = score_transform
        self.multiband_pca_ev = (
            None if multiband_pca_ev is None else float(multiband_pca_ev)
        )
        self.multiband_score_weight = float(multiband_score_weight)
        self.tail_score_quantile = (
            None if tail_score_quantile is None else float(tail_score_quantile)
        )
        self.tail_score_gain = float(tail_score_gain)
        self.branch_local_tail_quantile = (
            None
            if branch_local_tail_quantile is None
            else float(branch_local_tail_quantile)
        )
        self.branch_local_tail_gain = float(branch_local_tail_gain)
        self.branch_local_tail_min_position_variance = float(
            branch_local_tail_min_position_variance
        )
        self.branch_local_tail_max_position_variance = float(
            branch_local_tail_max_position_variance
        )
        self.position_local_tail_quantile = (
            None
            if position_local_tail_quantile is None
            else float(position_local_tail_quantile)
        )
        self.position_local_tail_gain = float(position_local_tail_gain)
        self.position_local_tail_min_spatial_correlation = float(
            position_local_tail_min_spatial_correlation
        )
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
        self.mixture_position_means_: Optional[np.ndarray] = None
        self.mixture_descriptor_centers_: Optional[np.ndarray] = None
        self.mixture_separation_: float = 0.0
        self.mixture_cluster_sizes_: Optional[np.ndarray] = None
        self.patch_grid_: Optional[tuple[int, int]] = None
        self.score_reference_: float = 1.0
        self.multiband_components_: Optional[int] = None
        self.multiband_score_reference_: float = 1.0
        self.tail_score_reference_: float = 0.0
        self.score_offset_: float = 0.0
        self.fit_max_score_: Optional[float] = None
        self.score_scale_: float = 1.0
        self.threshold_: float = self.calibration_target
        self.image_threshold_: float = self.calibration_target
        self.calibration_count_: int = 0
        self.branch_models_: list[SubspaceAD] = []
        self.branch_local_tail_thresholds_: Optional[np.ndarray] = None
        self.branch_local_tail_enabled_: bool = False
        self.position_local_tail_thresholds_: Optional[np.ndarray] = None
        self.position_local_tail_enabled_: bool = False
        self.position_variance_ratio_: float = 0.0
        self.spatial_score_correlation_: float = 0.0

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

        self.patch_grid_ = tuple(extracted[0][1])
        fit_extracted, calibration_extracted = self._split_fit_calibration(extracted)
        if extracted[0][0].ndim == 3:
            self._fit_branches(fit_extracted)
            self._fit_branch_local_tail(extracted)
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
        self.patch_grid_ = tuple(extracted[0][1])
        for branch_index in range(branch_count):
            branch = self._new_branch_model()
            branch_extracted = [
                (features[branch_index], grid_size, output_size)
                for features, grid_size, output_size in extracted
            ]
            branch._fit_position_center(branch_extracted)
            branch.patch_grid_ = tuple(branch_extracted[0][1])
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
            mixture_components=self.mixture_components,
            mixture_descriptor_grid=self.mixture_descriptor_grid,
            mixture_min_separation=self.mixture_min_separation,
            mixture_min_fraction=self.mixture_min_fraction,
            mixture_min_images=self.mixture_min_images,
            score_transform=self.score_transform,
            multiband_pca_ev=self.multiband_pca_ev,
            multiband_score_weight=self.multiband_score_weight,
            tail_score_quantile=self.tail_score_quantile,
            tail_score_gain=self.tail_score_gain,
            branch_local_tail_quantile=self.branch_local_tail_quantile,
            branch_local_tail_gain=self.branch_local_tail_gain,
            branch_local_tail_min_position_variance=(
                self.branch_local_tail_min_position_variance
            ),
            branch_local_tail_max_position_variance=(
                self.branch_local_tail_max_position_variance
            ),
            position_local_tail_quantile=self.position_local_tail_quantile,
            position_local_tail_gain=self.position_local_tail_gain,
            position_local_tail_min_spatial_correlation=(
                self.position_local_tail_min_spatial_correlation
            ),
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
        """Fit position-conditioned normal means, optionally as two modes.

        A small hard mixture captures discrete normal layouts without retaining
        training tokens.  The modes share one PCA loading matrix, so fitting still
        needs one eigendecomposition and inference still needs one projection.
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

        self.mixture_position_means_ = None
        self.mixture_descriptor_centers_ = None
        self.mixture_separation_ = 0.0
        self.mixture_cluster_sizes_ = None
        if self.mixture_components < 2 or len(extracted) < 4:
            return

        descriptors = np.stack([
            self._mixture_descriptor(features, first_grid)
            for features, _, _ in extracted
        ]).astype(np.float64)
        labels, centers, separation = self._fit_two_means(descriptors)
        cluster_sizes = np.bincount(labels, minlength=2)
        required_images = min(
            self.mixture_min_images,
            max(1, len(extracted) // 4),
        )
        required_images = max(
            required_images,
            int(math.ceil(len(extracted) * self.mixture_min_fraction)),
        )
        enabled = (
            separation >= self.mixture_min_separation
            and int(np.min(cluster_sizes)) >= required_images
        )
        self.mixture_separation_ = float(separation)
        self.mixture_cluster_sizes_ = cluster_sizes.astype(np.int32)
        print(
            "Shared-PPCA mixture candidate: "
            f"separation={self.mixture_separation_:.6g}, "
            f"cluster_sizes={cluster_sizes.tolist()}, enabled={enabled}"
        )
        if not enabled:
            return

        mode_means = []
        for mode in range(2):
            mode_features = [
                features
                for (features, _, _), label in zip(extracted, labels)
                if label == mode
            ]
            raw_mode_mean = np.mean(mode_features, axis=0, dtype=np.float64)
            mode_means.append(
                global_mean
                + self.spatial_centering * (raw_mode_mean - global_mean)
            )
        self.mixture_position_means_ = np.stack(mode_means).astype(np.float32)
        self.mixture_descriptor_centers_ = centers.astype(np.float32)

    def _mixture_descriptor(
        self,
        features: np.ndarray,
        grid_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a compact spatial descriptor for normal-mode selection."""
        if grid_size is None:
            grid_size = self.patch_grid_
        if grid_size is None:
            raise RuntimeError("patch grid is unavailable for mixture scoring.")
        h_p, w_p = grid_size
        if features.shape[0] != h_p * w_p:
            raise ValueError(
                "patch feature count mismatch for mixture scoring: "
                f"got {features.shape[0]}, expected {h_p * w_p}"
            )
        feature_map = features.reshape(h_p, w_p, features.shape[-1])
        size = min(self.mixture_descriptor_grid, h_p, w_p)
        if (h_p, w_p) != (size, size):
            # OpenCV builds commonly cap image channels at four for resize;
            # patch embeddings have hundreds. Explicit area bins are tiny
            # (at most 14x14 here) and keep this operation backend-independent.
            row_edges = np.linspace(0, h_p, size + 1, dtype=np.int32)
            col_edges = np.linspace(0, w_p, size + 1, dtype=np.int32)
            pooled = np.empty(
                (size, size, feature_map.shape[-1]),
                dtype=np.float64,
            )
            for row in range(size):
                for col in range(size):
                    pooled[row, col] = np.mean(
                        feature_map[
                            row_edges[row]:row_edges[row + 1],
                            col_edges[col]:col_edges[col + 1],
                        ],
                        axis=(0, 1),
                    )
        else:
            pooled = feature_map
        descriptor = pooled.reshape(-1).astype(np.float64)
        norm = float(np.linalg.norm(descriptor))
        if norm > self.eps:
            descriptor /= norm
        return descriptor

    def _fit_two_means(
        self,
        descriptors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit deterministic balanced-enough 2-means and report variance gain."""
        mean = np.mean(descriptors, axis=0)
        first = int(np.argmax(np.sum((descriptors - mean) ** 2, axis=1)))
        second = int(np.argmax(
            np.sum((descriptors - descriptors[first]) ** 2, axis=1)
        ))
        centers = descriptors[[first, second]].copy()
        labels = np.zeros(descriptors.shape[0], dtype=np.int32)
        for _ in range(20):
            distances = np.stack([
                np.sum((descriptors - center) ** 2, axis=1)
                for center in centers
            ], axis=1)
            new_labels = np.argmin(distances, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels) and np.all(
                np.bincount(new_labels, minlength=2) > 0
            ):
                break
            labels = new_labels
            for mode in range(2):
                selected = descriptors[labels == mode]
                if selected.size:
                    centers[mode] = np.mean(selected, axis=0)

        total = float(np.sum((descriptors - mean) ** 2))
        within = float(np.sum([
            np.sum((descriptors[labels == mode] - centers[mode]) ** 2)
            for mode in range(2)
        ]))
        separation = 0.0 if total <= self.eps else max(0.0, 1.0 - within / total)
        return labels, centers, separation

    def _center_features(self, features: np.ndarray) -> np.ndarray:
        if self.position_mean_ is None:
            return features
        if features.shape != self.position_mean_.shape:
            raise ValueError(
                "patch feature shape mismatch for spatial centering: "
                f"got {features.shape}, expected {self.position_mean_.shape}"
            )
        position_mean = self.position_mean_
        if (
            self.mixture_position_means_ is not None
            and self.mixture_descriptor_centers_ is not None
        ):
            descriptor = self._mixture_descriptor(features)
            distances = np.sum(
                (self.mixture_descriptor_centers_ - descriptor) ** 2,
                axis=1,
            )
            position_mean = self.mixture_position_means_[int(np.argmin(distances))]
        return features - position_mean

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
        print("Calibrating anomaly-map scale")
        fit_max = 0.0
        for scores, grid_size, output_size in scored:
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

        multiband_k = k
        if self.multiband_pca_ev is not None:
            total = float(np.sum(eigvals))
            if total > self.eps:
                multiband_cum = np.cumsum(eigvals) / total
                multiband_k = int(
                    np.searchsorted(multiband_cum, self.multiband_pca_ev) + 1
                )
                # The coarse band must be a subspace of the components already
                # projected for the main score. Clamping also makes explicit
                # pca_dim configurations degrade to the original single band.
                multiband_k = max(1, min(multiband_k, k))

        self.n_components_ = k
        self.multiband_components_ = multiband_k
        self.eigvals_ = eigvals[:k].astype(np.float64)
        self.components_ = eigvecs[:, :k].astype(np.float64)

    def _squared_spe_centered(self, X: np.ndarray) -> np.ndarray:
        """Return squared PCA residuals for position-centered features."""
        scores, _ = self._squared_spe_bands_centered(X)
        return scores

    def _squared_spe_bands_centered(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fine and coarse squared residuals from one PCA projection."""
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
        scores = np.maximum(scores, 0.0)

        multiband_k = self.multiband_components_
        if multiband_k is None or multiband_k >= z.shape[1]:
            return scores, scores

        # The 0.95 residual equals the 0.99 residual plus the energy captured
        # between their component cutoffs. Reuse z so inference needs no second
        # matrix multiplication and no reconstructed feature tensor.
        coarse_scores = scores + np.sum(z[:, multiband_k:] ** 2, axis=1)
        return scores, coarse_scores

    def _fit_score_reference(self, centered_features: np.ndarray) -> None:
        """Fit robust scales for spectral SPE bands and normal-score tail."""
        raw_scores, multiband_scores = self._squared_spe_bands_centered(
            centered_features
        )
        positive_scores = raw_scores[raw_scores > self.eps]
        if positive_scores.size == 0:
            self.score_reference_ = 1.0
        else:
            self.score_reference_ = max(
                float(np.median(positive_scores)),
                self.eps,
            )

        positive_multiband = multiband_scores[multiband_scores > self.eps]
        if positive_multiband.size == 0:
            self.multiband_score_reference_ = 1.0
        else:
            self.multiband_score_reference_ = max(
                float(np.median(positive_multiband)),
                self.eps,
            )

        base_scores = self._transform_spe_bands(raw_scores, multiband_scores)
        if self.tail_score_quantile is None:
            self.tail_score_reference_ = 0.0
        else:
            self.tail_score_reference_ = float(
                np.quantile(base_scores, self.tail_score_quantile)
            )

    def _transform_spe_bands(
        self,
        fine_scores: np.ndarray,
        coarse_scores: np.ndarray,
    ) -> np.ndarray:
        """Transform and combine PCA residual bands before tail calibration."""
        if self.score_transform == "sqrt":
            return np.sqrt(fine_scores)
        if self.score_transform == "squared":
            return fine_scores

        fine_log = np.log1p(fine_scores / self.score_reference_)
        multiband_k = self.multiband_components_
        use_multiband = (
            self.multiband_pca_ev is not None
            and self.multiband_score_weight > 0.0
            and multiband_k is not None
            and self.n_components_ is not None
            and multiband_k < self.n_components_
        )
        if not use_multiband:
            return fine_log

        coarse_log = np.log1p(
            coarse_scores / self.multiband_score_reference_
        )
        weight = self.multiband_score_weight
        return (1.0 - weight) * fine_log + weight * coarse_log

    def _local_score_context(self, scores: np.ndarray) -> np.ndarray:
        """Return the 3x3 Gaussian neighborhood of one patch-score map."""
        if self.patch_grid_ is None:
            raise RuntimeError("patch grid is unavailable for local-tail scoring.")
        h_p, w_p = self.patch_grid_
        if scores.size != h_p * w_p:
            raise ValueError(
                "patch score count mismatch for local-tail scoring: "
                f"got {scores.size}, expected {h_p * w_p}"
            )
        score_map = scores.reshape(h_p, w_p).astype(np.float32)
        return cv2.GaussianBlur(score_map, (3, 3), sigmaX=1.0).ravel()

    def _local_score_residual(self, scores: np.ndarray) -> np.ndarray:
        """Return positive patch-score contrast against a 3x3 neighborhood."""
        return np.maximum(scores - self._local_score_context(scores), 0.0)

    def _fit_branch_local_tail(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ) -> None:
        """Fit a normal-only gate for position or cross-branch local evidence.

        The gate is active only for semi-structured layouts. Homogeneous textures
        have little position-wise score variance, while rigid aligned objects have
        a dominant position template; both cases are better served by the base
        position-PCA score. Spatially coherent layouts use a per-position normal
        tail; the remaining eligible layouts require agreement between branches.
        """
        self.branch_local_tail_thresholds_ = None
        self.branch_local_tail_enabled_ = False
        self.position_local_tail_thresholds_ = None
        self.position_local_tail_enabled_ = False
        self.position_variance_ratio_ = 0.0
        self.spatial_score_correlation_ = 0.0
        branch_available = (
            self.branch_local_tail_quantile is not None
            and self.branch_local_tail_gain > 0.0
        )
        position_available = (
            self.position_local_tail_quantile is not None
            and self.position_local_tail_gain > 0.0
        )
        if len(self.branch_models_) < 2 or not (
            branch_available or position_available
        ):
            return

        scores_by_image = []
        for features, grid_size, _ in extracted:
            if tuple(grid_size) != self.patch_grid_:
                raise ValueError("All fit images must use one patch grid.")
            if features.ndim != 3 or features.shape[0] != len(self.branch_models_):
                raise ValueError("Invalid feature branches for local-tail fitting.")
            scores_by_image.append(np.stack([
                branch._score_features(features[index])
                for index, branch in enumerate(self.branch_models_)
            ]))

        branch_scores = np.stack(scores_by_image).astype(np.float32)
        fused_scores = np.mean(branch_scores, axis=1)
        total_variance = float(np.var(fused_scores))
        if total_variance > self.eps:
            self.position_variance_ratio_ = float(
                np.var(np.mean(fused_scores, axis=0)) / total_variance
            )

        contexts = np.stack([
            self._local_score_context(scores) for scores in fused_scores
        ])
        centered_scores = fused_scores.ravel() - float(np.mean(fused_scores))
        centered_contexts = contexts.ravel() - float(np.mean(contexts))
        correlation_scale = float(np.sqrt(
            np.sum(centered_scores * centered_scores)
            * np.sum(centered_contexts * centered_contexts)
        ))
        if correlation_scale > self.eps:
            self.spatial_score_correlation_ = float(
                np.sum(centered_scores * centered_contexts) / correlation_scale
            )

        eligible = (
            self.branch_local_tail_min_position_variance
            <= self.position_variance_ratio_
            <= self.branch_local_tail_max_position_variance
        )
        if (
            eligible
            and position_available
            and self.spatial_score_correlation_
            >= self.position_local_tail_min_spatial_correlation
        ):
            local_residuals = np.stack([
                self._local_score_residual(scores) for scores in fused_scores
            ])
            self.position_local_tail_thresholds_ = np.quantile(
                local_residuals,
                self.position_local_tail_quantile,
                axis=0,
            ).astype(np.float32)
            self.position_local_tail_enabled_ = True
        elif eligible and branch_available:
            thresholds = []
            for branch_index in range(branch_scores.shape[1]):
                local_residuals = np.concatenate([
                    self._local_score_residual(scores)
                    for scores in branch_scores[:, branch_index]
                ])
                thresholds.append(float(np.quantile(
                    local_residuals,
                    self.branch_local_tail_quantile,
                )))
            self.branch_local_tail_thresholds_ = np.asarray(
                thresholds,
                dtype=np.float32,
            )
            self.branch_local_tail_enabled_ = True

        mode = "position" if self.position_local_tail_enabled_ else (
            "branch" if self.branch_local_tail_enabled_ else "disabled"
        )
        print(
            "Adaptive local-tail fitted: "
            f"position_variance_ratio={self.position_variance_ratio_:.6g}, "
            f"spatial_correlation={self.spatial_score_correlation_:.6g}, "
            f"mode={mode}"
        )

    def _fuse_branch_scores(self, branch_scores: list[np.ndarray]) -> np.ndarray:
        """Fuse independent branch scores and optional agreed local-tail evidence."""
        stacked = np.stack(branch_scores, axis=0)
        scores = np.mean(stacked, axis=0)
        position_thresholds = self.position_local_tail_thresholds_
        if (
            self.position_local_tail_enabled_
            and position_thresholds is not None
            and position_thresholds.size == scores.size
        ):
            scores = scores + self.position_local_tail_gain * np.maximum(
                self._local_score_residual(scores) - position_thresholds,
                0.0,
            )
            return scores.astype(np.float32)

        thresholds = self.branch_local_tail_thresholds_
        if (
            not self.branch_local_tail_enabled_
            or thresholds is None
            or thresholds.size != stacked.shape[0]
        ):
            return scores.astype(np.float32)

        local_evidence = np.stack([
            np.maximum(
                self._local_score_residual(branch_score) - thresholds[index],
                0.0,
            )
            for index, branch_score in enumerate(branch_scores)
        ])
        scores = scores + self.branch_local_tail_gain * np.min(
            local_evidence,
            axis=0,
        )
        return scores.astype(np.float32)

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
            return self._fuse_branch_scores(branch_scores)

        centered = self._center_features(X)
        fine_scores, coarse_scores = self._squared_spe_bands_centered(centered)
        scores = self._transform_spe_bands(fine_scores, coarse_scores)

        if (
            self.score_transform == "log"
            and self.tail_score_quantile is not None
            and self.tail_score_gain > 0.0
        ):
            scores = scores + self.tail_score_gain * np.maximum(
                scores - self.tail_score_reference_,
                0.0,
            )

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
            "mixture_components": self.mixture_components,
            "mixture_descriptor_grid": self.mixture_descriptor_grid,
            "mixture_min_separation": self.mixture_min_separation,
            "mixture_min_fraction": self.mixture_min_fraction,
            "mixture_min_images": self.mixture_min_images,
            "score_transform": self.score_transform,
            "multiband_pca_ev": self.multiband_pca_ev,
            "multiband_score_weight": self.multiband_score_weight,
            "tail_score_quantile": self.tail_score_quantile,
            "tail_score_gain": self.tail_score_gain,
            "branch_local_tail_quantile": self.branch_local_tail_quantile,
            "branch_local_tail_gain": self.branch_local_tail_gain,
            "branch_local_tail_min_position_variance": (
                self.branch_local_tail_min_position_variance
            ),
            "branch_local_tail_max_position_variance": (
                self.branch_local_tail_max_position_variance
            ),
            "position_local_tail_quantile": self.position_local_tail_quantile,
            "position_local_tail_gain": self.position_local_tail_gain,
            "position_local_tail_min_spatial_correlation": (
                self.position_local_tail_min_spatial_correlation
            ),
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
            "mixture_position_means": self.mixture_position_means_,
            "mixture_descriptor_centers": self.mixture_descriptor_centers_,
            "mixture_separation": self.mixture_separation_,
            "mixture_cluster_sizes": self.mixture_cluster_sizes_,
            "patch_grid": self.patch_grid_,
            "score_reference": self.score_reference_,
            "multiband_components": self.multiband_components_,
            "multiband_score_reference": self.multiband_score_reference_,
            "tail_score_reference": self.tail_score_reference_,
            "score_offset": self.score_offset_,
            "fit_max_score": self.fit_max_score_,
            "score_scale": self.score_scale_,
            "threshold": self.threshold_,
            "image_threshold": self.image_threshold_,
            "calibration_count": self.calibration_count_,
            "branch_local_tail_thresholds": self.branch_local_tail_thresholds_,
            "branch_local_tail_enabled": self.branch_local_tail_enabled_,
            "position_local_tail_thresholds": self.position_local_tail_thresholds_,
            "position_local_tail_enabled": self.position_local_tail_enabled_,
            "position_variance_ratio": self.position_variance_ratio_,
            "spatial_score_correlation": self.spatial_score_correlation_,
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
            "format_version": 7,
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
            "mixture_components": int(state["mixture_components"]),
            "mixture_descriptor_grid": int(state["mixture_descriptor_grid"]),
            "mixture_min_separation": float(state["mixture_min_separation"]),
            "mixture_min_fraction": float(state["mixture_min_fraction"]),
            "mixture_min_images": int(state["mixture_min_images"]),
            "mixture_separation": float(state["mixture_separation"]),
            "mixture_cluster_sizes": (
                None
                if state["mixture_cluster_sizes"] is None
                else [int(value) for value in state["mixture_cluster_sizes"]]
            ),
            "score_transform": str(state["score_transform"]),
            "multiband_pca_ev": (
                None
                if state["multiband_pca_ev"] is None
                else float(state["multiband_pca_ev"])
            ),
            "multiband_score_weight": float(state["multiband_score_weight"]),
            "tail_score_quantile": (
                None
                if state["tail_score_quantile"] is None
                else float(state["tail_score_quantile"])
            ),
            "tail_score_gain": float(state["tail_score_gain"]),
            "branch_local_tail_quantile": (
                None
                if state["branch_local_tail_quantile"] is None
                else float(state["branch_local_tail_quantile"])
            ),
            "branch_local_tail_gain": float(state["branch_local_tail_gain"]),
            "branch_local_tail_min_position_variance": float(
                state["branch_local_tail_min_position_variance"]
            ),
            "branch_local_tail_max_position_variance": float(
                state["branch_local_tail_max_position_variance"]
            ),
            "position_local_tail_quantile": (
                None
                if state["position_local_tail_quantile"] is None
                else float(state["position_local_tail_quantile"])
            ),
            "position_local_tail_gain": float(
                state["position_local_tail_gain"]
            ),
            "position_local_tail_min_spatial_correlation": float(
                state["position_local_tail_min_spatial_correlation"]
            ),
            "patch_grid": (
                None
                if state["patch_grid"] is None
                else [int(value) for value in state["patch_grid"]]
            ),
            "branch_local_tail_thresholds": (
                None
                if state["branch_local_tail_thresholds"] is None
                else [
                    float(value)
                    for value in state["branch_local_tail_thresholds"]
                ]
            ),
            "branch_local_tail_enabled": bool(
                state["branch_local_tail_enabled"]
            ),
            "position_local_tail_thresholds": (
                None
                if state["position_local_tail_thresholds"] is None
                else [
                    float(value)
                    for value in state["position_local_tail_thresholds"]
                ]
            ),
            "position_local_tail_enabled": bool(
                state["position_local_tail_enabled"]
            ),
            "position_variance_ratio": float(
                state["position_variance_ratio"]
            ),
            "spatial_score_correlation": float(
                state["spatial_score_correlation"]
            ),
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
                current_metadata = {
                    key: value
                    for key, value in branch.items()
                    if key not in {
                        "mean",
                        "components",
                        "eigvals",
                        "position_mean",
                        "mixture_position_means",
                        "mixture_descriptor_centers",
                        "mixture_cluster_sizes",
                        "branches",
                    }
                }
                current_metadata["mixture_cluster_sizes"] = (
                    None
                    if branch["mixture_cluster_sizes"] is None
                    else [
                        int(value) for value in branch["mixture_cluster_sizes"]
                    ]
                )
                branch_metadata.append(current_metadata)
                arrays[f"branch_{index}_mean"] = branch["mean"]
                arrays[f"branch_{index}_components"] = branch["components"]
                arrays[f"branch_{index}_eigvals"] = branch["eigvals"]
                if branch["position_mean"] is not None:
                    arrays[f"branch_{index}_position_mean"] = branch["position_mean"]
                if branch["mixture_position_means"] is not None:
                    arrays[f"branch_{index}_mixture_position_means"] = branch[
                        "mixture_position_means"
                    ]
                if branch["mixture_descriptor_centers"] is not None:
                    arrays[f"branch_{index}_mixture_descriptor_centers"] = branch[
                        "mixture_descriptor_centers"
                    ]
            metadata["branch_metadata"] = branch_metadata
            arrays["metadata"] = np.asarray(json.dumps(metadata))
        else:
            metadata.update({
                "n_components": int(state["n_components"]),
                "feature_dim": int(state["feature_dim"]),
                "score_reference": float(state["score_reference"]),
                "multiband_components": int(
                    state["multiband_components"] or state["n_components"]
                ),
                "multiband_score_reference": float(
                    state["multiband_score_reference"]
                ),
                "tail_score_reference": float(state["tail_score_reference"]),
            })
            arrays.update({
                "metadata": np.asarray(json.dumps(metadata)),
                "mean": state["mean"],
                "components": state["components"],
                "eigvals": state["eigvals"],
            })
            if state["position_mean"] is not None:
                arrays["position_mean"] = state["position_mean"]
            if state["mixture_position_means"] is not None:
                arrays["mixture_position_means"] = state["mixture_position_means"]
            if state["mixture_descriptor_centers"] is not None:
                arrays["mixture_descriptor_centers"] = state[
                    "mixture_descriptor_centers"
                ]
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
            if format_version not in {1, 2, 3, 4, 5, 6, 7}:
                raise ValueError(
                    "Unsupported SubspaceAD NPZ format version: "
                    f"{metadata.get('format_version')!r}"
                )

            is_multi_branch = format_version == 4 or (
                format_version >= 5 and "branch_metadata" in metadata
            )
            if is_multi_branch:
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
                    mixture_means_key = prefix + "mixture_position_means"
                    if mixture_means_key in saved.files:
                        branch["mixture_position_means"] = np.array(
                            saved[mixture_means_key], copy=True
                        )
                    mixture_centers_key = prefix + "mixture_descriptor_centers"
                    if mixture_centers_key in saved.files:
                        branch["mixture_descriptor_centers"] = np.array(
                            saved[mixture_centers_key], copy=True
                        )
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
                if "mixture_position_means" in saved.files:
                    state["mixture_position_means"] = np.array(
                        saved["mixture_position_means"], copy=True
                    )
                if "mixture_descriptor_centers" in saved.files:
                    state["mixture_descriptor_centers"] = np.array(
                        saved["mixture_descriptor_centers"], copy=True
                    )

            # Versions 1-4 predate spectral-band and tail scoring. Loading one
            # must preserve its historical score rather than use new defaults.
            if format_version < 5:
                state.update({
                    "multiband_pca_ev": None,
                    "multiband_score_weight": 0.0,
                    "tail_score_quantile": None,
                    "tail_score_gain": 0.0,
                })
            # Versions 1-5 predate adaptive branch-local tail scoring. Keep
            # their original score exactly when loading historical files.
            if format_version < 6:
                state.update({
                    "branch_local_tail_quantile": None,
                    "branch_local_tail_gain": 0.0,
                    "branch_local_tail_enabled": False,
                    "branch_local_tail_thresholds": None,
                    "position_local_tail_quantile": None,
                    "position_local_tail_gain": 0.0,
                    "position_local_tail_enabled": False,
                    "position_local_tail_thresholds": None,
                    "position_variance_ratio": 0.0,
                    "spatial_score_correlation": 0.0,
                    "patch_grid": None,
                })
            if format_version < 7:
                state.update({
                    "mixture_components": 1,
                    "mixture_position_means": None,
                    "mixture_descriptor_centers": None,
                    "mixture_separation": 0.0,
                    "mixture_cluster_sizes": None,
                })
                for branch in state.get("branches", []):
                    branch.update({
                        "mixture_components": 1,
                        "mixture_position_means": None,
                        "mixture_descriptor_centers": None,
                        "mixture_separation": 0.0,
                        "mixture_cluster_sizes": None,
                    })

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
        self.mixture_components = int(state.get("mixture_components", 1))
        self.mixture_descriptor_grid = int(
            state.get("mixture_descriptor_grid", 4)
        )
        self.mixture_min_separation = float(
            state.get("mixture_min_separation", 0.08)
        )
        self.mixture_min_fraction = float(
            state.get("mixture_min_fraction", 0.15)
        )
        self.mixture_min_images = int(state.get("mixture_min_images", 8))
        self.score_transform = str(state.get("score_transform", "squared"))
        self.multiband_pca_ev = state.get("multiband_pca_ev")
        if self.multiband_pca_ev is not None:
            self.multiband_pca_ev = float(self.multiband_pca_ev)
        self.multiband_score_weight = float(
            state.get("multiband_score_weight", 0.0)
        )
        self.tail_score_quantile = state.get("tail_score_quantile")
        if self.tail_score_quantile is not None:
            self.tail_score_quantile = float(self.tail_score_quantile)
        self.tail_score_gain = float(state.get("tail_score_gain", 0.0))
        self.branch_local_tail_quantile = state.get(
            "branch_local_tail_quantile"
        )
        if self.branch_local_tail_quantile is not None:
            self.branch_local_tail_quantile = float(
                self.branch_local_tail_quantile
            )
        self.branch_local_tail_gain = float(
            state.get("branch_local_tail_gain", 0.0)
        )
        self.branch_local_tail_min_position_variance = float(
            state.get("branch_local_tail_min_position_variance", 0.1)
        )
        self.branch_local_tail_max_position_variance = float(
            state.get("branch_local_tail_max_position_variance", 0.5)
        )
        self.position_local_tail_quantile = state.get(
            "position_local_tail_quantile"
        )
        if self.position_local_tail_quantile is not None:
            self.position_local_tail_quantile = float(
                self.position_local_tail_quantile
            )
        self.position_local_tail_gain = float(
            state.get("position_local_tail_gain", 0.0)
        )
        self.position_local_tail_min_spatial_correlation = float(
            state.get("position_local_tail_min_spatial_correlation", 0.8)
        )
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
        mixture_position_means = state.get("mixture_position_means")
        self.mixture_position_means_ = (
            None
            if mixture_position_means is None
            else np.asarray(mixture_position_means, dtype=np.float32)
        )
        mixture_descriptor_centers = state.get("mixture_descriptor_centers")
        self.mixture_descriptor_centers_ = (
            None
            if mixture_descriptor_centers is None
            else np.asarray(mixture_descriptor_centers, dtype=np.float32)
        )
        self.mixture_separation_ = float(state.get("mixture_separation", 0.0))
        mixture_cluster_sizes = state.get("mixture_cluster_sizes")
        self.mixture_cluster_sizes_ = (
            None
            if mixture_cluster_sizes is None
            else np.asarray(mixture_cluster_sizes, dtype=np.int32)
        )
        patch_grid = state.get("patch_grid")
        self.patch_grid_ = (
            None
            if patch_grid is None
            else (int(patch_grid[0]), int(patch_grid[1]))
        )
        self.score_reference_ = float(
            state.get(
                "score_reference",
                self.eps if self.score_transform == "log" else 1.0,
            )
        )
        self.multiband_components_ = state.get("multiband_components")
        if self.multiband_components_ is not None:
            self.multiband_components_ = int(self.multiband_components_)
        self.multiband_score_reference_ = float(
            state.get("multiband_score_reference", self.score_reference_)
        )
        self.tail_score_reference_ = float(
            state.get("tail_score_reference", 0.0)
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
        branch_local_thresholds = state.get("branch_local_tail_thresholds")
        self.branch_local_tail_thresholds_ = (
            None
            if branch_local_thresholds is None
            else np.asarray(branch_local_thresholds, dtype=np.float32)
        )
        self.branch_local_tail_enabled_ = bool(
            state.get("branch_local_tail_enabled", False)
        )
        position_local_thresholds = state.get("position_local_tail_thresholds")
        self.position_local_tail_thresholds_ = (
            None
            if position_local_thresholds is None
            else np.asarray(position_local_thresholds, dtype=np.float32)
        )
        self.position_local_tail_enabled_ = bool(
            state.get("position_local_tail_enabled", False)
        )
        self.position_variance_ratio_ = float(
            state.get("position_variance_ratio", 0.0)
        )
        self.spatial_score_correlation_ = float(
            state.get("spatial_score_correlation", 0.0)
        )
