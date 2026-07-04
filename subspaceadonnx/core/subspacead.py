# subspacead_dinov3.py

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Callable, Optional, Iterable, Tuple
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
        normalize_map: bool = False,
        calibration_target: float = 0.5,
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
            normalize_map:
                Trueの場合、共通スケーリング後の異常マップをさらに画像ごとに
                0-1正規化する。画像間で共通の閾値を使う場合はFalseにする。
            calibration_target:
                fit画像全体の異常マップ最大値を合わせる値。
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

        self.pca_ev = pca_ev
        self.pca_dim = pca_dim
        self.max_fit_tokens = max_fit_tokens
        self.feature_l2_normalize = feature_l2_normalize
        self.normalize_map = normalize_map
        self.calibration_target = float(calibration_target)
        self.blur = blur
        self.eps = eps
        self.random_state = random_state

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigvals_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.feature_dim_: Optional[int] = None
        self.fit_max_score_: Optional[float] = None
        self.score_scale_: float = 1.0

    def fit(self, imgs: list[np.ndarray] | str | Path) -> "SubspaceAD":
        """
        正常画像リストまたはディレクトリから正常部分空間を作成する。

        Args:
            imgs:
                正常画像のlist、または正常画像を含むディレクトリ。
                listの各要素は np.ndarray で、DINOv3ラッパーが受け付ける
                形式に合わせてください。ディレクトリの場合は配下を再帰的に
                探索し、対応する画像ファイルをRGBで読み込みます。

        Returns:
            self
        """
        if isinstance(imgs, (str, Path)):
            imgs = self.load_images_from_directory(imgs)

        if len(imgs) == 0:
            raise ValueError("fitには少なくとも1枚の正常画像が必要です。")

        all_features = []
        extracted = []

        for img in tqdm(imgs, desc="Extracting features",unit="images"):
            feat, grid_size = self._extract_patch_features(img)
            all_features.append(feat)
            extracted.append((feat, grid_size, img.shape[:2]))

        X = np.concatenate(all_features, axis=0).astype(np.float64)

        if self.max_fit_tokens is not None and X.shape[0] > self.max_fit_tokens:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.max_fit_tokens, replace=False)
            X = X[idx]
        print(f"Fitting PCA on {X.shape[0]} patch tokens with feature dim {X.shape[1]}.")
        self._fit_pca(X)
        self._fit_score_scale(extracted)
        return self

    @classmethod
    def load_images_from_directory(
        cls,
        directory_path: str | Path,
    ) -> list[np.ndarray]:
        """ディレクトリ配下の対応画像ファイルをRGBで読み込む。

        サブディレクトリも再帰的に探索し、パス順に読み込みます。画像以外の
        ファイルは無視します。

        Args:
            directory_path: 画像を含むディレクトリ。

        Returns:
            RGB画像のlist。
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
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
        out = out * self.score_scale_

        do_norm = self.normalize_map if normalize_map is None else normalize_map
        if do_norm:
            out = self._minmax_norm(out)

        return out.astype(np.float32)

    def _fit_score_scale(
        self,
        extracted: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    ) -> None:
        """fit画像全体の最大異常値から、画像間で共通の係数を求める。"""
        fit_max = 0.0

        for feat, grid_size, output_size in tqdm(
            extracted,
            desc="Calibrating anomaly-map scale",
            unit="images"
        ):
            scores = self._score_features(feat)
            anomaly_map = self._scores_to_map(scores, grid_size, output_size)
            fit_max = max(fit_max, float(np.max(anomaly_map)))

        self.fit_max_score_ = fit_max
        if fit_max <= self.eps:
            self.score_scale_ = 1.0
            warnings.warn(
                "fit画像の最大異常値がほぼ0のため、異常マップの係数を1.0にしました。",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            self.score_scale_ = self.calibration_target / fit_max

        print(
            "Anomaly-map scale calibrated: "
            f"fit_max={self.fit_max_score_:.6g}, scale={self.score_scale_:.6g}, "
            f"scaled_max={self.fit_max_score_ * self.score_scale_:.6g}"
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
        method: str = "mtop1p",
    ) -> float:
        """
        異常マップから画像単位の異常スコアを作る補助関数。

        Args:
            target_img:
                入力画像。
            method:
                画像スコアの集約方法。
                - "max": 最大異常値をそのまま用いる。
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
        out = self.dino(img)

        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                "DINOv3(img) は (cls_token, patch_tokens) を返す必要があります。"
            )

        _, patch_tokens = out
        patch_grid, grid_size = self._patch_tokens_to_grid(patch_tokens, img)

        feat = patch_grid.reshape(-1, patch_grid.shape[-1]).astype(np.float32)

        if self.feature_l2_normalize:
            norm = np.linalg.norm(feat, axis=1, keepdims=True)
            feat = feat / (norm + self.eps)

        return feat, grid_size

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

    def _score_features(self, X: np.ndarray) -> np.ndarray:
        """
        PCA再構成残差をpatch異常スコアにする。
        """
        self._check_fitted()

        X = X.astype(np.float64)

        if X.shape[1] != self.feature_dim_:
            raise ValueError(
                f"feature dim mismatch: got {X.shape[1]}, expected {self.feature_dim_}"
            )

        Xc = X - self.mean_
        z = Xc @ self.components_
        x_recon = (z @ self.components_.T) + self.mean_

        residual = X - x_recon
        scores = np.sum(residual * residual, axis=1)

        return scores.astype(np.float32)

    def _check_fitted(self) -> None:
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
            "normalize_map": self.normalize_map,
            "calibration_target": self.calibration_target,
            "blur": self.blur,
            "eps": self.eps,
            "mean": self.mean_,
            "components": self.components_,
            "eigvals": self.eigvals_,
            "n_components": self.n_components_,
            "feature_dim": self.feature_dim_,
            "fit_max_score": self.fit_max_score_,
            "score_scale": self.score_scale_,
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
        metadata = {
            "format_version": 1,
            "model_name": str(state["model_name"]),
            "pca_ev": (
                None if state["pca_ev"] is None else float(state["pca_ev"])
            ),
            "pca_dim": (
                None if state["pca_dim"] is None else int(state["pca_dim"])
            ),
            "feature_l2_normalize": bool(state["feature_l2_normalize"]),
            "normalize_map": bool(state["normalize_map"]),
            "calibration_target": float(state["calibration_target"]),
            "blur": bool(state["blur"]),
            "eps": float(state["eps"]),
            "n_components": int(state["n_components"]),
            "feature_dim": int(state["feature_dim"]),
            "fit_max_score": float(state["fit_max_score"]),
            "score_scale": float(state["score_scale"]),
        }
        np.savez_compressed(
            npz_path,
            metadata=np.asarray(json.dumps(metadata)),
            mean=state["mean"],
            components=state["components"],
            eigvals=state["eigvals"],
        )

    def load_npz(self, npz_path: str | Path) -> "SubspaceAD":
        """Load PCA state and inference settings saved by :meth:`save_npz`.

        The current DINOv3 instance is retained, so it must be compatible with
        the feature dimension of the model used when fitting.

        Args:
            npz_path: Path to an NPZ file created by :meth:`save_npz`.

        Returns:
            self
        """
        required_keys = {"metadata", "mean", "components", "eigvals"}
        with np.load(npz_path, allow_pickle=False) as saved:
            missing_keys = required_keys.difference(saved.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"Invalid SubspaceAD NPZ: missing keys: {missing}")

            try:
                metadata = json.loads(str(saved["metadata"].item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("Invalid SubspaceAD NPZ metadata.") from exc

            if not isinstance(metadata, dict):
                raise ValueError("Invalid SubspaceAD NPZ metadata.")

            if metadata.get("format_version") != 1:
                raise ValueError(
                    "Unsupported SubspaceAD NPZ format version: "
                    f"{metadata.get('format_version')!r}"
                )

            state = {
                **metadata,
                "mean": np.array(saved["mean"], copy=True),
                "components": np.array(saved["components"], copy=True),
                "eigvals": np.array(saved["eigvals"], copy=True),
            }

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self(
            img,
            output_size=output_size,
            normalize_map=normalize_map,
        )

    def predict_mask(
        self,
        image_path: str,
        threshold: float = 0.5,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_map: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Reads an image from disk, computes the anomaly map, and returns a binary mask.

        Args:
            image_path: Path to the image file.
            threshold: Threshold applied to the anomaly map.
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
        return (anomaly_map >= threshold).astype(np.uint8) * 255

    def load_state_dict(self, state: dict) -> None:
        """
        DINOv3インスタンス作成後にPCA状態を復元する。
        """
        self.pca_ev = state["pca_ev"]
        self.pca_dim = state["pca_dim"]
        self.feature_l2_normalize = state["feature_l2_normalize"]
        self.normalize_map = state["normalize_map"]
        self.calibration_target = float(
            state["calibration_target"] if "calibration_target" in state else 0.5
        )
        self.blur = state["blur"]
        self.eps = state["eps"]

        self.mean_ = state["mean"]
        self.components_ = state["components"]
        self.eigvals_ = state["eigvals"]
        self.n_components_ = state["n_components"]
        self.feature_dim_ = state["feature_dim"]
        self.fit_max_score_ = float(
            state["fit_max_score"] if "fit_max_score" in state else 0.0
        )
        self.score_scale_ = float(
            state["score_scale"] if "score_scale" in state else 1.0
        )
