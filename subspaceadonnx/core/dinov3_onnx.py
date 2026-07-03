# dinov3_middle_onnx.py

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class DINOv3:
    """
    DINOv3 middle-layer ONNX wrapper.

    Returns:
        cls_token:
            [C]
        patch_tokens:
            [H_patch, W_patch, C]
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        providers: list[str] | None = None,
    ):
        """Create an ONNX feature extractor.

        Args:
            onnx_path: Path to the ONNX model.
            providers: ONNX Runtime Execution Providers in priority order.
                Unsupported providers are ignored. If none remain, CPU is used.
        """
        self.onnx_path = str(onnx_path)

        meta_path = Path(onnx_path).with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata json not found: {meta_path}")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.height = int(self.meta["height"])
        self.width = int(self.meta["width"])
        self.patch_size = int(self.meta["patch_size"])

        self.grid_h = self.height // self.patch_size
        self.grid_w = self.width // self.patch_size

        mean = self.meta.get("image_mean", None)
        std = self.meta.get("image_std", None)

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        if providers is None:
            # WebGPU can be listed by ONNX Runtime even when the host has no
            # usable adapter; initializing it then terminates the process in
            # some environments. Keep it available through an explicit
            # ``providers`` argument, but use safe defaults here.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.providers = list(providers)

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=self.providers,
        )

        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img: np.ndarray):
        x = self.preprocess(img)

        cls_token, patch_tokens = self.session.run(
            None,
            {self.input_name: x},
        )

        cls_token = cls_token[0]          # [C]
        patch_tokens = patch_tokens[0]    # [N, C]

        n, c = patch_tokens.shape
        expected_n = self.grid_h * self.grid_w

        if n != expected_n:
            # dynamic H/W export時などの保険
            s = int(np.sqrt(n))
            if s * s == n:
                grid_h, grid_w = s, s
            else:
                raise ValueError(
                    f"Cannot infer patch grid: N={n}, "
                    f"default grid={self.grid_h}x{self.grid_w}"
                )
        else:
            grid_h, grid_w = self.grid_h, self.grid_w

        patch_tokens = patch_tokens.reshape(grid_h, grid_w, c)

        return cls_token.astype(np.float32), patch_tokens.astype(np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img:
                RGB or BGRのHWC画像。
                既存pipelineに合わせて必要ならRGB/BGRを調整してください。

        Returns:
            pixel_values:
                [1, 3, H, W], float32
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"img must be HWC 3ch image, got {img.shape}")

        # ここでは入力をRGB想定にしています。
        # OpenCVで読んだBGR画像を渡す場合は、呼び出し前に
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # してください。
        x = cv2.resize(
            img,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        x = x.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = x[None, ...]
        return x.astype(np.float32)
