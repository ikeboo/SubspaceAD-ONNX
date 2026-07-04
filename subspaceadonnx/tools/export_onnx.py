'''
python subspaceadonnx/tools/export_onnx.py --model-name facebook/dinov3-vitb16-pretrain-lvd1689m --output models/dinov3_vitb_middle7.onnx --height 448 --width 448

'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModel


def parse_layers(s: str | None) -> list[int] | None:
    """
    hidden_states index:
        0 = patch embedding output
        1 = block1 output
        2 = block2 output
        ...
        num_hidden_layers = final block output

    例:
        --layers 4,5,6,7,8,9,10
    """
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def default_middle_layers(num_hidden_layers: int, k: int = 7) -> list[int]:
    """
    SubspaceADのMiddle-7に近い「中央7層」を自動選択する。
    hidden_states[0]はembeddingなので、block出力は1..num_hidden_layers。
    """
    k = min(k, num_hidden_layers)
    center = (num_hidden_layers + 1) // 2
    start = center - k // 2
    start = max(1, start)
    end = start + k - 1

    if end > num_hidden_layers:
        end = num_hidden_layers
        start = end - k + 1

    return list(range(start, end + 1))


class DINOv3MiddleLayerExport(nn.Module):
    """
    DINOv3 ViTの複数中間層patch tokenを取り出して平均するONNX export用ラッパー。

    ONNX出力:
        cls_token:
            [B, C]
        patch_tokens:
            [B, N, C]
            N = H_patch * W_patch

    既存の推論クラス側で
        cls_token, patch_tokens = dinov3(img)
    のように扱える形にする。
    """

    def __init__(
        self,
        model_name: str,
        selected_layers: Sequence[int],
        *,
        l2_normalize: bool = True,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.selected_layers = list(selected_layers)
        if not self.selected_layers:
            raise ValueError("selected_layers must contain at least one layer.")
        self.l2_normalize = bool(l2_normalize)

        self.num_register_tokens = int(
            getattr(self.backbone.config, "num_register_tokens", 0)
        )

        self.num_hidden_layers = int(self.backbone.config.num_hidden_layers)

        for i in self.selected_layers:
            if i < 1 or i > self.num_hidden_layers:
                raise ValueError(
                    f"Invalid layer index: {i}. "
                    f"Use 1..{self.num_hidden_layers}. "
                    "hidden_states[0] is embedding output."
                )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        patch_start = 1 + self.num_register_tokens

        patches = []
        for layer_idx in self.selected_layers:
            x = hidden_states[layer_idx]       # [B, 1 + R + N, C]
            x = x[:, patch_start:, :]          # [B, N, C]
            patches.append(x)

        # [B, L, N, C]
        patches = torch.stack(patches, dim=1)

        # SubspaceAD用: 中間層特徴を平均して1つのpatch token列にする
        patch_tokens = patches.mean(dim=1)     # [B, N, C]

        if self.l2_normalize:
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        # Returning the backbone's final CLS token would keep every block after
        # the last selected feature layer in the exported ONNX graph.  SubspaceAD
        # does not consume CLS, so use the last selected layer and let ONNX dead
        # code elimination prune those otherwise-unused blocks.
        cls_token_layer = max(self.selected_layers)
        cls_token = hidden_states[cls_token_layer][:, 0, :]  # [B, C]

        if self.l2_normalize:
            cls_token = F.normalize(cls_token, dim=-1)

        return cls_token, patch_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/dinov3_vitb_middle7.onnx",
    )
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=448)

    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Comma-separated hidden_states layer indices. "
            "Example: 4,5,6,7,8,9,10. "
            "If omitted, central 7 block outputs are used."
        ),
    )
    parser.add_argument(
        "--middle-k",
        type=int,
        default=7,
        help="Used only when --layers is omitted.",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization of exported features.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export dynamic batch axis.",
    )
    parser.add_argument(
        "--dynamic-hw",
        action="store_true",
        help=(
            "Try exporting dynamic H/W axes. "
            "Fixed H/W is safer for ONNX Runtime / TensorRT."
        ),
    )
    parser.add_argument("--opset", type=int, default=18)

    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    num_hidden_layers = int(config.num_hidden_layers)
    patch_size = int(config.patch_size)
    num_register_tokens = int(getattr(config, "num_register_tokens", 0))
    hidden_size = int(config.hidden_size)

    selected_layers = parse_layers(args.layers)
    if selected_layers is None:
        selected_layers = default_middle_layers(
            num_hidden_layers=num_hidden_layers,
            k=args.middle_k,
        )

    print("model_name:", args.model_name)
    print("num_hidden_layers:", num_hidden_layers)
    print("selected_layers:", selected_layers)
    print("patch_size:", patch_size)
    print("num_register_tokens:", num_register_tokens)
    print("hidden_size:", hidden_size)
    print("input:", (1, 3, args.height, args.width))

    if args.height % patch_size != 0 or args.width % patch_size != 0:
        raise ValueError(
            f"height/width must be multiples of patch_size={patch_size}: "
            f"got {(args.height, args.width)}"
        )

    model = DINOv3MiddleLayerExport(
        model_name=args.model_name,
        selected_layers=selected_layers,
        l2_normalize=not args.no_l2_normalize,
    )
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    dynamic_axes = None

    if args.dynamic_batch or args.dynamic_hw:
        dynamic_axes = {
            "pixel_values": {},
            "cls_token": {},
            "patch_tokens": {},
        }

        if args.dynamic_batch:
            dynamic_axes["pixel_values"][0] = "batch"
            dynamic_axes["cls_token"][0] = "batch"
            dynamic_axes["patch_tokens"][0] = "batch"

        if args.dynamic_hw:
            dynamic_axes["pixel_values"][2] = "height"
            dynamic_axes["pixel_values"][3] = "width"
            dynamic_axes["patch_tokens"][1] = "num_patches"
        else:
            dynamic_axes["patch_tokens"][1] = "num_patches"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["cls_token", "patch_tokens"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    meta = {
        "model_name": args.model_name,
        "height": args.height,
        "width": args.width,
        "patch_size": patch_size,
        "num_register_tokens": num_register_tokens,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "selected_layers": selected_layers,
        "cls_token_layer": max(selected_layers),
        "aggregation": "mean",
        "l2_normalize": not args.no_l2_normalize,
        "image_mean": getattr(processor, "image_mean", None),
        "image_std": getattr(processor, "image_std", None),
    }

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Exported:", output_path)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
