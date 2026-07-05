import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from subspaceadonnx.tools.export_onnx import (
    DINOv3MiddleLayerExport,
    default_middle_layers,
    parse_layer_groups,
    parse_layers,
)


class _FakeBackbone(nn.Module):
    def forward(self, pixel_values, output_hidden_states, return_dict):
        # [CLS, patch 1, patch 2], with deliberately distinct CLS values.
        hidden_states = (
            torch.zeros((1, 3, 2)),
            torch.tensor([[[10.0, 10.0], [1.0, 1.0], [3.0, 3.0]]]),
            torch.tensor([[[20.0, 20.0], [5.0, 5.0], [7.0, 7.0]]]),
            torch.tensor([[[99.0, 99.0], [9.0, 9.0], [9.0, 9.0]]]),
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            last_hidden_state=hidden_states[-1],
        )


class ExportOnnxTests(unittest.TestCase):
    def test_layer_parsing_and_middle_selection(self) -> None:
        self.assertEqual(parse_layers("3, 5,7"), [3, 5, 7])
        self.assertEqual(parse_layer_groups("2,3,4,5;6,7,8,9"), [[2, 3, 4, 5], [6, 7, 8, 9]])
        self.assertEqual(default_middle_layers(12, 7), [3, 4, 5, 6, 7, 8, 9])

    def test_cls_comes_from_last_selected_layer_for_graph_pruning(self) -> None:
        model = DINOv3MiddleLayerExport.__new__(DINOv3MiddleLayerExport)
        nn.Module.__init__(model)
        model.backbone = _FakeBackbone()
        model.selected_layers = [1, 2]
        model.num_register_tokens = 0
        model.l2_normalize = False

        cls_token, patch_tokens = model(torch.zeros((1, 3, 2, 2)))

        torch.testing.assert_close(cls_token, torch.tensor([[20.0, 20.0]]))
        torch.testing.assert_close(
            patch_tokens,
            torch.tensor([[[3.0, 3.0], [5.0, 5.0]]]),
        )

    def test_layer_groups_are_returned_separately(self) -> None:
        model = DINOv3MiddleLayerExport.__new__(DINOv3MiddleLayerExport)
        nn.Module.__init__(model)
        model.backbone = _FakeBackbone()
        model.layer_groups = [[1], [2]]
        model.selected_layers = [1, 2]
        model.num_register_tokens = 0
        model.l2_normalize = False

        cls_token, early, late = model(torch.zeros((1, 3, 2, 2)))

        torch.testing.assert_close(cls_token, torch.tensor([[20.0, 20.0]]))
        torch.testing.assert_close(early, torch.tensor([[[1.0, 1.0], [3.0, 3.0]]]))
        torch.testing.assert_close(late, torch.tensor([[[5.0, 5.0], [7.0, 7.0]]]))


if __name__ == "__main__":
    unittest.main()
