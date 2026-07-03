import unittest
from unittest.mock import patch

from subspaceadonnx import SubspaceAD


class ProviderForwardingTests(unittest.TestCase):
    def test_subspacead_forwards_providers_to_dinov3(self) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with patch("subspaceadonnx.core.subspacead.DINOv3") as dino_class:
            model = SubspaceAD("model.onnx", providers=providers)

        dino_class.assert_called_once_with("model.onnx", providers=providers)
        self.assertEqual(model.providers, providers)

    def test_providers_cannot_be_combined_with_custom_dino(self) -> None:
        with self.assertRaisesRegex(ValueError, "providers cannot be specified"):
            SubspaceAD(
                "model.onnx",
                dino=object(),
                providers=["CPUExecutionProvider"],
            )


if __name__ == "__main__":
    unittest.main()
