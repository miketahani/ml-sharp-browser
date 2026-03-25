"""Export ML-SHARP prediction model to ONNX format for browser inference.

Usage:
    python export_onnx.py                          # downloads default checkpoint
    python export_onnx.py -c path/to/checkpoint.pt # uses local checkpoint
    python export_onnx.py --fp16                   # export with fp16 weights
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from sharp.models import PredictorParams, create_predictor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INPUT_SIZE = 1536


class SharpONNXWrapper(nn.Module):
    """Wraps RGBGaussianPredictor to return flat tensors instead of NamedTuple.

    ONNX export doesn't support NamedTuple returns, so we unpack Gaussians3D
    into 5 separate output tensors.
    """

    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(
        self, image: torch.Tensor, disparity_factor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gaussians = self.predictor(image, disparity_factor)
        return (
            gaussians.mean_vectors,
            gaussians.singular_values,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacities,
        )


def load_predictor(checkpoint_path: Path | None, device: str = "cpu") -> nn.Module:
    if checkpoint_path is None:
        LOGGER.info("Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True
        )
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def export_onnx(
    checkpoint_path: Path | None,
    output_path: Path,
    fp16: bool = False,
    opset_version: int = 17,
):
    device = "cpu"  # Export on CPU for maximum compatibility
    predictor = load_predictor(checkpoint_path, device)
    wrapper = SharpONNXWrapper(predictor)
    wrapper.eval()

    # Dummy inputs matching predict_image() in cli/predict.py
    dummy_image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    dummy_disparity_factor = torch.tensor([1.0], device=device)

    LOGGER.info("Running torch.onnx.export (opset %d)...", opset_version)
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_disparity_factor),
        str(output_path),
        opset_version=opset_version,
        input_names=["image", "disparity_factor"],
        output_names=[
            "mean_vectors",
            "singular_values",
            "quaternions",
            "colors",
            "opacities",
        ],
        dynamic_axes=None,  # Fixed 1536x1536 input, no dynamic shapes
    )
    LOGGER.info("Exported to %s", output_path)

    if fp16:
        import onnx
        from onnxconverter_common import float16

        LOGGER.info("Converting to fp16...")
        model = onnx.load(str(output_path))
        model_fp16 = float16.convert_float_model_to_float16_model(model)
        fp16_path = output_path.with_suffix(".fp16.onnx")
        onnx.save(model_fp16, str(fp16_path))
        LOGGER.info("Saved fp16 model to %s", fp16_path)


def main():
    parser = argparse.ArgumentParser(description="Export SHARP to ONNX")
    parser.add_argument(
        "-c", "--checkpoint-path", type=Path, default=None,
        help="Path to .pt checkpoint (downloads default if omitted)",
    )
    parser.add_argument(
        "-o", "--output-path", type=Path, default=Path("sharp.onnx"),
        help="Output ONNX file path (default: sharp.onnx)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Also export fp16 quantized version",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()
    export_onnx(args.checkpoint_path, args.output_path, args.fp16, args.opset)


if __name__ == "__main__":
    main()
