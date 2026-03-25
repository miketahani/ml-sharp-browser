"""Verify ONNX export by comparing outputs against PyTorch model on a test image.

Usage:
    python verify_onnx.py -i path/to/image.jpg
    python verify_onnx.py -i path/to/image.jpg --onnx-path sharp.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INPUT_SIZE = 1536


def run_pytorch(image_path: Path) -> dict[str, np.ndarray]:
    """Run PyTorch model and return outputs as numpy arrays."""
    LOGGER.info("Loading PyTorch model...")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()

    image, _, f_px = io.load_rgb(image_path)
    image_pt = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float()

    image_resized = F.interpolate(
        image_pt[None],
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running PyTorch inference...")
    with torch.no_grad():
        gaussians = predictor(image_resized, disparity_factor)

    return {
        "mean_vectors": gaussians.mean_vectors.numpy(),
        "singular_values": gaussians.singular_values.numpy(),
        "quaternions": gaussians.quaternions.numpy(),
        "colors": gaussians.colors.numpy(),
        "opacities": gaussians.opacities.numpy(),
        # Pass through for ONNX run
        "_image": image_resized.numpy(),
        "_disparity_factor": disparity_factor.numpy(),
    }


def run_onnx(
    onnx_path: Path, image: np.ndarray, disparity_factor: np.ndarray
) -> dict[str, np.ndarray]:
    """Run ONNX model and return outputs."""
    LOGGER.info("Loading ONNX model from %s...", onnx_path)
    session = ort.InferenceSession(str(onnx_path))

    LOGGER.info("Running ONNX inference...")
    outputs = session.run(
        None,
        {"image": image, "disparity_factor": disparity_factor},
    )

    output_names = [o.name for o in session.get_outputs()]
    return dict(zip(output_names, outputs))


def compare_outputs(
    pytorch_outputs: dict[str, np.ndarray], onnx_outputs: dict[str, np.ndarray]
):
    """Compare PyTorch and ONNX outputs."""
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    all_close = True
    for name in ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]:
        pt = pytorch_outputs[name]
        ox = onnx_outputs[name]

        if pt.shape != ox.shape:
            print(f"\n{name}: SHAPE MISMATCH  pt={pt.shape} onnx={ox.shape}")
            all_close = False
            continue

        abs_diff = np.abs(pt - ox)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(pt), 1e-7)
        rel_diff = (abs_diff / denom).mean()

        close = np.allclose(pt, ox, atol=1e-4, rtol=1e-3)
        status = "PASS" if close else "FAIL"
        if not close:
            all_close = False

        print(f"\n{name}: {status}")
        print(f"  shape:     {pt.shape}")
        print(f"  max_diff:  {max_diff:.6e}")
        print(f"  mean_diff: {mean_diff:.6e}")
        print(f"  mean_rel:  {rel_diff:.6e}")
        print(f"  pt range:  [{pt.min():.4f}, {pt.max():.4f}]")
        print(f"  ox range:  [{ox.min():.4f}, {ox.max():.4f}]")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_close else 'SOME FAILED'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX export correctness")
    parser.add_argument(
        "-i", "--image", type=Path, required=True, help="Path to test image"
    )
    parser.add_argument(
        "--onnx-path", type=Path, default=Path("sharp.onnx"), help="ONNX model path"
    )
    args = parser.parse_args()

    pytorch_outputs = run_pytorch(args.image)
    onnx_outputs = run_onnx(
        args.onnx_path,
        pytorch_outputs.pop("_image"),
        pytorch_outputs.pop("_disparity_factor"),
    )
    compare_outputs(pytorch_outputs, onnx_outputs)


if __name__ == "__main__":
    main()
