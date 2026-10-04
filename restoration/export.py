"""Export best checkpoints and gate deployment on measured ONNX consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .prepare import checksum, write_json
from .train import load_checkpoint, model_from_checkpoint

NAMES = ("universal", "classifier", "salt", "blur", "occlusion", "soft", "gan")


class SoftInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output, weights, _ = self.model(image)
        return output, weights


def export_one(checkpoint_path, output_dir, permit_synthetic=False):
    saved = load_checkpoint(checkpoint_path)
    if saved.get("synthetic", False) and not permit_synthetic:
        raise ValueError("Refusing to deploy a synthetic smoke-test checkpoint")
    model = model_from_checkpoint(saved).cpu().eval()
    task = saved["task"]
    if task == "soft":
        model = SoftInference(model).eval()
    torch.manual_seed(42)
    image = torch.rand(1, 3, 128, 128)
    inputs = (image, torch.tensor([0], dtype=torch.int64)) if task == "gan" else (image,)
    input_names = ["image", "style"] if task == "gan" else ["image"]
    output_names = ["restored", "weights"] if task == "soft" else ["logits"] if task == "classifier" else ["output"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}.onnx"
    # Explicit legacy exporter is tested with pinned torch 2.8; no onnxscript dependency.
    torch.onnx.export(model, inputs, str(path), input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for batch in (1, 2):
        for condition in ("random", "black", "white"):
            image = torch.rand(batch, 3, 128, 128) if condition == "random" else torch.full((batch, 3, 128, 128), float(condition == "white"))
            for style in (range(3) if task == "gan" else [None]):
                feed = {"image": image.numpy()}
                inputs = (image,)
                if style is not None:
                    style_tensor = torch.full((batch,), style, dtype=torch.int64)
                    feed["style"] = style_tensor.numpy()
                    inputs = (image, style_tensor)
                with torch.no_grad():
                    expected = model(*inputs)
                expected = expected if isinstance(expected, tuple) else (expected,)
                actual = session.run(None, feed)
                for name, pytorch_value, onnx_value in zip(output_names, expected, actual):
                    ref = pytorch_value.numpy()
                    np.testing.assert_allclose(onnx_value, ref, rtol=1e-3, atol=1e-4)
                    errors.append({"batch": batch, "input": condition, "style": style, "output": name,
                                   "max_absolute_error": float(np.abs(onnx_value - ref).max())})
    return {"file": path.name, "sha256": checksum(path), "verified": True,
            "synthetic": saved.get("synthetic", False), "epoch": saved["epoch"] + 1,
            "config": saved["config"], "validation_objective": saved["best"],
            "manifest_sha256": saved["manifest_sha256"], "checks": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    manifest = {"format_version": 1, "image_size": 128, "models": {}}
    for task in NAMES:
        checkpoint = args.artifacts / "checkpoints" / task / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Train {task} first: {checkpoint}")
        manifest["models"][task] = export_one(checkpoint, args.artifacts / "models")
    write_json(args.artifacts / "models" / "manifest.json", manifest)
    print("Exported and checked all seven inference models")


if __name__ == "__main__":
    main()
