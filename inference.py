"""Run inference on images or a folder using trained checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import AUX_WEIGHT, CHECKPOINT_DIR, IMG_SIZE
from src.data import load_image_tensor
from src.model import EnsembleDetector, XceptionDeepFake, build_model


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    ensemble: bool,
) -> tuple[str, float]:
    x = load_image_tensor(path, IMG_SIZE, device)

    def _fuse_logits(lx, le):
        w = AUX_WEIGHT
        return (1.0 - w) * lx + w * le

    if ensemble:
        assert isinstance(model, EnsembleDetector)
        lx, le = model.xcep(x), model.eff(x)
        logits = _fuse_logits(lx, le)
        xf = torch.flip(x, dims=[3])
        lx2, le2 = model.xcep(xf), model.eff(xf)
        logits = 0.5 * (logits + _fuse_logits(lx2, le2))
    else:
        logits = model(x)
        xf = torch.flip(x, dims=[3])
        logits = 0.5 * (logits + model(xf))

    prob_fake = float(torch.softmax(logits, dim=1)[0, 1].cpu())
    label = "fake" if prob_fake >= 0.5 else "real"
    return label, prob_fake


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best.pt"))
    p.add_argument(
        "inputs",
        nargs="+",
        help="Image files or directories (optional: pass checkpoint.pt first — it is detected by extension)",
    )
    args = p.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    raw = [Path(x) for x in args.inputs]
    if raw and raw[0].suffix.lower() == ".pt":
        ckpt_path = raw[0]
        raw = raw[1:]
    if not raw:
        raise SystemExit("Provide at least one image or directory after the checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    ensemble = ckpt.get("ensemble")
    if ensemble is None:
        ensemble = any(k.startswith("xcep.") for k in sd)
    model = build_model(use_ensemble=ensemble, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)

    paths: list[Path] = []
    for inp in raw:
        pp = Path(inp)
        if pp.is_dir():
            paths.extend(sorted(pp.glob("*.jpg")) + sorted(pp.glob("*.png")))
        else:
            paths.append(pp)

    for path in paths:
        label, prob = predict_one(model, path, device, ensemble)
        print(f"{path.name}: {label} (P(fake)={prob:.4f})")


if __name__ == "__main__":
    main(sys.argv[1:])
