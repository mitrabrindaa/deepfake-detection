"""Dataset splits, heavy augmentation (train), deterministic eval transforms."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def collect_samples(root: Path) -> Tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    real_dir = root / "real"
    fake_dir = root / "fake"
    for p in sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.jpeg")) + sorted(real_dir.glob("*.png")):
        paths.append(p)
        labels.append(config.CLASS_REAL)
    for p in sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.jpeg")) + sorted(fake_dir.glob("*.png")):
        paths.append(p)
        labels.append(config.CLASS_FAKE)
    return paths, labels


def stratified_split(
    paths: list[Path],
    labels: list[int],
    seed: int,
) -> Tuple[list[Path], list[Path], list[Path], list[int], list[int], list[int]]:
    set_seed(seed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=(1.0 - config.TRAIN_FRAC),
        stratify=labels,
        random_state=seed,
    )
    val_ratio = config.VAL_FRAC / (1.0 - config.TRAIN_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio),
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_train_augmentation(img_size: int) -> A.Compose:
    """Strong augmentation: geometry, photometric, noise, compression, cutout."""
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.82, 1.0), ratio=(0.92, 1.08), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.04, 0.04),
                scale=(0.94, 1.06),
                rotate=(-12, 12),
                border_mode=4,  # reflect
                p=0.65,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=1.0),
                ],
                p=0.55,
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=22, val_shift_limit=12, p=0.45),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.35), p=1.0),
                ],
                p=0.35,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ImageCompression(quality_range=(55, 100), p=0.35),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.06, 0.12),
                hole_width_range=(0.06, 0.12),
                fill="random_uniform",
                p=0.25,
            ),
            A.Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(),
        ]
    )


def get_eval_augmentation(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(),
        ]
    )


class FaceDeepFakeDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        transform: Optional[A.Compose],
        is_train: bool,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = self.labels[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            t = self.transform(image=img)
            img = t["image"]
        return img, y


def mixup_data(x, y, alpha: float, device):
    """Returns mixed inputs, pairs of labels, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_image_tensor(path: Path, img_size: int, device):
    """Load single image for inference (eval transform)."""
    tfm = get_eval_augmentation(img_size)
    img = np.array(Image.open(path).convert("RGB"))
    t = tfm(image=img)["image"].unsqueeze(0).to(device)
    return t
