from pathlib import Path
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_path(root: Optional[str], p: str, append: str = "") -> str:
    if pd.isna(p):
        return p
    s = str(p) + append
    path = Path(s)
    if path.is_absolute():
        return str(path)
    if root is None:
        return str(path)
    return str(Path(root) / path)


def coerce_arkplus_label(value, uncertain_label: str, unknown_label: float) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return float(unknown_label)

    value = float(value)
    if value != -1:
        return value

    policy = uncertain_label.lower()
    if policy in ("ignore", "keep"):
        return -1.0
    if policy == "ones":
        return 1.0
    if policy == "zeros":
        return 0.0
    if policy == "lsr-ones":
        return random.uniform(0.55, 0.85)
    if policy == "lsr-zeros":
        return random.uniform(0.0, 0.3)
    raise ValueError(f"Unsupported uncertain_label policy: {uncertain_label}")


def _normalizer(normalize: str):
    normalize = normalize.lower()
    if normalize == "imagenet":
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    if normalize in ("chestx-ray", "chestxray"):
        return transforms.Normalize(
            mean=[0.5056, 0.5056, 0.5056],
            std=[0.252, 0.252, 0.252],
        )
    if normalize in ("none", "false", "null"):
        return None
    raise ValueError(f"Unsupported normalization: {normalize}")


def build_arkplus_transform(
    image_size: int,
    resize: int,
    normalize: str,
    mode: str,
):
    normalizer = _normalizer(normalize)
    t = []

    if mode == "student":
        t.extend(
            [
                transforms.Resize((resize, resize)),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
                transforms.ToTensor(),
            ]
        )
    elif mode == "teacher":
        # Ark+ uses the resized original image for the teacher signal.
        t.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
    elif mode in ("valid", "test"):
        t.append(transforms.Resize((resize, resize)))
        if resize != image_size:
            t.append(transforms.CenterCrop(image_size))
        t.append(transforms.ToTensor())
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")

    if normalizer is not None:
        t.append(normalizer)
    return transforms.Compose(t)


class ArkPlusCSVDataset(Dataset):
    """Dual-view CSV dataset for Ark+ teacher/student pretraining."""

    def __init__(
        self,
        csv_path: str,
        image_key: str,
        label_names: List[str],
        image_root: Optional[str] = None,
        image_append: str = "",
        student_transform=None,
        teacher_transform=None,
        uncertain_label: str = "Zeros",
        eval_uncertain_label: Optional[str] = None,
        split: str = "train",
        unknown_label: float = 0.0,
        validate_paths: bool = True,
        validate_samples: int = 5,
    ) -> None:
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_key = image_key
        self.label_names = list(label_names)
        self.image_root = image_root
        self.image_append = image_append
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.split = split
        self.uncertain_label = (
            eval_uncertain_label
            if split != "train" and eval_uncertain_label is not None
            else uncertain_label
        )
        self.unknown_label = unknown_label

        if "__key__" in self.df.columns:
            self.original_keys = self.df["__key__"].astype(str).tolist()
        else:
            self.original_keys = None
        self.original_paths = self.df[image_key].astype(str).tolist()

        self.df[image_key] = self.df[image_key].apply(
            lambda p: make_path(self.image_root, p, self.image_append)
        )
        self.labels = self._build_labels()

        if validate_paths and len(self.df) > 0:
            self._validate_paths(validate_samples)

    def _build_labels(self) -> torch.Tensor:
        labels = np.zeros((len(self.df), len(self.label_names)), dtype=np.float32)
        for label_idx, label_name in enumerate(self.label_names):
            if label_name in self.df.columns:
                values = self.df[label_name]
            else:
                values = pd.Series(np.nan, index=self.df.index)
            labels[:, label_idx] = values.map(
                lambda v: coerce_arkplus_label(v, self.uncertain_label, self.unknown_label)
            ).astype(np.float32)
        return torch.from_numpy(labels)

    def _validate_paths(self, n_samples: int) -> None:
        n_samples = min(n_samples, len(self.df))
        indices = np.random.choice(len(self.df), n_samples, replace=False)
        missing = []
        for idx in indices:
            path = self.df.iloc[idx][self.image_key]
            if pd.isna(path) or not Path(path).exists():
                missing.append(path)
        if missing:
            raise FileNotFoundError(f"Missing image paths. Example: {missing[0]}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image_path = self.df.iloc[idx][self.image_key]
        if pd.isna(image_path):
            raise FileNotFoundError("Image path is NaN")

        image = Image.open(image_path).convert("RGB")
        student_image = self.student_transform(image) if self.student_transform else image
        teacher_image = self.teacher_transform(image) if self.teacher_transform else student_image
        return student_image, teacher_image, self.labels[idx]
