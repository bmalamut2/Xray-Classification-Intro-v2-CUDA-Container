from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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


def coerce_csv_label(value, uncertain_label: str = "Ignore", unknown_label: float = -1.0) -> float:
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


class CSVDataset(Dataset):
    """
    Dataset class for CSV-based datasets.

    Args:
        df: pandas DataFrame containing the dataset
        image_key: column name in the DataFrame that contains the image paths
        label_names: list of column names in the DataFrame that contain the label values
        image_root: root directory of the images
        image_append: string to append to the image paths
        transform: torchvision transform to apply to the images
        validate_paths: whether to validate the image paths
        validate_samples: number of samples to validate the image paths
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_key: str,
        label_names: List[str],
        image_root: Optional[str] = None,
        image_append: str = "",
        transform=None,
        uncertain_label: str = "Ignore",
        eval_uncertain_label: Optional[str] = None,
        split: str = "train",
        unknown_label: float = -1.0,
        validate_paths: bool = True,
        validate_samples: int = 5,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_key = image_key
        self.label_names = label_names
        self.image_root = image_root
        self.image_append = image_append
        self.transform = transform
        self.split = split
        self.uncertain_label = (
            eval_uncertain_label
            if split != "train" and eval_uncertain_label is not None
            else uncertain_label
        )
        self.unknown_label = unknown_label

        # Store original paths before modification (for saving predictions)
        # Store both __key__ (if exists) and the original Path value
        if "__key__" in self.df.columns:
            self.original_keys = self.df["__key__"].astype(str).tolist()
        else:
            self.original_keys = None
        # Always store the original Path value (before image_root is applied)
        self.original_paths = self.df[image_key].astype(str).tolist()

        self.df[image_key] = self.df[image_key].apply(
            lambda p: make_path(self.image_root, p, self.image_append)
        )

        if validate_paths and len(self.df) > 0:
            self._validate_paths(validate_samples)

    def _validate_paths(self, n_samples: int = 5) -> None:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, Optional[str]]:
        row = self.df.iloc[idx]
        image_path = row[self.image_key]
        if pd.isna(image_path):
            raise FileNotFoundError("Image path is NaN")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        labels = []
        for lbl in self.label_names:
            v = row.get(lbl, float("nan"))
            labels.append(coerce_csv_label(v, self.uncertain_label, self.unknown_label))
        labels = torch.tensor(labels, dtype=torch.float32)

        # Get original path and key from stored values
        original_path = self.original_paths[idx]
        original_key = self.original_keys[idx] if self.original_keys is not None else None
        
        return image, labels, image_path, original_path, original_key
