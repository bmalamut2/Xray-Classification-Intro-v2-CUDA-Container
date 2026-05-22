from pathlib import Path
import sys

import pandas as pd
import torch
from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from arkplus_dataset import ArkPlusCSVDataset, build_arkplus_transform, coerce_arkplus_label
from arkplus_model import build_arkplus_model


def test_arkplus_label_policies():
    assert coerce_arkplus_label("", "Zeros", 0) == 0.0
    assert coerce_arkplus_label(float("nan"), "Ones", 0) == 0.0
    assert coerce_arkplus_label("-1", "Ones", 0) == 1.0
    assert coerce_arkplus_label("-1", "Zeros", 0) == 0.0

    lsr_value = coerce_arkplus_label("-1", "LSR-Ones", 0)
    assert 0.55 <= lsr_value <= 0.85


def test_arkplus_csv_dataset_dual_view(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (48, 48), color=(128, 128, 128)).save(image_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "__key__": "sample-1",
                "Path": image_path.name,
                "Finding A": 1,
                "Finding B": -1,
                "Finding C": "",
            }
        ]
    ).to_csv(csv_path, index=False)

    transform = build_arkplus_transform(32, 32, "none", "valid")
    ds = ArkPlusCSVDataset(
        csv_path=str(csv_path),
        image_key="Path",
        label_names=["Finding A", "Finding B", "Finding C"],
        image_root=str(tmp_path),
        student_transform=transform,
        teacher_transform=transform,
        uncertain_label="Zeros",
        unknown_label=0,
        validate_paths=True,
    )

    student_image, teacher_image, labels = ds[0]
    assert student_image.shape == (3, 32, 32)
    assert teacher_image.shape == (3, 32, 32)
    assert torch.equal(labels, torch.tensor([1.0, 0.0, 0.0]))


def test_arkplus_resnet50_multihead_forward():
    model = build_arkplus_model(
        backbone="resnet50",
        num_classes_list=[2, 3, 4],
        pretrained=False,
        projector_features=16,
        use_mlp=True,
    )
    features, logits = model(torch.randn(2, 3, 64, 64), head_index=1)
    assert features.shape == (2, 16)
    assert logits.shape == (2, 3)
    assert len(model.omni_heads) == 3
