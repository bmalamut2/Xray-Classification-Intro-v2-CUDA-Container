import argparse
import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from arkplus_dataset import ArkPlusCSVDataset, build_arkplus_transform
from arkplus_model import build_arkplus_model
from utils import ensure_dir, get_best_accelerator


TOP5_AUC_DATASETS = {"mimic", "chexpert"}
TOP5_AUC_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def get_absolute_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def apply_dataset_dir_override(cfg: DictConfig, dataset_dir: Optional[str]) -> None:
    if not dataset_dir:
        return

    cfg.dataset_dir = dataset_dir
    for task in cfg.ark.datasets:
        name = str(task.name).lower()
        task.dataset_root = dataset_dir
        if name == "mimic":
            root = f"{dataset_dir}/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0"
        elif name == "chexpert":
            root = f"{dataset_dir}/CheXpert/"
        elif name == "chestxray14":
            root = f"{dataset_dir}/nih_xray14/images/images/"
        else:
            continue
        task.train_image_root = root
        task.val_image_root = root
        task.test_image_root = root


def build_eval_dataset(
    cfg: DictConfig,
    task_cfg: DictConfig,
    split: str,
    base_dir: str,
    validate_paths: bool,
) -> ArkPlusCSVDataset:
    image_size = int(cfg.ark.preprocessing.get("image_size", cfg.model.get("image_size", 224)))
    resize = int(cfg.ark.preprocessing.get("resize", max(256, image_size)))
    normalize = str(cfg.ark.preprocessing.get("normalize", cfg.model.get("normalize", "imagenet")))
    transform = build_arkplus_transform(
        image_size=image_size,
        resize=resize,
        normalize=normalize,
        mode="valid" if split == "val" else "test",
    )

    # Match Ark+ pretraining at 228897cc: use the Ark+ dataset label policy
    # for validation/test too, without the newer eval_uncertain_label override.
    return ArkPlusCSVDataset(
        csv_path=get_absolute_path(task_cfg[f"{split}_ann"], base_dir),
        image_key=task_cfg.get("image_path_key", "Path"),
        label_names=list(task_cfg.labels),
        image_root=get_absolute_path(task_cfg.get(f"{split}_image_root"), base_dir),
        image_append=task_cfg.get("image_path_append", ""),
        student_transform=transform,
        teacher_transform=transform,
        uncertain_label=task_cfg.get("uncertain_label", "Zeros"),
        eval_uncertain_label=None,
        split=split,
        unknown_label=float(task_cfg.get("unknown_label", 0.0)),
        validate_paths=validate_paths,
    )


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "multi-class classification":
        return nn.CrossEntropyLoss()(logits, targets.argmax(dim=1).long())
    return nn.BCEWithLogitsLoss()(logits, targets)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def get_top5_auc(dataset_name: str, label_names: List[str], per_label_auc: object) -> Optional[float]:
    if dataset_name.lower() not in TOP5_AUC_DATASETS or not isinstance(per_label_auc, list):
        return None

    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    missing_labels = [label_name for label_name in TOP5_AUC_LABELS if label_name not in label_to_index]
    if missing_labels:
        raise ValueError(f"{dataset_name} is missing top-5 AUC labels: {missing_labels}")

    top5_values = np.array(
        [per_label_auc[label_to_index[label_name]] for label_name in TOP5_AUC_LABELS],
        dtype=np.float64,
    )
    return float(np.nanmean(top5_values)) if np.any(~np.isnan(top5_values)) else float("nan")


def format_float_metric(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.6f}"
    return value


def compute_multilabel_roc(
    probs: np.ndarray,
    targets: np.ndarray,
    label_names: List[str],
) -> Dict[str, object]:
    per_label_auc = []
    class_auc_rows = []
    curves = {}
    mean_fpr = np.linspace(0.0, 1.0, 101)
    interpolated_tprs = []

    for class_index, label_name in enumerate(label_names):
        raw_targets = targets[:, class_index]
        valid_mask = np.isfinite(raw_targets) & (raw_targets != -1)
        y_true = (raw_targets[valid_mask] > 0.5).astype(np.int64)
        y_score = probs[valid_mask, class_index]
        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)

        if positives == 0 or negatives == 0:
            class_auc = float("nan")
            fpr = np.array([], dtype=np.float32)
            tpr = np.array([], dtype=np.float32)
            thresholds = np.array([], dtype=np.float32)
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            class_auc = float(auc(fpr, tpr))
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interpolated_tprs.append(interp_tpr)

        per_label_auc.append(class_auc)
        class_auc_rows.append(
            {
                "class_index": class_index,
                "class_name": label_name,
                "positives": positives,
                "negatives": negatives,
                "auc": class_auc,
            }
        )
        curves[label_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": class_auc,
        }

    if interpolated_tprs:
        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_tpr[-1] = 1.0
        macro_curve_auc = float(auc(mean_fpr, mean_tpr))
    else:
        mean_tpr = np.array([], dtype=np.float32)
        macro_curve_auc = float("nan")

    mean_class_auc = float(np.nanmean(per_label_auc)) if np.any(~np.isnan(per_label_auc)) else float("nan")
    curves["macro_mean"] = {
        "fpr": mean_fpr if interpolated_tprs else np.array([], dtype=np.float32),
        "tpr": mean_tpr,
        "thresholds": np.array([], dtype=np.float32),
        "auc": macro_curve_auc,
    }

    return {
        "mean_class_auc": mean_class_auc,
        "macro_curve_auc": macro_curve_auc,
        "per_label_auc": per_label_auc,
        "class_auc_rows": class_auc_rows,
        "curves": curves,
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_index: int,
    task_type: str,
    num_labels: int,
    label_names: List[str],
) -> Dict[str, object]:
    model.eval()
    losses = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for _, images, targets in loader:
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            _, logits = model(images, head_index)
            loss = classification_loss(logits, targets, task_type)
            losses.append(float(loss.item()) * images.size(0))

            if task_type == "multi-class classification":
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_labels))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_labels))
    sample_count = int(targets_np.shape[0])
    mean_loss = float(np.sum(losses) / max(1, sample_count))

    if task_type == "multi-class classification":
        accuracy = float((probs_np.argmax(axis=1) == targets_np.argmax(axis=1)).mean())
        return {
            "loss": mean_loss,
            "accuracy": accuracy,
            "mean_auroc": "",
            "per_label_auroc": "",
            "samples": sample_count,
        }

    roc_stats = compute_multilabel_roc(probs_np, targets_np, label_names)
    return {
        "loss": mean_loss,
        "accuracy": "",
        "mean_auroc": roc_stats["mean_class_auc"],
        "macro_curve_auc": roc_stats["macro_curve_auc"],
        "per_label_auroc": roc_stats["per_label_auc"],
        "class_auc_rows": roc_stats["class_auc_rows"],
        "roc_curves": roc_stats["curves"],
        "samples": sample_count,
    }


def write_metrics_csv(output_path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(output_path) or ".")
    fieldnames = [
        "checkpoint",
        "epoch",
        "global_step",
        "dataset",
        "split",
        "samples",
        "loss",
        "mean_auroc",
        "mean_auroc_top5",
        "accuracy",
        "per_label_auroc",
        "date_time",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_per_class_auc_csv(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(output_path) or ".")
    fieldnames = [
        "checkpoint",
        "epoch",
        "global_step",
        "dataset",
        "split",
        "class_index",
        "class_name",
        "positives",
        "negatives",
        "auc",
        "macro_curve_auc",
        "date_time",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_roc_curve_files(
    output_dir: str,
    dataset_name: str,
    split: str,
    curves: Dict[str, Dict[str, object]],
) -> Dict[str, str]:
    ensure_dir(output_dir)
    base = f"{safe_filename(dataset_name)}_{split}"
    points_path = os.path.join(output_dir, f"{base}_roc_points.csv")
    plot_path = os.path.join(output_dir, f"{base}_roc_curves.png")

    with open(points_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "split", "class_name", "point_index", "fpr", "tpr", "threshold", "auc"],
        )
        writer.writeheader()
        for class_name, curve in curves.items():
            fpr = curve["fpr"]
            tpr = curve["tpr"]
            thresholds = curve["thresholds"]
            for point_index in range(len(fpr)):
                threshold = "" if class_name == "macro_mean" else thresholds[point_index]
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "class_name": class_name,
                        "point_index": point_index,
                        "fpr": f"{float(fpr[point_index]):.8f}",
                        "tpr": f"{float(tpr[point_index]):.8f}",
                        "threshold": "" if threshold == "" else f"{float(threshold):.8f}",
                        "auc": (
                            "nan"
                            if np.isnan(curve["auc"])
                            else f"{float(curve['auc']):.8f}"
                        ),
                    }
                )

    plt.figure(figsize=(10, 8))
    for class_name, curve in curves.items():
        if class_name == "macro_mean" or len(curve["fpr"]) == 0:
            continue
        auc_label = "nan" if np.isnan(curve["auc"]) else f"{curve['auc']:.3f}"
        plt.plot(curve["fpr"], curve["tpr"], linewidth=1.0, alpha=0.55, label=f"{class_name} ({auc_label})")

    macro = curves.get("macro_mean")
    if macro is not None and len(macro["fpr"]) > 0:
        auc_label = "nan" if np.isnan(macro["auc"]) else f"{macro['auc']:.3f}"
        plt.plot(
            macro["fpr"],
            macro["tpr"],
            color="black",
            linewidth=2.5,
            label=f"macro mean ({auc_label})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name} {split} ROC curves")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return {"points": points_path, "plot": plot_path}


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Ark+ teacher checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_teacher.pth.tar or last_teacher.pth.tar")
    parser.add_argument("--output", default=None, help="Metrics CSV path")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dataset-dir", default=None, help="Override dataset root directory")
    parser.add_argument("--device", default=None, help="cuda, cpu, or a torch device string")
    parser.add_argument("--no-validate-paths", action="store_true")
    parser.add_argument("--per-class-output", default=None, help="Per-class test AUC CSV path")
    parser.add_argument("--roc-output-dir", default=None, help="Directory for test ROC curve PNG/CSV files")
    parser.add_argument("--no-roc-curves", action="store_true", help="Disable test ROC curve files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or get_best_accelerator())
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = OmegaConf.create(checkpoint["config"])
    apply_dataset_dir_override(cfg, args.dataset_dir)

    task_metadata = checkpoint.get("task_metadata")
    if task_metadata:
        num_classes_list = [len(task["labels"]) for task in task_metadata]
    else:
        num_classes_list = [len(task.labels) for task in cfg.ark.datasets]

    model = build_arkplus_model(
        backbone=str(cfg.model.backbone),
        num_classes_list=num_classes_list,
        pretrained=False,
        projector_features=cfg.ark.get("projector_features", 1376),
        use_mlp=bool(cfg.ark.get("use_mlp", False)),
    )
    model.load_state_dict(checkpoint["teacher"], strict=True)
    model.to(device)
    model.eval()

    batch_size = int(args.batch_size or cfg.ark.get("eval_batch_size", 100))
    workers = int(args.workers if args.workers is not None else cfg.ark.get("workers", 8))
    validate_paths = not args.no_validate_paths
    base_dir = os.getcwd()

    rows = []
    per_class_rows = []
    for head_index, task_cfg in enumerate(cfg.ark.datasets):
        task_name = str(task_cfg.name)
        task_type = str(task_cfg.get("task_type", "multi-label classification"))
        label_names = list(task_cfg.labels)
        for split in args.splits:
            dataset = build_eval_dataset(cfg, task_cfg, split, base_dir, validate_paths)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=device.type == "cuda",
            )
            metrics = evaluate_split(
                model=model,
                loader=loader,
                device=device,
                head_index=head_index,
                task_type=task_type,
                num_labels=len(task_cfg.labels),
                label_names=label_names,
            )
            metrics["mean_auroc_top5"] = get_top5_auc(
                task_name, label_names, metrics.get("per_label_auroc")
            )
            per_label = metrics["per_label_auroc"]
            if isinstance(per_label, list):
                per_label = ";".join("nan" if np.isnan(x) else f"{x:.6f}" for x in per_label)
            rows.append(
                {
                    "checkpoint": args.checkpoint,
                    "epoch": checkpoint.get("epoch", ""),
                    "global_step": checkpoint.get("global_step", ""),
                    "dataset": task_name,
                    "split": split,
                    "samples": metrics["samples"],
                    "loss": f"{metrics['loss']:.6f}",
                    "mean_auroc": format_float_metric(metrics["mean_auroc"]),
                    "mean_auroc_top5": format_float_metric(metrics["mean_auroc_top5"]),
                    "accuracy": (
                        f"{metrics['accuracy']:.6f}"
                        if isinstance(metrics["accuracy"], float)
                        else metrics["accuracy"]
                    ),
                    "per_label_auroc": per_label,
                    "date_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )
            top5_text = (
                f" mean_auroc_top5={format_float_metric(metrics['mean_auroc_top5'])}"
                if isinstance(metrics["mean_auroc_top5"], float)
                else ""
            )
            print(
                f"{task_name} {split}: "
                f"loss={metrics['loss']:.6f} "
                f"mean_auroc={metrics['mean_auroc']} "
                f"{top5_text} "
                f"accuracy={metrics['accuracy']} "
                f"samples={metrics['samples']}"
            )

            if split == "test" and task_type != "multi-class classification":
                now = datetime.now().astimezone().isoformat(timespec="seconds")
                for class_row in metrics["class_auc_rows"]:
                    per_class_rows.append(
                        {
                            "checkpoint": args.checkpoint,
                            "epoch": checkpoint.get("epoch", ""),
                            "global_step": checkpoint.get("global_step", ""),
                            "dataset": task_name,
                            "split": split,
                            "class_index": class_row["class_index"],
                            "class_name": class_row["class_name"],
                            "positives": class_row["positives"],
                            "negatives": class_row["negatives"],
                            "auc": (
                                "nan"
                                if np.isnan(class_row["auc"])
                                else f"{class_row['auc']:.8f}"
                            ),
                            "macro_curve_auc": (
                                "nan"
                                if np.isnan(metrics["macro_curve_auc"])
                                else f"{metrics['macro_curve_auc']:.8f}"
                            ),
                            "date_time": now,
                        }
                    )

                per_class_rows.append(
                    {
                        "checkpoint": args.checkpoint,
                        "epoch": checkpoint.get("epoch", ""),
                        "global_step": checkpoint.get("global_step", ""),
                        "dataset": task_name,
                        "split": split,
                        "class_index": "mean",
                        "class_name": "mean_class_auc",
                        "positives": "",
                        "negatives": "",
                        "auc": (
                            "nan"
                            if np.isnan(metrics["mean_auroc"])
                            else f"{metrics['mean_auroc']:.8f}"
                        ),
                        "macro_curve_auc": (
                            "nan"
                            if np.isnan(metrics["macro_curve_auc"])
                            else f"{metrics['macro_curve_auc']:.8f}"
                        ),
                        "date_time": now,
                    }
                )

                if isinstance(metrics["mean_auroc_top5"], float):
                    per_class_rows.append(
                        {
                            "checkpoint": args.checkpoint,
                            "epoch": checkpoint.get("epoch", ""),
                            "global_step": checkpoint.get("global_step", ""),
                            "dataset": task_name,
                            "split": split,
                            "class_index": "top5",
                            "class_name": "top5_mean_auroc",
                            "positives": "",
                            "negatives": "",
                            "auc": (
                                "nan"
                                if np.isnan(metrics["mean_auroc_top5"])
                                else f"{metrics['mean_auroc_top5']:.8f}"
                            ),
                            "macro_curve_auc": "",
                            "date_time": now,
                        }
                    )

                if not args.no_roc_curves:
                    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
                    run_dir = os.path.dirname(checkpoint_dir)
                    roc_output_dir = args.roc_output_dir or os.path.join(run_dir, "roc_curves")
                    paths = write_roc_curve_files(
                        output_dir=roc_output_dir,
                        dataset_name=task_name,
                        split=split,
                        curves=metrics["roc_curves"],
                    )
                    print(f"Wrote ROC plot to {paths['plot']}")
                    print(f"Wrote ROC points to {paths['points']}")

    if args.output is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.output = os.path.join(os.path.dirname(checkpoint_dir), "arkplus_eval_metrics.csv")
    write_metrics_csv(args.output, rows)
    print(f"Wrote metrics to {args.output}")

    if args.per_class_output is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.per_class_output = os.path.join(
            os.path.dirname(checkpoint_dir), "arkplus_test_per_class_auc.csv"
        )
    write_per_class_auc_csv(args.per_class_output, per_class_rows)
    if per_class_rows:
        print(f"Wrote per-class test AUC to {args.per_class_output}")


if __name__ == "__main__":
    main()
