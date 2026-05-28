import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from arkplus_model import build_arkplus_model
from evaluate_ark_plus import (
    apply_dataset_dir_override,
    build_eval_dataset,
    evaluate_split,
    get_top5_auc,
)
from utils import ensure_dir, get_best_accelerator


DEFAULT_DATASETS = ["chestxray14", "mimic", "chexpert"]
TOP5_SELECTION_DATASETS = {"mimic", "chexpert"}

TARGET_DATASET_ALIASES = {
    "chestxray14": {"chestxray14", "chest-xray14", "xray14", "nih", "nih_chestxray14"},
    "mimic": {"mimic", "mimic-cxr", "mimiccxr"},
    "chexpert": {"chexpert"},
}

RESULT_FIELDNAMES = [
    "checkpoint",
    "checkpoint_name",
    "epoch",
    "global_step",
    "dataset",
    "selection_metric",
    "status",
    "error",
    "val_samples",
    "val_loss",
    "val_mean_auroc",
    "val_mean_auroc_top5",
    "val_selection_metric",
    "val_accuracy",
    "val_per_label_auroc",
    "test_samples",
    "test_loss",
    "test_mean_auroc",
    "test_mean_auroc_top5",
    "test_selection_metric",
    "test_accuracy",
    "test_per_label_auroc",
    "date_time",
]

SUMMARY_FIELDNAMES = [
    "criterion",
    "checkpoint",
    "checkpoint_name",
    "epoch",
    "global_step",
    "dataset",
    "selection_metric",
    "val_samples",
    "val_loss",
    "val_mean_auroc",
    "val_mean_auroc_top5",
    "val_selection_metric",
    "val_accuracy",
    "val_per_label_auroc",
    "test_samples",
    "test_loss",
    "test_mean_auroc",
    "test_mean_auroc_top5",
    "test_selection_metric",
    "test_accuracy",
    "test_per_label_auroc",
    "date_time",
]


def normalize_name(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(" ", "")


def torch_load_checkpoint(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_float(value: object, precision: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{float(value):.{precision}f}"
    return str(value)


def format_per_label(value: object) -> str:
    if not isinstance(value, list):
        return "" if value is None else str(value)
    return ";".join("nan" if np.isnan(x) else f"{float(x):.6f}" for x in value)


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def selection_metric_name(dataset_name: str) -> str:
    if normalize_name(dataset_name) in TOP5_SELECTION_DATASETS:
        return "mean_auroc_top5"
    return "mean_auroc"


def dataset_aliases(dataset_name: str) -> set:
    normalized = normalize_name(dataset_name)
    for aliases in TARGET_DATASET_ALIASES.values():
        normalized_aliases = {normalize_name(alias) for alias in aliases}
        if normalized in normalized_aliases:
            return aliases
    return {dataset_name}


def metric_column(split: str, metric_name: str) -> str:
    return f"{split}_{metric_name}"


def completion_key(checkpoint: str, dataset_name: str) -> str:
    checkpoint_path = str(Path(checkpoint).expanduser().resolve())
    return f"{checkpoint_path}\t{normalize_name(dataset_name)}"


def default_output_path(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    return run_dir / "arkplus_checkpoint_sweep.csv"


def default_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def discover_checkpoints(
    checkpoint_dir: Path,
    pattern: str,
    recursive: bool,
    newest_first: bool,
) -> List[Path]:
    iterator = checkpoint_dir.rglob(pattern) if recursive else checkpoint_dir.glob(pattern)
    paths = [path.resolve() for path in iterator if path.is_file()]
    paths.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=newest_first)
    return paths


def migrate_output_schema(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames == RESULT_FIELDNAMES:
            return

    ensure_dir(str(output_path.parent))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            metric_name = row.get("selection_metric") or selection_metric_name(row.get("dataset", ""))
            row["selection_metric"] = metric_name
            row["val_selection_metric"] = row.get("val_selection_metric") or row.get(
                metric_column("val", metric_name), ""
            )
            row["test_selection_metric"] = row.get("test_selection_metric") or row.get(
                metric_column("test", metric_name), ""
            )
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDNAMES})


def read_completed_checkpoints(output_path: Path) -> Tuple[set, List[Dict[str, str]]]:
    if not output_path.exists():
        return set(), []

    completed = set()
    rows = []
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            checkpoint = row.get("checkpoint", "")
            dataset = row.get("dataset", "")
            metric_name = row.get("selection_metric") or selection_metric_name(dataset)
            if (
                row.get("status") == "ok"
                and checkpoint
                and dataset
                and row.get(metric_column("val", metric_name), "") != ""
                and row.get(metric_column("test", metric_name), "") != ""
            ):
                completed.add(completion_key(checkpoint, dataset))
    return completed, rows


def write_csv_header(output_path: Path, fieldnames: Sequence[str]) -> None:
    ensure_dir(str(output_path.parent))
    with open(output_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_result_row(output_path: Path, row: Dict[str, object]) -> None:
    ensure_dir(str(output_path.parent))
    exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_FIELDNAMES})


def latest_ok_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest = {}
    for row in rows:
        checkpoint = row.get("checkpoint", "")
        dataset = row.get("dataset", "")
        if row.get("status") == "ok" and checkpoint and dataset:
            latest[completion_key(checkpoint, dataset)] = row
    return list(latest.values())


def write_summary(output_path: Path, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    ok_rows = latest_ok_rows(rows)
    dataset_order = {normalize_name(name): idx for idx, name in enumerate(DEFAULT_DATASETS)}
    datasets = sorted(
        {row.get("dataset", "") for row in ok_rows if row.get("dataset", "")},
        key=lambda name: (dataset_order.get(normalize_name(name), 99), normalize_name(name)),
    )

    summary_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in ok_rows if normalize_name(row.get("dataset", "")) == normalize_name(dataset)]
        metric_name = selection_metric_name(dataset)
        best_val = best_row(dataset_rows, metric_column("val", metric_name))
        best_test = best_row(dataset_rows, metric_column("test", metric_name))
        if best_val is not None:
            summary_rows.append(summary_row("best_validation", metric_name, best_val))
        if best_test is not None:
            summary_rows.append(summary_row("best_test", metric_name, best_test))

    ensure_dir(str(output_path.parent))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDNAMES})
    return summary_rows


def best_row(rows: List[Dict[str, str]], metric_name: str) -> Optional[Dict[str, str]]:
    candidates = []
    for row in rows:
        value = parse_float(row.get(metric_name))
        if value is not None and not np.isnan(value):
            candidates.append((value, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def summary_row(criterion: str, metric_name: str, row: Dict[str, str]) -> Dict[str, str]:
    summary = {"criterion": criterion, "selection_metric": metric_name}
    for key in SUMMARY_FIELDNAMES:
        if key not in summary:
            summary[key] = row.get(key, "")
    return summary


def load_config(checkpoint: Dict) -> DictConfig:
    if "config" not in checkpoint:
        raise KeyError("checkpoint does not contain a 'config' entry")
    return OmegaConf.create(checkpoint["config"])


def get_num_classes_list(checkpoint: Dict, cfg: DictConfig) -> List[int]:
    task_metadata = checkpoint.get("task_metadata")
    if task_metadata:
        return [len(task["labels"]) for task in task_metadata]
    return [len(task.labels) for task in cfg.ark.datasets]


def find_dataset(cfg: DictConfig, dataset_name: str) -> Tuple[int, DictConfig]:
    normalized_aliases = {normalize_name(alias) for alias in dataset_aliases(dataset_name)}

    for head_index, task_cfg in enumerate(cfg.ark.datasets):
        if normalize_name(str(task_cfg.name)) in normalized_aliases:
            return head_index, task_cfg

    available = [str(task.name) for task in cfg.ark.datasets]
    raise ValueError(f"Could not find dataset '{dataset_name}' in checkpoint config. Available: {available}")


def build_eval_contexts(
    checkpoint_path: Path,
    dataset_names: List[str],
    dataset_dir: Optional[str],
    batch_size_override: Optional[int],
    workers_override: Optional[int],
    validate_paths: bool,
    base_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    cfg = load_config(checkpoint)
    apply_dataset_dir_override(cfg, dataset_dir)

    batch_size = int(batch_size_override or cfg.ark.get("eval_batch_size", 100))
    workers = int(workers_override if workers_override is not None else cfg.ark.get("workers", 8))
    model_kwargs = {
        "backbone": str(cfg.model.backbone),
        "num_classes_list": get_num_classes_list(checkpoint, cfg),
        "pretrained": False,
        "projector_features": cfg.ark.get("projector_features", 1376),
        "use_mlp": bool(cfg.ark.get("use_mlp", False)),
    }

    contexts = []
    seen = set()
    for dataset_name in dataset_names:
        head_index, task_cfg = find_dataset(cfg, dataset_name)
        task_name = str(task_cfg.name)
        normalized_task_name = normalize_name(task_name)
        if normalized_task_name in seen:
            continue
        seen.add(normalized_task_name)

        task_type = str(task_cfg.get("task_type", "multi-label classification"))
        label_names = list(task_cfg.labels)
        loaders = {}
        for split in ("val", "test"):
            dataset = build_eval_dataset(cfg, task_cfg, split, str(base_dir), validate_paths)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=device.type == "cuda",
            )

        contexts.append(
            {
                "head_index": head_index,
                "task_name": task_name,
                "task_type": task_type,
                "label_names": label_names,
                "loaders": loaders,
                "selection_metric": selection_metric_name(task_name),
            }
        )

    return contexts, model_kwargs


def base_row(checkpoint_path: Path, checkpoint: Dict, context: Dict[str, object]) -> Dict[str, object]:
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "epoch": checkpoint.get("epoch", ""),
        "global_step": checkpoint.get("global_step", ""),
        "dataset": context["task_name"],
        "selection_metric": context["selection_metric"],
        "status": "ok",
        "error": "",
        "date_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    contexts: List[Dict[str, object]],
    model_kwargs: Dict[str, object],
    checkpoint_key: str,
    device: torch.device,
) -> List[Dict[str, object]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if checkpoint_key not in checkpoint:
        raise KeyError(f"checkpoint key '{checkpoint_key}' not found in {checkpoint_path}")

    model = build_arkplus_model(**model_kwargs)
    model.load_state_dict(checkpoint[checkpoint_key], strict=True)
    model.to(device)
    model.eval()

    rows = []
    try:
        for context in contexts:
            row = base_row(checkpoint_path, checkpoint, context)
            try:
                for split in ("val", "test"):
                    metrics = evaluate_split(
                        model=model,
                        loader=context["loaders"][split],
                        device=device,
                        head_index=int(context["head_index"]),
                        task_type=str(context["task_type"]),
                        num_labels=len(context["label_names"]),
                        label_names=list(context["label_names"]),
                    )
                    top5_auc = get_top5_auc(
                        str(context["task_name"]),
                        list(context["label_names"]),
                        metrics.get("per_label_auroc"),
                    )
                    selected_metric_name = str(context["selection_metric"])
                    selected_value = top5_auc if selected_metric_name == "mean_auroc_top5" else metrics["mean_auroc"]

                    row[f"{split}_samples"] = metrics["samples"]
                    row[f"{split}_loss"] = format_float(metrics["loss"])
                    row[f"{split}_mean_auroc"] = format_float(metrics["mean_auroc"])
                    row[f"{split}_mean_auroc_top5"] = format_float(top5_auc)
                    row[f"{split}_selection_metric"] = format_float(selected_value)
                    row[f"{split}_accuracy"] = format_float(metrics["accuracy"])
                    row[f"{split}_per_label_auroc"] = format_per_label(metrics["per_label_auroc"])
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    finally:
        del model
        del checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def error_row(checkpoint_path: Path, dataset_name: str, error: Exception) -> Dict[str, object]:
    metric_name = selection_metric_name(dataset_name)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "dataset": dataset_name,
        "selection_metric": metric_name,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "date_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Ark+ checkpoints on ChestXray14, MIMIC, and CheXpert validation/test splits. "
            "MIMIC and CheXpert are selected by top-5 AUROC; ChestXray14 is selected by mean AUROC."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint files")
    parser.add_argument("--pattern", default="*.pth.tar", help="Checkpoint glob pattern")
    parser.add_argument("--recursive", action="store_true", help="Search checkpoint directory recursively")
    parser.add_argument("--oldest-first", action="store_true", help="Evaluate oldest checkpoint first")
    parser.add_argument("--output", default=None, help="Resumable per-checkpoint/per-dataset results CSV")
    parser.add_argument("--summary-output", default=None, help="Best-checkpoint summary CSV")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate; defaults to chestxray14 mimic chexpert",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="dataset_overrides",
        default=None,
        help="Backward-compatible single dataset option; repeat to evaluate multiple datasets",
    )
    parser.add_argument("--dataset-dir", default=None, help="Override dataset root directory")
    parser.add_argument("--base-dir", default=None, help="Base directory for relative CSV/image paths")
    parser.add_argument("--checkpoint-key", default="teacher", help="Checkpoint state dict key to evaluate")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or a torch device string")
    parser.add_argument("--no-validate-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-evaluate checkpoints and overwrite CSV outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.dataset_overrides or args.datasets
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    checkpoints = discover_checkpoints(
        checkpoint_dir,
        args.pattern,
        args.recursive,
        newest_first=not args.oldest_first,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir}/{args.pattern}")

    output_path = Path(args.output).expanduser() if args.output else default_output_path(checkpoint_dir)
    summary_path = (
        Path(args.summary_output).expanduser()
        if args.summary_output
        else default_summary_path(output_path)
    )

    if args.force:
        write_csv_header(output_path, RESULT_FIELDNAMES)
        completed = set()
    else:
        migrate_output_schema(output_path)
        completed, _ = read_completed_checkpoints(output_path)

    device = torch.device(args.device or get_best_accelerator())
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    validate_paths = not args.no_validate_paths

    contexts, model_kwargs = build_eval_contexts(
        checkpoint_path=checkpoints[0],
        dataset_names=dataset_names,
        dataset_dir=args.dataset_dir,
        batch_size_override=args.batch_size,
        workers_override=args.workers,
        validate_paths=validate_paths,
        base_dir=base_dir,
        device=device,
    )

    print(f"Processing {len(checkpoints)} checkpoint(s) {'oldest' if args.oldest_first else 'newest'} first.")
    print("Selection metrics: ChestXray14=mean_auroc, MIMIC/CheXpert=mean_auroc_top5")

    evaluated_count = 0
    skipped_count = 0
    for checkpoint_path in checkpoints:
        pending_contexts = [
            context
            for context in contexts
            if completion_key(str(checkpoint_path), str(context["task_name"])) not in completed
        ]
        skipped_count += len(contexts) - len(pending_contexts)
        if not pending_contexts:
            print(f"Skipping completed checkpoint for all requested datasets: {checkpoint_path}")
            continue

        print(f"Evaluating checkpoint: {checkpoint_path}")
        try:
            rows = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                contexts=pending_contexts,
                model_kwargs=model_kwargs,
                checkpoint_key=args.checkpoint_key,
                device=device,
            )
        except Exception as exc:
            rows = [error_row(checkpoint_path, str(context["task_name"]), exc) for context in pending_contexts]

        for row in rows:
            append_result_row(output_path, row)
            if row.get("status") == "ok":
                evaluated_count += 1
                completed.add(completion_key(str(row["checkpoint"]), str(row["dataset"])))
                metric_name = str(row.get("selection_metric", ""))
                print(
                    f"  {row['dataset']} {metric_name}: "
                    f"val={row.get('val_selection_metric', '')} "
                    f"test={row.get('test_selection_metric', '')}"
                )
            else:
                print(f"  ERROR {row.get('dataset', '')}: {row.get('error', '')}")

    all_rows = []
    if output_path.exists():
        with open(output_path, newline="") as f:
            all_rows = list(csv.DictReader(f))
    summary_rows = write_summary(summary_path, all_rows)

    print(f"Wrote checkpoint results to {output_path}")
    print(f"Wrote best-checkpoint summary to {summary_path}")
    print(f"Evaluated {evaluated_count} dataset/checkpoint row(s); skipped {skipped_count}.")
    for row in summary_rows:
        print(
            f"{row['dataset']} {row['criterion']} ({row['selection_metric']}): "
            f"{row['checkpoint']} "
            f"val={row.get('val_selection_metric', '')} "
            f"test={row.get('test_selection_metric', '')}"
        )


if __name__ == "__main__":
    main()
