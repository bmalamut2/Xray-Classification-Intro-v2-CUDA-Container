import contextlib
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision import transforms

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from dataset import CSVDataset
from utils import ensure_dir, get_device, get_best_accelerator, seed_everything
from evaluate import evaluate, save_predictions, save_val_auroc

# Register custom resolver for formatting learning rate
def get_absolute_path(path: str) -> str:
    import os
    if os.path.isabs(path):
        return path
    try:
        from hydra.utils import get_original_cwd
        original_cwd = get_original_cwd()
    except Exception:
        original_cwd = os.environ.get("ORIGINAL_CWD", os.getcwd())
    return os.path.join(original_cwd, path)

def format_lr(lr: float) -> str:
    """Format learning rate by replacing dots with underscores."""
    return str(lr).replace(".", "_")

OmegaConf.register_new_resolver("format_lr", format_lr)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _state_dict_cpu(model: nn.Module) -> dict:
    """Copy model state dict to CPU. Avoids HPU/Gaudi serialization errors on torch.save."""
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {k: v.cpu().clone() for k, v in state_dict.items()}


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss that ignores targets == -1."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != -1
        targets_clamped = torch.where(mask, targets, torch.zeros_like(targets))
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_clamped,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss = loss * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def should_log(batch_idx: int, epoch: int) -> bool:
    iteration = batch_idx + 1
    if iteration <= 20:
        return True
    if epoch <= 2:
        return iteration % 10 == 0
    return iteration % 50 == 0


def get_lrs(optimizer: optim.Optimizer) -> Tuple[float, float]:
    if len(optimizer.param_groups) >= 2:
        return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
    lr = optimizer.param_groups[0]["lr"]
    return lr, lr


def abs_or_none(path: str):
    if path is None:
        return None
    return get_absolute_path(path)


def build_transforms(image_size: int, normalize: str, train: bool = True):
    t = []
    
    if train:
        t.append(transforms.Resize((image_size, image_size)))

        # Disable horizontal flipping as this is medical image
        t.append(transforms.RandomHorizontalFlip())
        
        t.append(transforms.RandomRotation(10))
        
        if image_size == 256:
            t.append(
                transforms.RandomChoice([
                    transforms.Resize((224, 224)),
                    transforms.RandomCrop(224)
                ], p=[0.1, 0.9])
            )
            
        t.append(transforms.ToTensor())
    else:
        # Valuation and test rules: remove 10-crop and center crop, use nocrop
        if image_size == 224 or image_size == 256:
            # For 256x256 model architecture, directly resize to 224x224
            t.append(transforms.Resize((224, 224)))
            t.append(transforms.ToTensor())
        else:
            # For models accepting 512x512, only resize to 512x512
            t.append(transforms.Resize((image_size, image_size)))
            t.append(transforms.ToTensor())

    if normalize == "imagenet":
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t.append(normalize_transform)
            
    return transforms.Compose(t)


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    model_name = backbone
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise ValueError(f"Unsupported backbone: {backbone} (timm name: {model_name})") from exc


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def load_arkplus_encoder_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    checkpoint_key: str = "teacher",
    load_mode: str = "encoder",
) -> Dict[str, object]:
    if load_mode != "encoder":
        raise ValueError(f"Unsupported Ark+ load mode: {load_mode}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
        raise KeyError(
            f"Checkpoint key '{checkpoint_key}' not found in {checkpoint_path}. "
            f"Available keys: {available}"
        )

    target_state = model.state_dict()
    loadable = {}
    skipped_non_encoder = []
    skipped_missing = []
    skipped_shape = []

    for raw_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = _strip_module_prefix(raw_key)
        if not key.startswith("encoder."):
            skipped_non_encoder.append(key)
            continue
        target_key = key[len("encoder.") :]
        if target_key not in target_state:
            skipped_missing.append(target_key)
            continue
        if target_state[target_key].shape != value.shape:
            skipped_shape.append((target_key, tuple(value.shape), tuple(target_state[target_key].shape)))
            continue
        loadable[target_key] = value

    if not loadable:
        raise ValueError(
            f"No compatible encoder weights found in Ark+ checkpoint {checkpoint_path} "
            f"with key '{checkpoint_key}'."
        )

    incompatible = model.load_state_dict(loadable, strict=False)
    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_key": checkpoint_key,
        "loaded": len(loadable),
        "skipped_non_encoder": len(skipped_non_encoder),
        "skipped_missing": len(skipped_missing),
        "skipped_shape": len(skipped_shape),
        "missing_after_load": len(incompatible.missing_keys),
        "unexpected_after_load": len(incompatible.unexpected_keys),
        "sample_loaded": list(loadable.keys())[:5],
        "sample_missing": skipped_missing[:5],
        "sample_shape": skipped_shape[:5],
    }
    return summary


def make_run_id(dataset_name: str, backbone: str, opt_type: str, lr: float) -> str:
    """Create a run ID with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = str(lr).replace(".", "_")
    return f"{dataset_name}_{backbone}_{opt_type}_{lr_str}_{ts}"


def _train_impl(cfg: DictConfig, rank: int = 0, world_size: int = 1, local_rank: int = 0) -> None:
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    train_cfg = cfg.train

    # Get dataset name
    dataset_name = dataset_cfg.get("name", "dataset")
    
    seed_everything(int(train_cfg.get("seed", 42)))

    label_names = dataset_cfg["labels"]
    num_classes = len(label_names)
    cfg_num_classes = dataset_cfg.get("num_classes")
    if cfg_num_classes is not None and int(cfg_num_classes) != num_classes:
        raise ValueError(
            f"Config num_classes ({cfg_num_classes}) != len(labels) ({num_classes})."
        )
    
    # Extract test_labels subsets if defined and convert label names to indices
    label_subsets = None
    if "test_labels" in dataset_cfg and dataset_cfg.test_labels is not None:
        label_subsets = {}
        test_labels_cfg = dataset_cfg.test_labels
        # Create a mapping from label name to index
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        
        # Process each subset (top5, top12, top6, top14, etc.)
        from omegaconf import ListConfig
        for subset_name, subset_labels in test_labels_cfg.items():
            if isinstance(subset_labels, (list, ListConfig)):
                # Convert label names to indices
                subset_indices = []
                for label_name in subset_labels:
                    if label_name in label_to_idx:
                        subset_indices.append(label_to_idx[label_name])
                    else:
                        logger.warning(f"Label '{label_name}' in {subset_name} not found in label_names, skipping")
                if subset_indices:
                    label_subsets[subset_name] = subset_indices
                    logger.info(f"Found {subset_name} with {len(subset_indices)} labels: {subset_labels}")

    # Get optimizer, lr, and backbone for later use
    opt_type = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("lr", 1e-4))
    backbone = model_cfg.get("backbone", "model")
    
    # Get run_name from config (constructed in config.yaml using Hydra resolvers)
    # Format: {dataset_name}_{backbone}_{optimizer}_{lr}_{date}_{time}
    run_name = cfg.get("run_name", "run_0")
    
    # Use make_run_id to construct run_id (for logging purposes)
    run_id = make_run_id(dataset_name, backbone, opt_type, lr)
    
    # Allow override via environment variable
    env_run_id = os.environ.get("RUN_ID")
    if env_run_id:
        # Override run_name if RUN_ID is set
        run_name = env_run_id
        run_id = env_run_id
        cfg.run_name = run_name
    
    # Create output directory: outputs/{dataset_name}/{run_name}/
    # Note: Hydra changes working directory to outputs/${dataset.name}, so we just use run_name
    # Note: .hydra directory is already in the correct location (outputs/{dataset_name}/{run_name}/.hydra)
    # because run_name is constructed in config.yaml and used for hydra.output_subdir
    output_dir = run_name
    ensure_dir(output_dir)
    
    # Create checkpoints subdirectory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(checkpoints_dir)
    
    # Setup file logging to output directory
    log_file = os.path.join(output_dir, "training.log")
    if rank != 0:
        logger.setLevel(logging.ERROR)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    train_csv = get_absolute_path(dataset_cfg.train_ann)
    val_csv = get_absolute_path(dataset_cfg.val_ann)
    test_csv = get_absolute_path(dataset_cfg.test_ann)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_tf = build_transforms(model_cfg["image_size"], model_cfg.get("normalize", "imagenet"), train=True)
    val_tf = build_transforms(model_cfg["image_size"], model_cfg.get("normalize", "imagenet"), train=False)
    arkplus_checkpoint = model_cfg.get("arkplus_checkpoint")
    use_label_policy = bool(arkplus_checkpoint) or bool(dataset_cfg.get("apply_label_policy", False))
    label_policy = {
        "uncertain_label": dataset_cfg.get("uncertain_label", "Ignore") if use_label_policy else "Ignore",
        "eval_uncertain_label": dataset_cfg.get("eval_uncertain_label") if use_label_policy else None,
        "unknown_label": float(dataset_cfg.get("unknown_label", -1.0)) if use_label_policy else -1.0,
    }

    train_ds = CSVDataset(
        train_df,
        image_key=dataset_cfg.get("image_path_key", "Path"),
        label_names=label_names,
        image_root=abs_or_none(dataset_cfg.get("train_image_root")),
        image_append=dataset_cfg.get("image_path_append", ""),
        transform=train_tf,
        split="train",
        **label_policy,
    )
    val_ds = CSVDataset(
        val_df,
        image_key=dataset_cfg.get("image_path_key", "Path"),
        label_names=label_names,
        image_root=abs_or_none(dataset_cfg.get("val_image_root")),
        image_append=dataset_cfg.get("image_path_append", ""),
        transform=val_tf,
        split="val",
        **label_policy,
    )
    test_ds = CSVDataset(
        test_df,
        image_key=dataset_cfg.get("image_path_key", "Path"),
        label_names=label_names,
        image_root=abs_or_none(dataset_cfg.get("test_image_root")),
        image_append=dataset_cfg.get("image_path_append", ""),
        transform=val_tf,
        split="test",
        **label_policy,
    )

    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    device = get_device()
    device_type = get_best_accelerator()

    model = build_model(model_cfg.get("backbone", "resnet50"), num_classes, model_cfg.get("pretrained", True))
    arkplus_load_summary = None
    if arkplus_checkpoint:
        arkplus_load_summary = load_arkplus_encoder_checkpoint(
            model,
            get_absolute_path(str(arkplus_checkpoint)),
            checkpoint_key=str(model_cfg.get("arkplus_checkpoint_key", "teacher")),
            load_mode=str(model_cfg.get("arkplus_load_mode", "encoder")),
        )
    
    if hasattr(model, "num_classes") and int(getattr(model, "num_classes")) != num_classes:
        raise ValueError(
            f"Model output classes ({model.num_classes}) != num_classes ({num_classes})."
        )
        
    if world_size > 1:
        if rank == 0: logger.info(f"Using {world_size} devices with DistributedDataParallel.")
        from torch.nn.parallel import DistributedDataParallel as DDP
        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
            model.to(device)
            model = DDP(model, device_ids=[local_rank])
        elif device_type == "hpu":
            model.to(device)
            model = DDP(model, bucket_cap_mb=100, gradient_as_bucket_view=True)
        else:
            model.to(device)
            model = DDP(model, bucket_cap_mb=100)

    model.to(device)


    loss_fn = MaskedBCEWithLogitsLoss()

    # Mixed precision: prefer bf16 when supported (Ampere+), else fp16 (AMP) when accelerator available
    use_amp = device_type != "cpu"
    bf16_supported = False
    if device_type == "cuda" and torch.cuda.is_available():
        bf16_supported = torch.cuda.is_bf16_supported()
        if bf16_supported:
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.info(
                    f"CUDA architecture (compute capability {major}.x) "
                    "reports bfloat16 support, but it may cause engine errors "
                    "on non-Ampere+ GPUs. Falling back to float16 (AMP)."
                )
                bf16_supported = False
    elif device_type == "hpu":
        bf16_supported = True
        
    use_bf16 = use_amp and bf16_supported
    if use_bf16:
        autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        scaler = None
    elif use_amp:
        autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        scaler = (
            torch.amp.GradScaler(device_type)
            if hasattr(torch.amp, "GradScaler")
            else (torch.amp.GradScaler() if device_type == "cuda" else None)
        )
    else:
        autocast_ctx = contextlib.nullcontext()
        scaler = None

    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

    epochs = int(train_cfg.get("epochs", 10))
    eval_every_iter = int(train_cfg.get("eval_every_iter", 1000))
    should_save_predictions = bool(train_cfg.get("save_predictions", True))
    debug = bool(OmegaConf.select(cfg, "debug", default=False))

    # Create scheduler
    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg and scheduler_cfg.get("name") != "none":
        scheduler_name = scheduler_cfg.name.lower()
        if scheduler_name == "reduce_lr_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=scheduler_cfg.get("factor", 0.2),
                patience=scheduler_cfg.get("patience", 2),
                threshold=scheduler_cfg.get("threshold", 0.0001),
                cooldown=scheduler_cfg.get("cooldown", 1),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
                verbose=scheduler_cfg.get("verbose", True)
            )
            
            warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs]
                )
        elif scheduler_name == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get("T_max", epochs),
                eta_min=float(scheduler_cfg.get("eta_min", 1e-5)),
            )
        else:
            logger.warning(f"Unsupported scheduler: {scheduler_name}, using none")


    monitor_name = train_cfg.get("monitor", "mean_auroc")
    monitor_mode = train_cfg.get("monitor_mode", "max")
    # Initialize best_metric based on monitor_mode
    # For "max": start with -inf (any value will be better)
    # For "min": start with +inf (any value will be better)
    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    best_val_auroc = -float("inf")  # Track best validation AUROC for val_auroc.csv
    patience = int(train_cfg.get("patience", 10))
    global_iter = 0
    eval_count = 0
    start_time = time.time()
    
    # Log all configuration before training starts
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("")
    logger.info("Dataset Configuration:")
    logger.info(f"  Name: {dataset_name}")
    logger.info(f"  Train CSV: {train_csv}")
    logger.info(f"  Val CSV: {val_csv}")
    logger.info(f"  Test CSV: {test_csv}")
    logger.info(f"  Train Image Root: {dataset_cfg.get('train_image_root', 'None')}")
    logger.info(f"  Val Image Root: {dataset_cfg.get('val_image_root', 'None')}")
    logger.info(f"  Test Image Root: {dataset_cfg.get('test_image_root', 'None')}")
    logger.info(f"  Image Path Key: {dataset_cfg.get('image_path_key', 'Path')}")
    logger.info(f"  Image Path Append: {dataset_cfg.get('image_path_append', '')}")
    logger.info(f"  Label Policy Enabled: {use_label_policy}")
    if use_label_policy:
        logger.info(f"  Train Uncertain Label: {label_policy['uncertain_label']}")
        logger.info(f"  Eval Uncertain Label: {label_policy['eval_uncertain_label']}")
        logger.info(f"  Unknown Label: {label_policy['unknown_label']}")
    logger.info(f"  Number of Classes: {num_classes}")
    logger.info(f"  Labels: {label_names}")
    logger.info(f"  Train Samples: {len(train_df):,}")
    logger.info(f"  Val Samples: {len(val_df):,}")
    logger.info(f"  Test Samples: {len(test_df):,}")
    if label_subsets:
        logger.info(f"  Label Subsets: {label_subsets}")
    logger.info("")
    logger.info("Model Configuration:")
    logger.info(f"  Backbone: {backbone}")
    logger.info(f"  Pretrained: {model_cfg.get('pretrained', True)}")
    logger.info(f"  Ark+ Checkpoint: {arkplus_checkpoint}")
    if arkplus_load_summary:
        logger.info(f"  Ark+ Checkpoint Key: {arkplus_load_summary['checkpoint_key']}")
        logger.info(f"  Ark+ Loaded Encoder Tensors: {arkplus_load_summary['loaded']}")
        logger.info(f"  Ark+ Skipped Non-Encoder Tensors: {arkplus_load_summary['skipped_non_encoder']}")
        logger.info(f"  Ark+ Skipped Missing Tensors: {arkplus_load_summary['skipped_missing']}")
        logger.info(f"  Ark+ Skipped Shape-Mismatched Tensors: {arkplus_load_summary['skipped_shape']}")
        if arkplus_load_summary["sample_shape"]:
            logger.info(f"  Ark+ Shape Mismatch Examples: {arkplus_load_summary['sample_shape']}")
    logger.info(f"  Image Size: {model_cfg.get('image_size', 224)}")
    logger.info(f"  Normalize: {model_cfg.get('normalize', 'imagenet')}")
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {train_cfg.get('batch_size', 32)}")
    logger.info(f"  Num Workers: {train_cfg.get('num_workers', 4)}")
    logger.info(f"  Optimizer: {opt_type}")
    logger.info(f"  Learning Rate: {lr}")
    logger.info(f"  Weight Decay: {weight_decay}")
    if opt_type == "sgd":
        logger.info(f"  Momentum: {train_cfg.get('momentum', 0.9)}")
    logger.info(f"  Seed: {train_cfg.get('seed', 42)}")
    logger.info(f"  Eval Every Iter: {eval_every_iter}")
    logger.info(f"  Monitor: {monitor_name} ({monitor_mode})")
    logger.info(f"  Patience: {patience} evals")
    logger.info(f"  Save Predictions: {should_save_predictions}")
    if scheduler:
        logger.info(f"  Scheduler: {scheduler_cfg.name}")
        for k, v in scheduler_cfg.items():
            if k != "name":
                logger.info(f"    {k}: {v}")
    else:
        logger.info("  Scheduler: None")
    
    logger.info(f"  Save Best Model: {train_cfg.get('save_best_model', True)}")
    logger.info(f"  Debug (1 batch + eval + exit): {debug}")
    precision_str = "bf16-mixed" if use_bf16 else ("16-mixed (AMP)" if use_amp else "32-true")
    logger.info(f"  Precision (AMP): {precision_str}")
    logger.info("")
    logger.info("Device Configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        iters_per_epoch = len(train_loader)
        for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(images)
                loss = loss_fn(logits, labels)
            
            # Record global loss for logging (average across all ranks)
            if world_size > 1:
                import torch.distributed as dist
                dist_loss = loss.detach().clone()
                dist.all_reduce(dist_loss, op=dist.ReduceOp.SUM)
                dist_loss /= world_size
                loss_val = dist_loss.item()
            else:
                loss_val = loss.item()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_iter += 1

            bs = images.size(0)
            loss_meter.update(loss_val, bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if should_log(batch_idx, epoch):
                elapsed = time.time() - start_time
                total_iters = max(1, iters_per_epoch * epochs)
                eta = (elapsed / max(1, global_iter)) * (total_iters - global_iter)
                lr_head, lr_back = get_lrs(optimizer)
                temp_str = ""
                logger.info(
                    f"Rank {rank} {dataset_name} {opt_type} E={epoch}/{epochs} "
                    f"g_step={global_iter} [B {batch_idx}/{iters_per_epoch}] "
                    f"BT={batch_time.val:.2f}({batch_time.avg:.2f}) "
                    f"DT={data_time.val:.2f}({data_time.avg:.2f}) "
                    f"LR={lr_head:.2e}/{lr_back:.2e} Loss={loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                    f"Time {seconds_to_hms(elapsed)}/{seconds_to_hms(eta)} {temp_str}"
                )

            if debug or (global_iter % eval_every_iter == 0):
                eval_count += 1
                val_metrics, val_probs, val_targets, val_paths, val_keys = evaluate(model, val_loader, device, label_names, label_subsets)
                test_metrics, test_probs, test_targets, test_paths, test_keys = evaluate(model, test_loader, device, label_names, label_subsets)
                
                # Prefix metrics with split name for clarity
                val_metrics_prefixed = {f"val_{key}": value for key, value in val_metrics.items()}
                test_metrics_prefixed = {f"test_{key}": value for key, value in test_metrics.items()}
                
                metric_value = val_metrics.get(monitor_name, float("nan"))
                
                # Display mean AUROC and per-label AUROC
                val_mean_auroc = val_metrics.get("mean_auroc", float("nan"))
                val_per_label_auroc = val_metrics.get("per_label_auroc", [])
                test_mean_auroc = test_metrics.get("mean_auroc", float("nan"))
                test_per_label_auroc = test_metrics.get("per_label_auroc", [])
                
                logger.info("=" * 80)
                logger.info(f"Iter {global_iter} (Epoch {epoch}) Evaluation Results")
                logger.info("=" * 80)
                logger.info(f"Validation Mean AUROC: {val_mean_auroc:.4f}")
                logger.info(f"Test Mean AUROC: {test_mean_auroc:.4f}")
                logger.info("")
                logger.info("Validation Per-Label AUROC:")
                for i, (label_name, auroc) in enumerate(zip(label_names, val_per_label_auroc)):
                    auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
                    logger.info(f"  {label_name}: {auroc_str}")
                logger.info("")
                logger.info("Test Per-Label AUROC:")
                for i, (label_name, auroc) in enumerate(zip(label_names, test_per_label_auroc)):
                    auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
                    logger.info(f"  {label_name}: {auroc_str}")
                logger.info("=" * 80)
                
                if should_save_predictions and rank == 0:
                    # Save both validation and test predictions together
                    if rank == 0:
                        save_predictions(output_dir, "val", epoch, global_iter, val_paths, val_probs, val_targets, label_names, keys=val_keys)
                    save_predictions(output_dir, "test", epoch, global_iter, test_paths, test_probs, test_targets, label_names, keys=test_keys)

                # Check if this is the best validation AUROC
                val_mean_auroc = val_metrics.get("mean_auroc", -float("inf"))
                is_best_auroc = val_mean_auroc > best_val_auroc
                if is_best_auroc:
                    best_val_auroc = val_mean_auroc
                
                # Save val_auroc.csv
                if rank == 0:
                    save_val_auroc(
                        output_dir,
                        epoch,
                        global_iter,
                        val_metrics,
                        test_metrics=test_metrics,
                        label_subsets=label_subsets,
                        is_best=is_best_auroc,
                        label_names=label_names,
                    )

                is_better = metric_value > best_metric if monitor_mode == "max" else metric_value < best_metric
                if is_better:
                    best_metric = metric_value
                    best_epoch = eval_count
                    if rank == 0:
                        state = _state_dict_cpu(model)
                        torch.save(
                            state,
                            os.path.join(
                                checkpoints_dir,
                                f"best_model_{epoch:03d}_{global_iter:06d}.pt",
                            ),
                        )
                        torch.save(
                            state,
                            os.path.join(
                                checkpoints_dir,
                                f"model_{epoch:03d}_{global_iter:06d}.pt",
                            ),
                        )
                    logger.info(f"New best {monitor_name}: {best_metric:.4f} (iter {global_iter})")
                    logger.info(f"Saved model for iteration {global_iter} because it improved")
                else:
                    logger.info(f"No improvement in {monitor_name} for {eval_count - best_epoch}/{patience} evals")
                
                # Early stopping check
                if eval_count - best_epoch >= patience:
                    logger.info(f"Early stopping triggered: {monitor_name} did not improve for {patience} evals")
                    logger.info(f"Best {monitor_name}: {best_metric:.4f} at eval {best_epoch}")
                    break

                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.SequentialLR):
                        # The second scheduler is ReduceLROnPlateau
                        if eval_count > warmup_epochs:
                            scheduler.schedulers[1].step(val_metrics[monitor_name])
                        else:
                            scheduler.step()
                    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics[monitor_name])
                    else:
                        scheduler.step()
                
                if debug:
                    logger.info("Debug: evaluation done, exiting.")
                    return

                # Switch model back to training mode after evaluation
                model.train()

        if eval_count - best_epoch >= patience:
            break

        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from utils import get_best_accelerator
    import torch
    import os
    import torch.distributed as dist
    
    device_type = get_best_accelerator()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if device_type == 'hpu':
            backend = 'hccl'
            import habana_frameworks.torch.core as htcore
            import habana_frameworks.torch.distributed.hccl
        elif device_type == 'cuda':
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    _train_impl(cfg, rank, world_size, local_rank)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
