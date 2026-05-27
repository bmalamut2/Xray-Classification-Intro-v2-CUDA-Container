import contextlib
import csv
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from arkplus_dataset import ArkPlusCSVDataset, build_arkplus_transform
from arkplus_model import build_arkplus_model, unwrap_model
from evaluate import calculate_multilabel_metrics, calculate_per_label_auroc
from utils import ensure_dir, get_best_accelerator, seed_everything


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_lr(lr: float) -> str:
    return str(lr).replace(".", "_")


try:
    OmegaConf.register_new_resolver("format_lr", format_lr)
except ValueError:
    pass


def get_absolute_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    try:
        from hydra.utils import get_original_cwd

        original_cwd = get_original_cwd()
    except Exception:
        original_cwd = os.environ.get("ORIGINAL_CWD", os.getcwd())
    return os.path.join(original_cwd, path)


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
        self.avg = self.sum / self.count if self.count else 0.0


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def should_log(batch_idx: int, epoch: int, log_every: int) -> bool:
    iteration = batch_idx + 1
    if iteration <= 10:
        return True
    if epoch <= 2:
        return iteration % max(1, log_every // 5) == 0
    return iteration % max(1, log_every) == 0


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = unwrap_model(model).state_dict()
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def reduce_float(value: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        import torch.distributed as dist

        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return float(value.item())


def get_rank_device(device_type: str, local_rank: int) -> torch.device:
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if device_type == "hpu":
        return torch.device("hpu")
    return torch.device("cpu")


def make_autocast(device_type: str):
    use_amp = device_type != "cpu"
    bf16_supported = False
    if device_type == "cuda" and torch.cuda.is_available():
        bf16_supported = torch.cuda.is_bf16_supported()
        if bf16_supported:
            major, _ = torch.cuda.get_device_capability()
            bf16_supported = major >= 8
    elif device_type == "hpu":
        bf16_supported = True

    if use_amp and bf16_supported:
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16), None, "bf16-mixed"
    if use_amp:
        scaler = (
            torch.amp.GradScaler(device_type)
            if hasattr(torch.amp, "GradScaler")
            else (torch.amp.GradScaler() if device_type == "cuda" else None)
        )
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16), scaler, "16-mixed"
    return contextlib.nullcontext(), None, "32-true"


def cosine_schedule(base_value: float, final_value: float, total_steps: int) -> np.ndarray:
    if total_steps <= 1:
        return np.array([final_value], dtype=np.float32)
    steps = np.linspace(0.0, np.pi, total_steps)
    values = final_value + 0.5 * (base_value - final_value) * (1.0 + np.cos(steps))
    return values.astype(np.float32)


def consistency_weight(momentum: float, base_momentum: float, max_weight: float) -> float:
    if base_momentum >= 1.0:
        return float(max_weight)
    ratio = (momentum - base_momentum) / (1.0 - base_momentum)
    return float(np.clip(ratio, 0.0, 1.0) * max_weight)


def ema_update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    student_model = unwrap_model(student)
    with torch.no_grad():
        for student_param, teacher_param in zip(student_model.parameters(), teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.detach().data, alpha=1.0 - momentum)
        for student_buffer, teacher_buffer in zip(student_model.buffers(), teacher.buffers()):
            teacher_buffer.copy_(student_buffer)


def task_to_dict(task_cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(task_cfg, resolve=True)


def build_dataset(
    task_cfg: DictConfig,
    split: str,
    image_size: int,
    resize: int,
    normalize: str,
    validate_paths: bool,
    validate_samples: int,
) -> ArkPlusCSVDataset:
    task = task_to_dict(task_cfg)
    ann_key = f"{split}_ann"
    root_key = f"{split}_image_root"
    if split == "train":
        student_tf = build_arkplus_transform(image_size, resize, normalize, mode="student")
        teacher_tf = build_arkplus_transform(image_size, resize, normalize, mode="teacher")
    else:
        mode = "valid" if split == "val" else "test"
        student_tf = build_arkplus_transform(image_size, resize, normalize, mode=mode)
        teacher_tf = student_tf

    return ArkPlusCSVDataset(
        csv_path=get_absolute_path(task[ann_key]),
        image_key=task.get("image_path_key", "Path"),
        label_names=task["labels"],
        image_root=get_absolute_path(task.get(root_key)),
        image_append=task.get("image_path_append", ""),
        student_transform=student_tf,
        teacher_transform=teacher_tf,
        uncertain_label=task.get("uncertain_label", "Zeros"),
        unknown_label=float(task.get("unknown_label", 0.0)),
        validate_paths=validate_paths,
        validate_samples=validate_samples,
    )


def build_loaders(
    cfg: DictConfig,
    per_rank_batch_size: int,
    rank: int,
    world_size: int,
) -> Tuple[List[Dict], List[Optional[DistributedSampler]]]:
    image_size = int(cfg.ark.preprocessing.get("image_size", cfg.model.get("image_size", 224)))
    resize = int(cfg.ark.preprocessing.get("resize", max(256, image_size)))
    normalize = str(cfg.ark.preprocessing.get("normalize", cfg.model.get("normalize", "imagenet")))
    validate_paths = bool(cfg.ark.get("validate_paths", True))
    validate_samples = int(cfg.ark.get("validate_samples", 5))
    num_workers = int(cfg.ark.get("workers", 8))
    pin_memory = get_best_accelerator() != "cpu"
    drop_last = bool(cfg.ark.get("drop_last", False))

    tasks = []
    samplers = []
    for task_cfg in cfg.ark.datasets:
        train_ds = build_dataset(
            task_cfg, "train", image_size, resize, normalize, validate_paths, validate_samples
        )
        val_ds = build_dataset(
            task_cfg, "val", image_size, resize, normalize, validate_paths, validate_samples
        )
        test_ds = build_dataset(
            task_cfg, "test", image_size, resize, normalize, validate_paths, validate_samples
        )

        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if world_size > 1
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=per_rank_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        eval_batch_size = max(1, int(cfg.ark.get("eval_batch_size", per_rank_batch_size)))
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        task = task_to_dict(task_cfg)
        tasks.append(
            {
                "name": task["name"],
                "labels": task["labels"],
                "task_type": task.get("task_type", "multi-label classification"),
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
            }
        )
        samplers.append(train_sampler)
    return tasks, samplers


def build_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    opt_type = str(cfg.ark.get("optimizer", "sgd")).lower()
    lr = float(cfg.ark.get("lr", 0.3))
    weight_decay = float(cfg.ark.get("weight_decay", 0.0))
    if opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.ark.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_type}")


def build_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
    epochs = int(cfg.ark.get("pretrain_epochs", 50))
    warmup_epochs = int(cfg.ark.get("warmup_epochs", 20))
    min_lr = float(cfg.ark.get("min_lr", 1e-5))
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=min_lr
        )

    lr = float(cfg.ark.get("lr", 0.3))
    warmup_lr = float(cfg.ark.get("warmup_lr", 1e-6))
    start_factor = max(warmup_lr / lr, 1e-8)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    criterion: nn.Module,
) -> torch.Tensor:
    if task_type == "multi-class classification":
        return criterion(logits, targets.argmax(dim=1).long())
    return criterion(logits, targets)


def make_criterion(task_type: str) -> nn.Module:
    if task_type == "multi-class classification":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()


def train_one_task(
    student: nn.Module,
    teacher: nn.Module,
    task: Dict,
    head_idx: int,
    optimizer: optim.Optimizer,
    autocast_ctx,
    scaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    task_step: int,
    momentum_schedule: np.ndarray,
    cfg: DictConfig,
    rank: int,
    world_size: int,
    start_time: float,
) -> Tuple[int, Dict[str, float]]:
    student.train()
    teacher.eval()

    criterion = make_criterion(task["task_type"]).to(device)
    mse = nn.MSELoss()
    momentum = float(momentum_schedule[min(task_step, len(momentum_schedule) - 1)])
    cons_weight = consistency_weight(
        momentum,
        float(cfg.ark.get("teacher_momentum", 0.9)),
        float(cfg.ark.get("consistency_max_weight", 0.5)),
    )

    loss_meter = AverageMeter()
    cls_meter = AverageMeter()
    cons_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    loader = task["train_loader"]
    log_every = int(cfg.ark.get("log_every", 50))
    debug = bool(cfg.get("debug", False))

    for batch_idx, (student_images, teacher_images, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        student_images = student_images.float().to(device, non_blocking=True)
        teacher_images = teacher_images.float().to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            with torch.no_grad():
                teacher_features, _ = teacher(teacher_images, head_idx)
            student_features, logits = student(student_images, head_idx)
            cls_loss = classification_loss(logits, targets, task["task_type"], criterion)
            cons_loss = mse(student_features, teacher_features)
            loss = (1.0 - cons_weight) * cls_loss + cons_weight * cons_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = reduce_float(loss.detach(), world_size)
        cls_val = reduce_float(cls_loss.detach(), world_size)
        cons_val = reduce_float(cons_loss.detach(), world_size)
        batch_size = student_images.size(0)
        loss_meter.update(loss_val, batch_size)
        cls_meter.update(cls_val, batch_size)
        cons_meter.update(cons_val, batch_size)
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and should_log(batch_idx, epoch, log_every):
            elapsed = time.time() - start_time
            logger.info(
                f"Ark+ {task['name']} head={head_idx} E={epoch} "
                f"g_step={global_step} task_step={task_step} "
                f"[B {batch_idx + 1}/{len(loader)}] "
                f"BT={batch_time.val:.2f}({batch_time.avg:.2f}) "
                f"DT={data_time.val:.2f}({data_time.avg:.2f}) "
                f"LR={optimizer.param_groups[0]['lr']:.2e} "
                f"m={momentum:.5f} w_cons={cons_weight:.3f} "
                f"Loss={loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                f"Cls={cls_meter.val:.4f} Cons={cons_meter.val:.4f} "
                f"Elapsed={seconds_to_hms(elapsed)}"
            )

        if debug:
            break

    ema_update_teacher(student, teacher, momentum)
    return global_step, {
        "loss": loss_meter.avg,
        "classification_loss": cls_meter.avg,
        "consistency_loss": cons_meter.avg,
        "teacher_momentum": momentum,
        "consistency_weight": cons_weight,
    }


def evaluate_loss(
    model: nn.Module,
    task: Dict,
    split: str,
    head_idx: int,
    device: torch.device,
) -> float:
    model.eval()
    loader = task[f"{split}_loader"]
    criterion = make_criterion(task["task_type"]).to(device)
    meter = AverageMeter()
    with torch.no_grad():
        for _, images, targets in loader:
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            _, logits = model(images, head_idx)
            loss = classification_loss(logits, targets, task["task_type"], criterion)
            meter.update(float(loss.item()), images.size(0))
    return meter.avg


def evaluate_predictions(
    model: nn.Module,
    task: Dict,
    split: str,
    head_idx: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    loader = task[f"{split}_loader"]
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for _, images, targets in loader:
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            _, logits = model(images, head_idx)
            if task["task_type"] == "multi-class classification":
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(task["labels"])))
    targets_np = (
        np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(task["labels"])))
    )

    if task["task_type"] == "multi-class classification":
        acc = float((probs_np.argmax(axis=1) == targets_np.argmax(axis=1)).mean())
        return {"accuracy": acc}

    mean_auroc = calculate_multilabel_metrics(probs_np, targets_np, device)
    per_label_auroc = calculate_per_label_auroc(probs_np, targets_np, device)
    return {"mean_auroc": mean_auroc, "per_label_auroc": per_label_auroc}


def append_metrics_csv(output_dir: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path = os.path.join(output_dir, "arkplus_metrics.csv")
    fieldnames = [
        "epoch",
        "global_step",
        "dataset",
        "val_loss",
        "test_mean_auroc",
        "test_accuracy",
        "test_per_label_auroc",
        "is_best",
        "date_time",
    ]
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def checkpoint_payload(
    cfg: DictConfig,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    task_step: int,
    best_val_loss: float,
    tasks: List[Dict],
) -> Dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "task_step": task_step,
        "best_val_loss": best_val_loss,
        "student": state_dict_cpu(student),
        "teacher": state_dict_cpu(teacher),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "task_metadata": [
            {
                "name": task["name"],
                "labels": task["labels"],
                "task_type": task["task_type"],
                "head_index": idx,
            }
            for idx, task in enumerate(tasks)
        ],
    }


def save_checkpoint(
    checkpoints_dir: str,
    payload: Dict,
    is_best: bool,
) -> None:
    torch.save(payload, os.path.join(checkpoints_dir, "last_teacher.pth.tar"))
    if is_best:
        torch.save(payload, os.path.join(checkpoints_dir, "best_teacher.pth.tar"))


def load_resume(
    resume_path: str,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, int, int, float]:
    checkpoint = torch.load(resume_path, map_location=device)
    unwrap_model(student).load_state_dict(checkpoint["student"], strict=True)
    teacher.load_state_dict(checkpoint["teacher"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        int(checkpoint.get("global_step", 0)),
        int(checkpoint.get("task_step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def _train_impl(cfg: DictConfig, rank: int, world_size: int, local_rank: int) -> None:
    seed_everything(int(cfg.ark.get("seed", 42)) + rank)

    device_type = get_best_accelerator()
    device = get_rank_device(device_type, local_rank)
    global_batch_size = int(cfg.ark.get("global_batch_size", 200))
    per_rank_batch_size = max(1, global_batch_size // max(1, world_size))
    if rank == 0 and global_batch_size % max(1, world_size) != 0:
        logger.warning(
            f"global_batch_size={global_batch_size} is not divisible by world_size={world_size}; "
            f"using per-rank batch size {per_rank_batch_size}"
        )

    run_name = str(cfg.get("run_name", "arkplus_run"))
    env_run_id = os.environ.get("RUN_ID")
    if env_run_id:
        run_name = env_run_id
        cfg.run_name = run_name

    output_dir = run_name
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(checkpoints_dir)

    if rank != 0:
        logger.setLevel(logging.ERROR)
    log_name = "training.log" if rank == 0 else f"training_rank{rank}.log"
    file_handler = logging.FileHandler(os.path.join(output_dir, log_name), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    tasks, samplers = build_loaders(cfg, per_rank_batch_size, rank, world_size)
    num_classes_list = [len(task["labels"]) for task in tasks]

    backbone = str(cfg.model.get("backbone", "resnet50"))
    pretrained = bool(cfg.model.get("pretrained", True))
    projector_features = cfg.ark.get("projector_features", 1376)
    student = build_arkplus_model(
        backbone=backbone,
        num_classes_list=num_classes_list,
        pretrained=pretrained,
        projector_features=projector_features,
        use_mlp=bool(cfg.ark.get("use_mlp", False)),
    )
    teacher = build_arkplus_model(
        backbone=backbone,
        num_classes_list=num_classes_list,
        pretrained=False,
        projector_features=projector_features,
        use_mlp=bool(cfg.ark.get("use_mlp", False)),
    )
    teacher.load_state_dict(student.state_dict(), strict=True)

    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        if device_type == "cuda":
          student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)
        elif device_type == "hpu":
          student = DDP(
              student,
              bucket_cap_mb=100,
              gradient_as_bucket_view=True,
              find_unused_parameters=True,
          )
        else:
          student = DDP(student, bucket_cap_mb=100, find_unused_parameters=True)

    optimizer = build_optimizer(cfg, student)
    scheduler = build_scheduler(cfg, optimizer)
    autocast_ctx, scaler, precision = make_autocast(device_type)

    start_epoch = 1
    global_step = 0
    task_step = 0
    best_val_loss = float("inf")
    resume_path = cfg.ark.get("resume")
    if resume_path:
        start_epoch, global_step, task_step, best_val_loss = load_resume(
            get_absolute_path(str(resume_path)), student, teacher, optimizer, scheduler, device
        )

    pretrain_epochs = int(cfg.ark.get("pretrain_epochs", 50))
    task_count = len(tasks)
    momentum_schedule = cosine_schedule(
        float(cfg.ark.get("teacher_momentum", 0.9)),
        1.0,
        max(1, pretrain_epochs * task_count),
    )
    debug = bool(cfg.get("debug", False))

    if rank == 0:
        logger.info("=" * 80)
        logger.info("ARK+ PRETRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Run Name: {run_name}")
        logger.info(f"Output Directory: {os.path.abspath(output_dir)}")
        logger.info(f"Backbone: {backbone}")
        logger.info(f"Pretrained: {pretrained}")
        logger.info(f"Projector Features: {projector_features}")
        logger.info(f"Datasets: {[task['name'] for task in tasks]}")
        logger.info(f"Class counts: {num_classes_list}")
        logger.info(f"Global batch size: {global_batch_size}")
        logger.info(f"Per-rank batch size: {per_rank_batch_size}")
        logger.info(f"Epochs: {pretrain_epochs}")
        logger.info(f"Optimizer: {cfg.ark.get('optimizer', 'sgd')}")
        logger.info(f"Learning rate: {cfg.ark.get('lr', 0.3)}")
        logger.info(f"Warmup epochs: {cfg.ark.get('warmup_epochs', 20)}")
        logger.info(f"Teacher momentum base: {cfg.ark.get('teacher_momentum', 0.9)}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Device: {device}")
        for task in tasks:
            logger.info(
                f"Dataset {task['name']}: train={task['train_size']:,} "
                f"val={task['val_size']:,} test={task['test_size']:,} "
                f"labels={len(task['labels'])}"
            )
        logger.info("=" * 80)

    start_time = time.time()
    for epoch in range(start_epoch, pretrain_epochs + 1):
        for head_idx, task in enumerate(tasks):
            sampler = samplers[head_idx]
            if sampler is not None:
                sampler.set_epoch((epoch - 1) * task_count + head_idx)

            global_step, train_metrics = train_one_task(
                student=student,
                teacher=teacher,
                task=task,
                head_idx=head_idx,
                optimizer=optimizer,
                autocast_ctx=autocast_ctx,
                scaler=scaler,
                device=device,
                epoch=epoch,
                global_step=global_step,
                task_step=task_step,
                momentum_schedule=momentum_schedule,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                start_time=start_time,
            )
            if rank == 0:
                logger.info(
                    f"Finished task {task['name']} E={epoch}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"cls={train_metrics['classification_loss']:.4f}, "
                    f"cons={train_metrics['consistency_loss']:.4f}, "
                    f"teacher_m={train_metrics['teacher_momentum']:.5f}"
                )
            task_step += 1

        val_losses = []
        metric_rows = []
        should_test = (
            bool(cfg.ark.get("save_test_metrics", True))
            and (
                epoch % int(cfg.ark.get("test_epoch", 10)) == 0
                or epoch == pretrain_epochs
                or debug
            )
        )
        for head_idx, task in enumerate(tasks):
            val_loss = evaluate_loss(teacher, task, "val", head_idx, device)
            val_losses.append(val_loss)
            test_metrics = evaluate_predictions(teacher, task, "test", head_idx, device) if should_test else {}
            per_label = test_metrics.get("per_label_auroc", "")
            if isinstance(per_label, list):
                per_label = ";".join("nan" if np.isnan(x) else f"{x:.6f}" for x in per_label)
            metric_rows.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "dataset": task["name"],
                    "val_loss": f"{val_loss:.6f}",
                    "test_mean_auroc": (
                        f"{test_metrics['mean_auroc']:.6f}"
                        if "mean_auroc" in test_metrics
                        else ""
                    ),
                    "test_accuracy": (
                        f"{test_metrics['accuracy']:.6f}" if "accuracy" in test_metrics else ""
                    ),
                    "test_per_label_auroc": per_label,
                    "date_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )

        avg_val_loss = float(np.average(val_losses))
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        for row in metric_rows:
            row["is_best"] = is_best

        if rank == 0:
            append_metrics_csv(output_dir, metric_rows)

        scheduler.step()

        if rank == 0:
            payload = checkpoint_payload(
                cfg,
                student,
                teacher,
                optimizer,
                scheduler,
                epoch,
                global_step,
                task_step,
                best_val_loss,
                tasks,
            )
            save_checkpoint(checkpoints_dir, payload, is_best)
            logger.info(
                f"Epoch {epoch}/{pretrain_epochs}: avg_val_loss={avg_val_loss:.6f} "
                f"best_val_loss={best_val_loss:.6f} is_best={is_best}"
            )

        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()

        if debug:
            if rank == 0:
                logger.info("Debug mode finished one Ark+ cycle.")
            break


@hydra.main(version_base=None, config_path="../configs", config_name="config_arkplus")
def main(cfg: DictConfig) -> None:
    import torch.distributed as dist

    device_type = get_best_accelerator()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if device_type == "hpu":
            backend = "hccl"
            import habana_frameworks.torch.distributed.hccl  # noqa: F401
        elif device_type == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"

        timeout_minutes = int(cfg.ark.get("ddp_timeout_minutes", 180))
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(minutes=timeout_minutes))

    try:
        _train_impl(cfg, rank, world_size, local_rank)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
