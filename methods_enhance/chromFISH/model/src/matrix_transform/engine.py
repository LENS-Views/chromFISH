from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.amp import autocast 
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import NormalizationStats
from .metrics import compute_metrics, mean_metrics
from .losses import MatrixTransformationLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用CUDA，但当前环境没有可用CUDA")
    return device


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MatrixTransformationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    gradient_clip_norm: float,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    totals = []
    for batch in tqdm(loader, desc="training", leave=False):
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=amp_enabled):
            prediction = model(source)
            loss, parts = criterion(prediction, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        totals.append(parts)
    return mean_metrics(totals)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: NormalizationStats,
    device: torch.device,
    save_dir: Optional[Union[str, Path]] = None,
    prefix: str = "eval",
) -> Dict[str, float]:
    model.eval()
    all_metrics = []
    all_predictions = []
    all_targets = []
    all_sources = []
    all_indices = []

    for batch in tqdm(loader, desc=f"evaluating:{prefix}", leave=False):
        source = batch["source"].to(device, non_blocking=True)
        prediction = model(source).squeeze(1).cpu().numpy()
        target = np.asarray(batch["target_original"], dtype=np.float32)
        source_original = np.asarray(batch["source_original"], dtype=np.float32)
        prediction = stats.denormalize_target(prediction)

        for index in range(prediction.shape[0]):
            all_metrics.append(
                compute_metrics(prediction[index], target[index], stats.target_range)
            )
        all_predictions.append(prediction)
        all_targets.append(target)
        all_sources.append(source_original)
        all_indices.extend(np.asarray(batch["index"]).reshape(-1).tolist())

    summary = mean_metrics(all_metrics)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        sources = np.concatenate(all_sources, axis=0)
        np.save(save_dir / f"{prefix}_predictions.npy", predictions)
        np.save(save_dir / f"{prefix}_targets.npy", targets)
        np.save(save_dir / f"{prefix}_sources.npy", sources)
        np.save(save_dir / f"{prefix}_indices.npy", np.asarray(all_indices))
        with (save_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "per_sample": all_metrics}, handle, indent=2)
    return summary


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict,
    metrics: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": metrics or {},
        },
        path,
    )
