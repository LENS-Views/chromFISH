from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# 支持直接执行 `python scripts/train.py`，无需依赖当前工作目录的PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from matrix_transform.config import load_config
from matrix_transform.data import MatrixDataset
from matrix_transform.engine import (
    evaluate,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from matrix_transform.losses import MatrixTransformationLoss
from matrix_transform.model import SwinMatrixTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train matrix transformation model")
    parser.add_argument("--config", required=True, help="YAML配置文件")
    parser.add_argument("--resume", default=None, help="可选：从checkpoint继续训练")
    parser.add_argument("--output-dir", default=None, help="覆盖配置中的输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device: {device}")

    common_dataset_args = {
        "source_filename": config.data.source_filename,
        "target_filename": config.data.target_filename,
        "matrix_size": config.data.matrix_size,
    }
    train_dataset, stats = MatrixDataset.training_dataset(
        config.data.train_dir, **common_dataset_args
    )
    valid_dataset = MatrixDataset(
        config.data.valid_dir, stats=stats, **common_dataset_args
    )
    test_dataset = MatrixDataset(
        config.data.test_dir, stats=stats, **common_dataset_args
    )
    stats.save(output_dir / "normalization.json")

    loader_kwargs = {
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.data.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = SwinMatrixTransformer(config.data.matrix_size, config.model).to(device)
    criterion = MatrixTransformationLoss(config.loss)
    if config.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    elif config.optimizer.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optimizer.learning_rate
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer.name}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.optimizer.warm_restart_period,
        T_mult=2,
        eta_min=config.optimizer.min_learning_rate,
    )
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_score = float("-inf")
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        previous_metrics = checkpoint.get("metrics", {})
        best_score = float(previous_metrics.get("score", best_score))
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config.training.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_clip_norm=config.training.gradient_clip_norm,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        print(
            f"epoch {epoch + 1}/{config.training.epochs} "
            f"loss={train_metrics.get('total', float('nan')):.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        should_validate = (
            (epoch + 1) % config.training.validate_every == 0
            or epoch == config.training.epochs - 1
        )
        if not should_validate:
            continue

        valid_metrics = evaluate(
            model,
            valid_loader,
            stats,
            device,
            save_dir=output_dir / "validation" / f"epoch_{epoch + 1:04d}",
            prefix="valid",
        )
        score = valid_metrics["ssim"] + valid_metrics["pcc"]
        print(
            f"validation: score={score:.6f} "
            f"ssim={valid_metrics['ssim']:.6f} pcc={valid_metrics['pcc']:.6f}"
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                config.to_dict(),
                {**valid_metrics, "score": score},
            )
            print(f"saved best checkpoint: {best_checkpoint}")

        if config.training.save_last:
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                config.to_dict(),
                {**valid_metrics, "score": score},
            )

    if not best_checkpoint.exists():
        raise RuntimeError("训练结束但没有生成best checkpoint")
    best = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        stats,
        device,
        save_dir=output_dir / "test_best",
        prefix="test",
    )
    (output_dir / "test_best" / "summary.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print("best-model test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
