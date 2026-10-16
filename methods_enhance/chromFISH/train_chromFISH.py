"""我们自己的模型(SwinMatrixTransformer)的统一配置训练。

复用 model/src/matrix_transform 的全部组件: 数据管线、损失、优化器调度、train_one_epoch、
evaluate —— 模型与损失一行不改。只把训练设置对齐到 common.py 的统一口径:
    batch_size 32 与 epochs 100 本来就一致 —— 统一口径正是取自这里, 所以本文件只加日志。

原版 model/scripts/train.py 的 epoch 数写在 config 里; 这里用命令行覆盖, 不改编排。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))  # 共用代码

import common  # noqa: E402

sys.path.insert(0, str(HERE / "model" / "src"))
from matrix_transform.config import load_config  # noqa: E402
from matrix_transform.data import MatrixDataset  # noqa: E402
from matrix_transform.engine import (  # noqa: E402
    evaluate,
    resolve_device,
    seed_everything,
    train_one_epoch,
)
from matrix_transform.losses import MatrixTransformationLoss  # noqa: E402
from matrix_transform.model import SwinMatrixTransformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="我们的模型 统一配置训练")
    ap.add_argument("--config", default=str(HERE / "model" / "configs" / "default.yaml"))
    ap.add_argument("--reads", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=common.EPOCHS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    common.EPOCHS = args.epochs

    config = load_config(args.config)
    config.data.batch_size = args.batch_size
    config.data.source_filename = f"dawn_sampled_{args.reads}.npy"

    seed_everything(config.seed)
    device = resolve_device(args.device or config.device)
    print(f"device={device}  reads={args.reads}  {args.epochs} epoch x "
          f"{common.STEPS_PER_EPOCH} 步 = {args.epochs * common.STEPS_PER_EPOCH:,} 步")

    dataset_args = {
        "source_filename": config.data.source_filename,
        "target_filename": config.data.target_filename,
        "matrix_size": config.data.matrix_size,
    }
    train_dataset, stats = MatrixDataset.training_dataset(config.data.train_dir, **dataset_args)
    valid_dataset = MatrixDataset(config.data.valid_dir, stats=stats, **dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=config.data.eval_batch_size,
                              shuffle=False, num_workers=config.data.num_workers)

    model = SwinMatrixTransformer(config.data.matrix_size, config.model).to(device)
    criterion = MatrixTransformationLoss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate,
                                  weight_decay=config.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.optimizer.warm_restart_period, T_mult=2,
        eta_min=config.optimizer.min_learning_rate)
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    logger = common.RunLogger(args.out_dir, "chromFISH")
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        logger.begin_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model, loader=train_loader, criterion=criterion, optimizer=optimizer,
            device=device, scaler=scaler,
            gradient_clip_norm=config.training.gradient_clip_norm, amp_enabled=amp_enabled)
        scheduler.step()
        print(f"epoch {epoch}/{args.epochs} loss={train_metrics.get('total', float('nan')):.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.3e}")

        if not logger.should_eval():
            continue
        metrics = evaluate(model, valid_loader, stats, device)  # save_dir=None: 只取指标不落盘
        logger.log_epoch(metrics)
        score = metrics["ssim"] + metrics["pcc"]
        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "metrics": metrics, "score": score},
                       Path(args.out_dir) / "best.pt")

    common.print_report(
        logger.write_report(model, args.batch_size, device,
                            extra={"reads": args.reads, "best_score": round(best_score, 4)}))
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()},
               Path(args.out_dir) / "final.pt")


if __name__ == "__main__":
    main()
