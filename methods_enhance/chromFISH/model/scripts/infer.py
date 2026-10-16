from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

# 支持直接执行 `python scripts/infer.py`，无需依赖当前工作目录的PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from matrix_transform.config import load_config
from matrix_transform.data import NormalizationStats
from matrix_transform.engine import resolve_device
from matrix_transform.metrics import compute_metrics, mean_metrics
from matrix_transform.model import SwinMatrixTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Run matrix transformation inference")
    parser.add_argument("--config", required=True, help="与训练时一致的YAML配置")
    parser.add_argument("--checkpoint", required=True, help="best.pt或其他checkpoint")
    parser.add_argument("--stats", required=True, help="训练集normalization.json")
    parser.add_argument("--input", required=True, help="输入[N,H,W]或[H,W]的npy文件")
    parser.add_argument("--output", required=True, help="输出预测npy文件")
    parser.add_argument("--target", default=None, help="可选：目标npy，用于计算指标")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stats = NormalizationStats.load(args.stats)
    stats.validate()
    device = resolve_device(args.device or config.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None or "model" not in checkpoint_config:
        warnings.warn(
            "checkpoint未包含模型配置，无法验证其与当前YAML是否一致",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        checkpoint_model_config = checkpoint_config["model"]
        current_model_config = config.to_dict()["model"]
        differences = {
            key: {
                "checkpoint": checkpoint_model_config.get(key),
                "config": current_model_config.get(key),
            }
            for key in sorted(set(checkpoint_model_config) | set(current_model_config))
            if checkpoint_model_config.get(key) != current_model_config.get(key)
        }
        if differences:
            raise ValueError(
                "checkpoint模型配置与当前YAML不一致: "
                + json.dumps(differences, ensure_ascii=False, sort_keys=True)
            )

    model = SwinMatrixTransformer(config.data.matrix_size, config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source = np.asarray(np.load(args.input), dtype=np.float32)
    single_sample = source.ndim == 2
    if single_sample:
        source = source[None, ...]
    if source.ndim != 3 or source.shape[1] <= 0 or source.shape[2] <= 0:
        raise ValueError(f"输入应为[N, H, W]且H/W大于0，实际为{source.shape}")

    normalized = stats.normalize_source(source)
    tensor = torch.from_numpy(normalized[:, None, ...])
    loader = DataLoader(TensorDataset(tensor), batch_size=args.batch_size, shuffle=False)
    predictions = []
    for (batch,) in loader:
        output = model(batch.to(device)).squeeze(1).cpu().numpy()
        predictions.append(stats.denormalize_target(output))
    prediction = np.concatenate(predictions, axis=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction[0] if single_sample else prediction)
    print(f"saved predictions: {args.output}")

    if args.target:
        target = np.asarray(np.load(args.target), dtype=np.float32)
        if target.ndim == 2:
            target = target[None, ...]
        if target.shape != prediction.shape:
            raise ValueError(f"target形状不一致: {target.shape} vs {prediction.shape}")
        metrics = mean_metrics(
            [
                compute_metrics(prediction[index], target[index], stats.target_range)
                for index in range(prediction.shape[0])
            ]
        )
        metrics_path = Path(args.output).with_suffix(".metrics.json")
        payload = {
            "summary": metrics,
            "provenance": {
                "experiment_id": checkpoint.get("experiment_id"),
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "normalization": str(Path(args.stats).resolve()),
            },
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
