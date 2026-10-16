"""四个 baseline 的统一推理/评估入口。

指标直接复用我们模型那套 (model/src/matrix_transform/metrics.py), data_range=1.0
—— 与 model/scripts/infer.py 完全一致, 所以结果可以直接和我们模型的 test 指标对比。
刻意不做 clip: 我们的模型 denormalize 后也不裁剪, 这里保持一致。

用法:
    python eval_baselines.py --method HiCNN --checkpoint checkpoints/HiCNN_300/best_model.pth
    python eval_baselines.py --method all --reads 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chromFISH" / "model" / "src"))  # 复用我们模型的指标

from matrix_transform.metrics import compute_metrics, mean_metrics  # noqa: E402

sys.path.insert(0, str(HERE))
from common import METHOD_DIRS  # noqa: E402

# 每个方法: checkpoint 路径模板 + 模型构造函数
CHECKPOINTS = {
    "HiCNN": "HiCNN_software_package/checkpoints/HiCNN_{reads}/best_model.pth",
    "HiCARN": "HiCARN/checkpoints/HiCARN_{reads}/fish_best.pytorch",
    "ScUnicorn": "Unicorn/sc_checkpoint/scunicorn_model_{reads}_64.pth",
    "ScHiCEDRN": "ScHiCEDRN/checkpoints/ScHiCEDRN_{reads}/bestg_40kb_c40_s40_Human1_hiedsr.pytorch",
}


def build_model(method: str) -> torch.nn.Module:
    """按方法构造网络。sys.path 用绝对路径, 所以不依赖 cwd。"""
    if method == "HiCNN":
        sys.path.insert(0, str(HERE.parent / "HiCNN_software_package" / "legacy"))
        from HiCNN_training_f import Net
        return Net()
    if method == "HiCARN":
        sys.path.insert(0, str(HERE.parent / "HiCARN"))
        from Models.HiCARN_1 import Generator
        return Generator(num_channels=64)
    if method == "ScUnicorn":
        sys.path.insert(0, str(HERE.parent / "Unicorn" / "ScUnicorn" / "scripts" / "training"))
        import train as scunicorn_train
        return scunicorn_train.ScUnicorn(num_channels=64)
    if method == "ScHiCEDRN":
        sys.path.insert(0, str(HERE.parent / "ScHiCEDRN"))
        from Models.schicedrn_gan import Generator
        return Generator()
    raise ValueError(f"未知方法: {method}")


def load_state_dict(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "netG"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 不是 state_dict: {path}")
    return payload


@torch.no_grad()
def predict(model, source: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, source.shape[0], batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(source[start:start + batch_size]))
        outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline 统一推理与评估")
    parser.add_argument("--method", required=True, choices=sorted(CHECKPOINTS) + ["all"])
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--checkpoint", default=None, help="不给则按方法+读深用默认路径")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    split_path = HERE.parent / "data" / args.split / f"{args.split}_{args.reads}_64.npz"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with np.load(split_path) as payload:
        source, target = payload["data"], payload["target"]
    print(f"device={device}  {args.split}: {source.shape}")

    methods = sorted(CHECKPOINTS) if args.method == "all" else [args.method]
    results = {}
    for method in methods:
        ckpt_fallback = HERE.parent / CHECKPOINTS[method].format(reads=args.reads)
        ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_fallback
        if not ckpt.exists():
            print(f"[跳过] {method}: checkpoint 不存在 {ckpt}")
            continue

        model = build_model(method).to(device)
        model.load_state_dict(load_state_dict(ckpt))
        pred = predict(model, source, args.batch_size, device)
        if pred.shape != source.shape:
            raise ValueError(f"{method} 输出形状 {pred.shape} != 输入 {source.shape}")

        pred_2d = pred[:, 0]
        target_2d = target[:, 0]
        per_sample = [
            compute_metrics(pred_2d[i], target_2d[i], 1.0) for i in range(pred_2d.shape[0])
        ]
        summary = mean_metrics(per_sample)

        out_dir = HERE.parent / METHOD_DIRS[method] / "eval" / f"{method}_{args.reads}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{args.split}_predictions.npy", pred_2d)
        np.save(out_dir / f"{args.split}_targets.npy", target_2d)
        (out_dir / "metrics.json").write_text(
            json.dumps({"summary": summary, "per_sample": per_sample, "checkpoint": str(ckpt)},
                       indent=2), encoding="utf-8")
        results[method] = summary
        print(f"  {method:11s} pcc={summary['pcc']:.4f} ssim={summary['ssim']:.4f} "
              f"rmse={summary['rmse']:.4f} psnr={summary['psnr']:.2f}  -> {out_dir}")

    if results:
        print("\n=== 汇总 (reads={}) {}".format(args.reads, args.split))
        print(f"{'method':12s} {'pcc':>8s} {'spearman':>9s} {'rmse':>8s} {'mae':>8s} {'ssim':>8s} {'psnr':>8s}")
        for method, s in results.items():
            print(f"{method:12s} {s['pcc']:8.4f} {s['spearman']:9.4f} {s['rmse']:8.4f} "
                  f"{s['mae']:8.4f} {s['ssim']:8.4f} {s['psnr']:8.2f}")

if __name__ == "__main__":
    main()
