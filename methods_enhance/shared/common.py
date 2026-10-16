"""统一训练口径: 五个方法用同一份超参, 并统一记录用时/显存。

各方法仍是各自的训练循环(方法特有的 loss/优化器/lr schedule 一律不动),
只把"跑多少"和"怎么记账"抽到这里。

统一口径的由来:
    train 集 4091 条, batch_size=32, drop_last=False -> 每个 epoch 都是 ceil(4091/32)=128 步
    (最后一批 27 条不满, 照常参与)。所以只要五个方法都用这个配置, 总步数天然相等,
    不需要中途截断。128 x 100 = 12,800 步 —— 与我们原版 run 的步数完全相同。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"

# 方法名 -> 该方法在 methods_enhance/ 下的目录名。各脚本据此找模型和读写产出。
METHOD_DIRS = {
    "chromFISH": "chromFISH",
    "HiCARN": "HiCARN",
    "HiCNN": "HiCNN_software_package",
    "ScUnicorn": "Unicorn",
    "ScHiCEDRN": "ScHiCEDRN",
}

# 统一配置 —— 五个方法共用, 改这里就都改了
BATCH_SIZE = 32
EPOCHS = 100
EVAL_EVERY = 5  # 每 5 epoch 验证一次 -> 20 个点, 和我们原版 run 的粒度一致
TRAIN_SIZE = 4091
STEPS_PER_EPOCH = -(-TRAIN_SIZE // BATCH_SIZE)  # 向上取整 = 128
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS          # 12,800

sys.path.insert(0, str(HERE.parent / "chromFISH" / "model" / "src"))  # 复用我们模型的指标
from matrix_transform.metrics import compute_metrics, mean_metrics  # noqa: E402


def npz_loader(split: str, reads: int, batch_size: int, shuffle: bool,
               drop_last: bool = False) -> DataLoader:
    """五个方法共用的数据加载。npz 里 data/target 已是 (N,1,64,64) float32。"""
    path = DATA_ROOT / split / f"{split}_{reads}_64.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        data = torch.from_numpy(payload["data"]).float()
        target = torch.from_numpy(payload["target"]).float()
    return DataLoader(TensorDataset(data, target), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def predict_and_target(model, loader, device, keys=None):
    """跑一遍 loader 拿 (预测, 目标)。baseline 的 loader 出 (data, target) 元组;
    我们模型的出 dict, 传 keys=('source','target')。"""
    was_training = model.training
    model.eval()
    preds, targets = [], []
    for batch in loader:
        if keys is not None:
            x, y = batch[keys[0]], batch[keys[1]]
        else:
            x, y = batch[0], batch[1]
        preds.append(model(x.to(device)).float().cpu())
        targets.append(y.float().cpu())
    if was_training:
        model.train()
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def mean_metrics_of(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> dict:
    """逐样本算指标再平均 —— 与 model/scripts/infer.py 完全一致。"""
    p = pred.numpy()[:, 0]
    t = target.numpy()[:, 0]
    if p.shape != t.shape:
        raise ValueError(f"预测/目标形状不一致: {p.shape} vs {t.shape}")
    return mean_metrics([compute_metrics(p[i], t[i], data_range) for i in range(p.shape[0])])


class RunLogger:
    """记录每个 epoch 的耗时与验证指标, 结束时写资源账单。"""

    def __init__(self, out_dir, method: str, eval_every: int = EVAL_EVERY) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.eval_every = eval_every
        self.epoch = 0
        self.history: list = []
        self._t0 = time.time()
        self._log_path = self.out_dir / "train_log.jsonl"
        self._log_path.unlink(missing_ok=True)

    def elapsed(self) -> float:
        return time.time() - self._t0

    def begin_epoch(self, epoch: int) -> None:
        """每个 epoch 都打一行, 免得远程跑的时候长时间看不到进度。"""
        self.epoch = epoch
        print(f"[epoch {epoch:>4d}/{EPOCHS}] start  ({self.elapsed():.0f}s elapsed)", flush=True)

    def should_eval(self) -> bool:
        return self.epoch % self.eval_every == 0 or self.epoch == EPOCHS

    def log_epoch(self, metrics: dict) -> None:
        record = {"epoch": self.epoch, "step": self.epoch * STEPS_PER_EPOCH,
                  "seconds": round(self.elapsed(), 1),
                  **{k: float(v) for k, v in metrics.items()}}
        self.history.append(record)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  [epoch {self.epoch:>4d}/{EPOCHS}] pcc={record['pcc']:.4f} "
              f"ssim={record['ssim']:.4f} rmse={record['rmse']:.4f}  {record['seconds']:.0f}s")

    def write_report(self, model, batch_size: int, device: torch.device,
                     extra: dict | None = None) -> dict:
        seconds = self.elapsed()
        report = {
            "method": self.method,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_steps": EPOCHS * STEPS_PER_EPOCH,
            "batch_size": batch_size,
            "params": sum(p.numel() for p in model.parameters()),
            "wall_seconds": round(seconds, 1),
            "sec_per_epoch": round(seconds / max(self.epoch, 1), 2),
            "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if device.type == "cuda" else None),
            "device": str(device),
            "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            **(extra or {}),
        }
        (self.out_dir / "run_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report


def print_report(report: dict) -> None:
    mem = f"{report['peak_gpu_mb']} MB" if report["peak_gpu_mb"] else "CPU/不可用"
    print(f"\n=== {report['method']} ===")
    print(f"  {report['epochs']} epoch x {report['steps_per_epoch']} 步 = "
          f"{report['total_steps']:,} 步  batch {report['batch_size']}")
    print(f"  参数 {report['params'] / 1e6:.2f}M  用时 {report['wall_seconds']:.0f}s "
          f"({report['sec_per_epoch']:.1f} s/epoch)  峰值显存 {mem}")
