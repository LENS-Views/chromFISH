"""把各方法的 run_report.json 汇成一张表(用时 / 显存 / 参数 / 步数)。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))
from common import METHOD_DIRS  # noqa: E402

METHODS = ["chromFISH", "HiCARN", "HiCNN", "ScUnicorn", "ScHiCEDRN"]


def main() -> None:
    ap = argparse.ArgumentParser()

    args = ap.parse_args()


    rows = []
    for name, sub_dir in METHOD_DIRS.items():
        for path in sorted((ROOT / sub_dir / "runs").glob("*/run_report.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        print("没找到 run_report.json, 先跑训练")
        return

    rows.sort(
        key=lambda r: (
            METHODS.index(r["method"]) if r["method"] in METHODS else 99,
            r.get("experiment_id", ""),
        )
    )
    header = f"{'实验/方法':<56}{'参数':>8}{'步数':>9}{'用时':>10}{'s/epoch':>9}{'峰值显存':>10}"
    print(header)
    print("-" * len(header) * 2)
    for r in rows:
        mem = f"{r['peak_gpu_mb']} MB" if r["peak_gpu_mb"] else "—"
        mins = r["wall_seconds"] / 60
        display_name = r.get("experiment_id", r["method"])
        print(f"{display_name:<56}{r['params'] / 1e6:>7.2f}M{r['total_steps']:>9,}"
              f"{mins:>9.1f}m{r['sec_per_epoch']:>9.1f}{mem:>10}")
    print(f"\n统一口径: batch {rows[0]['batch_size']} / {rows[0]['epochs']} epoch / "
          f"{rows[0]['steps_per_epoch']} 步每 epoch")

    (HERE / "resource_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
