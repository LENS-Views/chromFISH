"""把 data/mask_2018_science 的 [N,64,64] npy 打包成各 baseline 要的 npz。

产出 {out-root}/{split}/{split}_{depth}_64.npz (HiCNN 也直接读这个):
    data    (N,1,64,64) float32   低读深输入
    target  (N,1,64,64) float32   稠密目标
    inds/compacts/sizes           凑 HiCARN schema; 我们是独立样本故为恒等映射

数据本身已在 [0,1], 不再 clamp/缩放 —— HiCARN 的 Generate.py 那套缩放是对 raw count 做的。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")


def pack(source: np.ndarray, target: np.ndarray, side: int) -> dict:
    """[N,H,W] -> [N,1,H,W] float32, 并造 HiCARN 的索引字段。"""
    if source.shape != target.shape:
        raise ValueError(f"source/target 形状不一致: {source.shape} vs {target.shape}")
    if source.ndim != 3 or source.shape[1:] != (side, side):
        raise ValueError(f"矩阵尺寸应为 ({side},{side}), 实际为 {source.shape[1:]}")

    n = source.shape[0]
    # 每个样本是独立矩阵、不来自染色体切片, 所以还原是恒等: 单条 "chr0", 偏移 0。
    inds = np.tile(np.array([[0, side, 0, 0]], dtype=np.int64), (n, 1))
    return {
        "data": source[:, None, ...].astype(np.float32),
        "target": target[:, None, ...].astype(np.float32),
        "inds": inds,
        "compacts": np.array({0: np.arange(side)}, dtype=object),
        "sizes": np.array({0: side}, dtype=object),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="打包 baseline 训练数据")
    ap.add_argument("--data-root", required=True, help="原始 npy 所在目录, 如 ../data/mask_2018_science")
    ap.add_argument("--out-root", required=True, help="输出根目录, 如 data")
    ap.add_argument("--depths", type=int, nargs="+", default=[50, 100, 300],
                    help="输入读深, 对应 dawn_sampled_{N}.npy")
    ap.add_argument("--target-file", default="ground_distance_normalized.npy")
    ap.add_argument("--side", type=int, default=64)
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    for depth in args.depths:
        for split in SPLITS:
            src_path = data_root / split / f"dawn_sampled_{depth}.npy"
            tgt_path = data_root / split / args.target_file
            if not src_path.exists() or not tgt_path.exists():
                raise FileNotFoundError(f"{src_path} 或 {tgt_path} 不存在")

            payload = pack(np.load(src_path), np.load(tgt_path), args.side)
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = f"{split}_{depth}_{args.side}"
            np.savez_compressed(out_dir / f"{stem}.npz", **payload)
            d, t = payload["data"], payload["target"]
            print(f"{stem}: data {d.shape} [{d.min():.3f},{d.max():.3f}] "
                  f"非零率 {np.count_nonzero(d) / d.size:.3f} | "
                  f"target [{t.min():.3f},{t.max():.3f}]")


if __name__ == "__main__":
    main()
