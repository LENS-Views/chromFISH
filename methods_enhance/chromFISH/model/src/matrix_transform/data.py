from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NormalizationStats:
    source_min: float
    source_max: float
    target_min: float
    target_max: float

    @classmethod
    def from_arrays(cls, source: np.ndarray, target: np.ndarray) -> "NormalizationStats":
        return cls(
            source_min=float(np.min(source)),
            source_max=float(np.max(source)),
            target_min=float(np.min(target)),
            target_max=float(np.max(target)),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormalizationStats":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    def validate(self) -> None:
        if self.source_max <= self.source_min:
            raise ValueError("source归一化范围无效")
        if self.target_max <= self.target_min:
            raise ValueError("target归一化范围无效")

    @staticmethod
    def _normalize(value: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
        return ((value - value_min) / (value_max - value_min) * 2.0 - 1.0).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _denormalize(value: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
        return ((value + 1.0) / 2.0 * (value_max - value_min) + value_min).astype(
            np.float32, copy=False
        )

    def normalize_source(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.source_min, self.source_max)

    def normalize_target(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.target_min, self.target_max)

    def denormalize_target(self, value: np.ndarray) -> np.ndarray:
        return self._denormalize(value, self.target_min, self.target_max)

    @property
    def target_range(self) -> float:
        return self.target_max - self.target_min


class MatrixDataset(Dataset):
    """Loads [N, H, W] numpy matrices and returns [C, H, W] tensors."""

    def __init__(
        self,
        split_dir: Union[str, Path],
        source_filename: str,
        target_filename: str,
        matrix_size: int,
        stats: Optional[NormalizationStats] = None,
    ) -> None:
        split_dir = Path(split_dir)
        source_path = split_dir / source_filename
        target_path = split_dir / target_filename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not target_path.exists():
            raise FileNotFoundError(target_path)

        self.source = np.asarray(np.load(source_path), dtype=np.float32)
        self.target = np.asarray(np.load(target_path), dtype=np.float32)
        expected_shape = (matrix_size, matrix_size)
        if self.source.ndim != 3 or self.target.ndim != 3:
            raise ValueError("source和target必须是[N, H, W]数组")
        if self.source.shape != self.target.shape:
            raise ValueError(f"source与target形状不一致: {self.source.shape} vs {self.target.shape}")
        if self.source.shape[1:] != expected_shape:
            raise ValueError(f"矩阵尺寸应为{expected_shape}，实际为{self.source.shape[1:]}")

        self.stats = stats or NormalizationStats.from_arrays(self.source, self.target)
        self.stats.validate()

    @classmethod
    def training_dataset(cls, split_dir: Union[str, Path], **kwargs) -> Tuple["MatrixDataset", NormalizationStats]:
        dataset = cls(split_dir, stats=None, **kwargs)
        return dataset, dataset.stats

    def __len__(self) -> int:
        return int(self.source.shape[0])

    def __getitem__(self, index: int):
        source_original = self.source[index]
        target_original = self.target[index]
        source = self.stats.normalize_source(source_original)[None, ...]
        target = self.stats.normalize_target(target_original)[None, ...]
        return {
            "source": torch.from_numpy(np.ascontiguousarray(source)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "source_original": source_original,
            "target_original": target_original,
            "index": index,
        }
