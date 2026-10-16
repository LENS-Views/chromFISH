from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


@dataclass
class DataConfig:
    train_dir: str
    valid_dir: str
    test_dir: str
    source_filename: str = "dawn_sampled_100.npy"
    target_filename: str = "ground_distance_normalized.npy"
    matrix_size: int = 64
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    in_chans: int = 1
    embed_dim: int = 64
    patch_size: int = 1
    patch_norm: bool = True
    depths: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])
    num_heads: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])
    window_size: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    # 原模型消融开关；默认值保持现有模型行为完全不变。
    use_shifted_windows: bool = True
    use_relative_position_bias: bool = True
    use_rstb_residual: bool = True
    use_long_skip: bool = True
    use_reconstruction_head: bool = True


@dataclass
class LossConfig:
    lambda_mse: float = 1.0
    lambda_corr: float = 0.2
    lambda_structure: float = 0.1
    lambda_ssim: float = 0.8


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    min_learning_rate: float = 1e-5
    warm_restart_period: int = 10


@dataclass
class TrainingConfig:
    epochs: int = 100
    validate_every: int = 5
    gradient_clip_norm: float = 1.0
    amp: bool = True
    save_last: bool = True


@dataclass
class AppConfig:
    seed: int = 100
    device: str = "auto"
    output_dir: str = "outputs/run"
    data: DataConfig = field(default_factory=lambda: DataConfig("", "", ""))
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not self.data.train_dir or not self.data.valid_dir or not self.data.test_dir:
            raise ValueError("data.train_dir、data.valid_dir、data.test_dir 必须配置")
        if self.data.matrix_size <= 0:
            raise ValueError("data.matrix_size 必须大于0")
        if self.model.patch_size != 1:
            raise ValueError("当前矩阵级模型要求patch_size=1，以保留每个矩阵单元的位置分辨率")
        if len(self.model.depths) != len(self.model.num_heads):
            raise ValueError("model.depths 与 model.num_heads 长度必须一致")
        if any(self.model.embed_dim % heads != 0 for heads in self.model.num_heads):
            raise ValueError("embed_dim 必须能被每个 num_heads 整除")
        if self.training.epochs <= 0 or self.training.validate_every <= 0:
            raise ValueError("epochs 和 validate_every 必须大于0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, values: Dict[str, Any]):
    return cls(**(values or {}))


def load_config(path: Union[str, Path]) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = AppConfig(
        seed=raw.get("seed", 100),
        device=raw.get("device", "auto"),
        output_dir=raw.get("output_dir", "outputs/run"),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        loss=_merge_dataclass(LossConfig, raw.get("loss", {})),
        optimizer=_merge_dataclass(OptimizerConfig, raw.get("optimizer", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
    )
    config.validate()
    return config
