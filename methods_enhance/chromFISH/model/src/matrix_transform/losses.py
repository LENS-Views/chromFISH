from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 5, value_range: float = 2.0) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("SSIM window_size必须为奇数")
        self.window_size = window_size
        self.value_range = value_range
        self.channels = 1
        self.register_buffer("window", self._create_window(1), persistent=False)

    @staticmethod
    def _gaussian(size: int, sigma: float = 1.5) -> torch.Tensor:
        values = [math.exp(-((index - size // 2) ** 2) / (2 * sigma**2)) for index in range(size)]
        result = torch.tensor(values, dtype=torch.float32)
        return result / result.sum()

    def _create_window(self, channels: int) -> torch.Tensor:
        vector = self._gaussian(self.window_size).unsqueeze(1)
        window = vector @ vector.t()
        return window.unsqueeze(0).unsqueeze(0).expand(
            channels, 1, self.window_size, self.window_size
        ).contiguous()

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        if image_a.shape != image_b.shape:
            raise ValueError("SSIM的两个输入形状必须一致")
        channels = image_a.shape[1]
        if (
            channels != self.channels
            or self.window.device != image_a.device
            or self.window.dtype != image_a.dtype
        ):
            self.window = self._create_window(channels).to(
                device=image_a.device, dtype=image_a.dtype
            )
            self.channels = channels

        padding = self.window_size // 2
        mean_a = F.conv2d(image_a, self.window, padding=padding, groups=channels)
        mean_b = F.conv2d(image_b, self.window, padding=padding, groups=channels)
        mean_a_sq = mean_a.square()
        mean_b_sq = mean_b.square()
        mean_ab = mean_a * mean_b
        variance_a = F.conv2d(image_a.square(), self.window, padding=padding, groups=channels) - mean_a_sq
        variance_b = F.conv2d(image_b.square(), self.window, padding=padding, groups=channels) - mean_b_sq
        covariance = F.conv2d(image_a * image_b, self.window, padding=padding, groups=channels) - mean_ab
        variance_a = variance_a.clamp_min(0.0)
        variance_b = variance_b.clamp_min(0.0)

        c1 = (0.01 * self.value_range) ** 2
        c2 = (0.03 * self.value_range) ** 2
        score = ((2 * mean_ab + c1) * (2 * covariance + c2)) / (
            (mean_a_sq + mean_b_sq + c1) * (variance_a + variance_b + c2)
        )
        return 1.0 - score.mean()


class MatrixTransformationLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(window_size=5, value_range=2.0)

    @staticmethod
    def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.flatten(1)
        truth = target.flatten(1)
        pred = pred - pred.mean(dim=1, keepdim=True)
        truth = truth - truth.mean(dim=1, keepdim=True)
        covariance = (pred * truth).sum(dim=1)
        denominator = torch.sqrt(pred.square().sum(dim=1) * truth.square().sum(dim=1))
        valid = denominator > 1e-8
        correlation = torch.zeros_like(denominator)
        correlation[valid] = covariance[valid] / denominator[valid]
        return 1.0 - correlation.clamp(-1.0, 1.0).mean()

    @staticmethod
    def structure_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x = prediction[..., 1:] - prediction[..., :-1]
        pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_x = target[..., 1:] - target[..., :-1]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        loss_mse = self.mse(prediction, target)
        loss_corr = self.correlation_loss(prediction, target)
        loss_structure = self.structure_loss(prediction, target)
        loss_ssim = self.ssim(prediction, target)
        total = (
            self.config.lambda_mse * loss_mse
            + self.config.lambda_corr * loss_corr
            + self.config.lambda_structure * loss_structure
            + self.config.lambda_ssim * loss_ssim
        )
        parts = {
            "total": float(total.detach().item()),
            "mse": float(loss_mse.detach().item()),
            "corr": float(loss_corr.detach().item()),
            "structure": float(loss_structure.detach().item()),
            "ssim": float(loss_ssim.detach().item()),
        }
        return total, parts
