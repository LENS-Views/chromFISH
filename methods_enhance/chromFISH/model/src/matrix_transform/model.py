from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchEmbed(nn.Module):
    """二维特征图到token序列的转换，patch_size=1时不丢失矩阵单元分辨率。"""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("特征图尺寸必须能被patch_size整除")
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens) if self.norm is not None else tokens


class PatchUnEmbed(nn.Module):
    """token序列恢复为二维特征图。"""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = tokens.shape
        if length != height * width or channels != self.embed_dim:
            raise ValueError("token数量或通道数与目标特征图尺寸不一致")
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError("特征图尺寸必须能被window_size整除")
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, channels
    )


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    windows_per_image = (height // window_size) * (width // window_size)
    batch = windows.shape[0] // windows_per_image
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        dropout: float,
        use_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim必须能被num_heads整除")
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_relative_position_bias = use_relative_position_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        table_size = (2 * window_size - 1) ** 2
        if use_relative_position_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(table_size, num_heads)
            )
        else:
            self.register_parameter("relative_position_bias_table", None)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        if self.relative_position_bias_table is not None:
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale

        if self.relative_position_bias_table is not None:
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(tokens, tokens, -1)
            attention = attention + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            windows_per_image = mask.shape[0]
            attention = attention.view(
                batch_windows // windows_per_image,
                windows_per_image,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)

        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ value).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.proj_drop(self.proj(output))


class SwinBlock(nn.Module):
    """支持动态空间尺寸的普通/移位窗口Transformer块。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        use_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim,
            window_size,
            num_heads,
            dropout,
            use_relative_position_bias=use_relative_position_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def _make_mask(
        self, height: int, width: int, reference: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        mask = reference.new_zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        counter = 0
        for h_slice in h_slices:
            for w_slice in h_slices:
                mask[:, h_slice, w_slice, :] = counter
                counter += 1
        mask_windows = window_partition(mask, self.window_size).view(
            -1, self.window_size * self.window_size
        )
        difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0).masked_fill(
            difference == 0, 0.0
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("输入token数量与当前空间尺寸不一致")
        if height % self.window_size or width % self.window_size:
            raise ValueError("输入空间尺寸必须能被window_size整除")

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, channels
        )
        mask = self._make_mask(height, width, windows)
        windows = self.attention(windows, mask)
        x = window_reverse(
            windows.view(-1, self.window_size, self.window_size, channels),
            self.window_size,
            height,
            width,
        )
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_paths: Sequence[float],
        use_shifted_windows: bool = True,
        use_relative_position_bias: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0
                        if not use_shifted_windows or index % 2 == 0
                        else window_size // 2
                    ),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_paths[index],
                    use_relative_position_bias=use_relative_position_bias,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        shortcut = x
        for block in self.blocks:
            x = block(x, x_size)
        x = self.norm(x)
        batch, _, channels = x.shape
        height, width = x_size
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        shortcut = shortcut.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.conv(x)
        if self.use_residual:
            x = x + shortcut
        return x.flatten(2).transpose(1, 2)


class SwinMatrixTransformer(nn.Module):
    """支持自动padding/crop的Swin矩阵到矩阵转换模型。"""

    def __init__(self, matrix_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.matrix_size = matrix_size
        self.in_chans = config.in_chans
        self.window_size = config.window_size
        self.use_long_skip = config.use_long_skip
        dim = config.embed_dim
        total_depth = sum(config.depths)
        drop_paths = torch.linspace(0, config.drop_path_rate, total_depth).tolist()

        self.shallow = nn.Sequential(
            nn.Conv2d(config.in_chans, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.projection = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            embed_dim=dim,
            norm_layer=nn.LayerNorm if config.patch_norm else None,
        )
        self.patch_unembed = PatchUnEmbed(dim)

        self.blocks = nn.ModuleList()
        offset = 0
        for depth, heads in zip(config.depths, config.num_heads):
            self.blocks.append(
                RSTB(
                    dim=dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_paths=drop_paths[offset : offset + depth],
                    use_shifted_windows=config.use_shifted_windows,
                    use_relative_position_bias=config.use_relative_position_bias,
                    use_residual=config.use_rstb_residual,
                )
            )
            offset += depth

        # 所有RSTB完成后的最终归一化
        self.final_norm = nn.LayerNorm(dim)
        self.body = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.reconstruction = (
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            if config.use_reconstruction_head
            else nn.Identity()
        )
        self.output = nn.Conv2d(dim, config.in_chans, kernel_size=3, padding=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pad_to_window(self, x: torch.Tensor):
        height, width = x.shape[-2:]
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        if pad_height or pad_width:
            # 对非图像矩阵使用边缘复制，避免reflect制造镜像结构。
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(
                f"输入应为[B, {self.in_chans}, H, W]，实际为{tuple(x.shape)}"
            )
        x, original_size = self._pad_to_window(x)
        padded_size = (x.shape[-2], x.shape[-1])

        shallow = self.shallow(x)
        features = self.patch_embed(self.projection(shallow))
        for block in self.blocks:
            features = block(features, padded_size)
        features = self.final_norm(features)
        deep = self.patch_unembed(features, padded_size)

        features = self.body(deep)
        if self.use_long_skip:
            features = features + shallow
        features = self.reconstruction(features)
        output = torch.tanh(self.output(features))
        height, width = original_size
        return output[..., :height, :width]

