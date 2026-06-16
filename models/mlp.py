from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from utils import webster_batch_to_xy


class ProfileMLP(nn.Module):
    """
    Baseline MLP for Webster transfer-function regression.

    Input:
        profile features [B, Nx, C], usually [log(area), x].

    Output:
        transfer function [B, Nf] when out_channels=1,
        otherwise [B, Nf, out_channels].
    """

    def __init__(
        self,
        n_profile_points: int,
        in_channels: int,
        n_frequencies: int,
        *,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.05,
        out_channels: int = 1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_profile_points < 2:
            raise ValueError("n_profile_points must be >= 2")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if n_frequencies < 1:
            raise ValueError("n_frequencies must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        self.n_profile_points = n_profile_points
        self.in_channels = in_channels
        self.n_frequencies = n_frequencies
        self.out_channels = out_channels
        self.model_name = "profile_mlp"

        output_dim = n_frequencies * out_channels
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(n_profile_points * in_channels, hidden_dim),
            nn.GELU(),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, profile: Tensor) -> Tensor:
        if profile.ndim != 3:
            raise ValueError(
                f"profile must have shape [B, Nx, C], got {tuple(profile.shape)}"
            )
        if profile.shape[1] != self.n_profile_points:
            raise ValueError(
                f"expected {self.n_profile_points} profile points, got {profile.shape[1]}"
            )
        if profile.shape[2] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {profile.shape[2]}"
            )

        out = self.net(profile)
        if self.out_channels == 1:
            return out
        return out.view(profile.shape[0], self.n_frequencies, self.out_channels)


def webster_mlp_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    n_points: int = 128,
    log_area: bool = True,
    include_x: bool = True,
    target_key: str = "target",
) -> tuple[Tensor, Tensor]:
    """
    Adapter for ProfileMLP.

    Returns profile features [B, Nx, C] and target from WebsterTorchDataset.
    """

    return webster_batch_to_xy(
        batch,
        device,
        n_points=n_points,
        log_area=log_area,
        include_x=include_x,
        channel_first=False,
        target_key=target_key,
    )
