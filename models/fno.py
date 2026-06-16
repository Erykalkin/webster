from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from neuralop.models import FNO

from utils import make_webster_profile_features


class FrequencyEmbedding(nn.Module):
    """
    Преобразует одну нормализованную частоту в набор гармонических признаков.

    Input:
        kappa: [B, Nf, 1]

    Output:
        features: [B, Nf, 1 + 2 * n_bands]
    """

    def __init__(self, n_bands: int = 8) -> None:
        super().__init__()

        bands = 2.0 ** torch.arange(n_bands, dtype=torch.float32)
        self.register_buffer("bands", math.pi * bands)

    def forward(self, kappa: Tensor) -> Tensor:
        if kappa.ndim != 3 or kappa.shape[-1] != 1:
            raise ValueError(
                f"kappa must have shape [B, Nf, 1], got {tuple(kappa.shape)}"
            )

        angles = kappa * self.bands.view(1, 1, -1)

        return torch.cat(
            [
                kappa,
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )


class TransferFunctionFNO(nn.Module):
    """
    Аппроксимирует оператор:

        S(x) -> H(f)

    Output:
        [B, Nf] when out_channels=1
        [B, Nf, out_channels] otherwise
    """

    def __init__(
        self,
        n_modes: int = 24,
        hidden_channels: int = 64,
        latent_dim: int = 128,
        pooling_bins: int = 8,
        frequency_bands: int = 8,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        self.out_channels = out_channels

        self.encoder = FNO(
            n_modes=(n_modes,),
            in_channels=1,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=4,
            positional_embedding="grid",
        )

        # Сохраняем грубую пространственную структуру профиля.
        # Это информативнее, чем обычный mean pooling.
        self.geometry_pool = nn.AdaptiveAvgPool1d(pooling_bins)

        self.geometry_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooling_bins, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
            nn.GELU(),
        )

        self.frequency_embedding = FrequencyEmbedding(
            n_bands=frequency_bands
        )

        frequency_input_dim = 1 + 2 * frequency_bands

        self.frequency_head = nn.Sequential(
            nn.Linear(frequency_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        decoder_input_dim = latent_dim + 64

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, out_channels),
        )

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        """
        area:
            Нормализованный профиль площади [B, 1, Nx].

        kappa:
            Нормализованная безразмерная частота [B, Nf, 1].

        returns:
            Передаточная функция [B, Nf] или [B, Nf, out_channels].
        """

        if area.ndim != 3:
            raise ValueError(
                f"area must have shape [B, 1, Nx], got {tuple(area.shape)}"
            )

        if area.shape[1] != 1:
            raise ValueError(
                f"area must have one input channel, got {area.shape[1]}"
            )

        if area.shape[0] != kappa.shape[0]:
            raise ValueError("area and kappa batch sizes must match")

        # [B, 64, Nx]
        spatial_features = self.encoder(area)

        # [B, 64, pooling_bins]
        pooled_features = self.geometry_pool(spatial_features)

        # [B, latent_dim]
        geometry_latent = self.geometry_head(pooled_features)

        # [B, Nf, 64]
        frequency_features = self.frequency_head(
            self.frequency_embedding(kappa)
        )

        n_frequencies = kappa.shape[1]

        # [B, Nf, latent_dim]
        geometry_latent = geometry_latent.unsqueeze(1).expand(
            -1,
            n_frequencies,
            -1,
        )

        # [B, Nf, latent_dim + 64]
        decoder_input = torch.cat(
            [geometry_latent, frequency_features],
            dim=-1,
        )

        out = self.decoder(decoder_input)
        if self.out_channels == 1:
            return out.squeeze(-1)
        return out


def webster_fno_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    n_points: int = 128,
    log_area: bool = True,
    target_key: str = "target",
    frequency_key: str = "frequencies_hz",
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    """
    Адаптер батча WebsterTorchDataset под TransferFunctionFNO.

    Возвращает:
        inputs = (area, kappa)
            area: [B, 1, Nx]
            kappa: [B, Nf, 1], частота нормирована в [0, 1]
        target:
            [B, Nf] для dB/magnitude/phase или [B, Nf, 2] для real/imag.
    """

    area = make_webster_profile_features(
        batch,
        n_points=n_points,
        log_area=log_area,
        include_x=False,
        channel_first=True,
        device=device,
    )

    frequencies = batch[frequency_key].to(device).float()
    f_min = frequencies.amin(dim=1, keepdim=True)
    f_span = (frequencies.amax(dim=1, keepdim=True) - f_min).clamp_min(1e-12)
    kappa = ((frequencies - f_min) / f_span).unsqueeze(-1)

    target = batch[target_key].to(device)
    if torch.is_complex(target):
        target = torch.view_as_real(target)
    else:
        target = target.float()

    return (area, kappa), target
