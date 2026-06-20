import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from utils import make_webster_profile_features


class FrequencyEmbedding(nn.Module):
    """
    Преобразует нормализованную частоту в гармонические признаки.

    Input:
        kappa: [B, Nf, 1]

    Output:
        features: [B, Nf, 1 + 2 * n_bands]
    """

    def __init__(self, n_bands: int = 8) -> None:
        super().__init__()

        bands = 2.0 ** torch.arange(
            n_bands,
            dtype=torch.float32,
        )
        self.register_buffer(
            "bands",
            math.pi * bands,
        )

    def forward(self, kappa: Tensor) -> Tensor:
        if kappa.ndim != 3 or kappa.shape[-1] != 1:
            raise ValueError(
                "kappa must have shape [B, Nf, 1], "
                f"got {tuple(kappa.shape)}"
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


class ResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(
                num_groups=min(8, channels),
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(
                num_groups=min(8, channels),
                num_channels=channels,
            ),
        )

        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class GeometryBranchNet(nn.Module):
    """
    Branch-сеть DeepONet.

    Преобразует профиль площади:

        area: [B, in_channels, Nx]

    в коэффициенты базисного разложения:

        coefficients: [B, out_channels, basis_dim]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.basis_dim = basis_dim
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(
                num_groups=min(8, hidden_channels),
                num_channels=hidden_channels,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            ResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=1,
            ),
            ResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=2,
            ),
            ResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=4,
            ),
            ResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=8,
            ),
        )

        self.pool = nn.AdaptiveAvgPool1d(pooling_bins)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                hidden_channels * pooling_bins,
                256,
            ),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(
                256,
                out_channels * basis_dim,
            ),
        )

    def forward(self, area: Tensor) -> Tensor:
        if area.ndim != 3:
            raise ValueError(
                "area must have shape [B, C, Nx], "
                f"got {tuple(area.shape)}"
            )

        x = self.stem(area)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.head(x)

        return x.view(
            area.shape[0],
            self.out_channels,
            self.basis_dim,
        )


class FrequencyTrunkNet(nn.Module):
    """
    Trunk-сеть DeepONet.

    Преобразует частоты:

        kappa: [B, Nf, 1]

    в значения обучаемых базисных функций:

        basis: [B, Nf, basis_dim]
    """

    def __init__(
        self,
        basis_dim: int = 128,
        frequency_bands: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.frequency_embedding = FrequencyEmbedding(
            n_bands=frequency_bands,
        )

        frequency_input_dim = 1 + 2 * frequency_bands

        self.network = nn.Sequential(
            nn.Linear(
                frequency_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                basis_dim,
            ),
        )

    def forward(self, kappa: Tensor) -> Tensor:
        features = self.frequency_embedding(kappa)
        return self.network(features)


class TransferFunctionDeepONet(nn.Module):
    """
    DeepONet для оператора:

        S(x) -> H(f)

    Формула:

        H_c(f) = sum_j branch_c,j(S) * trunk_j(f) + bias_c

    Output:
        [B, Nf] при out_channels=1
        [B, Nf, out_channels] иначе
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        frequency_bands: int = 8,
        trunk_hidden_dim: int = 128,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.model_name = "transfer_function_deeponet"

        self.branch_net = GeometryBranchNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            basis_dim=basis_dim,
            pooling_bins=pooling_bins,
            out_channels=out_channels,
        )

        self.trunk_net = FrequencyTrunkNet(
            basis_dim=basis_dim,
            frequency_bands=frequency_bands,
            hidden_dim=trunk_hidden_dim,
        )

        self.output_bias = nn.Parameter(
            torch.zeros(out_channels)
        )

        self.output_scale = basis_dim ** -0.5

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        """
        area:
            [B, in_channels, Nx]

        kappa:
            [B, Nf, 1]

        returns:
            [B, Nf] при out_channels=1
            [B, Nf, out_channels] иначе
        """

        if area.ndim != 3:
            raise ValueError(
                "area must have shape [B, C, Nx], "
                f"got {tuple(area.shape)}"
            )

        if area.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} area channels, "
                f"got {area.shape[1]}"
            )

        if kappa.ndim != 3 or kappa.shape[-1] != 1:
            raise ValueError(
                "kappa must have shape [B, Nf, 1], "
                f"got {tuple(kappa.shape)}"
            )

        if area.shape[0] != kappa.shape[0]:
            raise ValueError(
                "area and kappa batch sizes must match"
            )

        # [B, out_channels, basis_dim]
        branch_coefficients = self.branch_net(area)

        # [B, Nf, basis_dim]
        trunk_basis = self.trunk_net(kappa)

        # Для каждого выходного канала:
        #
        # H[b, f, c] =
        #     sum_p branch[b, c, p] * trunk[b, f, p]
        #
        # [B, Nf, out_channels]
        output = torch.einsum(
            "bcp,bfp->bfc",
            branch_coefficients,
            trunk_basis,
        )

        output = (
            output * self.output_scale
            + self.output_bias.view(1, 1, -1)
        )

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


def webster_deeponet_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    n_points: int = 128,
    log_area: bool = True,
    target_key: str = "target",
    frequency_key: str = "frequencies_hz",
    frequency_min_hz: float | None = None,
    frequency_max_hz: float | None = None,
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    """
    Адаптер WebsterTorchDataset под TransferFunctionDeepONet.

    Returns:
        inputs:
            area: [B, 1, Nx]
            kappa: [B, Nf, 1]

        target:
            [B, Nf] или [B, Nf, 2]
    """

    if (
        frequency_min_hz is not None
        and frequency_max_hz is not None
        and frequency_max_hz <= frequency_min_hz
    ):
        raise ValueError(
            "frequency_max_hz must be greater than "
            "frequency_min_hz"
        )

    area = make_webster_profile_features(
        batch,
        n_points=n_points,
        log_area=log_area,
        include_x=False,
        channel_first=True,
        device=device,
    )

    frequencies = (
        batch[frequency_key]
        .to(device=device, dtype=torch.float32)
    )

    if frequency_min_hz is None:
        f_min = frequencies.amin(dim=1, keepdim=True)
    else:
        f_min = torch.full(
            (frequencies.shape[0], 1),
            float(frequency_min_hz),
            device=device,
            dtype=frequencies.dtype,
        )

    if frequency_max_hz is None:
        f_max = frequencies.amax(dim=1, keepdim=True)
    else:
        f_max = torch.full(
            (frequencies.shape[0], 1),
            float(frequency_max_hz),
            device=device,
            dtype=frequencies.dtype,
        )

    frequency_span = (f_max - f_min).clamp_min(1e-12)
    kappa = ((frequencies - f_min) / frequency_span).unsqueeze(-1)

    target = batch[target_key].to(device)

    if torch.is_complex(target):
        target = torch.view_as_real(target)
    else:
        target = target.float()

    return (area, kappa), target    
