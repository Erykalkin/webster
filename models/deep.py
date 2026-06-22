import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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


class DynamicConv1d(nn.Module):
    """
    Conv1d with input-dependent kernels.

    A small routing network predicts a convex combination of several expert
    kernels for every sample in the batch. This lets the branch encoder adapt
    its local filters to different geometry families without changing the
    public DeepONet input format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        *,
        padding: int | None = None,
        dilation: int = 1,
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.dilation = dilation
        self.n_experts = n_experts
        self.temperature = temperature

        self.weight = nn.Parameter(
            torch.empty(
                n_experts,
                out_channels,
                in_channels,
                kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(n_experts, out_channels))
            if bias
            else None
        )

        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, routing_hidden_dim),
            nn.GELU(),
            nn.Linear(routing_hidden_dim, n_experts),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_idx in range(self.n_experts):
            nn.init.kaiming_uniform_(
                self.weight[expert_idx],
                a=math.sqrt(5),
            )

        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, C, Nx], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        batch_size, _, n_points = x.shape
        routing_logits = self.routing(x)
        routing_weights = torch.softmax(
            routing_logits / self.temperature,
            dim=-1,
        )

        # [B, out_channels, in_channels, kernel_size]
        mixed_weight = torch.einsum(
            "be,eock->bock",
            routing_weights,
            self.weight,
        )
        mixed_weight = mixed_weight.reshape(
            batch_size * self.out_channels,
            self.in_channels,
            self.kernel_size,
        )

        mixed_bias = None
        if self.bias is not None:
            mixed_bias = torch.einsum(
                "be,eo->bo",
                routing_weights,
                self.bias,
            ).reshape(batch_size * self.out_channels)

        grouped_x = x.reshape(
            1,
            batch_size * self.in_channels,
            n_points,
        )
        out = torch.nn.functional.conv1d(
            grouped_x,
            mixed_weight,
            mixed_bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch_size,
        )

        return out.reshape(
            batch_size,
            self.out_channels,
            out.shape[-1],
        )


class DynamicResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.conv1 = DynamicConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            n_experts=n_experts,
            routing_hidden_dim=routing_hidden_dim,
            temperature=temperature,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, channels),
            num_channels=channels,
        )
        self.conv2 = DynamicConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            n_experts=n_experts,
            routing_hidden_dim=routing_hidden_dim,
            temperature=temperature,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, channels),
            num_channels=channels,
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(residual + x)


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


class DynamicGeometryBranchNet(nn.Module):
    """
    DeepONet branch network with dynamic convolution.

    Input:
        area: [B, in_channels, Nx]

    Output:
        coefficients: [B, out_channels, basis_dim]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        out_channels: int = 1,
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.basis_dim = basis_dim
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            DynamicConv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            nn.GroupNorm(
                num_groups=min(8, hidden_channels),
                num_channels=hidden_channels,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=1,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=2,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=4,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=8,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
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


class DeformableConv1d(nn.Module):
    """
    Lightweight 1D deformable convolution.

    The layer predicts a small offset for every kernel tap and every spatial
    position, samples the input profile with linear interpolation, and then
    applies a learned convolution kernel to the sampled values. With zero
    offsets it behaves like a standard same-length Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        *,
        padding: int | None = None,
        dilation: int = 1,
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        if max_offset < 0.0:
            raise ValueError("max_offset must be >= 0")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.dilation = dilation
        self.max_offset = float(max_offset)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(out_channels))
            if bias
            else None
        )

        offset_hidden_channels = offset_hidden_channels or max(16, in_channels)
        offset_padding = dilation * (kernel_size - 1) // 2
        self.offset_net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                offset_hidden_channels,
                kernel_size=kernel_size,
                padding=offset_padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv1d(
                offset_hidden_channels,
                kernel_size,
                kernel_size=3,
                padding=1,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

        last_offset_layer = self.offset_net[-1]
        if isinstance(last_offset_layer, nn.Conv1d):
            nn.init.zeros_(last_offset_layer.weight)
            nn.init.zeros_(last_offset_layer.bias)

    def _sample_input(self, x: Tensor, offsets: Tensor) -> Tensor:
        batch_size, channels, n_points = x.shape

        padded = torch.nn.functional.pad(
            x,
            (self.padding, self.padding),
        )
        padded_points = padded.shape[-1]

        base_positions = torch.arange(
            n_points,
            device=x.device,
            dtype=x.dtype,
        )
        kernel_positions = torch.arange(
            self.kernel_size,
            device=x.device,
            dtype=x.dtype,
        ) * self.dilation
        positions = (
            base_positions.view(1, 1, n_points)
            + kernel_positions.view(1, self.kernel_size, 1)
            + offsets
        )
        positions = positions.clamp(0.0, float(padded_points - 1))

        left = positions.floor().long()
        right = (left + 1).clamp(max=padded_points - 1)
        alpha = (positions - left.to(dtype=x.dtype)).unsqueeze(1)

        gather_left = left.unsqueeze(1).expand(
            batch_size,
            channels,
            self.kernel_size,
            n_points,
        )
        gather_right = right.unsqueeze(1).expand_as(gather_left)
        expanded = padded.unsqueeze(2).expand(
            batch_size,
            channels,
            self.kernel_size,
            padded_points,
        )

        left_values = torch.gather(
            expanded,
            dim=-1,
            index=gather_left,
        )
        right_values = torch.gather(
            expanded,
            dim=-1,
            index=gather_right,
        )

        return left_values * (1.0 - alpha) + right_values * alpha

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, C, Nx], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        offsets = torch.tanh(self.offset_net(x)) * self.max_offset
        sampled = self._sample_input(x, offsets)

        output = torch.einsum(
            "bckn,ock->bon",
            sampled,
            self.weight,
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output


class DeformableResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = DeformableConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            max_offset=max_offset,
            offset_hidden_channels=offset_hidden_channels,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, channels),
            num_channels=channels,
        )
        self.conv2 = DeformableConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            max_offset=max_offset,
            offset_hidden_channels=offset_hidden_channels,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, channels),
            num_channels=channels,
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(residual + x)


class DeformableGeometryBranchNet(nn.Module):
    """
    DeepONet branch network with deformable 1D convolutions.

    Input:
        area: [B, in_channels, Nx]

    Output:
        coefficients: [B, out_channels, basis_dim]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        out_channels: int = 1,
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.basis_dim = basis_dim
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            DeformableConv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
            nn.GroupNorm(
                num_groups=min(8, hidden_channels),
                num_channels=hidden_channels,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=1,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=2,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=4,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=8,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
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


class SineLayer(nn.Module):
    """
    Linear layer with SIREN sine activation.

    Input:
        [..., in_features]

    Output:
        [..., out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega_0: float = 30.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features < 1:
            raise ValueError("in_features must be >= 1")
        if out_features < 1:
            raise ValueError("out_features must be >= 1")
        if omega_0 <= 0.0:
            raise ValueError("omega_0 must be positive")

        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class FrequencySIRENTrunkNet(nn.Module):
    """
    SIREN trunk network for DeepONet.

    Converts dimensionless frequency

        kappa: [B, Nf, input_dim]

    to learned basis functions

        basis: [B, Nf, basis_dim]
    """

    def __init__(
        self,
        basis_dim: int = 128,
        input_dim: int = 1,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
    ) -> None:
        super().__init__()

        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        layers: list[nn.Module] = [
            SineLayer(
                input_dim,
                hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        for _ in range(n_hidden_layers - 1):
            layers.append(
                SineLayer(
                    hidden_dim,
                    hidden_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        final_layer = nn.Linear(hidden_dim, basis_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            final_layer.weight.uniform_(-bound, bound)

            if final_layer.bias is not None:
                final_layer.bias.uniform_(-bound, bound)

        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, kappa: Tensor) -> Tensor:
        if kappa.ndim != 3:
            raise ValueError(
                "kappa must have shape [B, Nf, input_dim], "
                f"got {tuple(kappa.shape)}"
            )

        return self.network(kappa)


class GeometryTokenEncoder(nn.Module):
    """
    Geometry encoder that keeps the spatial sequence.

    Input:
        area: [B, C, Nx]

    Output:
        tokens: [B, Nx, D]
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        hidden_channels = hidden_channels or d_model
        self.in_channels = in_channels

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
            ResidualConvBlock1d(hidden_channels, kernel_size=5, dilation=1),
            ResidualConvBlock1d(hidden_channels, kernel_size=5, dilation=2),
            ResidualConvBlock1d(hidden_channels, kernel_size=5, dilation=4),
            ResidualConvBlock1d(hidden_channels, kernel_size=5, dilation=8),
        )

        self.projection = nn.Conv1d(
            hidden_channels,
            d_model,
            kernel_size=1,
        )

    def forward(self, area: Tensor) -> Tensor:
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

        x = self.stem(area)
        x = self.blocks(x)
        x = self.projection(x)
        return x.transpose(1, 2).contiguous()


class FrequencyTokenEncoder(nn.Module):
    """
    Frequency encoder for frequency-conditioned sequence operators.

    Input:
        kappa: [B, Nf, 1]

    Output:
        tokens: [B, Nf, D]
    """

    def __init__(
        self,
        d_model: int = 128,
        frequency_bands: int = 16,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.frequency_embedding = FrequencyEmbedding(
            n_bands=frequency_bands,
        )
        input_dim = 1 + 2 * frequency_bands
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, kappa: Tensor) -> Tensor:
        if kappa.ndim != 3 or kappa.shape[-1] != 1:
            raise ValueError(
                "kappa must have shape [B, Nf, 1], "
                f"got {tuple(kappa.shape)}"
            )
        return self.network(self.frequency_embedding(kappa))


class LightweightMambaBlock(nn.Module):
    """
    Local Mamba-like sequence block with the same [B, L, D] interface.

    This is intentionally dependency-free. It is not a drop-in replacement for
    mamba-ssm internals, but gives the project a cheap gated state/conv-style
    sequence block that can later be swapped for a real Mamba module.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 2,
        expansion: int = 2,
        kernel_size: int = 9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if expansion < 1:
            raise ValueError("expansion must be >= 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        hidden_dim = d_model * expansion
        padding = kernel_size // 2

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "in_proj": nn.Linear(d_model, 2 * hidden_dim),
                        "conv": nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=kernel_size,
                            padding=padding,
                            groups=hidden_dim,
                        ),
                        "out_proj": nn.Linear(hidden_dim, d_model),
                        "dropout": nn.Dropout(dropout),
                    }
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, L, D], "
                f"got {tuple(x.shape)}"
            )

        for layer in self.layers:
            residual = x
            x_norm = layer["norm"](x)
            value, gate = layer["in_proj"](x_norm).chunk(2, dim=-1)
            value = value.transpose(1, 2).contiguous()
            value = layer["conv"](value)
            value = value.transpose(1, 2).contiguous()
            value = torch.nn.functional.silu(value)
            value = value * torch.sigmoid(gate)
            value = layer["out_proj"](value)
            x = residual + layer["dropout"](value)

        return x


def _mamba2_segsum(values: Tensor) -> Tensor:
    sequence_length = values.shape[-1]
    repeated = values.unsqueeze(-1).expand(*values.shape, sequence_length)
    lower_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=values.device,
            dtype=torch.bool,
        ),
        diagonal=-1,
    )
    repeated = repeated.masked_fill(~lower_mask, 0.0)
    segment_sum = torch.cumsum(repeated, dim=-2)
    inclusive_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=values.device,
            dtype=torch.bool,
        ),
        diagonal=0,
    )
    return segment_sum.masked_fill(~inclusive_mask, -torch.inf)


def _mamba2_ssd_minimal_discrete(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    block_len: int,
) -> Tensor:
    """
    Minimal SSD scan used by Mamba-2.

    Shapes:
        x: [B, L, H, P]
        a: [B, L, H]
        b: [B, L, G, N]
        c: [B, L, G, N]

    This mirrors the official SSD-minimal algorithm, but is kept local and
    pure PyTorch so the project does not depend on fused CUDA/Triton kernels.
    """

    if x.ndim != 4 or a.ndim != 3 or b.ndim != 4 or c.ndim != 4:
        raise ValueError("invalid SSD tensor ranks")
    if x.shape[1] % block_len != 0:
        raise ValueError("sequence length must be divisible by block_len")

    batch_size, sequence_length, n_heads, head_dim = x.shape
    n_chunks = sequence_length // block_len

    x_chunks = x.reshape(batch_size, n_chunks, block_len, n_heads, head_dim)
    a_chunks = a.reshape(batch_size, n_chunks, block_len, n_heads)
    b_chunks = b.reshape(batch_size, n_chunks, block_len, b.shape[2], b.shape[3])
    c_chunks = c.reshape(batch_size, n_chunks, block_len, c.shape[2], c.shape[3])

    # The current operator uses one group. Expand to heads so the equations are
    # explicit and easy to audit.
    if b_chunks.shape[3] == 1:
        b_heads = b_chunks.expand(batch_size, n_chunks, block_len, n_heads, b_chunks.shape[-1])
        c_heads = c_chunks.expand(batch_size, n_chunks, block_len, n_heads, c_chunks.shape[-1])
    elif b_chunks.shape[3] == n_heads:
        b_heads = b_chunks
        c_heads = c_chunks
    else:
        raise ValueError("SSD group count must be 1 or equal to n_heads")

    # [B, H, C, L]
    a_by_head = a_chunks.permute(0, 3, 1, 2).contiguous()
    a_cumsum = torch.cumsum(a_by_head, dim=-1)

    # Intra-chunk outputs.
    lower_triangular = torch.exp(_mamba2_segsum(a_by_head))
    y_diag = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        c_heads,
        b_heads,
        lower_triangular,
        x_chunks,
    )

    # Chunk states.
    decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
    states = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        b_heads,
        decay_states,
        x_chunks,
    )

    initial_states = torch.zeros_like(states[:, :1])
    states_with_initial = torch.cat([initial_states, states], dim=1)
    padded_chunk_end = torch.nn.functional.pad(a_cumsum[:, :, :, -1], (1, 0))
    decay_chunk = torch.exp(_mamba2_segsum(padded_chunk_end))
    propagated_states = torch.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk,
        states_with_initial,
    )
    states = propagated_states[:, :-1]

    # State-to-output conversion.
    state_decay_out = torch.exp(a_cumsum)
    y_off = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        c_heads,
        states,
        state_decay_out,
    )

    return (y_diag + y_off).reshape(batch_size, sequence_length, n_heads, head_dim)


class MinimalMamba2Layer(nn.Module):
    """
    Pure PyTorch Mamba-2/SSD layer with [B, L, D] input/output.

    It follows the official Mamba2Simple structure: projection into z/x/B/C/dt,
    depthwise causal convolution, SSD scan, skip D, gate, normalization, and
    output projection. The fused CUDA/Triton kernels are intentionally avoided.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        a_init_range: tuple[float, float] = (1.0, 16.0),
    ) -> None:
        super().__init__()

        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if d_state < 1:
            raise ValueError("d_state must be >= 1")
        if d_conv < 1:
            raise ValueError("d_conv must be >= 1")
        if expand < 1:
            raise ValueError("expand must be >= 1")
        if headdim < 1:
            raise ValueError("headdim must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        if self.d_inner % headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        if ngroups not in (1, self.nheads):
            raise ValueError("ngroups must be 1 or equal to nheads")

        self.norm_in = nn.LayerNorm(d_model)

        projection_dim = (
            2 * self.d_inner
            + 2 * ngroups * d_state
            + self.nheads
        )
        self.in_proj = nn.Linear(d_model, projection_dim, bias=False)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        inverse_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inverse_dt)

        a = torch.empty(self.nheads).uniform_(*a_init_range)
        self.a_log = nn.Parameter(torch.log(a))
        self.d_skip = nn.Parameter(torch.ones(self.nheads))

        self.norm_out = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_chunk(self, values: Tensor) -> tuple[Tensor, int]:
        sequence_length = values.shape[1]
        remainder = sequence_length % self.chunk_size
        if remainder == 0:
            return values, sequence_length

        pad_length = self.chunk_size - remainder
        padding = values[:, -1:, :].expand(values.shape[0], pad_length, values.shape[2])
        return torch.cat([values, padding], dim=1), sequence_length

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, L, D], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.shape[-1]}")

        residual = x
        x = self.norm_in(x)
        x, original_length = self._pad_to_chunk(x)
        batch_size, sequence_length, _ = x.shape

        projected = self.in_proj(x)
        z, xbc, dt = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        xbc = self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)
        xbc = torch.nn.functional.silu(xbc[:, :sequence_length, :])

        x_ssm, b, c = torch.split(
            xbc,
            [
                self.d_inner,
                self.ngroups * self.d_state,
                self.ngroups * self.d_state,
            ],
            dim=-1,
        )

        dt = torch.nn.functional.softplus(dt + self.dt_bias.view(1, 1, -1))
        a = -torch.exp(self.a_log).view(1, 1, self.nheads) * dt

        x_heads = x_ssm.reshape(
            batch_size,
            sequence_length,
            self.nheads,
            self.headdim,
        )
        b = b.reshape(batch_size, sequence_length, self.ngroups, self.d_state)
        c = c.reshape(batch_size, sequence_length, self.ngroups, self.d_state)

        y = _mamba2_ssd_minimal_discrete(
            x_heads * dt.unsqueeze(-1),
            a,
            b,
            c,
            block_len=self.chunk_size,
        )
        y = y + x_heads * self.d_skip.view(1, 1, self.nheads, 1)
        y = y.reshape(batch_size, sequence_length, self.d_inner)
        y = y * torch.nn.functional.silu(z)
        y = self.out_proj(self.norm_out(y))
        y = y[:, :original_length, :]

        return residual + self.dropout(y)


class MinimalMamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        depth: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.layers = nn.Sequential(
            *[
                MinimalMamba2Layer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    chunk_size=chunk_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ExternalMamba2Block(nn.Module):
    """
    Thin wrapper around mamba-ssm's Mamba2Simple.

    Use this only in environments where mamba-ssm is installed and its CUDA /
    Triton kernels are healthy. The local MinimalMamba2Block is the portable
    default for this project.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
        use_mem_eff_path: bool = False,
    ) -> None:
        super().__init__()

        try:
            from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "mamba_ssm is not available. Use mamba_backend='minimal_mamba2' "
                "or install a working mamba-ssm environment."
            ) from exc

        self.layers = nn.Sequential(
            *[
                Mamba2Simple(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    chunk_size=chunk_size,
                    use_mem_eff_path=use_mem_eff_path,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x.contiguous())


class FrequencyConditionedMambaOperator(nn.Module):
    """
    Frequency-conditioned bidirectional sequence operator over the tube axis.

    The geometry tokens are FiLM-conditioned by each frequency, then processed
    along x in both directions. To avoid materializing all frequencies at once,
    use frequency_chunk_size.
    """

    def __init__(
        self,
        geometry_encoder: nn.Module,
        frequency_encoder: nn.Module,
        mamba_forward: nn.Module,
        mamba_backward: nn.Module,
        d_model: int = 128,
        out_channels: int = 2,
        frequency_chunk_size: int | None = 32,
    ) -> None:
        super().__init__()

        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if frequency_chunk_size is not None and frequency_chunk_size < 1:
            raise ValueError("frequency_chunk_size must be >= 1 or None")

        self.geometry_encoder = geometry_encoder
        self.frequency_encoder = frequency_encoder
        self.d_model = d_model
        self.out_channels = out_channels
        self.frequency_chunk_size = frequency_chunk_size

        self.gamma_head = nn.Linear(d_model, d_model)
        self.beta_head = nn.Linear(d_model, d_model)

        self.mamba_forward = mamba_forward
        self.mamba_backward = mamba_backward

        self.fusion = nn.Linear(2 * d_model, d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def _forward_frequency_chunk(
        self,
        geometry_tokens: Tensor,
        frequency_tokens: Tensor,
    ) -> Tensor:
        batch_size, n_x, d_model = geometry_tokens.shape
        n_frequencies = frequency_tokens.shape[1]

        gamma = self.gamma_head(frequency_tokens)
        beta = self.beta_head(frequency_tokens)

        conditioned_geometry = (
            geometry_tokens[:, None, :, :]
            * (1.0 + gamma[:, :, None, :])
            + beta[:, :, None, :]
        )

        conditioned_geometry = conditioned_geometry.reshape(
            batch_size * n_frequencies,
            n_x,
            d_model,
        )

        forward_features = self.mamba_forward(conditioned_geometry)

        backward_input = torch.flip(
            conditioned_geometry,
            dims=[1],
        )
        backward_features = self.mamba_backward(backward_input)
        backward_features = torch.flip(
            backward_features,
            dims=[1],
        )

        spatial_features = self.fusion(
            torch.cat(
                [forward_features, backward_features],
                dim=-1,
            )
        )

        pooled = spatial_features.mean(dim=1)
        output = self.output_head(pooled)

        return output.reshape(
            batch_size,
            n_frequencies,
            self.out_channels,
        )

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        geometry_tokens = self.geometry_encoder(area)
        frequency_tokens = self.frequency_encoder(kappa)

        if geometry_tokens.ndim != 3:
            raise ValueError(
                "geometry_encoder must return [B, Nx, D], "
                f"got {tuple(geometry_tokens.shape)}"
            )
        if frequency_tokens.ndim != 3:
            raise ValueError(
                "frequency_encoder must return [B, Nf, D], "
                f"got {tuple(frequency_tokens.shape)}"
            )
        if geometry_tokens.shape[0] != frequency_tokens.shape[0]:
            raise ValueError("geometry and frequency batch sizes must match")
        if geometry_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected geometry token dim {self.d_model}, "
                f"got {geometry_tokens.shape[-1]}"
            )
        if frequency_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected frequency token dim {self.d_model}, "
                f"got {frequency_tokens.shape[-1]}"
            )

        n_frequencies = frequency_tokens.shape[1]
        chunk_size = self.frequency_chunk_size or n_frequencies
        outputs = []

        for start in range(0, n_frequencies, chunk_size):
            stop = min(start + chunk_size, n_frequencies)
            outputs.append(
                self._forward_frequency_chunk(
                    geometry_tokens,
                    frequency_tokens[:, start:stop, :],
                )
            )

        return torch.cat(outputs, dim=1)


class TransferFunctionMambaOperator(nn.Module):
    """
    Frequency-conditioned bidirectional Mamba-like operator.

    Input:
        area: [B, C, Nx]
        kappa: [B, Nf, 1]

    Output:
        [B, Nf] when out_channels=1
        [B, Nf, out_channels] otherwise
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        hidden_channels: int | None = None,
        frequency_bands: int = 16,
        frequency_hidden_dim: int | None = None,
        mamba_backend: str = "minimal_mamba2",
        mamba_depth: int = 2,
        mamba_expansion: int = 2,
        mamba_kernel_size: int = 9,
        mamba_d_state: int = 64,
        mamba_headdim: int = 64,
        mamba_chunk_size: int = 64,
        mamba_use_mem_eff_path: bool = False,
        dropout: float = 0.0,
        out_channels: int = 1,
        frequency_chunk_size: int | None = 32,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.model_name = "transfer_function_mamba_operator"

        geometry_encoder = GeometryTokenEncoder(
            in_channels=in_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
        )
        frequency_encoder = FrequencyTokenEncoder(
            d_model=d_model,
            frequency_bands=frequency_bands,
            hidden_dim=frequency_hidden_dim,
        )

        def make_mamba_block() -> nn.Module:
            if mamba_backend == "minimal_mamba2":
                return MinimalMamba2Block(
                    d_model=d_model,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    dropout=dropout,
                )

            if mamba_backend == "external_mamba2":
                return ExternalMamba2Block(
                    d_model=d_model,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    use_mem_eff_path=mamba_use_mem_eff_path,
                )

            if mamba_backend == "lightweight":
                return LightweightMambaBlock(
                    d_model=d_model,
                    depth=mamba_depth,
                    expansion=mamba_expansion,
                    kernel_size=mamba_kernel_size,
                    dropout=dropout,
                )

            raise ValueError(
                "mamba_backend must be 'minimal_mamba2', "
                "'external_mamba2', or 'lightweight'"
            )

        mamba_forward = make_mamba_block()
        mamba_backward = make_mamba_block()

        self.operator = FrequencyConditionedMambaOperator(
            geometry_encoder=geometry_encoder,
            frequency_encoder=frequency_encoder,
            mamba_forward=mamba_forward,
            mamba_backward=mamba_backward,
            d_model=d_model,
            out_channels=out_channels,
            frequency_chunk_size=frequency_chunk_size,
        )

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        output = self.operator(area, kappa)
        if self.out_channels == 1:
            return output.squeeze(-1)
        return output


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


class TransferFunctionFiLMDeepONet(nn.Module):
    """
    DeepONet with FiLM modulation at the branch/trunk intersection.

    Geometry branch coefficients predict per-basis gamma/beta parameters that
    modulate the frequency trunk basis before the DeepONet dot product.
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
        self.model_name = "transfer_function_film_deeponet"

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
        self.film_head = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, 2 * basis_dim),
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.output_scale = basis_dim ** -0.5

    def forward(self, area: Tensor, kappa: Tensor) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)
        gamma, beta = self.film_head(branch_coefficients).chunk(2, dim=-1)

        modulated_trunk = (
            trunk_basis[:, None, :, :]
            * (1.0 + gamma[:, :, None, :])
            + beta[:, :, None, :]
        )
        output = (
            branch_coefficients[:, :, None, :]
            * modulated_trunk
        ).sum(dim=-1)

        output = output.transpose(1, 2).contiguous()
        output = (
            output * self.output_scale
            + self.output_bias.view(1, 1, -1)
        )

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


class TransferFunctionBilinearFusionDeepONet(nn.Module):
    """
    DeepONet with low-rank bilinear branch/trunk fusion.

    Instead of only multiplying equal basis coordinates, branch and trunk
    coefficients are projected into a shared fusion space before multiplication.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        frequency_bands: int = 8,
        trunk_hidden_dim: int = 128,
        fusion_rank: int | None = None,
        out_channels: int = 1,
        residual_dot: bool = True,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        fusion_rank = basis_dim if fusion_rank is None else fusion_rank
        if fusion_rank < 1:
            raise ValueError("fusion_rank must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.fusion_rank = fusion_rank
        self.residual_dot = residual_dot
        self.model_name = "transfer_function_bilinear_fusion_deeponet"

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
        self.branch_projection = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, fusion_rank),
        )
        self.trunk_projection = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, fusion_rank),
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.output_scale = fusion_rank ** -0.5
        self.residual_scale = basis_dim ** -0.5

    def forward(self, area: Tensor, kappa: Tensor) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

        branch_fused = self.branch_projection(branch_coefficients)
        trunk_fused = self.trunk_projection(trunk_basis)
        output = torch.einsum(
            "bcr,bfr->bfc",
            branch_fused,
            trunk_fused,
        ) * self.output_scale

        if self.residual_dot:
            output = output + torch.einsum(
                "bcp,bfp->bfc",
                branch_coefficients,
                trunk_basis,
            ) * self.residual_scale

        output = output + self.output_bias.view(1, 1, -1)

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


class TransferFunctionAttentionFusionDeepONet(nn.Module):
    """
    Lightweight cross-attention at the branch/trunk intersection.

    Trunk basis vectors are frequency queries. Geometry branch coefficients are
    converted to a small memory of key/value tokens, so the attention cost is
    O(B * out_channels * Nf * memory_tokens), not O(Nf^2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        frequency_bands: int = 8,
        trunk_hidden_dim: int = 128,
        attention_dim: int = 64,
        memory_tokens: int = 8,
        out_channels: int = 1,
        dropout: float = 0.0,
        residual_dot: bool = True,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")
        if attention_dim < 1:
            raise ValueError("attention_dim must be >= 1")
        if memory_tokens < 1:
            raise ValueError("memory_tokens must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.attention_dim = attention_dim
        self.memory_tokens = memory_tokens
        self.residual_dot = residual_dot
        self.model_name = "transfer_function_attention_fusion_deeponet"

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
        self.query_projection = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, attention_dim),
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(
                basis_dim,
                2 * memory_tokens * attention_dim,
            ),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.residual_scale = basis_dim ** -0.5

    def forward(self, area: Tensor, kappa: Tensor) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        batch_size = area.shape[0]
        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

        query = self.query_projection(trunk_basis)
        memory = self.memory_projection(branch_coefficients)
        memory = memory.view(
            batch_size,
            self.out_channels,
            self.memory_tokens,
            2,
            self.attention_dim,
        )
        key = memory[:, :, :, 0, :]
        value = memory[:, :, :, 1, :]

        attention_logits = torch.einsum(
            "bfd,bcmd->bcfm",
            query,
            key,
        ) * (self.attention_dim ** -0.5)
        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )
        attended = torch.einsum(
            "bcfm,bcmd->bcfd",
            attention_weights,
            value,
        )

        output = self.output_head(attended).squeeze(-1)

        if self.residual_dot:
            output = output + torch.einsum(
                "bcp,bfp->bcf",
                branch_coefficients,
                trunk_basis,
            ) * self.residual_scale

        output = output + self.output_bias.view(1, -1, 1)
        output = output.transpose(1, 2).contiguous()

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


class TransferFunctionSIRENDeepONet(nn.Module):
    """
    DeepONet variant with a SIREN trunk network.

    The branch encoder is the same geometry CNN as in TransferFunctionDeepONet,
    while the trunk maps kappa directly through sine layers instead of using
    harmonic embedding plus GELU MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        trunk_hidden_dim: int = 128,
        trunk_hidden_layers: int = 3,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
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
        self.model_name = "transfer_function_siren_deeponet"

        self.branch_net = GeometryBranchNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            basis_dim=basis_dim,
            pooling_bins=pooling_bins,
            out_channels=out_channels,
        )

        self.trunk_net = FrequencySIRENTrunkNet(
            basis_dim=basis_dim,
            input_dim=1,
            hidden_dim=trunk_hidden_dim,
            n_hidden_layers=trunk_hidden_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.output_scale = basis_dim ** -0.5

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

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


class TransferFunctionMambaFusionDeepONet(nn.Module):
    """
    DeepONet with Mamba placed at the branch/trunk intersection.

    Standard DeepONet computes:

        output[f] = sum_p branch[p] * trunk[f, p]

    This variant first forms per-frequency fused tokens:

        token[f, p] = branch[p] * trunk[f, p]

    Then a sequence block processes token[f] along the frequency axis before
    the final scalar head. This keeps Mamba at the branch/trunk intersection
    without rerunning geometry processing for every frequency.
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
        mamba_backend: str = "minimal_mamba2",
        mamba_depth: int = 1,
        mamba_expansion: int = 2,
        mamba_kernel_size: int = 4,
        mamba_d_state: int = 32,
        mamba_headdim: int = 32,
        mamba_chunk_size: int = 64,
        mamba_use_mem_eff_path: bool = False,
        dropout: float = 0.0,
        residual_dot: bool = True,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.residual_dot = residual_dot
        self.model_name = "transfer_function_mamba_fusion_deeponet"

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

        def make_mamba_block() -> nn.Module:
            if mamba_backend == "minimal_mamba2":
                return MinimalMamba2Block(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    dropout=dropout,
                )

            if mamba_backend == "external_mamba2":
                return ExternalMamba2Block(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    use_mem_eff_path=mamba_use_mem_eff_path,
                )

            if mamba_backend == "lightweight":
                return LightweightMambaBlock(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    expansion=mamba_expansion,
                    kernel_size=mamba_kernel_size,
                    dropout=dropout,
                )

            raise ValueError(
                "mamba_backend must be 'minimal_mamba2', "
                "'external_mamba2', or 'lightweight'"
            )

        self.fusion_norm = nn.LayerNorm(basis_dim)
        self.fusion_mamba = make_mamba_block()
        self.output_head = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, basis_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(basis_dim, 1),
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.output_scale = basis_dim ** -0.5
    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        batch_size = area.shape[0]
        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

        # [B, C, Nf, P]
        fused_tokens = (
            branch_coefficients[:, :, None, :]
            * trunk_basis[:, None, :, :]
            * self.output_scale
        )

        base_output = fused_tokens.sum(dim=-1) if self.residual_dot else None

        # Mamba processes the branch/trunk intersection along frequency.
        n_frequencies = trunk_basis.shape[1]
        fused_tokens = fused_tokens.reshape(
            batch_size * self.out_channels,
            n_frequencies,
            self.basis_dim,
        )
        fused_tokens = self.fusion_norm(fused_tokens)
        fused_tokens = self.fusion_mamba(fused_tokens)
        correction = self.output_head(fused_tokens).squeeze(-1)
        correction = correction.reshape(
            batch_size,
            self.out_channels,
            n_frequencies,
        )

        if base_output is not None:
            output = base_output + correction
        else:
            output = correction

        output = output + self.output_bias.view(1, -1, 1)
        output = output.transpose(1, 2).contiguous()

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


class DynamicDeformableGeometryBranchNet(nn.Module):
    """
    Branch network that combines dynamic and deformable 1D convolutions.

    Dynamic layers adapt convolution kernels to the current geometry, while
    deformable layers adapt sampling positions along the profile axis.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        out_channels: int = 1,
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.basis_dim = basis_dim
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            DynamicConv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            nn.GroupNorm(
                num_groups=min(8, hidden_channels),
                num_channels=hidden_channels,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=1,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=2,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
            DynamicResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=4,
                n_experts=n_experts,
                routing_hidden_dim=routing_hidden_dim,
                temperature=temperature,
            ),
            DeformableResidualConvBlock1d(
                hidden_channels,
                kernel_size=5,
                dilation=8,
                max_offset=max_offset,
                offset_hidden_channels=offset_hidden_channels,
            ),
        )

        self.pool = nn.AdaptiveAvgPool1d(pooling_bins)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooling_bins, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, out_channels * basis_dim),
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


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("checkpoint must be a mapping or contain model_state")


def _resolve_checkpoint_path(
    checkpoint: str | Path,
    checkpoint_dir: str | Path,
) -> Path:
    path = Path(checkpoint)
    if path.exists():
        return path

    checkpoint_dir = Path(checkpoint_dir)
    if path.suffix:
        return checkpoint_dir / path

    best_candidate = checkpoint_dir / f"{path.name}_best.pt"
    if best_candidate.exists():
        return best_candidate

    return checkpoint_dir / f"{path.name}.pt"


def _copy_matching_checkpoint_entries(
    model: nn.Module,
    *,
    state_dict: Mapping[str, Any],
    block_name: str,
    prefix_pairs: Sequence[tuple[str, str]],
    verbose: bool,
) -> tuple[int, int]:
    own_state = model.state_dict()
    updated_state = dict(own_state)
    copied = 0
    skipped = 0

    for source_key, value in state_dict.items():
        if source_key == "_metadata":
            continue

        target_key = None
        for source_prefix, target_prefix in prefix_pairs:
            if source_key == source_prefix:
                target_key = target_prefix
                break
            if source_key.startswith(source_prefix):
                target_key = target_prefix + source_key[len(source_prefix):]
                break

        if target_key is None:
            continue

        if target_key not in own_state or not isinstance(value, Tensor):
            skipped += 1
            continue

        target_value = own_state[target_key]
        if target_value.shape != value.shape:
            skipped += 1
            continue

        updated_state[target_key] = value.detach().to(
            device=target_value.device,
            dtype=target_value.dtype,
        )
        copied += 1

    model.load_state_dict(updated_state, strict=True)

    if verbose:
        print(
            f"warm-start {block_name}: copied={copied}, "
            f"skipped={skipped}"
        )

    return copied, skipped


def _warm_start_one_checkpoint(
    model: nn.Module,
    *,
    checkpoint: str | Path | None,
    block_name: str,
    prefix_pairs: Sequence[tuple[str, str]],
    checkpoint_dir: str | Path,
    map_location: str | torch.device,
    verbose: bool,
) -> tuple[int, int]:
    if checkpoint is None:
        if verbose:
            print(f"warm-start {block_name}: skipped, checkpoint is None")
        return 0, 0

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint,
        checkpoint_dir,
    )
    if not checkpoint_path.exists():
        if verbose:
            print(
                f"warm-start {block_name}: skipped, "
                f"checkpoint not found: {checkpoint_path}"
            )
        return 0, 0

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state_dict = _checkpoint_state_dict(loaded_checkpoint)
    copied, skipped = _copy_matching_checkpoint_entries(
        model,
        state_dict=state_dict,
        block_name=block_name,
        prefix_pairs=prefix_pairs,
        verbose=verbose,
    )

    if verbose:
        print(f"warm-start {block_name}: source={checkpoint_path}")

    return copied, skipped


def warm_start_mamba_siren_dynamic_deformable_deeponet(
    model: nn.Module,
    *,
    mamba_fusion_checkpoint: str | Path | None = None,
    siren_checkpoint: str | Path | None = None,
    dynamic_checkpoint: str | Path | None = None,
    deformable_checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    map_location: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, tuple[int, int]]:
    """
    Load compatible blocks from separately trained models.

    Checkpoint arguments accept either a full path or a checkpoint stem such
    as "siren_deeponet_db"; stems resolve to checkpoints/<stem>_best.pt.
    Passing None skips that donor.
    """

    return {
        "mamba_fusion": _warm_start_one_checkpoint(
            model,
            checkpoint=mamba_fusion_checkpoint,
            block_name="mamba_fusion",
            prefix_pairs=(
                ("fusion_norm.", "fusion_norm."),
                ("fusion_mamba.", "fusion_mamba."),
                ("output_head.", "output_head."),
                ("output_bias", "output_bias"),
            ),
            checkpoint_dir=checkpoint_dir,
            map_location=map_location,
            verbose=verbose,
        ),
        "siren": _warm_start_one_checkpoint(
            model,
            checkpoint=siren_checkpoint,
            block_name="siren",
            prefix_pairs=(("trunk_net.", "trunk_net."),),
            checkpoint_dir=checkpoint_dir,
            map_location=map_location,
            verbose=verbose,
        ),
        "dynamic": _warm_start_one_checkpoint(
            model,
            checkpoint=dynamic_checkpoint,
            block_name="dynamic",
            prefix_pairs=(
                ("branch_net.stem.", "branch_net.stem."),
                ("branch_net.blocks.0.", "branch_net.blocks.0."),
                ("branch_net.blocks.2.", "branch_net.blocks.2."),
            ),
            checkpoint_dir=checkpoint_dir,
            map_location=map_location,
            verbose=verbose,
        ),
        "deformable": _warm_start_one_checkpoint(
            model,
            checkpoint=deformable_checkpoint,
            block_name="deformable",
            prefix_pairs=(
                ("branch_net.blocks.1.", "branch_net.blocks.1."),
                ("branch_net.blocks.3.", "branch_net.blocks.3."),
            ),
            checkpoint_dir=checkpoint_dir,
            map_location=map_location,
            verbose=verbose,
        ),
    }


class TransferFunctionMambaSIRENDynamicDeformableDeepONet(nn.Module):
    """
    Hybrid DeepONet:

    * dynamic + deformable CNN branch for geometry,
    * SIREN trunk for frequency,
    * Mamba fusion at the branch/trunk intersection.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        basis_dim: int = 128,
        pooling_bins: int = 8,
        trunk_hidden_dim: int = 128,
        trunk_hidden_layers: int = 3,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
        out_channels: int = 1,
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
        mamba_backend: str = "minimal_mamba2",
        mamba_depth: int = 1,
        mamba_expansion: int = 2,
        mamba_kernel_size: int = 4,
        mamba_d_state: int = 32,
        mamba_headdim: int = 32,
        mamba_chunk_size: int = 64,
        mamba_use_mem_eff_path: bool = False,
        dropout: float = 0.0,
        residual_dot: bool = True,

        warm_start_mamba_fusion_checkpoint: str | Path | None = None,
        warm_start_siren_checkpoint: str | Path | None = None,
        warm_start_dynamic_checkpoint: str | Path | None = None,
        warm_start_deformable_checkpoint: str | Path | None = None,
        warm_start_checkpoint_dir: str | Path = "checkpoints",
        warm_start_map_location: str | torch.device = "cpu",
        warm_start_verbose: bool = True,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.residual_dot = residual_dot
        self.model_name = "transfer_function_mamba_siren_dynamic_deformable_deeponet"

        self.branch_net = DynamicDeformableGeometryBranchNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            basis_dim=basis_dim,
            pooling_bins=pooling_bins,
            out_channels=out_channels,
            n_experts=n_experts,
            routing_hidden_dim=routing_hidden_dim,
            temperature=temperature,
            max_offset=max_offset,
            offset_hidden_channels=offset_hidden_channels,
        )

        self.trunk_net = FrequencySIRENTrunkNet(
            basis_dim=basis_dim,
            input_dim=1,
            hidden_dim=trunk_hidden_dim,
            n_hidden_layers=trunk_hidden_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        def make_mamba_block() -> nn.Module:
            if mamba_backend == "minimal_mamba2":
                return MinimalMamba2Block(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    dropout=dropout,
                )

            if mamba_backend == "external_mamba2":
                return ExternalMamba2Block(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    d_state=mamba_d_state,
                    d_conv=mamba_kernel_size,
                    expand=mamba_expansion,
                    headdim=mamba_headdim,
                    chunk_size=mamba_chunk_size,
                    use_mem_eff_path=mamba_use_mem_eff_path,
                )

            if mamba_backend == "lightweight":
                return LightweightMambaBlock(
                    d_model=basis_dim,
                    depth=mamba_depth,
                    expansion=mamba_expansion,
                    kernel_size=mamba_kernel_size,
                    dropout=dropout,
                )

            raise ValueError(
                "mamba_backend must be 'minimal_mamba2', "
                "'external_mamba2', or 'lightweight'"
            )

        self.fusion_norm = nn.LayerNorm(basis_dim)
        self.fusion_mamba = make_mamba_block()
        self.output_head = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, basis_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(basis_dim, 1),
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.output_scale = basis_dim ** -0.5

        self.warm_start_summary = warm_start_mamba_siren_dynamic_deformable_deeponet(
            self,
            mamba_fusion_checkpoint=warm_start_mamba_fusion_checkpoint,
            siren_checkpoint=warm_start_siren_checkpoint,
            dynamic_checkpoint=warm_start_dynamic_checkpoint,
            deformable_checkpoint=warm_start_deformable_checkpoint,
            checkpoint_dir=warm_start_checkpoint_dir,
            map_location=warm_start_map_location,
            verbose=warm_start_verbose,
        )

    def forward(
        self,
        area: Tensor,
        kappa: Tensor,
    ) -> Tensor:
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
            raise ValueError("area and kappa batch sizes must match")

        batch_size = area.shape[0]
        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

        fused_tokens = (
            branch_coefficients[:, :, None, :]
            * trunk_basis[:, None, :, :]
            * self.output_scale
        )

        base_output = fused_tokens.sum(dim=-1) if self.residual_dot else None

        n_frequencies = trunk_basis.shape[1]
        fused_tokens = fused_tokens.reshape(
            batch_size * self.out_channels,
            n_frequencies,
            self.basis_dim,
        )
        fused_tokens = self.fusion_norm(fused_tokens)
        fused_tokens = self.fusion_mamba(fused_tokens)
        correction = self.output_head(fused_tokens).squeeze(-1)
        correction = correction.reshape(
            batch_size,
            self.out_channels,
            n_frequencies,
        )

        if base_output is not None:
            output = base_output + correction
        else:
            output = correction

        output = output + self.output_bias.view(1, -1, 1)
        output = output.transpose(1, 2).contiguous()

        if self.out_channels == 1:
            return output.squeeze(-1)

        return output


class TransferFunctionDynamicDeepONet(nn.Module):
    """
    DeepONet variant with a dynamic-CNN branch encoder.

    Public input/output format is the same as TransferFunctionDeepONet:

        area: [B, in_channels, Nx]
        kappa: [B, Nf, 1]

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
        n_experts: int = 4,
        routing_hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.model_name = "transfer_function_dynamic_deeponet"

        self.branch_net = DynamicGeometryBranchNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            basis_dim=basis_dim,
            pooling_bins=pooling_bins,
            out_channels=out_channels,
            n_experts=n_experts,
            routing_hidden_dim=routing_hidden_dim,
            temperature=temperature,
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

        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

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


class TransferFunctionDeformableDeepONet(nn.Module):
    """
    DeepONet variant with a deformable-CNN branch encoder.

    Unlike the dynamic variant, this model keeps one learned convolution kernel
    but learns input-dependent sampling offsets along the profile axis.
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
        max_offset: float = 2.0,
        offset_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        if basis_dim < 1:
            raise ValueError("basis_dim must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.model_name = "transfer_function_deformable_deeponet"

        self.branch_net = DeformableGeometryBranchNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            basis_dim=basis_dim,
            pooling_bins=pooling_bins,
            out_channels=out_channels,
            max_offset=max_offset,
            offset_hidden_channels=offset_hidden_channels,
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

        branch_coefficients = self.branch_net(area)
        trunk_basis = self.trunk_net(kappa)

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
