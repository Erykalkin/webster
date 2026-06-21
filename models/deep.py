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
