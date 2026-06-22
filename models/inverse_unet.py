from __future__ import annotations

import math
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import vt_all_solvers_wrapper as vt
from utils import make_webster_profile_features


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TransferMapInverseUNet(nn.Module):
    """
    UNet-style inverse model for Webster geometry reconstruction.

    Input:
        transfer_map: [B, C, Nf, Nx]

    The channels are expected to contain transfer-function values H(f), the
    normalized frequency coordinate f, and the normalized spatial coordinate x.

    Output:
        profile: [B, out_channels, Nx]

    Usually out_channels=1 and the target is log(area) or physical area.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        bottleneck_multiplier: int = 16,
        dropout: float = 0.0,
        frequency_pool: Literal["mean", "max", "meanmax"] = "mean",
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        if bottleneck_multiplier < 1:
            raise ValueError("bottleneck_multiplier must be >= 1")
        if frequency_pool not in {"mean", "max", "meanmax"}:
            raise ValueError("frequency_pool must be 'mean', 'max', or 'meanmax'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_pool = frequency_pool
        self.model_name = "transfer_map_inverse_unet"

        encoder_channels = [base_channels * mult for mult in channel_multipliers]
        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for channels in encoder_channels:
            self.encoder.append(
                ConvBlock2d(
                    prev_channels,
                    channels,
                    dropout=dropout,
                )
            )
            prev_channels = channels

        bottleneck_channels = base_channels * bottleneck_multiplier
        self.bottleneck = ConvBlock2d(
            prev_channels,
            bottleneck_channels,
            dropout=dropout,
        )

        self.decoder = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.decoder.append(
                ConvBlock2d(
                    current_channels + skip_channels,
                    skip_channels,
                    dropout=dropout,
                )
            )
            current_channels = skip_channels

        pooled_channels = current_channels
        if frequency_pool == "meanmax":
            pooled_channels *= 2

        self.profile_head = nn.Sequential(
            nn.Conv1d(pooled_channels, pooled_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(pooled_channels, out_channels, kernel_size=1),
        )

    def forward(self, transfer_map: Tensor) -> Tensor:
        if transfer_map.ndim != 4:
            raise ValueError(
                "transfer_map must have shape [B, C, Nf, Nx], "
                f"got {tuple(transfer_map.shape)}"
            )
        if transfer_map.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {transfer_map.shape[1]}"
            )

        x = transfer_map
        skips: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        if self.frequency_pool == "mean":
            profile_features = x.mean(dim=2)
        elif self.frequency_pool == "max":
            profile_features = x.amax(dim=2)
        else:
            profile_features = torch.cat(
                [
                    x.mean(dim=2),
                    x.amax(dim=2),
                ],
                dim=1,
            )

        return self.profile_head(profile_features)


def make_transfer_inverse_map(
    batch: Mapping[str, Any],
    *,
    n_profile_points: int = 128,
    target_key: str = "target",
    frequency_key: str = "frequencies_hz",
    map_mode: Literal["broadcast", "prefix_solver"] = "broadcast",
    solver_config: vt.SolverConfig | None = None,
    prefix_target_mode: Literal["magnitude", "db", "phase", "complex", "realimag"] = "db",
    min_prefix_length_fraction: float = 1.0e-3,
    normalize_h: bool = True,
    include_frequency: bool = True,
    include_x: bool = True,
    device: str | torch.device | None = None,
) -> Tensor:
    """
    Build a 2D inverse-problem map from a Webster batch.

    Output:
        [B, C, Nf, Nx], where C is H channels plus optional f and x channels.

    map_mode:
        "broadcast" repeats the full transfer function along x.
        "prefix_solver" solves every truncated geometry [0, x_j], so column j
        contains the response of the channel prefix up to x_j.
    """

    device = torch.device(device) if device is not None else None

    if map_mode == "broadcast":
        h = batch[target_key]
        h = h.to(device) if device is not None else h
        if torch.is_complex(h):
            h = torch.view_as_real(h)

        h = h.float()
        if h.ndim == 2:
            h_channels = h.unsqueeze(1)
        elif h.ndim == 3:
            h_channels = h.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(
                f"target must have shape [B, Nf] or [B, Nf, C], got {tuple(h.shape)}"
            )

        batch_size, _, n_frequencies = h_channels.shape
        h_map = h_channels.unsqueeze(-1).expand(
            batch_size,
            h_channels.shape[1],
            n_frequencies,
            n_profile_points,
        )
    elif map_mode == "prefix_solver":
        h_map = _make_prefix_solver_transfer_map(
            batch,
            n_profile_points=n_profile_points,
            solver_config=solver_config,
            target_mode=prefix_target_mode,
            min_prefix_length_fraction=min_prefix_length_fraction,
            device=device,
        )
        batch_size, _, n_frequencies, _ = h_map.shape
        h_channels = h_map.flatten(start_dim=2)
    else:
        raise ValueError("map_mode must be 'broadcast' or 'prefix_solver'")

    if normalize_h:
        if map_mode == "broadcast":
            mean = h_channels.mean(dim=2, keepdim=True)
            std = h_channels.std(dim=2, keepdim=True).clamp_min(1.0e-6)
            h_channels = (h_channels - mean) / std
            h_map = h_channels.unsqueeze(-1).expand(
                batch_size,
                h_channels.shape[1],
                n_frequencies,
                n_profile_points,
            )
        else:
            mean = h_map.flatten(start_dim=2).mean(dim=2).view(batch_size, -1, 1, 1)
            std = (
                h_map.flatten(start_dim=2)
                .std(dim=2)
                .clamp_min(1.0e-6)
                .view(batch_size, -1, 1, 1)
            )
            h_map = (h_map - mean) / std

    channels = [h_map]

    if include_frequency:
        frequencies = batch[frequency_key]
        frequencies = frequencies.to(device) if device is not None else frequencies
        frequencies = frequencies.float()
        if frequencies.ndim != 2:
            raise ValueError(
                "frequencies_hz must have shape [B, Nf], "
                f"got {tuple(frequencies.shape)}"
            )
        f_min = frequencies.amin(dim=1, keepdim=True)
        f_span = (frequencies.amax(dim=1, keepdim=True) - f_min).clamp_min(1.0e-12)
        f_norm = (frequencies - f_min) / f_span
        f_map = f_norm[:, None, :, None].expand(
            batch_size,
            1,
            n_frequencies,
            n_profile_points,
        )
        channels.append(f_map)

    if include_x:
        x_grid = torch.linspace(
            0.0,
            1.0,
            n_profile_points,
            device=h_map.device,
            dtype=h_map.dtype,
        )
        x_map = x_grid.view(1, 1, 1, n_profile_points).expand(
            batch_size,
            1,
            n_frequencies,
            n_profile_points,
        )
        channels.append(x_map)

    return torch.cat(channels, dim=1).contiguous()


def _make_prefix_solver_transfer_map(
    batch: Mapping[str, Any],
    *,
    n_profile_points: int,
    solver_config: vt.SolverConfig | None,
    target_mode: Literal["magnitude", "db", "phase", "complex", "realimag"],
    min_prefix_length_fraction: float,
    device: torch.device | None,
) -> Tensor:
    if solver_config is None:
        raise ValueError("solver_config is required when map_mode='prefix_solver'")
    if "geometry" not in batch:
        raise ValueError("batch must contain geometry when map_mode='prefix_solver'")
    if min_prefix_length_fraction <= 0.0:
        raise ValueError("min_prefix_length_fraction must be positive")

    geometry_batch = batch["geometry"]
    x_batch = geometry_batch["x_m"]
    area_batch = geometry_batch["area_m2"]
    node_counts = geometry_batch["node_count"]

    batch_size = int(x_batch.shape[0])
    x_grid = torch.linspace(0.0, 1.0, n_profile_points)
    sample_maps: list[Tensor] = []

    for sample_idx in range(batch_size):
        node_count = int(node_counts[sample_idx])
        x_nodes = x_batch[sample_idx, :node_count].detach().cpu().float()
        area_nodes = area_batch[sample_idx, :node_count].detach().cpu().float()
        sample_maps.append(
            _solve_prefixes_for_one_geometry(
                x_nodes=x_nodes,
                area_nodes=area_nodes,
                x_grid=x_grid,
                solver_config=solver_config,
                target_mode=target_mode,
                min_prefix_length_fraction=min_prefix_length_fraction,
            )
        )

    out = torch.stack(sample_maps, dim=0)
    return out.to(device) if device is not None else out


def _solve_prefixes_for_one_geometry(
    *,
    x_nodes: Tensor,
    area_nodes: Tensor,
    x_grid: Tensor,
    solver_config: vt.SolverConfig,
    target_mode: Literal["magnitude", "db", "phase", "complex", "realimag"],
    min_prefix_length_fraction: float,
) -> Tensor:
    x_nodes = x_nodes.double()
    area_nodes = area_nodes.double()
    order = torch.argsort(x_nodes)
    x_nodes = x_nodes[order]
    area_nodes = area_nodes[order].clamp_min(1.0e-12)

    x0 = float(x_nodes[0])
    x1 = float(x_nodes[-1])
    length = x1 - x0
    if length <= 0.0:
        raise ValueError("geometry length must be positive")

    normalized_x = (x_nodes - x0) / length
    min_cut = min_prefix_length_fraction

    prefix_columns: list[Tensor] = []
    for x_fraction in x_grid.tolist():
        cut = max(float(x_fraction), min_cut)
        prefix_x_norm, prefix_area = _make_prefix_geometry_arrays(
            normalized_x=normalized_x,
            area_nodes=area_nodes,
            cut=cut,
        )
        prefix_geometry = vt.ExplicitGeometry(
            x_m=(prefix_x_norm * length).tolist(),
            area_m2=prefix_area.tolist(),
        )
        result = vt.solve(prefix_geometry, config=solver_config)
        prefix_columns.append(_solver_result_to_target_tensor(result, target_mode))

    return torch.stack(prefix_columns, dim=-1)


def _make_prefix_geometry_arrays(
    *,
    normalized_x: Tensor,
    area_nodes: Tensor,
    cut: float,
) -> tuple[Tensor, Tensor]:
    cut = min(max(cut, 1.0e-12), 1.0)
    inside = normalized_x < cut
    prefix_x = normalized_x[inside]
    prefix_area = area_nodes[inside]

    cut_area = _interp1d_scalar(normalized_x, area_nodes, cut)
    if prefix_x.numel() == 0 or float(prefix_x[0]) > 0.0:
        prefix_x = torch.cat([normalized_x[:1] * 0.0, prefix_x])
        prefix_area = torch.cat([area_nodes[:1], prefix_area])

    if prefix_x.numel() == 0 or not torch.isclose(prefix_x[-1], torch.tensor(cut, dtype=prefix_x.dtype)):
        prefix_x = torch.cat([prefix_x, torch.tensor([cut], dtype=prefix_x.dtype)])
        prefix_area = torch.cat([prefix_area, torch.tensor([cut_area], dtype=prefix_area.dtype)])
    else:
        prefix_area[-1] = cut_area

    if prefix_x.numel() == 1:
        prefix_x = torch.cat([prefix_x, torch.tensor([cut], dtype=prefix_x.dtype)])
        prefix_area = torch.cat([prefix_area, torch.tensor([cut_area], dtype=prefix_area.dtype)])

    return prefix_x, prefix_area


def _interp1d_scalar(x: Tensor, y: Tensor, x_new: float) -> float:
    if x_new <= float(x[0]):
        return float(y[0])
    if x_new >= float(x[-1]):
        return float(y[-1])

    right = int(torch.searchsorted(x, torch.tensor(x_new, dtype=x.dtype), right=False))
    left = max(right - 1, 0)
    x_left = float(x[left])
    x_right = float(x[right])
    if math.isclose(x_left, x_right):
        return float(y[left])
    weight = (x_new - x_left) / (x_right - x_left)
    return float(y[left]) * (1.0 - weight) + float(y[right]) * weight


def _solver_result_to_target_tensor(
    result: vt.SpectrumResult,
    target_mode: Literal["magnitude", "db", "phase", "complex", "realimag"],
) -> Tensor:
    if target_mode == "magnitude":
        return torch.tensor(result.magnitude, dtype=torch.float32).view(1, -1)
    if target_mode == "db":
        values = [20.0 * math.log10(max(v, 1.0e-12)) for v in result.magnitude]
        return torch.tensor(values, dtype=torch.float32).view(1, -1)
    if target_mode == "phase":
        values = result.phase_rad
        return torch.tensor(values, dtype=torch.float32).view(1, -1)
    if target_mode == "complex":
        values = [[z.real, z.imag] for z in result.transfer_complex]
        return torch.tensor(values, dtype=torch.float32).T.contiguous()
    if target_mode == "realimag":
        values = [[z.real, z.imag] for z in result.transfer_complex]
        return torch.tensor(values, dtype=torch.float32).T.contiguous()
    raise ValueError(f"unsupported target_mode: {target_mode!r}")


def webster_inverse_unet_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    n_profile_points: int = 128,
    input_target_key: str = "target",
    frequency_key: str = "frequencies_hz",
    target_log_area: bool = True,
    map_mode: Literal["broadcast", "prefix_solver"] = "broadcast",
    solver_config: vt.SolverConfig | None = None,
    prefix_target_mode: Literal["magnitude", "db", "phase", "complex", "realimag"] = "db",
    min_prefix_length_fraction: float = 1.0e-3,
    normalize_h: bool = True,
    include_frequency: bool = True,
    include_x: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Adapter for training TransferMapInverseUNet.

    The input is built from the transfer function H(f), frequency f, and spatial
    coordinate x. The target is the resampled area profile [B, 1, Nx].
    """

    transfer_map = make_transfer_inverse_map(
        batch,
        n_profile_points=n_profile_points,
        target_key=input_target_key,
        frequency_key=frequency_key,
        map_mode=map_mode,
        solver_config=solver_config,
        prefix_target_mode=prefix_target_mode,
        min_prefix_length_fraction=min_prefix_length_fraction,
        normalize_h=normalize_h,
        include_frequency=include_frequency,
        include_x=include_x,
        device=device,
    )

    profile = make_webster_profile_features(
        batch,
        n_points=n_profile_points,
        log_area=target_log_area,
        include_x=False,
        channel_first=True,
        device=device,
    )

    return transfer_map, profile.float()


__all__ = [
    "TransferMapInverseUNet",
    "make_transfer_inverse_map",
    "webster_inverse_unet_batch_to_xy",
]
