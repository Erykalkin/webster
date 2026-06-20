from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ModelKind = Literal["operator", "mlp"]
ModelInputs = Tensor | tuple[Tensor, ...]
AreaPreprocessFn = Callable[[Tensor], ModelInputs]
ForwardFn = Callable[[nn.Module, ModelInputs, Tensor], Tensor]
TransferLossFn = Callable[[Tensor, Tensor], Tensor]


DEFAULT_MIN_AREA_M2 = 1.0e-4
DEFAULT_MAX_AREA_M2 = 5.0e-2


class AreaParameterization(nn.Module):
    """
    Positive physical area profile parameterization.

    The optimized variables are control points in physical units, m^2.
    They are interpolated to the profile grid expected by a surrogate model.

    Output:
        area_m2: [B, 1, Nx]
    """

    def __init__(
        self,
        batch_size: int,
        n_points: int = 128,
        n_control_points: int = 16,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
        max_area_m2: float = DEFAULT_MAX_AREA_M2,
        initial_area_m2: Tensor | None = None,
        initialization_noise: float = 0.1,
        fixed_inlet_area_m2: float | None = None,
        fixed_outlet_area_m2: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        if n_control_points < 2:
            raise ValueError("n_control_points must be >= 2")
        if max_area_m2 <= min_area_m2:
            raise ValueError("max_area_m2 must be greater than min_area_m2")

        self.n_points = int(n_points)
        self.n_control_points = int(n_control_points)
        self.min_area_m2 = float(min_area_m2)
        self.max_area_m2 = float(max_area_m2)
        self.fixed_inlet_area_m2 = fixed_inlet_area_m2
        self.fixed_outlet_area_m2 = fixed_outlet_area_m2

        if initial_area_m2 is None:
            raw = initialization_noise * torch.randn(
                batch_size,
                1,
                n_control_points,
                device=device,
                dtype=dtype,
            )
        else:
            initial_area_m2 = initial_area_m2.to(device=device, dtype=dtype)
            if initial_area_m2.ndim == 2:
                initial_area_m2 = initial_area_m2.unsqueeze(1)
            if initial_area_m2.ndim != 3:
                raise ValueError(
                    "initial_area_m2 must have shape [B, 1, Nx] or [B, Nx], "
                    f"got {tuple(initial_area_m2.shape)}"
                )
            if initial_area_m2.shape[:2] != (batch_size, 1):
                raise ValueError(
                    "initial_area_m2 batch/channel dimensions must be "
                    f"{(batch_size, 1)}, got {tuple(initial_area_m2.shape[:2])}"
                )

            control_area = F.interpolate(
                initial_area_m2,
                size=n_control_points,
                mode="linear",
                align_corners=True,
            )
            normalized = (
                (control_area - self.min_area_m2)
                / (self.max_area_m2 - self.min_area_m2)
            ).clamp(1.0e-5, 1.0 - 1.0e-5)
            raw = torch.logit(normalized)

        self.raw_control_points = nn.Parameter(raw)

    def forward(self) -> Tensor:
        control_area = self.min_area_m2 + (
            self.max_area_m2 - self.min_area_m2
        ) * torch.sigmoid(self.raw_control_points)

        area = F.interpolate(
            control_area,
            size=self.n_points,
            mode="linear",
            align_corners=True,
        )

        if self.fixed_inlet_area_m2 is not None:
            inlet = torch.full_like(area[..., :1], float(self.fixed_inlet_area_m2))
            area = torch.cat([inlet, area[..., 1:]], dim=-1)

        if self.fixed_outlet_area_m2 is not None:
            outlet = torch.full_like(area[..., -1:], float(self.fixed_outlet_area_m2))
            area = torch.cat([area[..., :-1], outlet], dim=-1)

        return area


def operator_area_input(
    area_m2: Tensor,
    *,
    log_area: bool = True,
    eps: float = 1.0e-12,
) -> Tensor:
    """
    Convert physical area to FNO/DeepONet input:
        [B, 1, Nx]
    """

    if area_m2.ndim != 3 or area_m2.shape[1] != 1:
        raise ValueError(f"area_m2 must have shape [B, 1, Nx], got {tuple(area_m2.shape)}")

    if log_area:
        return torch.log(area_m2.clamp_min(eps))
    return area_m2


def mlp_profile_input(
    area_m2: Tensor,
    *,
    log_area: bool = True,
    include_x: bool = True,
    eps: float = 1.0e-12,
) -> Tensor:
    """
    Convert physical area to ProfileMLP input:
        [B, Nx, C], usually C=2: log(area), x.
    """

    area = operator_area_input(area_m2, log_area=log_area, eps=eps)
    profile = area.transpose(1, 2).contiguous()

    if not include_x:
        return profile

    batch_size, n_points, _ = profile.shape
    x = torch.linspace(
        0.0,
        1.0,
        n_points,
        device=profile.device,
        dtype=profile.dtype,
    ).view(1, n_points, 1)
    x = x.expand(batch_size, -1, -1)
    return torch.cat([profile, x], dim=-1)


def make_kappa(
    frequencies_hz: Tensor,
    *,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> Tensor:
    """
    Normalize frequencies to the [B, Nf, 1] kappa format used by FNO/DeepONet.
    """

    if frequencies_hz.ndim == 1:
        frequencies_hz = frequencies_hz.unsqueeze(0)
    if frequencies_hz.ndim != 2:
        raise ValueError(
            "frequencies_hz must have shape [Nf] or [B, Nf], "
            f"got {tuple(frequencies_hz.shape)}"
        )

    if f_min_hz is None:
        f_min = frequencies_hz.amin(dim=1, keepdim=True)
    else:
        f_min = torch.full_like(frequencies_hz[:, :1], float(f_min_hz))

    if f_max_hz is None:
        f_max = frequencies_hz.amax(dim=1, keepdim=True)
    else:
        f_max = torch.full_like(frequencies_hz[:, :1], float(f_max_hz))

    return ((frequencies_hz - f_min) / (f_max - f_min).clamp_min(1.0e-12)).unsqueeze(-1)


def make_area_preprocess_fn(
    model_kind: ModelKind,
    *,
    log_area: bool = True,
    include_x: bool = True,
) -> AreaPreprocessFn:
    if model_kind == "operator":
        return lambda area: operator_area_input(area, log_area=log_area)
    if model_kind == "mlp":
        return lambda area: mlp_profile_input(
            area,
            log_area=log_area,
            include_x=include_x,
        )
    raise ValueError(f"Unsupported model_kind: {model_kind!r}")


def make_forward_fn(model_kind: ModelKind) -> ForwardFn:
    if model_kind == "operator":
        return lambda model, area_input, kappa: model(area_input, kappa)
    if model_kind == "mlp":
        return lambda model, profile, kappa: model(profile)
    raise ValueError(f"Unsupported model_kind: {model_kind!r}")


def transfer_mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if torch.is_complex(prediction):
        prediction = torch.view_as_real(prediction)
    if torch.is_complex(target):
        target = torch.view_as_real(target)
    return torch.mean((prediction - target) ** 2)


def area_regularization(
    area: Tensor,
    *,
    reference_area: Tensor | None = None,
    reference_area_m2: Tensor | None = None,
    eps: float = 1.0e-8,
) -> dict[str, Tensor]:
    """
    Compute regularization for a physical area profile.

    Derivatives are taken with respect to normalized coordinate x in [0, 1].
    The regularization is applied to log(area), so it penalizes relative
    rather than absolute area changes.
    """

    if reference_area is not None and reference_area_m2 is not None:
        raise ValueError("Provide only one of reference_area or reference_area_m2")

    if reference_area is None:
        reference_area = reference_area_m2

    if area.ndim != 3:
        raise ValueError(
            "area must have shape [B, 1, Nx], "
            f"got {tuple(area.shape)}"
        )

    if area.shape[1] != 1:
        raise ValueError(
            "area must have exactly one channel, "
            f"got {area.shape[1]}"
        )

    n_points = area.shape[-1]
    if n_points < 3:
        raise ValueError(
            "At least 3 area points are required to compute the second derivative"
        )

    dx = 1.0 / float(n_points - 1)
    log_area = torch.log(area.clamp_min(eps))

    first_derivative = torch.diff(log_area, dim=-1) / dx
    second_derivative = torch.diff(log_area, n=2, dim=-1) / (dx ** 2)

    smoothness = torch.mean(first_derivative.square())
    curvature = torch.mean(second_derivative.square())

    if reference_area is None:
        prior = area.new_zeros(())
    else:
        reference_area = reference_area.to(
            device=area.device,
            dtype=area.dtype,
        )

        if reference_area.ndim == 2:
            reference_area = reference_area.unsqueeze(1)

        if reference_area.ndim != 3:
            raise ValueError(
                "reference_area must have shape [B, Nx] or [B, 1, Nx], "
                f"got {tuple(reference_area.shape)}"
            )

        if reference_area.shape[1] != 1:
            raise ValueError(
                "reference_area must have exactly one channel, "
                f"got {reference_area.shape[1]}"
            )

        if reference_area.shape[0] not in (1, area.shape[0]):
            raise ValueError(
                "reference_area batch size must be 1 or match area. "
                f"Got {reference_area.shape[0]} and {area.shape[0]}"
            )

        if reference_area.shape[-1] != n_points:
            reference_area = F.interpolate(
                reference_area,
                size=n_points,
                mode="linear",
                align_corners=True,
            )

        reference_log_area = torch.log(reference_area.clamp_min(eps))
        prior = torch.mean(
            (log_area - reference_log_area).square()
        )

    return {
        "smoothness": smoothness,
        "curvature": curvature,
        "prior": prior,
    }


@dataclass
class InverseSolution:
    area_m2: Tensor
    model_inputs: ModelInputs
    prediction: Tensor
    target: Tensor
    total_loss: float
    transfer_loss: float
    history: list[dict[str, float]]


class InverseAreaSolver:
    """
    Inverse geometry solver through a frozen neural surrogate.

    Supported project models:
        model_kind="mlp"      -> ProfileMLP(profile)
        model_kind="operator" -> TransferFunctionFNO(area, kappa)
                              -> TransferFunctionDeepONet(area, kappa)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        model_kind: ModelKind = "operator",
        area_preprocess_fn: AreaPreprocessFn | None = None,
        forward_fn: ForwardFn | None = None,
        transfer_loss_fn: TransferLossFn | None = None,
        log_area: bool = True,
        include_x: bool = True,
    ) -> None:
        self.model = model
        self.model_kind = model_kind
        self.area_preprocess_fn = area_preprocess_fn or make_area_preprocess_fn(
            model_kind,
            log_area=log_area,
            include_x=include_x,
        )
        self.forward_fn = forward_fn or make_forward_fn(model_kind)
        self.transfer_loss_fn = transfer_loss_fn or transfer_mse_loss

    def solve(
        self,
        target: Tensor,
        frequencies_hz: Tensor,
        *,
        n_points: int = 128,
        n_control_points: int = 16,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
        max_area_m2: float = DEFAULT_MAX_AREA_M2,
        initial_area_m2: Tensor | None = None,
        reference_area_m2: Tensor | None = None,
        fixed_inlet_area_m2: float | None = None,
        fixed_outlet_area_m2: float | None = None,
        n_steps: int = 3000,
        learning_rate: float = 1.0e-2,
        smoothness_weight: float = 1.0e-4,
        curvature_weight: float = 1.0e-8,
        prior_weight: float = 0.0,
        gradient_clip_norm: float | None = 10.0,
        log_every: int = 100,
        initialization_noise: float = 0.1,
        f_min_hz: float | None = None,
        f_max_hz: float | None = None,
    ) -> InverseSolution:
        if target.ndim < 2:
            raise ValueError("target must have shape [B, Nf] or [B, Nf, C]")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if log_every < 1:
            raise ValueError("log_every must be >= 1")

        try:
            model_parameter = next(self.model.parameters())
            device = model_parameter.device
            dtype = model_parameter.dtype
        except StopIteration:
            device = target.device
            dtype = torch.float32

        if torch.is_complex(target):
            target = torch.view_as_real(target)
        target = target.to(device=device, dtype=dtype)

        frequencies_hz = frequencies_hz.to(device=device, dtype=dtype)
        if frequencies_hz.ndim == 1:
            frequencies_hz = frequencies_hz.unsqueeze(0).expand(target.shape[0], -1)
        if frequencies_hz.ndim != 2:
            raise ValueError(
                "frequencies_hz must have shape [Nf] or [B, Nf], "
                f"got {tuple(frequencies_hz.shape)}"
            )
        if target.shape[0] != frequencies_hz.shape[0]:
            raise ValueError("target and frequencies_hz batch sizes must match")
        if target.shape[1] != frequencies_hz.shape[1]:
            raise ValueError("target and frequencies_hz frequency dimensions must match")

        kappa = make_kappa(
            frequencies_hz,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
        )

        area_parameterization = AreaParameterization(
            batch_size=target.shape[0],
            n_points=n_points,
            n_control_points=n_control_points,
            min_area_m2=min_area_m2,
            max_area_m2=max_area_m2,
            initial_area_m2=initial_area_m2,
            initialization_noise=initialization_noise,
            fixed_inlet_area_m2=fixed_inlet_area_m2,
            fixed_outlet_area_m2=fixed_outlet_area_m2,
            device=device,
            dtype=dtype,
        ).to(device)

        optimizer = torch.optim.Adam(
            area_parameterization.parameters(),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_steps,
            eta_min=learning_rate * 0.01,
        )

        previous_training_state = self.model.training
        previous_requires_grad = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        best_total_loss = float("inf")
        best_transfer_loss = float("inf")
        best_area: Tensor | None = None
        best_prediction: Tensor | None = None
        best_model_inputs: ModelInputs | None = None
        history: list[dict[str, float]] = []

        try:
            for step in range(n_steps):
                optimizer.zero_grad(set_to_none=True)

                area_m2 = area_parameterization()
                model_inputs = self.area_preprocess_fn(area_m2)
                prediction = self.forward_fn(self.model, model_inputs, kappa)

                transfer_loss = self.transfer_loss_fn(prediction, target)
                regularization = area_regularization(
                    area_m2,
                    reference_area=reference_area_m2,
                )

                weighted_smoothness = (
                    smoothness_weight * regularization["smoothness"]
                )
                weighted_curvature = (
                    curvature_weight * regularization["curvature"]
                )
                weighted_prior = (
                    prior_weight * regularization["prior"]
                )

                total_loss = (
                    transfer_loss
                    + weighted_smoothness
                    + weighted_curvature
                    + weighted_prior
                )

                if not torch.isfinite(total_loss):
                    raise RuntimeError("Inverse optimization produced a non-finite loss")

                total_loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        area_parameterization.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()
                scheduler.step()

                current_total = float(total_loss.detach())
                if current_total < best_total_loss:
                    best_total_loss = current_total
                    best_transfer_loss = float(transfer_loss.detach())
                    best_area = area_m2.detach().clone()
                    best_prediction = prediction.detach().clone()
                    best_model_inputs = _detach_model_inputs(model_inputs)

                if step % log_every == 0 or step == n_steps - 1:
                    history.append(
                        {
                            "step": float(step),
                            "total_loss": current_total,
                            "transfer_loss": float(transfer_loss.detach()),
                            "smoothness": float(regularization["smoothness"].detach()),
                            "curvature": float(regularization["curvature"].detach()),
                            "prior": float(regularization["prior"].detach()),
                            "weighted_smoothness": float(weighted_smoothness.detach()),
                            "weighted_curvature": float(weighted_curvature.detach()),
                            "weighted_prior": float(weighted_prior.detach()),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        }
                    )
        finally:
            for parameter, requires_grad in zip(
                self.model.parameters(),
                previous_requires_grad,
            ):
                parameter.requires_grad_(requires_grad)
            self.model.train(previous_training_state)

        if best_area is None or best_prediction is None or best_model_inputs is None:
            raise RuntimeError("Inverse optimization did not produce a valid solution")

        return InverseSolution(
            area_m2=best_area,
            model_inputs=best_model_inputs,
            prediction=best_prediction,
            target=target.detach(),
            total_loss=best_total_loss,
            transfer_loss=best_transfer_loss,
            history=history,
        )


def _detach_model_inputs(model_inputs: ModelInputs) -> ModelInputs:
    if isinstance(model_inputs, tuple):
        return tuple(item.detach().clone() for item in model_inputs)
    return model_inputs.detach().clone()


def make_inverse_solver(
    model: nn.Module,
    *,
    model_kind: ModelKind,
    log_area: bool = True,
    include_x: bool = True,
    transfer_loss_fn: TransferLossFn | None = None,
) -> InverseAreaSolver:
    return InverseAreaSolver(
        model,
        model_kind=model_kind,
        log_area=log_area,
        include_x=include_x,
        transfer_loss_fn=transfer_loss_fn,
    )


__all__ = [
    "AreaParameterization",
    "DEFAULT_MAX_AREA_M2",
    "DEFAULT_MIN_AREA_M2",
    "InverseAreaSolver",
    "InverseSolution",
    "area_regularization",
    "make_area_preprocess_fn",
    "make_forward_fn",
    "make_inverse_solver",
    "make_kappa",
    "mlp_profile_input",
    "operator_area_input",
    "transfer_mse_loss",
]
