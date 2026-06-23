from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


ProfileFn = Callable[[Tensor], Tensor]
ModelInputs = Tensor | tuple[Tensor, ...]
PreprocessFn = Callable[[Tensor], ModelInputs]
ForwardFn = Callable[[nn.Module, ModelInputs, Tensor], Tensor]
LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class MetropolisResult:
    samples: Tensor
    losses: Tensor
    trace_losses: Tensor
    accepted_fraction: float
    best_parameters: Tensor
    best_area: Tensor
    best_prediction: Tensor
    best_loss: float


@torch.no_grad()
def metropolis_inverse(
    model: nn.Module,
    target: Tensor,
    frequency_input: Tensor,
    initial_parameters: Tensor,
    profile_fn: ProfileFn,
    transfer_loss_fn: LossFn,
    *,
    area_preprocess_fn: PreprocessFn | None = None,
    forward_fn: ForwardFn | None = None,
    n_steps: int = 50_000,
    burn_in: int = 10_000,
    thinning: int = 10,
    proposal_std: float = 0.05,
    temperature: float = 1.0,
) -> MetropolisResult:
    """
    Метод Метрополиса для обратного восстановления профиля площади.

    Args:
        model:
            Замороженная прямая модель:
                area, frequency -> transfer function.

        target:
            Целевая передаточная функция:
                [1, Nf]
                или [1, Nf, C].

        frequency_input:
            Частотный вход прямой модели:
                [1, Nf, D].

        initial_parameters:
            Начальные параметры профиля:
                [1, M].

            Например, M контрольных точек или B-spline коэффициентов.

        profile_fn:
            Дифференцируемая или обычная функция:
                parameters -> physical area [1, 1, Nx].

        transfer_loss_fn:
            Функция ошибки передаточной функции.

        area_preprocess_fn:
            Тот же preprocessing площади, который применялся
            при обучении прямой модели.

        forward_fn:
            Универсальный вызов модели:
                model, model_inputs, frequency_input -> prediction.

            Если None, используется операторный формат:
                model(area_input, frequency_input).

        temperature:
            Масштаб распределения.

            Малое значение концентрирует выборку около минимумов.
            Большое значение позволяет исследовать больше профилей.
    """

    if n_steps <= burn_in:
        raise ValueError("n_steps must be greater than burn_in")

    if thinning < 1:
        raise ValueError("thinning must be >= 1")

    if proposal_std <= 0.0:
        raise ValueError("proposal_std must be positive")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    try:
        model_parameter = next(model.parameters())
        device = model_parameter.device
        dtype = model_parameter.dtype
    except StopIteration:
        device = target.device
        dtype = target.dtype

    target = target.to(
        device=device,
        dtype=dtype,
    )

    frequency_input = frequency_input.to(
        device=device,
        dtype=dtype,
    )

    current_parameters = initial_parameters.to(
        device=device,
        dtype=dtype,
    ).clone()

    preprocess = (
        area_preprocess_fn
        if area_preprocess_fn is not None
        else lambda area: area
    )
    forward = (
        forward_fn
        if forward_fn is not None
        else lambda model, model_inputs, frequency: model(
            model_inputs,
            frequency,
        )
    )

    previous_training_state = model.training
    model.eval()

    def evaluate(
        parameters: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        area_physical = profile_fn(parameters)
        area_model_input = preprocess(area_physical)

        prediction = forward(
            model,
            area_model_input,
            frequency_input,
        )

        loss = transfer_loss_fn(
            prediction,
            target,
        )

        return loss, area_physical, prediction

    try:
        current_loss, current_area, current_prediction = evaluate(
            current_parameters
        )

        best_loss = float(current_loss)
        best_parameters = current_parameters.clone()
        best_area = current_area.clone()
        best_prediction = current_prediction.clone()

        stored_samples: list[Tensor] = []
        stored_losses: list[Tensor] = []
        trace_losses: list[Tensor] = []

        accepted = 0

        for step in range(n_steps):
            proposal = (
                current_parameters
                + proposal_std
                * torch.randn_like(current_parameters)
            )

            proposal_loss, proposal_area, proposal_prediction = evaluate(
                proposal
            )

            # Не нормированная логарифмическая апостериорная вероятность:
            #
            # log p(q | H) = -loss / temperature + const
            current_log_probability = (
                -current_loss / temperature
            )

            proposal_log_probability = (
                -proposal_loss / temperature
            )

            log_acceptance_ratio = (
                proposal_log_probability
                - current_log_probability
            )

            log_uniform = torch.log(
                torch.rand(
                    (),
                    device=device,
                    dtype=dtype,
                )
            )

            if log_uniform < torch.minimum(
                torch.zeros_like(log_acceptance_ratio),
                log_acceptance_ratio,
            ):
                current_parameters = proposal
                current_loss = proposal_loss
                current_area = proposal_area
                current_prediction = proposal_prediction
                accepted += 1

            if float(current_loss) < best_loss:
                best_loss = float(current_loss)
                best_parameters = current_parameters.clone()
                best_area = current_area.clone()
                best_prediction = current_prediction.clone()

            trace_losses.append(
                current_loss.detach().cpu().clone()
            )

            if step >= burn_in:
                if (step - burn_in) % thinning == 0:
                    stored_samples.append(
                        current_parameters.cpu().clone()
                    )
                    stored_losses.append(
                        current_loss.cpu().clone()
                    )
    finally:
        model.train(previous_training_state)

    if not stored_samples:
        raise RuntimeError(
            "No samples were stored. Check burn_in and thinning."
        )

    return MetropolisResult(
        samples=torch.cat(
            stored_samples,
            dim=0,
        ),
        losses=torch.stack(
            stored_losses,
        ),
        trace_losses=torch.stack(
            trace_losses,
        ),
        accepted_fraction=accepted / n_steps,
        best_parameters=best_parameters,
        best_area=best_area,
        best_prediction=best_prediction,
        best_loss=best_loss,
    )


def metropolis_inverse_for_project_model(
    model: nn.Module,
    target: Tensor,
    frequencies_hz: Tensor,
    initial_parameters: Tensor,
    profile_fn: ProfileFn,
    transfer_loss_fn: LossFn,
    *,
    model_kind: str = "operator",
    log_area: bool = True,
    include_x: bool = True,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    n_steps: int = 50_000,
    burn_in: int = 10_000,
    thinning: int = 10,
    proposal_std: float = 0.05,
    temperature: float = 1.0,
) -> MetropolisResult:
    """
    Convenience wrapper for project models.

    model_kind="operator" supports FNO/DeepONet/Mamba-like models:
        model(area, kappa)

    model_kind="mlp" supports ProfileMLP:
        model(profile)
    """

    from models.back_base import (
        make_area_preprocess_fn,
        make_forward_fn,
        make_kappa,
    )

    try:
        model_parameter = next(model.parameters())
        device = model_parameter.device
        dtype = model_parameter.dtype
    except StopIteration:
        device = target.device
        dtype = target.dtype

    kappa = make_kappa(
        frequencies_hz.to(device=device, dtype=dtype),
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )

    return metropolis_inverse(
        model=model,
        target=target,
        frequency_input=kappa,
        initial_parameters=initial_parameters,
        profile_fn=profile_fn,
        transfer_loss_fn=transfer_loss_fn,
        area_preprocess_fn=make_area_preprocess_fn(
            model_kind,
            log_area=log_area,
            include_x=include_x,
        ),
        forward_fn=make_forward_fn(model_kind),
        n_steps=n_steps,
        burn_in=burn_in,
        thinning=thinning,
        proposal_std=proposal_std,
        temperature=temperature,
    )
