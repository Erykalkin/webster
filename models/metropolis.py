from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


ProfileFn = Callable[[Tensor], Tensor]
PreprocessFn = Callable[[Tensor], Tensor]
LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class MetropolisResult:
    samples: Tensor
    losses: Tensor
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

    previous_training_state = model.training
    model.eval()

    def evaluate(
        parameters: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        area_physical = profile_fn(parameters)
        area_model_input = preprocess(area_physical)

        prediction = model(
            area_model_input,
            frequency_input,
        )

        loss = transfer_loss_fn(
            prediction,
            target,
        )

        return loss, area_physical, prediction

    current_loss, current_area, current_prediction = evaluate(
        current_parameters
    )

    best_loss = float(current_loss)
    best_parameters = current_parameters.clone()
    best_area = current_area.clone()
    best_prediction = current_prediction.clone()

    stored_samples: list[Tensor] = []
    stored_losses: list[Tensor] = []

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

        if step >= burn_in:
            if (step - burn_in) % thinning == 0:
                stored_samples.append(
                    current_parameters.cpu().clone()
                )
                stored_losses.append(
                    current_loss.cpu().clone()
                )

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
        accepted_fraction=accepted / n_steps,
        best_parameters=best_parameters,
        best_area=best_area,
        best_prediction=best_prediction,
        best_loss=best_loss,
    )