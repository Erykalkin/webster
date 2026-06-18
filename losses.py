from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OutputType = Literal["db", "complex"]


class UniversalTransferFunctionLoss(nn.Module):
    """
    Универсальная функция потерь для передаточной функции.

    Поддерживаемые представления:

    1. output_type="db"

       prediction, target:
           [B, Nf]
           [B, Nf, Nout]

       Значения являются модулем передаточной функции в dB.

    2. output_type="complex"

       prediction, target:
           [B, Nf, 2]
           [B, Nf, Nout, 2]

       Последнее измерение:
           [..., 0] = Re(H)
           [..., 1] = Im(H)

    Общие компоненты:
        - Smooth L1 в dB;
        - относительная L2-ошибка линейного модуля.

    Дополнительный компонент для complex:
        - относительная комплексная L2-ошибка.
    """

    def __init__(
        self,
        output_type: OutputType,
        db_weight: float = 1.0,
        magnitude_weight: float = 0.1,
        complex_weight: float = 1.0,
        db_error_scale: float = 10.0,
        min_db: float = -100.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if output_type not in ("db", "complex"):
            raise ValueError(
                "output_type должен быть 'db' или 'complex'"
            )

        self.output_type = output_type

        self.db_weight = db_weight
        self.magnitude_weight = magnitude_weight
        self.complex_weight = complex_weight

        self.db_error_scale = db_error_scale
        self.min_db = min_db
        self.eps = eps

    def _validate(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> None:
        """Проверяет совместимость форм входных тензоров."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"Формы prediction и target не совпадают: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        if prediction.ndim < 2:
            raise ValueError(
                "Ожидается как минимум форма [B, Nf]"
            )

        if self.output_type == "complex":
            if prediction.ndim < 3:
                raise ValueError(
                    "Комплексный выход должен иметь форму "
                    "[B, Nf, 2] или [B, Nf, Nout, 2]"
                )

            if prediction.shape[-1] != 2:
                raise ValueError(
                    "Последнее измерение должно содержать [Re(H), Im(H)]"
                )

    def _complex_magnitude(
        self,
        values: Tensor,
    ) -> Tensor:
        """Вычисляет модуль из каналов Re(H) и Im(H)."""
        return torch.sqrt(
            values[..., 0].square()
            + values[..., 1].square()
            + self.eps
        )

    def _db_to_magnitude(
        self,
        values_db: Tensor,
    ) -> Tensor:
        """Переводит dB в линейный модуль."""
        return torch.pow(
            10.0,
            values_db / 20.0,
        )

    def _magnitude_to_db(
        self,
        magnitude: Tensor,
    ) -> Tensor:
        """Переводит линейный модуль в dB с ограничением снизу."""
        magnitude_floor = 10.0 ** (self.min_db / 20.0)

        return 20.0 * torch.log10(
            magnitude.clamp_min(magnitude_floor)
        )

    def _extract_db_and_magnitude(
        self,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Приводит любое поддерживаемое представление к:
            - модулю в dB;
            - модулю в линейном масштабе.
        """
        if self.output_type == "db":
            values_db = values
            magnitude = self._db_to_magnitude(values_db)
            return values_db, magnitude

        magnitude = self._complex_magnitude(values)
        values_db = self._magnitude_to_db(magnitude)

        return values_db, magnitude

    def db_loss(
        self,
        prediction_db: Tensor,
        target_db: Tensor,
    ) -> Tensor:
        """
        Smooth L1-ошибка АЧХ в dB.

        Ошибка делится на db_error_scale, чтобы компонент имел
        сопоставимый порядок величины с относительными ошибками.

        Например, при db_error_scale=10 ошибка 10 dB соответствует
        нормализованной ошибке 1.
        """
        normalized_error = (
            prediction_db - target_db
        ) / self.db_error_scale

        return F.smooth_l1_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            beta=1.0,
        )

    def relative_magnitude_l2(
        self,
        prediction_magnitude: Tensor,
        target_magnitude: Tensor,
    ) -> Tensor:
        """
        Относительная L2-ошибка линейного модуля.

        Формула:
            |||H_pred| - |H_true|||_2
            -------------------------
                  |||H_true|||_2

        В отличие от dB-компонента сильнее учитывает области
        с большой линейной амплитудой.
        """
        difference = (
            prediction_magnitude - target_magnitude
        ).flatten(start_dim=1)

        target_flat = target_magnitude.flatten(start_dim=1)

        numerator = torch.linalg.vector_norm(
            difference,
            ord=2,
            dim=1,
        )

        denominator = torch.linalg.vector_norm(
            target_flat,
            ord=2,
            dim=1,
        )

        return (
            numerator
            / denominator.clamp_min(self.eps)
        ).mean()

    def relative_complex_l2(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Относительная комплексная L2-ошибка.

        Одновременно контролирует Re(H), Im(H), модуль и фазу:

            ||H_pred - H_true||_2
            --------------------
                 ||H_true||_2
        """
        difference = (
            prediction - target
        ).flatten(start_dim=1)

        target_flat = target.flatten(start_dim=1)

        numerator = torch.linalg.vector_norm(
            difference,
            ord=2,
            dim=1,
        )

        denominator = torch.linalg.vector_norm(
            target_flat,
            ord=2,
            dim=1,
        )

        return (
            numerator
            / denominator.clamp_min(self.eps)
        ).mean()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Вычисляет итоговую функцию потерь.

        Для dB:
            L = w_db * L_db
              + w_mag * L_relative_magnitude

        Для complex:
            L = w_db * L_db
              + w_mag * L_relative_magnitude
              + w_complex * L_relative_complex
        """
        self._validate(prediction, target)

        prediction_db, prediction_magnitude = (
            self._extract_db_and_magnitude(prediction)
        )

        target_db, target_magnitude = (
            self._extract_db_and_magnitude(target)
        )

        db_component = self.db_loss(
            prediction_db,
            target_db,
        )

        magnitude_component = self.relative_magnitude_l2(
            prediction_magnitude,
            target_magnitude,
        )

        total = (
            self.db_weight * db_component
            + self.magnitude_weight * magnitude_component
        )

        complex_component = prediction.new_zeros(())

        if self.output_type == "complex":
            complex_component = self.relative_complex_l2(
                prediction,
                target,
            )

            total = (
                total
                + self.complex_weight * complex_component
            )

        if not return_components:
            return total

        components = {
            "loss_total": total.detach(),
            "loss_db": db_component.detach(),
            "loss_relative_magnitude": magnitude_component.detach(),
            "loss_relative_complex": complex_component.detach(),
        }

        return total, components