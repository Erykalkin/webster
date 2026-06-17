"""
metrics.py

Метрики для сравнения FNO и MLP при прогнозировании передаточной функции.

Поддерживаемые постановки
------------------------
1. Выход в децибелах:
       A(f) = 20 * log10(|H(f)|)

   Ожидаемые формы:
       [B, Nf]
       [B, Nf, Nout]

2. Комплексный выход в формате Re/Im:

   Ожидаемые формы:
       [B, Nf, 2]
       [B, Nf, Nout, 2]

   Последнее измерение:
       [..., 0] = Re(H)
       [..., 1] = Im(H)

Обозначения
-----------
B     — размер батча;
Nf    — число частот;
Nout  — число выходных ветвей/каналов.

Важно
-----
Все метрики FNO и MLP нужно вычислять:
- на одной и той же тестовой выборке;
- после обратной нормализации выходов;
- на одной и той же частотной сетке;
- с одинаковыми настройками поиска резонансов.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn


Reduction = Literal["mean", "none"]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _validate_same_shape(prediction: Tensor, target: Tensor) -> None:
    """Проверяет совпадение форм предсказания и эталона."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Формы prediction и target должны совпадать: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )


def _validate_complex_ri(values: Tensor) -> None:
    """Проверяет, что последнее измерение содержит Re(H) и Im(H)."""
    if values.ndim < 3:
        raise ValueError(
            "Комплексный тензор должен иметь форму [B, Nf, 2] "
            "или [B, Nf, Nout, 2]."
        )
    if values.shape[-1] != 2:
        raise ValueError(
            "Последнее измерение комплексного тензора должно иметь размер 2: "
            "[Re(H), Im(H)]."
        )


def _normalize_frequencies(
    frequencies: Tensor,
    n_frequencies: int,
) -> Tensor:
    """
    Приводит частотную сетку к форме [Nf].

    Поддерживает:
    - [Nf]
    - [B, Nf], если все строки одинаковы
    """
    if frequencies.ndim == 1:
        normalized = frequencies
    elif frequencies.ndim == 2:
        if frequencies.shape[1] != n_frequencies:
            raise ValueError(
                f"Ожидалось {n_frequencies} частот в каждой строке, "
                f"получено {frequencies.shape[1]}."
            )
        if frequencies.shape[0] == 0:
            raise ValueError("frequencies не должен быть пустым.")
        normalized = frequencies[0]
        if not torch.allclose(
            frequencies,
            normalized.unsqueeze(0).expand_as(frequencies),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "Для batched frequencies ожидается одинаковая частотная сетка "
                "во всех элементах батча."
            )
    else:
        raise ValueError("frequencies должен иметь форму [Nf] или [B, Nf].")

    if normalized.numel() != n_frequencies:
        raise ValueError(
            f"Ожидалось {n_frequencies} частот, получено {normalized.numel()}."
        )
    if torch.any(normalized[1:] <= normalized[:-1]):
        raise ValueError("Частоты должны строго возрастать.")
    return normalized


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    """Применяет усреднение по примерам либо возвращает значения по примерам."""
    if reduction == "mean":
        return values.mean()
    if reduction == "none":
        return values
    raise ValueError("reduction должен быть 'mean' или 'none'.")


def _flatten_per_sample(values: Tensor) -> Tensor:
    """Сохраняет размер батча и объединяет все остальные измерения."""
    if values.ndim < 2:
        raise ValueError("Ожидается тензор с размерностью не меньше 2.")
    return values.flatten(start_dim=1)


def _to_complex_tensor(values: Tensor) -> Tensor:
    """
    Приводит комплексное представление к complex Tensor.

    Поддерживает:
    - torch.complex tensor формы [B, Nf] или [B, Nf, Nout]
    - real tensor формы [B, Nf, 2] или [B, Nf, Nout, 2]
    """
    if torch.is_complex(values):
        if values.ndim < 2:
            raise ValueError(
                "Комплексный тензор должен иметь форму [B, Nf] "
                "или [B, Nf, Nout]."
            )
        return values

    _validate_complex_ri(values)
    return torch.complex(values[..., 0], values[..., 1])


def complex_ri_to_complex(values: Tensor) -> Tensor:
    """
    Преобразует два действительных канала [Re(H), Im(H)] в complex-тензор.

    Это вспомогательная функция, а не метрика.
    """
    return _to_complex_tensor(values)


def complex_ri_to_db(
    values: Tensor,
    min_db: float = -100.0,
) -> Tensor:
    """
    Преобразует комплексный выход [Re(H), Im(H)] в модуль в децибелах.

    Значения ниже min_db ограничиваются снизу, чтобы избежать log10(0).
    Это вспомогательная функция, а не метрика.
    """
    complex_values = _to_complex_tensor(values)
    magnitude_floor = 10.0 ** (min_db / 20.0)
    magnitude = complex_values.abs().clamp_min(magnitude_floor)
    return 20.0 * torch.log10(magnitude)


# ---------------------------------------------------------------------------
# Метрики для выхода в децибелах
# ---------------------------------------------------------------------------

def mae_db(
    prediction_db: Tensor,
    target_db: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    MAE в децибелах — средняя абсолютная ошибка АЧХ.

    Показывает среднюю величину отклонения предсказанного спектра
    от эталонного в дБ. Хорошо интерпретируется физически.

    Возвращает:
        reduction='mean' — одно среднее значение по тестовой выборке;
        reduction='none' — значение для каждого примера батча.
    """
    _validate_same_shape(prediction_db, target_db)
    error = _flatten_per_sample(prediction_db - target_db)
    per_sample = error.abs().mean(dim=1)
    return _reduce(per_sample, reduction)


def rmse_db(
    prediction_db: Tensor,
    target_db: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    RMSE в децибелах — среднеквадратичная ошибка АЧХ.

    Сильнее MAE штрафует большие локальные ошибки, например плохо
    восстановленные резонансные пики или глубокие провалы.
    """
    _validate_same_shape(prediction_db, target_db)
    error = _flatten_per_sample(prediction_db - target_db)
    per_sample = error.square().mean(dim=1).sqrt()
    return _reduce(per_sample, reduction)


def max_abs_error_db(
    prediction_db: Tensor,
    target_db: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    MaxAE в децибелах — максимальная абсолютная ошибка по спектру.

    Показывает худшее локальное отклонение на частотной сетке для каждого
    примера. При reduction='mean' усредняет эти худшие ошибки по батчу.
    """
    _validate_same_shape(prediction_db, target_db)
    error = _flatten_per_sample(prediction_db - target_db)
    per_sample = error.abs().amax(dim=1)
    return _reduce(per_sample, reduction)


# ---------------------------------------------------------------------------
# Метрики для комплексного выхода
# ---------------------------------------------------------------------------

def relative_complex_l2(
    prediction: Tensor,
    target: Tensor,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Относительная комплексная L2-ошибка.

    Формула для каждого примера:
        ||H_pred - H_true||_2 / (||H_true||_2 + eps)

    Одновременно учитывает действительную и мнимую части передаточной функции.
    Это основная интегральная метрика для комплексного выхода.
    """
    _validate_same_shape(prediction, target)
    pred_complex = _to_complex_tensor(prediction)
    target_complex = _to_complex_tensor(target)

    error_norm = torch.linalg.vector_norm(
        _flatten_per_sample(pred_complex - target_complex),
        ord=2,
        dim=1,
    )
    target_norm = torch.linalg.vector_norm(
        _flatten_per_sample(target_complex),
        ord=2,
        dim=1,
    )
    per_sample = error_norm / target_norm.clamp_min(eps)
    return _reduce(per_sample, reduction)


def relative_complex_l2_percent(
    prediction: Tensor,
    target: Tensor,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Относительная комплексная L2-ошибка в процентах.

    Это та же метрика, что relative_complex_l2, умноженная на 100.
    Удобна для таблиц диплома.
    """
    return 100.0 * relative_complex_l2(
        prediction=prediction,
        target=target,
        eps=eps,
        reduction=reduction,
    )


def magnitude_mae_db_from_complex(
    prediction: Tensor,
    target: Tensor,
    min_db: float = -100.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    MAE модуля в децибелах для комплексного выхода.

    Сначала Re/Im переводятся в |H| и затем в дБ, после чего считается MAE.
    Показывает качество амплитудно-частотной характеристики.
    """
    prediction_db = complex_ri_to_db(prediction, min_db=min_db)
    target_db = complex_ri_to_db(target, min_db=min_db)
    return mae_db(prediction_db, target_db, reduction=reduction)


def magnitude_rmse_db_from_complex(
    prediction: Tensor,
    target: Tensor,
    min_db: float = -100.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    RMSE модуля в децибелах для комплексного выхода.

    Сильнее штрафует большие ошибки АЧХ, чем magnitude_mae_db_from_complex.
    """
    prediction_db = complex_ri_to_db(prediction, min_db=min_db)
    target_db = complex_ri_to_db(target, min_db=min_db)
    return rmse_db(prediction_db, target_db, reduction=reduction)


def magnitude_max_abs_error_db_from_complex(
    prediction: Tensor,
    target: Tensor,
    min_db: float = -100.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Максимальная абсолютная ошибка модуля в децибелах.

    Показывает худшее локальное отличие комплексного предсказания от эталона
    после перехода к амплитудно-частотной характеристике.
    """
    prediction_db = complex_ri_to_db(prediction, min_db=min_db)
    target_db = complex_ri_to_db(target, min_db=min_db)
    return max_abs_error_db(prediction_db, target_db, reduction=reduction)


def phase_mae_degrees(
    prediction: Tensor,
    target: Tensor,
    dynamic_range_db: float = 40.0,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Средняя циклическая ошибка фазы в градусах.

    Разность фаз вычисляется через arg(H_pred * conj(H_true)), поэтому переход
    между -pi и +pi обрабатывается корректно.

    Частоты, где эталонный модуль ниже максимума более чем на
    dynamic_range_db, исключаются: около глубоких провалов фаза плохо
    определена и её ошибка физически малоинформативна.
    """
    _validate_same_shape(prediction, target)

    pred_complex = _to_complex_tensor(prediction)
    target_complex = _to_complex_tensor(target)

    phase_error_rad = torch.angle(
        pred_complex * torch.conj(target_complex)
    ).abs()

    target_magnitude = target_complex.abs()

    # Максимум по частоте отдельно для каждого примера и каждого выхода.
    maximum = target_magnitude.amax(dim=1, keepdim=True).clamp_min(eps)
    relative_db = 20.0 * torch.log10(
        (target_magnitude / maximum).clamp_min(eps)
    )
    mask = relative_db >= -dynamic_range_db

    # Все измерения, кроме батча, объединяются.
    masked_error = (phase_error_rad * mask).flatten(start_dim=1)
    valid_count = mask.flatten(start_dim=1).sum(dim=1).clamp_min(1)

    per_sample_rad = masked_error.sum(dim=1) / valid_count
    per_sample_deg = torch.rad2deg(per_sample_rad)
    return _reduce(per_sample_deg, reduction)


# ---------------------------------------------------------------------------
# Метрики формы спектра
# ---------------------------------------------------------------------------

def relative_frequency_derivative_l2(
    prediction: Tensor,
    target: Tensor,
    frequencies: Tensor,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Относительная L2-ошибка производной по частоте.

    Формула:
        ||D_f H_pred - D_f H_true||_2 /
        (||D_f H_true||_2 + eps)

    где
        D_f H_j = (H_{j+1} - H_j) / (f_{j+1} - f_j).

    Метрика оценивает качество формы спектра: наклоны, резонансные переходы
    и способность модели не сглаживать узкие особенности.
    """
    _validate_same_shape(prediction, target)
    frequencies = _normalize_frequencies(frequencies, prediction.shape[1])

    df = frequencies[1:] - frequencies[:-1]
    df_shape = [1, df.numel()] + [1] * (prediction.ndim - 2)
    df = df.view(*df_shape).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )

    pred_derivative = (prediction[:, 1:] - prediction[:, :-1]) / df
    target_derivative = (target[:, 1:] - target[:, :-1]) / df

    numerator = torch.linalg.vector_norm(
        _flatten_per_sample(pred_derivative - target_derivative),
        ord=2,
        dim=1,
    )
    denominator = torch.linalg.vector_norm(
        _flatten_per_sample(target_derivative),
        ord=2,
        dim=1,
    )

    per_sample = numerator / denominator.clamp_min(eps)
    return _reduce(per_sample, reduction)


def derivative_mse(
    prediction: Tensor,
    target: Tensor,
    frequencies: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Абсолютная среднеквадратичная ошибка производной по частоте.

    В отличие от relative_frequency_derivative_l2 эта величина не нормирована
    на эталонную производную. Она зависит от масштаба H и единиц частоты.
    Чаще используется как дополнительный training loss, а не как основная
    метрика сравнения разных задач.
    """
    _validate_same_shape(prediction, target)
    frequencies = _normalize_frequencies(frequencies, prediction.shape[1])

    df = frequencies[1:] - frequencies[:-1]
    df_shape = [1, df.numel()] + [1] * (prediction.ndim - 2)
    df = df.view(*df_shape).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )

    pred_derivative = (prediction[:, 1:] - prediction[:, :-1]) / df
    target_derivative = (target[:, 1:] - target[:, :-1]) / df

    error = _flatten_per_sample(pred_derivative - target_derivative)
    per_sample = error.square().mean(dim=1)
    return _reduce(per_sample, reduction)


# ---------------------------------------------------------------------------
# Метрики доминирующего резонанса и антирезонанса
# ---------------------------------------------------------------------------

def dominant_peak_frequency_mae_hz(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> Tensor:
    """
    Средняя абсолютная ошибка частоты доминирующего резонанса, Гц.

    Доминирующий резонанс определяется как глобальный максимум АЧХ
    в рассматриваемом частотном диапазоне.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )
    pred_idx = pred.argmax(dim=-1)
    target_idx = target.argmax(dim=-1)
    error = (frequencies[pred_idx] - frequencies[target_idx]).abs()
    return error.mean()


def dominant_peak_frequency_relative_mae_percent(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Относительная ошибка частоты доминирующего резонанса, %.

    Нормирует абсолютное смещение резонанса на истинную резонансную частоту.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )
    pred_idx = pred.argmax(dim=-1)
    target_idx = target.argmax(dim=-1)

    pred_frequency = frequencies[pred_idx]
    target_frequency = frequencies[target_idx]

    error_percent = (
        (pred_frequency - target_frequency).abs()
        / target_frequency.abs().clamp_min(eps)
        * 100.0
    )
    return error_percent.mean()


def dominant_peak_level_mae_db(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> Tensor:
    """
    Средняя абсолютная ошибка уровня доминирующего резонанса, дБ.

    Сравнивает высоту максимального пика предсказанной и эталонной АЧХ.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )
    pred_level = pred.amax(dim=-1)
    target_level = target.amax(dim=-1)
    return (pred_level - target_level).abs().mean()


def dominant_notch_frequency_mae_hz(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> Tensor:
    """
    Средняя абсолютная ошибка частоты доминирующего антирезонанса, Гц.

    Доминирующий антирезонанс определяется как глобальный минимум АЧХ
    в рассматриваемом диапазоне.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )
    pred_idx = pred.argmin(dim=-1)
    target_idx = target.argmin(dim=-1)
    error = (frequencies[pred_idx] - frequencies[target_idx]).abs()
    return error.mean()


def dominant_notch_level_mae_db(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> Tensor:
    """
    Средняя абсолютная ошибка глубины доминирующего антирезонанса, дБ.

    Сравнивает минимальные уровни предсказанной и эталонной АЧХ.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )
    pred_level = pred.amin(dim=-1)
    target_level = target.amin(dim=-1)
    return (pred_level - target_level).abs().mean()


def _curves_with_frequency_last(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Приводит АЧХ к форме [число_кривых, Nf].

    Вход может иметь форму [B, Nf] или [B, Nf, Nout].
    """
    _validate_same_shape(prediction_db, target_db)
    frequencies = _normalize_frequencies(frequencies, prediction_db.shape[1])

    pred = prediction_db.movedim(1, -1)
    target = target_db.movedim(1, -1)

    n_frequencies = frequencies.numel()
    return (
        pred.reshape(-1, n_frequencies),
        target.reshape(-1, n_frequencies),
    )


# ---------------------------------------------------------------------------
# Ширина резонанса и добротность
# ---------------------------------------------------------------------------

def _dominant_peak_bandwidth_3db(
    curve_db: Tensor,
    frequencies: Tensor,
) -> tuple[float, float]:
    """
    Находит частоту доминирующего пика и его ширину по уровню -3 дБ.

    Возвращает:
        peak_frequency_hz,
        bandwidth_hz

    Если пересечение уровня -3 дБ отсутствует с одной из сторон,
    bandwidth_hz возвращается как NaN.
    """
    y = curve_db.detach().cpu().double().numpy()
    f = frequencies.detach().cpu().double().numpy()

    peak_index = int(np.argmax(y))
    peak_level = float(y[peak_index])
    threshold = peak_level - 3.0

    left_frequency = math.nan
    for i in range(peak_index - 1, -1, -1):
        if y[i] <= threshold <= y[i + 1] or y[i] >= threshold >= y[i + 1]:
            denom = y[i + 1] - y[i]
            if abs(denom) < 1e-15:
                left_frequency = float(f[i])
            else:
                alpha = (threshold - y[i]) / denom
                left_frequency = float(f[i] + alpha * (f[i + 1] - f[i]))
            break

    right_frequency = math.nan
    for i in range(peak_index, len(y) - 1):
        if y[i] >= threshold >= y[i + 1] or y[i] <= threshold <= y[i + 1]:
            denom = y[i + 1] - y[i]
            if abs(denom) < 1e-15:
                right_frequency = float(f[i + 1])
            else:
                alpha = (threshold - y[i]) / denom
                right_frequency = float(f[i] + alpha * (f[i + 1] - f[i]))
            break

    bandwidth = right_frequency - left_frequency
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        bandwidth = math.nan

    return float(f[peak_index]), bandwidth


def dominant_peak_bandwidth_mae_hz(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> Tensor:
    """
    Средняя абсолютная ошибка ширины доминирующего резонанса по уровню -3 дБ.

    Характеризует, насколько правильно модель воспроизводит ширину пика,
    затухание и резонансную полосу. Кривые без двух пересечений уровня -3 дБ
    исключаются из усреднения.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )

    errors: list[float] = []
    for pred_curve, target_curve in zip(pred, target):
        _, pred_bw = _dominant_peak_bandwidth_3db(pred_curve, frequencies)
        _, target_bw = _dominant_peak_bandwidth_3db(target_curve, frequencies)
        if np.isfinite(pred_bw) and np.isfinite(target_bw):
            errors.append(abs(pred_bw - target_bw))

    if not errors:
        return torch.tensor(
            float("nan"),
            device=prediction_db.device,
            dtype=prediction_db.dtype,
        )

    return torch.tensor(
        float(np.mean(errors)),
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


def dominant_peak_quality_factor_relative_mae_percent(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Средняя относительная ошибка добротности доминирующего резонанса, %.

    Добротность определяется как:
        Q = f_res / bandwidth_3dB

    Кривые без корректно определяемой ширины -3 дБ исключаются.
    """
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )

    errors: list[float] = []
    for pred_curve, target_curve in zip(pred, target):
        pred_fr, pred_bw = _dominant_peak_bandwidth_3db(
            pred_curve, frequencies
        )
        target_fr, target_bw = _dominant_peak_bandwidth_3db(
            target_curve, frequencies
        )

        if (
            np.isfinite(pred_bw)
            and np.isfinite(target_bw)
            and pred_bw > 0.0
            and target_bw > 0.0
        ):
            pred_q = pred_fr / pred_bw
            target_q = target_fr / target_bw
            errors.append(
                abs(pred_q - target_q) / max(abs(target_q), eps) * 100.0
            )

    if not errors:
        return torch.tensor(
            float("nan"),
            device=prediction_db.device,
            dtype=prediction_db.dtype,
        )

    return torch.tensor(
        float(np.mean(errors)),
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


# ---------------------------------------------------------------------------
# Метрики нескольких резонансов
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeakMatchResult:
    """Результат поиска и сопоставления резонансных пиков одной пары кривых."""

    target_count: int
    prediction_count: int
    matched_target_indices: np.ndarray
    matched_prediction_indices: np.ndarray
    missed_count: int
    false_count: int


def _import_scipy_peak_tools() -> tuple[Any, Any]:
    """Импортирует SciPy только при использовании многопиковых метрик."""
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError(
            "Для многопиковых метрик требуется SciPy: pip install scipy"
        ) from exc
    return find_peaks, linear_sum_assignment


def _match_resonances_single(
    prediction_curve_db: np.ndarray,
    target_curve_db: np.ndarray,
    frequencies_hz: np.ndarray,
    prominence_db: float,
    min_distance_bins: int,
    max_match_distance_hz: float,
) -> PeakMatchResult:
    """
    Находит резонансы и сопоставляет их по минимальному расстоянию частот.

    Используется внутри функций многопиковых метрик.
    """
    find_peaks, linear_sum_assignment = _import_scipy_peak_tools()

    pred_indices, _ = find_peaks(
        prediction_curve_db,
        prominence=prominence_db,
        distance=min_distance_bins,
    )
    target_indices, _ = find_peaks(
        target_curve_db,
        prominence=prominence_db,
        distance=min_distance_bins,
    )

    if len(pred_indices) == 0 or len(target_indices) == 0:
        return PeakMatchResult(
            target_count=len(target_indices),
            prediction_count=len(pred_indices),
            matched_target_indices=np.empty(0, dtype=np.int64),
            matched_prediction_indices=np.empty(0, dtype=np.int64),
            missed_count=len(target_indices),
            false_count=len(pred_indices),
        )

    distance_matrix = np.abs(
        frequencies_hz[target_indices, None]
        - frequencies_hz[pred_indices][None, :]
    )

    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    valid = distance_matrix[row_indices, col_indices] <= max_match_distance_hz

    matched_target = target_indices[row_indices[valid]]
    matched_prediction = pred_indices[col_indices[valid]]

    matched_count = int(valid.sum())
    return PeakMatchResult(
        target_count=len(target_indices),
        prediction_count=len(pred_indices),
        matched_target_indices=matched_target,
        matched_prediction_indices=matched_prediction,
        missed_count=len(target_indices) - matched_count,
        false_count=len(pred_indices) - matched_count,
    )


def _all_peak_matches(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    prominence_db: float,
    min_distance_bins: int,
    max_match_distance_hz: float,
) -> list[tuple[PeakMatchResult, np.ndarray, np.ndarray, np.ndarray]]:
    """Готовит результаты сопоставления пиков для всех кривых."""
    pred, target = _curves_with_frequency_last(
        prediction_db, target_db, frequencies
    )

    pred_np = pred.detach().cpu().double().numpy()
    target_np = target.detach().cpu().double().numpy()
    frequencies_np = frequencies.detach().cpu().double().numpy()

    results = []
    for pred_curve, target_curve in zip(pred_np, target_np):
        match = _match_resonances_single(
            prediction_curve_db=pred_curve,
            target_curve_db=target_curve,
            frequencies_hz=frequencies_np,
            prominence_db=prominence_db,
            min_distance_bins=min_distance_bins,
            max_match_distance_hz=max_match_distance_hz,
        )
        results.append((match, pred_curve, target_curve, frequencies_np))
    return results


def matched_resonance_frequency_mae_hz(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    prominence_db: float = 3.0,
    min_distance_bins: int = 3,
    max_match_distance_hz: float = 100.0,
) -> Tensor:
    """
    MAE частот всех сопоставленных резонансов, Гц.

    Пики сначала находятся по prominence, затем пары формируются по
    минимальному расстоянию частот. Пары дальше max_match_distance_hz
    считаются несопоставленными.
    """
    errors: list[float] = []
    for match, _, _, f in _all_peak_matches(
        prediction_db,
        target_db,
        frequencies,
        prominence_db,
        min_distance_bins,
        max_match_distance_hz,
    ):
        if match.matched_target_indices.size:
            errors.extend(
                np.abs(
                    f[match.matched_prediction_indices]
                    - f[match.matched_target_indices]
                ).tolist()
            )

    value = float(np.mean(errors)) if errors else float("nan")
    return torch.tensor(
        value,
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


def matched_resonance_level_mae_db(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    prominence_db: float = 3.0,
    min_distance_bins: int = 3,
    max_match_distance_hz: float = 100.0,
) -> Tensor:
    """
    MAE уровней всех сопоставленных резонансов, дБ.

    Оценивает, насколько точно модель восстанавливает высоту найденных пиков.
    """
    errors: list[float] = []
    for match, pred_curve, target_curve, _ in _all_peak_matches(
        prediction_db,
        target_db,
        frequencies,
        prominence_db,
        min_distance_bins,
        max_match_distance_hz,
    ):
        if match.matched_target_indices.size:
            errors.extend(
                np.abs(
                    pred_curve[match.matched_prediction_indices]
                    - target_curve[match.matched_target_indices]
                ).tolist()
            )

    value = float(np.mean(errors)) if errors else float("nan")
    return torch.tensor(
        value,
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


def missed_resonance_rate(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    prominence_db: float = 3.0,
    min_distance_bins: int = 3,
    max_match_distance_hz: float = 100.0,
) -> Tensor:
    """
    Доля пропущенных резонансов.

    Формула:
        число несопоставленных эталонных пиков /
        общее число эталонных пиков

    Значение 0 означает, что все эталонные резонансы найдены.
    """
    missed = 0
    target_total = 0

    for match, _, _, _ in _all_peak_matches(
        prediction_db,
        target_db,
        frequencies,
        prominence_db,
        min_distance_bins,
        max_match_distance_hz,
    ):
        missed += match.missed_count
        target_total += match.target_count

    value = missed / target_total if target_total > 0 else float("nan")
    return torch.tensor(
        value,
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


def false_resonance_rate(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
    prominence_db: float = 3.0,
    min_distance_bins: int = 3,
    max_match_distance_hz: float = 100.0,
) -> Tensor:
    """
    Доля ложных резонансов.

    Формула:
        число несопоставленных предсказанных пиков /
        общее число предсказанных пиков

    Значение 0 означает отсутствие лишних резонансных пиков.
    """
    false = 0
    prediction_total = 0

    for match, _, _, _ in _all_peak_matches(
        prediction_db,
        target_db,
        frequencies,
        prominence_db,
        min_distance_bins,
        max_match_distance_hz,
    ):
        false += match.false_count
        prediction_total += match.prediction_count

    value = (
        false / prediction_total
        if prediction_total > 0
        else float("nan")
    )
    return torch.tensor(
        value,
        device=prediction_db.device,
        dtype=prediction_db.dtype,
    )


# ---------------------------------------------------------------------------
# Метрики вычислительной эффективности
# ---------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    """
    Число обучаемых параметров модели.

    Используется для сравнения сложности FNO и MLP.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def inference_time_ms(
    model: nn.Module,
    model_inputs: Sequence[Any],
    warmup_runs: int = 20,
    measured_runs: int = 100,
) -> float:
    """
    Среднее время одного forward-прохода в миллисекундах.

    Перед измерением выполняются прогревочные запуски. Для CUDA используются
    synchronize-вызовы, чтобы измерять фактическое время вычислений.
    """
    if measured_runs <= 0:
        raise ValueError("measured_runs должен быть положительным.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs не может быть отрицательным.")

    model.eval()
    uses_cuda = any(
        isinstance(value, Tensor) and value.is_cuda
        for value in model_inputs
    )

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(*model_inputs)

        if uses_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(measured_runs):
            model(*model_inputs)

        if uses_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / measured_runs


def throughput_samples_per_second(
    model: nn.Module,
    model_inputs: Sequence[Any],
    batch_size: int,
    warmup_runs: int = 20,
    measured_runs: int = 100,
) -> float:
    """
    Пропускная способность модели, примеров в секунду.

    Рассчитывается по среднему времени forward-прохода и размеру батча.
    """
    if batch_size <= 0:
        raise ValueError("batch_size должен быть положительным.")

    time_ms = inference_time_ms(
        model=model,
        model_inputs=model_inputs,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    return batch_size / (time_ms / 1000.0)


def peak_gpu_memory_mb(
    model: nn.Module,
    model_inputs: Sequence[Any],
) -> float:
    """
    Пиковое дополнительное потребление GPU-памяти при одном forward-проходе.

    Возвращает значение в мегабайтах. Работает только для CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна.")

    uses_cuda = any(
        isinstance(value, Tensor) and value.is_cuda
        for value in model_inputs
    )
    if not uses_cuda:
        raise ValueError("Хотя бы один входной тензор должен находиться на CUDA.")

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    with torch.inference_mode():
        model(*model_inputs)
        torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - baseline) / (1024.0 ** 2)


# ---------------------------------------------------------------------------
# Удобные наборы метрик
# ---------------------------------------------------------------------------

def evaluate_db_output(
    prediction_db: Tensor,
    target_db: Tensor,
    frequencies: Tensor,
) -> dict[str, float]:
    """
    Вычисляет рекомендуемый базовый набор метрик для выхода в децибелах.

    Возвращает обычный dict, удобный для логирования и таблиц.
    """
    return {
        "mae_db": float(mae_db(prediction_db, target_db)),
        "rmse_db": float(rmse_db(prediction_db, target_db)),
        "max_abs_error_db": float(
            max_abs_error_db(prediction_db, target_db)
        ),
        "relative_derivative_l2": float(
            relative_frequency_derivative_l2(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_peak_frequency_mae_hz": float(
            dominant_peak_frequency_mae_hz(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_peak_level_mae_db": float(
            dominant_peak_level_mae_db(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_notch_frequency_mae_hz": float(
            dominant_notch_frequency_mae_hz(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_notch_level_mae_db": float(
            dominant_notch_level_mae_db(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
    }


def evaluate_complex_output(
    prediction: Tensor,
    target: Tensor,
    frequencies: Tensor,
    min_db: float = -100.0,
    phase_dynamic_range_db: float = 40.0,
) -> dict[str, float]:
    """
    Вычисляет рекомендуемый базовый набор метрик для комплексного выхода Re/Im.

    Включает общую комплексную ошибку, ошибки модуля в дБ, фазу,
    производную по частоте и характеристики доминирующих экстремумов.
    """
    prediction_db = complex_ri_to_db(prediction, min_db=min_db)
    target_db = complex_ri_to_db(target, min_db=min_db)

    return {
        "relative_complex_l2_percent": float(
            relative_complex_l2_percent(prediction, target)
        ),
        "magnitude_mae_db": float(
            magnitude_mae_db_from_complex(
                prediction,
                target,
                min_db=min_db,
            )
        ),
        "magnitude_rmse_db": float(
            magnitude_rmse_db_from_complex(
                prediction,
                target,
                min_db=min_db,
            )
        ),
        "magnitude_max_abs_error_db": float(
            magnitude_max_abs_error_db_from_complex(
                prediction,
                target,
                min_db=min_db,
            )
        ),
        "phase_mae_degrees": float(
            phase_mae_degrees(
                prediction,
                target,
                dynamic_range_db=phase_dynamic_range_db,
            )
        ),
        "relative_complex_derivative_l2": float(
            relative_frequency_derivative_l2(
                prediction,
                target,
                frequencies,
            )
        ),
        "dominant_peak_frequency_mae_hz": float(
            dominant_peak_frequency_mae_hz(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_peak_level_mae_db": float(
            dominant_peak_level_mae_db(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_notch_frequency_mae_hz": float(
            dominant_notch_frequency_mae_hz(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
        "dominant_notch_level_mae_db": float(
            dominant_notch_level_mae_db(
                prediction_db,
                target_db,
                frequencies,
            )
        ),
    }


__all__ = [
    # Преобразования.
    "complex_ri_to_complex",
    "complex_ri_to_db",

    # АЧХ в дБ.
    "mae_db",
    "rmse_db",
    "max_abs_error_db",

    # Комплексный выход.
    "relative_complex_l2",
    "relative_complex_l2_percent",
    "magnitude_mae_db_from_complex",
    "magnitude_rmse_db_from_complex",
    "magnitude_max_abs_error_db_from_complex",
    "phase_mae_degrees",

    # Форма спектра.
    "relative_frequency_derivative_l2",
    "derivative_mse",

    # Доминирующий резонанс/антирезонанс.
    "dominant_peak_frequency_mae_hz",
    "dominant_peak_frequency_relative_mae_percent",
    "dominant_peak_level_mae_db",
    "dominant_notch_frequency_mae_hz",
    "dominant_notch_level_mae_db",

    # Ширина и добротность.
    "dominant_peak_bandwidth_mae_hz",
    "dominant_peak_quality_factor_relative_mae_percent",

    # Несколько резонансов.
    "matched_resonance_frequency_mae_hz",
    "matched_resonance_level_mae_db",
    "missed_resonance_rate",
    "false_resonance_rate",

    # Вычислительная эффективность.
    "count_trainable_parameters",
    "inference_time_ms",
    "throughput_samples_per_second",
    "peak_gpu_memory_mb",

    # Готовые наборы.
    "evaluate_db_output",
    "evaluate_complex_output",
]
