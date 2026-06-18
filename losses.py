from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OutputType = Literal["db", "complex"]


class UniversalTransferFunctionLoss(nn.Module):
    """
    РЈРЅРёРІРµСЂСЃР°Р»СЊРЅС‹Р№ СЃРѕСЃС‚Р°РІРЅРѕР№ loss РґР»СЏ РїРµСЂРµРґР°С‚РѕС‡РЅРѕР№ С„СѓРЅРєС†РёРё.

    output_type="db":
        prediction, target:
            [B, Nf]
            [B, Nf, Nout]

    output_type="complex":
        prediction, target:
            [B, Nf, 2]
            [B, Nf, Nout, 2]

        РџРѕСЃР»РµРґРЅРµРµ РёР·РјРµСЂРµРЅРёРµ:
            [..., 0] = Re(H)
            [..., 1] = Im(H)

    РћР±С‰РёРµ РєРѕРјРїРѕРЅРµРЅС‚С‹:
        1. Smooth L1 РїРѕ РђР§РҐ РІ dB.
        2. РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ L2-РѕС€РёР±РєР° Р»РёРЅРµР№РЅРѕРіРѕ РјРѕРґСѓР»СЏ.
        3. РћС€РёР±РєР° РїСЂРѕРёР·РІРѕРґРЅРѕР№ РђР§РҐ РїРѕ С‡Р°СЃС‚РѕС‚Рµ.
        4. Р”РёС„С„РµСЂРµРЅС†РёСЂСѓРµРјР°СЏ РѕС€РёР±РєР° РїРѕР»РѕР¶РµРЅРёСЏ РґРѕРјРёРЅРёСЂСѓСЋС‰РµРіРѕ РїРёРєР°.

    РўРѕР»СЊРєРѕ РґР»СЏ РєРѕРјРїР»РµРєСЃРЅРѕРіРѕ РІС‹С…РѕРґР°:
        5. РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ РєРѕРјРїР»РµРєСЃРЅР°СЏ L2-РѕС€РёР±РєР°.
        6. РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ РѕС€РёР±РєР° РєРѕРјРїР»РµРєСЃРЅРѕР№ РїСЂРѕРёР·РІРѕРґРЅРѕР№.
        7. Р¦РёРєР»РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° С„Р°Р·С‹.
    """

    def __init__(
        self,
        output_type: OutputType,

        # РћР±С‰РёРµ РєРѕРјРїРѕРЅРµРЅС‚С‹.
        db_weight: float = 1.0,
        magnitude_weight: float = 0.1,
        db_derivative_weight: float = 0.05,
        peak_weight: float = 0.0,

        # РўРѕР»СЊРєРѕ complex.
        complex_weight: float = 1.0,
        complex_derivative_weight: float = 0.05,
        phase_weight: float = 0.05,

        # РњР°СЃС€С‚Р°Р±С‹ Рё С‡РёСЃР»РµРЅРЅР°СЏ СѓСЃС‚РѕР№С‡РёРІРѕСЃС‚СЊ.
        db_error_scale: float = 10.0,
        db_derivative_scale: float = 0.01,
        peak_temperature_db: float = 3.0,
        phase_dynamic_range_db: float = 40.0,
        min_db: float = -100.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if output_type not in ("db", "complex"):
            raise ValueError(
                "output_type РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ 'db' РёР»Рё 'complex'"
            )

        self.output_type = output_type

        self.db_weight = db_weight
        self.magnitude_weight = magnitude_weight
        self.db_derivative_weight = db_derivative_weight
        self.peak_weight = peak_weight

        self.complex_weight = complex_weight
        self.complex_derivative_weight = complex_derivative_weight
        self.phase_weight = phase_weight

        self.db_error_scale = db_error_scale
        self.db_derivative_scale = db_derivative_scale
        self.peak_temperature_db = peak_temperature_db
        self.phase_dynamic_range_db = phase_dynamic_range_db
        self.min_db = min_db
        self.eps = eps

    def _validate(
        self,
        prediction: Tensor,
        target: Tensor,
        frequencies: Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes do not match: {tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        frequencies = self._normalize_frequencies(
            frequencies,
            prediction.shape[1],
        )

        if prediction.shape[1] != frequencies.numel():
            raise ValueError("Frequency axis length does not match frequencies.")

        if torch.any(frequencies[1:] <= frequencies[:-1]):
            raise ValueError("Frequencies must be strictly increasing.")

        if self.output_type == "complex":
            if prediction.ndim < 3 or prediction.shape[-1] != 2:
                raise ValueError(
                    "Complex output must have shape [B, Nf, 2] or [B, Nf, Nout, 2]."
                )

    def _normalize_frequencies(
        self,
        frequencies: Tensor,
        n_frequencies: int,
    ) -> Tensor:
        if frequencies.ndim == 1:
            normalized = frequencies
        elif frequencies.ndim == 2:
            if frequencies.shape[1] != n_frequencies:
                raise ValueError(
                    f"Expected {n_frequencies} frequencies per row, got {frequencies.shape[1]}."
                )
            if frequencies.shape[0] == 0:
                raise ValueError("frequencies must not be empty.")
            normalized = frequencies[0]
            if not torch.allclose(
                frequencies,
                normalized.unsqueeze(0).expand_as(frequencies),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(
                    "Batched frequencies must use the same frequency grid in every sample."
                )
        else:
            raise ValueError("frequencies must have shape [Nf] or [B, Nf].")

        if normalized.numel() != n_frequencies:
            raise ValueError(
                f"Expected {n_frequencies} frequencies, got {normalized.numel()}."
            )
        if torch.any(normalized[1:] <= normalized[:-1]):
            raise ValueError("Frequencies must be strictly increasing.")

        return normalized

    def _complex_magnitude(self, values: Tensor) -> Tensor:
        """РњРѕРґСѓР»СЊ РєРѕРјРїР»РµРєСЃРЅРѕРіРѕ С‡РёСЃР»Р° РёР· РєР°РЅР°Р»РѕРІ Re/Im."""
        return torch.sqrt(
            values[..., 0].square()
            + values[..., 1].square()
            + self.eps
        )

    def _db_to_magnitude(self, values_db: Tensor) -> Tensor:
        """РџРµСЂРµРІРѕРґРёС‚ dB РІ Р»РёРЅРµР№РЅС‹Р№ РјРѕРґСѓР»СЊ."""
        return torch.pow(10.0, values_db / 20.0)

    def _magnitude_to_db(self, magnitude: Tensor) -> Tensor:
        """РџРµСЂРµРІРѕРґРёС‚ Р»РёРЅРµР№РЅС‹Р№ РјРѕРґСѓР»СЊ РІ dB."""
        floor = 10.0 ** (self.min_db / 20.0)

        return 20.0 * torch.log10(
            magnitude.clamp_min(floor)
        )

    def _extract_db_and_magnitude(
        self,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Р’РѕР·РІСЂР°С‰Р°РµС‚ РґР»СЏ Р»СЋР±РѕРіРѕ С„РѕСЂРјР°С‚Р°:
            values_db вЂ” РђР§РҐ РІ dB;
            magnitude вЂ” Р»РёРЅРµР№РЅС‹Р№ РјРѕРґСѓР»СЊ.
        """
        if self.output_type == "db":
            values_db = values
            magnitude = self._db_to_magnitude(values_db)
            return values_db, magnitude

        magnitude = self._complex_magnitude(values)
        values_db = self._magnitude_to_db(magnitude)

        return values_db, magnitude

    def _frequency_step(
        self,
        values: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """Р¤РѕСЂРјРёСЂСѓРµС‚ df СЃ РЅСѓР¶РЅС‹Рј С‡РёСЃР»РѕРј РёР·РјРµСЂРµРЅРёР№."""
        frequencies = self._normalize_frequencies(
            frequencies,
            values.shape[1],
        )
        df = frequencies[1:] - frequencies[:-1]

        shape = [1, df.numel()] + [1] * (values.ndim - 2)

        return df.view(*shape).to(
            device=values.device,
            dtype=values.dtype,
        )

    def db_loss(
        self,
        prediction_db: Tensor,
        target_db: Tensor,
    ) -> Tensor:
        """
        Smooth L1 РїРѕ Р·РЅР°С‡РµРЅРёСЏРј РђР§РҐ РІ dB.

        РљРѕРЅС‚СЂРѕР»РёСЂСѓРµС‚ РѕР±С‰РµРµ СЃРѕРІРїР°РґРµРЅРёРµ СЃРїРµРєС‚СЂРѕРІ РІ Р»РѕРіР°СЂРёС„РјРёС‡РµСЃРєРѕР№ С€РєР°Р»Рµ.
        """
        error = (
            prediction_db - target_db
        ) / self.db_error_scale

        return F.smooth_l1_loss(
            error,
            torch.zeros_like(error),
            beta=1.0,
        )

    def relative_magnitude_l2(
        self,
        prediction_magnitude: Tensor,
        target_magnitude: Tensor,
    ) -> Tensor:
        """
        РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ L2-РѕС€РёР±РєР° Р»РёРЅРµР№РЅРѕРіРѕ РјРѕРґСѓР»СЏ.

        РЎРёР»СЊРЅРµРµ dB-loss СѓС‡РёС‚С‹РІР°РµС‚ РѕР±Р»Р°СЃС‚Рё СЃ РІС‹СЃРѕРєРѕР№ Р°РјРїР»РёС‚СѓРґРѕР№.
        """
        difference = (
            prediction_magnitude - target_magnitude
        ).flatten(start_dim=1)

        target_flat = target_magnitude.flatten(start_dim=1)

        numerator = torch.linalg.vector_norm(
            difference,
            dim=1,
        )

        denominator = torch.linalg.vector_norm(
            target_flat,
            dim=1,
        )

        return (
            numerator / denominator.clamp_min(self.eps)
        ).mean()

    def db_derivative_loss(
        self,
        prediction_db: Tensor,
        target_db: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """
        Smooth L1 РїРѕ РїСЂРѕРёР·РІРѕРґРЅРѕР№ РђР§РҐ РІ dB РїРѕ С‡Р°СЃС‚РѕС‚Рµ.

        РљРѕРЅС‚СЂРѕР»РёСЂСѓРµС‚ РЅР°РєР»РѕРЅС‹, С„РѕСЂРјСѓ СЂРµР·РѕРЅР°РЅСЃРѕРІ Рё СЂРµР·РєРёРµ РїРµСЂРµС…РѕРґС‹.
        """
        df = self._frequency_step(
            prediction_db,
            frequencies,
        )

        prediction_derivative = (
            prediction_db[:, 1:] - prediction_db[:, :-1]
        ) / df

        target_derivative = (
            target_db[:, 1:] - target_db[:, :-1]
        ) / df

        error = (
            prediction_derivative - target_derivative
        ) / self.db_derivative_scale

        return F.smooth_l1_loss(
            error,
            torch.zeros_like(error),
            beta=1.0,
        )

    def soft_peak_loss(
        self,
        prediction_db: Tensor,
        target_db: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """
        Р”РёС„С„РµСЂРµРЅС†РёСЂСѓРµРјР°СЏ РѕС€РёР±РєР° РїРѕР»РѕР¶РµРЅРёСЏ РґРѕРјРёРЅРёСЂСѓСЋС‰РµРіРѕ РїРёРєР°.

        РСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ soft-argmax РІРјРµСЃС‚Рѕ РѕР±С‹С‡РЅРѕРіРѕ argmax.
        """
        frequencies = self._normalize_frequencies(
            frequencies,
            prediction_db.shape[1],
        )
        prediction_weights = torch.softmax(
            prediction_db / self.peak_temperature_db,
            dim=1,
        )

        target_weights = torch.softmax(
            target_db / self.peak_temperature_db,
            dim=1,
        )

        shape = (
            [1, frequencies.numel()]
            + [1] * (prediction_db.ndim - 2)
        )

        frequency_grid = frequencies.view(*shape).to(
            device=prediction_db.device,
            dtype=prediction_db.dtype,
        )

        prediction_peak = (
            prediction_weights * frequency_grid
        ).sum(dim=1)

        target_peak = (
            target_weights * frequency_grid
        ).sum(dim=1)

        frequency_range = (
            frequencies[-1] - frequencies[0]
        ).abs().clamp_min(self.eps)

        return (
            prediction_peak - target_peak
        ).abs().mean() / frequency_range

    def relative_complex_l2(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ РєРѕРјРїР»РµРєСЃРЅР°СЏ L2-РѕС€РёР±РєР°.

        РљРѕРЅС‚СЂРѕР»РёСЂСѓРµС‚ РѕРґРЅРѕРІСЂРµРјРµРЅРЅРѕ Re(H), Im(H), РјРѕРґСѓР»СЊ Рё С„Р°Р·Сѓ.
        """
        difference = (
            prediction - target
        ).flatten(start_dim=1)

        target_flat = target.flatten(start_dim=1)

        numerator = torch.linalg.vector_norm(
            difference,
            dim=1,
        )

        denominator = torch.linalg.vector_norm(
            target_flat,
            dim=1,
        )

        return (
            numerator / denominator.clamp_min(self.eps)
        ).mean()

    def relative_complex_derivative_l2(
        self,
        prediction: Tensor,
        target: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        """
        РћС‚РЅРѕСЃРёС‚РµР»СЊРЅР°СЏ РѕС€РёР±РєР° РєРѕРјРїР»РµРєСЃРЅРѕР№ РїСЂРѕРёР·РІРѕРґРЅРѕР№ dH/df.

        РљРѕРЅС‚СЂРѕР»РёСЂСѓРµС‚ РёР·РјРµРЅРµРЅРёРµ Re(H) Рё Im(H) РІРґРѕР»СЊ С‡Р°СЃС‚РѕС‚РЅРѕР№ РѕСЃРё.
        """
        df = self._frequency_step(
            prediction,
            frequencies,
        )

        prediction_derivative = (
            prediction[:, 1:] - prediction[:, :-1]
        ) / df

        target_derivative = (
            target[:, 1:] - target[:, :-1]
        ) / df

        numerator = torch.linalg.vector_norm(
            (
                prediction_derivative
                - target_derivative
            ).flatten(start_dim=1),
            dim=1,
        )

        denominator = torch.linalg.vector_norm(
            target_derivative.flatten(start_dim=1),
            dim=1,
        )

        return (
            numerator / denominator.clamp_min(self.eps)
        ).mean()

    def phase_loss(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Р¦РёРєР»РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° С„Р°Р·С‹: 1 - cos(delta_phi).

        РљРѕСЂСЂРµРєС‚РЅРѕ РѕР±СЂР°Р±Р°С‚С‹РІР°РµС‚ РїРµСЂРµС…РѕРґ РјРµР¶РґСѓ -pi Рё +pi.
        Р§Р°СЃС‚РѕС‚С‹ СЃ РѕС‡РµРЅСЊ РјР°Р»РѕР№ Р°РјРїР»РёС‚СѓРґРѕР№ РјР°СЃРєРёСЂСѓСЋС‚СЃСЏ.
        """
        prediction_complex = torch.complex(
            prediction[..., 0],
            prediction[..., 1],
        )

        target_complex = torch.complex(
            target[..., 0],
            target[..., 1],
        )

        product = (
            prediction_complex
            * torch.conj(target_complex)
        )

        cosine_difference = (
            product.real
            / (
                prediction_complex.abs()
                * target_complex.abs()
                + self.eps
            )
        ).clamp(-1.0, 1.0)

        circular_error = 1.0 - cosine_difference

        target_magnitude = target_complex.abs()

        maximum = target_magnitude.amax(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        target_relative_db = 20.0 * torch.log10(
            (target_magnitude / maximum).clamp_min(self.eps)
        )

        mask = (
            target_relative_db
            >= -self.phase_dynamic_range_db
        )

        weights = (
            mask.to(target_magnitude.dtype)
            * target_magnitude / maximum
        )

        return (
            circular_error * weights
        ).sum() / weights.sum().clamp_min(self.eps)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        frequencies: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        frequencies = self._normalize_frequencies(
            frequencies,
            prediction.shape[1],
        ).to(device=prediction.device, dtype=prediction.dtype)

        self._validate(
            prediction,
            target,
            frequencies,
        )

        prediction_db, prediction_magnitude = (
            self._extract_db_and_magnitude(prediction)
        )

        target_db, target_magnitude = (
            self._extract_db_and_magnitude(target)
        )

        components = {
            "db": self.db_loss(
                prediction_db,
                target_db,
            ),
            "magnitude": self.relative_magnitude_l2(
                prediction_magnitude,
                target_magnitude,
            ),
            "db_derivative": self.db_derivative_loss(
                prediction_db,
                target_db,
                frequencies,
            ),
            "peak": self.soft_peak_loss(
                prediction_db,
                target_db,
                frequencies,
            ),
        }

        total = (
            self.db_weight * components["db"]
            + self.magnitude_weight * components["magnitude"]
            + self.db_derivative_weight
            * components["db_derivative"]
            + self.peak_weight * components["peak"]
        )

        if self.output_type == "complex":
            components["complex"] = self.relative_complex_l2(
                prediction,
                target,
            )

            components["complex_derivative"] = (
                self.relative_complex_derivative_l2(
                    prediction,
                    target,
                    frequencies,
                )
            )

            components["phase"] = self.phase_loss(
                prediction,
                target,
            )

            total = (
                total
                + self.complex_weight * components["complex"]
                + self.complex_derivative_weight
                * components["complex_derivative"]
                + self.phase_weight * components["phase"]
            )

        components["total"] = total

        if not return_components:
            return total

        detached_components = {
            name: value.detach()
            for name, value in components.items()
        }

        return total, detached_components
