from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

import vt_all_solvers_wrapper as vt


TargetMode = Literal["magnitude", "db", "phase", "complex", "realimag"]


@dataclass(frozen=True)
class GeometryDatasetConfig:
    n_samples: int
    geometry_kind: vt.GeometryKind = "random"
    solver_config: vt.SolverConfig = field(default_factory=vt.SolverConfig)
    acoustics: vt.AcousticConfig = field(default_factory=vt.AcousticConfig)
    target_mode: TargetMode = "db"
    return_geometry: bool = True
    return_metadata: bool = True
    cache: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class StreamingGeometryDatasetConfig:
    geometry_kind: vt.GeometryKind = "random"
    solver_config: vt.SolverConfig = field(default_factory=vt.SolverConfig)
    acoustics: vt.AcousticConfig = field(default_factory=vt.AcousticConfig)
    target_mode: TargetMode = "db"
    return_geometry: bool = True
    return_metadata: bool = True
    seed: int | None = None
    start_index: int = 0
    max_samples: int | None = None


class WebsterTorchDataset(Dataset):
    """
    Генерирует случайные трубы через vt.make_geometry_from_range_library(...)
    и считает для них спектр через vt.solve(...).

    Возвращает словарь:
      sample["target"]          : torch.Tensor
      sample["frequencies_hz"]  : torch.Tensor [Nf]
      sample["geometry"]        : dict с x_m / area_m2 / segment_lengths (если return_geometry=True)
      sample["meta"]            : служебная информация (если return_metadata=True)
    """

    def __init__(
        self,
        config: GeometryDatasetConfig,
        range_library: dict[str, dict[str, object]],
    ) -> None:
        super().__init__()

        if config.n_samples < 1:
            raise ValueError("config.n_samples must be >= 1")
        if config.target_mode not in ("magnitude", "db", "phase", "complex", "realimag"):
            raise ValueError(f"Unsupported target_mode: {config.target_mode!r}")
        if not range_library:
            raise ValueError("range_library must not be empty")

        self.config = config
        self.range_library = range_library
        self._cache: dict[int, dict[str, Any]] | None = {} if config.cache else None

    def __len__(self) -> int:
        return self.config.n_samples

    def _sample_seed(self, idx: int) -> int | None:
        if self.config.seed is None:
            return None
        return self.config.seed + int(idx)

    def _make_target_tensor(self, result: vt.SpectrumResult) -> torch.Tensor:
        mode = self.config.target_mode

        if mode == "magnitude":
            return torch.tensor(result.magnitude, dtype=torch.float32)

        if mode == "db":
            values = [20.0 * torch.log10(torch.tensor(max(v, 1e-12), dtype=torch.float32)) for v in result.magnitude]
            return torch.stack(values)

        if mode == "phase":
            return torch.tensor(result.phase_rad, dtype=torch.float32)

        if mode == "complex":
            return torch.tensor(result.transfer_complex, dtype=torch.complex64)

        # realimag
        values = [
            [z.real, z.imag]
            for z in result.transfer_complex
        ]
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        sample_seed = self._sample_seed(idx)
        geometry = vt.make_geometry_from_range_library(
            self.config.geometry_kind,
            self.range_library,
            seed=sample_seed,
        )
        result = vt.solve(
            geometry=geometry,
            config=self.config.solver_config,
            acoustics=self.config.acoustics,
        )

        sample: dict[str, Any] = {
            "target": self._make_target_tensor(result),
            "frequencies_hz": torch.tensor(result.frequencies_hz, dtype=torch.float32),
        }

        if self.config.return_geometry:
            x_m, area_m2, segment_lengths = vt.geometry_to_tube_tuple(geometry)
            sample["geometry"] = {
                "x_m": torch.tensor(x_m, dtype=torch.float32),
                "area_m2": torch.tensor(area_m2, dtype=torch.float32),
                "segment_lengths_m": torch.tensor(segment_lengths, dtype=torch.float32),
            }

        if self.config.return_metadata:
            sample["meta"] = {
                "idx": int(idx),
                "seed": sample_seed,
                "solver": self.config.solver_config.solver,
                "geometry_kind_requested": self.config.geometry_kind,
            }

        if self._cache is not None:
            self._cache[idx] = sample

        return sample


class StreamingWebsterTorchDataset(IterableDataset):
    """
    Потоковая версия WebsterTorchDataset.

    Подходит для on-the-fly обучения: новые геометрии и спектры считаются
    при итерации, без фиксированного размера датасета и без хранения в памяти.
    Для бесконечного потока задайте max_samples=None, а в обучении используйте
    steps_per_epoch.
    """

    def __init__(
        self,
        config: StreamingGeometryDatasetConfig,
        range_library: dict[str, dict[str, object]],
    ) -> None:
        super().__init__()

        if config.target_mode not in ("magnitude", "db", "phase", "complex", "realimag"):
            raise ValueError(f"Unsupported target_mode: {config.target_mode!r}")
        if config.start_index < 0:
            raise ValueError("config.start_index must be >= 0")
        if config.max_samples is not None and config.max_samples < 1:
            raise ValueError("config.max_samples must be None or >= 1")
        if not range_library:
            raise ValueError("range_library must not be empty")

        self.config = config
        self.range_library = range_library

    def _sample_seed(self, idx: int) -> int | None:
        if self.config.seed is None:
            return None
        return self.config.seed + int(idx)

    def _make_target_tensor(self, result: vt.SpectrumResult) -> torch.Tensor:
        mode = self.config.target_mode

        if mode == "magnitude":
            return torch.tensor(result.magnitude, dtype=torch.float32)

        if mode == "db":
            values = [20.0 * torch.log10(torch.tensor(max(v, 1e-12), dtype=torch.float32)) for v in result.magnitude]
            return torch.stack(values)

        if mode == "phase":
            return torch.tensor(result.phase_rad, dtype=torch.float32)

        if mode == "complex":
            return torch.tensor(result.transfer_complex, dtype=torch.complex64)

        values = [
            [z.real, z.imag]
            for z in result.transfer_complex
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _make_sample(self, idx: int) -> dict[str, Any]:
        sample_seed = self._sample_seed(idx)
        geometry = vt.make_geometry_from_range_library(
            self.config.geometry_kind,
            self.range_library,
            seed=sample_seed,
        )
        result = vt.solve(
            geometry=geometry,
            config=self.config.solver_config,
            acoustics=self.config.acoustics,
        )

        sample: dict[str, Any] = {
            "target": self._make_target_tensor(result),
            "frequencies_hz": torch.tensor(result.frequencies_hz, dtype=torch.float32),
        }

        if self.config.return_geometry:
            x_m, area_m2, segment_lengths = vt.geometry_to_tube_tuple(geometry)
            sample["geometry"] = {
                "x_m": torch.tensor(x_m, dtype=torch.float32),
                "area_m2": torch.tensor(area_m2, dtype=torch.float32),
                "segment_lengths_m": torch.tensor(segment_lengths, dtype=torch.float32),
            }

        if self.config.return_metadata:
            sample["meta"] = {
                "idx": int(idx),
                "seed": sample_seed,
                "solver": self.config.solver_config.solver,
                "geometry_kind_requested": self.config.geometry_kind,
                "streaming": True,
            }

        return sample

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker.id
            worker_count = worker.num_workers

        idx = self.config.start_index + worker_id
        stop_index = None
        if self.config.max_samples is not None:
            stop_index = self.config.start_index + self.config.max_samples

        while stop_index is None or idx < stop_index:
            yield self._make_sample(idx)
            idx += worker_count


def collate_geometry_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty")

    out: dict[str, Any] = {
        "target": torch.stack([sample["target"] for sample in batch], dim=0),
        "frequencies_hz": torch.stack([sample["frequencies_hz"] for sample in batch], dim=0),
    }

    if "geometry" in batch[0]:
        xs = [sample["geometry"]["x_m"] for sample in batch]
        areas = [sample["geometry"]["area_m2"] for sample in batch]
        seg_lengths = [sample["geometry"]["segment_lengths_m"] for sample in batch]

        out["geometry"] = {
            "x_m": pad_sequence(xs, batch_first=True),
            "area_m2": pad_sequence(areas, batch_first=True),
            "segment_lengths_m": pad_sequence(seg_lengths, batch_first=True),
            "node_count": torch.tensor([len(v) for v in xs], dtype=torch.int64),
            "segment_count": torch.tensor([len(v) for v in seg_lengths], dtype=torch.int64),
        }

    if "meta" in batch[0]:
        out["meta"] = [sample["meta"] for sample in batch]

    return out


def make_dataloader(
    config: GeometryDatasetConfig,
    range_library: dict[str, dict[str, object]],
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = WebsterTorchDataset(config=config, range_library=range_library)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_geometry_batch,
    )


def make_streaming_dataloader(
    config: StreamingGeometryDatasetConfig,
    range_library: dict[str, dict[str, object]],
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = StreamingWebsterTorchDataset(config=config, range_library=range_library)
    kwargs: dict[str, Any] = {}
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_geometry_batch,
        **kwargs,
    )
