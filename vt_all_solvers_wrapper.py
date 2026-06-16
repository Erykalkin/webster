from __future__ import annotations

import argparse
import cmath
import csv
import math
import random as py_random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"
BIN_DIR = ROOT / "bin"
BINARY_PATH = BIN_DIR / "vt_all_solvers_cli.exe"

SolverKind = Literal["cylinder", "cone", "arma", "webster"]
GridKind = Literal["linear", "log"]
PlotKind = Literal["magnitude", "db", "phase"]
GeometryPlotKind = Literal["graph", "symmetric"]


@dataclass(frozen=True)
class AcousticConfig:
    rho_kg_m3: float = 1.225
    c_m_s: float = 343.0


@dataclass(frozen=True)
class SolverConfig:
    solver: SolverKind = "cylinder"
    sections: int = 20
    points: int = 256
    f_min_hz: float = 50.0
    f_max_hz: float = 5000.0
    grid: GridKind = "linear"
    beta_loss_np_per_m: float = 0.0
    signal_sample_rate_hz: float = 48000.0
    signal_duration_s: float = 0.12
    signal_f0_hz: float | None = None
    signal_f1_hz: float | None = None
    signal_amplitude: float = 1.0
    spatial_nodes: int = 21
    cfl: float = 0.95
    observation_node: int = 0


@dataclass(frozen=True)
class ThreePointGeometry:
    length_m: float
    area_left_m2: float
    area_middle_m2: float
    area_right_m2: float


@dataclass(frozen=True)
class LinearGeometry:
    length_m: float
    area_left_m2: float
    area_right_m2: float


@dataclass(frozen=True)
class ConicalGeometry:
    length_m: float
    area_in_m2: float
    area_out_m2: float
    sample_count: int = 257


@dataclass(frozen=True)
class UniformAreasGeometry:
    length_m: float
    area_samples_m2: Sequence[float]


@dataclass(frozen=True)
class ExplicitGeometry:
    x_m: Sequence[float]
    area_m2: Sequence[float]


GeometrySpec = ThreePointGeometry | LinearGeometry | ConicalGeometry | UniformAreasGeometry | ExplicitGeometry | Path | str

GeometryKind = Literal[
    "cylinder",
    "conical",
    "cone",
    "three_point",
    "tube_with_hole",
    "random_smooth",
    "random_piecewise",
    "random",
]


def _sample_parameter_value(value, rng: py_random.Random):
    if isinstance(value, range):
        items = list(value)
        if not items:
            raise ValueError("range parameter must not be empty")
        return rng.choice(items)

    if isinstance(value, tuple) and len(value) == 2:
        lo, hi = value
        if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool) and not isinstance(hi, bool):
            if hi < lo:
                raise ValueError(f"invalid integer range: {value!r}")
            return rng.randint(lo, hi)
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            raise ValueError(f"invalid float range: {value!r}")
        return rng.uniform(lo_f, hi_f)

    if isinstance(value, list):
        if not value:
            raise ValueError("list parameter must not be empty")
        return rng.choice(value)

    return value


def make_cylinder_geometry(length_m: float, area_m2: float) -> LinearGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if area_m2 <= 0.0:
        raise ValueError("area_m2 must be positive")
    return LinearGeometry(length_m=length_m, area_left_m2=area_m2, area_right_m2=area_m2)


def make_conical_geometry(
    length_m: float,
    area_in_m2: float,
    area_out_m2: float,
    sample_count: int = 257,
) -> ConicalGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if area_in_m2 <= 0.0 or area_out_m2 <= 0.0:
        raise ValueError("areas must be positive")
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2")
    return ConicalGeometry(
        length_m=length_m,
        area_in_m2=area_in_m2,
        area_out_m2=area_out_m2,
        sample_count=sample_count,
    )


def make_three_point_geometry(
    length_m: float,
    area_left_m2: float,
    area_middle_m2: float,
    area_right_m2: float,
) -> ThreePointGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if area_left_m2 <= 0.0 or area_middle_m2 <= 0.0 or area_right_m2 <= 0.0:
        raise ValueError("areas must be positive")
    return ThreePointGeometry(
        length_m=length_m,
        area_left_m2=area_left_m2,
        area_middle_m2=area_middle_m2,
        area_right_m2=area_right_m2,
    )


def _sample_conical_arrays(
    length_m: float,
    area_in_m2: float,
    area_out_m2: float,
    sample_count: int,
) -> tuple[list[float], list[float]]:
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2")

    radius_in_m = math.sqrt(area_in_m2 / math.pi)
    radius_out_m = math.sqrt(area_out_m2 / math.pi)

    x_nodes: list[float] = []
    area_nodes: list[float] = []
    for i in range(sample_count):
        t = i / float(sample_count - 1)
        x_nodes.append(length_m * t)
        radius_m = radius_in_m * (1.0 - t) + radius_out_m * t
        area_nodes.append(math.pi * radius_m * radius_m)

    return x_nodes, area_nodes


def _sample_two_cone_arrays(
    length_m: float,
    area_left_m2: float,
    area_middle_m2: float,
    area_right_m2: float,
    sample_count_per_segment: int = 129,
) -> tuple[list[float], list[float]]:
    if sample_count_per_segment < 2:
        raise ValueError("sample_count_per_segment must be >= 2")

    half_length_m = 0.5 * length_m
    left_x, left_area = _sample_conical_arrays(
        half_length_m,
        area_left_m2,
        area_middle_m2,
        sample_count_per_segment,
    )
    right_x, right_area = _sample_conical_arrays(
        half_length_m,
        area_middle_m2,
        area_right_m2,
        sample_count_per_segment,
    )

    x_nodes = left_x + [half_length_m + x for x in right_x[1:]]
    area_nodes = left_area + right_area[1:]
    return x_nodes, area_nodes


def make_tube_with_hole_geometry(
    length_m: float,
    base_area_m2: float | None = None,
    base_width_m: float | None = None,
    hole_center_m: float | None = None,
    hole_width_m: float | None = None,
    hole_area_gain_m2: float | None = None,
    hole_height_m: float | None = None,
    transition_width_m: float | None = None,
    random: bool = False,
    random_position: bool = False,
    seed: int | None = None,
) -> ExplicitGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if base_area_m2 is None and base_width_m is None:
        raise ValueError("Either base_area_m2 or base_width_m must be provided")
    if base_area_m2 is not None and base_width_m is not None:
        raise ValueError("Provide only one of base_area_m2 or base_width_m")

    if base_width_m is not None:
        if base_width_m <= 0.0:
            raise ValueError("base_width_m must be positive")
        base_radius_m = 0.5 * base_width_m
        base_area_m2 = math.pi * base_radius_m * base_radius_m
    else:
        if base_area_m2 is None or base_area_m2 <= 0.0:
            raise ValueError("base_area_m2 must be positive")

    if hole_width_m is None:
        hole_width_m = 0.01 * length_m
    if hole_width_m <= 0.0:
        raise ValueError("hole_width_m must be positive")

    base_radius_m = math.sqrt(base_area_m2 / math.pi)
    if hole_height_m is None:
        hole_height_m = 10.0 * base_radius_m
    if hole_height_m < 0.0:
        raise ValueError("hole_height_m must be non-negative")

    if hole_area_gain_m2 is None:
        target_radius_m = base_radius_m + hole_height_m
        hole_area_gain_m2 = math.pi * target_radius_m * target_radius_m - base_area_m2
    if hole_area_gain_m2 < 0.0:
        raise ValueError("hole_area_gain_m2 must be non-negative")

    taper = 0.5 * hole_width_m if transition_width_m is None else transition_width_m
    if taper < 0.0:
        raise ValueError("transition_width_m must be non-negative")

    half_width = 0.5 * hole_width_m
    use_random_position = random or random_position
    if use_random_position:
        rng = py_random.Random(seed)
        hole_center_m = rng.uniform(half_width, length_m - half_width)
    elif hole_center_m is None:
        hole_center_m = 0.5 * length_m

    if not (0.0 <= hole_center_m <= length_m):
        raise ValueError("hole_center_m must lie inside the tube")

    hole_left = max(0.0, hole_center_m - 0.5 * hole_width_m)
    hole_right = min(length_m, hole_center_m + 0.5 * hole_width_m)
    rise_left = max(0.0, hole_left - taper)
    fall_right = min(length_m, hole_right + taper)

    x_nodes = sorted(
        {
            0.0,
            rise_left,
            hole_left,
            hole_center_m,
            hole_right,
            fall_right,
            length_m,
        }
    )

    area_nodes: list[float] = []
    for x_m in x_nodes:
        if x_m <= rise_left or x_m >= fall_right:
            area = base_area_m2
        elif rise_left < x_m < hole_left and hole_left > rise_left:
            t = (x_m - rise_left) / (hole_left - rise_left)
            area = base_area_m2 + t * hole_area_gain_m2
        elif hole_left <= x_m <= hole_right:
            area = base_area_m2 + hole_area_gain_m2
        elif hole_right < x_m < fall_right and fall_right > hole_right:
            t = (x_m - hole_right) / (fall_right - hole_right)
            area = base_area_m2 + (1.0 - t) * hole_area_gain_m2
        else:
            area = base_area_m2
        area_nodes.append(area)

    return ExplicitGeometry(x_m=x_nodes, area_m2=area_nodes)


def explicit_geometry_from_arrays(x_m: Sequence[float], area_m2: Sequence[float]) -> ExplicitGeometry:
    xs = [float(v) for v in x_m]
    areas = [float(v) for v in area_m2]

    if len(xs) != len(areas):
        raise ValueError("x_m and area_m2 must have the same length")
    if len(xs) < 2:
        raise ValueError("at least two profile points are required")
    if any(area <= 0.0 for area in areas):
        raise ValueError("all areas must be positive")
    if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
        raise ValueError("x_m must be strictly increasing")

    return ExplicitGeometry(x_m=xs, area_m2=areas)


def tube_tuple_to_geometry(
    tube: tuple[Sequence[float], Sequence[float], Sequence[float]] | Sequence[Sequence[float]],
    *,
    check_lengths: bool = True,
    tol: float = 1e-12,
) -> ExplicitGeometry:
    if len(tube) != 3:
        raise ValueError("tube must be a tuple (x_nodes, area_nodes, segment_lengths)")

    x_nodes, area_nodes, segment_lengths = tube
    geometry = explicit_geometry_from_arrays(x_nodes, area_nodes)
    lengths = [float(v) for v in segment_lengths]

    if len(lengths) != len(geometry.x_m) - 1:
        raise ValueError("segment_lengths must have len(x_nodes) - 1 elements")

    if check_lengths:
        for i, seg_len in enumerate(lengths):
            expected = geometry.x_m[i + 1] - geometry.x_m[i]
            if seg_len <= 0.0:
                raise ValueError("all segment lengths must be positive")
            if abs(seg_len - expected) > tol:
                raise ValueError(
                    "segment_lengths must match consecutive x-node differences; "
                    f"mismatch at index {i}: got {seg_len}, expected {expected}"
                )

    return geometry


def tube_factory_to_geometry(tube_factory) -> ExplicitGeometry:
    if not callable(tube_factory):
        raise TypeError("tube_factory must be callable")
    return tube_tuple_to_geometry(tube_factory())


def make_random_smooth_geometry(
    length_m: float = 0.17,
    dx_m: float = 0.005,
    area0_m2: float = 2.0e-4,
    amp: float = 0.3,
    n_harmonics: int = 5,
    seed: int | None = None,
) -> ExplicitGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if dx_m <= 0.0:
        raise ValueError("dx_m must be positive")
    if area0_m2 <= 0.0:
        raise ValueError("area0_m2 must be positive")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    if amp < 0.0:
        raise ValueError("amp must be non-negative")

    rng = py_random.Random(seed)

    step_count = max(1, int(round(length_m / dx_m)))
    actual_dx = length_m / step_count
    x_nodes = [i * actual_dx for i in range(step_count + 1)]

    coeffs = [rng.gauss(0.0, 1.0) for _ in range(n_harmonics)]
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_harmonics)]

    noise: list[float] = []
    for x in x_nodes:
        sample = 0.0
        for k in range(1, n_harmonics + 1):
            sample += coeffs[k - 1] * math.sin(2.0 * math.pi * k * x / length_m + phases[k - 1])
        noise.append(sample)

    max_abs = max((abs(v) for v in noise), default=0.0)
    if max_abs > 0.0:
        noise = [v / max_abs for v in noise]

    area_floor = 0.1 * area0_m2
    areas = [max(area_floor, area0_m2 * (1.0 + amp * v)) for v in noise]

    return ExplicitGeometry(x_m=x_nodes, area_m2=areas)


def make_random_test_geometry(seed: int) -> ExplicitGeometry:
    return make_random_smooth_geometry(
        length_m=0.17,
        dx_m=0.005,
        area0_m2=2.0e-4,
        amp=0.25,
        n_harmonics=6,
        seed=seed,
    )


def make_random_piecewise_geometry(
    length_m: float = 0.17,
    mean_width_m: float = 0.02,
    section_count: int = 6,
    width_spread: float = 0.25,
    seed: int | None = None,
) -> ExplicitGeometry:
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    if mean_width_m <= 0.0:
        raise ValueError("mean_width_m must be positive")
    if section_count < 1:
        raise ValueError("section_count must be >= 1")
    if width_spread < 0.0:
        raise ValueError("width_spread must be non-negative")

    rng = py_random.Random(seed)
    dx = length_m / float(section_count)
    x_nodes = [i * dx for i in range(section_count + 1)]

    min_width_m = 0.1 * mean_width_m
    area_nodes: list[float] = []
    for _ in x_nodes:
        width_scale = 1.0 + rng.uniform(-width_spread, width_spread)
        width_m = max(min_width_m, mean_width_m * width_scale)
        radius_m = 0.5 * width_m
        area_nodes.append(math.pi * radius_m * radius_m)

    return ExplicitGeometry(x_m=x_nodes, area_m2=area_nodes)


def make_geometry_from_ranges(
    kind: GeometryKind,
    parameter_ranges: dict[str, object],
    *,
    seed: int | None = None,
) -> GeometrySpec:
    rng = py_random.Random(seed)
    params = {
        key: _sample_parameter_value(value, rng)
        for key, value in parameter_ranges.items()
    }

    if kind == "random":
        choices = params.pop(
            "kind_choices",
            [
                "cylinder",
                "conical",
                "three_point",
                "tube_with_hole",
                "random_smooth",
                "random_piecewise",
            ],
        )
        if not isinstance(choices, list) or not choices:
            raise ValueError("kind_choices must be a non-empty list")
        sampled_kind = rng.choice(choices)
        if not isinstance(sampled_kind, str):
            raise ValueError("kind_choices must contain strings")
        kind = sampled_kind  # type: ignore[assignment]

    if kind == "cylinder":
        return make_cylinder_geometry(
            length_m=float(params["length_m"]),
            area_m2=float(params["area_m2"]),
        )

    if kind in ("conical", "cone"):
        kwargs = {
            "length_m": float(params["length_m"]),
            "area_in_m2": float(params["area_in_m2"]),
            "area_out_m2": float(params["area_out_m2"]),
        }
        if "sample_count" in params:
            kwargs["sample_count"] = int(params["sample_count"])
        return make_conical_geometry(**kwargs)

    if kind == "three_point":
        return make_three_point_geometry(
            length_m=float(params["length_m"]),
            area_left_m2=float(params["area_left_m2"]),
            area_middle_m2=float(params["area_middle_m2"]),
            area_right_m2=float(params["area_right_m2"]),
        )

    if kind == "tube_with_hole":
        kwargs = {
            "length_m": float(params["length_m"]),
        }
        if "base_area_m2" in params:
            kwargs["base_area_m2"] = float(params["base_area_m2"])
        if "base_width_m" in params:
            kwargs["base_width_m"] = float(params["base_width_m"])
        optional_keys = (
            "hole_center_m",
            "hole_width_m",
            "hole_area_gain_m2",
            "hole_height_m",
            "transition_width_m",
            "random",
            "random_position",
        )
        for key in optional_keys:
            if key in params:
                kwargs[key] = params[key]
        kwargs["seed"] = seed
        return make_tube_with_hole_geometry(**kwargs)

    if kind == "random_smooth":
        kwargs = {
            "length_m": float(params.get("length_m", 0.17)),
            "dx_m": float(params.get("dx_m", 0.005)),
            "area0_m2": float(params.get("area0_m2", 2.0e-4)),
            "amp": float(params.get("amp", 0.3)),
            "n_harmonics": int(params.get("n_harmonics", 5)),
            "seed": seed,
        }
        return make_random_smooth_geometry(**kwargs)

    if kind == "random_piecewise":
        kwargs = {
            "length_m": float(params.get("length_m", 0.17)),
            "mean_width_m": float(params.get("mean_width_m", 0.02)),
            "section_count": int(params.get("section_count", 6)),
            "width_spread": float(params.get("width_spread", 0.25)),
            "seed": seed,
        }
        return make_random_piecewise_geometry(**kwargs)

    raise ValueError(f"Unsupported geometry kind: {kind!r}")


def make_geometry_from_range_library(
    kind: GeometryKind,
    range_library: dict[str, dict[str, object]],
    *,
    seed: int | None = None,
) -> GeometrySpec:
    if not range_library:
        raise ValueError("range_library must not be empty")

    rng = py_random.Random(seed)
    if kind == "random":
        available_kinds = [name for name in range_library.keys() if name != "random"]
        if not available_kinds:
            raise ValueError("range_library must contain at least one non-random geometry kind")
        sampled_kind = rng.choice(available_kinds)
        return make_geometry_from_ranges(sampled_kind, range_library[sampled_kind], seed=seed)

    if kind not in range_library:
        raise KeyError(f"Geometry kind {kind!r} is missing in range_library")

    return make_geometry_from_ranges(kind, range_library[kind], seed=seed)


def make_all_test_geometries(seed: int = 1) -> dict[str, GeometrySpec]:
    return {
        "cylinder": make_cylinder_geometry(length_m=0.17, area_m2=2.0e-4),
        "conical": make_conical_geometry(length_m=0.17, area_in_m2=2.0e-4, area_out_m2=8.0e-4),
        "three_point": make_three_point_geometry(
            length_m=0.17,
            area_left_m2=2.0e-4,
            area_middle_m2=5.0e-4,
            area_right_m2=3.0e-4,
        ),
        "tube_with_hole": make_tube_with_hole_geometry(
            length_m=0.17,
            base_area_m2=2.5e-4,
            hole_center_m=0.085,
            hole_width_m=0.03,
            hole_area_gain_m2=2.0e-4,
        ),
        "explicit": explicit_geometry_from_arrays(
            x_m=[0.0, 0.03, 0.07, 0.11, 0.14, 0.17],
            area_m2=[2.2e-4, 2.8e-4, 4.5e-4, 3.0e-4, 5.8e-4, 7.0e-4],
        ),
        "random_smooth": make_random_test_geometry(seed=seed),
        "random_piecewise": make_random_piecewise_geometry(
            length_m=0.17,
            mean_width_m=0.02,
            section_count=6,
            width_spread=0.25,
            seed=seed,
        ),
    }


def geometry_to_arrays(geometry: GeometrySpec) -> tuple[list[float], list[float]]:
    if isinstance(geometry, ThreePointGeometry):
        return _sample_two_cone_arrays(
            geometry.length_m,
            geometry.area_left_m2,
            geometry.area_middle_m2,
            geometry.area_right_m2,
        )

    if isinstance(geometry, ConicalGeometry):
        return _sample_conical_arrays(
            geometry.length_m,
            geometry.area_in_m2,
            geometry.area_out_m2,
            geometry.sample_count,
        )

    if isinstance(geometry, LinearGeometry):
        return [0.0, geometry.length_m], [geometry.area_left_m2, geometry.area_right_m2]

    if isinstance(geometry, UniformAreasGeometry):
        n = len(geometry.area_samples_m2)
        if n < 2:
            raise ValueError("uniform geometry requires at least two area samples")
        dx = geometry.length_m / float(n - 1)
        return [i * dx for i in range(n)], [float(v) for v in geometry.area_samples_m2]

    if isinstance(geometry, ExplicitGeometry):
        return [float(v) for v in geometry.x_m], [float(v) for v in geometry.area_m2]

    if isinstance(geometry, (Path, str)):
        path = Path(geometry)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            xs: list[float] = []
            areas: list[float] = []
            for row in reader:
                xs.append(float(row["x_m"]))
                areas.append(float(row["area_m2"]))
        return xs, areas

    raise TypeError(f"Unsupported geometry spec: {type(geometry)!r}")


def geometry_to_tube_tuple(geometry: GeometrySpec) -> tuple[list[float], list[float], list[float]]:
    x_m, area_m2 = geometry_to_arrays(geometry)
    lengths = [x_m[i + 1] - x_m[i] for i in range(len(x_m) - 1)]
    return x_m, area_m2, lengths


def plot_geometry(
    geometry: GeometrySpec,
    mode: GeometryPlotKind = "graph",
    *,
    ax=None,
    title: str | None = None,
    equal_aspect: bool = False,
    linewidth: float = 1.5,
):
    x_m, area_m2 = geometry_to_arrays(geometry)

    if len(x_m) != len(area_m2):
        raise ValueError("geometry must provide matching x and area arrays")
    if len(x_m) < 2:
        raise ValueError("geometry must contain at least two points")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if mode == "graph":
        ax.plot(x_m, area_m2, linewidth=linewidth, color="black")
        ax.set_xlabel("x, m")
        ax.set_ylabel("area, m^2")
        ax.grid(True, alpha=0.3)
        ax.set_title(title or "Tube area profile")
        return ax

    if mode == "symmetric":
        radii_m = [math.sqrt(area / math.pi) for area in area_m2]
        upper = radii_m
        lower = [-r for r in radii_m]

        ax.plot(x_m, upper, color="black", linewidth=linewidth)
        ax.plot(x_m, lower, color="black", linewidth=linewidth)
        ax.fill_between(x_m, lower, upper, color="C0", alpha=0.2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("x, m")
        ax.set_ylabel("radius, m")
        ax.set_title(title or "Symmetric tube shape")
        ax.grid(True, alpha=0.3)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")
        return ax

    raise ValueError("mode must be 'graph' or 'symmetric'")


def approximate_with_cylinders(
    geometry: GeometrySpec,
    section_count: int,
    *,
    beta_loss_np_per_m: float = 0.0,
) -> list[dict[str, float]]:
    if section_count < 1:
        raise ValueError("section_count must be >= 1")

    x_m, area_m2 = geometry_to_arrays(geometry)
    x0 = x_m[0]
    x1 = x_m[-1]
    dx = (x1 - x0) / float(section_count)

    sections: list[dict[str, float]] = []
    for i in range(section_count):
        left_x = x0 + i * dx
        right_x = left_x + dx
        center_x = 0.5 * (left_x + right_x)
        center_area = _area_at_linear_arrays(x_m, area_m2, center_x)
        sections.append(
            {
                "left_x_m": left_x,
                "right_x_m": right_x,
                "center_x_m": center_x,
                "length_m": dx,
                "area_m2": center_area,
                "beta_loss_np_per_m": float(beta_loss_np_per_m),
            }
        )

    return sections


def approximate_with_cones(
    geometry: GeometrySpec,
    section_count: int,
) -> list[dict[str, float]]:
    if section_count < 1:
        raise ValueError("section_count must be >= 1")

    x_m, area_m2 = geometry_to_arrays(geometry)
    x0 = x_m[0]
    x1 = x_m[-1]
    dx = (x1 - x0) / float(section_count)

    sections: list[dict[str, float]] = []
    for i in range(section_count):
        left_x = x0 + i * dx
        right_x = left_x + dx
        area_in = _area_at_linear_arrays(x_m, area_m2, left_x)
        area_out = _area_at_linear_arrays(x_m, area_m2, right_x)
        sections.append(
            {
                "left_x_m": left_x,
                "right_x_m": right_x,
                "length_m": dx,
                "area_in_m2": area_in,
                "area_out_m2": area_out,
            }
        )

    return sections


def plot_geometry_approximation(
    geometry: GeometrySpec,
    section_count: int,
    approx: Literal["cylinders", "cones"] = "cylinders",
    *,
    ax=None,
    title: str | None = None,
    mode: GeometryPlotKind = "graph",
    equal_aspect: bool = False,
    linewidth_original: float = 2.0,
    linewidth_approx: float = 1.5,
):
    x_m, area_m2 = geometry_to_arrays(geometry)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if mode == "graph":
        ax.plot(x_m, area_m2, linewidth=linewidth_original, color="black", label="original")

        if approx == "cylinders":
            sections = approximate_with_cylinders(geometry, section_count)
            xs: list[float] = []
            ys: list[float] = []
            for sec in sections:
                xs.extend([sec["left_x_m"], sec["right_x_m"]])
                ys.extend([sec["area_m2"], sec["area_m2"]])
            ax.plot(xs, ys, linewidth=linewidth_approx, label=f"cylinders ({section_count})")
        elif approx == "cones":
            sections = approximate_with_cones(geometry, section_count)
            xs = [sections[0]["left_x_m"]]
            ys = [sections[0]["area_in_m2"]]
            for sec in sections:
                xs.append(sec["right_x_m"])
                ys.append(sec["area_out_m2"])
            ax.plot(xs, ys, linewidth=linewidth_approx, label=f"cones ({section_count})")
        else:
            raise ValueError("approx must be 'cylinders' or 'cones'")

        ax.set_xlabel("x, m")
        ax.set_ylabel("area, m^2")
        ax.set_title(title or f"{approx} approximation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    if mode == "symmetric":
        radii_m = [math.sqrt(area / math.pi) for area in area_m2]
        upper = radii_m
        lower = [-r for r in radii_m]
        ax.plot(x_m, upper, color="black", linewidth=linewidth_original, label="original")
        ax.plot(x_m, lower, color="black", linewidth=linewidth_original)
        approx_color = "C1"

        if approx == "cylinders":
            sections = approximate_with_cylinders(geometry, section_count)
            xs: list[float] = []
            upper_y: list[float] = []
            lower_y: list[float] = []
            for sec in sections:
                radius = math.sqrt(sec["area_m2"] / math.pi)
                xs.extend([sec["left_x_m"], sec["right_x_m"]])
                upper_y.extend([radius, radius])
                lower_y.extend([-radius, -radius])
            ax.plot(xs, upper_y, linewidth=linewidth_approx, color=approx_color, label=f"cylinders ({section_count})")
            ax.plot(xs, lower_y, linewidth=linewidth_approx, color=approx_color)
            ax.fill_between(xs, lower_y, upper_y, color=approx_color, alpha=0.18)
        elif approx == "cones":
            sections = approximate_with_cones(geometry, section_count)
            xs = [sections[0]["left_x_m"]]
            upper_y = [math.sqrt(sections[0]["area_in_m2"] / math.pi)]
            for sec in sections:
                xs.append(sec["right_x_m"])
                upper_y.append(math.sqrt(sec["area_out_m2"] / math.pi))
            lower_y = [-v for v in upper_y]
            ax.plot(xs, upper_y, linewidth=linewidth_approx, color=approx_color, label=f"cones ({section_count})")
            ax.plot(xs, lower_y, linewidth=linewidth_approx, color=approx_color)
            ax.fill_between(xs, lower_y, upper_y, color=approx_color, alpha=0.18)
        else:
            raise ValueError("approx must be 'cylinders' or 'cones'")

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("x, m")
        ax.set_ylabel("radius, m")
        ax.set_title(title or f"{approx} approximation")
        ax.grid(True, alpha=0.3)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        return ax

    raise ValueError("mode must be 'graph' or 'symmetric'")


@dataclass
class SpectrumResult:
    solver: SolverKind
    frequencies_hz: list[float]
    transfer_real: list[float]
    transfer_imag: list[float]
    magnitude: list[float]
    phase_rad: list[float]
    stdout_csv: str

    @property
    def transfer_complex(self) -> list[complex]:
        return [complex(r, i) for r, i in zip(self.transfer_real, self.transfer_imag)]


@dataclass
class TransferFieldResult:
    x_m: list[float]
    frequencies_hz: list[float]
    transfer_by_position: list[list[complex]]

    @property
    def magnitude_by_position(self) -> list[list[float]]:
        return [[abs(value) for value in row] for row in self.transfer_by_position]


@dataclass
class WebsterFieldResult:
    sample_rate_hz: float
    x_m: list[float]
    area_m2: list[float]
    input_pressure: list[float]
    pressure_by_time: list[list[float]]

    @property
    def time_s(self) -> list[float]:
        return [i / self.sample_rate_hz for i in range(len(self.pressure_by_time))]


@dataclass
class ComparisonResult:
    results: dict[SolverKind, SpectrumResult]


@dataclass
class RelativeErrorSummary:
    reference_solver: SolverKind
    solver: SolverKind
    mean_rel_mag_err: float
    max_rel_mag_err: float


@dataclass
class GeometryBenchmarkResult:
    geometry_name: str
    geometry: GeometrySpec
    times_s: dict[SolverKind, float]
    comparison: ComparisonResult
    relative_errors: dict[SolverKind, RelativeErrorSummary]


def _require_gpp() -> str:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ was not found. Install MinGW-w64 or add g++ to PATH.")
    return compiler


def build_binary(binary_path: Path = BINARY_PATH, rebuild: bool = False) -> Path:
    if binary_path.exists() and not rebuild:
        return binary_path

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = _require_gpp()
    sources = [
        CPP_DIR / "vt_all_solvers_cli.cpp",
        CPP_DIR / "vt_geometry_simple.cpp",
        CPP_DIR / "vt_cylinder_tlm_solver.cpp",
        CPP_DIR / "vt_cone_reference_solver.cpp",
        CPP_DIR / "vt_arma_solver.cpp",
        CPP_DIR / "vt_webster_fdtd_solver.cpp",
    ]

    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT),
        *map(str, sources),
        "-o",
        str(binary_path),
    ]

    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
    return binary_path


def _geometry_args(geometry: GeometrySpec) -> list[str]:
    if isinstance(geometry, ThreePointGeometry):
        x_m, area_m2 = geometry_to_arrays(geometry)
        xs = ",".join(str(v) for v in x_m)
        areas = ",".join(str(v) for v in area_m2)
        return [
            "--geometry", "explicit",
            "--x-m", xs,
            "--areas-m2", areas,
        ]

    if isinstance(geometry, ConicalGeometry):
        x_m, area_m2 = geometry_to_arrays(geometry)
        xs = ",".join(str(v) for v in x_m)
        areas = ",".join(str(v) for v in area_m2)
        return [
            "--geometry", "explicit",
            "--x-m", xs,
            "--areas-m2", areas,
        ]

    if isinstance(geometry, LinearGeometry):
        return [
            "--geometry", "linear",
            "--length-m", str(geometry.length_m),
            "--areas-m2", f"{geometry.area_left_m2},{geometry.area_right_m2}",
        ]

    if isinstance(geometry, UniformAreasGeometry):
        areas = ",".join(str(v) for v in geometry.area_samples_m2)
        return [
            "--geometry", "uniform-areas",
            "--length-m", str(geometry.length_m),
            "--areas-m2", areas,
        ]

    if isinstance(geometry, ExplicitGeometry):
        xs = ",".join(str(v) for v in geometry.x_m)
        areas = ",".join(str(v) for v in geometry.area_m2)
        return [
            "--geometry", "explicit",
            "--x-m", xs,
            "--areas-m2", areas,
        ]

    if isinstance(geometry, (Path, str)):
        return ["--profile-csv", str(geometry)]

    raise TypeError(f"Unsupported geometry spec: {type(geometry)!r}")


def _validate_config(config: SolverConfig) -> None:
    if config.sections < 1:
        raise ValueError("sections must be >= 1")
    if config.points < 2:
        raise ValueError("points must be >= 2")
    if config.f_max_hz <= config.f_min_hz:
        raise ValueError("f_max_hz must be greater than f_min_hz")
    if config.grid not in ("linear", "log"):
        raise ValueError("grid must be 'linear' or 'log'")
    if config.solver in ("cone", "arma") and config.f_min_hz <= 0.0:
        raise ValueError(f"solver '{config.solver}' requires f_min_hz > 0")
    if config.signal_sample_rate_hz <= 0.0:
        raise ValueError("signal_sample_rate_hz must be positive")
    if config.signal_duration_s <= 0.0:
        raise ValueError("signal_duration_s must be positive")
    if config.signal_amplitude <= 0.0:
        raise ValueError("signal_amplitude must be positive")
    if config.spatial_nodes < 3:
        raise ValueError("spatial_nodes must be >= 3")
    if not (0.0 < config.cfl <= 1.0):
        raise ValueError("cfl must be in (0, 1]")
    if config.observation_node < 0:
        raise ValueError("observation_node must be >= 0")

    signal_f0_hz = config.f_min_hz if config.signal_f0_hz is None else config.signal_f0_hz
    signal_f1_hz = config.f_max_hz if config.signal_f1_hz is None else config.signal_f1_hz
    if signal_f0_hz < 0.0:
        raise ValueError("signal_f0_hz must be >= 0")
    if signal_f1_hz <= signal_f0_hz:
        raise ValueError("signal_f1_hz must be greater than signal_f0_hz")


def _make_frequency_grid(config: SolverConfig) -> list[float]:
    if config.grid == "linear":
        if config.points < 2:
            raise ValueError("points must be >= 2")
        step = (config.f_max_hz - config.f_min_hz) / float(config.points - 1)
        return [config.f_min_hz + i * step for i in range(config.points)]

    if config.grid == "log":
        if config.f_min_hz <= 0.0:
            raise ValueError("log grid requires f_min_hz > 0")
        log_min = math.log(config.f_min_hz)
        log_max = math.log(config.f_max_hz)
        dlog = (log_max - log_min) / float(config.points - 1)
        return [math.exp(log_min + i * dlog) for i in range(config.points)]

    raise ValueError(f"Unsupported grid kind: {config.grid}")


def _area_at_linear_arrays(x_nodes: Sequence[float], area_nodes: Sequence[float], x_m: float) -> float:
    if x_m <= x_nodes[0]:
        return float(area_nodes[0])
    if x_m >= x_nodes[-1]:
        return float(area_nodes[-1])

    for i in range(len(x_nodes) - 1):
        x0 = float(x_nodes[i])
        x1 = float(x_nodes[i + 1])
        if x0 <= x_m <= x1:
            a0 = float(area_nodes[i])
            a1 = float(area_nodes[i + 1])
            t = (x_m - x0) / (x1 - x0)
            return a0 * (1.0 - t) + a1 * t

    return float(area_nodes[-1])


def solve_cylinder_transfer_field(
    geometry: GeometrySpec,
    config: SolverConfig = SolverConfig(solver="cylinder"),
    acoustics: AcousticConfig = AcousticConfig(),
) -> TransferFieldResult:
    _validate_config(config)

    x_nodes, area_nodes = geometry_to_arrays(geometry)
    if len(x_nodes) < 2:
        raise ValueError("geometry must contain at least two points")
    if config.sections < 1:
        raise ValueError("sections must be >= 1")

    frequencies_hz = _make_frequency_grid(config)
    total_length_m = x_nodes[-1] - x_nodes[0]
    dx = total_length_m / float(config.sections)

    x_grid_m = [x_nodes[0] + i * dx for i in range(config.sections + 1)]
    section_areas_m2 = [
        _area_at_linear_arrays(x_nodes, area_nodes, 0.5 * (x_grid_m[i] + x_grid_m[i + 1]))
        for i in range(config.sections)
    ]

    zc = acoustics.rho_kg_m3 * acoustics.c_m_s
    transfer_by_position: list[list[complex]] = [
        [0j for _ in frequencies_hz] for _ in x_grid_m
    ]

    for fi, frequency_hz in enumerate(frequencies_hz):
        omega = 2.0 * math.pi * frequency_hz
        a11 = 1.0 + 0.0j
        a12 = 0.0 + 0.0j
        a21 = 0.0 + 0.0j
        a22 = 1.0 + 0.0j

        transfer_by_position[0][fi] = 1.0 + 0.0j

        for si, area_m2 in enumerate(section_areas_m2):
            gamma = complex(config.beta_loss_np_per_m, omega / acoustics.c_m_s)
            gl = gamma * dx
            ch = cmath.cosh(gl)
            sh = cmath.sinh(gl)

            local11 = ch
            local12 = -(zc / area_m2) * sh
            local21 = -(area_m2 / zc) * sh
            local22 = ch

            next11 = local11 * a11 + local12 * a21
            next12 = local11 * a12 + local12 * a22
            next21 = local21 * a11 + local22 * a21
            next22 = local21 * a12 + local22 * a22

            a11 = next11
            a12 = next12
            a21 = next21
            a22 = next22

            transfer_by_position[si + 1][fi] = 0j if abs(a11) < 1e-15 else 1.0 / a11

    return TransferFieldResult(
        x_m=x_grid_m,
        frequencies_hz=frequencies_hz,
        transfer_by_position=transfer_by_position,
    )


def solve_webster_field(
    geometry: GeometrySpec,
    input_signal: Sequence[float],
    config: SolverConfig = SolverConfig(solver="webster"),
    acoustics: AcousticConfig = AcousticConfig(),
) -> WebsterFieldResult:
    _validate_config(config)

    x_nodes, area_nodes = geometry_to_arrays(geometry)
    if len(x_nodes) < 2:
        raise ValueError("geometry must contain at least two points")

    sample_rate_hz = float(config.signal_sample_rate_hz)
    signal = [float(v) for v in input_signal]
    if not signal:
        raise ValueError("input_signal must not be empty")

    x_left = x_nodes[0]
    x_right = x_nodes[-1]
    nx = int(config.spatial_nodes)
    dx = (x_right - x_left) / float(nx - 1)
    dt = 1.0 / sample_rate_hz
    courant = acoustics.c_m_s * dt / dx
    if courant > config.cfl + 1e-12:
        raise ValueError(
            "Time step is too large for the chosen spatial grid: c*dt/dx exceeds cfl"
        )

    x_grid = [x_left + i * dx for i in range(nx)]
    area_grid = [_area_at_linear_arrays(x_nodes, area_nodes, x) for x in x_grid]
    area_half = [0.5 * (area_grid[i] + area_grid[i + 1]) for i in range(nx - 1)]

    p_prev = [0.0] * nx
    p_cur = [0.0] * nx
    p_next = [0.0] * nx

    c2dt2 = acoustics.c_m_s * acoustics.c_m_s * dt * dt
    inv_dx2 = 1.0 / (dx * dx)

    pressure_by_time: list[list[float]] = []

    for sample in signal:
        p_cur[0] = sample
        p_cur[-1] = 0.0

        for i in range(1, nx - 1):
            flux_right = area_half[i] * (p_cur[i + 1] - p_cur[i])
            flux_left = area_half[i - 1] * (p_cur[i] - p_cur[i - 1])
            laplacian = (flux_right - flux_left) * inv_dx2 / area_grid[i]
            p_next[i] = 2.0 * p_cur[i] - p_prev[i] + c2dt2 * laplacian

        p_next[0] = sample
        p_next[-1] = 0.0

        pressure_by_time.append(list(p_cur))

        p_prev, p_cur, p_next = p_cur, p_next, p_prev
        for i in range(nx):
            p_next[i] = 0.0

    return WebsterFieldResult(
        sample_rate_hz=sample_rate_hz,
        x_m=x_grid,
        area_m2=area_grid,
        input_pressure=signal,
        pressure_by_time=pressure_by_time,
    )


def _run_solver(
    geometry: GeometrySpec,
    solver: SolverKind,
    config: SolverConfig,
    acoustics: AcousticConfig,
    rebuild: bool = False,
    binary_path: Path = BINARY_PATH,
) -> str:
    _validate_config(config)
    binary = build_binary(binary_path=binary_path, rebuild=rebuild)

    command = [
        str(binary),
        "--solver", solver,
        "--sections", str(config.sections),
        "--points", str(config.points),
        "--f-min-hz", str(config.f_min_hz),
        "--f-max-hz", str(config.f_max_hz),
        "--grid", config.grid,
        "--beta-loss", str(config.beta_loss_np_per_m),
        "--signal-sample-rate-hz", str(config.signal_sample_rate_hz),
        "--signal-duration-s", str(config.signal_duration_s),
        "--signal-amplitude", str(config.signal_amplitude),
        "--spatial-nodes", str(config.spatial_nodes),
        "--cfl", str(config.cfl),
        "--observation-node", str(config.observation_node),
        "--rho", str(acoustics.rho_kg_m3),
        "--c", str(acoustics.c_m_s),
        *_geometry_args(geometry),
    ]

    if config.signal_f0_hz is not None:
        command.extend(["--signal-f0-hz", str(config.signal_f0_hz)])
    if config.signal_f1_hz is not None:
        command.extend(["--signal-f1-hz", str(config.signal_f1_hz)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _parse_solver_csv(stdout_csv: str) -> SpectrumResult:
    reader = csv.DictReader(stdout_csv.splitlines())
    frequencies_hz: list[float] = []
    transfer_real: list[float] = []
    transfer_imag: list[float] = []
    magnitude: list[float] = []
    phase_rad: list[float] = []
    solver_name: str | None = None

    for row in reader:
        solver_name = row["solver"]
        frequencies_hz.append(float(row["frequency_hz"]))
        transfer_real.append(float(row["real"]))
        transfer_imag.append(float(row["imag"]))
        magnitude.append(float(row["magnitude"]))
        phase_rad.append(float(row["phase_rad"]))

    if not frequencies_hz or solver_name is None:
        raise RuntimeError("Failed to parse solver CSV output.")

    return SpectrumResult(
        solver=solver_name,  # type: ignore[arg-type]
        frequencies_hz=frequencies_hz,
        transfer_real=transfer_real,
        transfer_imag=transfer_imag,
        magnitude=magnitude,
        phase_rad=phase_rad,
        stdout_csv=stdout_csv,
    )


def solve(
    geometry: GeometrySpec,
    config: SolverConfig = SolverConfig(),
    acoustics: AcousticConfig = AcousticConfig(),
    rebuild: bool = False,
    binary_path: Path = BINARY_PATH,
) -> SpectrumResult:
    stdout_csv = _run_solver(
        geometry=geometry,
        solver=config.solver,
        config=config,
        acoustics=acoustics,
        rebuild=rebuild,
        binary_path=binary_path,
    )
    return _parse_solver_csv(stdout_csv)


def relative_magnitude_error(
    reference: SpectrumResult,
    candidate: SpectrumResult,
    eps: float = 1e-12,
) -> RelativeErrorSummary:
    if len(reference.frequencies_hz) != len(candidate.frequencies_hz):
        raise ValueError("reference and candidate must use the same frequency grid length")

    rel_errors: list[float] = []
    for f_ref, f_cmp, mag_ref, mag_cmp in zip(
        reference.frequencies_hz,
        candidate.frequencies_hz,
        reference.magnitude,
        candidate.magnitude,
    ):
        if abs(f_ref - f_cmp) > 1e-9:
            raise ValueError("reference and candidate must use the same frequency grid")
        denom = max(abs(mag_ref), eps)
        rel_errors.append(abs(mag_cmp - mag_ref) / denom)

    if not rel_errors:
        raise ValueError("reference and candidate must not be empty")

    return RelativeErrorSummary(
        reference_solver=reference.solver,
        solver=candidate.solver,
        mean_rel_mag_err=sum(rel_errors) / len(rel_errors),
        max_rel_mag_err=max(rel_errors),
    )


def benchmark_random_geometries(
    n_runs: int = 100,
    solvers: Sequence[SolverKind] = ("cylinder", "cone", "arma"),
    base_config: SolverConfig = SolverConfig(),
    acoustics: AcousticConfig = AcousticConfig(),
    rebuild: bool = False,
    binary_path: Path = BINARY_PATH,
) -> tuple[dict[SolverKind, list[float]], list[ExplicitGeometry]]:
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if not solvers:
        raise ValueError("solvers must not be empty")

    # Сначала гарантируем, что бинарник уже собран, и не включаем сборку в бенчмарк.
    build_binary(binary_path=binary_path, rebuild=rebuild)

    times_by_solver: dict[SolverKind, list[float]] = {solver: [] for solver in solvers}
    geometries: list[ExplicitGeometry] = []

    for run_idx in range(n_runs):
        geom = make_random_test_geometry(seed=run_idx + 1)
        geometries.append(geom)

        for solver_name in solvers:
            config = SolverConfig(
                solver=solver_name,
                sections=base_config.sections,
                points=base_config.points,
                f_min_hz=base_config.f_min_hz,
                f_max_hz=base_config.f_max_hz,
                grid=base_config.grid,
                beta_loss_np_per_m=base_config.beta_loss_np_per_m,
                signal_sample_rate_hz=base_config.signal_sample_rate_hz,
                signal_duration_s=base_config.signal_duration_s,
                signal_f0_hz=base_config.signal_f0_hz,
                signal_f1_hz=base_config.signal_f1_hz,
                signal_amplitude=base_config.signal_amplitude,
                spatial_nodes=base_config.spatial_nodes,
                cfl=base_config.cfl,
                observation_node=base_config.observation_node,
            )

            t0 = time.perf_counter()
            _ = solve(
                geometry=geom,
                config=config,
                acoustics=acoustics,
                rebuild=False,
                binary_path=binary_path,
            )
            t1 = time.perf_counter()

            times_by_solver[solver_name].append(t1 - t0)

    return times_by_solver, geometries


def benchmark_geometry_suite(
    solvers: Sequence[SolverKind] = ("cylinder", "cone", "arma"),
    base_config: SolverConfig = SolverConfig(),
    acoustics: AcousticConfig = AcousticConfig(),
    geometry_suite: dict[str, GeometrySpec] | None = None,
    reference_solver: SolverKind = "cylinder",
    rebuild: bool = False,
    binary_path: Path = BINARY_PATH,
) -> list[GeometryBenchmarkResult]:
    if not solvers:
        raise ValueError("solvers must not be empty")
    if reference_solver not in solvers:
        raise ValueError("reference_solver must be included in solvers")

    suite = make_all_test_geometries() if geometry_suite is None else geometry_suite
    if not suite:
        raise ValueError("geometry_suite must not be empty")

    build_binary(binary_path=binary_path, rebuild=rebuild)

    out: list[GeometryBenchmarkResult] = []
    for geometry_name, geometry in suite.items():
        times_s: dict[SolverKind, float] = {}
        results: dict[SolverKind, SpectrumResult] = {}

        for solver_name in solvers:
            config = SolverConfig(
                solver=solver_name,
                sections=base_config.sections,
                points=base_config.points,
                f_min_hz=base_config.f_min_hz,
                f_max_hz=base_config.f_max_hz,
                grid=base_config.grid,
                beta_loss_np_per_m=base_config.beta_loss_np_per_m,
                signal_sample_rate_hz=base_config.signal_sample_rate_hz,
                signal_duration_s=base_config.signal_duration_s,
                signal_f0_hz=base_config.signal_f0_hz,
                signal_f1_hz=base_config.signal_f1_hz,
                signal_amplitude=base_config.signal_amplitude,
                spatial_nodes=base_config.spatial_nodes,
                cfl=base_config.cfl,
                observation_node=base_config.observation_node,
            )

            t0 = time.perf_counter()
            result = solve(
                geometry=geometry,
                config=config,
                acoustics=acoustics,
                rebuild=False,
                binary_path=binary_path,
            )
            t1 = time.perf_counter()

            times_s[solver_name] = t1 - t0
            results[solver_name] = result

        comparison = ComparisonResult(results=results)
        reference = results[reference_solver]
        relative_errors = {
            solver_name: relative_magnitude_error(reference, result)
            for solver_name, result in results.items()
        }

        out.append(
            GeometryBenchmarkResult(
                geometry_name=geometry_name,
                geometry=geometry,
                times_s=times_s,
                comparison=comparison,
                relative_errors=relative_errors,
            )
        )

    return out


def compare_solvers(
    geometry: GeometrySpec,
    solvers: Sequence[SolverKind],
    base_config: SolverConfig = SolverConfig(),
    acoustics: AcousticConfig = AcousticConfig(),
    rebuild: bool = False,
    binary_path: Path = BINARY_PATH,
) -> ComparisonResult:
    if not solvers:
        raise ValueError("solvers must not be empty")

    results: dict[SolverKind, SpectrumResult] = {}
    built_once = False
    for solver in solvers:
        config = SolverConfig(
            solver=solver,
            sections=base_config.sections,
            points=base_config.points,
            f_min_hz=base_config.f_min_hz,
            f_max_hz=base_config.f_max_hz,
            grid=base_config.grid,
            beta_loss_np_per_m=base_config.beta_loss_np_per_m,
            signal_sample_rate_hz=base_config.signal_sample_rate_hz,
            signal_duration_s=base_config.signal_duration_s,
            signal_f0_hz=base_config.signal_f0_hz,
            signal_f1_hz=base_config.signal_f1_hz,
            signal_amplitude=base_config.signal_amplitude,
            spatial_nodes=base_config.spatial_nodes,
            cfl=base_config.cfl,
            observation_node=base_config.observation_node,
        )
        result = solve(
            geometry=geometry,
            config=config,
            acoustics=acoustics,
            rebuild=(rebuild and not built_once),
            binary_path=binary_path,
        )
        built_once = True
        results[solver] = result

    return ComparisonResult(results=results)


def _comparison_csv(comparison: ComparisonResult) -> str:
    lines: list[str] = ["solver,frequency_hz,real,imag,magnitude,phase_rad"]

    for result in comparison.results.values():
        rows = result.stdout_csv.splitlines()
        if len(rows) <= 1:
            continue
        lines.extend(rows[1:])

    return "\n".join(lines) + "\n"


def save_spectrum_csv(result: SpectrumResult, path: str | Path) -> None:
    Path(path).write_text(result.stdout_csv, encoding="utf-8")


def _y_values(result: SpectrumResult, mode: PlotKind) -> tuple[list[float], str]:
    if mode == "magnitude":
        return result.magnitude, "|H(f)|"
    if mode == "db":
        values = [20.0 * math.log10(max(v, 1e-15)) for v in result.magnitude]
        return values, "20 log10 |H(f)|, dB"
    if mode == "phase":
        return result.phase_rad, "phase, rad"
    raise ValueError(f"Unsupported plot mode: {mode}")


def plot_spectrum(
    result: SpectrumResult,
    y: PlotKind = "magnitude",
    ax=None,
    label: str | None = None,
) -> None:
    values, ylabel = _y_values(result, y)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result.frequencies_hz, values, linewidth=1.5, label=label or result.solver)
    ax.set_xlabel("f, Hz")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Transfer Function: {result.solver}")
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend()
    plt.tight_layout()


def plot_comparison(comparison: ComparisonResult, y: PlotKind = "magnitude") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for solver, result in comparison.results.items():
        values, ylabel = _y_values(result, y)
        ax.plot(result.frequencies_hz, values, linewidth=1.5, label=solver)
    ax.set_xlabel("f, Hz")
    ax.set_ylabel(ylabel)
    ax.set_title("Solver comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Python wrapper for cylinder, cone, ARMA, and Webster solvers.")
    parser.add_argument("--solver", choices=["cylinder", "cone", "arma", "webster", "compare"], default="cylinder")
    parser.add_argument(
        "--compare-solvers",
        default="cylinder,cone,webster",
        help="Comma-separated solver list for --solver compare",
    )
    parser.add_argument(
        "--geometry",
        choices=["three-point", "linear", "uniform-areas", "explicit", "csv"],
        default="three-point",
    )
    parser.add_argument("--profile-csv", default=None)
    parser.add_argument("--x-m", default=None)
    parser.add_argument("--length-m", type=float, default=0.17)
    parser.add_argument("--areas-m2", default="3.0e-4,8.0e-5,4.0e-4")
    parser.add_argument("--sections", type=int, default=20)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--f-min-hz", type=float, default=50.0)
    parser.add_argument("--f-max-hz", type=float, default=5000.0)
    parser.add_argument("--grid", choices=["linear", "log"], default="linear")
    parser.add_argument("--beta-loss", type=float, default=0.0)
    parser.add_argument("--signal-sample-rate-hz", type=float, default=48000.0)
    parser.add_argument("--signal-duration-s", type=float, default=0.12)
    parser.add_argument("--signal-f0-hz", type=float, default=None)
    parser.add_argument("--signal-f1-hz", type=float, default=None)
    parser.add_argument("--signal-amplitude", type=float, default=1.0)
    parser.add_argument("--spatial-nodes", type=int, default=21)
    parser.add_argument("--cfl", type=float, default=0.95)
    parser.add_argument("--observation-node", type=int, default=0)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("--c", type=float, default=343.0)
    parser.add_argument("--plot", choices=["none", "magnitude", "db", "phase"], default="magnitude")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    acoustics = AcousticConfig(rho_kg_m3=args.rho, c_m_s=args.c)
    base_config = SolverConfig(
        solver="cylinder",
        sections=args.sections,
        points=args.points,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        grid=args.grid,
        beta_loss_np_per_m=args.beta_loss,
        signal_sample_rate_hz=args.signal_sample_rate_hz,
        signal_duration_s=args.signal_duration_s,
        signal_f0_hz=args.signal_f0_hz,
        signal_f1_hz=args.signal_f1_hz,
        signal_amplitude=args.signal_amplitude,
        spatial_nodes=args.spatial_nodes,
        cfl=args.cfl,
        observation_node=args.observation_node,
    )

    if args.geometry == "csv":
        if args.profile_csv is None:
            print("--profile-csv is required when --geometry=csv", file=sys.stderr)
            return 1
        geometry: GeometrySpec = Path(args.profile_csv)
    else:
        areas = [float(v) for v in args.areas_m2.split(",") if v.strip()]
        if args.geometry == "three-point":
            if len(areas) != 3:
                print("three-point requires exactly 3 areas", file=sys.stderr)
                return 1
            geometry = ThreePointGeometry(args.length_m, areas[0], areas[1], areas[2])
        elif args.geometry == "linear":
            if len(areas) != 2:
                print("linear requires exactly 2 areas", file=sys.stderr)
                return 1
            geometry = LinearGeometry(args.length_m, areas[0], areas[1])
        elif args.geometry == "uniform-areas":
            geometry = UniformAreasGeometry(args.length_m, areas)
        else:
            if args.x_m is None:
                print("--x-m is required when --geometry=explicit", file=sys.stderr)
                return 1
            xs = [float(v) for v in args.x_m.split(",") if v.strip()]
            geometry = ExplicitGeometry(xs, areas)

    try:
        if args.solver == "compare":
            solvers = [s.strip() for s in args.compare_solvers.split(",") if s.strip()]
            comparison = compare_solvers(
                geometry=geometry,
                solvers=solvers,  # type: ignore[arg-type]
                base_config=base_config,
                acoustics=acoustics,
                rebuild=args.rebuild,
            )
            print(_comparison_csv(comparison), end="")
            if args.plot != "none":
                plot_comparison(comparison, y=args.plot)  # type: ignore[arg-type]
            return 0

        result = solve(
            geometry=geometry,
            config=SolverConfig(
                solver=args.solver,  # type: ignore[arg-type]
                sections=args.sections,
                points=args.points,
                f_min_hz=args.f_min_hz,
                f_max_hz=args.f_max_hz,
                grid=args.grid,
                beta_loss_np_per_m=args.beta_loss,
                signal_sample_rate_hz=args.signal_sample_rate_hz,
                signal_duration_s=args.signal_duration_s,
                signal_f0_hz=args.signal_f0_hz,
                signal_f1_hz=args.signal_f1_hz,
                signal_amplitude=args.signal_amplitude,
                spatial_nodes=args.spatial_nodes,
                cfl=args.cfl,
                observation_node=args.observation_node,
            ),
            acoustics=acoustics,
            rebuild=args.rebuild,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(result.stdout_csv, end="")
    if args.save_csv:
        save_spectrum_csv(result, args.save_csv)
    if args.plot != "none":
        plot_spectrum(result, y=args.plot)  # type: ignore[arg-type]
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
