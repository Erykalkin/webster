from __future__ import annotations

import copy
import re
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, MutableMapping, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import metrics as spectral_metrics

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional for scripts/train jobs
    plt = None

try:
    from IPython.display import clear_output
except ImportError:  # live notebook plotting is optional
    clear_output = None


BatchToXY = Callable[[Any, torch.device], tuple[Any, torch.Tensor]]
MetricFn = Callable[..., torch.Tensor | float]
CriterionFn = Callable[..., torch.Tensor | tuple[torch.Tensor, Any]]
MetricSpec = Mapping[str, MetricFn] | Sequence[str] | None
StepCallback = Callable[[int, float], None]


@dataclass
class TrainConfig:
    epochs: int = 50
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    steps_per_epoch: int | None = None
    val_steps: int | None = None
    val_every: int = 1
    grad_clip_norm: float | None = None
    use_amp: bool = False
    show_progress: bool = True
    scheduler_on: str = "val_loss"  # "val_loss", "train_loss", or "epoch"
    min_lr: float | None = None
    early_stopping_patience: int | None = None
    restore_best: bool = True
    live_plot_every_steps: int | None = None
    live_plot_show_epoch: bool = True
    checkpoint_dir: str | Path = "checkpoints"
    checkpoint_name: str | None = None
    save_best: bool = True
    save_every_steps: int | None = None
    validation_metrics: MetricSpec = None


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    step_train_loss: list[float] = field(default_factory=list)
    step_val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    planned_train_steps: int | None = None
    planned_val_steps: int | None = None
    planned_epochs: int | None = None
    planned_validations: int | None = None
    batch_size: int | None = None
    best_checkpoint_path: str | None = None
    intermediate_checkpoint_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_train_history(history: TrainHistory | Mapping[str, Any] | None = None) -> TrainHistory:
    if history is None:
        return TrainHistory()
    if isinstance(history, TrainHistory):
        return copy.deepcopy(history)
    if not isinstance(history, Mapping):
        raise TypeError("history must be a TrainHistory, a mapping, or None")

    return TrainHistory(
        train_loss=list(history.get("train_loss", []) or []),
        val_loss=list(history.get("val_loss", []) or []),
        step_train_loss=list(history.get("step_train_loss", []) or []),
        step_val_loss=list(history.get("step_val_loss", []) or []),
        lr=list(history.get("lr", []) or []),
        metrics={
            str(name): list(values or [])
            for name, values in (history.get("metrics", {}) or {}).items()
        },
        planned_train_steps=history.get("planned_train_steps"),
        planned_val_steps=history.get("planned_val_steps"),
        planned_epochs=history.get("planned_epochs"),
        planned_validations=history.get("planned_validations"),
        batch_size=history.get("batch_size"),
        best_checkpoint_path=history.get("best_checkpoint_path"),
        intermediate_checkpoint_paths=list(
            history.get("intermediate_checkpoint_paths", []) or []
        ),
    )


def _best_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return min(finite) if finite else float("inf")


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def clean_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if key != "_metadata"
    }


def load_model_state(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    return model.load_state_dict(clean_state_dict(state_dict), strict=strict)


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def checkpoint_history(checkpoint: Any) -> Any | None:
    if isinstance(checkpoint, Mapping):
        return checkpoint.get("history")
    return None


def history_get(history: Any | None, key: str) -> Any:
    if history is None:
        return []
    if hasattr(history, "as_dict"):
        return history.as_dict().get(key, []) or []
    if isinstance(history, Mapping):
        return history.get(key, []) or []
    return getattr(history, key, []) or []


def history_to_mapping(history: Any | None) -> Mapping[str, Any]:
    if history is None:
        return {}
    if hasattr(history, "as_dict"):
        return history.as_dict()
    if isinstance(history, Mapping):
        return history
    return {
        "train_loss": getattr(history, "train_loss", []),
        "val_loss": getattr(history, "val_loss", []),
        "step_train_loss": getattr(history, "step_train_loss", []),
        "step_val_loss": getattr(history, "step_val_loss", []),
        "lr": getattr(history, "lr", []),
        "metrics": getattr(history, "metrics", {}),
        "planned_train_steps": getattr(history, "planned_train_steps", None),
        "planned_val_steps": getattr(history, "planned_val_steps", None),
        "planned_epochs": getattr(history, "planned_epochs", None),
        "planned_validations": getattr(history, "planned_validations", None),
        "batch_size": getattr(history, "batch_size", None),
    }


def load_history_from_checkpoint(
    checkpoint_name: str,
    *,
    checkpoint_dir: str | Path = "checkpoints",
    map_location: str | torch.device = "cpu",
) -> Any | None:
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}_best.pt"
    if not checkpoint_path.exists():
        print(f"history checkpoint not found: {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        print(f"failed to load history from {checkpoint_path}: {exc}")
        return None

    history = checkpoint_history(checkpoint)
    if history is not None:
        return history

    print(f"checkpoint has no history: {checkpoint_path}")
    return None


def get_or_load_history(
    namespace: MutableMapping[str, Any],
    history_variable_name: str,
    checkpoint_name: str,
    *,
    checkpoint_dir: str | Path = "checkpoints",
    map_location: str | torch.device = "cpu",
) -> Any | None:
    history = namespace.get(history_variable_name)
    if history is not None:
        return history

    history = load_history_from_checkpoint(
        checkpoint_name,
        checkpoint_dir=checkpoint_dir,
        map_location=map_location,
    )
    if history is not None:
        namespace[history_variable_name] = history
    return history


def get_or_load_model(
    namespace: MutableMapping[str, Any],
    *,
    variable_name: str,
    checkpoint_name: str,
    factory: Callable[[], nn.Module],
    device: str | torch.device,
    history_variable_name: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    strict: bool = True,
    verbose: bool = True,
) -> tuple[nn.Module | None, Any | None]:
    existing_model = namespace.get(variable_name)
    existing_history = (
        namespace.get(history_variable_name)
        if history_variable_name is not None
        else None
    )

    device = torch.device(device)
    if existing_model is not None:
        existing_model = existing_model.to(device)
        existing_model.eval()
        if verbose:
            print(f"{variable_name}: using model from current notebook session")
        return existing_model, existing_history

    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}_best.pt"
    if not checkpoint_path.exists():
        if verbose:
            print(f"{variable_name}: checkpoint not found, skipping: {checkpoint_path}")
        return None, existing_history

    try:
        model = factory().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state(model, checkpoint_state_dict(checkpoint), strict=strict)
        model.eval()

        history = checkpoint_history(checkpoint)
        namespace[variable_name] = model
        if history_variable_name is not None and history is not None:
            namespace[history_variable_name] = history

        if verbose:
            print(f"{variable_name}: loaded checkpoint {checkpoint_path}")
            if history is not None:
                print(f"{variable_name}: loaded training history from checkpoint")
        return model, history
    except Exception as exc:
        if verbose:
            print(f"{variable_name}: failed to load checkpoint {checkpoint_path}: {exc}")
        return None, existing_history


def _safe_checkpoint_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned or "model"


def _checkpoint_stem(model: nn.Module, config: TrainConfig) -> str:
    name = config.checkpoint_name or getattr(model, "model_name", None) or model.__class__.__name__
    return _safe_checkpoint_stem(str(name))


def _loader_batch_size(loader: Any) -> int | None:
    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        return None
    try:
        return int(batch_size)
    except (TypeError, ValueError):
        return None


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: TrainHistory,
    epoch: int,
    global_step: int,
    best_loss: float,
    kind: str,
    batch_size: int | None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "epoch": epoch,
        "global_step": global_step,
        "batch_size": batch_size,
        "best_loss": best_loss,
        "model_class": model.__class__.__name__,
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "model_state": clean_state_dict(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "history": history.as_dict(),
    }


def save_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: TrainHistory,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    best_loss: float,
    kind: str,
    batch_size: int | None = None,
) -> Path:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = _checkpoint_stem(model, config)
    if kind == "best":
        path = checkpoint_dir / f"{stem}_best.pt"
    else:
        path = checkpoint_dir / f"{stem}_{kind}_step_{global_step:08d}.pt"

    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            history=history,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            kind=kind,
            batch_size=batch_size,
        ),
        path,
    )
    return path


def make_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def make_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def create_model_and_optimizer(
    model_or_factory: nn.Module | Callable[[], nn.Module],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    device: str | torch.device | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = model_or_factory() if callable(model_or_factory) and not isinstance(model_or_factory, nn.Module) else model_or_factory
    if not isinstance(model, nn.Module):
        raise TypeError("model_or_factory must be an nn.Module or a callable returning nn.Module")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay, betas=betas)
    return model, optimizer


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    return obj


def _real_view(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(tensor):
        return torch.view_as_real(tensor)
    return tensor


def _call_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple | list):
        return model(*inputs)
    return model(inputs)


def _profile_interpolate(
    x_nodes: torch.Tensor,
    area_nodes: torch.Tensor,
    node_count: torch.Tensor | None,
    *,
    n_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample padded variable-length tube profiles to a fixed normalized grid."""
    batch_size = area_nodes.shape[0]
    grid = torch.linspace(0.0, 1.0, n_points, device=area_nodes.device, dtype=area_nodes.dtype)
    areas = []
    xs_out = []

    for item_idx in range(batch_size):
        count = int(node_count[item_idx].item()) if node_count is not None else area_nodes.shape[1]
        count = max(count, 2)

        xs = x_nodes[item_idx, :count]
        ys = area_nodes[item_idx, :count]

        length = torch.clamp(xs[-1] - xs[0], min=torch.finfo(xs.dtype).eps)
        xs = (xs - xs[0]) / length
        xs = torch.clamp(xs, 0.0, 1.0)

        right = torch.searchsorted(xs.contiguous(), grid, right=False)
        right = torch.clamp(right, 1, count - 1)
        left = right - 1

        x0 = xs[left]
        x1 = xs[right]
        y0 = ys[left]
        y1 = ys[right]
        weight = (grid - x0) / torch.clamp(x1 - x0, min=torch.finfo(xs.dtype).eps)

        areas.append(y0 + weight * (y1 - y0))
        xs_out.append(grid)

    return torch.stack(xs_out, dim=0), torch.stack(areas, dim=0)


def make_webster_profile_features(
    batch: Mapping[str, Any],
    *,
    n_points: int = 128,
    log_area: bool = True,
    include_x: bool = True,
    channel_first: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert WebsterTorchDataset batches to fixed-size profile tensors.

    Output shape is [B, X, C] by default, or [B, C, X] when channel_first=True.
    This works both for MLPs (flatten in the model) and 1D FNOs/CNNs.
    """
    if "geometry" not in batch:
        raise KeyError("batch must contain geometry; set return_geometry=True in GeometryDatasetConfig")

    device = torch.device(device) if device is not None else None
    geometry = move_to_device(batch["geometry"], device) if device is not None else batch["geometry"]

    x_nodes = geometry["x_m"].float()
    area_nodes = geometry["area_m2"].float()
    node_count = geometry.get("node_count")
    if node_count is not None:
        node_count = node_count.to(x_nodes.device)

    x_grid, area_grid = _profile_interpolate(
        x_nodes,
        area_nodes,
        node_count,
        n_points=n_points,
    )

    if log_area:
        area_grid = torch.log(torch.clamp(area_grid, min=1e-12))

    channels = [area_grid]
    if include_x:
        channels.append(x_grid)

    features = torch.stack(channels, dim=-1)
    if channel_first:
        features = features.transpose(1, 2).contiguous()
    return features


def webster_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    n_points: int = 128,
    log_area: bool = True,
    include_x: bool = True,
    channel_first: bool = False,
    target_key: str = "target",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_webster_profile_features(
        batch,
        n_points=n_points,
        log_area=log_area,
        include_x=include_x,
        channel_first=channel_first,
        device=device,
    )
    y = batch[target_key].to(device)
    if not torch.is_complex(y):
        y = y.float()
    return x, y


def mapping_batch_to_xy(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    input_key: str = "input",
    target_key: str = "target",
) -> tuple[Any, torch.Tensor]:
    y = batch[target_key].to(device)
    if not torch.is_complex(y):
        y = y.float()
    return move_to_device(batch[input_key], device), y


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((_real_view(pred) - _real_view(target)) ** 2)


def mae_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(_real_view(pred) - _real_view(target)))


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse_loss(pred, target))


def relative_l2_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = _real_view(pred)
    target = _real_view(target)
    error = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return torch.mean(error / denom)


def _batch_frequencies(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    if "frequencies_hz" not in batch:
        raise KeyError("batch must contain frequencies_hz for spectral validation metrics")
    return batch["frequencies_hz"].to(device).float()


def _is_complex_like_target(target: torch.Tensor) -> bool:
    return torch.is_complex(target) or (target.ndim >= 3 and target.shape[-1] == 2)


def _magnitude_mae_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    if _is_complex_like_target(target):
        return spectral_metrics.magnitude_mae_db_from_complex(pred, target)
    return spectral_metrics.mae_db(pred.float(), target.float())


def _magnitude_rmse_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    if _is_complex_like_target(target):
        return spectral_metrics.magnitude_rmse_db_from_complex(pred, target)
    return spectral_metrics.rmse_db(pred.float(), target.float())


def _magnitude_max_abs_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    if _is_complex_like_target(target):
        return spectral_metrics.magnitude_max_abs_error_db_from_complex(pred, target)
    return spectral_metrics.max_abs_error_db(pred.float(), target.float())


def _relative_derivative_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = _batch_frequencies(batch, pred.device)
    return spectral_metrics.relative_frequency_derivative_l2(pred, target, frequencies)


def _dominant_peak_frequency_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = _batch_frequencies(batch, pred.device)
    if _is_complex_like_target(target):
        pred_db = spectral_metrics.complex_ri_to_db(pred)
        target_db = spectral_metrics.complex_ri_to_db(target)
    else:
        pred_db = pred.float()
        target_db = target.float()
    return spectral_metrics.dominant_peak_frequency_mae_hz(pred_db, target_db, frequencies)


def _dominant_peak_level_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = _batch_frequencies(batch, pred.device)
    if _is_complex_like_target(target):
        pred_db = spectral_metrics.complex_ri_to_db(pred)
        target_db = spectral_metrics.complex_ri_to_db(target)
    else:
        pred_db = pred.float()
        target_db = target.float()
    return spectral_metrics.dominant_peak_level_mae_db(pred_db, target_db, frequencies)


def _dominant_notch_frequency_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = _batch_frequencies(batch, pred.device)
    if _is_complex_like_target(target):
        pred_db = spectral_metrics.complex_ri_to_db(pred)
        target_db = spectral_metrics.complex_ri_to_db(target)
    else:
        pred_db = pred.float()
        target_db = target.float()
    return spectral_metrics.dominant_notch_frequency_mae_hz(pred_db, target_db, frequencies)


def _dominant_notch_level_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = _batch_frequencies(batch, pred.device)
    if _is_complex_like_target(target):
        pred_db = spectral_metrics.complex_ri_to_db(pred)
        target_db = spectral_metrics.complex_ri_to_db(target)
    else:
        pred_db = pred.float()
        target_db = target.float()
    return spectral_metrics.dominant_notch_level_mae_db(pred_db, target_db, frequencies)


def _relative_complex_l2_percent_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    if not _is_complex_like_target(target):
        return torch.tensor(float("nan"), device=pred.device)
    return spectral_metrics.relative_complex_l2_percent(pred, target)


def _phase_mae_degrees_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    if not _is_complex_like_target(target):
        return torch.tensor(float("nan"), device=pred.device)
    return spectral_metrics.phase_mae_degrees(pred, target)


DEFAULT_REGRESSION_METRICS: dict[str, MetricFn] = {
    "mae": mae_metric,
    "rmse": rmse_metric,
    "rel_l2": relative_l2_metric,
}


DEFAULT_VALIDATION_METRICS: dict[str, MetricFn] = {
    **DEFAULT_REGRESSION_METRICS,
    "magnitude_mae_db": _magnitude_mae_metric,
    "magnitude_rmse_db": _magnitude_rmse_metric,
    "magnitude_max_abs_error_db": _magnitude_max_abs_metric,
    "relative_derivative_l2": _relative_derivative_metric,
    "dominant_peak_frequency_mae_hz": _dominant_peak_frequency_metric,
    "dominant_peak_level_mae_db": _dominant_peak_level_metric,
    "dominant_notch_frequency_mae_hz": _dominant_notch_frequency_metric,
    "dominant_notch_level_mae_db": _dominant_notch_level_metric,
    "relative_complex_l2_percent": _relative_complex_l2_percent_metric,
    "phase_mae_degrees": _phase_mae_degrees_metric,
}


def resolve_metrics(metrics: MetricSpec) -> dict[str, MetricFn]:
    if metrics is None:
        return dict(DEFAULT_VALIDATION_METRICS)
    if isinstance(metrics, Mapping):
        return dict(metrics)

    resolved: dict[str, MetricFn] = {}
    unknown: list[str] = []
    for name in metrics:
        if name in DEFAULT_VALIDATION_METRICS:
            resolved[name] = DEFAULT_VALIDATION_METRICS[name]
        else:
            unknown.append(str(name))

    if unknown:
        available = ", ".join(sorted(DEFAULT_VALIDATION_METRICS))
        missing = ", ".join(unknown)
        raise KeyError(f"Unknown metric name(s): {missing}. Available metrics: {available}")

    return resolved


def _compute_metric(
    metric_fn: MetricFn,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor | float:
    try:
        return metric_fn(pred, target, batch)
    except TypeError:
        return metric_fn(pred, target)


def _compute_loss(
    criterion: CriterionFn,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    frequencies = batch.get("frequencies_hz")

    if frequencies is not None:
        try:
            loss = criterion(pred, target, frequencies.to(pred.device).float())
        except TypeError:
            loss = None
        else:
            return loss[0] if isinstance(loss, tuple) else loss

    try:
        loss = criterion(pred, target, batch)
    except TypeError:
        loss = criterion(pred, target)

    return loss[0] if isinstance(loss, tuple) else loss


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    criterion: CriterionFn = mse_loss,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    show_progress: bool = True,
    max_steps: int | None = None,
    loss_log: list[float] | None = None,
    on_step_end: StepCallback | None = None,
) -> float:
    device = torch.device(device or next(model.parameters()).device)
    model.train()
    losses: list[float] = []

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    iterator = tqdm(loader, total=max_steps, leave=False, disable=not show_progress)

    for step, batch in enumerate(iterator, start=1):
        inputs, target = batch_to_xy(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = _call_model(model, inputs)
            loss = _compute_loss(criterion, pred, target, batch)

        scaler.scale(loss).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_log is not None:
            loss_log.append(loss_value)
        iterator.set_postfix(loss=f"{loss_value:.4g}", lr=f"{get_lr(optimizer):.3g}")

        if on_step_end is not None:
            on_step_end(step, loss_value)

        if max_steps is not None and step >= max_steps:
            break

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CriterionFn = mse_loss,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    metrics: MetricSpec = None,
    show_progress: bool = True,
    max_steps: int | None = None,
    loss_log: list[float] | None = None,
    on_step_end: StepCallback | None = None,
) -> tuple[float, dict[str, float]]:
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    resolved_metrics = resolve_metrics(metrics)

    losses: list[float] = []
    metric_values: dict[str, list[float]] = {name: [] for name in resolved_metrics}

    iterator = tqdm(loader, total=max_steps, leave=False, disable=not show_progress)
    for step, batch in enumerate(iterator, start=1):
        inputs, target = batch_to_xy(batch, device)
        pred = _call_model(model, inputs)
        loss = _compute_loss(criterion, pred, target, batch)

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_log is not None:
            loss_log.append(loss_value)
        iterator.set_postfix(loss=f"{loss_value:.4g}")

        if on_step_end is not None:
            on_step_end(step, loss_value)

        for name, metric_fn in resolved_metrics.items():
            value = _compute_metric(metric_fn, pred, target, batch)
            if torch.is_tensor(value):
                value = float(value.detach().cpu())
            metric_values[name].append(float(value))

        if max_steps is not None and step >= max_steps:
            break

    mean_metrics = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in metric_values.items()
    }
    return float(np.mean(losses)) if losses else float("nan"), mean_metrics


def _step_scheduler(
    scheduler: Any,
    *,
    scheduler_on: str,
    train_loss: float,
    val_loss: float | None,
) -> None:
    if scheduler is None:
        return

    if scheduler_on == "epoch":
        scheduler.step()
        return

    value = val_loss if scheduler_on == "val_loss" else train_loss
    if value is None:
        return

    try:
        scheduler.step(value)
    except TypeError:
        scheduler.step()


def _live_plot_history(
    history: TrainHistory,
    *,
    show_epoch: bool,
) -> None:
    if plt is None:
        return
    if clear_output is not None:
        clear_output(wait=True)
    plot_history(
        history,
        show_steps=True,
        show_epoch=show_epoch,
        normalize_metrics=True,
    )


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    criterion: CriterionFn = mse_loss,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    scheduler: Any = None,
    metrics: MetricSpec = None,
    config: TrainConfig | None = None,
    resume_history: TrainHistory | Mapping[str, Any] | None = None,
) -> TrainHistory:
    config = config or TrainConfig()
    device = torch.device(config.device)
    model.to(device)
    resolved_metrics = resolve_metrics(metrics if metrics is not None else config.validation_metrics)

    history = make_train_history(resume_history)
    for name in resolved_metrics:
        history.metrics.setdefault(name, [])
    current_batch_size = _loader_batch_size(train_loader)
    if current_batch_size is not None:
        history.batch_size = current_batch_size

    initial_epoch_count = len(history.train_loss)
    initial_global_step = len(history.step_train_loss)
    initial_validation_count = len(history.val_loss)

    history.planned_epochs = initial_epoch_count + config.epochs
    if config.steps_per_epoch is not None:
        history.planned_train_steps = initial_global_step + config.epochs * config.steps_per_epoch
    if val_loader is not None:
        new_validation_count = sum(
            1
            for epoch in range(1, config.epochs + 1)
            if epoch % config.val_every == 0
        )
        history.planned_validations = initial_validation_count + new_validation_count
        if config.val_steps is not None:
            history.planned_val_steps = (
                len(history.step_val_loss) + new_validation_count * config.val_steps
            )
    if history.val_loss:
        best_loss = _best_finite(history.val_loss)
    elif history.train_loss:
        best_loss = _best_finite(history.train_loss)
    else:
        best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    if config.restore_best and np.isfinite(best_loss):
        best_state = clean_state_dict(copy.deepcopy(model.state_dict()))
    epochs_without_improvement = 0
    global_step = initial_global_step
    current_epoch = initial_epoch_count
    needs_train_step_callback = (
        (
            config.save_every_steps is not None
            and config.save_every_steps > 0
        )
        or (
            config.live_plot_every_steps is not None
            and config.live_plot_every_steps > 0
        )
    )
    needs_val_step_callback = (
        config.live_plot_every_steps is not None
        and config.live_plot_every_steps > 0
    )

    def on_train_step_end(_local_step: int, _loss_value: float) -> None:
        nonlocal global_step
        global_step += 1
        if (
            config.save_every_steps is not None
            and config.save_every_steps > 0
            and global_step % config.save_every_steps == 0
        ):
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stem = _checkpoint_stem(model, config)
            expected_path = checkpoint_dir / f"{stem}_intermediate_step_{global_step:08d}.pt"
            history.intermediate_checkpoint_paths.append(str(expected_path))
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                history=history,
                config=config,
                epoch=current_epoch,
                global_step=global_step,
                best_loss=best_loss,
                kind="intermediate",
                batch_size=current_batch_size,
            )

        if (
            config.live_plot_every_steps is not None
            and config.live_plot_every_steps > 0
            and global_step % config.live_plot_every_steps == 0
        ):
            _live_plot_history(
                history,
                show_epoch=config.live_plot_show_epoch,
            )

    def on_val_step_end(_local_step: int, _loss_value: float) -> None:
        if config.live_plot_every_steps is not None and config.live_plot_every_steps > 0:
            _live_plot_history(
                history,
                show_epoch=config.live_plot_show_epoch,
            )

    for epoch in range(1, config.epochs + 1):
        current_epoch = initial_epoch_count + epoch
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            criterion,
            batch_to_xy=batch_to_xy,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
            use_amp=config.use_amp,
            show_progress=config.show_progress,
            max_steps=config.steps_per_epoch,
            loss_log=history.step_train_loss,
            on_step_end=on_train_step_end if needs_train_step_callback else None,
        )
        history.train_loss.append(train_loss)
        history.lr.append(get_lr(optimizer))

        val_loss: float | None = None
        if val_loader is not None and epoch % config.val_every == 0:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                batch_to_xy=batch_to_xy,
                device=device,
                metrics=resolved_metrics,
                show_progress=config.show_progress,
                max_steps=config.val_steps,
                loss_log=history.step_val_loss,
                on_step_end=on_val_step_end if needs_val_step_callback else None,
            )
            history.val_loss.append(val_loss)
            for name, value in val_metrics.items():
                history.metrics.setdefault(name, []).append(value)

        _step_scheduler(
            scheduler,
            scheduler_on=config.scheduler_on,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        monitored_loss = val_loss if val_loss is not None else train_loss
        if monitored_loss < best_loss:
            best_loss = monitored_loss
            epochs_without_improvement = 0
            if config.restore_best:
                best_state = clean_state_dict(copy.deepcopy(model.state_dict()))
        else:
            epochs_without_improvement += 1

        total_epochs = history.planned_epochs or (initial_epoch_count + config.epochs)
        parts = [f"epoch {current_epoch:03d}/{total_epochs}", f"train={train_loss:.6g}"]
        if val_loss is not None:
            parts.append(f"val={val_loss:.6g}")
        parts.append(f"lr={get_lr(optimizer):.3g}")
        print(" | ".join(parts))

        if config.min_lr is not None and get_lr(optimizer) <= config.min_lr:
            print(f"stopped: learning rate reached {config.min_lr:g}")
            break

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(f"stopped: no improvement for {config.early_stopping_patience} epochs")
            break

    if config.restore_best and best_state is not None:
        load_model_state(model, best_state)

    if config.save_best:
        path = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            history=history,
            config=config,
            epoch=current_epoch,
            global_step=global_step,
            best_loss=best_loss,
            kind="best",
            batch_size=current_batch_size,
        )
        history.best_checkpoint_path = str(path)

    return history


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch in tqdm(loader, leave=False, disable=not show_progress):
        inputs, target = batch_to_xy(batch, device)
        preds.append(_call_model(model, inputs).detach().cpu())
        targets.append(target.detach().cpu())

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def validation_preview_batch(loader: torch.utils.data.DataLoader) -> Any:
    return next(iter(loader))


def predict_on_batch(
    model: nn.Module,
    batch_to_xy: BatchToXY,
    batch: Any,
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return predict(
        model,
        [batch],
        batch_to_xy=batch_to_xy,
        device=device,
        show_progress=False,
    )


def batch_geometry_to_explicit(batch: Mapping[str, Any], sample_idx: int) -> Any:
    if "geometry" not in batch:
        raise KeyError("batch does not contain geometry")

    import vt_all_solvers_wrapper as vt

    geom = batch["geometry"]
    n = int(geom["node_count"][sample_idx])

    return vt.ExplicitGeometry(
        x_m=geom["x_m"][sample_idx, :n].detach().cpu().tolist(),
        area_m2=geom["area_m2"][sample_idx, :n].detach().cpu().tolist(),
    )


def plot_batch_geometry(
    batch: Mapping[str, Any],
    sample_idx: int,
    ax: Any = None,
    *,
    title: str = "Channel geometry",
    mode: str = "symmetric",
    equal_aspect: bool = False,
    linewidth: float = 1.5,
) -> Any:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    import vt_all_solvers_wrapper as vt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    geometry = batch_geometry_to_explicit(batch, sample_idx)
    vt.plot_geometry(
        geometry,
        mode=mode,
        equal_aspect=equal_aspect,
        linewidth=linewidth,
        ax=ax,
        title=title,
    )
    ax.grid(True, alpha=0.25)
    return ax


def plot_model_prediction_on_channel(
    batch: Mapping[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
    model_label: str,
    *,
    sample_idx: int = 0,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")
    if sample_idx < 0 or sample_idx >= target.shape[0]:
        raise IndexError(
            f"sample_idx={sample_idx} is outside validation batch size {target.shape[0]}"
        )

    freq = batch["frequencies_hz"][sample_idx].detach().cpu()
    pred = prediction[sample_idx]
    y = target[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    plot_batch_geometry(
        batch,
        sample_idx,
        axes[0],
        title=f"Validation sample {sample_idx}",
    )

    axes[1].plot(freq, y, label="target dB", linewidth=2.0, color="black")
    axes[1].plot(freq, pred, label=model_label, linewidth=1.8)
    axes[1].set_xlabel("Frequency, Hz")
    axes[1].set_ylabel("Transfer function, dB")
    axes[1].set_title(model_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    mae = torch.mean(torch.abs(pred - y)).item()
    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
    print(f"{model_label} MAE dB:  {mae:.4f}")
    print(f"{model_label} RMSE dB: {rmse:.4f}")


def apply_diploma_plot_style(
    *,
    style_dir: str | Path = "style",
    font_size: int = 12,
    fig_width_cm: float = 18.0,
    fig_height_cm: float = 10.0,
    dpi: int = 140,
    closed_frame: bool = True,
    orange: str = "#FF7A00",
) -> dict[str, str]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    style_path = Path(style_dir).resolve()
    if str(style_path) not in sys.path:
        sys.path.insert(0, str(style_path))

    import diploma_style as ds

    ds.apply_style(
        font_size=font_size,
        fig_width_cm=fig_width_cm,
        fig_height_cm=fig_height_cm,
        dpi=dpi,
        closed_frame=closed_frame,
    )

    colors = dict(ds.COLORS)
    colors["orange"] = orange
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            colors["blue"],
            colors["orange"],
            colors["green"],
            colors["red"],
            colors["purple"],
            colors["cyan"],
            colors["pink"],
        ]
    )
    return colors


def figure_slug(title: str) -> str:
    text = str(title).lower()
    text = re.sub(r"[^0-9a-zа-яё]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "figure"


def save_figure(
    fig: Any = None,
    *,
    filename_title: str | None = None,
    output_dir: str | Path = "article/images",
    dpi: int = 300,
    hide_titles: bool = False,
    overwrite: bool = True,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    fig = plt.gcf() if fig is None else fig
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename_title is None:
        if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text().strip():
            filename_title = fig._suptitle.get_text().strip()
        else:
            filename_title = next(
                (ax.get_title().strip() for ax in fig.axes if ax.get_title().strip()),
                "figure",
            )

    if hide_titles:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
        for ax in fig.axes:
            ax.set_title("")

    stem = figure_slug(filename_title)
    path = output_path / f"{stem}.png"
    if not overwrite:
        index = 2
        while path.exists():
            path = output_path / f"{stem}_{index}.png"
            index += 1

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"saved figure: {path}")
    return path


def _iter_model_specs(model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]):
    if isinstance(model_specs, Mapping):
        for label, spec in model_specs.items():
            merged = dict(spec)
            merged.setdefault("label", str(label))
            yield merged
    else:
        for spec in model_specs:
            yield dict(spec)


def enabled_model_specs(
    model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    enabled = []
    for spec in _iter_model_specs(model_specs):
        label = str(spec.get("label", "model"))
        if spec.get("checkpoint_name") is None:
            if verbose:
                print(f"skip {label}: checkpoint_name is None")
            continue
        enabled.append(spec)
    return enabled


def load_models_from_specs(
    namespace: MutableMapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
    checkpoint_dir: str | Path = "checkpoints",
    verbose: bool = True,
) -> list[tuple[str, nn.Module | None, Any | None, BatchToXY]]:
    loaded = []
    for spec in enabled_model_specs(model_specs, verbose=verbose):
        label = str(spec["label"])
        model, history = get_or_load_model(
            namespace,
            variable_name=str(spec["variable_name"]),
            checkpoint_name=str(spec["checkpoint_name"]),
            factory=spec["factory"],
            history_variable_name=spec.get("history_name"),
            checkpoint_dir=checkpoint_dir,
            device=device,
            verbose=verbose,
        )
        loaded.append((label, model, history, spec["batch_to_xy"]))
    return loaded


def compare_forward_models(
    namespace: MutableMapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    batch: Mapping[str, Any] | None = None,
    loader: torch.utils.data.DataLoader | None = None,
    seed: int | None = None,
    target_batch_to_xy: BatchToXY,
    sample_idx: int = 0,
    device: str | torch.device | None = None,
    image_title: str = "Forward model comparison",
    output_dir: str | Path = "article/images",
    checkpoint_dir: str | Path = "checkpoints",
    style: bool = True,
    save: bool = True,
    show: bool = True,
    hide_titles: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")
    if batch is None:
        if loader is None:
            raise ValueError("Either batch or loader must be provided")
        if seed is not None:
            set_seed(seed)
        batch = validation_preview_batch(loader)

    colors = apply_diploma_plot_style() if style else {"black": "black"}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    models = load_models_from_specs(
        namespace,
        model_specs,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    _, target = target_batch_to_xy(batch, device)
    target = target.detach().cpu()
    if sample_idx < 0 or sample_idx >= target.shape[0]:
        raise IndexError(
            f"sample_idx={sample_idx} is outside validation batch size {target.shape[0]}"
        )

    predictions: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}
    freq = batch["frequencies_hz"][sample_idx].detach().cpu()
    y = target[sample_idx]

    for label, model, _, batch_to_xy in models:
        if model is None:
            continue

        prediction, _ = predict_on_batch(
            model,
            batch_to_xy,
            batch,
            device=device,
        )
        prediction = prediction.detach().cpu()
        predictions[label] = prediction

        pred = prediction[sample_idx]
        mae = torch.mean(torch.abs(pred - y)).item()
        rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
        metrics[label] = {"mae_db": mae, "rmse_db": rmse}
        print(f"{label:28s} MAE dB: {mae:8.4f} | RMSE dB: {rmse:8.4f}")

    if not predictions:
        print("No trained models are available for comparison. Run training cells or add checkpoints.")
    else:
        n_plots = len(predictions) + 1
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.6 * n_rows),
            squeeze=False,
        )
        flat_axes = list(axes.ravel())

        plot_batch_geometry(
            batch,
            sample_idx,
            flat_axes[0],
            title=f"Validation sample {sample_idx}",
            equal_aspect=False,
            linewidth=1.8,
        )

        for ax, (label, prediction) in zip(flat_axes[1:], predictions.items()):
            pred = prediction[sample_idx]
            ax.plot(
                freq,
                y,
                label="target dB",
                linewidth=2.4,
                color=colors.get("black", "black"),
            )
            ax.plot(
                freq,
                pred,
                label=label,
                linewidth=1.8,
                color=colors.get("orange", "C1"),
            )
            ax.set_xlabel("Frequency, Hz")
            ax.set_ylabel("Transfer function, dB")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in flat_axes[n_plots:]:
            ax.axis("off")

        plt.tight_layout()

        if save:
            save_figure(
                fig,
                filename_title=image_title,
                output_dir=output_dir,
                hide_titles=hide_titles,
                overwrite=overwrite,
            )
        if show:
            plt.show()
        plt.close(fig)

    return {
        "batch": batch,
        "target": target,
        "predictions": predictions,
        "metrics": metrics,
    }


def compare_training_histories(
    namespace: MutableMapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    image_title: str = "Forward models training history comparison",
    output_dir: str | Path = "article/images",
    checkpoint_dir: str | Path = "checkpoints",
    style: bool = True,
    save: bool = True,
    show: bool = True,
    hide_titles: bool = False,
    overwrite: bool = True,
    yscale: str | None = "log",
    curves: Sequence[str] = ("val", "train"),
) -> list[tuple[str, Any]]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    if style:
        apply_diploma_plot_style()

    requested_curves = {str(curve).lower() for curve in curves}
    unknown_curves = requested_curves - {"train", "val"}
    if unknown_curves:
        raise ValueError(f"unknown history curve(s): {sorted(unknown_curves)}")

    history_specs = []
    for spec in enabled_model_specs(model_specs):
        history = get_or_load_history(
            namespace,
            str(spec["history_name"]),
            str(spec["checkpoint_name"]),
            checkpoint_dir=checkpoint_dir,
        )
        history_specs.append((str(spec["label"]), history))

    print("=== Loaded history lengths ===")
    for name, history in history_specs:
        print(
            f"{name:28s} "
            f"train={len(history_get(history, 'train_loss')):3d} "
            f"val={len(history_get(history, 'val_loss')):3d} "
            f"train_steps={len(history_get(history, 'step_train_loss')):4d} "
            f"val_steps={len(history_get(history, 'step_val_loss')):4d}"
        )

    available = [
        (name, history)
        for name, history in history_specs
        if history_get(history, "train_loss") or history_get(history, "val_loss")
    ]

    if not available:
        print("No training histories are available. Run training cells or load checkpoints with history.")
        return []

    fig, ax = plt.subplots(figsize=(11, 5.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if len(available) > len(color_cycle):
        color_cycle = [plt.get_cmap("tab20")(i) for i in range(20)]
    for index, (name, history) in enumerate(available):
        color = color_cycle[index % len(color_cycle)]
        val_loss = history_get(history, "val_loss")
        train_loss = history_get(history, "train_loss")
        if "val" in requested_curves and val_loss:
            ax.plot(
                val_loss,
                marker="o",
                linewidth=1.8,
                color=color,
                label=f"{name} val",
            )
        if "train" in requested_curves and train_loss:
            ax.plot(
                train_loss,
                linestyle="--",
                linewidth=1.4,
                alpha=0.65,
                color=color,
                label=f"{name} train",
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_title("Training history comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    if save:
        save_figure(
            fig,
            filename_title=image_title,
            output_dir=output_dir,
            hide_titles=hide_titles,
            overwrite=overwrite,
        )
    if show:
        plt.show()
    plt.close(fig)

    print("=== Best losses ===")
    for name, history in available:
        val_loss = history_get(history, "val_loss")
        train_loss = history_get(history, "train_loss")
        best_train = min(train_loss) if train_loss else None
        best_val = min(val_loss) if val_loss else None
        print(f"{name:28s} best train loss: {best_train} | best val loss: {best_val}")

    return available


def compare_training_metrics(
    namespace: MutableMapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    metric_names: Sequence[str] = ("mae", "rmse", "magnitude_mae_db"),
    *,
    image_title: str = "Forward models validation metrics",
    output_dir: str | Path = "article/images",
    checkpoint_dir: str | Path = "checkpoints",
    style: bool = True,
    save: bool = True,
    show: bool = True,
    hide_titles: bool = False,
    overwrite: bool = True,
    yscale: str | None = None,
    best_mode: str = "min",
) -> dict[str, list[dict[str, Any]]]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")
    if best_mode not in {"min", "max"}:
        raise ValueError("best_mode must be 'min' or 'max'")

    if style:
        apply_diploma_plot_style()

    history_specs = []
    for spec in enabled_model_specs(model_specs):
        history = get_or_load_history(
            namespace,
            str(spec["history_name"]),
            str(spec["checkpoint_name"]),
            checkpoint_dir=checkpoint_dir,
        )
        history_specs.append((str(spec["label"]), history))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if len(history_specs) > len(color_cycle):
        color_cycle = [plt.get_cmap("tab20")(i) for i in range(20)]

    summary: dict[str, list[dict[str, Any]]] = {}

    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        metric_rows: list[dict[str, Any]] = []
        plotted = False

        for index, (model_name, history) in enumerate(history_specs):
            metrics = history_to_mapping(history).get("metrics", {}) if history is not None else {}
            values = metrics.get(metric_name, []) if isinstance(metrics, Mapping) else []
            xs, ys = _finite_series(values)
            if not ys:
                continue

            color = color_cycle[index % len(color_cycle)]
            xs_one_based = _one_based(xs)
            ax.plot(
                xs_one_based,
                ys,
                marker="o",
                linewidth=1.8,
                color=color,
                label=model_name,
            )
            plotted = True

            best_index = int(np.argmin(ys) if best_mode == "min" else np.argmax(ys))
            metric_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "best_value": float(ys[best_index]),
                    "validation": int(xs_one_based[best_index]),
                }
            )

        summary[metric_name] = metric_rows

        if not plotted:
            plt.close(fig)
            print(f"metric not found or empty for all models: {metric_name}")
            continue

        ax.set_xlabel("validation")
        ax.set_ylabel(metric_name)
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.set_title(metric_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        if save:
            save_figure(
                fig,
                filename_title=f"{image_title} {metric_name}",
                output_dir=output_dir,
                hide_titles=hide_titles,
                overwrite=overwrite,
            )
        if show:
            plt.show()
        plt.close(fig)

        print(f"=== Best {metric_name} ({best_mode}) ===")
        print(f"{'model':28s} | {'best_value':>14s} | {'validation':>10s}")
        print("-" * 59)
        for row in sorted(metric_rows, key=lambda item: item["best_value"], reverse=(best_mode == "max")):
            print(
                f"{row['model']:28s} | "
                f"{row['best_value']:14.6g} | "
                f"{row['validation']:10d}"
            )

    return summary


def plot_single_model_preview(
    model: nn.Module,
    batch_to_xy: BatchToXY,
    model_label: str,
    *,
    loader: torch.utils.data.DataLoader,
    device: str | torch.device | None = None,
    sample_idx: int = 0,
) -> None:
    batch = validation_preview_batch(loader)
    prediction, target = predict_on_batch(
        model,
        batch_to_xy,
        batch,
        device=device,
    )
    plot_model_prediction_on_channel(
        batch,
        prediction,
        target,
        model_label,
        sample_idx=sample_idx,
    )


def make_single_geometry_batch(
    geometry: Any,
    solver_config: Any,
    *,
    geometry_kind: str | None = None,
) -> dict[str, Any]:
    import vt_all_solvers_wrapper as vt

    result = vt.solve(
        geometry,
        config=solver_config,
    )

    target_db = 20.0 * torch.log10(
        torch.tensor(result.magnitude, dtype=torch.float32).clamp_min(1e-12)
    )

    x_m, area_m2, segment_lengths_m = vt.geometry_to_tube_tuple(geometry)

    return {
        "target": target_db.unsqueeze(0),
        "frequencies_hz": torch.tensor(result.frequencies_hz, dtype=torch.float32).unsqueeze(0),
        "geometry": {
            "x_m": torch.tensor([x_m], dtype=torch.float32),
            "area_m2": torch.tensor([area_m2], dtype=torch.float32),
            "segment_lengths_m": torch.tensor([segment_lengths_m], dtype=torch.float32),
            "node_count": torch.tensor([len(x_m)], dtype=torch.int64),
            "segment_count": torch.tensor([len(segment_lengths_m)], dtype=torch.int64),
        },
        "meta": [
            {
                "idx": 0,
                "seed": None,
                "solver": solver_config.solver,
                "geometry_kind_requested": geometry_kind,
            }
        ],
    }


def make_unexpected_geometry(kind: str) -> Any:
    import vt_all_solvers_wrapper as vt

    if kind == "tube_with_hole":
        return vt.make_tube_with_hole_geometry(
            length_m=1.0,
            base_width_m=0.015,
            hole_center_m=0.58,
        )

    if kind == "sharp_bottleneck":
        return vt.explicit_geometry_from_arrays(
            x_m=[0.0, 0.18, 0.32, 0.50, 0.68, 0.82, 1.0],
            area_m2=[8.0e-4, 8.0e-4, 1.6e-4, 1.2e-4, 1.6e-4, 8.0e-4, 8.0e-4],
        )

    raise ValueError(f"Unsupported unexpected geometry kind: {kind!r}")


def _finite_series(values: Sequence[float]) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for idx, value in enumerate(values):
        value = float(value)
        if np.isfinite(value):
            xs.append(idx)
            ys.append(value)
    return xs, ys


def _one_based(values: list[int]) -> list[int]:
    return [value + 1 for value in values]


def _set_integer_xlim(ax: Any, xs: Sequence[int]) -> None:
    if not xs:
        return
    if len(xs) == 1:
        ax.set_xlim(xs[0] - 0.5, xs[0] + 0.5)
    else:
        ax.set_xlim(min(xs), max(xs))


def _set_planned_xlim(
    ax: Any,
    planned_count: int | None,
    fallback_xs: Sequence[int],
) -> None:
    if planned_count is not None and planned_count > 0:
        if planned_count == 1:
            ax.set_xlim(0.5, 1.5)
        else:
            ax.set_xlim(1, planned_count)
        return
    _set_integer_xlim(ax, fallback_xs)


def _has_finite_values(values: Sequence[float]) -> bool:
    return bool(_finite_series(values)[1])


def _normalize_to_first(values: Sequence[float], eps: float = 1e-12) -> list[float]:
    if not values:
        return []
    first = float(values[0])
    denominator = first if abs(first) > eps else 1.0
    return [float(value) / denominator for value in values]


def plot_history(
    history: TrainHistory | Mapping[str, Any],
    *,
    show_steps: bool = True,
    show_epoch: bool = True,
    normalize_metrics: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    data = history_to_mapping(history)
    train_loss = data.get("train_loss", [])
    val_loss = data.get("val_loss", [])
    step_train_loss = data.get("step_train_loss", [])
    step_val_loss = data.get("step_val_loss", [])
    metrics = data.get("metrics", {})
    planned_train_steps = data.get("planned_train_steps")
    planned_val_steps = data.get("planned_val_steps")
    planned_epochs = data.get("planned_epochs")
    planned_validations = data.get("planned_validations")

    panels = 0
    if show_steps and (_has_finite_values(step_train_loss) or _has_finite_values(step_val_loss)):
        panels += 1
    if show_epoch and (_has_finite_values(train_loss) or _has_finite_values(val_loss)):
        panels += 1
    if metrics:
        panels += 1

    if panels == 0:
        raise ValueError(
            "No finite values to plot. "
            f"Lengths: train_loss={len(train_loss)}, val_loss={len(val_loss)}, "
            f"step_train_loss={len(step_train_loss)}, step_val_loss={len(step_val_loss)}. "
            "Check that fit(...) finished at least one batch and that you are plotting the returned history."
        )

    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4))
    if panels == 1:
        axes = [axes]
    axis_idx = 0

    if show_steps and (_has_finite_values(step_train_loss) or _has_finite_values(step_val_loss)):
        ax = axes[axis_idx]
        all_step_xs: list[int] = []
        xs, ys = _finite_series(step_train_loss)
        if ys:
            xs = _one_based(xs)
            all_step_xs.extend(xs)
            ax.plot(xs, ys, label="train step", linewidth=1)
        xs, ys = _finite_series(step_val_loss)
        if ys:
            xs = _one_based(xs)
            all_step_xs.extend(xs)
            ax.plot(xs, ys, label="val step", linewidth=1)
        planned_step_count = None
        if planned_train_steps is not None or planned_val_steps is not None:
            planned_step_count = max(int(planned_train_steps or 0), int(planned_val_steps or 0))
        _set_planned_xlim(ax, planned_step_count, all_step_xs)
        ax.set_title("loss per batch")
        ax.set_xlabel("optimizer step")
        ax.grid(True)
        ax.legend()
        axis_idx += 1

    if show_epoch and (_has_finite_values(train_loss) or _has_finite_values(val_loss)):
        ax = axes[axis_idx]
        all_epoch_xs: list[int] = []
        xs, ys = _finite_series(train_loss)
        if ys:
            xs = _one_based(xs)
            all_epoch_xs.extend(xs)
            ax.plot(xs, ys, ".-", label="train epoch")
        xs, ys = _finite_series(val_loss)
        if ys:
            xs = _one_based(xs)
            all_epoch_xs.extend(xs)
            ax.plot(xs, ys, ".-", label="val epoch")
        _set_planned_xlim(ax, planned_epochs, all_epoch_xs)
        ax.set_title("mean loss")
        ax.set_xlabel("epoch")
        ax.grid(True)
        ax.legend()
        axis_idx += 1

    if metrics:
        ax = axes[axis_idx]
        all_metric_xs: list[int] = []
        for name, values in metrics.items():
            xs, ys = _finite_series(values)
            if ys:
                xs = _one_based(xs)
                if normalize_metrics:
                    ys = _normalize_to_first(ys)
                all_metric_xs.extend(xs)
                ax.plot(xs, ys, ".-", label=name)
        _set_planned_xlim(ax, planned_validations, all_metric_xs)
        ax.set_title("metrics / first value" if normalize_metrics else "metrics")
        ax.set_xlabel("validation")
        if normalize_metrics:
            ax.set_ylabel("relative to first value")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_selected_metrics(
    history: TrainHistory | Mapping[str, Any] | None,
    metric_names: Sequence[str],
    *,
    title: str = "Selected validation metrics",
    xlabel: str = "validation",
    figsize: tuple[float, float] = (10, 4),
    normalize: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    if history is None:
        print("history is None")
        return

    data = history_to_mapping(history)
    metrics = data.get("metrics", {})
    if not metrics:
        print("history has no metrics")
        return

    fig, ax = plt.subplots(figsize=figsize)
    plotted = False

    for name in metric_names:
        values = metrics.get(name, [])
        xs, ys = _finite_series(values)
        if not ys:
            print(f"metric not found or empty: {name}")
            continue

        if normalize:
            ys = _normalize_to_first(ys)
        ax.plot(_one_based(xs), ys, ".-", label=name)
        plotted = True

    if not plotted:
        available = ", ".join(sorted(metrics))
        print(f"nothing to plot. Available metrics: {available}")
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative to first value" if normalize else "metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    criterion: CriterionFn = mse_loss,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, torch.optim.Optimizer, float]:
    loss = train_one_epoch(
        model,
        optimizer,
        loader,
        criterion,
        batch_to_xy=batch_to_xy,
        device=device,
        **kwargs,
    )
    return model, optimizer, loss


def val(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CriterionFn = mse_loss,
    *,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    metric_names: Sequence[str] | None = None,
    metrics: MetricSpec = None,
    **kwargs: Any,
) -> tuple[float, dict[str, float] | None]:
    selected_metrics = metrics if metrics is not None else metric_names
    loss, metric_values = evaluate(
        model,
        loader,
        criterion,
        batch_to_xy=batch_to_xy,
        device=device,
        metrics=selected_metrics,
        **kwargs,
    )
    return loss, metric_values if selected_metrics is not None else None


def learning_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    criterion: CriterionFn = mse_loss,
    *,
    scheduler: Any = None,
    epochs: int = 10,
    val_every: int = 1,
    metric_names: Sequence[str] | None = None,
    metrics: MetricSpec = None,
    batch_to_xy: BatchToXY = webster_batch_to_xy,
    device: str | torch.device | None = None,
    min_lr: float | None = None,
    steps_per_epoch: int | None = None,
    val_steps: int | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, torch.optim.Optimizer, dict[str, list[float]], list[float], dict[str, list[float]] | None]:
    selected_metrics = metrics if metrics is not None else metric_names

    history = fit(
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        batch_to_xy=batch_to_xy,
        scheduler=scheduler,
        metrics=selected_metrics,
        config=TrainConfig(
            epochs=epochs,
            val_every=val_every,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            min_lr=min_lr,
            **kwargs,
        ),
    )

    losses = {"train": history.train_loss, "val": history.val_loss}
    return model, optimizer, losses, history.lr, history.metrics if selected_metrics is not None else None
