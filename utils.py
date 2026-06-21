from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


def _safe_checkpoint_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned or "model"


def _checkpoint_stem(model: nn.Module, config: TrainConfig) -> str:
    name = config.checkpoint_name or getattr(model, "model_name", None) or model.__class__.__name__
    return _safe_checkpoint_stem(str(name))


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: TrainHistory,
    epoch: int,
    global_step: int,
    best_loss: float,
    kind: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "epoch": epoch,
        "global_step": global_step,
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
    plot_history(history, show_steps=True, show_epoch=show_epoch)


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


def plot_history(
    history: TrainHistory | Mapping[str, Any],
    *,
    show_steps: bool = True,
    show_epoch: bool = True,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    data = history.as_dict() if isinstance(history, TrainHistory) else history
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
                all_metric_xs.extend(xs)
                ax.plot(xs, ys, ".-", label=name)
        _set_planned_xlim(ax, planned_validations, all_metric_xs)
        ax.set_title("metrics")
        ax.set_xlabel("validation")
        ax.grid(True)
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
