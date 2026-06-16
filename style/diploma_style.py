"""
Стиль графиков для дипломной работы (Nature / IEEE).

Использование
=============

  import diploma_style as ds
  ds.apply_style()          # один раз в начале ноутбука

  fig, ax = plt.subplots()
  ax.plot(x, y, color=ds.COLORS["blue"])
  ...
  ds.save(fig, "spectrum")  # -> figures/spectrum.pdf + .png

Палитра
-------
Тёплые, но приглушённые оттенки, различимые при ч/б печати
и для людей с дальтонизмом (palette close to «Wong»).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Цветовая палитра (вдохновлена Wong palette + Nature guidelines)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "blue":       "#0072B2",   # основной
    "orange":     "#E69F00",   # акцент
    "green":      "#009E73",   # третий ряд
    "red":        "#D55E00",   # ошибка / выделение
    "purple":     "#CC79A7",   # дополнительный
    "cyan":       "#56B4E9",   # светлый акцент
    "yellow":     "#F0E442",   # маркер
    "black":      "#000000",
    "grey":       "#999999",
}

# Последовательность для cycler — порядок подобран для максимальной
# различимости соседних кривых.
COLOR_CYCLE: list[str] = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["cyan"],
]

# Последовательность маркеров для дополнительной различимости.
MARKER_CYCLE: list[str] = ["o", "s", "^", "D", "v", "P"]

# ---------------------------------------------------------------------------
# Стили линий для отдельных серий
# ---------------------------------------------------------------------------
LINE_STYLES: list[str] = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]


def _font_family() -> list[str]:
    """Предпочтительные шрифты: STIX Two Text → CMU Serif → DejaVu Serif."""
    return ["STIX Two Text", "CMU Serif", "DejaVu Serif", "serif"]


# ---------------------------------------------------------------------------
# Основная функция стилизации
# ---------------------------------------------------------------------------
def apply_style(
    *,
    font_size: int = 11,
    fig_width_cm: float = 16.0,
    fig_height_cm: float = 10.0,
    dpi: int = 150,
    use_latex: bool = False,
) -> None:
    """Применить стиль ко всем последующим графикам.

    Parameters
    ----------
    font_size : int
        Базовый размер шрифта (pt).
    fig_width_cm, fig_height_cm : float
        Размер фигуры по умолчанию в сантиметрах.
    dpi : int
        Разрешение растровых графиков.
    use_latex : bool
        Если ``True``, включается рендеринг подписей через LaTeX.
        Требует установленного TeX-дистрибутива.
    """
    w_inch = fig_width_cm / 2.54
    h_inch = fig_height_cm / 2.54

    # Cycler для автоматического чередования цветов + маркеров
    from cycler import cycler

    color_cy = cycler(color=COLOR_CYCLE)

    rc: dict = {
        # --- Размер фигуры ---
        "figure.figsize": (w_inch, h_inch),
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # --- Шрифты ---
        "font.family": "serif",
        "font.serif": _font_family(),
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,

        # --- Математика ---
        "mathtext.fontset": "stix",

        # --- Цвета ---
        "axes.prop_cycle": color_cy,
        "axes.facecolor": "white",
        "figure.facecolor": "white",

        # --- Оси ---
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "axes.titlepad": 8,
        "axes.labelpad": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # --- Сетка ---
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,

        # --- Тики ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # --- Линии ---
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # --- Легенда ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.8,
    }

    if use_latex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })

    mpl.rcParams.update(rc)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def save(
    fig: plt.Figure,
    name: str,
    *,
    folder: str | Path = "figures",
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """Сохранить фигуру в ``figures/<name>.pdf`` и ``figures/<name>.png``.

    Возвращает список путей к сохранённым файлам.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(str(p))
        paths.append(p)
    return paths


def legend_outside(
    ax: plt.Axes,
    loc: str = "upper right",
    *,
    ncol: int = 1,
) -> None:
    """Разместить легенду снаружи рабочей области графика.

    Parameters
    ----------
    loc : str
        ``'upper right'`` — справа сверху (по умолчанию).
        ``'lower center'`` — под графиком.
    """
    if loc == "upper right":
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            ncol=ncol,
        )
    elif loc == "lower center":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            borderaxespad=0,
            ncol=ncol,
        )
    else:
        ax.legend(loc=loc, ncol=ncol)


def annotate_max(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fmt: str = "{:.1f}",
    offset: tuple[float, float] = (10, 10),
    color: str | None = None,
) -> None:
    """Поставить стрелку-аннотацию в точку максимума кривой."""
    idx = int(np.argmax(y))
    ax.annotate(
        fmt.format(y[idx]),
        xy=(x[idx], y[idx]),
        xytext=offset,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=color or COLORS["red"], lw=1.2),
        fontsize=9,
        color=color or COLORS["red"],
    )


def dual_y_axis(
    ax: plt.Axes,
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    label1: str = "",
    label2: str = "",
    ylabel1: str = "",
    ylabel2: str = "",
) -> plt.Axes:
    """Построить два ряда данных на одном графике с двумя осями Y.

    Возвращает правую ось (ax2).
    """
    ax.plot(x, y1, color=COLORS["blue"], label=label1)
    ax.set_ylabel(ylabel1, color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])

    ax2 = ax.twinx()
    ax2.plot(x, y2, color=COLORS["orange"], label=label2)
    ax2.set_ylabel(ylabel2, color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["orange"])
    return ax2
