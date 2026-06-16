"""
Стиль графиков по ГОСТ 7.32-2017 / ГОСТ 2.105-2019 для дипломной работы.

Использование
=============

  import diploma_style_gost as gost
  gost.apply_style()

  fig, ax = plt.subplots()
  ax.plot(x, y1, **gost.line(0))   # сплошная + кружок
  ax.plot(x, y2, **gost.line(1))   # штрих  + квадрат
  ...
  gost.save(fig, "spectrum")       # -> figures/spectrum.pdf + .png

Требования ГОСТ
----------------
- Шрифт Times New Roman (или STIX Two Text как замена)
- Все 4 рамки осей видны
- Чёрно-белые линии, различаемые стилем и маркерами
- Подписи осей: «Название, единица измерения»
- Размер шрифта 12–14 pt
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Стили линий: каждая серия отличается ТИПОМ линии + МАРКЕРОМ
# (обязательно для ГОСТ, чтобы различать кривые в ч/б печати)
# ---------------------------------------------------------------------------
_LINE_DEFS: list[dict] = [
    dict(linestyle="-",   marker="o", markevery=0.12, markersize=5),
    dict(linestyle="--",  marker="s", markevery=0.12, markersize=5),
    dict(linestyle="-.",  marker="^", markevery=0.12, markersize=5.5),
    dict(linestyle=":",   marker="D", markevery=0.12, markersize=4.5),
    dict(linestyle="-",   marker="v", markevery=0.12, markersize=5),
    dict(linestyle="--",  marker="P", markevery=0.12, markersize=5.5),
]


def line(index: int, **overrides) -> dict:
    """Вернуть kwargs для ``ax.plot(...)`` с ГОСТ-различимым стилем.

    >>> ax.plot(x, y, **gost.line(0), label="Cylinder TLM")
    """
    d = dict(
        _LINE_DEFS[index % len(_LINE_DEFS)],
        color="black",
        linewidth=1.4,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
    )
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Заливки для столбчатых диаграмм (штриховка вместо цвета)
# ---------------------------------------------------------------------------
_HATCH_CYCLE = ["///", "\\\\\\", "xxx", "...", "---", "+++"]


def bar_style(index: int, **overrides) -> dict:
    """Kwargs для ``ax.bar(...)`` с ГОСТ-штриховкой."""
    d = dict(
        color="white",
        edgecolor="black",
        linewidth=0.9,
        hatch=_HATCH_CYCLE[index % len(_HATCH_CYCLE)],
    )
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Основная функция стилизации
# ---------------------------------------------------------------------------
def _font_family() -> list[str]:
    return ["Times New Roman", "STIX Two Text", "CMU Serif", "DejaVu Serif", "serif"]


def apply_style(
    *,
    font_size: int = 13,
    fig_width_cm: float = 16.0,
    fig_height_cm: float = 10.0,
    dpi: int = 150,
    use_latex: bool = False,
) -> None:
    """Применить ГОСТ-стиль ко всем последующим графикам.

    Parameters
    ----------
    font_size : int
        Базовый размер шрифта (pt). ГОСТ рекомендует 12–14.
    fig_width_cm, fig_height_cm : float
        Размер фигуры по умолчанию (см). 16 см ≈ ширина текстового поля A4.
    """
    w_inch = fig_width_cm / 2.54
    h_inch = fig_height_cm / 2.54

    rc: dict = {
        # --- Размер ---
        "figure.figsize": (w_inch, h_inch),
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,

        # --- Шрифты (Times New Roman — ГОСТ) ---
        "font.family": "serif",
        "font.serif": _font_family(),
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,

        # --- Математика ---
        "mathtext.fontset": "stix",

        # --- Фон ---
        "axes.facecolor": "white",
        "figure.facecolor": "white",

        # --- Все 4 рамки (ГОСТ) ---
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,

        # --- Сетка ---
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,

        # --- Тики внутрь (ГОСТ) ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",

        # --- Линии ---
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "lines.color": "black",

        # --- Cycler: всё чёрное ---
        "axes.prop_cycle": mpl.cycler(color=["black"]),

        # --- Легенда ---
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 2.5,

        # --- Заголовок без жирности ---
        "axes.titleweight": "normal",
        "axes.titlepad": 10,
        "axes.labelpad": 6,
    }

    if use_latex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}"
                r"\usepackage{amssymb}"
                r"\usepackage{mathptmx}"   # Times для LaTeX
            ),
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
    """Сохранить фигуру в ``figures/<name>.pdf`` и ``.png``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(str(p))
        paths.append(p)
    return paths


def caption(fig: plt.Figure, number: int, text: str) -> None:
    """Добавить подпись под графиком в формате ГОСТ:
    ``Рисунок <number> — <text>``
    """
    fig.text(
        0.5, -0.02,
        f"Рисунок {number} \u2014 {text}",
        ha="center",
        fontsize=mpl.rcParams["font.size"],
        fontstyle="normal",
    )


def legend_outside(
    ax: plt.Axes,
    loc: str = "upper right",
    *,
    ncol: int = 1,
) -> None:
    """Разместить легенду снаружи рабочей области."""
    if loc == "upper right":
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0, ncol=ncol)
    elif loc == "lower center":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  borderaxespad=0, ncol=ncol)
    else:
        ax.legend(loc=loc, ncol=ncol)


def annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    offset: tuple[float, float] = (12, 12),
) -> None:
    """Аннотация со стрелкой в чёрном стиле."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
        fontsize=mpl.rcParams["font.size"] - 2,
        color="black",
    )
