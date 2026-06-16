"""
Демо-скрипт: показывает стиль diploma_style на типичных графиках диплома.

Запуск:
    python diploma_style_demo.py

Результат — сохранённые PDF/PNG в папке figures/ и plt.show().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import style.diploma_style as ds

ds.apply_style()

# ---------------------------------------------------------------------------
# 1. Передаточная функция — сравнение солверов
# ---------------------------------------------------------------------------
f = np.linspace(50, 5000, 512)

# Синтетические данные, имитирующие акустическую передаточную функцию трубы
def synth_tf(f, peaks, widths, amps):
    tf = np.ones_like(f) * 0.3
    for p, w, a in zip(peaks, widths, amps):
        tf += a * np.exp(-0.5 * ((f - p) / w) ** 2)
    return tf

H_cyl = synth_tf(f, [420, 1250, 2200, 3500], [80, 120, 150, 200], [1.0, 0.7, 0.5, 0.35])
H_cone = synth_tf(f, [430, 1280, 2250, 3550], [75, 110, 145, 195], [0.95, 0.65, 0.48, 0.30])
H_web = synth_tf(f, [415, 1240, 2180, 3480], [85, 125, 155, 205], [1.05, 0.72, 0.52, 0.38])

fig1, ax1 = plt.subplots()
ax1.plot(f, 20 * np.log10(H_cyl),  label="Cylinder TLM")
ax1.plot(f, 20 * np.log10(H_cone), label="Cone model")
ax1.plot(f, 20 * np.log10(H_web),  label="Webster equation", linestyle="--")
ax1.set_xlabel("Частота, Гц")
ax1.set_ylabel("Амплитуда, дБ")
ax1.set_title("Передаточная функция трубы переменного сечения")
ax1.legend()
ds.save(fig1, "transfer_function_comparison")

# ---------------------------------------------------------------------------
# 2. Профиль площади сечения + аппроксимация
# ---------------------------------------------------------------------------
x = np.linspace(0, 0.17, 200)
area = 3e-4 + 1e-4 * np.sin(2 * np.pi * x / 0.17) - 1.5e-4 * x / 0.17

n_sec = 10
x_sec = np.linspace(0, 0.17, n_sec + 1)
area_sec = np.interp(x_sec, x, area)
# ступенчатая аппроксимация
x_step, area_step = [], []
for i in range(n_sec):
    mid_area = 0.5 * (area_sec[i] + area_sec[i + 1])
    x_step.extend([x_sec[i], x_sec[i + 1]])
    area_step.extend([mid_area, mid_area])

fig2, ax2 = plt.subplots()
ax2.plot(x * 100, area * 1e4, color=ds.COLORS["black"], linewidth=2, label="Исходный профиль")
ax2.plot(np.array(x_step) * 100, np.array(area_step) * 1e4,
         color=ds.COLORS["blue"], linewidth=1.5, label=f"Цилиндры ({n_sec})")
ax2.set_xlabel("Длина, см")
ax2.set_ylabel(r"Площадь сечения, см$^2$")
ax2.set_title("Аппроксимация геометрии трубы")
ax2.legend()
ds.save(fig2, "geometry_approximation")

# ---------------------------------------------------------------------------
# 3. Сходимость обучения нейросети (loss)
# ---------------------------------------------------------------------------
np.random.seed(42)
epochs = np.arange(1, 81)
train_loss = 1.2 * np.exp(-0.05 * epochs) + 0.02 + 0.015 * np.random.randn(80)
val_loss = 1.3 * np.exp(-0.045 * epochs) + 0.035 + 0.02 * np.random.randn(80)
train_loss = np.maximum(train_loss, 0.01)
val_loss = np.maximum(val_loss, 0.02)

fig3, ax3 = plt.subplots()
ax3.plot(epochs, train_loss, label="Train loss")
ax3.plot(epochs, val_loss,   label="Validation loss")
ax3.set_xlabel("Эпоха")
ax3.set_ylabel("MSE")
ax3.set_title("Сходимость обучения MLP-FNO модели")
ax3.legend()
# аннотация минимума
best_epoch = int(np.argmin(val_loss))
ax3.annotate(
    f"best = {val_loss[best_epoch]:.4f}",
    xy=(epochs[best_epoch], val_loss[best_epoch]),
    xytext=(15, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color=ds.COLORS["red"], lw=1.2),
    fontsize=9,
    color=ds.COLORS["red"],
)
ds.save(fig3, "training_convergence")

# ---------------------------------------------------------------------------
# 4. Ошибка предсказания модели (scatter)
# ---------------------------------------------------------------------------
y_true = np.random.rand(120)
y_pred = y_true + 0.06 * np.random.randn(120)

fig4, ax4 = plt.subplots(figsize=(5.5, 5.5))
ax4.scatter(y_true, y_pred, s=18, alpha=0.7, color=ds.COLORS["blue"], edgecolors="none")
lims = [min(y_true.min(), y_pred.min()) - 0.05, max(y_true.max(), y_pred.max()) + 0.05]
ax4.plot(lims, lims, "--", color=ds.COLORS["grey"], linewidth=1, label="Идеальное совпадение")
ax4.set_xlim(lims)
ax4.set_ylim(lims)
ax4.set_xlabel("Истинное значение")
ax4.set_ylabel("Предсказание модели")
ax4.set_title("Предсказание vs. истина")
ax4.set_aspect("equal")
ax4.legend()
ds.save(fig4, "prediction_scatter")

# ---------------------------------------------------------------------------
# 5. Столбчатая диаграмма — метрики по солверам
# ---------------------------------------------------------------------------
solvers = ["Cylinder\nTLM", "Cone", "Webster", "MLP-FNO"]
rmse_values = [0.042, 0.038, 0.051, 0.029]
colors = [ds.COLORS["blue"], ds.COLORS["orange"], ds.COLORS["green"], ds.COLORS["red"]]

fig5, ax5 = plt.subplots()
bars = ax5.bar(solvers, rmse_values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
ax5.set_ylabel("RMSE")
ax5.set_title("Сравнение точности методов")
ax5.set_ylim(0, 0.065)
# подписи на столбцах
for bar, val in zip(bars, rmse_values):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.0015,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
ds.save(fig5, "solver_accuracy_comparison")

# ---------------------------------------------------------------------------
plt.show()
print("\nDone! All figures saved to ./figures/")
