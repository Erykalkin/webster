# -*- coding: utf-8 -*-
"""
Demo: GOST style charts for diploma thesis.
Run:  python diploma_style_gost_demo.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diploma_style_gost as gost

gost.apply_style()

# ---------------------------------------------------------------------------
# 1. Transfer function comparison (3 solvers)
# ---------------------------------------------------------------------------
f = np.linspace(50, 5000, 512)

def synth_tf(f, peaks, widths, amps):
    tf = np.ones_like(f) * 0.3
    for p, w, a in zip(peaks, widths, amps):
        tf += a * np.exp(-0.5 * ((f - p) / w) ** 2)
    return tf

H_cyl  = synth_tf(f, [420, 1250, 2200, 3500], [80, 120, 150, 200], [1.0, 0.7, 0.5, 0.35])
H_cone = synth_tf(f, [430, 1280, 2250, 3550], [75, 110, 145, 195], [0.95, 0.65, 0.48, 0.30])
H_web  = synth_tf(f, [415, 1240, 2180, 3480], [85, 125, 155, 205], [1.05, 0.72, 0.52, 0.38])

fig1, ax1 = plt.subplots()
ax1.plot(f, 20*np.log10(H_cyl),  **gost.line(0), label="Cylinder TLM")
ax1.plot(f, 20*np.log10(H_cone), **gost.line(1), label="Cone model")
ax1.plot(f, 20*np.log10(H_web),  **gost.line(2), label="Webster equation")
ax1.set_xlabel("f, Hz")
ax1.set_ylabel("Amplitude, dB")
ax1.set_title("Transfer function comparison")
ax1.legend()
gost.caption(fig1, 1, "Transfer function of variable cross-section tube")
gost.save(fig1, "gost_transfer_function")

# ---------------------------------------------------------------------------
# 2. Geometry approximation
# ---------------------------------------------------------------------------
x = np.linspace(0, 0.17, 200)
area = 3e-4 + 1e-4 * np.sin(2*np.pi*x/0.17) - 1.5e-4 * x/0.17

n_sec = 10
x_sec = np.linspace(0, 0.17, n_sec+1)
area_sec = np.interp(x_sec, x, area)
x_step, area_step = [], []
for i in range(n_sec):
    mid_area = 0.5*(area_sec[i] + area_sec[i+1])
    x_step.extend([x_sec[i], x_sec[i+1]])
    area_step.extend([mid_area, mid_area])

fig2, ax2 = plt.subplots()
ax2.plot(x*100, area*1e4, **gost.line(0), label="Original profile")
ax2.plot(np.array(x_step)*100, np.array(area_step)*1e4,
         **gost.line(1, markevery=0.2), label=f"Cylinders ({n_sec})")
ax2.set_xlabel("x, cm")
ax2.set_ylabel(r"S, cm$^2$")
ax2.set_title("Tube geometry approximation")
ax2.legend()
gost.caption(fig2, 2, "Cylinder approximation of the tube profile")
gost.save(fig2, "gost_geometry")

# ---------------------------------------------------------------------------
# 3. Training convergence
# ---------------------------------------------------------------------------
np.random.seed(42)
epochs = np.arange(1, 81)
train_loss = 1.2*np.exp(-0.05*epochs) + 0.02 + 0.015*np.random.randn(80)
val_loss   = 1.3*np.exp(-0.045*epochs) + 0.035 + 0.02*np.random.randn(80)
train_loss = np.maximum(train_loss, 0.01)
val_loss   = np.maximum(val_loss, 0.02)

fig3, ax3 = plt.subplots()
ax3.plot(epochs, train_loss, **gost.line(0), label="Train loss")
ax3.plot(epochs, val_loss,   **gost.line(1), label="Validation loss")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("MSE")
ax3.set_title("MLP-FNO model training convergence")
ax3.legend()

best_idx = int(np.argmin(val_loss))
gost.annotate_point(ax3, epochs[best_idx], val_loss[best_idx],
                    f"min = {val_loss[best_idx]:.4f}", offset=(12, 18))
gost.caption(fig3, 3, "Training convergence of the neural network model")
gost.save(fig3, "gost_training")

# ---------------------------------------------------------------------------
# 4. Prediction scatter
# ---------------------------------------------------------------------------
np.random.seed(7)
y_true = np.random.rand(120)
y_pred = y_true + 0.06*np.random.randn(120)

fig4, ax4 = plt.subplots(figsize=(5.5, 5.5))
ax4.scatter(y_true, y_pred, s=20, facecolors="white", edgecolors="black", linewidths=0.8)
lims = [min(y_true.min(), y_pred.min())-0.05, max(y_true.max(), y_pred.max())+0.05]
ax4.plot(lims, lims, "k--", linewidth=0.9, label="y = x")
ax4.set_xlim(lims); ax4.set_ylim(lims)
ax4.set_xlabel("True value")
ax4.set_ylabel("Model prediction")
ax4.set_title("Prediction vs. ground truth")
ax4.set_aspect("equal")
ax4.legend()
gost.caption(fig4, 4, "Scatter plot of predicted vs. true values")
gost.save(fig4, "gost_scatter")

# ---------------------------------------------------------------------------
# 5. Bar chart with hatching
# ---------------------------------------------------------------------------
solvers = ["Cylinder\nTLM", "Cone", "Webster", "MLP-FNO"]
rmse = [0.042, 0.038, 0.051, 0.029]

fig5, ax5 = plt.subplots()
x_pos = np.arange(len(solvers))
for i, (s, v) in enumerate(zip(solvers, rmse)):
    ax5.bar(x_pos[i], v, width=0.55, **gost.bar_style(i))
    ax5.text(x_pos[i], v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(solvers)
ax5.set_ylabel("RMSE")
ax5.set_title("Solver accuracy comparison")
ax5.set_ylim(0, 0.065)
gost.caption(fig5, 5, "RMSE comparison across methods")
gost.save(fig5, "gost_bar_chart")

# ---------------------------------------------------------------------------
plt.close("all")
print("Done! GOST-style figures saved to ./figures/")
