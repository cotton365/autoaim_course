#!/usr/bin/env python3
"""Visibility windows for a spinning 4-plate target.

A plate is "visible" when its bearing is inside the gimbal's usable sector
(+-60 deg here).  Each plate is therefore hittable only during part of the
rotation cycle.  The chart is the scheduling picture behind plate choice.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

omega = 2.0  # rad/s
phi = np.deg2rad([0.0, 90.0, 180.0, 270.0])
t = np.linspace(0.0, 2.0, 4000)
sector = 60.0
colors = ["#8c2d2d", "#2f6f8f", "#3f7a4f", "#9a6a2f"]


def windows(mask, t):
    """Return [(t0, t1), ...] for contiguous True runs."""
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


fig, ax = plt.subplots(figsize=(9.2, 3.8))
for i, (p, c) in enumerate(zip(phi, colors)):
    bearing = np.rad2deg(np.arctan2(np.sin(omega * t + p), np.cos(omega * t + p)))
    visible = np.abs(bearing) <= sector
    y = 3 - i
    for t0, t1 in windows(visible, t):
        ax.barh(y, t1 - t0, left=t0, height=0.5, color=c, alpha=0.75,
                edgecolor="white", linewidth=0.6)
        if t1 - t0 > 0.12:
            ax.text((t0 + t1) / 2, y, f"{t0:.2f}–{t1:.2f}s",
                    ha="center", va="center", color="white", fontsize=8)

ax.axvline(0.15, color="#222222", lw=1.0, ls="--")
ax.text(0.16, 3.55, "现在", fontsize=9)
ax.set_yticks([3, 2, 1, 0])
ax.set_yticklabels(["板 0 (0°)", "板 1 (90°)", "板 2 (180°)", "板 3 (270°)"])
ax.set_xlim(0, 2)
ax.set_ylim(-0.55, 3.75)
ax.set_xlabel("时间 [s]")
ax.set_title("每块板只在窗口内可打（目标 2 rad/s，扇区 ±60°）")
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("assets/figures/plate_windows.png", dpi=160)
print("saved assets/figures/plate_windows.png")
