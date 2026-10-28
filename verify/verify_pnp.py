#!/usr/bin/env python3
"""Numerical checks for pinhole projection, PnP, yaw search, coordinate chains and armor pitch.

Run:
    MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_pnp.py

Only writes PNG files under assets/figures/ (prefix pnp_).  All random streams are
seeded and every number quoted in verify/pnp_report.md is printed by this script.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-zh")

from pathlib import Path
import platform

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

np.set_printoptions(precision=6, suppress=True)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
    }
)

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------
K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
IMAGE_W, IMAGE_H = 1280, 720
DIST0 = np.zeros(5, dtype=np.float64)
DIST_TEST = np.array([-0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)

ARMOR_W, ARMOR_H = 0.135, 0.056
# Armor frame A: x right, y up, z outward (towards the camera when facing it).
# Corner order: top-left, bottom-left, bottom-right, top-right.
OBJ = np.array(
    [
        [-ARMOR_W / 2, +ARMOR_H / 2, 0.0],
        [-ARMOR_W / 2, -ARMOR_H / 2, 0.0],
        [+ARMOR_W / 2, -ARMOR_H / 2, 0.0],
        [+ARMOR_W / 2, +ARMOR_H / 2, 0.0],
    ],
    dtype=np.float64,
)
CORNER_NAMES = ["TL", "BL", "BR", "TR"]
CORNER_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

# Base orientation for a front-facing plate: x_A -> x_C, y_A -> -y_C,
# z_A -> -z_C.  Camera uses OpenCV convention (x right, y down, z forward).
R_BASE = np.diag([1.0, -1.0, -1.0])


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def armor_pose(yaw_deg, pitch_deg, t):
    """Return R_armor2camera and t_armor2camera.

    R = R_base * Ry(yaw) * Rx(pitch).  Positive yaw turns the face normal
    towards camera +x; positive pitch tips the top edge towards the camera.
    """
    R = R_BASE @ rot_y(np.deg2rad(yaw_deg)) @ rot_x(np.deg2rad(pitch_deg))
    return R, np.asarray(t, dtype=np.float64)


def project_pose(R, t, dist_coeffs=None):
    rvec = cv2.Rodrigues(R)[0]
    uv, _ = cv2.projectPoints(
        OBJ, rvec, np.asarray(t, dtype=np.float64), K,
        DIST0 if dist_coeffs is None else dist_coeffs,
    )
    return uv.reshape(-1, 2).astype(np.float64)


def project_yaw_pitch(yaw_deg, pitch_deg, t):
    R, _ = armor_pose(yaw_deg, pitch_deg, t)
    return project_pose(R, t)


def add_pixel_noise(uv, rng, sigma=0.5):
    return uv + rng.normal(0.0, sigma, uv.shape)


def solve_pnp(uv, dist_coeffs=None):
    dist = DIST0 if dist_coeffs is None else dist_coeffs
    ok, rvec, tvec = cv2.solvePnP(
        OBJ, uv.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    R = cv2.Rodrigues(rvec)[0]
    return R, tvec.reshape(3).astype(np.float64)


def solve_pnp_generic(uv):
    n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        OBJ, uv.reshape(-1, 1, 2), K, DIST0, flags=cv2.SOLVEPNP_IPPE
    )
    cands = []
    for i in range(n):
        R = cv2.Rodrigues(rvecs[i])[0]
        t = tvecs[i].reshape(3).astype(np.float64)
        yaw, pitch = yaw_pitch_of_rotation(R)
        normal = R @ np.array([0.0, 0.0, 1.0])
        cands.append(
            {
                "R": R,
                "t": t,
                "reproj": float(np.asarray(reproj[i]).ravel()[0]),
                "yaw": yaw,
                "pitch": pitch,
                "faces_camera": bool(normal @ (-t) > 0.0),
            }
        )
    return cands


def yaw_pitch_of_rotation(R):
    """Read plate yaw/pitch (deg) from the outward normal R[:,2]."""
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = np.rad2deg(np.arctan2(n[0], -n[2]))
    pitch = np.rad2deg(np.arcsin(np.clip(n[1], -1.0, 1.0)))
    return float(yaw), float(pitch)


def rotation_error_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def corner_errors(uv_pred, uv_obs):
    return np.linalg.norm(uv_pred - uv_obs, axis=1)


def corner_rms(uv_pred, uv_obs):
    d = corner_errors(uv_pred, uv_obs)
    return float(np.sqrt(np.mean(d * d)))


def stats(a, prefix=""):
    a = np.asarray(a, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "p95abs": float(np.percentile(np.abs(a), 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "prefix": prefix,
    }


def print_stat_line(name, a, unit=""):
    s = stats(a)
    print(
        f"  {name:<34s} mean={s['mean']:9.4f}{unit}  std={s['std']:8.4f}{unit} "
        f"median={s['median']:9.4f}{unit}  p95|.|={s['p95abs']:9.4f}{unit}"
    )
    return s


def arrow2d(ax, start, end, color, label=None, ls="-", lw=1.4, alpha=1.0):
    p = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, color=color,
        linewidth=lw, linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0,
        label=label, zorder=6,
    )
    ax.add_patch(p)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print("=" * 78)
print("verify_pnp.py - pinhole / PnP / yaw / frames / armor pitch verification")
print("=" * 78)
print(f"python      : {platform.python_version()}")
print(f"numpy       : {np.__version__}")
print(f"scipy       : {__import__('scipy').__version__}")
print(f"opencv      : {cv2.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"figures dir : {FIG_DIR}")
print("camera K    :")
print(K)
print("armor W x H : %.3f m x %.3f m" % (ARMOR_W, ARMOR_H))
print("armor frame : x right, y up, z outward; corners TL, BL, BR, TR")
print("R_base      : diag(1,-1,-1), mapping front-facing armor axes to camera axes")
print("noise       : independent N(0, 0.5^2) on every u and v pixel coordinate")
print()

# ---------------------------------------------------------------------------
# 1. Closed loop projection -> noisy PnP
# ---------------------------------------------------------------------------
print("-" * 78)
print("[1] closed-loop projection and PnP (200 runs, 3 m, yaw=20 deg, pitch=15 deg)")
print("-" * 78)
SEED_CLOSED = 20261026
rng_closed = np.random.default_rng(SEED_CLOSED)
print(f"seed        : {SEED_CLOSED}")

T_TRUE = np.array([0.0, 0.0, 3.0])
YAW_TRUE, PITCH_TRUE = 20.0, 15.0
R_TRUE, _ = armor_pose(YAW_TRUE, PITCH_TRUE, T_TRUE)
UV_TRUE = project_pose(R_TRUE, T_TRUE)
print("true R_armor2camera:")
print(R_TRUE)
print("true t (m):", T_TRUE)
print("true normal (camera):", R_TRUE @ np.array([0.0, 0.0, 1.0]))
print("true yaw/pitch from normal (deg):", yaw_pitch_of_rotation(R_TRUE))
print("clean pixels (u,v):")
for name, p in zip(CORNER_NAMES, UV_TRUE):
    print(f"  {name}: ({p[0]:9.4f}, {p[1]:9.4f})")
print(
    "clean projected extent: width=%.4f px, height=%.4f px, center=(%.3f, %.3f)"
    % (
        UV_TRUE[:, 0].max() - UV_TRUE[:, 0].min(),
        UV_TRUE[:, 1].max() - UV_TRUE[:, 1].min(),
        UV_TRUE[:, 0].mean(),
        UV_TRUE[:, 1].mean(),
    )
)

raw_rot, raw_trans, raw_rep = [], [], []
br_rot, br_trans, br_rep = [], [], []
br_yaw_err, br_pitch_err = [], []
n_solutions = []
all_both_face = 0
raw_pitch_negative = 0
raw_is_lower_reproj = 0
generic_rot_gap, generic_reproj_gap = [], []
both_face_runs = 0

for _ in range(200):
    uv_obs = add_pixel_noise(UV_TRUE, rng_closed, sigma=0.5)
    R_raw, t_raw = solve_pnp(uv_obs)
    uv_raw = project_pose(R_raw, t_raw)
    raw_rot.append(rotation_error_deg(R_TRUE, R_raw))
    raw_trans.append(float(np.linalg.norm(t_raw - T_TRUE) * 1000.0))
    raw_rep.append(corner_rms(uv_raw, uv_obs))
    yaw_raw, pitch_raw = yaw_pitch_of_rotation(R_raw)
    if pitch_raw < 0.0:
        raw_pitch_negative += 1

    cands = solve_pnp_generic(uv_obs)
    n_solutions.append(len(cands))
    if len(cands) >= 2:
        generic_rot_gap.append(rotation_error_deg(cands[0]["R"], cands[1]["R"]))
        generic_reproj_gap.append(abs(cands[0]["reproj"] - cands[1]["reproj"]))
        if all(c["faces_camera"] for c in cands[:2]):
            both_face_runs += 1
    # The valid mechanical prior is armor pitch = +15 deg: choose the branch
    # whose estimated pitch is closest to +15 deg.
    sel = min(cands, key=lambda c: abs(c["pitch"] - PITCH_TRUE))
    R_sel, t_sel = sel["R"], sel["t"]
    uv_sel = project_pose(R_sel, t_sel)
    br_rot.append(rotation_error_deg(R_TRUE, R_sel))
    br_trans.append(float(np.linalg.norm(t_sel - T_TRUE) * 1000.0))
    br_rep.append(corner_rms(uv_sel, uv_obs))
    yaw_sel, pitch_sel = sel["yaw"], sel["pitch"]
    br_yaw_err.append(((yaw_sel - YAW_TRUE + 180.0) % 360.0) - 180.0)
    br_pitch_err.append(pitch_sel - PITCH_TRUE)
    lower = min(cands, key=lambda c: c["reproj"])
    if np.linalg.norm(lower["t"] - t_raw) < 1e-9 and np.linalg.norm(
        lower["R"] - R_raw
    ) < 1e-9:
        raw_is_lower_reproj += 1

print(f"\ngeneric solution count: {dict(zip(*np.unique(n_solutions, return_counts=True)))}")
print(
    "runs where BOTH generic solutions face the camera: %d/200"
    % both_face_runs
)
print(
    "raw solvePnP chose negative-pitch (mirror) branch: %d/200" % raw_pitch_negative
)
print("raw solvePnP equals the lower-reprojection generic solution: %d/200" % raw_is_lower_reproj)
print_stat_line("generic branch rotation gap", generic_rot_gap, " deg")
print_stat_line("generic branch reprojection gap", generic_reproj_gap, " px")
print()

print("raw solvePnP (no branch prior):")
print_stat_line("rotation error", raw_rot, " deg")
print_stat_line("translation error", raw_trans, " mm")
print_stat_line("reprojection RMS", raw_rep, " px")
print("\nIPPE branch chosen with known +15 deg pitch prior:")
print_stat_line("rotation error", br_rot, " deg")
print_stat_line("translation error", br_trans, " mm")
print_stat_line("reprojection RMS", br_rep, " px")
print_stat_line("yaw error (estimate - truth)", br_yaw_err, " deg")
print_stat_line("pitch error (estimate - truth)", br_pitch_err, " deg")
print()

# --- Fig 1: 3D camera-armor geometry ---------------------------------------
fig = plt.figure(figsize=(9.2, 7.0))
ax = fig.add_subplot(111, projection="3d")
z_img = 0.5
cam_axes_len = (0.32, 0.24, 0.30)
# plot coordinates: (depth z, lateral x, up -y)
def p3(P):
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        return np.array([P[2], P[0], -P[1]])
    return np.column_stack([P[:, 2], P[:, 0], -P[:, 1]])


# image plane
img_corners = []
for u, v in [(0, 0), (IMAGE_W - 1, 0), (IMAGE_W - 1, IMAGE_H - 1), (0, IMAGE_H - 1)]:
    x = (u - K[0, 2]) / K[0, 0] * z_img
    y = (v - K[1, 2]) / K[1, 1] * z_img
    img_corners.append([x, y, z_img])
img_corners = np.array(img_corners)
ip = p3(img_corners)
ax.add_collection3d(
    Poly3DCollection([ip], facecolors="lightsteelblue", edgecolors="steelblue", alpha=0.35)
)
for i in range(4):
    ax.plot(
        [0, ip[i, 0]], [0, ip[i, 1]], [0, ip[i, 2]],
        color="steelblue", lw=0.8, alpha=0.5,
    )

# armor face
armor_pts = (R_TRUE @ OBJ.T).T + T_TRUE
ap = p3(armor_pts)
ax.add_collection3d(
    Poly3DCollection([ap], facecolors="coral", edgecolors="darkred", alpha=0.55, linewidths=1.2)
)
for i in range(4):
    j = (i + 1) % 4
    ax.plot([ap[i, 0], ap[j, 0]], [ap[i, 1], ap[j, 1]], [ap[i, 2], ap[j, 2]],
            color="darkred", lw=1.4)

# projected image points and rays
img_pts = []
for u, v in UV_TRUE:
    x = (u - K[0, 2]) / K[0, 0] * z_img
    y = (v - K[1, 2]) / K[1, 1] * z_img
    img_pts.append([x, y, z_img])
img_pts = np.array(img_pts)
for i in range(4):
    ax.plot(
        [0, ap[i, 0]], [0, ap[i, 1]], [0, ap[i, 2]],
        color=CORNER_COLORS[i], lw=0.9, alpha=0.55,
    )
    q = p3(img_pts[i])
    ax.scatter(q[0], q[1], q[2], color=CORNER_COLORS[i], s=16, depthshade=False)

# camera center and axes
ax.scatter([0], [0], [0], color="black", s=40, depthshade=False)
# Zc forward
ax.quiver(0, 0, 0, cam_axes_len[2], 0, 0, color="blue", arrow_length_ratio=0.14)
ax.text(cam_axes_len[2] + 0.03, 0, 0, "Zc", color="blue")
# Xc right
ax.quiver(0, 0, 0, 0, cam_axes_len[0], 0, color="red", arrow_length_ratio=0.14)
ax.text(0.01, cam_axes_len[0] + 0.03, 0, "Xc", color="red")
# -Yc up
ax.quiver(0, 0, 0, 0, 0, cam_axes_len[1], color="green", arrow_length_ratio=0.14)
ax.text(0.01, 0, cam_axes_len[1] + 0.03, "-Yc (up)", color="green")

# armor normal and local axes
center = T_TRUE
normal = R_TRUE @ np.array([0.0, 0.0, 1.0])
n_plot = p3(center + 0.16 * normal)
c_plot = p3(center)
ax.quiver(
    c_plot[0], c_plot[1], c_plot[2],
    n_plot[0] - c_plot[0], n_plot[1] - c_plot[1], n_plot[2] - c_plot[2],
    color="purple", arrow_length_ratio=0.18, lw=1.3,
)
# depth marker
ax.plot([0, 3.0], [-0.28, -0.28], [0, 0], "k--", lw=0.9)
ax.text(1.45, -0.30, 0.02, "distance Zc = 3.0 m", color="black", fontsize=9)

legend_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
           markeredgecolor="black", markersize=6, label="camera origin O_c"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="coral",
           markeredgecolor="darkred", markersize=8, label="armor face"),
    Line2D([0], [0], color="steelblue", lw=2.0, label="image plane (schematic)"),
    Line2D([0], [0], color="purple", lw=1.3, label="armor normal n_A"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)
ax.set_xlabel("depth Zc [m]")
ax.set_ylabel("Xc [m]")
ax.set_zlabel("-Yc (up) [m]")
ax.set_xlim(0.0, 3.35)
ax.set_ylim(-0.45, 0.45)
ax.set_zlim(-0.35, 0.35)
ax.set_box_aspect((3.2, 1.15, 0.85))
ax.view_init(elev=19, azim=-58)
ax.set_title(
    "Camera-armor geometry at 3 m (yaw 20 deg, pitch 15 deg)\n"
    "image plane drawn at Zc=0.5 m for readability"
)
fig.savefig(FIG_DIR / "pnp_geometry.png")
plt.close(fig)

# --- Fig 2: image projection ------------------------------------------------
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.6, 5.4))
ax_full.set_xlim(0, IMAGE_W - 1)
ax_full.set_ylim(IMAGE_H - 1, 0)
ax_full.set_aspect("equal")
ax_full.add_patch(
    Rectangle((0, 0), IMAGE_W - 1, IMAGE_H - 1, fill=False, edgecolor="black", lw=1.2)
)
ax_full.scatter(
    [K[0, 2]], [K[1, 2]], marker="+", s=90, color="gray", linewidths=1.2,
    label="principal point (640,360)",
)
poly = np.vstack([UV_TRUE, UV_TRUE[0]])
ax_full.plot(poly[:, 0], poly[:, 1], color="darkred", lw=1.6)
for name, p, c in zip(CORNER_NAMES, UV_TRUE, CORNER_COLORS):
    ax_full.scatter(p[0], p[1], s=22, color=c, label=name, zorder=5)
cx, cy = UV_TRUE.mean(axis=0)
ax_full.add_patch(
    Rectangle((cx - 30, cy - 20), 60, 40, fill=False, edgecolor="darkorange", lw=1.0, ls="--")
)
ax_full.text(cx + 34, cy - 24, "zoom ROI", color="darkorange", fontsize=8)
ax_full.set_xlabel("u [px]")
ax_full.set_ylabel("v [px]")
ax_full.set_title("Full 1280x720 projection")
ax_full.legend(loc="upper right", ncol=2, fontsize=8)
ax_full.set_facecolor("white")

ax_zoom.set_xlim(cx - 38, cx + 38)
ax_zoom.set_ylim(cy + 25, cy - 25)
ax_zoom.set_aspect("equal")
ax_zoom.plot(poly[:, 0], poly[:, 1], color="darkred", lw=1.8)
for name, p, c in zip(CORNER_NAMES, UV_TRUE, CORNER_COLORS):
    ax_zoom.scatter(p[0], p[1], s=42, color=c, zorder=5)
    ax_zoom.annotate(
        f"{name}\n({p[0]:.2f},{p[1]:.2f})",
        xy=p,
        xytext=(8 if name in ("BR", "TR") else -8, 8 if name in ("TL", "TR") else -18),
        textcoords="offset points",
        fontsize=8,
        ha="left" if name in ("BR", "TR") else "right",
    )
ax_zoom.plot([cx - 4, cx + 4], [cy, cy], color="gray", lw=1)
ax_zoom.plot([cx, cx], [cy - 4, cy + 4], color="gray", lw=1)
ax_zoom.text(cx + 1.2, cy + 1.2, "center", color="gray", fontsize=8)
ax_zoom.set_xlabel("u [px]")
ax_zoom.set_ylabel("v [px]")
ax_zoom.set_title("Zoom: projected small armor (corner order TL-BL-BR-TR)")
ax_zoom.set_facecolor("white")
fig.suptitle("Pinhole projection of a 0.135 x 0.056 m armor at 3 m", y=1.02)
fig.savefig(FIG_DIR / "pnp_projection.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Depth error vs distance
# ---------------------------------------------------------------------------
print("-" * 78)
print("[2] depth error vs distance (0.5 px noise, 200 runs per distance)")
print("-" * 78)
SEED_DEPTH = 20261027
rng_depth = np.random.default_rng(SEED_DEPTH)
print(f"seed        : {SEED_DEPTH}")
distances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
depth_rows = []
for d in distances:
    t_d = np.array([0.0, 0.0, d])
    uv_clean = project_pose(R_TRUE, t_d)
    rel = []
    abs_mm = []
    for _ in range(200):
        uv_obs = add_pixel_noise(uv_clean, rng_depth, sigma=0.5)
        _, t_est = solve_pnp(uv_obs)
        rel.append((t_est[2] - d) / d * 100.0)
        abs_mm.append(abs(t_est[2] - d) * 1000.0)
    rel = np.asarray(rel)
    abs_mm = np.asarray(abs_mm)
    s = stats(rel)
    row = {
        "d": d,
        "mean": s["mean"],
        "std": s["std"],
        "mean_abs": float(np.mean(np.abs(rel))),
        "p95_abs": float(np.percentile(np.abs(rel), 95)),
        "p5": float(np.percentile(rel, 5)),
        "p95": float(np.percentile(rel, 95)),
        "std_mm": float(np.std(abs_mm, ddof=1)),
        "mean_abs_mm": float(np.mean(abs_mm)),
    }
    depth_rows.append(row)
    print(
        f"  d={d:.0f} m: mean={row['mean']:+7.3f}%  std={row['std']:6.3f}%  "
        f"mean|.|={row['mean_abs']:6.3f}%  p95|.|={row['p95_abs']:6.3f}%  "
        f"mean|z_err|={row['mean_abs_mm']:6.2f} mm"
    )
row6 = depth_rows[-1]
print(
    f"\n6 m headline: signed mean={row6['mean']:+.3f}%, std={row6['std']:.3f}%, "
    f"mean|error|={row6['mean_abs']:.3f}%, p95|error|={row6['p95_abs']:.3f}%, "
    f"mean |z error|={row6['mean_abs_mm']:.2f} mm"
)

fig, ax = plt.subplots(figsize=(9.0, 5.2))
ds = np.array([r["d"] for r in depth_rows])
means = np.array([r["mean"] for r in depth_rows])
stds = np.array([r["std"] for r in depth_rows])
mean_abs = np.array([r["mean_abs"] for r in depth_rows])
p5 = np.array([r["p5"] for r in depth_rows])
p95 = np.array([r["p95"] for r in depth_rows])
ax.fill_between(ds, p5, p95, color="tab:blue", alpha=0.13, label="5th-95th percentile (signed)")
ax.errorbar(
    ds, means, yerr=stds, fmt="o-", color="tab:blue", capsize=4,
    lw=1.6, label="signed mean +/- 1 std",
)
ax.plot(ds, mean_abs, "s--", color="tab:red", lw=1.4, label="mean absolute error")
ax.axhline(0.0, color="black", lw=0.8)
ax.annotate(
    f"6 m: mean={row6['mean']:+.2f}%\nstd={row6['std']:.2f}%\nmean|e|={row6['mean_abs']:.2f}%",
    xy=(6.0, row6["mean"]),
    xytext=(4.10, -7.9),
    fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
)
ax.set_xlabel("Distance to armor center [m]")
ax.set_ylabel("Depth relative error (t_z - d) / d [%]")
ax.set_title("Monocular PnP depth uncertainty vs distance (0.5 px noise, 200 trials each)")
ax.set_xticks(ds)
ax.set_xlim(0.7, 6.3)
ax.legend(loc="upper left")
fig.savefig(FIG_DIR / "pnp_depth_noise.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Yaw scan with fixed translation and pitch
# ---------------------------------------------------------------------------
print("-" * 78)
print("[3] yaw reprojection search (t fixed, pitch=15 deg, true yaw=30 deg)")
print("-" * 78)
SEED_YAW = 20261028
rng_yaw = np.random.default_rng(SEED_YAW)
print(f"seed        : {SEED_YAW}")
YAW_SCAN_TRUE = 30.0
T_SCAN = np.array([0.0, 0.0, 3.0])
UV_SCAN_TRUE = project_yaw_pitch(YAW_SCAN_TRUE, 15.0, T_SCAN)
yaw_grid = np.arange(-70.0, 70.0 + 1e-9, 0.5)
cost_clean = np.array(
    [corner_rms(project_yaw_pitch(y, 15.0, T_SCAN), UV_SCAN_TRUE) for y in yaw_grid]
)
i_best = int(np.argmin(cost_clean))
print("grid points :", len(yaw_grid))
print("noise-free minimum yaw: %.3f deg, RMS: %.6f px" % (yaw_grid[i_best], cost_clean[i_best]))
print("cost at true 30.0 deg: %.6f px" % cost_clean[np.argmin(np.abs(yaw_grid - 30.0))])

estimates, estimates_q, best_rms = [], [], []
for _ in range(100):
    uv_obs = add_pixel_noise(UV_SCAN_TRUE, rng_yaw, sigma=0.5)
    cost = np.array(
        [corner_rms(project_yaw_pitch(y, 15.0, T_SCAN), uv_obs) for y in yaw_grid]
    )
    j = int(np.argmin(cost))
    estimates.append(yaw_grid[j])
    best_rms.append(cost[j])
    if 0 < j < len(yaw_grid) - 1:
        denom = cost[j - 1] - 2.0 * cost[j] + cost[j + 1]
        offset = 0.25 * (cost[j - 1] - cost[j + 1]) / denom if abs(denom) > 1e-14 else 0.0
        estimates_q.append(yaw_grid[j] + offset)
    else:
        estimates_q.append(yaw_grid[j])
estimates = np.asarray(estimates)
estimates_q = np.asarray(estimates_q)
yaw_bias = estimates - YAW_SCAN_TRUE
yaw_bias_q = estimates_q - YAW_SCAN_TRUE
print("\n100 noisy scans (0.5 px):")
print_stat_line("grid-argmin yaw error", yaw_bias, " deg")
print_stat_line("quadratic-refined yaw error", yaw_bias_q, " deg")
print_stat_line("minimum corner RMS", best_rms, " px")
print(
    "grid estimates min/max: %.1f / %.1f deg; unique values: %s"
    % (estimates.min(), estimates.max(), np.unique(estimates).tolist())
)
print(
    "raw bias=%.4f deg, raw std=%.4f deg, refined bias=%.4f deg, refined std=%.4f deg"
    % (np.mean(yaw_bias), np.std(yaw_bias, ddof=1), np.mean(yaw_bias_q), np.std(yaw_bias_q, ddof=1))
)

fig, axs = plt.subplots(1, 3, figsize=(15.3, 4.5))
ax = axs[0]
ax.plot(yaw_grid, cost_clean, color="tab:blue", lw=1.6)
ax.axvline(YAW_SCAN_TRUE, color="black", ls="--", lw=0.9, label="true yaw 30 deg")
ax.scatter(
    [yaw_grid[i_best]], [cost_clean[i_best]], color="tab:red", zorder=5,
    label=f"minimum at {yaw_grid[i_best]:.1f} deg",
)
ax.set_xlabel("Candidate yaw [deg]")
ax.set_ylabel("4-corner reprojection RMS [px]")
ax.set_title("Full search range")
ax.legend(loc="upper center")
ax.set_xlim(-70, 70)
ax.set_ylim(bottom=-0.05)

ax = axs[1]
m = (yaw_grid >= 25.0) & (yaw_grid <= 35.0)
ax.plot(yaw_grid[m], cost_clean[m], "o-", color="tab:blue", ms=3.2, lw=1.2)
ax.axvline(YAW_SCAN_TRUE, color="black", ls="--", lw=0.9)
ax.scatter([yaw_grid[i_best]], [cost_clean[i_best]], color="tab:red", s=40, zorder=5)
ax.set_xlabel("Candidate yaw [deg]")
ax.set_ylabel("4-corner reprojection RMS [px]")
ax.set_title("Zoom near the minimum")
ax.text(
    0.03, 0.88,
    f"grid step = 0.5 deg\nmin = {yaw_grid[i_best]:.1f} deg\nRMS = {cost_clean[i_best]:.2e} px",
    transform=ax.transAxes, fontsize=8, va="top",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
)

ax = axs[2]
if len(np.unique(estimates)) > 1:
    bins = np.arange(estimates.min() - 0.75, estimates.max() + 1.26, 0.5)
    ax.hist(estimates, bins=bins, color="tab:blue", alpha=0.7, edgecolor="white")
else:
    ax.axvline(estimates[0], color="tab:blue", lw=3)
ax.axvline(YAW_SCAN_TRUE, color="black", ls="--", lw=1.0, label="true yaw")
ax.axvline(estimates.mean(), color="tab:red", ls=":", lw=1.3, label="estimate mean")
ax.set_xlabel("Estimated yaw [deg]")
ax.set_ylabel("Count")
ax.set_title("100 noisy yaw estimates")
ax.text(
    0.03, 0.80,
    f"bias = {np.mean(yaw_bias):+.2f} deg\nstd = {np.std(yaw_bias, ddof=1):.2f} deg",
    transform=ax.transAxes, fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
)
ax.legend(loc="upper right")
fig.suptitle("Yaw search at fixed t and pitch=15 deg (true yaw 30 deg)", y=1.02)
fig.savefig(FIG_DIR / "pnp_yaw_scan.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Coordinate chain
# ---------------------------------------------------------------------------
print("-" * 78)
print("[4] camera -> gimbal -> world coordinate chain")
print("-" * 78)
R_C2G = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_C2G = np.array([0.20, 0.08, 0.05])
R_G2W = rot_z(np.deg2rad(25.0))
T_G2W = np.zeros(3)
print("R_camera2gimbal:")
print(R_C2G)
print("t_camera2gimbal (m):", T_C2G)
print("R_gimbal2world (yaw 25 deg):")
print(R_G2W)
print("t_gimbal2world (m):", T_G2W)
print("det(R_c2g)=%.15f, det(R_g2w)=%.15f" % (np.linalg.det(R_C2G), np.linalg.det(R_G2W)))

T_G_C = np.eye(4)
T_G_C[:3, :3] = R_C2G
T_G_C[:3, 3] = T_C2G
T_W_G = np.eye(4)
T_W_G[:3, :3] = R_G2W
T_W_G[:3, 3] = T_G2W
T_W_C = T_W_G @ T_G_C

p_cam = np.array([0.0, 0.0, 3.0])
p_gimbal = R_C2G @ p_cam + T_C2G
p_world = R_G2W @ p_gimbal
p_direct = T_W_C[:3, :3] @ p_cam + T_W_C[:3, 3]
print("p_camera (m):", p_cam)
print("p_gimbal (m):", p_gimbal)
print("p_world  (m):", p_world)
print("direct T_world_camera * p_camera (m):", p_direct)
print("target point chain residual (m): %.3e" % np.linalg.norm(p_world - p_direct))

rng_frames = np.random.default_rng(20261029)
errs = []
for _ in range(1000):
    p = rng_frames.uniform(-2.0, 2.0, 3)
    a = R_G2W @ (R_C2G @ p + T_C2G)
    b = T_W_C[:3, :3] @ p + T_W_C[:3, 3]
    errs.append(np.max(np.abs(a - b)))
errs = np.asarray(errs)
print("max |chain - homogeneous T| over 1000 points (m): %.3e" % errs.max())
print("mean max abs error (m): %.3e" % errs.mean())
print("T_world_camera:")
print(T_W_C)
print("world origin in gimbal / camera coordinates:",
      T_W_G[:3, 3], np.linalg.inv(T_G_C)[:3, 3])

# --- Fig 5: top view frames -------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 6.6))
# screen coordinates are gimbal XY (x forward, y left)
origin = np.array([0.0, 0.0])
cam_xy = T_C2G[:2]
tgt_xy = p_gimbal[:2]

# gimbal axes
arrow2d(ax, origin, origin + np.array([1.15, 0.0]), "black", lw=1.6)
arrow2d(ax, origin, origin + np.array([0.0, 1.15]), "black", lw=1.6)
ax.text(1.18, -0.05, "Xg (forward)", fontsize=9)
ax.text(-0.02, 1.18, "Yg (left)", fontsize=9, rotation=90, va="bottom", ha="left")

# world axes expressed in gimbal coordinates
xw = R_G2W.T @ np.array([1.0, 0.0, 0.0])
yw = R_G2W.T @ np.array([0.0, 1.0, 0.0])
arrow2d(ax, origin, origin + 0.95 * xw[:2], "tab:orange", ls="--", lw=1.4)
arrow2d(ax, origin, origin + 0.95 * yw[:2], "tab:orange", ls="--", lw=1.4)
ax.text(*(origin + 0.97 * xw[:2] + np.array([0.02, -0.08])), "Xw", color="tab:orange", fontsize=9)
ax.text(*(origin + 0.97 * yw[:2] + np.array([-0.09, 0.0])), "Yw", color="tab:orange", fontsize=9)

# vector chain
arrow2d(ax, origin, cam_xy, "tab:blue", lw=1.6)
arrow2d(ax, cam_xy, tgt_xy, "tab:green", lw=1.6)
ax.plot([origin[0], tgt_xy[0]], [origin[1], tgt_xy[1]], ":", color="gray", lw=1.0)

# origin markers: gimbal and world coincide
ax.scatter(
    [0], [0], marker="s", s=95, facecolor="black", edgecolor="black",
    zorder=8, label="gimbal origin O_g",
)
ax.scatter(
    [0], [0], marker="*", s=360, facecolor="none", edgecolor="tab:orange",
    linewidths=2.0, zorder=9, label="world origin O_w (same point)",
)
ax.annotate(
    "O_g", xy=origin, xytext=(-0.38, -0.23), fontsize=11,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
)
ax.annotate(
    "O_w (=O_g)", xy=origin, xytext=(-0.50, 0.66), fontsize=9.5, color="tab:orange",
    arrowprops=dict(arrowstyle="-", color="tab:orange", lw=0.8),
)

# camera and target
ax.scatter(
    [cam_xy[0]], [cam_xy[1]], marker="^", s=110, color="tab:blue",
    edgecolor="black", zorder=8, label="camera origin O_c",
)
ax.annotate(
    "O_c (camera)\n" + f"({cam_xy[0]:.2f}, {cam_xy[1]:.2f})",
    xy=cam_xy, xytext=(cam_xy[0] - 0.10, cam_xy[1] - 0.50),
    fontsize=9, color="tab:blue",
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
)
ax.scatter(
    [tgt_xy[0]], [tgt_xy[1]], marker="D", s=90, color="tab:red",
    edgecolor="black", zorder=8, label="target point",
)
ax.annotate(
    "target p\n" + f"({tgt_xy[0]:.2f}, {tgt_xy[1]:.2f})",
    xy=tgt_xy, xytext=(tgt_xy[0] - 0.55, tgt_xy[1] + 0.24),
    fontsize=9, color="tab:red",
    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8),
)
ax.text(
    (origin[0] + cam_xy[0]) / 2, (origin[1] + cam_xy[1]) / 2 - 0.17,
    "t_c2g", color="tab:blue", fontsize=9,
)
ax.text(
    (cam_xy[0] + tgt_xy[0]) / 2, (cam_xy[1] + tgt_xy[1]) / 2 + 0.10,
    "R_c2g p_cam", color="tab:green", fontsize=9,
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.55, 3.65)
ax.set_ylim(-1.25, 1.75)
ax.set_xlabel("Gimbal X (forward) [m]")
ax.set_ylabel("Gimbal Y (left) [m]")
ax.set_title(
    "Top view of the frame chain\n"
    "O_w and O_g coincide; only R_g2w (25 deg yaw) changes the axes"
)
box = (
    "p_cam   = [0, 0, 3.00] m\n"
    f"p_gimbal= [{p_gimbal[0]:.3f}, {p_gimbal[1]:.3f}, {p_gimbal[2]:.3f}] m\n"
    f"p_world = [{p_world[0]:.3f}, {p_world[1]:.3f}, {p_world[2]:.3f}] m\n"
    f"max chain error = {errs.max():.1e} m"
)
ax.text(
    0.98, 0.03, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.92),
)
ax.legend(loc="upper right", framealpha=0.95)
fig.savefig(FIG_DIR / "pnp_frames.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Effect of 15 degree armor pitch
# ---------------------------------------------------------------------------
print("-" * 78)
print("[5] effect of 15 deg armor pitch")
print("-" * 78)
SEED_PITCH = 20261030
rng_pitch = np.random.default_rng(SEED_PITCH)
print(f"seed        : {SEED_PITCH}")
T_PITCH = np.array([0.0, 0.0, 3.0])
YAW_P = 30.0
UV_P15 = project_yaw_pitch(YAW_P, 15.0, T_PITCH)
UV_P0 = project_yaw_pitch(YAW_P, 0.0, T_PITCH)
delta = UV_P15 - UV_P0
dist_delta = np.linalg.norm(delta, axis=1)
print("corner pixel shifts (pitch=15 minus pitch=0), du, dv, norm:")
for name, duv, dn in zip(CORNER_NAMES, delta, dist_delta):
    print(f"  {name}: du={duv[0]:+7.4f} px  dv={duv[1]:+7.4f} px  |d|={dn:.4f} px")
print(
    "mean |shift|=%.4f px, max |shift|=%.4f px (%s), mean du=%.4f, mean dv=%.4f"
    % (
        dist_delta.mean(), dist_delta.max(), CORNER_NAMES[int(np.argmax(dist_delta))],
        delta[:, 0].mean(), delta[:, 1].mean(),
    )
)

cost_fixed_15 = np.array(
    [corner_rms(project_yaw_pitch(y, 15.0, T_PITCH), UV_P15) for y in yaw_grid]
)
cost_fixed_0 = np.array(
    [corner_rms(project_yaw_pitch(y, 0.0, T_PITCH), UV_P15) for y in yaw_grid]
)
i15 = int(np.argmin(cost_fixed_15))
i0 = int(np.argmin(cost_fixed_0))
i_mirror = int(np.argmin(np.abs(yaw_grid + YAW_P)))
print(
    "fixed-t scan pitch15: min yaw=%.1f deg, RMS=%.6f px" % (yaw_grid[i15], cost_fixed_15[i15])
)
print(
    "fixed-t scan pitch0 : min yaw=%.1f deg, RMS=%.6f px; "
    "mirror yaw=%.1f deg, RMS=%.6f px"
    % (yaw_grid[i0], cost_fixed_0[i0], yaw_grid[i_mirror], cost_fixed_0[i_mirror])
)


def pitch0_fit(uv_obs, x0=(30.0, 0.0, 0.0, 3.0)):
    """Fit yaw and t with pitch constrained to zero (local branch near truth)."""
    def residual(p):
        yaw, tx, ty, tz = p
        pred = project_yaw_pitch(yaw, 0.0, np.array([tx, ty, tz]))
        return (pred - uv_obs).ravel()

    sol = least_squares(
        residual,
        np.asarray(x0, dtype=float),
        bounds=([15.0, -0.20, -0.20, 2.50], [45.0, 0.20, 0.20, 3.50]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=300,
    )
    yaw, tx, ty, tz = sol.x
    pred = project_yaw_pitch(yaw, 0.0, np.array([tx, ty, tz]))
    d = corner_errors(pred, uv_obs)
    return {
        "yaw": float(yaw),
        "t": np.array([tx, ty, tz]),
        "rms": float(np.sqrt(np.mean(d * d))),
        "max": float(np.max(d)),
        "corners": d,
        "success": bool(sol.success),
    }


fit_clean = pitch0_fit(UV_P15)
print(
    "pitch0 + free t, noise-free fit: yaw=%.4f deg (bias %+0.4f), "
    "t=[%.5f, %.5f, %.5f] m (dz=%+.3f%%), reproj RMS=%.4f px, max=%.4f px"
    % (
        fit_clean["yaw"], fit_clean["yaw"] - YAW_P,
        fit_clean["t"][0], fit_clean["t"][1], fit_clean["t"][2],
        (fit_clean["t"][2] - 3.0) / 3.0 * 100.0,
        fit_clean["rms"], fit_clean["max"],
    )
)
fit_noisy = []
for _ in range(100):
    uv_obs = add_pixel_noise(UV_P15, rng_pitch, sigma=0.5)
    fit_noisy.append(pitch0_fit(uv_obs, x0=(30.0, 0.0, 0.0, 3.0)))
fit_yaw = np.array([f["yaw"] for f in fit_noisy])
fit_rms = np.array([f["rms"] for f in fit_noisy])
fit_tz = np.array([f["t"][2] for f in fit_noisy])
print("\n100 noisy pitch0 + free-t local fits (0.5 px):")
print_stat_line("yaw error", fit_yaw - YAW_P, " deg")
print_stat_line("t_z relative error", (fit_tz - 3.0) / 3.0 * 100.0, " %")
print_stat_line("reprojection RMS", fit_rms, " px")

fig, (ax_img, ax_cost) = plt.subplots(1, 2, figsize=(14.2, 5.4))
poly15 = np.vstack([UV_P15, UV_P15[0]])
poly0 = np.vstack([UV_P0, UV_P0[0]])
ax_img.plot(poly15[:, 0], poly15[:, 1], "o-", color="tab:red", lw=1.6, ms=5, label="pitch = 15 deg (truth)")
ax_img.plot(poly0[:, 0], poly0[:, 1], "s--", color="tab:blue", lw=1.3, ms=5, label="pitch = 0 deg (model)")
for name, a, b, c in zip(CORNER_NAMES, UV_P15, UV_P0, CORNER_COLORS):
    ax_img.annotate(
        "", xy=a, xytext=b, arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.85)
    )
    ax_img.annotate(
        f"{name}: {np.linalg.norm(a-b):.2f} px",
        xy=a, xytext=(6, 5 if name in ("TL", "TR") else -12), textcoords="offset points",
        fontsize=8, color=c,
    )
cx, cy = UV_P15.mean(axis=0)
ax_img.set_xlim(cx - 25, cx + 25)
ax_img.set_ylim(cy + 18, cy - 18)
ax_img.set_aspect("equal")
ax_img.set_xlabel("u [px]")
ax_img.set_ylabel("v [px]")
ax_img.set_title("Same pose, different pitch model")
ax_img.legend(loc="lower left")
ax_img.text(
    0.03, 0.97,
    f"mean |shift| = {dist_delta.mean():.2f} px\nmax |shift| = {dist_delta.max():.2f} px",
    transform=ax_img.transAxes, va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
)

ax_cost.plot(yaw_grid, cost_fixed_15, color="tab:red", lw=1.5, label="pitch 15 deg model")
ax_cost.plot(yaw_grid, cost_fixed_0, color="tab:blue", lw=1.5, label="pitch 0 deg model")
ax_cost.axvline(YAW_P, color="black", ls="--", lw=0.9, label="true yaw 30 deg")
ax_cost.scatter(
    [yaw_grid[i0], yaw_grid[i_mirror]],
    [cost_fixed_0[i0], cost_fixed_0[i_mirror]],
    color="tab:blue", s=34, zorder=5,
)
ax_cost.annotate(
    "mirror basin\npitch0", xy=(yaw_grid[i_mirror], cost_fixed_0[i_mirror]),
    xytext=(-67, 1.55), fontsize=8, color="tab:blue", ha="left",
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
)
ax_cost.annotate(
    "true basin", xy=(yaw_grid[i0], cost_fixed_0[i0]),
    xytext=(38, 0.42), fontsize=8, color="tab:blue",
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
)
ax_cost.set_xlim(-70, 70)
ax_cost.set_ylim(0.0, 3.1)
ax_cost.set_xlabel("Candidate yaw [deg]")
ax_cost.set_ylabel("4-corner reprojection RMS [px]")
ax_cost.set_title("Fixed-t yaw cost: wrong pitch creates a mirror basin")
ax_cost.text(
    0.03, 0.72,
    "free-t pitch0 fit (near truth):\n"
    f"yaw = {fit_clean['yaw']:.2f} deg  (bias {fit_clean['yaw'] - YAW_P:+.2f} deg)\n"
    f"RMS = {fit_clean['rms']:.2f} px\n"
    "(curve is clipped above 3.1 px)",
    transform=ax_cost.transAxes, fontsize=8.2, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.92),
)
ax_cost.legend(loc="upper right")
fig.suptitle("Armor pitch 15 deg vs 0 deg at 3 m, yaw 30 deg", y=1.02)
fig.savefig(FIG_DIR / "pnp_pitch_bias.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Distortion (optional)
# ---------------------------------------------------------------------------
print("-" * 78)
print("[6] distortion k1=-0.1, k2=0.05 (200 runs per distance)")
print("-" * 78)
SEED_DIST = 20261031
rng_dist = np.random.default_rng(SEED_DIST)
print(f"seed        : {SEED_DIST}")
print("D =", DIST_TEST)
for d in [1.0, 3.0, 6.0]:
    t_d = np.array([0.0, 0.0, d])
    uv_clean = project_pose(R_TRUE, t_d, DIST_TEST)
    ignore, undist, with_d = [], [], []
    for _ in range(200):
        uv_obs = add_pixel_noise(uv_clean, rng_dist, sigma=0.5)
        _, t_ignore = solve_pnp(uv_obs, DIST0)
        ignore.append((t_ignore[2] - d) / d * 100.0)
        uv_und = cv2.undistortPoints(
            uv_obs.reshape(-1, 1, 2), K, DIST_TEST, P=K
        ).reshape(-1, 2)
        _, t_und = solve_pnp(uv_und, DIST0)
        undist.append((t_und[2] - d) / d * 100.0)
        _, t_wd = solve_pnp(uv_obs, DIST_TEST)
        with_d.append((t_wd[2] - d) / d * 100.0)
    for name, arr in [("ignore D", ignore), ("explicit undistort", undist), ("solvePnP with D", with_d)]:
        s = stats(np.asarray(arr))
        print(
            f"  d={d:.0f} m  {name:<20s} mean={s['mean']:+7.3f}%  std={s['std']:6.3f}%  "
            f"mean|.|={np.mean(np.abs(arr)):6.3f}%"
        )
    print(
        "    ignored-vs-undistorted mean|error| change: %+.3f percentage points"
        % (np.mean(np.abs(ignore)) - np.mean(np.abs(undist)))
    )
print("\nNote: the centered target makes low-order radial distortion nearly symmetric;")
print("its depth effect is small in this setup.  The corrected and ignored branches")
print("differ mainly in tangential pixel position, not in z.")

print()
print("=" * 78)
print("all figures written:")
for name in [
    "pnp_geometry.png",
    "pnp_projection.png",
    "pnp_depth_noise.png",
    "pnp_yaw_scan.png",
    "pnp_frames.png",
    "pnp_pitch_bias.png",
]:
    p = FIG_DIR / name
    print(f"  {p}  ({p.stat().st_size} bytes)")
print("=" * 78)
