#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical verification of the code-reading conclusions in the autoaim course.

Run:
    MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_projects.py

The script only reads the three project repositories.  It writes figures to
assets/figures/proj_*.png and a machine-readable summary to
verify/verify_results.json.  All numbers quoted in verify/projects_report.md
come from this script's stdout.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Paths / global constants
# ---------------------------------------------------------------------------
ROOT = Path("/home/zh/RM")
COURSE = ROOT / "autoaim_course"
VERIFY = COURSE / "verify"
FIGS = COURSE / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

SP = ROOT / "sp_vision_25"
WUHAN = ROOT / "wuhan"
SHTECH = ROOT / "SHtech_auto_aim"

RESULTS: dict = {}


def rec(key: str, value, note: str = "") -> None:
    """Record a result and print it in a stable, copy-pasteable form."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    RESULTS[key] = value
    if isinstance(value, float):
        print(f"{key:52s} = {value:.12g} {note}")
    else:
        print(f"{key:52s} = {value} {note}")


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_line(path: Path, needle: str) -> tuple[int, str]:
    """Print the first source line containing needle; fail loudly if absent."""
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            print(f"  {path.relative_to(ROOT)}:{i}: {line.strip()}")
            return i, line.strip()
    raise RuntimeError(f"source pattern not found: {path}: {needle!r}")


# ---------------------------------------------------------------------------
# 1. sp_vision_25: ekf_x()[8] vs [7]
# ---------------------------------------------------------------------------
def section_sp_index() -> None:
    head("[1] sp_vision_25: choose_aim_point() tests x[8] (r), not x[7] (w)")
    show_line(SP / "tasks/auto_aim/target.cpp", "// x vx y vy z vz a w r l h")
    show_line(SP / "tasks/auto_aim/target.cpp", "Eigen::VectorXd x0{{center_x")
    show_line(SP / "tasks/auto_aim/target.cpp", "auto r_ok = ekf_.x[8] > 0.05")
    show_line(SP / "tasks/auto_aim/target.cpp", "auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05")
    show_line(SP / "tasks/auto_aim/target.cpp", "std::abs(this->ekf_.x[7]) > 2")
    show_line(SP / "tasks/auto_aim/aimer.cpp", "std::abs(target.ekf_x()[8]) <= 2")
    show_line(SP / "tasks/auto_aim/aimer.cpp", "target.ekf_x()[7] > decision_speed_")
    show_line(SP / "tasks/auto_aim/aimer.cpp", "if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle)")

    r_lo, r_hi, thr = 0.05, 0.5, 2.0
    rec("sp.r_lo_m", r_lo)
    rec("sp.r_hi_m", r_hi)
    rec("sp.threshold", thr)
    rec("sp.threshold_minus_r_hi_m", thr - r_hi)
    rec("sp.threshold_over_r_hi", thr / r_hi)

    r_samples = np.array([0.05, 0.10, 0.20, 0.26, 0.50])
    pred_r = np.abs(r_samples) <= thr
    print(f"  r samples [m]                 : {r_samples}")
    print(f"  |x[8]| <= 2 for these samples : {pred_r}")
    rec("sp.all_r_samples_pass", bool(np.all(pred_r)))

    w_samples = np.array([0.0, 1.0, 1.99, 2.0, 2.51, 4.0, 6.28, 10.0])
    pred_w = np.abs(w_samples) <= thr
    print(f"  w samples [rad/s]             : {w_samples}")
    print(f"  |x[7]| <= 2 for these samples : {pred_w}")
    rec("sp.w_threshold_deg_s", float(np.degrees(thr)))
    rec("sp.outpost_clamp_abs_w", 2.51)
    rec("sp.outpost_clamp_above_threshold", bool(2.51 > thr))

    # min_x in Tracker::update_target()
    tracker_lines = (SP / "tasks/auto_aim/tracker.cpp").read_text().splitlines()
    hits = [(i, line.strip()) for i, line in enumerate(tracker_lines, 1) if "min_x" in line]
    print("  min_x occurrences in tracker.cpp:")
    for i, line in hits:
        print(f"    {i}: {line}")
    non_comment = [(i, line) for i, line in hits if not line.lstrip().startswith("//")]
    rec("sp.min_x_occurrences", len(hits))
    rec("sp.min_x_non_comment_occurrences", len(non_comment))
    rec("sp.min_x_effective_reads", 0)
    rec("sp.min_x_decl_line", hits[0][0])
    rec("sp.min_x_use_is_commented", any(line.lstrip().startswith("//") for _, line in hits))


# ---------------------------------------------------------------------------
# 2. sp_vision_25: NIS threshold
# ---------------------------------------------------------------------------
def section_nis() -> None:
    head("[2] sp_vision_25: NIS threshold 0.711 for chi-square df=4")
    show_line(SP / "tools/extended_kalman_filter.cpp", "constexpr double nis_threshold = 0.711;")
    show_line(SP / "tools/extended_kalman_filter.cpp", "Eigen::VectorXd residual = z_subtract(z, h(x));")
    show_line(SP / "tools/extended_kalman_filter.cpp", "Eigen::MatrixXd S = H * P * H.transpose() + R;")
    show_line(SP / "tools/extended_kalman_filter.cpp", "recent_nis_failures.push_back")
    show_line(SP / "tools/extended_kalman_filter.hpp", "std::deque<int> recent_nis_failures{0};")
    show_line(SP / "tools/extended_kalman_filter.hpp", "size_t window_size = 100;")
    show_line(SP / "tasks/auto_aim/tracker.cpp", "(0.4 * target_.ekf().window_size)")

    df = 4
    thr = 0.711
    q05 = float(chi2.ppf(0.05, df))
    q95 = float(chi2.ppf(0.95, df))
    p_gt = float(chi2.sf(thr, df))
    p_lt = float(chi2.cdf(thr, df))
    rec("nis.df", df)
    rec("nis.threshold", thr)
    rec("nis.chi2_4_q05", q05)
    rec("nis.chi2_4_q95", q95)
    rec("nis.threshold_percentile", 100.0 * p_lt)
    rec("nis.P_NIS_gt_0.711_consistent", p_gt)
    rec("nis.P_NIS_gt_9.4877_consistent", float(chi2.sf(q95, df)))
    rec("nis.P_NIS_lt_0.711_consistent", p_lt)
    rec("nis.q05_matches_0.711_abs_err", abs(q05 - thr))

    # Consequence of the 40 % "bad converge" gate, assuming a standard NIS.
    window = 100
    gate = 0.4 * window
    mean_fail = window * p_gt
    rec("nis.tracker_window", window)
    rec("nis.tracker_gate_count", gate)
    rec("nis.expected_fail_count_if_consistent", mean_fail)
    rec("nis.binom_P_sum_lt_gate_if_consistent", float(binom.cdf(int(gate) - 1, window, p_gt)))

    # The code actually forms a post-fit quantity with posterior P.
    # For the isotropic linear case H=I, P=I, R=I:
    #   K=0.5I, P_post=0.5I, r_post=0.5 r_pre, S_wrong=1.5I
    #   => NIS_code = |r_pre|^2 / 6 = chi2_4 / 3.
    rec("nis.postfit_scale_example", 1.0 / 3.0)
    rec("nis.postfit_iso_P_gt_0.711", float(chi2.sf(3.0 * thr, df)))

    # The same 0.711 is used for NEES, whose state dimension is 11.
    rec("nis.nees_df", 11)
    rec("nis.nees_q05", float(chi2.ppf(0.05, 11)))
    rec("nis.nees_q95", float(chi2.ppf(0.95, 11)))
    rec("nis.nees_0.711_percentile", 100.0 * float(chi2.cdf(thr, 11)))


# ---------------------------------------------------------------------------
# 3. SHtech: h_ vs predict_closest_armor geometry
# ---------------------------------------------------------------------------
def shtech_h_points(center: np.ndarray, yaw: float, r: float, l: float, h: float) -> np.ndarray:
    """Tracker::h_armor_xyz, Tracker.cpp:290-300, reordered to physical (x, y, z).

    h_ builds its return vector as {y_a, x_a, z_a}:
        armor_x = x[0] - r*cos(angle)   // x[0] is y_c
        armor_y = x[2] - r*sin(angle)   // x[2] is x_c
    The [y, x, z] ordering is confirmed by whole_state_2_measurement() and by
    match_armor_id() swapping components [1] and [0].  For a top view we need
    (x_a, y_a) = (x_c - r*sin(angle), y_c - r*cos(angle)).
    """
    pts = []
    for i in range(4):
        angle = math.atan2(math.sin(yaw + i * math.pi / 2), math.cos(yaw + i * math.pi / 2))
        rr = r + l if i in (1, 3) else r
        pts.append(
            (
                center[0] - rr * math.sin(angle),
                center[1] - rr * math.cos(angle),
                center[2] + (h if i in (1, 3) else 0.0),
            )
        )
    return np.asarray(pts, dtype=float)


def shtech_planner_points(
    center: np.ndarray, yaw: float, r: float, l: float, h: float, use_157: bool = True
) -> np.ndarray:
    """Planner::predict_closest_armor, Planner.cpp:472-521."""
    step = 1.57 if use_157 else math.pi / 2.0
    pts = []
    for i in range(4):
        angle = yaw + step * i
        rr = r + l if i in (1, 3) else r
        pts.append(
            (
                center[0] + rr * math.sin(angle),
                center[1] + rr * math.cos(angle),
                center[2] + (h if i in (1, 3) else 0.0),
            )
        )
    return np.asarray(pts, dtype=float)


def hausdorff_2d(a: np.ndarray, b: np.ndarray) -> float:
    dab = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    return float(max(dab.min(axis=1).max(), dab.min(axis=0).max()))


def section_shtech_geometry() -> None:
    head("[3] SHtech: h_armor_xyz() vs Planner::predict_closest_armor()")
    show_line(SHTECH / "predict/Tracker.cpp", "auto armor_x = x[0] - r * std::cos(angle);")
    show_line(SHTECH / "predict/Tracker.cpp", "auto armor_y = x[2] - r * std::sin(angle);")
    show_line(SHTECH / "predict/Tracker.cpp", "Pos3D standard_armor_pos = {standard_armors[j][1], standard_armors[j][0], standard_armors[j][2]};")
    show_line(SHTECH / "planner/Planner.cpp", "double ya = x(0, 0) - x(8, 0) * cos(x(6, 0));")
    show_line(SHTECH / "planner/Planner.cpp", "predicted_center_pos(0, 0) + x(8, 0) * sin")
    show_line(SHTECH / "planner/Planner.cpp", "predicted_center_pos(1, 0) + x(8, 0) * cos")
    show_line(SHTECH / "planner/Planner.cpp", "armor_index = closest_armor_idx;")

    center = np.array([0.0, 3.0, 0.0])
    r, l, h = 0.26, 0.0, 0.0
    rec("shtech.r_m", r)
    rec("shtech.l_m", l)
    rec("shtech.h_m", h)
    rec("shtech.center_x_m", center[0])
    rec("shtech.center_y_m", center[1])

    for yaw in (0.0, 0.7):
        tag = f"{yaw:.1f}"
        hp = shtech_h_points(center, yaw, r, l, h)
        pp = shtech_planner_points(center, yaw, r, l, h, use_157=True)
        pp_exact = shtech_planner_points(center, yaw, r, l, h, use_157=False)

        print(f"\n  yaw = {yaw} rad")
        print("  id | h_armor_xyz (x,y) | predict_closest_armor (x,y) | |delta|")
        for i in range(4):
            d = float(np.linalg.norm(hp[i, :2] - pp[i, :2]))
            print(
                f"   {i} | ({hp[i,0]: .9f}, {hp[i,1]: .9f}) | "
                f"({pp[i,0]: .9f}, {pp[i,1]: .9f}) | {d:.9f}"
            )

        same_id = np.linalg.norm(hp[:, :2] - pp[:, :2], axis=1)
        rec(f"shtech.yaw_{tag}.same_id_delta_m", same_id)
        rec(f"shtech.yaw_{tag}.same_id_delta_mean_m", float(same_id.mean()))
        rec(f"shtech.yaw_{tag}.same_id_delta_max_err_from_2r_m", float(np.max(np.abs(same_id - 2 * r))))

        # nearest h point for every planner point (id permutation)
        dist = np.linalg.norm(hp[:, None, :2] - pp[None, :, :2], axis=2)
        nearest = dist.argmin(axis=1)
        print(f"  planner id -> nearest h id: {nearest.tolist()}")
        rec(f"shtech.yaw_{tag}.planner_to_h_id", nearest.tolist())

        rec(f"shtech.yaw_{tag}.set_hausdorff_code_1.57_m", hausdorff_2d(hp, pp))
        rec(f"shtech.yaw_{tag}.set_hausdorff_exact_pi2_m", hausdorff_2d(hp, pp_exact))

        # Fit the linear map v_planner = M @ v_h for the same id.
        vh = (hp[:, :2] - center[:2]).T  # 2x4
        vp = (pp[:, :2] - center[:2]).T  # 2x4
        M = np.linalg.lstsq(vh.T, vp.T, rcond=None)[0].T  # 2x2
        print(f"  fitted M (same id):\n{M}")
        rec(f"shtech.yaw_{tag}.M_00", float(M[0, 0]))
        rec(f"shtech.yaw_{tag}.M_01", float(M[0, 1]))
        rec(f"shtech.yaw_{tag}.M_10", float(M[1, 0]))
        rec(f"shtech.yaw_{tag}.M_11", float(M[1, 1]))
        rec(f"shtech.yaw_{tag}.M_det", float(np.linalg.det(M)))
        rec(f"shtech.yaw_{tag}.M_rotation_deg", float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        rec(f"shtech.yaw_{tag}.M_max_abs_minus_I", float(np.max(np.abs(M + np.eye(2)))))

        # A pure sin/cos swap is a reflection; show its determinant for contrast.
        M_swap = np.array([[0.0, -1.0], [-1.0, 0.0]])
        rec(f"shtech.yaw_{tag}.swap_matrix_det", float(np.linalg.det(M_swap)))


# ---------------------------------------------------------------------------
# 4. SHtech: armor_jump_interval int truncation
# ---------------------------------------------------------------------------
def section_armor_jump() -> None:
    head("[4] SHtech: const int armor_jump_interval = 0.1f")
    show_line(SHTECH / "planner/Planner.hpp", "const int armor_jump_interval = 0.1f;")
    show_line(SHTECH / "planner/Planner.cpp", "if (dt_since_jump > armor_jump_interval)")

    cpp = VERIFY / "armor_jump_check.cpp"
    exe = Path(tempfile.gettempdir()) / "armor_jump_check_zh"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-Wall", str(cpp), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    print("  C++ output:")
    for line in proc.stdout.splitlines():
        print("   ", line)
    kv = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    rec("armor_jump.interval_as_int", int(kv["armor_jump_interval_as_int"]))
    rec("armor_jump.float_0.1f_as_double", float(kv["float_0.1f_as_double"]))
    rec("armor_jump.dt_zero_predicate", int(kv["dt_zero_gt_interval"]))
    rec("armor_jump.armor_jump_at_dt_zero", int(kv["armor_jump_at_dt_zero"]))
    rec("armor_jump.post_jump_frames", int(kv["post_jump_frames"]))
    rec("armor_jump.armor_jump_true_frames", int(kv["armor_jump_true_frames"]))


# ---------------------------------------------------------------------------
# 5. SHtech: IESEKF prior term sign
# ---------------------------------------------------------------------------
def iekf_scalar(x_pri: float, P: float, H: float, R: float, z: float, n: int, sign: float):
    """sign=+1 is the code; sign=-1 is the standard IEKF."""
    K = P * H / (H * P * H + R)
    x = x_pri
    xs = [x]
    rows = []
    for _ in range(n):
        innovation = z - H * x
        dx_pri = x_pri - x
        correction = K * (innovation + sign * H * dx_pri)
        x_next = x_pri + correction
        rows.append((x, innovation, dx_pri, correction, x_next))
        x = x_next
        xs.append(x)
    return K, xs, rows


def section_iekf() -> None:
    head("[5] SHtech IESEKF_Double_Armor: prior term sign")
    show_line(SHTECH / "mathutils/IESEKF_Double_Armor.hpp", "Matrix_x1d dx_pri = x_minus(X_pri, X_cur);")
    show_line(SHTECH / "mathutils/IESEKF_Double_Armor.hpp", "Matrix_x1d dx = K * (innovation + H * dx_pri);")
    show_line(SHTECH / "mathutils/IESEKF_Double_Armor.hpp", "Matrix_x1d X_next = x_add(X_pri, dx);")
    show_line(SHTECH / "mathutils/IESEKF_Double_Armor.hpp", "int max_iter_ = 5, double stop_threshold_ = 1e-4")

    x_pri, P, H, R, z = 0.0, 1.0, 1.0, 1.0, 1.0
    rec("iekf.x_pri", x_pri)
    rec("iekf.P", P)
    rec("iekf.H", H)
    rec("iekf.R", R)
    rec("iekf.z", z)

    K_std, xs_std, rows_std = iekf_scalar(x_pri, P, H, R, z, 5, sign=-1.0)
    K_code, xs_code, rows_code = iekf_scalar(x_pri, P, H, R, z, 5, sign=+1.0)
    _, xs_code6, _ = iekf_scalar(x_pri, P, H, R, z, 6, sign=+1.0)

    rec("iekf.K", float(K_std))
    rec("iekf.true_posterior", 0.5)
    rec("iekf.code_fixed_point", 0.25)
    rec("iekf.code_map_derivative", -2.0 * K_code)

    print("\n  iteration | standard x_k | code x_k")
    for k in range(6):
        print(f"      {k:2d}     |   {xs_std[k]: .12f} |  {xs_code[k]: .12f}")

    print("\n  code details (x_cur, innovation, dx_pri, correction, x_next):")
    for i, (x, innov, dxp, corr, xn) in enumerate(rows_code, 1):
        print(f"    {i}: {x:.6f}  {innov:.6f}  {dxp:.6f}  {corr:.6f}  {xn:.6f}")

    rec("iekf.standard_final", float(xs_std[-1]))
    rec("iekf.code_final_5iter", float(xs_code[-1]))
    rec("iekf.code_final_6iter", float(xs_code6[-1]))
    rec("iekf.code_final_parity_dependent", bool(abs(xs_code[-1] - xs_code6[-1]) > 1e-9))
    rec("iekf.code_max_step_change", float(np.max(np.abs(np.diff(xs_code)))))
    rec("iekf.standard_max_step_change", float(np.max(np.abs(np.diff(xs_std)))))


# ---------------------------------------------------------------------------
# 6. wuhan: ballistic solve_pitch high-arc preference
# ---------------------------------------------------------------------------
def wuhan_roots(d: float, h: float, v0: float, k: float, g: float):
    A = math.expm1(k * d) / k
    C = A / v0
    B = 0.5 * g * C * C
    disc = A * A - 4.0 * B * (B + h)
    if disc < 0:
        return None
    s = math.sqrt(disc)
    u1 = (A + s) / (2.0 * B)
    u2 = (A - s) / (2.0 * B)
    p1 = math.atan(u1)
    p2 = math.atan(u2)
    t1 = math.expm1(k * d) / (k * v0 * math.cos(p1))
    t2 = math.expm1(k * d) / (k * v0 * math.cos(p2))
    return dict(A=A, B=B, C=C, disc=disc, u1=u1, u2=u2, p1=p1, p2=p2, t1=t1, t2=t2)


def wuhan_forward(pitch: float, v0: float, t: float, k: float, g: float):
    x = math.log1p(k * v0 * math.cos(pitch) * t) / k
    y = v0 * math.sin(pitch) * t - 0.5 * g * t * t
    return x, y


def section_wuhan() -> None:
    head("[6] wuhan: solve_pitch() returns p1 (high arc) before p2")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "const double u1 = (A + sqrt_disc) / denom;")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "const double u2 = (A - sqrt_disc) / denom;")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "if (valid(p1))")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "if (valid(p2))")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "const double x = std::log1p")
    show_line(WUHAN / "src/tasks/base/ballistic_trajectory.hpp", "const double y = v0 * std::sin(pitch) * t")

    g = 9.8
    k = 0.092
    v0 = 30.0
    d = 23.5
    h = 1.0
    rec("wuhan.g", g)
    rec("wuhan.k", k)
    rec("wuhan.v0", v0)
    rec("wuhan.d", d)
    rec("wuhan.h", h)

    roots = wuhan_roots(d, h, v0, k, g)
    assert roots is not None
    for key, val in roots.items():
        rec(f"wuhan.{key}", float(val))

    p1, p2, t1, t2 = roots["p1"], roots["p2"], roots["t1"], roots["t2"]
    lo, hi = math.radians(-45.0), math.radians(60.0)
    rec("wuhan.p1_deg", math.degrees(p1))
    rec("wuhan.p2_deg", math.degrees(p2))
    rec("wuhan.t1_s", t1)
    rec("wuhan.t2_s", t2)
    rec("wuhan.p1_in_range", bool(lo <= p1 <= hi))
    rec("wuhan.p2_in_range", bool(lo <= p2 <= hi))
    rec("wuhan.p1_gt_p2", bool(p1 > p2))
    rec("wuhan.delta_pitch_deg", math.degrees(p1 - p2))
    rec("wuhan.delta_t_s", t1 - t2)

    returned = p1 if (lo <= p1 <= hi) else p2
    rec("wuhan.returned_pitch_deg", math.degrees(returned))
    rec("wuhan.returned_is_p1", bool(returned == p1))

    x1, y1 = wuhan_forward(p1, v0, t1, k, g)
    x2, y2 = wuhan_forward(p2, v0, t2, k, g)
    rec("wuhan.forward_p1_dx_err_m", x1 - d)
    rec("wuhan.forward_p1_dy_err_m", y1 - h)
    rec("wuhan.forward_p2_dx_err_m", x2 - d)
    rec("wuhan.forward_p2_dy_err_m", y2 - h)
    rec("wuhan.forward_p1_abs_max_err_m", max(abs(x1 - d), abs(y1 - h)))
    rec("wuhan.forward_p2_abs_max_err_m", max(abs(x2 - d), abs(y2 - h)))
    rec("wuhan.forward_check_pass_1e-6", bool(max(abs(x1 - d), abs(y1 - h)) < 1e-6))

    # sp_vision_25's drag-free solver at the same (d,h,v0,g); it picks min(t1,t2).
    g_sp = 9.7833
    a = g_sp * d * d / (2 * v0 * v0)
    b = -d
    c = a + h
    delta = b * b - 4 * a * c
    tp1 = (-b + math.sqrt(delta)) / (2 * a)
    tp2 = (-b - math.sqrt(delta)) / (2 * a)
    pp1 = math.atan(tp1)
    pp2 = math.atan(tp2)
    tt1 = d / (v0 * math.cos(pp1))
    tt2 = d / (v0 * math.cos(pp2))
    sp_pitch, sp_t = (pp1, tt1) if tt1 < tt2 else (pp2, tt2)
    rec("wuhan.sp_dragfree_pitch_deg", math.degrees(sp_pitch))
    rec("wuhan.sp_dragfree_fly_time_s", sp_t)
    rec("wuhan.sp_picks_low_arc", bool(sp_pitch < p1))

    # A representative high-arc apex, useful for the report.
    t_apex = np.linspace(0.0, t1, 20000)
    xa, ya = [], []
    for t in t_apex:
        xx, yy = wuhan_forward(p1, v0, t, k, g)
        xa.append(xx)
        ya.append(yy)
    rec("wuhan.high_arc_apex_x_m", float(xa[int(np.argmax(ya))]))
    rec("wuhan.high_arc_apex_y_m", float(np.max(ya)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_sp_index() -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 4.8), gridspec_kw={"height_ratios": [1.0, 1.15]}, constrained_layout=True
    )
    fig.suptitle("sp_vision_25 choose_aim_point(): x[8] is r, but the threshold 2 is an angular-speed test", fontsize=12)

    ax1.broken_barh(
        [(0.05, 0.45)],
        (-0.16, 0.32),
        facecolors="#9ecae1",
        edgecolors="#3182bd",
        linewidth=1.2,
    )
    ax1.axvline(2.0, color="crimson", lw=2.4)
    ax1.set_xlim(0.0, 2.2)
    ax1.set_ylim(-0.34, 0.58)
    ax1.set_yticks([])
    ax1.set_xlabel("radius  r = x[8]  [m]")
    ax1.text(
        0.275,
        0.0,
        "r accepted by diverged()",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#1f4e79",
    )
    ax1.text(
        0.28,
        0.22,
        "r in (0.05, 0.5) m\nmax |x[8]| = 0.5",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f4e79",
    )
    ax1.annotate(
        "threshold 2\n4x above r_max",
        xy=(2.0, 0.06),
        xytext=(1.28, 0.40),
        fontsize=9,
        color="crimson",
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="crimson"),
    )
    ax1.set_title("(a) for every converged non-outpost target, |x[8]| <= 2 is always true", fontsize=10)

    w = np.linspace(0.0, 10.0, 800)
    ax2.axvspan(2.0, 10.0, color="#fdd0a2", alpha=0.75, label="|w| > 2 rad/s: a real small-gyro gate")
    ax2.axvline(2.0, color="crimson", lw=2.4, label="same threshold 2, applied to |x[7]| = |w|")
    ax2.set_xlim(0.0, 10.0)
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_yticks([])
    ax2.set_xlabel("angular speed  |w| = |x[7]|  [rad/s]")
    ax2.annotate(
        "2 rad/s\n= 114.6 deg/s",
        xy=(2.0, 0.18),
        xytext=(2.7, 0.36),
        fontsize=9,
        color="crimson",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="crimson"),
    )
    ax2.annotate(
        "outpost clamp 2.51\n(this is w, not r)",
        xy=(2.51, 0.0),
        xytext=(4.2, -0.33),
        fontsize=9,
        color="#555555",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"),
    )
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax2.set_title("(b) if the intent was 'small gyro', the code should read x[7]", fontsize=10)

    path = FIGS / "proj_sp_index.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [figure] {path}")


def fig_nis_threshold() -> None:
    df = 4
    x = np.linspace(1e-4, 16.0, 2000)
    pdf = chi2.pdf(x, df)
    q05 = float(chi2.ppf(0.05, df))
    q95 = float(chi2.ppf(0.95, df))

    fig, ax = plt.subplots(figsize=(9.8, 5.6), constrained_layout=True)
    ax.plot(x, pdf, color="#1f77b4", lw=2.0, label=r"$\chi^2_4$ pdf")
    ax.fill_between(x, 0, pdf, where=(x <= q05), color="#d62728", alpha=0.28, label="lower 5% tail: 0 ... 0.711")
    ax.fill_between(x, 0, pdf, where=(x >= q95), color="#2ca02c", alpha=0.28, label="upper 5% tail: 9.49 ...")
    ax.axvline(q05, color="#d62728", lw=2.0)
    ax.axvline(q95, color="#2ca02c", lw=2.0)
    ax.annotate(
        "0.711 = 5.00th percentile\n(code calls this '95% confidence')",
        xy=(q05, chi2.pdf(q05, df)),
        xytext=(1.6, 0.118),
        fontsize=9,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#d62728"),
    )
    ax.annotate(
        "9.49 = 95th percentile",
        xy=(q95, chi2.pdf(q95, df)),
        xytext=(9.7, 0.155),
        fontsize=9,
        color="#2ca02c",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#2ca02c"),
    )
    ax.text(
        0.985,
        0.93,
        "consistent filter (standard NIS ~ chi2_4):\n"
        "P(NIS > 0.711) = 0.950\n"
        "P(NIS > 9.49)  = 0.050",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.95),
    )
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.205)
    ax.set_xlabel("NIS value")
    ax.set_ylabel("probability density")
    ax.set_title(r"NIS threshold in sp_vision_25: 0.711 is the 5% quantile of $\chi^2_4$, not the 95% one", fontsize=11)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=8.3,
        frameon=False,
    )

    path = FIGS / "proj_nis_threshold.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


def fig_shtech_geometry() -> None:
    center = np.array([0.0, 3.0, 0.0])
    r, l, h = 0.26, 0.0, 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)
    fig.suptitle(
        "SHtech: same-id armor positions from h_armor_xyz() and predict_closest_armor()\n"
        "r = 0.26 m, l = 0, h = 0, center = (0, 3); h_ returns [y, x, z], reordered to physical (x, y, z)",
        fontsize=11.5,
    )

    for ax, yaw in zip(axes, (0.0, 0.7)):
        hp = shtech_h_points(center, yaw, r, l, h)
        pp = shtech_planner_points(center, yaw, r, l, h, use_157=True)

        ring = Circle((center[0], center[1]), r, fill=False, ls="--", lw=0.9, color="#999999", zorder=1)
        ax.add_patch(ring)
        ax.axhline(center[1], color="#cccccc", lw=0.7, ls=":", zorder=0)
        ax.axvline(center[0], color="#cccccc", lw=0.7, ls=":", zorder=0)

        for i in range(4):
            ax.annotate(
                "",
                xy=(pp[i, 0], pp[i, 1]),
                xytext=(hp[i, 0], hp[i, 1]),
                arrowprops=dict(arrowstyle="<->", color="#777777", lw=0.9, alpha=0.65, shrinkA=0, shrinkB=0),
                zorder=2,
            )

        ax.scatter(
            hp[:, 0],
            hp[:, 1],
            s=150,
            facecolors="none",
            edgecolors="#1f77b4",
            linewidths=2.0,
            label="h_armor_xyz()",
            zorder=4,
        )
        ax.scatter(
            pp[:, 0],
            pp[:, 1],
            marker="x",
            s=85,
            color="#d62728",
            linewidths=2.3,
            label="predict_closest_armor()",
            zorder=5,
        )
        ax.scatter([center[0]], [center[1]], marker="*", s=170, color="black", zorder=6, label="vehicle center")

        # Combined labels: h_i and p_(i+2) occupy the same location (up to 0.6 mm).
        for i in range(4):
            v = hp[i, :2] - center[:2]
            ang = math.atan2(v[1], v[0])
            off = 0.105 * np.array([math.cos(ang), math.sin(ang)])
            ax.text(
                hp[i, 0] + off[0],
                hp[i, 1] + off[1],
                f"h{i} / p{(i + 2) % 4}",
                fontsize=8.5,
                color="#1f77b4",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
                zorder=7,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.78, 0.78)
        ax.set_ylim(2.22, 3.78)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.set_title(
            f"({chr(97) if yaw == 0.0 else chr(98)}) yaw = {yaw:.1f} rad\n"
            "same-id |delta| = 0.520 m; p_i = h_(i+2) (mod 4)",
            fontsize=9.5,
        )
        ax.plot([], [], color="#777777", lw=0.9, label="same-id h_i -> p_i")
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.95)

        if yaw == 0.0:
            ax.text(
                0.975,
                0.03,
                "same-id map M = -I + O(1.6e-3)\n"
                "det = +0.999999, rotation = 180 deg\n"
                "no translation; 4-point set identical\n"
                "0.62 mm residual: literal 1.57 vs pi/2",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#999999", alpha=0.94),
                zorder=8,
            )

    path = FIGS / "proj_shtech_geometry.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


def fig_armor_jump() -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 5.6), gridspec_kw={"height_ratios": [1.0, 1.25]}, constrained_layout=True
    )
    fig.suptitle("SHtech armor_jump_interval: int(0.1f) = 0, so the intended 0.1 s gate has zero width", fontsize=11.5)

    ax1.barh([1], [0.1], left=0.0, height=0.34, color="#9ecae1", edgecolor="#3182bd", label="intended: 0.1 s (double)")
    ax1.plot([0.0], [0.0], marker="o", color="crimson", ms=11, label="actual: width = 0 (int truncation)")
    ax1.axvline(0.0, color="black", ls="--", lw=1.0)
    ax1.set_xlim(-0.01, 0.25)
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["actual\nint = 0", "intended\ndouble = 0.1"])
    ax1.set_xlabel("time since armor jump [s]")
    ax1.text(0.052, 1.0, "0.1 s", va="center", fontsize=9, color="#1f4e79")
    ax1.text(0.012, 0.03, "0 s: only dt = 0 would set armor_jump = true", fontsize=8.5, color="crimson")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax1.set_title("(a) fire-disable window: intended 0.1 s, actual 0 s", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)

    dt = np.linspace(0.0, 0.25, 2000)
    ax2.step(dt, (dt > 0.0).astype(float), where="post", color="crimson", lw=2.0, label="actual: dt > 0")
    ax2.step(dt, (dt > 0.1).astype(float), where="post", color="#3182bd", lw=2.0, ls="--", label="intended: dt > 0.1")
    ax2.set_xlim(0.0, 0.25)
    ax2.set_ylim(-0.08, 1.18)
    ax2.set_xlabel("dt since jump [s]")
    ax2.set_ylabel("dt > interval ?")
    ax2.set_yticks([0, 1])
    ax2.legend(loc="center right", fontsize=8.5, framealpha=0.95)
    ax2.set_title("(b) armor_jump = NOT(dt > interval): actual predicate is true for every dt > 0", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)

    path = FIGS / "proj_armor_jump.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


def fig_iekf_sign() -> None:
    x_pri, P, H, R, z = 0.0, 1.0, 1.0, 1.0, 1.0
    _, xs_std, _ = iekf_scalar(x_pri, P, H, R, z, 5, sign=-1.0)
    _, xs_code, _ = iekf_scalar(x_pri, P, H, R, z, 5, sign=+1.0)
    it = np.arange(0, 6)

    fig, ax = plt.subplots(figsize=(10.4, 5.2), constrained_layout=True)
    ax.plot(it, xs_std, "o-", lw=2.2, ms=8, color="#1f77b4", label="standard IEKF: + H(x_cur - x_pri)")
    ax.plot(it, xs_code, "s--", lw=2.2, ms=7, color="#d62728", label="code: + H(x_pri - x_cur)")
    ax.axhline(0.5, color="#555555", ls=":", lw=1.3, label="true posterior = 0.5")
    ax.axhline(0.25, color="#ff7f0e", ls=":", lw=1.3, label="code map fixed point = 0.25")
    ax.set_xticks(it)
    ax.set_xlim(-0.2, 5.4)
    ax.set_ylim(-0.08, 0.62)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("state estimate x_k")
    ax.set_title(
        "Scalar IEKF: x_pri = 0, P = H = R = 1, z = 1 (K = 0.5)\n"
        "code map x_(k+1) = 0.5 - x_k has slope -1: period-2, not convergence",
        fontsize=10.5,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.2, framealpha=0.95)

    path = FIGS / "proj_iekf_sign.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


def fig_wuhan_ballistic() -> None:
    g, k, v0, d, h = 9.8, 0.092, 30.0, 23.5, 1.0
    roots = wuhan_roots(d, h, v0, k, g)
    assert roots is not None
    p1, p2, t1, t2 = roots["p1"], roots["p2"], roots["t1"], roots["t2"]

    def traj(pitch, tf, n=1200):
        ts = np.linspace(0.0, tf, n)
        xs, ys = [], []
        for t in ts:
            x, y = wuhan_forward(pitch, v0, t, k, g)
            xs.append(x)
            ys.append(y)
        return np.asarray(xs), np.asarray(ys)

    x1, y1 = traj(p1, t1)
    x2, y2 = traj(p2, t2)

    fig, ax = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    ax.plot(x1, y1, color="#1f77b4", lw=2.2, label=f"p1 = {math.degrees(p1):.2f} deg, t1 = {t1:.3f} s (returned)")
    ax.plot(x2, y2, color="#ff7f0e", lw=2.2, label=f"p2 = {math.degrees(p2):.2f} deg, t2 = {t2:.3f} s")
    ax.scatter([d], [h], marker="*", s=210, color="black", zorder=5, label=f"target ({d:.1f}, {h:.1f}) m")
    ax.axhline(0.0, color="#666666", lw=0.9)
    ax.set_xlim(0.0, d * 1.08)
    ax.set_ylim(0.0, 40.0)
    ax.set_xlabel("horizontal distance x [m]")
    ax.set_ylabel("height y [m]")
    ax.set_title("wuhan solve_pitch(): p1 (high arc) is returned before p2")
    ax.annotate(
        f"high-arc apex: y = {np.max(y1):.1f} m",
        xy=(x1[int(np.argmax(y1))], np.max(y1)),
        xytext=(2.0, np.max(y1) * 0.86),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f77b4"),
    )
    ax.text(
        0.985,
        0.965,
        f"v0 = {v0:.0f} m/s, k = {k:.3f} 1/m, g = {g:.1f} m/s^2\n"
        f"forward_model residual at target:\n"
        f"|dx| = {abs(x1[-1] - d):.1e} m, |dy| = {abs(y1[-1] - h):.1e} m",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.95),
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)

    path = FIGS / "proj_wuhan_ballistic.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("autoaim_course numerical verification")
    print(f"  python: {subprocess.check_output(['/usr/bin/python3', '--version'], text=True).strip()}")
    print(f"  numpy={np.__version__}, scipy={__import__('scipy').__version__}, "
          f"matplotlib={matplotlib.__version__}")
    print(f"  repository root: {ROOT}")

    section_sp_index()
    section_nis()
    section_shtech_geometry()
    section_armor_jump()
    section_iekf()
    section_wuhan()

    head("Figures")
    fig_sp_index()
    fig_nis_threshold()
    fig_shtech_geometry()
    fig_armor_jump()
    fig_iekf_sign()
    fig_wuhan_ballistic()

    out_json = VERIFY / "verify_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n  [json] {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
