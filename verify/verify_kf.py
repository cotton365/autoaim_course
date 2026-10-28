#!/usr/bin/env python3
"""KF / EKF / IEKF / ESKF numerical verification for the auto-aim course.

Run:  MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_kf.py
Outputs six figures in assets/figures/ and verify/kf_results.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-zh")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets" / "figures"
VER = ROOT / "verify"
SEED = 20261026

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 160,
})


def save(fig, name):
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[figure] {path}")
    return str(path.relative_to(ROOT))


def gauss(x, m, v):
    return np.exp(-0.5 * (np.asarray(x) - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


# --------------------------------------------------------------------------- 1D
def one_d():
    print("\n=== 1. 1-D KF ===")
    mu0, P0, z, R = 0.0, 1.0, 1.2, 0.5
    P_prod = 1.0 / (1.0 / P0 + 1.0 / R)
    mu_prod = P_prod * (mu0 / P0 + z / R)
    K = P0 / (P0 + R)
    mu_kf = mu0 + K * (z - mu0)
    P_kf = (1 - K) * P0
    errs = (abs(mu_prod - mu_kf), abs(P_prod - P_kf))
    print(f"analytic: mu={mu_prod:.17g}  var={P_prod:.17g}")
    print(f"KF      : mu={mu_kf:.17g}  var={P_kf:.17g}  K={K:.17g}")
    print(f"abs err : mean={errs[0]:.3e}  var={errs[1]:.3e}  max={max(errs):.3e}")
    assert max(errs) < 1e-12

    x = np.linspace(-3.2, 4.2, 901)
    pp, pl, po = gauss(x, mu0, P0), gauss(x, z, R), gauss(x, mu_prod, P_prod)
    cp, cl, co = "#1f77b4", "#ff7f0e", "#2ca02c"
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(x, pp, color=cp, lw=2.0, label="prior  N(0, 1)")
    ax.plot(x, pl, color=cl, lw=2.0, label="likelihood  N(1.2, 0.5)")
    ax.plot(x, po, color=co, lw=2.4, label="posterior  N(0.8, 1/3)")
    ax.fill_between(x, 0, po, where=(x >= mu_prod - np.sqrt(P_prod)) & (x <= mu_prod + np.sqrt(P_prod)),
                    color=co, alpha=0.15, label="posterior 1-sigma region")
    for mu, c in [(mu0, cp), (z, cl), (mu_prod, co)]:
        ax.vlines(mu, 0, 0.76, color=c, ls=":", lw=1.2)
    rows = [(0.985, mu0, np.sqrt(P0), cp, "prior"),
            (0.930, z, np.sqrt(R), cl, "likelihood"),
            (0.875, mu_prod, np.sqrt(P_prod), co, "posterior")]
    for y, mu, sig, c, lab in rows:
        ax.errorbar(mu, y, xerr=sig, fmt="o", color=c, ms=4, capsize=5,
                    elinewidth=1.4, label=f"{lab}: {mu:.3f} +/- {sig:.3f}")
    ax.set_xlim(-3.2, 4.2)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("state x")
    ax.set_ylabel("probability density")
    ax.set_title("1-D Gaussian update: prior x likelihood = posterior")
    ax.legend(loc="center right", framealpha=0.92)
    fig.tight_layout()
    fig_rel = save(fig, "kf_prior_posterior.png")
    return {"prior_mean": mu0, "prior_var": P0, "z": z, "R": R, "K": float(K),
            "analytic": {"mean": float(mu_prod), "var": float(P_prod)},
            "kf": {"mean": float(mu_kf), "var": float(P_kf)},
            "abs_error_max": float(max(errs)), "figure": fig_rel}


# --------------------------------------------------------------------------- 2D
def ellipse(ax, mean, cov, ns, color, ls, alpha, lw=1.7):
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    ax.add_patch(Ellipse(mean, 2 * ns * np.sqrt(w[0]), 2 * ns * np.sqrt(w[1]),
                         angle=ang, facecolor=color, edgecolor=color,
                         alpha=alpha, lw=lw, ls=ls))


def two_d():
    print("\n=== 2. 2-D covariance update ===")
    P0 = np.array([[2.0, 0.8], [0.8, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.3]])
    z, x0 = 1.0, np.zeros(2)
    S = H @ P0 @ H.T + R
    K = P0 @ H.T @ np.linalg.inv(S)
    x1 = x0 + K @ np.array([z])
    P1 = (np.eye(2) - K @ H) @ P0
    P1 = 0.5 * (P1 + P1.T)
    print(f"S={S[0,0]:.17g}  K=[{K[0,0]:.17g}, {K[1,0]:.17g}]")
    print(f"P+ = [[{P1[0,0]:.17g}, {P1[0,1]:.17g}], [{P1[1,0]:.17g}, {P1[1,1]:.17g}]]")
    print(f"x+ = [{x1[0]:.17g}, {x1[1]:.17g}]")

    cp, cn = "#1f77b4", "#d62728"
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    for ns, ls in [(1, "-"), (2, "--")]:
        ellipse(ax, x0, P0, ns, cp, ls, 0.20 if ns == 1 else 0.08)
        ellipse(ax, x1, P1, ns, cn, ls, 0.20 if ns == 1 else 0.08)
    ax.axvspan(z - np.sqrt(R[0, 0]), z + np.sqrt(R[0, 0]), color="0.2", alpha=0.10, lw=0)
    ax.axvline(z, color="0.15", ls="--", lw=1.5)
    ax.plot(*x0, "o", color=cp, ms=6)
    ax.plot(*x1, "s", color=cn, ms=6)
    ax.annotate("", xy=x1, xytext=x0, arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
    ax.text(z + 0.08, 2.72, "measurement z = 1.0\nshaded: z +/- sqrt(R)", fontsize=9)
    handles = [
        Patch(facecolor=cp, edgecolor=cp, alpha=0.20, label="prior 1-sigma"),
        Patch(facecolor=cp, edgecolor=cp, alpha=0.08, ls="--", label="prior 2-sigma"),
        Patch(facecolor=cn, edgecolor=cn, alpha=0.20, label="posterior 1-sigma"),
        Patch(facecolor=cn, edgecolor=cn, alpha=0.08, ls="--", label="posterior 2-sigma"),
        Line2D([0], [0], color="0.15", ls="--", lw=1.5, label="measurement z = 1.0"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.92)
    ax.text(0.015, 0.025,
            "K = [0.8696, 0.3478]\nP+ = [[0.2609, 0.1043],\n      [0.1043, 0.7217]]",
            transform=ax.transAxes, family="monospace", fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="0.7"))
    ax.set_xlim(-4, 4.8)
    ax.set_ylim(-3, 3.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2-D prior and posterior covariance ellipses")
    fig.tight_layout()
    fig_rel = save(fig, "kf_cov_ellipse.png")
    return {"K": K[:, 0].tolist(), "P_post": P1.tolist(), "x_post": x1.tolist(),
            "S": float(S[0, 0]), "figure": fig_rel}


# --------------------------------------------------------------------------- NIS/NEES
def nis_nees(nr=200, ns=50):
    print("\n=== 3. NIS / NEES ===")
    rng = np.random.default_rng(SEED)
    dt, q, rv = 0.1, 0.05, 0.2
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = q * G @ G.T
    R = rv * np.eye(2)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    P0 = np.diag([0.25, 0.25, 0.5, 0.5])
    xfix = np.array([2.0, -1.0, 1.5, 0.8])
    nis = np.zeros((nr, ns))
    nees = np.zeros((nr, ns))
    for i in range(nr):
        xt = xfix.copy()
        xh = xt + rng.multivariate_normal(np.zeros(4), P0)
        P = P0.copy()
        for k in range(ns):
            xt = F @ xt + G @ rng.normal(0, np.sqrt(q), 2)
            xp = F @ xh
            Pp = F @ P @ F.T + Q
            z = H @ xt + rng.normal(0, np.sqrt(rv), 2)
            nu = z - H @ xp
            S = H @ Pp @ H.T + R
            nis[i, k] = nu @ np.linalg.solve(S, nu)
            K = Pp @ H.T @ np.linalg.inv(S)
            xh = xp + K @ nu
            P = (np.eye(4) - K @ H) @ Pp
            P = 0.5 * (P + P.T)
            e = xh - xt
            nees[i, k] = e @ np.linalg.solve(P, e)
    b2, b4 = chi2.ppf([0.025, 0.975], 2), chi2.ppf([0.025, 0.975], 4)
    cn = np.mean((nis >= b2[0]) & (nis <= b2[1]))
    ce = np.mean((nees >= b4[0]) & (nees <= b4[1]))
    print(f"mean NIS={nis.mean():.10f} (df=2), bounds={b2.tolist()}, coverage={cn:.10f}")
    print(f"mean NEES={nees.mean():.10f} (df=4), bounds={b4.tolist()}, coverage={ce:.10f}")
    print(f"per-step NIS mean range [{nis.mean(0).min():.6f}, {nis.mean(0).max():.6f}]")
    print(f"per-step NEES mean range [{nees.mean(0).min():.6f}, {nees.mean(0).max():.6f}]")
    q05_4, q95_4 = chi2.ppf(0.05, 4), chi2.ppf(0.95, 4)
    print(f"cross-check chi2_4 5%= {q05_4:.12f}, 95%= {q95_4:.12f}; "
          f"sp_vision_25 threshold 0.711 is the 5% lower quantile.")

    t = np.arange(1, ns + 1)
    fig, axs = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    for ax, dat, bd, df, ylab, col, cover in [
        (axs[0], nis.mean(0), b2, 2, "NIS (2-D position innovation)", "#1f77b4", cn),
        (axs[1], nees.mean(0), b4, 4, "NEES (4-D state error)", "#d62728", ce)]:
        ax.fill_between(t, bd[0], bd[1], color="0.55", alpha=0.16, label="chi-square 95% interval")
        ax.axhline(bd[0], color="0.35", ls="--", lw=1.0)
        ax.axhline(bd[1], color="0.35", ls="--", lw=1.0)
        ax.axhline(df, color="0.15", ls=":", lw=1.3, label=f"theoretical mean = {df}")
        ax.plot(t, dat, color=col, lw=1.8, marker="o", ms=2.4, label="Monte Carlo mean")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, bd[1] * 1.22)
        ax.text(0.985, 0.95, f"mean = {dat.mean():.4f}\n95% coverage = {cover*100:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.75"))
        ax.legend(loc="center left", ncol=1, framealpha=0.92)
    axs[0].set_title(f"NIS / NEES consistency over {nr} runs x {ns} steps")
    axs[1].set_xlabel("time step k")
    fig.tight_layout()
    fig_rel = save(fig, "kf_nis_nees.png")
    return {"runs": nr, "steps": ns, "nis_mean": float(nis.mean()), "nees_mean": float(nees.mean()),
            "nis_bounds": b2.tolist(), "nees_bounds": b4.tolist(),
            "nis_coverage": float(cn), "nees_coverage": float(ce),
            "nis_time_mean": nis.mean(0).tolist(), "nees_time_mean": nees.mean(0).tolist(),
            "chi2_4_q05": float(q05_4), "chi2_4_q95": float(q95_4),
            "figure": fig_rel}


# --------------------------------------------------------------------------- IEKF
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def h_rb(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def H_rb(x):
    px, py = x[:2]
    r2 = px * px + py * py
    r = np.sqrt(r2)
    return np.array([[px / r, py / r, 0, 0], [-py / r2, px / r2, 0, 0]])


def rb_update(xpri, Ppri, z, R, kind, max_iter=5, tol=1e-8):
    x = xpri.copy()
    hist = []
    K, Hk = None, H_rb(x)
    for _ in range(max_iter):
        Hk = H_rb(x)
        S = Hk @ Ppri @ Hk.T + R
        K = Ppri @ Hk.T @ np.linalg.inv(S)
        inn = z - h_rb(x)
        inn[1] = wrap(inn[1])
        if kind == "ekf":
            x = xpri + K @ inn
            hist.append(x.copy())
            break
        pr = (x - xpri) if kind == "iekf" else (xpri - x)
        xn = xpri + K @ (inn + Hk @ pr)
        hist.append(xn.copy())
        if np.linalg.norm(xn - x) < tol:
            x = xn
            break
        x = xn
    P = (np.eye(4) - K @ Hk) @ Ppri
    P = 0.5 * (P + P.T)
    return x, P, hist


def rb_run(seed, kind, ns=30):
    rng = np.random.default_rng(seed)
    dt, q = 0.1, 0.5
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = q * G @ G.T
    R = np.diag([0.3 ** 2, 0.006 ** 2])
    xt = np.array([30.0, 10.0, -0.8, 0.4])
    xh = xt + np.array([4.0, -3.0, 0.8, -0.4])
    P = np.diag([16.0, 16.0, 4.0, 4.0])
    err = np.zeros((ns, 4))
    prior_pos = None
    first_pos = []
    iters = []
    for k in range(ns):
        xt = F @ xt + G @ rng.normal(0, np.sqrt(q), 2)
        xp = F @ xh
        Pp = F @ P @ F.T + Q
        z = h_rb(xt) + rng.normal(0, [0.3, 0.006])
        z[1] = wrap(z[1])
        xh, P, hist = rb_update(xp, Pp, z, R, kind)
        err[k] = xh - xt
        iters.append(len(hist))
        if k == 0:
            prior_pos = float(np.linalg.norm((xp - xt)[:2]))
            first_pos = [float(np.linalg.norm((h - xt)[:2])) for h in hist]
    return err, prior_pos, first_pos, iters


def scalar_iekf():
    print("\n=== 4a. Scalar IEKF ===")
    xpri, P, H, R, z = 0.0, 1.0, 1.0, 1.0, 1.0
    K = P * H / (H * P * H + R)
    ekf = [xpri] + [xpri + K * (z - H * xpri)] * 5
    iekf = [xpri]
    x = xpri
    for _ in range(5):
        x = xpri + K * (z - H * x + H * (x - xpri))
        iekf.append(x)
    sht = [xpri]
    x = xpri
    for _ in range(5):
        x = xpri + K * (z - H * x + H * (xpri - x))
        sht.append(x)
    print(f"K={K:.17g}, correct posterior={xpri+K*(z-H*xpri):.17g}")
    print(" k | EKF | standard IEKF | SHtech sign")
    for i in range(6):
        print(f"{i:2d} | {ekf[i]:.12f} | {iekf[i]:.12f} | {sht[i]:.12f}")
    print(f"SHtech 2-cycle: odd={sht[1]:.17g}, even={sht[2]:.17g}; algebraic fixed point={(xpri+K*z)/(1+2*K*H):.17g}")

    k = np.arange(6)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(k, ekf, color="0.2", ls="--", marker="o", ms=5, lw=1.6, label="EKF (one update, repeated reference)")
    ax.plot(k, iekf, color="#1f77b4", marker="s", ms=4.5, lw=1.9, label="standard IEKF")
    ax.plot(k, sht, color="#d62728", marker="D", ms=5, lw=2.0, label="SHtech code sign (x_pri - x_k)")
    ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
    ax.text(3.35, 0.525, "correct fixed point 0.5", fontsize=9, color="0.35")
    for i, val in enumerate(sht):
        ax.annotate(f"{val:.1f}", xy=(i, val), xytext=(4, 8), textcoords="offset points",
                    fontsize=8.5, color="#d62728")
    ax.set_xlim(-0.15, 5.25)
    ax.set_ylim(-0.08, 0.65)
    ax.set_xticks(k)
    ax.set_xlabel("iteration k (value produced by that update)")
    ax.set_ylabel("state estimate")
    ax.set_title("Scalar linear IEKF: correct sign converges, SHtech sign forms a 2-cycle")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig_rel = save(fig, "kf_iekf_scalar.png")
    return {"K": K, "ekf": [float(v) for v in ekf], "iekf": [float(v) for v in iekf],
            "shtech": [float(v) for v in sht], "figure": fig_rel}


def iekf():
    scalar = scalar_iekf()
    print("\n=== 4b. Range-bearing 100-run Monte Carlo ===")
    kinds = ["ekf", "iekf", "shtech"]
    runs = {m: [] for m in kinds}
    for m in kinds:
        for s in range(100):
            runs[m].append(rb_run(SEED + s, m))
    stats = {}
    print("method   firstRMSE timeRMSE finalPosRMSE finalVelRMSE medianFinal meanIters")
    for m in kinds:
        E = np.array([r[0] for r in runs[m]])
        pos = np.linalg.norm(E[:, :, :2], axis=2)
        vel = np.linalg.norm(E[:, :, 2:], axis=2)
        st = {"first_pos_rmse": float(np.sqrt(np.mean(pos[:, 0] ** 2))),
              "time_pos_rmse": float(np.sqrt(np.mean(pos ** 2))),
              "final_pos_rmse": float(np.sqrt(np.mean(pos[:, -1] ** 2))),
              "final_vel_rmse": float(np.sqrt(np.mean(vel[:, -1] ** 2))),
              "median_final_pos": float(np.median(pos[:, -1])),
              "mean_iters": float(np.mean([it for r in runs[m] for it in r[3]]))}
        stats[m] = st
        print(f"{m:7s} {st['first_pos_rmse']:9.6f} {st['time_pos_rmse']:8.6f} "
              f"{st['final_pos_rmse']:12.6f} {st['final_vel_rmse']:12.6f} "
              f"{st['median_final_pos']:11.6f} {st['mean_iters']:9.4f}")

    time_rmse = {}
    for m in kinds:
        E = np.array([r[0] for r in runs[m]])
        pos = np.linalg.norm(E[:, :, :2], axis=2)
        time_rmse[m] = np.sqrt(np.mean(pos ** 2, axis=0)).tolist()

    prior = float(np.sqrt(np.mean([r[1] ** 2 for r in runs["iekf"]])))
    iter_rmse = {}
    for m in kinds:
        vals = []
        for i in range(5):
            e = [r[2][min(i, len(r[2]) - 1)] for r in runs[m]]
            vals.append(float(np.sqrt(np.mean(np.square(e)))))
        iter_rmse[m] = vals
    print(f"prior position RMSE={prior:.6f}")
    for m in kinds:
        print(f"  {m}: " + ", ".join(f"{v:.6f}" for v in iter_rmse[m]))

    cols = {"ekf": "0.15", "iekf": "#1f77b4", "shtech": "#d62728"}
    labs = {"ekf": "EKF (one linearization)", "iekf": "standard IEKF", "shtech": "SHtech code sign"}
    fig, axs = plt.subplots(2, 1, figsize=(9.4, 8.3), gridspec_kw={"height_ratios": [1.12, 1.0]})
    ax = axs[0]
    k = np.arange(1, 31)
    for m in kinds:
        ax.semilogy(k, time_rmse[m], color=cols[m], lw=2.0, marker="o", ms=2.6, label=labs[m])
    ax.set_xlabel("time step k")
    ax.set_ylabel("position RMSE [m]")
    ax.set_title("100-run MC range-bearing tracking: position RMSE vs time")
    ax.legend(loc="center right", framealpha=0.92)

    ax = axs[1]
    ax.semilogy([0], [prior], "o", color="0.0", ms=6, label="common prior")
    ax.semilogy([1], [iter_rmse["ekf"][0]], "o", color=cols["ekf"], ms=6, label="EKF after one update")
    ax.hlines(iter_rmse["ekf"][0], 1, 5, colors=cols["ekf"], linestyles="--", lw=1.2, alpha=0.8)
    ax.semilogy(np.arange(1, 6), iter_rmse["iekf"], "-s", color=cols["iekf"], ms=4.5, lw=2.0, label="standard IEKF")
    ax.semilogy(np.arange(1, 6), iter_rmse["shtech"], "-D", color=cols["shtech"], ms=4.5, lw=2.0, label="SHtech code sign")
    ax.set_xlim(-0.15, 6.4)
    ax.set_xticks(np.arange(6))
    ax.set_xlabel("iteration index at the first update")
    ax.set_ylabel("position RMSE [m]")
    ax.set_title("First update only: position RMSE across the same 100 runs")
    ax.text(0.08, prior * 1.16, "prior", color="0.0", fontsize=9, va="bottom")
    ax.text(2.15, iter_rmse["ekf"][0] * 1.18, "EKF\n(1 update)", color=cols["ekf"], fontsize=8.5, va="bottom")
    ax.text(5.1, iter_rmse["iekf"][-1], "IEKF", color=cols["iekf"], fontsize=9, va="center")
    ax.text(5.1, iter_rmse["shtech"][-1], "SHtech", color=cols["shtech"], fontsize=9, va="center")
    ax.text(0.02, 0.96, f"prior RMSE = {prior:.3f} m; EKF first-update RMSE = {iter_rmse['ekf'][0]:.3f} m",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.75"))
    fig.tight_layout()
    fig_rel = save(fig, "kf_iekf.png")
    scalar["range_bearing"] = {"stats": stats, "prior_rmse": prior, "iter_rmse": iter_rmse,
                               "time_rmse": time_rmse, "figure": fig_rel}
    return scalar


# --------------------------------------------------------------------------- ESKF
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def expm(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    K = skew(v)
    if th < 1e-12:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def logm(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-10:
        return 0.5 * v
    return th / (2 * np.sin(th)) * v


def inject(delta, x):
    return x[0] + delta[:3], x[1] @ expm(delta[3:])


def box_minus(x, y):
    """Tangent vector from x to y (so box_minus(x, inject(d,x)) == d)."""
    return np.r_[y[0] - x[0], logm(x[1].T @ y[1])]


def eskf():
    print("\n=== 5. ESKF / SO(3) ===")
    rng = np.random.default_rng(SEED)
    ie, le = [], []
    for _ in range(200):
        p = rng.normal(0, 2, 3)
        R = expm(rng.normal(0, 1, 3))
        phi = rng.normal(size=3)
        phi = phi / np.linalg.norm(phi) * rng.uniform(1e-6, 2.8)
        d = np.r_[rng.normal(0, 1, 3), phi]
        x = (p, R)
        ie.append(float(np.max(np.abs(box_minus(x, inject(d, x)) - d))))
        le.append(float(np.max(np.abs(logm(expm(phi)) - phi))))
    print(f"inject/boxminus max err={max(ie):.3e}; Log(Exp) max err={max(le):.3e}")
    assert max(ie) < 1e-9 and max(le) < 1e-9

    dt, n = 0.05, 100
    v = np.array([1.2, 0.3, -0.2])
    w = np.array([1.0, -0.6, 0.8])
    p0 = np.array([1.0, 2.0, 3.0])
    R0 = expm([0.0, 0.0, 1.4])
    dp = np.array([0.2, -0.1, 0.15])
    dth = np.array([0.08, -0.05, 0.06])
    pt, Rt = p0.copy(), R0.copy()
    pe, Re = p0 + dp, R0 @ expm(dth)
    pn, phi = pe.copy(), logm(Re)
    Pe = np.diag([0.02] * 3 + [0.01] * 3)
    Pn = Pe.copy()
    Q = 1e-6 * np.eye(6)
    oe, on, pen, pnn = [], [], [], []
    cpe, cpn, coe, con = [], [], [], []

    def rec():
        oe.append(float(np.linalg.norm(logm(Rt.T @ Re))))
        on.append(float(np.linalg.norm(logm(Rt.T @ expm(phi)))))
        pen.append(float(np.linalg.norm(pe - pt)))
        pnn.append(float(np.linalg.norm(pn - pt)))
        cpe.append(float(np.trace(Pe[:3, :3])))
        cpn.append(float(np.trace(Pn[:3, :3])))
        coe.append(float(np.trace(Pe[3:, 3:])))
        con.append(float(np.trace(Pn[3:, 3:])))

    rec()
    for _ in range(n):
        pt = pt + Rt @ v * dt
        Rt = Rt @ expm(w * dt)
        Fe = np.eye(6)
        Fe[:3, 3:] = -Re @ skew(v) * dt
        Fe[3:, 3:] = expm(-w * dt)
        Pe = Fe @ Pe @ Fe.T + Q
        Pe = 0.5 * (Pe + Pe.T)
        pe = pe + Re @ v * dt
        Re = Re @ expm(w * dt)

        Rp = expm(phi)
        Fn = np.eye(6)
        Fn[:3, 3:] = -Rp @ skew(v) * dt
        Pn = Fn @ Pn @ Fn.T + Q
        Pn = 0.5 * (Pn + Pn.T)
        pn = pn + Rp @ v * dt
        phi = phi + w * dt
        rec()

    print(f"100-step: ESKF pos err={pen[-1]:.6f}, naive pos err={pnn[-1]:.6f}")
    print(f"100-step: ESKF ori err={oe[-1]:.6f}, naive ori err={on[-1]:.6f} rad")
    print(f"100-step: Ppos trace ESKF={cpe[-1]:.6f}, naive={cpn[-1]:.6f}; "
          f"Pori trace ESKF={coe[-1]:.6f}, naive={con[-1]:.6f}")
    print(f"naive unwrapped |phi|={np.linalg.norm(phi):.6f} rad; true rotation={np.linalg.norm(w)*dt*n:.6f} rad")

    s = np.arange(n + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.2))
    ax = axs[0]
    ax.semilogy(s, oe, color="#1f77b4", ls="--", lw=1.8, label="ESKF orientation")
    ax.semilogy(s, pen, color="#1f77b4", ls="-", lw=2.0, label="ESKF position")
    ax.semilogy(s, on, color="#d62728", ls="--", lw=1.8, label="naive orientation")
    ax.semilogy(s, pnn, color="#d62728", ls="-", lw=2.0, label="naive position")
    ax.set_xlabel("prediction step")
    ax.set_ylabel("error norm")
    ax.set_title("Prediction error vs truth")
    ax.legend(loc="upper left", framealpha=0.92)
    ax = axs[1]
    ax.semilogy(s, np.sqrt(cpe), color="#1f77b4", ls="-", lw=2.0, label="ESKF position")
    ax.semilogy(s, np.sqrt(coe), color="#1f77b4", ls="--", lw=1.8, label="ESKF orientation")
    ax.semilogy(s, np.sqrt(cpn), color="#d62728", ls="-", lw=2.0, label="naive position")
    ax.semilogy(s, np.sqrt(con), color="#d62728", ls="--", lw=1.8, label="naive orientation")
    ax.set_xlabel("prediction step")
    ax.set_ylabel("sqrt(trace P_block)")
    ax.set_title("Filter-predicted 1-sigma scale")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.suptitle("ESKF vs additive-rotvec EKF: 100-step prediction", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_rel = save(fig, "kf_eskf.png")
    return {"inject_max_error": float(max(ie)), "log_exp_max_error": float(max(le)),
            "final": {"pos_err_eskf": pen[-1], "pos_err_naive": pnn[-1],
                      "ori_err_eskf": oe[-1], "ori_err_naive": on[-1],
                      "pos_trace_eskf": cpe[-1], "pos_trace_naive": cpn[-1],
                      "ori_trace_eskf": coe[-1], "ori_trace_naive": con[-1],
                      "phi_norm": float(np.linalg.norm(phi))},
            "figure": fig_rel}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    VER.mkdir(parents=True, exist_ok=True)
    out = {"one_d": one_d(), "cov_2d": two_d(), "nis_nees": nis_nees(),
           "iekf": iekf(), "eskf": eskf()}
    with (VER / "kf_results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote", VER / "kf_results.json")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
