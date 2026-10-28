import os, math
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-zh')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG = '/home/zh/RM/autoaim_course/assets/figures'
plt.rcParams.update({
    'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 10.5,
    'figure.dpi': 160, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

def timeline():
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    rows = ['sp_vision_25', 'wuhan', 'SHtech']
    y = [2.0, 1.0, 0.0]
    colors = ['#8b3a2e', '#2f5d7c', '#3f6b4a']
    tk, now, lat, fly = 0.0, 0.020, 0.015, 0.120
    hit = now + lat + fly
    for yi, name, c in zip(y, rows, colors):
        ax.plot([tk, tk], [yi-0.16, yi+0.16], color=c, lw=3)
        ax.annotate('', xy=(now, yi), xytext=(tk, yi),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2))
        ax.text((tk+now)/2, yi-0.26, 'detect + track', ha='center', va='top', fontsize=8.2, color=c)
        ax.annotate('', xy=(hit, yi), xytext=(now, yi),
                    arrowprops=dict(arrowstyle='<|-|>', color=c, lw=1.5, ls='--'))
        ax.text((now+hit)/2, yi+0.09, 'latency + $T_{fly}$ (iterate)', ha='center', va='bottom', fontsize=8.2, color=c)
        ax.plot([hit], [yi], marker='o', ms=6, color=c)
        ax.text(hit+0.004, yi, 'hit', va='center', fontsize=9, color=c)
    ax.plot([], [], color='#8b3a2e', lw=3, label='sp: 11D EKF, 10 iters')
    ax.plot([], [], color='#2f5d7c', lw=3, label='wuhan: 13D ESKF, 1 kHz predict + 10 iters')
    ax.plot([], [], color='#3f6b4a', lw=3, label='SHtech: dual model, one-shot')
    ax.set_yticks(y); ax.set_yticklabels(rows)
    ax.set_xlim(-0.02, hit+0.03); ax.set_ylim(-0.55, 2.35)
    ax.set_xlabel('time [s]  (example: latency 15 ms, fly 120 ms)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=7.8)
    ax.grid(axis='x', color='#e6e1d6', lw=0.6)
    ax.set_title('Three prediction timelines: who extrapolates to the hit time, and when')
    fig.savefig(f'{FIG}/timeline.png'); plt.close(fig)

def armor_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    for ax, N, r, title in [(axes[0], 4, 0.26, 'N = 4 (standard / balance)'),
                            (axes[1], 3, 0.2765, 'N = 3 (outpost)')]:
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(r*np.cos(th), r*np.sin(th), color='#cfc7b8', lw=1, ls=':')
        L = 0.42 if N == 4 else 0.34
        W = 0.34 if N == 4 else 0.34
        ax.add_patch(Rectangle((-L/2, -W/2), L, W, fill=False, ec='#8a8378', lw=1.2))
        ax.plot(0, 0, 'o', color='#1b1b1b', ms=5)
        ax.text(0.018, -0.045, 'car center $c$', fontsize=8.5, va='top')
        for i in range(N):
            phi = i*2*np.pi/N
            px, py = -r*np.cos(phi), -r*np.sin(phi)
            tx, ty = -np.sin(phi), np.cos(phi)
            half = 0.0675
            ax.plot([px-half*tx, px+half*tx], [py-half*ty, py+half*ty], color='#8b3a2e', lw=4, solid_capstyle='butt')
            ax.plot([0, px], [0, py], color='#2f5d7c', lw=0.9, ls='--')
            ha = 'right' if px < -1e-6 else ('left' if px > 1e-6 else 'center')
            va = 'bottom' if py > 1e-6 else ('top' if py < -1e-6 else 'center')
            ax.text(px*1.20, py*1.20, f'$p_{i}$', color='#8b3a2e', fontsize=10, ha=ha, va=va)
        # theta arrow at 35 deg, away from labels
        ax.annotate('', xy=(0.13*math.cos(math.radians(35)), 0.13*math.sin(math.radians(35))),
                    xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='#3f6b4a', lw=1.5))
        ax.text(0.14*math.cos(math.radians(35)), 0.14*math.sin(math.radians(35)), r'$\theta$', color='#3f6b4a', fontsize=12)
        phi_txt = '   '.join([f'$\\phi_{i}={int(round(np.degrees(i*2*np.pi/N)))}^{{\\circ}}$' for i in range(N)])
        ax.text(0.02, 0.03, phi_txt, transform=ax.transAxes, fontsize=8.2, color='#6d6a64', va='bottom')
        ax.set_aspect('equal'); ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45)
        ax.set_title(title); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.grid(color='#eee9dd', lw=0.5)
    fig.suptitle(r'Armor layout: $p_i = c - r_i(\cos\phi_i,\ \sin\phi_i)$', y=1.02)
    fig.savefig(f'{FIG}/armor_geometry.png'); plt.close(fig)

def ballistic_compare():
    g, v0, d, h = 9.8, 23.0, 4.0, 0.5
    a = g*d*d/(2*v0*v0); b = -d; cc = a+h
    disc = b*b - 4*a*cc
    u1 = (-b + math.sqrt(disc))/(2*a); u2 = (-b - math.sqrt(disc))/(2*a)
    th_high, th_low = math.atan(u1), math.atan(u2)
    def vac(th):
        T = d/(v0*math.cos(th)); t = np.linspace(0, T, 300)
        return v0*math.cos(th)*t, v0*math.sin(th)*t - 0.5*g*t*t, T
    xh, zh, Th = vac(th_high); xl, zl, Tl = vac(th_low)
    k = 0.05
    A = (math.exp(k*d)-1)/k; C = A/v0; B = 0.5*g*C*C
    disc2 = A*A - 4*B*(B+h)
    Td = None
    if disc2 > 0:
        uu = [(A+math.sqrt(disc2))/(2*B), (A-math.sqrt(disc2))/(2*B)]
        uu = [u for u in uu if abs(math.atan(u)) < math.pi/2]
        thd = min(uu, key=lambda u: (math.exp(k*d)-1)/(k*v0*math.cos(math.atan(u))))
        Td = (math.exp(k*d)-1)/(k*v0*math.cos(thd))
        t = np.linspace(0, Td, 400)
        xd = np.log1p(k*v0*math.cos(thd)*t)/k
        zd = v0*math.sin(thd)*t - 0.5*g*t*t

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={'width_ratios':[1.15, 1]})
    ax.plot(xh, zh, color='#b98a80', lw=1.6, ls='--', label=f'vacuum high arc, T={Th:.2f}s')
    ax.plot(xl, zl, color='#8b3a2e', lw=2.0, label=f'vacuum low arc, T={Tl:.3f}s')
    if Td: ax.plot(xd, zd, color='#2f5d7c', lw=2.0, label=f'drag (wuhan), T={Td:.3f}s')
    ax.plot([d], [h], 'k*', ms=13, label='target (d,h)')
    ax.set_xlabel('horizontal distance d [m]'); ax.set_ylabel('height z [m]')
    ax.set_title('(a) full view'); ax.grid(color='#eee9dd', lw=0.5)
    ax.legend(frameon=False, fontsize=8.0, loc='upper right')
    ax.set_xlim(0, d*1.05); ax.set_ylim(-0.1, 28)

    ax2.plot(xl, zl, color='#8b3a2e', lw=2.2, label=f'vacuum low, T={Tl:.3f}s')
    if Td: ax2.plot(xd, zd, color='#2f5d7c', lw=2.2, label=f'drag, T={Td:.3f}s')
    ax2.plot([d], [h], 'k*', ms=13)
    ax2.set_xlim(3.2, 4.06); ax2.set_ylim(0.2, 1.0)
    ax2.set_xlabel('horizontal distance d [m]'); ax2.set_ylabel('height z [m]')
    ax2.set_title('(b) zoom near the target'); ax2.grid(color='#eee9dd', lw=0.5)
    ax2.legend(frameon=False, fontsize=8.0, loc='upper left')
    fig.suptitle('Ballistic solutions: high vs low arc, and the drag model', y=1.02)
    fig.savefig(f'{FIG}/ballistic_compare.png'); plt.close(fig)
    return Tl, Th, Td

def fixed_point():
    v0 = 23.0
    def fly_time(T, vt=2.0):
        x = 4.0 - vt*T
        return math.hypot(x, 0.5)/v0
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for vt, c in [(0.0, '#8b3a2e'), (2.0, '#2f5d7c'), (4.0, '#3f6b4a')]:
        Ts = [0.18]
        for _ in range(12): Ts.append(fly_time(Ts[-1], vt))
        ax.plot(range(len(Ts)), Ts, 'o-', ms=4, color=c, label=f'target speed {vt:.0f} m/s')
        ax.axhline(Ts[-1], color=c, lw=0.8, ls=':')
    ax.set_xlabel('iteration k'); ax.set_ylabel('$T_{fly}$ [s]')
    ax.set_title('Flight-time fixed point: $T_{k+1}=$ range$(t_{fire}+T_k)/v_0$')
    ax.grid(color='#eee9dd', lw=0.5); ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(f'{FIG}/fixed_point.png'); plt.close(fig)

def qr_gain():
    P = 1.0; R = np.logspace(-2, 2, 300); K = P/(P+R)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.semilogx(R, K, color='#2f5d7c', lw=2)
    ax.set_xlabel('R / P  (measurement noise / prior uncertainty)')
    ax.set_ylabel('Kalman gain K')
    ax.set_title('K = P / (P + R): trust the measurement only when R is small')
    ax.grid(color='#eee9dd', lw=0.5)
    for x, lab in [(0.05, 'R << P\nK→1'), (20, 'R >> P\nK→0')]:
        ax.axvline(x, color='#b98a80', lw=0.8, ls=':')
        ax.text(x*1.15, 0.5, lab, fontsize=8.5, color='#6d6a64')
    fig.savefig(f'{FIG}/qr_gain.png'); plt.close(fig)

def noise_anisotropy():
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    p = np.array([3.0, 2.0]); d = np.linalg.norm(p)
    ur = p/d; ut = np.array([-ur[1], ur[0]])
    sr, st = 0.20*d, 0.0628*d
    R = sr*sr*np.outer(ur, ur) + st*st*np.outer(ut, ut)
    w, V = np.linalg.eigh(R); order = np.argsort(w)[::-1]; w = w[order]; V = V[:, order]
    th = np.linspace(0, 2*np.pi, 300); circle = np.vstack([np.cos(th), np.sin(th)])
    ell = V @ np.diag(np.sqrt(w)) @ circle
    ax.plot(p[0]+ell[0], p[1]+ell[1], color='#2f5d7c', lw=2, label='1σ ellipse of R (SHtech)')
    ax.plot([0, p[0]], [0, p[1]], color='#8a8378', lw=1, ls='--')
    ax.arrow(p[0], p[1], ur[0]*0.5, ur[1]*0.5, color='#8b3a2e', width=0.01, label='radial $u_r$')
    ax.arrow(p[0], p[1], ut[0]*0.5, ut[1]*0.5, color='#3f6b4a', width=0.01, label='tangential $u_t$')
    ax.plot(p[0], p[1], 'ko', ms=5); ax.text(p[0]+0.1, p[1]+0.12, 'target', fontsize=9)
    ax.set_aspect('equal'); ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.2, 3.4)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Anisotropic measurement noise: long axis points at the camera')
    ax.grid(color='#eee9dd', lw=0.5); ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.savefig(f'{FIG}/noise_anisotropy.png'); plt.close(fig)

if __name__ == '__main__':
    timeline(); armor_geometry(); print('ballistic', ballistic_compare())
    fixed_point(); qr_gain(); noise_anisotropy()
    print('wrote figures to', FIG)
