import os, math
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-zh')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

FIG = '/home/zh/RM/autoaim_course/assets/figures'
RNG = np.random.default_rng(20261028)
plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 10.5,
                     'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

def draw_ellipse(ax, cov, mean=(0,0), nstd=1.0, **kw):
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    ang = np.degrees(np.arctan2(V[1,0], V[0,0]))
    e = Ellipse(mean, 2*nstd*math.sqrt(w[0]), 2*nstd*math.sqrt(w[1]),
                angle=ang, fill=False, **kw)
    ax.add_patch(e)
    return w, V

def cov_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharex=True, sharey=True)
    specs = [(0.85, 'positive covariance', '#2f5d7c'),
             (-0.85, 'negative covariance', '#8b3a2e'),
             (0.0, 'near-zero covariance', '#3f6b4a')]
    for ax, (rho, title, color) in zip(axes, specs):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        pts = RNG.multivariate_normal([0, 0], cov, 180)
        ax.scatter(pts[:,0], pts[:,1], s=10, alpha=.45, color=color, edgecolors='none')
        draw_ellipse(ax, cov, nstd=1.0, color=color, lw=2)
        ax.axhline(0, color='#cfc7b8', lw=.7); ax.axvline(0, color='#cfc7b8', lw=.7)
        ax.set_title(title); ax.set_aspect('equal')
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
        ax.grid(color='#eee9dd', lw=.4)
        ax.text(0.04, 0.95, f'Cov(X,Y)={rho:+.2f}', transform=ax.transAxes,
                fontsize=9, va='top', color=color)
    axes[0].set_ylabel('Y'); axes[1].set_xlabel('X')
    fig.suptitle('Covariance sign: do X and Y move together, oppositely, or independently?', y=1.02)
    fig.savefig(f'{FIG}/cov_scatter.png'); plt.close(fig)

def cov_product():
    cov = np.array([[1.0, .75], [.75, 1.0]])
    pts = RNG.multivariate_normal([0,0], cov, 240)
    prod = pts[:,0]*pts[:,1]
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.axhline(0, color='#8a8378', lw=1); ax.axvline(0, color='#8a8378', lw=1)
    ax.scatter(pts[prod>=0,0], pts[prod>=0,1], s=16, color='#2f5d7c', alpha=.55,
               label='dx·dy ≥ 0  (Q1/Q3)')
    ax.scatter(pts[prod<0,0], pts[prod<0,1], s=16, color='#8b3a2e', alpha=.55,
               label='dx·dy < 0  (Q2/Q4)')
    draw_ellipse(ax, cov, nstd=1.0, color='#1b1b1b', lw=1.6)
    # highlight one point and its deviations
    px, py = 1.25, 1.55
    ax.plot([px,px], [0,py], color='#3f6b4a', lw=1.6, ls='--')
    ax.plot([0,px], [py,py], color='#3f6b4a', lw=1.6, ls='--')
    ax.plot(px, py, 'o', color='#3f6b4a', ms=7)
    ax.annotate('(dx, dy)', xy=(px,py), xytext=(px+.15, py+.35), fontsize=9, color='#3f6b4a')
    ax.text(0.03, 0.97, 'Q2: dx·dy < 0', transform=ax.transAxes, fontsize=9, va='top', color='#8b3a2e')
    ax.text(0.72, 0.97, 'Q1: dx·dy > 0', transform=ax.transAxes, fontsize=9, va='top', color='#2f5d7c')
    ax.text(0.03, 0.03, 'Q3: dx·dy > 0', transform=ax.transAxes, fontsize=9, va='bottom', color='#2f5d7c')
    ax.text(0.72, 0.03, 'Q4: dx·dy < 0', transform=ax.transAxes, fontsize=9, va='bottom', color='#8b3a2e')
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect('equal')
    ax.set_xlabel('X - E[X]'); ax.set_ylabel('Y - E[Y]')
    ax.set_title('Cov(X,Y) = E[(X-EX)(Y-EY)]: average of signed rectangle areas')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5)
    ax.grid(color='#eee9dd', lw=.4)
    fig.savefig(f'{FIG}/cov_product.png'); plt.close(fig)

def cov_ellipse_axes():
    # 5-point example from the lesson
    P = np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]])
    mean = P.mean(0)
    X = P - mean
    S = X.T @ X / (len(P)-1)
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(P[:,0], P[:,1], color='#2f5d7c', s=35, zorder=3, label='data')
    ax.plot(mean[0], mean[1], 'o', color='#1b1b1b', ms=7, zorder=4)
    ax.text(mean[0]+.08, mean[1]-.22, 'mean', fontsize=9)
    for ns, ls in [(1, '-'), (2, '--')]:
        draw_ellipse(ax, S, mean=mean, nstd=ns, color='#8b3a2e', lw=2 if ns==1 else 1, ls=ls)
    for i in range(2):
        v = V[:,i]; L = math.sqrt(w[i])
        ax.annotate('', xy=mean + 1.0*L*v, xytext=mean,
                    arrowprops=dict(arrowstyle='-|>', color='#3f6b4a', lw=2))
        ax.text(*(mean + 1.08*L*v), f'$\\sqrt{{\\lambda_{i+1}}}v_{i+1}$', color='#3f6b4a', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(0, 7.4); ax.set_ylim(0, 7.4)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('P = VΛVᵀ: ellipse axes are ±√λᵢ vᵢ (1σ solid, 2σ dashed)')
    ax.text(.04, .95, f'Σ = [[{S[0,0]:.1f}, {S[0,1]:.1f}],\n     [{S[1,0]:.1f}, {S[1,1]:.1f}]]\n'
                      f'λ = [{w[0]:.2f}, {w[1]:.2f}]', transform=ax.transAxes,
            va='top', fontsize=8.5, family='monospace')
    ax.grid(color='#eee9dd', lw=.4); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(f'{FIG}/cov_ellipse_axes.png'); plt.close(fig)

def cov_transform():
    A = np.array([[1.4, .9], [.1, 1.0]])
    X = RNG.normal(0, 1, (500, 2))
    Y = X @ A.T
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, pts, title, cov in [(axes[0], X, 'input: Cov(x) = I', np.eye(2)),
                                (axes[1], Y, 'output: Cov(Ax) = A Aᵀ', A @ A.T)]:
        ax.scatter(pts[:,0], pts[:,1], s=10, alpha=.35, color='#2f5d7c')
        draw_ellipse(ax, cov, nstd=1.0, color='#8b3a2e', lw=2)
        ax.axhline(0, color='#cfc7b8', lw=.7); ax.axvline(0, color='#cfc7b8', lw=.7)
        ax.set_aspect('equal'); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.grid(color='#eee9dd', lw=.4); ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
    # basis vectors and their images
    for e, c, lab in [(np.array([1,0]), '#3f6b4a', 'e1'), (np.array([0,1]), '#8b3a2e', 'e2')]:
        axes[0].annotate('', xy=e, xytext=(0,0), arrowprops=dict(arrowstyle='-|>', color=c, lw=2))
        axes[0].text(e[0]+.08, e[1]+.08, lab, color=c)
        axes[1].annotate('', xy=A@e, xytext=(0,0), arrowprops=dict(arrowstyle='-|>', color=c, lw=2))
        axes[1].text(*(A@e + .12), f'A{lab}', color=c)
    fig.suptitle('Linear transform: covariance is pushed by A, not by b', y=1.02)
    fig.savefig(f'{FIG}/cov_transform.png'); plt.close(fig)

if __name__ == '__main__':
    cov_scatter(); cov_product(); cov_ellipse_axes(); cov_transform()
    print('wrote covariance figures to', FIG)
