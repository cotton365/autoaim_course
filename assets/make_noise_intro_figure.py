import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-zh')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = '/home/zh/RM/autoaim_course/assets/figures'
plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 10.5,
                     'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

mu_p, var_p = 100.0, 25.0
mu_z, var_z = 108.0, 4.0
S = var_p + var_z
K = var_p/S
mu_post = mu_p + K*(mu_z - mu_p)
var_post = (1-K)*var_p

def gauss(x, mu, var):
    return np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)

x = np.linspace(88, 118, 1000)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(x, gauss(x, mu_p, var_p), color='#8a8378', lw=2.2, label=f'prediction 100 ± 5 m (P⁻=25)')
ax.plot(x, gauss(x, mu_z, var_z), color='#2f5d7c', lw=2.2, label=f'GPS 108 ± 2 m (R=4)')
ax.plot(x, gauss(x, mu_post, var_post), color='#8b3a2e', lw=2.6,
        label=f'posterior {mu_post:.1f} ± {np.sqrt(var_post):.2f} m')
for mu, var, c in [(mu_p, var_p, '#8a8378'), (mu_z, var_z, '#2f5d7c'),
                   (mu_post, var_post, '#8b3a2e')]:
    ax.axvline(mu, color=c, lw=0.8, ls=':')
ax.set_xlabel('position [m]'); ax.set_ylabel('probability density')
ax.set_title('Navigation example: prediction + GPS observation → posterior')
ax.set_xlim(88, 118); ax.set_ylim(0, 0.24)
ax.grid(color='#eee9dd', lw=0.5); ax.legend(frameon=False, fontsize=8.2, loc='upper left')
fig.savefig(f'{FIG}/noise_intro_fusion.png'); plt.close(fig)
print('K=%.6f mu=%.6f P=%.6f sigma=%.6f NIS=%.6f' % (K, mu_post, var_post, np.sqrt(var_post), (8**2)/S))
