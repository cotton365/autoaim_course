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
dt = np.linspace(0.01, 0.30, 300)
sa2 = 1.0
qpp = sa2*dt**4/4
qpv = sa2*dt**3/2
qvv = sa2*dt**2
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.semilogy(dt, qpp, color='#2f5d7c', lw=2.2, label=r'$Q_{pp}=\sigma_a^2\,dt^4/4$')
ax.semilogy(dt, qpv, color='#8b3a2e', lw=2.2, label=r'$Q_{pv}=\sigma_a^2\,dt^3/2$')
ax.semilogy(dt, qvv, color='#3f6b4a', lw=2.2, label=r'$Q_{vv}=\sigma_a^2\,dt^2$')
i = np.argmin(np.abs(dt-0.1))
for y, c in [(qpp[i], '#2f5d7c'), (qpv[i], '#8b3a2e'), (qvv[i], '#3f6b4a')]:
    ax.plot(0.1, y, 'o', color=c, ms=5)
ax.axvline(0.1, color='#b9b1a3', lw=0.8, ls=':')
ax.text(0.103, qpp[i]*1.6, 'dt=0.1', fontsize=8.5, color='#6d6a64')
ax.set_xlabel('dt [s]'); ax.set_ylabel(r'$Q$ entry (with $\sigma_a^2=1$)')
ax.set_title('Process-noise entries grow at different rates with dt')
ax.grid(True, which='both', color='#eee9dd', lw=0.5)
ax.legend(frameon=False, fontsize=8.8, loc='lower right')
fig.savefig(f'{FIG}/q_terms.png')
print('wrote', f'{FIG}/q_terms.png')
