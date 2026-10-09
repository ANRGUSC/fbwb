"""Make a compact scientific comparison from recorded simulator telemetry."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for mode, color, label in [('upstream', '#b54c2c', 'Original / test room'), ('stable', '#266d9b', 'Integration only / test room'), ('corrected','#238554','Corrected / corridor')]:
    d = pd.read_csv(ROOT/'results'/mode/'telemetry.csv')
    initial_x = d.x.iloc[0]
    d = d[d.t >= 6]
    t = d.t-6
    axes[0,0].plot(t, d.x-initial_x, label=label, color=color)
    axes[0,1].plot(t, d.z, label=label, color=color)
    axes[1,0].plot(t, d.spikes, label=label, color=color)
    if mode == 'corrected':
        axes[1,1].plot(t, d.vx, label='Measured forward speed', color=color)
        axes[1,1].plot(t, d.vx_target, label='Filtered velocity target', color='#333333', linestyle='--')
axes[0,0].set_ylabel('Forward displacement (m)')
axes[0,1].set_ylabel('Altitude (m)')
axes[1,0].set_ylabel('Mean spikes / neural step')
axes[1,1].set_ylabel('Body forward speed (m/s)')
for ax in axes.flat:
    ax.set_xlabel('Simulation seconds after brain handoff')
    ax.grid(alpha=0.2)
axes[0,0].legend(fontsize=8)
axes[1,1].legend(fontsize=8)
fig.suptitle('Flybrain → Crazyflie integration checks\nDifferent scenes; not a controlled navigation-performance comparison', fontsize=12)
fig.savefig(ROOT/'results/comparison.png', dpi=160)
