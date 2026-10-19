"""Summarize both turn proposals recorded from the same neural trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory/'decoder_comparison.csv')
    if frame.empty:
        raise ValueError('No neural decoder samples in this run')
    phases = {}
    for name,rows in frame.groupby('phase',sort=False):
        p = rows.population_yaw_rad_s.to_numpy()
        d = rows.dna02_yaw_rad_s.to_numpy()
        valid = np.isfinite(p) & np.isfinite(d)
        active = valid & (np.abs(p)>.01) & (np.abs(d)>.01)
        phases[name] = dict(samples=len(rows),
            population_mean_rad_s=float(np.mean(p)),
            dna02_mean_rad_s=float(np.mean(d[valid])) if valid.any() else None,
            dna02_left_mean_hz=float(rows.dna02_left_hz.mean()),
            dna02_right_mean_hz=float(rows.dna02_right_hz.mean()),
            both_active_samples=int(active.sum()),
            direction_agreement=float(np.mean(np.sign(p[active])==np.sign(d[active]))) if active.any() else None,
            mean_abs_difference_rad_s=float(np.mean(np.abs(p[valid]-d[valid]))) if valid.any() else None,
            mean_vision_ms=float(rows.vision_ms.mean()),mean_brain_ms=float(rows.brain_ms.mean()))
    report = dict(vision=frame.vision.iloc[0],selected_decoder=frame.selected_decoder.iloc[0],
        comparison='Same spikes, raw commands before fixture/PID/filter. Not two independent flight outcomes.',
        sign='positive = left / counterclockwise',phases=phases)
    (directory/'decoder_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    axes[0].plot(frame.t,frame.population_yaw_rad_s,label='Population proposal',lw=1)
    axes[0].plot(frame.t,frame.dna02_yaw_rad_s,label='DNa02 proposal',lw=1)
    axes[0].plot(frame.t,frame.applied_yaw_rad_s,label='Applied (fixture + filter)',color='black',lw=1)
    axes[0].set_ylabel('Yaw rate (rad/s), + left'); axes[0].legend(loc='upper left',ncol=3)
    axes[0].axhline(0,color='gray',lw=.5)
    axes[1].plot(frame.t,frame.dna02_left_hz,label='DNa02 left')
    axes[1].plot(frame.t,frame.dna02_right_hz,label='DNa02 right')
    axes[1].set_ylabel('Trailing 50 ms rate (Hz)'); axes[1].set_xlabel('Simulation time (s)')
    axes[1].legend(loc='upper left')
    for _,rows in frame.groupby('phase',sort=False):
        for ax in axes: ax.axvline(rows.t.iloc[0],color='gray',lw=.5,alpha=.5)
    fig.suptitle(f"{report['vision']} input | applied decoder: {report['selected_decoder']} | same-spike comparison")
    fig.savefig(directory/'decoder_comparison.png',dpi=150)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
