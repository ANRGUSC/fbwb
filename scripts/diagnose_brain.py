"""Contrast upstream versus stable integration with controlled stimuli.

The toy graph isolates numerical behavior. --full additionally tests the real
prepared connectome. Neither test claims physiological validity.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from fbwb.brain import build_brain, load_brain


def toy(mode):
    neurons = pd.DataFrame({'root_id': np.arange(16), 'primary_type': ['R1-6', 'R1-6', 'R7', 'R8'] + ['T4a']*10 + ['DNa01']*2, 'nt_type': ['ACH']*16})
    # No edges: any firing here must be driven locally, not by recurrence.
    graph = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (16, 16)).coalesce()
    return build_brain(neurons, graph, mode, period_ms=20)


def run_case(brain, stimulus, cycles):
    brain.reset_state()
    start = time.monotonic()
    outputs = []
    for _ in range(cycles):
        with torch.no_grad():
            motor, metrics = brain.compute(np.array(stimulus, dtype=np.float32))
        outputs.append({'motor': motor, **metrics})
    return {'stimulus': stimulus, 'simulated_ms': cycles*brain.substeps, 'wall_seconds': time.monotonic()-start, 'max_spikes_per_step': max(o['total_spikes'] for o in outputs), 'max_dn_activity': max(o['dn_activity'] for o in outputs), 'outputs': outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'upstream_tau_m_seconds': 10.0*2e-6, 'upstream_dt_seconds': 0.001, 'passive_euler_multiplier': 1-0.001/(10.0*2e-6), 'note': 'Stable mode changes integration only. The full graph uses official neuron ordering, which can differ from upstream private CSVs.'}
    for mode in ['upstream', 'stable']:
        brain = load_brain(mode) if args.full else toy(mode)
        cases = {}
        for name, stimulus in [('zero', [0,0,0,0]), ('uniform', [1,1,1,1]), ('left', [0,1,0,0]), ('right', [0,0,1,0])]:
            cases[name] = run_case(brain, stimulus, 5)
            print(mode, name, {k:v for k,v in cases[name].items() if k != 'outputs'}, flush=True)
        report[mode] = cases
        del brain
    directory = ROOT / 'results/diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('full.json' if args.full else 'isolated.json')
    path.write_text(json.dumps(report, indent=2))
    print(path)


if __name__ == '__main__':
    main()
