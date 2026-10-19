"""Selectable sensory drive and two readouts around the unchanged LIF brain.

External visual inputs obey the existing brain's refractory/synaptic rules.
This deliberately does not substitute the zero-shot repository's LIF engine.
"""
from collections import deque

import numpy as np

from fbwb.corrected import CorrectedFlyBrain


class VisualFlyBrain(CorrectedFlyBrain):
    def __init__(self, *args, decoder='population', dna02_gain=2.0, **kwargs):
        if decoder not in ('population', 'dna02'):
            raise ValueError('Unknown turn decoder')
        if not np.isfinite(dna02_gain) or dna02_gain <= 0:
            raise ValueError('DNa02 gain must be finite and positive')
        self.decoder = decoder
        self.dna02_gain = dna02_gain
        super().__init__(*args, **kwargs)
        types = self.neurons_df.primary_type.fillna('')
        sides = self.neurons_df.side.fillna('')
        self.dna02_groups = [np.flatnonzero(types.eq('DNa02') & sides.eq(s)) for s in ('left', 'right')]
        if decoder == 'dna02' and any(len(g) == 0 for g in self.dna02_groups):
            raise ValueError('DNa02 requires identified left and right DNa02 neurons')

    def reset_state(self):
        super().reset_state()
        self.external_indices = None
        self.external_probability = None
        self.dna02_history = deque(maxlen=max(1, round(0.050 / self.dt)))

    def set_visual_rates(self, indices, rates_hz):
        indices = np.asarray(indices, dtype=np.int64)
        rates = np.asarray(rates_hz, dtype=float)
        if (indices.ndim != 1 or rates.shape != indices.shape or
                len(np.unique(indices)) != len(indices) or
                np.any(indices < 0) or np.any(indices >= self.n_neurons) or
                not np.all(np.isfinite(rates)) or np.any(rates < 0)):
            raise ValueError('Invalid external visual neuron indices/rates')
        self.external_indices = indices
        self.external_probability = -np.expm1(-rates * self.dt)
        self.external_rates = rates.copy()

    def _step(self, forced=None):
        if self.external_indices is not None:
            external = self.external_indices[self.rng.random(len(self.external_indices)) < self.external_probability]
            forced = np.union1d(forced, external) if forced is not None else external
            if hasattr(self, 'external_events'):
                self.external_events += np.isin(self.external_indices,external)
        spikes = super()._step(forced)
        if self.external_indices is not None and hasattr(self,'external_counts'):
            self.external_counts += spikes[self.external_indices]
        if hasattr(self, 'dna02_groups'):
            self.dna02_history.append([float(spikes[g].mean()) if len(g) else 0.0 for g in self.dna02_groups])
        return spikes

    def compute(self, optic_flow):
        if self.external_indices is not None and np.any(optic_flow):
            raise ValueError('Use either FlyVis handoff or optical-flow receptor drive, not both')
        if self.external_indices is not None:
            self.external_counts = np.zeros(len(self.external_indices))
            self.external_events = np.zeros(len(self.external_indices),dtype=int)
        motor, metrics = super().compute(optic_flow)
        population_rc = motor['yaw']
        available = all(len(g) for g in self.dna02_groups)
        # A trailing 50 ms window, zero padded on reset, independent of control period.
        hz = np.sum(self.dna02_history, axis=0) / (self.dna02_history.maxlen * self.dt)
        left, right = map(float, hz)
        # Reference repo: asym = clip(-k*(L-R)/100, +/-.8).
        # Here the same dimensionless signal is explicitly adapted to CCW rad/s.
        dna02_ccw = float(np.clip(self.dna02_gain * (left-right) / 100, -0.8, 0.8))
        dna02_rc = -100 * dna02_ccw
        if self.decoder == 'dna02':
            motor['yaw'] = dna02_rc
        self.observation['decode'].update(
            selected_decoder=self.decoder,
            population_yaw_rc=population_rc,
            population_yaw_rad_s=-population_rc / 100,
            dna02_available=bool(available), dna02_left_hz=left, dna02_right_hz=right,
            dna02_yaw_rc=dna02_rc if available else None,
            dna02_yaw_rad_s=dna02_ccw if available else None,
            dna02_window_ms=1000*self.dna02_history.maxlen*self.dt,
            dna02_gain=self.dna02_gain,
            dna02_side_counts=[len(g) for g in self.dna02_groups],
        )
        if self.external_indices is not None:
            table = self.neurons_df.iloc[self.external_indices]
            actual = self.external_counts/(self.substeps*self.dt)
            groups, sizes, cells = [], [], []
            for side in ('left','right'):
                positions = np.flatnonzero(table.side.eq(side))
                sample = positions[np.linspace(0,len(positions)-1,min(48,len(positions))).astype(int)] if len(positions) else []
                groups.append(dict(name=f'Visual handoff {side}',size=len(positions),
                    target_hz=float(self.external_rates[positions].mean()) if len(positions) else 0,
                    observed_hz=float(actual[positions].mean()) if len(positions) else 0))
                sizes.append(len(sample))
                for j in sample:
                    cells.append(dict(id=str(table.iloc[j].get('root_id',self.external_indices[j])),
                        hz=float(actual[j]),target_hz=float(self.external_rates[j]),drive_events=int(self.external_events[j])))
            self.observation['visual_inputs'] = dict(groups=groups,sample_group_sizes=sizes,receptors=cells)
        return motor, metrics
