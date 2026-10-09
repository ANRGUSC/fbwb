"""Unit-consistent voltage-based LIF dynamics for the Flybrain adapter.

Uses the 20 ms membrane / 5 ms synaptic equations and 0.275 mV synaptic
increments documented in philshiu/Drosophila_brain_model/model.py. Camera
features -> Poisson receptor activation and the drone decoder remain engineering
interfaces, not a reproduction of the reference paper's experiments.
"""
import math
import numpy as np


def bilateral_turn(left_hz, right_hz):
    """Engineering readout of measured motion-cell rates, not a motor-circuit claim.

    Normalize per cell before comparing sides. Positive Tello yaw is right.
    A common-mode response cancels; no camera features enter this decoder.
    """
    return 2.0*(right_hz-left_hz)/max(left_hz+right_hz,1.0)


class CorrectedFlyBrain:
    def __init__(self, neurons, matrix, period_ms=20, dt_ms=1, seed=7, turn_reference_hz=(1.0,1.0)):
        if not all(math.isfinite(x) and x > 0 for x in (period_ms, dt_ms)):
            raise ValueError('Neural and control timesteps must be positive and finite')
        self.neurons_df = neurons
        self.n_neurons = len(neurons)
        self.conn_T = matrix.T.tocsr()
        self.dt = dt_ms / 1000
        self.turn_reference_hz = np.asarray(turn_reference_hz,dtype=float)
        if self.turn_reference_hz.shape != (2,) or not np.all(np.isfinite(self.turn_reference_hz)) or np.any(self.turn_reference_hz <= 0):
            raise ValueError('Turn reference must contain two positive finite rates')
        self.substeps = round(period_ms / dt_ms)
        if self.substeps < 1 or not math.isclose(self.substeps*self.dt, period_ms/1000):
            raise ValueError('Control period must be an integer multiple of neural dt')
        self.V_rest, self.V_thresh = -0.052, -0.045
        self.tau_m, self.tau_syn = 0.020, 0.005
        self.w_syn = 0.000275  # VOLTS per signed synapse, never treated as amperes.
        self.tau_ref = 0.0022
        # Grid-quantized event times: 1.8 ms -> 2 ms at the default 1 ms dt.
        self.delay_steps = max(1, round(0.0018/self.dt))
        self.refractory_steps = math.ceil(self.tau_ref/self.dt)
        types = neurons.primary_type.fillna('')
        if 'side' not in neurons:
            raise ValueError('Corrected mode requires anatomical side annotations')
        sides = neurons.side.fillna('')
        receptor_left = np.flatnonzero(types.eq('R1-6') & sides.eq('left'))
        receptor_right = np.flatnonzero(types.eq('R1-6') & sides.eq('right'))
        self.r16_idx = np.concatenate([receptor_left,receptor_right])
        self.r78_idx = np.flatnonzero(types.isin(['R7', 'R8']))
        self.pr_idx = np.concatenate([self.r16_idx, self.r78_idx])
        self.dn_indices = np.flatnonzero(types.str.startswith('DN'))
        self.motion_indices = np.flatnonzero(types.str.contains(r'^(?:T4|T5|Tm|Mi|Lo)', regex=True))
        self.motion_groups = [np.intersect1d(self.motion_indices,np.flatnonzero(sides.eq(side))) for side in ['left','right']]
        self.receptor_groups = [receptor_left, receptor_right, self.r78_idx]
        self.sample_indices = np.concatenate([g[np.linspace(0,len(g)-1,min(48,len(g))).astype(int)] for g in self.receptor_groups if len(g)])
        self.sample_group_sizes = [min(48,len(g)) for g in self.receptor_groups]
        if not len(self.pr_idx) or not len(self.dn_indices):
            raise ValueError('Missing receptor/descending annotations')
        self.seed = seed
        self.a = math.exp(-self.dt/self.tau_m)
        self.b = math.exp(-self.dt/self.tau_syn)
        self.c = self.tau_syn/(self.tau_syn-self.tau_m)*(self.b-self.a)
        self.reset_state()

    def reset_state(self):
        self.v = np.full(self.n_neurons, self.V_rest, dtype=np.float64)
        self.g = np.zeros(self.n_neurons, dtype=np.float64)
        self.refractory = np.zeros(self.n_neurons, dtype=np.int32)
        # A spike is generated at the END of a step. One extra queue slot puts
        # its arrival at a boundary delay_steps later, not one step too early.
        self.delay = [np.zeros(self.n_neurons, dtype=np.float32) for _ in range(self.delay_steps+1)]
        self.delay_cursor = 0
        self.rng = np.random.default_rng(self.seed)
        self.spikes = np.zeros(self.n_neurons, dtype=bool)
        self.observation = None

    def _step(self, forced=None):
        due = self.delay[self.delay_cursor]
        incoming = self.conn_T @ due if np.any(due) else np.zeros(self.n_neurons)
        self.g += np.asarray(incoming).ravel()*self.w_syn
        available = self.refractory == 0
        # Exact solution of the coupled linear v/g system between spike events.
        # Refractory neurons hold v at reset and freeze g, matching the reference
        # '(unless refractory)' equations; received synaptic events still accumulate.
        self.v[available] = self.V_rest + (self.v[available]-self.V_rest)*self.a + self.g[available]*self.c
        self.g[available] *= self.b
        self.refractory = np.maximum(self.refractory-1, 0)
        if forced is not None:
            # Poisson receptor activation: one strong depolarizing event, as in
            # the reference input interface. Receptors have no refractory hold.
            self.v[forced] += self.w_syn*250
            available[forced] = True
        self.spikes = available & (self.v > self.V_thresh)
        self.v[self.spikes] = self.V_rest
        self.g[self.spikes] = 0
        self.refractory[self.spikes] = self.refractory_steps
        self.refractory[self.pr_idx] = 0
        self.delay[self.delay_cursor] = self.spikes.astype(np.float32)
        self.delay_cursor = (self.delay_cursor+1) % len(self.delay)
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():
            raise FloatingPointError('Non-finite neural state')
        return self.spikes

    def compute(self, optic_flow):
        flow = np.asarray(optic_flow, dtype=np.float64)
        if flow.shape != (4,) or not np.isfinite(flow).all():
            raise ValueError('Expected four finite optic-flow values')
        flow = np.clip(flow, 0, 1)
        rates = np.zeros(len(self.pr_idx))
        split = len(self.receptor_groups[0])
        # Image halves drive anatomically labelled receptor sides; this remains
        # a coarse camera interface, not an ommatidial visual-field mapping.
        rates[:split] = 150*np.maximum(flow[0], flow[1])
        rates[split:len(self.r16_idx)] = 150*np.maximum(flow[0], flow[2])
        rates[len(self.r16_idx):] = 150*flow[3]
        probability = -np.expm1(-rates*self.dt)
        accum = np.zeros(self.n_neurons)
        forced_counts = np.zeros(self.n_neurons, dtype=np.int32)
        for _ in range(self.substeps):
            forced = self.pr_idx[self.rng.random(len(self.pr_idx)) < probability]
            forced_counts[forced] += 1
            accum += self._step(forced)
        # Express readout in spikes per 1 ms, making it independent of neural dt.
        rate = accum/(self.substeps*self.dt)*0.001
        dn = rate[self.dn_indices]
        motion = rate[self.motion_indices]
        forward_drive = dn.sum()/(len(dn)/100)
        motion_hz = [float(rate[g].mean()*1000) if len(g) else 0.0 for g in self.motion_groups]
        relative_motion = np.asarray(motion_hz)/self.turn_reference_hz
        turn = bilateral_turn(*relative_motion)
        climb = dn.sum()/(len(dn)/50)-1
        motor = {'forward': float(np.tanh(forward_drive/100)*100), 'yaw': float(np.tanh(turn)*100), 'vertical': float(np.clip(np.tanh(climb/50)*100,-80,80)), 'roll':0}
        metrics = {'total_spikes': float(rate.sum()), 'dn_activity': float(dn.sum()), 'motion_activity': float(motion.sum()), 'voltage_min': float(self.v.min()), 'voltage_max': float(self.v.max())}
        window_s = self.substeps*self.dt
        receptor_target = np.zeros(self.n_neurons)
        receptor_target[self.pr_idx] = rates
        group_names = ['R1-6 anatomical left','R1-6 anatomical right','R7/R8 (vertical-input proxy)']
        self.observation = {
            'window_ms':window_s*1000,
            'spike_count':int(accum.sum()),
            'active_neurons':int(np.count_nonzero(accum)),
            'population_hz':{'receptors':float(accum[self.pr_idx].mean()/window_s), 'motion':float(accum[self.motion_indices].mean()/window_s) if len(self.motion_indices) else 0.0, 'descending':float(accum[self.dn_indices].mean()/window_s)},
            'groups':[{'name':name,'size':len(g),'target_hz':float(receptor_target[g].mean()) if len(g) else 0,'observed_hz':float(accum[g].mean()/window_s) if len(g) else 0} for name,g in zip(group_names,self.receptor_groups)],
            'sample_group_sizes':self.sample_group_sizes,
            'receptors':[{'id':str(self.neurons_df.iloc[i]['root_id']) if 'root_id' in self.neurons_df else str(i),'index':int(i),'hz':float(accum[i]/window_s),'drive_events':int(forced_counts[i]),'target_hz':float(receptor_target[i])} for i in self.sample_indices],
            'decode':{'decoder':'anatomical-balanced-rate-v1','dn_count':len(dn),'dn_sum_per_ms':float(dn.sum()),'motion_left_hz':motion_hz[0],'motion_right_hz':motion_hz[1],'motion_left_relative':float(relative_motion[0]),'motion_right_relative':float(relative_motion[1]),'turn_reference_hz':self.turn_reference_hz.tolist(),'motion_side_counts':[len(g) for g in self.motion_groups],'turn_signal':float(turn),'forward_drive':float(forward_drive),'climb_drive':float(climb)},
            'voltage_min':float(self.v.min()),'voltage_max':float(self.v.max()),
        }
        return motor, metrics
