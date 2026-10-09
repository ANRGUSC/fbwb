"""Load only the upstream neural class, never its Tello or plotting entry points."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = '97e02fb7c8d19fd5267c3fa9194f8001e08fb613'


def upstream_class():
    source_path = ROOT / 'vendor/flybrain/flybrain_tello_real_brain.py'
    source = source_path.read_text(encoding='utf-8')
    provenance = json.loads((ROOT / 'upstream.json').read_text())
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != provenance['source_sha256']:
        raise RuntimeError('Upstream source changed: review it and update upstream.json before running.')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RealFlyBrain')
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace['RealFlyBrain']


RealFlyBrain = upstream_class()


class StableFlyBrain(RealFlyBrain):
    """Numerical diagnostic: exponential membrane update, all upstream parameters unchanged.

    This fixes passive Euler instability, NOT physiological units, sensory gain,
    or the arbitrary bilateral decoder. It is not a validated biological model.
    """

    def _step(self, i_input):
        self.in_refractory = torch.clamp(self.in_refractory - self.dt, min=0.0)
        incoming = torch.sparse.mm(self.conn_T, self.spikes.float().unsqueeze(1)).squeeze(-1)
        self.i_syn = self.i_syn * self._syn_decay + incoming * self.W_syn / self.tau_syn
        equilibrium = self.V_rest + (i_input + self.i_syn) * self.R_m
        decay = float(np.exp(-self.dt / (self.R_m * self.C_m)))
        self.v = torch.clamp(equilibrium + (self.v - equilibrium) * decay, -0.1, 0.05)
        self.spikes = (self.v > self.V_thresh) & (self.in_refractory <= 0.0)
        self.v[self.spikes] = self.V_rest
        self.in_refractory[self.spikes] = self.tau_ref


def build_brain(neurons, connectivity, mode='upstream', period_ms=20):
    if mode not in ('upstream', 'stable'):
        raise ValueError(f'Unknown brain mode: {mode}')
    if period_ms < 1 or int(period_ms) != period_ms:
        raise ValueError('Brain period must be a positive integer number of milliseconds.')
    types = neurons.primary_type.fillna('')
    pr = set(neurons.index[types.isin(['R1-6', 'R7', 'R8'])])
    motion = set(neurons.index[types.str.contains(r'^(?:T4|T5|Tm|Mi|Lo)', regex=True)])
    dn = set(neurons.index[types.str.startswith('DN')])
    if not pr or not dn:
        raise ValueError('Connectome needs photoreceptors and descending neurons.')
    cls = RealFlyBrain if mode == 'upstream' else StableFlyBrain
    return cls(connectivity, neurons, pr, dn, motion, len(neurons), torch.device('cpu'), substeps=int(period_ms))


def load_brain(mode='upstream', period_ms=20, calibrating=False):
    """Load all prepared neurons/edges without subsampling; preserve source row order."""
    directory = ROOT / 'data/prepared'
    if not (directory / 'connectivity.npz').exists():
        raise FileNotFoundError('Missing connectome. Run scripts/fetch_assets.py --data then scripts/prepare_data.py.')
    torch.set_num_threads(4)
    neurons = pd.read_csv(directory / 'neurons.csv', dtype={'root_id': 'int64'})
    if mode == 'corrected':
        from fbwb.corrected import CorrectedFlyBrain
        annotations = pd.read_csv(ROOT/'data/raw/classification.csv.gz',usecols=['root_id','side'],dtype={'root_id':'int64'})
        # Join by exact ID without changing the connectivity matrix's row order.
        root_order = neurons.root_id.to_numpy().copy()
        neurons = neurons.drop(columns=['side'],errors='ignore').merge(annotations,on='root_id',how='left',sort=False,validate='one_to_one')
        if not np.array_equal(root_order,neurons.root_id.to_numpy()):
            raise ValueError('Annotation join changed connectivity row order')
        if not (directory / 'counts.npz').exists():
            raise FileNotFoundError('Run scripts/prepare_counts.py for corrected-mode weights.')
        calibration = directory/'turn_calibration.json'
        reference = [1.0,1.0]
        if not calibrating:
            if not calibration.exists():
                raise FileNotFoundError('Run scripts/calibrate_turn.py before corrected-mode flight.')
            provenance = json.loads(calibration.read_text())
            if not provenance.get('passed'):
                raise ValueError('Turn calibration has not passed validation; rerun scripts/calibrate_turn.py.')
            for name,digest in provenance['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Turn calibration is stale for {name}; rerun scripts/calibrate_turn.py.')
            reference = provenance['reference_hz']
        return CorrectedFlyBrain(neurons, sp.load_npz(directory / 'counts.npz'), period_ms=period_ms,turn_reference_hz=reference)
    matrix = sp.load_npz(directory / 'connectivity.npz').tocoo()
    indices = torch.from_numpy(np.stack([matrix.row, matrix.col]).astype(np.int64))
    connectivity = torch.sparse_coo_tensor(indices, torch.from_numpy(matrix.data), matrix.shape).coalesce()
    brain = build_brain(neurons, connectivity, mode, period_ms)
    # Same matrix, substantially faster on this CPU. Verified against COO.
    brain.conn_T = brain.conn_T.to_sparse_csr()
    return brain
