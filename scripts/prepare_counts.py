"""Prepare count-based synaptic increments for unit-consistent LIF dynamics.

All source synapses retained. GABA-negative sign convention is inherited from
flybrain. This is not the reference paper's neurotransmitter model or dataset.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT/'data/prepared'
rows = np.memmap(directory/'synapses.bin', mode='r', dtype=[('pre','<i4'),('post','<i4'),('size','<f4')])
neurons = pd.read_csv(directory/'neurons.csv')
values = np.ones(len(rows), dtype=np.float32)
inhibitory = neurons.nt_type.eq('GABA').to_numpy()
for start in range(0,len(rows),1_000_000):
    stop = min(len(rows),start+1_000_000)
    values[start:stop] = np.where(inhibitory[rows['pre'][start:stop]],-1,1)
matrix = sp.coo_matrix((values,(rows['pre'],rows['post'])),shape=(len(neurons),len(neurons))).tocsr()
sp.save_npz(directory/'counts.npz',matrix)
report = {'source_synapses':len(rows),'directed_pairs':matrix.nnz,'weight':'signed number of synapses; GABA negative, others positive, inherited from flybrain','note':'Corrected mode uses 0.275 mV increments per signed synapse; no size normalization.'}
(directory/'counts_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
