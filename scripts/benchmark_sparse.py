"""Check CPU sparse layout speed and numerical agreement on the actual graph."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from fbwb.brain import load_brain

brain = load_brain()
torch.manual_seed(7)
x = (torch.rand(brain.n_neurons, 1) > 0.9).float()
outputs = {}
for layout in ['coo', 'csr']:
    matrix = brain.conn_T.to_sparse_coo() if layout == 'coo' else brain.conn_T.to_sparse_csr()
    start = time.monotonic()
    for _ in range(5):
        outputs[layout] = torch.sparse.mm(matrix, x)
    print(layout, (time.monotonic()-start)/5, flush=True)
error = (outputs['coo']-outputs['csr']).abs().max().item()
print('max absolute difference', error, flush=True)
torch.testing.assert_close(outputs['coo'], outputs['csr'], rtol=1e-5, atol=1e-4)
