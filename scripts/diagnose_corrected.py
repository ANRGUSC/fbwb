"""Measure responsiveness and recovery of corrected dynamics on the full graph."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from fbwb.brain import load_brain

brain = load_brain('corrected')
report = {'mode':'corrected', 'tau_membrane_ms':20, 'tau_synapse_ms':5, 'dt_ms':1, 'effective_delay_ms':2, 'effective_refractory_ms':3, 'sensory_max_hz':150, 'seed':7, 'cases':{}}
for name, stimulus in [('zero',[0,0,0,0]),('uniform',[1,1,1,1]),('left',[0,1,0,0]),('right',[0,0,1,0])]:
    brain.reset_state()
    records = []
    start = time.monotonic()
    for cycle in range(25):
        # 200 ms stimulus followed by 300 ms recovery.
        motor, metrics = brain.compute(np.array(stimulus if cycle<10 else [0,0,0,0],dtype=np.float32))
        records.append({'time_ms':(cycle+1)*20,'input_on':cycle<10, **motor, **metrics})
    summary = {'max_spikes_per_ms':max(r['total_spikes'] for r in records),'max_dn_activity':max(r['dn_activity'] for r in records),'max_forward':max(r['forward'] for r in records),'final_spikes_per_ms':records[-1]['total_spikes'],'wall_seconds':time.monotonic()-start,'records':records}
    report['cases'][name] = summary
    print(name,{k:v for k,v in summary.items() if k!='records'},flush=True)
    (ROOT/'results/diagnostics/corrected.json').write_text(json.dumps(report,indent=2))
assert report['cases']['zero']['max_spikes_per_ms']==0
assert report['cases']['uniform']['max_spikes_per_ms']>0
assert report['cases']['uniform']['max_dn_activity']>0
