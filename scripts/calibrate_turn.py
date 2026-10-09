"""Balance anatomical motion-pool gains using only a symmetric visual stimulus.

Unilateral stimuli are held-out validation, never used to fit these gains.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fbwb.brain import load_brain

brain = load_brain('corrected',calibrating=True)
brain.turn_reference_hz[:] = 1
brain.reset_state()
rates = []
for _ in range(30):
    brain.compute(np.ones(4))
    d = brain.observation['decode']
    rates.append([d['motion_left_hz'],d['motion_right_hz']])
reference = np.mean(rates[10:],axis=0)
assert np.all(reference > 1)
report = {'method':'anatomical side means / symmetric-stimulus reference; 200 ms warmup + 400 ms fit','stimulus':[1,1,1,1],'seed':brain.seed,'reference_hz':reference.tolist(),'motion_side_counts':[len(g) for g in brain.motion_groups],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'fbwb/corrected.py',ROOT/'data/raw/classification.csv.gz',ROOT/'data/prepared/neurons.csv']},'validation':{}}
brain.turn_reference_hz = reference
path = ROOT/'data/prepared/turn_calibration.json'
path.write_text(json.dumps(report,indent=2))
print('Reference Hz:',reference,flush=True)
for seed in [7,19]:
    for name,flow in [('left',[.2,1,.01,.98]),('right',[.2,.01,1,.98]),('symmetric',[1,1,1,1]),('zero',[0,0,0,0])]:
        brain.seed = seed
        brain.reset_state()
        rows = []
        for i in range(20):
            motor,_ = brain.compute(np.array(flow))
            rows.append(motor['yaw'])
        mean = float(np.mean(rows[10:]))
        report['validation'][f'{name}-seed-{seed}'] = {'mean_tello_yaw':mean,'samples':rows}
        path.write_text(json.dumps(report,indent=2))
        print(name,seed,mean,flush=True)
        if name == 'left': assert mean < -20
        if name == 'right': assert mean > 20
        if name == 'symmetric': assert abs(mean) < 10
        if name == 'zero': assert mean == 0
report['passed'] = True
path.write_text(json.dumps(report,indent=2))
