"""Acceptance checks on the actual recorded controlled-trial / flight run."""
import json
from pathlib import Path
import sys
import numpy as np


def validate(directory):
    directory = Path(directory)
    records = [json.loads(line) for line in (directory/'live.ndjson').read_text().splitlines()]
    summary = json.loads((directory/'summary.json').read_text())
    assert summary.get('vision','optic-flow') == 'optic-flow', 'Historical acceptance checks require --vision optic-flow; use validate_visual_run.py for FlyVis'
    assert all(p['run_id'] == summary['run_id'] for p in records)
    assert [p['seq'] for p in records] == list(range(1,len(records)+1))
    active = [p for p in records if p['state']['t'] >= 6]
    assert np.allclose(np.diff([p['state']['t'] for p in active]),.020)
    grouped = {name:[p for p in active if p['phase']['name']==name] for name in dict.fromkeys(p['phase']['name'] for p in active)}
    bilateral = 'RIGHT TURN' in grouped
    assert len(grouped) == (8 if bilateral else 6)
    means = {name:np.mean([p['inputs'] for p in packets[1:]],axis=0) for name,packets in grouped.items()}
    assert np.max(means['STILL BASELINE']) == 0
    assert means['LEFT SCREEN'][1] > .8 and means['LEFT SCREEN'][2] < .1
    assert means['RIGHT SCREEN'][2] > .8 and means['RIGHT SCREEN'][1] < .1
    assert means['OUTWARD MOTION'][0] > .2
    assert means['VERTICAL MOTION'][3] > .8
    held = [p for p in active if p['phase']['held']]
    assert max(abs(p['state']['z']-1) for p in held) < .001
    assert max(abs(p['state']['x']+.8)+abs(p['state']['y']) for p in held) < .001
    assert all(p['applied']['vx'] == p['applied']['vy'] == p['applied']['yaw'] == 0 for p in held)
    assert any(p['decoded']['forward'] > 1 for p in held)
    assert all(p['brain']['spike_count'] == 0 for p in grouped['STILL BASELINE'])
    for p in active:
        brain = p['brain']
        assert brain['window_ms'] == 20
        assert np.isclose(brain['spike_count'],p['state']['spikes']*20)
        assert sum(brain['sample_group_sizes']) == len(brain['receptors']) == 144
        assert len(p['applied']['motors']) == 4
    free = grouped['FREE FLIGHT']
    assert all(.85 < p['state']['z'] < 1.15 for p in free)
    assert summary['airborne_contact_steps'] == 0
    path_length = float(np.linalg.norm(np.diff([[p['state']['x'],p['state']['y']] for p in free],axis=0),axis=1).sum())
    assert path_length > .03
    assert summary['max_abs_roll_pitch'] < .15
    turns = {}
    if bilateral:
        assert np.mean([p['decoded']['yaw'] for p in grouped['LEFT SCREEN'][-10:]]) < -20
        assert np.mean([p['decoded']['yaw'] for p in grouped['RIGHT SCREEN'][-10:]]) > 20
        assert abs(np.mean([p['decoded']['yaw'] for p in grouped['OUTWARD MOTION'][-10:]])) < 10
        assert free[-1]['state']['t']-free[0]['state']['t'] >= 6-1e-8
        for name,sign in [('LEFT TURN',1),('RIGHT TURN',-1)]:
            packets = grouped[name]
            heading = np.unwrap([p['state']['yaw'] for p in packets])
            delta = float(heading[-1]-heading[0])
            assert sign*delta > .3,(name,delta)
            assert all(p['applied']['vx'] == 0 and p['phase']['camera'] == 'stabilized test camera' for p in packets)
            assert sign*np.mean([p['applied']['yaw'] for p in packets]) > .1
            turns[name] = {'measured_turn_degrees':float(np.degrees(delta))}
        assert all(not p['phase']['held'] and not p['phase']['turn_trial'] and p['phase']['camera']=='onboard camera' for p in free)
        assert path_length > .2
    report = {'run_id':summary['run_id'],'passed':True,'recorded_frames':len(records),'mean_inputs_by_phase':{k:v.tolist() for k,v in means.items()},'turns':turns,'free_flight_path_m':path_length,'forward_progress_m':summary['forward_progress_m'],'final_height_m':summary['final_position'][2]}
    (directory/'acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    validate(sys.argv[1] if len(sys.argv)>1 else 'results/explained-bilateral')
