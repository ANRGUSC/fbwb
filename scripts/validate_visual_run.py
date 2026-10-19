"""Integration checks for recorded visual-input/decoder configurations."""
import argparse
import json
from pathlib import Path
import numpy as np


def validate(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    records = [json.loads(line) for line in (directory/'live.ndjson').read_text().splitlines()]
    active = [p for p in records if p.get('brain')]
    assert active, 'No neural observations recorded'
    assert all(p['run_id']==summary['run_id'] for p in records)
    assert np.allclose(np.diff([p['state']['t'] for p in active]),.02)
    for p in active:
        state,brain = p['state'],p['brain']
        d = brain['decode']
        assert d['selected_decoder']==summary['selected_decoder']
        assert np.all(np.isfinite(list(state.values())))
        selected = d[f"{d['selected_decoder']}_yaw_rad_s"]
        assert selected is not None and np.isclose(state['yaw_raw'],selected)
        assert np.isclose(brain['spike_count'],state['spikes']*20)
        if p['phase']['held']:
            assert p['applied']['yaw']==p['applied']['vx']==0
        if p['phase']['turn_trial']:
            assert p['applied']['vx']==0
        if summary['vision']=='flyvis':
            assert p['visual']['device']=='cpu'
            assert p['visual']['handoff_neurons']>30000
            assert sum(g['size'] for g in brain['visual_inputs']['groups'])==p['visual']['handoff_neurons']
            assert max(r['target_hz'] for r in brain['receptors'])==0
    if summary['vision']=='flyvis':
        assert any(p['visual']['max_drive_hz']>1 for p in active)
        assert any(g['observed_hz']>0 for p in active for g in p['brain']['visual_inputs']['groups'])
    report = dict(passed=True,vision=summary['vision'],decoder=summary['selected_decoder'],
                  neural_frames=len(active),phases=list(dict.fromkeys(p['phase']['name'] for p in active)),
                  free_flight_observed=any(p['phase']['name']=='FREE FLIGHT' for p in active),
                  interpretation='Signal-path integrity only; not biological or navigation-performance validation')
    (directory/'visual_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    validate(parser.parse_args().directory)
