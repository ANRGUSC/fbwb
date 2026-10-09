import json
import numpy as np
from fbwb.experiment import PHASES, phase_at, explain_step
from fbwb.telemetry import Publisher
from test_corrected import model


def test_observation_is_measured_and_identified():
    brain = model(weight=100)
    motor, metrics = brain.compute(np.ones(4))
    observation = brain.observation
    assert observation['spike_count'] == round(metrics['total_spikes']*20)
    assert observation['active_neurons'] <= brain.n_neurons
    assert all(r['hz']*0.020 == round(r['hz']*0.020) for r in observation['receptors'])
    assert len(observation['receptors']) == 2
    assert observation['decode']['dn_sum_per_ms'] == metrics['dn_activity']
    assert {r['id'] for r in observation['receptors']} == {'0','1'}


def test_phase_boundaries_and_fixture_ownership():
    assert phase_at(5.98).name == 'TAKEOFF'
    assert phase_at(6.0).held
    assert phase_at(6.4).name == 'LEFT SCREEN'
    assert phase_at(6.8).name == 'RIGHT SCREEN'
    assert not phase_at(8.0).held
    assert all(p.held for p in PHASES[1:6])


def test_narrative_does_not_attribute_pid_altitude_to_brain():
    state = {'t':6.5,'z':1,'vx':0}
    applied = {'height':1,'vx':0,'yaw':0}
    motor = {'yaw':50,'forward':20}
    decode = {'motion_a_sum_per_ms':10,'motion_b_sum_per_ms':5,'dn_sum_per_ms':20}
    text = explain_step(phase_at(6.5),state,[0,1,0,0],motor,applied,decode)
    assert 'shadow commands' in text and 'fixture prevents motion' in text
    assert 'Neural climb is unused' in text and 'cannot reverse' in text


def test_atomic_snapshot_matches_recorded_frame(tmp_path):
    publisher = Publisher(tmp_path,'test-run')
    publisher.publish({'state':{'t':6.4},'phase':{'name':'LEFT SCREEN'},'narrative':'test narrative'},np.zeros((20,20,4),dtype=np.uint8))
    publisher.close()
    packet = json.loads((tmp_path/'latest.json').read_text())
    assert packet == json.loads((tmp_path/'live.ndjson').read_text())
    assert packet['run_id'] == 'test-run' and packet['seq'] == 1
    assert packet['camera_png'].startswith('iVBOR')
