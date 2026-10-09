import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fbwb.corrected import CorrectedFlyBrain, bilateral_turn


def model(dt=1, weight=0):
    neurons = pd.DataFrame({'primary_type':['R1-6','R1-6','DNa01'],'nt_type':['ACH']*3,'side':['left','right','left']})
    graph = sp.csr_matrix(([weight],([0],[2])),shape=(3,3))
    return CorrectedFlyBrain(neurons, graph, dt_ms=dt)


def test_passive_relaxation_matches_20ms_analytic_solution_at_multiple_steps():
    for dt in [1,0.5,0.2]:
        brain = model(dt)
        brain.v[2] += 0.002
        for _ in range(round(20/dt)):
            brain._step()
        assert brain.v[2] == pytest.approx(-0.052 + 0.002*np.exp(-1), abs=1e-12)


def test_synaptic_decay_and_membrane_response_are_timestep_invariant():
    values = []
    for dt in [1,0.5,0.2]:
        brain = model(dt)
        brain.g[2] = 0.001
        for _ in range(round(10/dt)):
            brain._step()
        values.append(brain.v[2])
        assert brain.g[2] == pytest.approx(0.001*np.exp(-2), abs=1e-12)
    assert np.ptp(values) < 1e-12


def test_synapse_sign_and_delay_are_respected():
    for sign in [-1,1]:
        brain = model(weight=sign*10)
        brain._step(np.array([0]))  # receptor spike at t=1ms
        brain._step()
        brain._step()
        assert brain.g[2] == 0  # arrival is at t=3ms, next integration interval
        brain._step()
        assert np.sign(brain.g[2]) == sign
        assert np.sign(brain.v[2]-brain.V_rest) == sign


def test_reset_zero_stimulus_and_seeded_input():
    brain = model(weight=100)
    _, zero = brain.compute(np.zeros(4))
    assert zero['total_spikes'] == 0
    motor, active = brain.compute(np.ones(4))
    assert active['total_spikes'] > 0
    brain.reset_state()
    brain.compute(np.zeros(4))
    assert brain.compute(np.ones(4)) == (motor,active)


def test_refractory_neuron_holds_reset_despite_input():
    brain = model()
    brain.v[2] = -0.040
    brain._step()
    assert brain.spikes[2]
    brain.g[2] = 1
    for _ in range(brain.refractory_steps):
        brain._step()
        assert not brain.spikes[2]
        assert brain.v[2] == brain.V_rest


def test_invalid_control_period_is_rejected():
    with pytest.raises(ValueError):
        model(dt=3)


def test_bilateral_decoder_is_symmetric_and_silent_without_spikes():
    assert bilateral_turn(0,0) == 0
    assert bilateral_turn(1,1) == 0
    assert bilateral_turn(1,.5) == -bilateral_turn(.5,1)
    assert bilateral_turn(1,.5) < 0  # Tello negative -> Webots left


def test_decoder_uses_anatomy_and_mean_rates_not_list_halves_or_pool_sizes():
    neurons = pd.DataFrame({'primary_type':['R1-6','T4a','T4a','DNa01','R1-6','T4a'],
                            'side':['right','left','right','left','left','left']})
    brain = CorrectedFlyBrain(neurons,sp.csr_matrix((6,6)))
    assert brain.receptor_groups[0].tolist() == [4]
    assert brain.receptor_groups[1].tolist() == [0]
    # All three motion neurons have identical activity, despite 2:1 pool size.
    brain._step = lambda forced: np.array([0,1,1,0,0,1],dtype=bool)
    motor,_ = brain.compute(np.zeros(4))
    assert motor['yaw'] == 0
    # Silence the right motion neuron; direction now follows measured spikes.
    brain._step = lambda forced: np.array([0,1,0,0,0,1],dtype=bool)
    assert brain.compute(np.zeros(4))[0]['yaw'] < 0
