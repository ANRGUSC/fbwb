import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pytest
import torch
from fbwb.brain import build_brain
from fbwb.commands import map_commands, CommandFilter
from fbwb.vision import OpticFlow


def isolated(mode):
    neurons = pd.DataFrame({'primary_type': ['R1-6','R1-6','DNa01'], 'nt_type': ['ACH']*3})
    graph = torch.sparse_coo_tensor(torch.empty((2,0),dtype=torch.long), torch.empty(0), (3,3)).coalesce()
    return build_brain(neurons, graph, mode)


def test_stable_passive_membrane_converges_without_spurious_spikes():
    brain = isolated('stable')
    brain.v[:] = brain.V_rest + 0.001
    brain._step(torch.zeros(3))
    assert torch.allclose(brain.v, torch.full((3,), brain.V_rest), atol=1e-7)
    assert not brain.spikes.any()


def test_upstream_passive_error_is_amplified():
    brain = isolated('upstream')
    brain.v[:] = brain.V_rest + 0.00001
    before = abs(float(brain.v[0])-brain.V_rest)
    brain._step(torch.zeros(3))
    after = abs(float(brain.v[0])-brain.V_rest)
    assert after > 40*before


def test_neural_time_matches_webots_period():
    brain = isolated('upstream')
    assert brain.dt*brain.substeps == pytest.approx(0.020)


def test_command_axes_limits_and_invalid_input():
    assert map_commands({'forward':100,'yaw':100}) == (0.3,0,-1)
    assert map_commands({'forward':-200,'yaw':-200}) == (-0.3,0,1)
    with pytest.raises(ValueError):
        map_commands({'forward':float('nan'),'yaw':0})


def test_static_camera_has_no_flow_and_moving_texture_has_flow():
    rng = np.random.default_rng(12)
    frame = rng.integers(0, 256, (120,160,4), dtype=np.uint8)
    frame[:,:,3] = 255
    vision = OpticFlow()
    assert np.array_equal(vision.compute(frame), np.zeros(4))
    still = vision.compute(frame)
    assert np.max(still) < 0.05
    moved = vision.compute(np.roll(frame, 4, axis=1))
    assert moved[1] > 0.5 and moved[2] > 0.5
    assert np.isfinite(moved).all() and (moved >= 0).all() and (moved <= 1).all()


def test_synapse_ids_reconstruct_without_float_rounding():
    spec = importlib.util.spec_from_file_location('prepare_data', ROOT/'scripts/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    restored = module.restore_ids(pd.Series([610757204, 620797269], dtype='int64'))
    assert restored.tolist() == [720575940610757204, 720575940620797269]


def test_filter_limits_acceleration_and_attenuates_window_jitter():
    filter_ = CommandFilter()
    outputs = [filter_.update(0.1, (-1)**i, 0.02) for i in range(100)]
    vx, yaw = np.array(outputs).T
    assert np.max(np.abs(np.diff(np.r_[0,vx]))) <= 0.006001
    assert np.max(np.abs(np.diff(np.r_[0,yaw]))) <= 0.020001
    assert vx[-1] == pytest.approx(0.1, abs=0.001)
    assert np.max(np.abs(yaw)) < 0.1
