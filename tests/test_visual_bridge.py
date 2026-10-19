import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fbwb.corrected import CorrectedFlyBrain
from fbwb.visual_brain import VisualFlyBrain
from fbwb.flyvis_input import remap_root_ids, evoked_rates, sample_cube, CUBE_AXES, prepare_flyvis_world


def models():
    neurons = pd.DataFrame({'primary_type':['R1-6','R1-6','T4a','T4a','DNa02','DNa02'],
                            'side':['left','right','left','right','left','right']})
    graph = sp.csr_matrix(([30,30,20,20],([0,1,2,3],[2,3,4,5])),shape=(6,6))
    return neurons, graph


def test_legacy_adapter_preserves_spikes_voltages_rng_and_commands():
    n,g = models()
    old = CorrectedFlyBrain(n,g)
    new = VisualFlyBrain(n,g)
    for flow in [np.zeros(4),np.array([.2,1,0,.3]),np.ones(4)]*3:
        assert old.compute(flow) == new.compute(flow)
        np.testing.assert_array_equal(old.v,new.v)
        np.testing.assert_array_equal(old.g,new.g)
        np.testing.assert_array_equal(old.spikes,new.spikes)


def test_dna02_reads_50ms_spikes_and_selection_does_not_change_brain():
    n,g = models()
    pop = VisualFlyBrain(n,g)
    dn = VisualFlyBrain(n,g,decoder='dna02')
    for brain in (pop,dn):
        brain.set_visual_rates([4],[150])
    for _ in range(4):
        pm,_ = pop.compute(np.zeros(4)); dm,_ = dn.compute(np.zeros(4))
        np.testing.assert_array_equal(pop.v,dn.v)
        np.testing.assert_array_equal(pop.spikes,dn.spikes)
        assert pm['forward'] == dm['forward']
    d = dn.observation['decode']
    assert d['dna02_left_hz'] > 0 and d['dna02_right_hz'] == 0
    assert dm['yaw'] < 0  # negative RC -> CCW Webots yaw
    assert d['dna02_yaw_rad_s'] <= .8
    dn.reset_state()
    assert not dn.dna02_history and dn.external_indices is None


def test_handoff_cannot_mix_with_receptor_drive():
    n,g = models(); brain=VisualFlyBrain(n,g)
    brain.set_visual_rates([2],[100])
    with pytest.raises(ValueError): brain.compute(np.ones(4))
    with pytest.raises(ValueError): brain.set_visual_rates([2,2],[1,2])
    with pytest.raises(ValueError): brain.set_visual_rates([2],[np.nan])


def test_root_ids_are_remapped_and_missing_neurons_fail():
    assert remap_root_ids([99,42],[42,5,99]).tolist() == [2,0]
    with pytest.raises(ValueError): remap_root_ids([99],[42])
    with pytest.raises(ValueError): remap_root_ids([42,42],[42])


def test_handoff_baseline_rectification_and_saturation():
    np.testing.assert_allclose(evoked_rates(np.array([0,.5,1,3]),.5,np.ones(4)),[0,0,75,150])


def test_cube_axes_and_projection_preserve_side_and_elevation():
    frames=[]
    for i in range(6):
        frame=np.full((96,96,4),i*40,dtype=np.uint8); frame[:,:,3]=255
        frames.append(frame)
    np.testing.assert_allclose(sample_cube(frames,CUBE_AXES[:,0]),np.arange(6)*40/255,atol=1e-6)
    # Forward face gets brighter towards the image left (positive robot Y).
    ramp=np.tile(np.linspace(255,0,96,dtype=np.uint8),(96,1))
    frames[0]=np.stack([ramp,ramp,ramp,np.full_like(ramp,255)],axis=-1)
    sampled=sample_cube(frames,np.array([[1,.5,0],[1,-.5,0]]))
    assert sampled[0] > sampled[1]


def test_generated_world_keeps_original_and_probe_camera(tmp_path):
    original = 'DEF DRONE Crazyflie { extensionSlot [ DEF PROBE_CAMERA_POSE Pose { children [ Camera { name "probe_camera" } ] } ] }'
    source=tmp_path/'explain.wbt'; source.write_text(original)
    generated=prepare_flyvis_world(source)
    text=generated.read_text()
    assert source.read_text()==original
    assert text.count('extensionSlot [')==1
    assert text.count('name "flyvis_')==6
    assert 'name "probe_camera"' in text
    assert text.count('{')==text.count('}')
