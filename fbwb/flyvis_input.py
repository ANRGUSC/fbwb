"""CPU FlyVis -> spatially mapped Poisson drive; no optical-flow preprocessing.

Handoff maps/scales are from vib2810/fly-brain-zero-shot (MIT). The mapping is
approximate, not a measured visual receptive field for each FlyWire neuron.
"""
import os
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'data/flyvis_bridge'
MODEL_SHA256 = {
    'best_chkpt':'d8a57a022aa18ca338599be713af9db29b89cde4b00098ffe42b78f0d583e46a',
    '_meta.yaml':'2aa8a9505324c82bbb4011ee2f7c375b5f8b1bddde1a5281e038baecf91fc5dd',
}


def verify_assets(assets=ASSETS):
    for name,digest in MODEL_SHA256.items():
        path = ROOT/'data/flyvis/results/flow/0000/000'/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or changed pretrained model {path}; run scripts/setup_flyvis.py --download-models')
    assets = Path(assets)
    manifest = json.loads((assets/'manifest.json').read_text())
    if hashlib.sha256((assets/'handoff.npz').read_bytes()).hexdigest() != manifest['handoff_sha256']:
        raise ValueError('FlyVis handoff checksum mismatch; rerun setup')


def windows_datamate_compat():
    # datamate 1.0.0 unlinks an open HDF5 file on every initial write. Windows
    # rejects that. Keep the fix scoped to this process; never edit site-packages.
    if os.name != 'nt':
        return
    import datamate.directory
    import datamate.io
    import h5py

    def write_h5(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w', libver='latest') as stream:
            stream.create_dataset('data', data=np.asarray(value))
            stream.swmr_mode = True

    datamate.io._write_h5 = write_h5
    datamate.directory._write_h5 = write_h5


def remap_root_ids(source_ids, target_ids):
    """Never reuse row indices from a differently ordered connectome."""
    lookup = {int(root): i for i, root in enumerate(target_ids)}
    if len(lookup) != len(target_ids) or len(np.unique(source_ids)) != len(source_ids):
        raise ValueError('Duplicate neuron IDs in handoff or target')
    missing = [int(root) for root in source_ids if int(root) not in lookup]
    if missing:
        raise ValueError(f'{len(missing)} FlyVis handoff neurons absent from local brain')
    return np.array([lookup[int(root)] for root in source_ids], dtype=np.int64)


def evoked_rates(activity, baseline, scale):
    if not np.all(np.isfinite(activity)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Invalid FlyVis activity or calibration')
    return 150 * np.clip(np.maximum(activity-baseline, 0)/scale, 0, 1)


# Camera forward, image-left, image-up axes in robot coordinates (+X forward).
CUBE_AXES = np.array([
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,1,0],[-1,0,0],[0,0,1]],
    [[-1,0,0],[0,-1,0],[0,0,1]],
    [[0,-1,0],[1,0,0],[0,0,1]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
], dtype=float)
CUBE_ROTATIONS = [(0,0,1,0),(0,0,1,np.pi/2),(0,0,1,np.pi),
                  (0,0,1,-np.pi/2),(0,1,0,-np.pi/2),(0,1,0,np.pi/2)]


def sample_cube(frames, directions):
    """Sample local luminance from six 90-degree planar cameras.

Small spatial averaging approximates receptor acceptance, not phototransduction.
"""
    rays = np.asarray(directions)
    dots = rays @ CUBE_AXES[:, 0].T
    faces = dots.argmax(axis=-1)
    values = np.empty(rays.shape[:-1], dtype=np.float32)
    for face, frame in enumerate(frames):
        mask = faces == face
        if not np.any(mask):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY).astype(np.float32)/255
        gray = cv2.GaussianBlur(gray, (3,3), 0.8)
        h,w = gray.shape
        ray = rays[mask]
        depth = ray @ CUBE_AXES[face,0]
        # Webots planar projection uses focal length width/(2*tan(FOV/2)).
        x = (w-1)/2 - w/2*(ray @ CUBE_AXES[face,1])/depth
        y = (h-1)/2 - w/2*(ray @ CUBE_AXES[face,2])/depth
        values[mask] = cv2.remap(gray,x.astype('float32')[None],y.astype('float32')[None],
                                cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
    return values


class WebotsFlyVisCameras:
    """Enable sensors in the launcher's generated FlyVis world."""
    def __init__(self, robot, timestep):
        self.pose = robot.getFromDef('FLYVIS_RIG')
        self.cameras = [robot.getDevice(f'flyvis_{i}') for i in range(6)]
        if self.pose is None or any(c is None for c in self.cameras):
            raise RuntimeError('Webots did not register the panoramic FlyVis cameras')
        for camera in self.cameras:
            camera.enable(timestep)

    def stabilize(self, yaw, held):
        self.pose.getField('rotation').setSFRotation([0,0,1,-yaw if held else 0])

    def frames(self):
        return [np.frombuffer(c.getImage(),dtype=np.uint8).reshape(c.getHeight(),c.getWidth(),4) for c in self.cameras]


def prepare_flyvis_world(source):
    """Add static devices before Webots loads the PROTO; preserve original worlds."""
    source = Path(source)
    text = source.read_text()
    marker = 'DEF DRONE Crazyflie {'
    if text.count(marker) != 1:
        raise ValueError('Expected one DEF DRONE Crazyflie in the source world')
    prefix,drone = text.split(marker)
    cameras = ['Camera { name "flyvis_%d" rotation %s fieldOfView %.16f width 96 height 96 }' %
               (i,' '.join(map(str,rotation)),np.pi/2) for i,rotation in enumerate(CUBE_ROTATIONS)]
    rig = '\n    DEF FLYVIS_RIG Pose { translation .03 0 .08 children [\n      '+'\n      '.join(cameras)+'\n    ] }\n'
    if 'extensionSlot [' in drone:
        drone = drone.replace('extensionSlot [','extensionSlot ['+rig,1)
    else:
        drone = '\n  extensionSlot ['+rig+'  ]\n'+drone
    target = source.with_name('.flyvis-'+source.name)
    target.write_text(prefix+marker+drone)
    return target


class FlyVisInput:
    def __init__(self, neurons, period_ms=20, assets=ASSETS):
        verify_assets(assets)
        import torch
        os.environ['FLYVIS_ROOT_DIR'] = str(ROOT/'data/flyvis')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        windows_datamate_compat()
        import flyvis
        flyvis.device = torch.device('cpu')
        torch.set_default_device('cpu')
        torch.set_num_threads(4)
        from flyvis import NetworkView
        self.torch = torch
        self.dt = period_ms / 1000
        if period_ms <= 0 or period_ms % 10:
            raise ValueError('FlyVis control period must be a multiple of 10 ms')
        self.steps = int(period_ms/10)
        self.net = NetworkView('flow/0000/000').init_network().cpu()
        self.net.eval()
        self.net.requires_grad_(False)
        z = np.load(Path(assets)/'handoff.npz',allow_pickle=False)
        self.indices = remap_root_ids(z['root_id'],neurons.root_id.to_numpy())
        self.eye = z['eye'].astype(int)
        self.types = z['type'].astype(str)
        if not np.array_equal(neurons.primary_type.iloc[self.indices].to_numpy().astype(str),self.types):
            raise ValueError('Handoff cell types changed; rerun setup')
        nodes = self.net.connectome.nodes
        node_types = np.asarray(nodes.type[:]).astype(str)
        self.node = np.array([np.flatnonzero(node_types == ty)[col] for ty,col in zip(self.types,z['column'])])
        self.scale = z['scale']
        self.directions = z['directions']
        self.metadata = dict(model='flow/0000/000',device='cpu',handoff_neurons=len(self.indices),
                             mapping='Mi1/connectivity-derived approximate columns',
                             sensory_camera='six 90-degree cameras; binocular angular samples')
        with torch.no_grad():
            self.initial_state = None
            for _ in range(100):
                self.initial_state = self.net.steady_state(.01,.01,2,value=.8,state=self.initial_state)
        self.baseline = self.initial_state.nodes.activity.detach().cpu().numpy()[self.eye,self.node].copy()
        self.reset()

    def reset(self):
        from flyvis.utils.tensor_utils import clone
        self.state = clone(self.initial_state)
        self.observation = None

    def compute_luminance(self, luminance):
        from flyvis.utils.nn_utils import simulation
        lum = np.asarray(luminance,dtype=np.float32)
        if lum.shape != (2,721) or not np.all(np.isfinite(lum)) or np.any(lum < 0) or np.any(lum > 1):
            raise ValueError('Expected binocular 2x721 luminance in [0,1]')
        with self.torch.no_grad(), simulation(self.net):
            self.net.stimulus.zero(2,1)
            self.net.stimulus.add_input(self.torch.tensor(lum[:,None,None,:],device='cpu'))
            for _ in range(self.steps):
                self.state = self.net.forward(self.net.stimulus(),.01,self.state,as_states=True)[-1]
            activity = self.state.nodes.activity.detach().cpu().numpy()[self.eye,self.node]
        rates = evoked_rates(activity,self.baseline,self.scale)
        self.observation = dict(self.metadata,mean_luminance=lum.mean(axis=1).tolist(),
                                mean_drive_hz=[float(rates[self.eye==e].mean()) for e in (0,1)],
                                max_drive_hz=float(rates.max()),active_inputs=int(np.count_nonzero(rates)))
        return rates

    def compute(self, frames):
        return self.compute_luminance(sample_cube(frames,self.directions))
