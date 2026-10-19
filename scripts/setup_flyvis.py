"""Fetch verified bridge assets and prepare a root-ID keyed handoff on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
COMMIT = '0c63ba5cdd123f8d38eea62deb574353f0ec2632'
BASE = f'https://raw.githubusercontent.com/vib2810/fly-brain-zero-shot/{COMMIT}/'
FILES = {
    'data/eyemap/flywire_columns.npz':'29c36f5dca105e55bf074a0716c7be8f569539e0d96427176b4988fe8cd3b2a2',
    'data/eyemap/ommatidia_dirs.npz':'effbc52b0671f2633d2e029ba26a0fba4891f8e8c46d8cb04b0b44571e705433',
    'data/eyemap/handoff_scale_bars.npz':'337c7026a8cf0804e69fa7a94aff9b9f71792436a3b30fbf637fdaa3d3b4cd40',
    'LICENSE':'3a19eafc077233f0a66bb0d9a8fe06f0fa504b88b78987204cd3c550b1471832',
}


def fetch(url, path, sha):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,partial)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != sha:
            raise ValueError(f'Checksum mismatch: {url}')
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f'Checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-models',action='store_true')
    args = parser.parse_args()
    reference = ROOT/'vendor/zero-shot-reference'
    for name,sha in FILES.items():
        fetch(BASE+name,reference/name,sha)
    completeness = reference/'Completeness_783.csv'
    fetch('https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Completeness_783.csv',
          completeness,'bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311')
    os.environ['FLYVIS_ROOT_DIR'] = str(ROOT/'data/flyvis')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.download_models:
        subprocess.run([sys.executable,'-m','flyvis_cli.download_pretrained_models','--skip_large_files'],check=True)
    import numpy as np
    import pandas as pd
    from fbwb.flyvis_input import windows_datamate_compat, remap_root_ids, verify_assets, MODEL_SHA256
    windows_datamate_compat()
    from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel
    xy = np.array(hex_to_pixel(*get_hex_coords(15))).T
    maps = np.load(reference/'data/eyemap/flywire_columns.npz')
    facets = np.load(reference/'data/eyemap/ommatidia_dirs.npz')
    scales = np.load(reference/'data/eyemap/handoff_scale_bars.npz')
    scale_lookup = dict(zip(scales['types'].astype(str),scales['scale']))
    ids = pd.read_csv(completeness,index_col=0).index.to_numpy(np.int64)
    arrays = {k:[] for k in ('root_id','eye','type','column','scale')}
    directions = []
    for e,eye in enumerate(('left','right')):
        seen = facets['seen'][e]
        alo,ahi = np.percentile(np.abs(facets['az'][e][seen]),[2,98])
        elo,ehi = np.percentile(facets['el'][e][seen],[2,98])
        xlo,xhi = np.percentile(xy[:,0],[2,98]); ylo,yhi = np.percentile(xy[:,1],[2,98])
        az = np.radians((ahi-(xy[:,0]-xlo)/(xhi-xlo)*(ahi-alo))*(1 if e==0 else -1))
        el = np.radians(elo+(xy[:,1]-ylo)/(yhi-ylo)*(ehi-elo))
        directions.append(np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)],axis=-1))
        types = maps[f'{eye}_type'].astype(str)
        arrays['root_id'].append(ids[maps[f'{eye}_neuron']])
        arrays['eye'].append(np.full(len(types),e,dtype=np.int32))
        arrays['type'].append(types)
        arrays['column'].append(maps[f'{eye}_column'])
        arrays['scale'].append(np.array([scale_lookup[t] for t in types]))
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    local = pd.read_csv(ROOT/'data/prepared/neurons.csv')
    indices = remap_root_ids(arrays['root_id'],local.root_id)
    mismatches = local.primary_type.iloc[indices].to_numpy().astype(str) != arrays['type']
    excluded = [dict(root_id=int(root),reference=str(ref),local=str(loc)) for root,ref,loc in
                zip(arrays['root_id'][mismatches],arrays['type'][mismatches],local.primary_type.iloc[indices[mismatches]])]
    # Preserve our annotation/brain table. Do not force a disputed subtype match.
    arrays = {k:v[~mismatches] for k,v in arrays.items()}
    target = ROOT/'data/flyvis_bridge'
    target.mkdir(exist_ok=True,parents=True)
    np.savez(target/'handoff.npz',**arrays,directions=np.asarray(directions))
    report = dict(reference_commit=COMMIT,source_sha256=FILES,model='flow/0000/000',
                  model_sha256=MODEL_SHA256,
                  handoff_sha256=hashlib.sha256((target/'handoff.npz').read_bytes()).hexdigest(),
                  matched_neurons=len(arrays['root_id']),excluded_type_mismatches=excluded,device='cpu',
                  mapping='Approximate Mi1/connectivity-derived visual columns; source map remapped by root_id')
    (target/'manifest.json').write_text(json.dumps(report,indent=2))
    verify_assets(target)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
