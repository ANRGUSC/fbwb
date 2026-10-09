"""Download pinned Webots assets and public FAFB v783 source data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch_flybrain():
    spec = json.loads((ROOT/'upstream.json').read_text())
    target = ROOT/'vendor/flybrain'
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',spec['repository'],str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',spec['commit']],check=True)
    actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    if actual != spec['commit']:
        raise RuntimeError('Existing vendor/flybrain is at a different commit; inspect it before checking out the pinned revision.')
    source = (target/'flybrain_tello_real_brain.py').read_text(encoding='utf-8')
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != spec['source_sha256']:
        raise RuntimeError('Flybrain source does not match the pinned UTF-8/LF checksum.')
    print('Verified Flybrain '+actual,flush=True)


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f'Already present: {destination.name}', flush=True)
        return
    partial = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
        total = int(response.headers.get('Content-Length', 0))
        done = 0
        report = 0
        while block := response.read(4 * 1024 * 1024):
            out.write(block)
            done += len(block)
            if done - report >= 128 * 1024 * 1024:
                print(f'  {done / 1e6:.0f}/{total / 1e6:.0f} MB', flush=True)
                report = done
        if total and done != total:
            raise IOError(f'Incomplete download: {done} != {total}')
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--flybrain',action='store_true',help='Clone and verify the pinned upstream source (not its runtime environment)')
    args = parser.parse_args()
    base = 'https://raw.githubusercontent.com/cyberbotics/webots/R2025a/'
    prefix = 'projects/robots/bitcraze/crazyflie/protos/'
    for name in ['Crazyflie.proto', 'meshes/cf2_assembly.dae', 'meshes/ccw_prop.dae']:
        download(base + prefix + name, ROOT / 'protos' / name)
    download(base + 'projects/default/protos/textures/fast_helix.png', ROOT / 'protos/textures/fast_helix.png')
    proto = ROOT / 'protos/Crazyflie.proto'
    text = proto.read_text()
    text = text.replace('webots://projects/default/protos/textures/fast_helix.png', 'textures/fast_helix.png')
    proto.write_text(text)
    target = ROOT / 'controllers/flybrain_webots/pid_controller.py'
    download(base+'projects/robots/bitcraze/crazyflie/controllers/crazyflie_py_wallfollowing/pid_controller.py',target)
    if args.flybrain:
        fetch_flybrain()
    if args.data:
        data_base = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/'
        manifest = {'dataset': 'FAFB v783', 'files': []}
        for name in ['neurons.csv.gz', 'consolidated_cell_types.csv.gz', 'classification.csv.gz', 'fafb_v783_princeton_synapse_table.csv.gz']:
            target = ROOT / 'data/raw' / name
            download(data_base + name, target)
            with target.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest['files'].append({'name': name, 'url': data_base + name, 'bytes': target.stat().st_size, 'sha256': digest})
        (ROOT / 'data/raw/manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
