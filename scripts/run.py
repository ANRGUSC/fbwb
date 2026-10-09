"""Launch one finite, logged Webots experiment in a separate simulator instance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hover', 'scripted', 'upstream', 'stable', 'corrected'], default='hover')
    parser.add_argument('--scene', choices=['diagnostic', 'corridor', 'explain'], default='corridor')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--panel',action='store_true',help='Open the side-by-side live/replay visualization')
    parser.add_argument('--record',action='store_true',help='Record detailed live telemetry without opening the panel')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    out = args.output or ROOT / 'results' / args.mode
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    run_id = str(uuid.uuid4())
    env.update(FBWB_MODE=args.mode, FBWB_SCENE=args.scene, FBWB_DURATION=str(args.duration), FBWB_OUTPUT=str(out), FBWB_RUN_ID=run_id, PYTHONIOENCODING='utf-8')
    env['FBWB_RECORD'] = '1' if args.panel or args.record or args.scene == 'explain' else '0'
    if args.scene == 'explain' and args.mode not in ('corrected','hover'):
        parser.error('Use --mode corrected for the demo, or hover for fixture validation.')
    webots_home = Path(env.get('WEBOTS_HOME', 'C:/Program Files/Webots'))
    executable = webots_home/'msys64/mingw64/bin/webots.exe'
    if not executable.exists():
        parser.error('Webots R2025a was not found. Install it and set WEBOTS_HOME to its installation directory. The launcher/panel currently target Windows.')
    if not (ROOT/'protos/Crazyflie.proto').exists() or not (ROOT/'controllers/flybrain_webots/pid_controller.py').exists():
        parser.error('Missing downloaded Webots assets; run python scripts/fetch_assets.py --flybrain first.')
    env['WEBOTS_HOME'] = str(webots_home)
    # runtime.ini is project local; no change to Webots global preferences.
    (ROOT / 'controllers/flybrain_webots/runtime.ini').write_text('[python]\nCOMMAND = "' + sys.executable.replace('\\', '/') + '"\n')
    env['PATH'] = str(webots_home / 'msys64/mingw64/bin') + os.pathsep + env['PATH']
    command = [str(webots_home / 'msys64/mingw64/bin/webots.exe'), '--batch', '--mode=fast', '--stdout', '--stderr']
    if args.minimize:
        command.append('--minimize')
    world = ROOT/'worlds'/{'diagnostic':'flybrain.wbt','corridor':'corridor.wbt','explain':'explain.wbt'}[args.scene]
    command.append(str(world))
    status_path = out/'status.json'
    status_path.write_text(json.dumps({'run_id':run_id,'state':'starting'}))
    if args.panel:
        with (out/'panel.log').open('w') as log:
            process = subprocess.Popen([sys.executable,str(ROOT/'scripts/live_panel.py'),'--directory',str(out),'--run-id',run_id,'--world',str(world)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/'panel.pid').write_text(str(process.pid))
    print('Starting ' + args.mode + '; output: ' + str(out), flush=True)
    completed = subprocess.run(command, env=env, cwd=ROOT)
    if completed.returncode or not (out / 'summary.json').exists():
        status_path.write_text(json.dumps({'run_id':run_id,'state':'failed'}))
        raise SystemExit(completed.returncode or 1)
    summary = json.loads((out / 'summary.json').read_text())
    if summary.get('run_id') != run_id:
        status_path.write_text(json.dumps({'run_id':run_id,'state':'failed'}))
        raise SystemExit('No summary for this run; inspect Webots controller output.')
    status_path.write_text(json.dumps({'run_id':run_id,'state':'finished'}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
