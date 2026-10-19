"""Webots camera -> flybrain -> body velocity PID -> Crazyflie motors."""
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from controller import Supervisor
from pid_controller import pid_velocity_fixed_height_controller
from fbwb.commands import map_commands, CommandFilter
from fbwb.vision import OpticFlow
from fbwb.experiment import VisualExperiment, Phase, PHASES, explain_step
from fbwb.telemetry import Publisher

simulator = None


def main():
    global simulator
    mode = os.environ.get('FBWB_MODE', 'hover')
    vision_mode = os.environ.get('FBWB_VISION', 'optic-flow')
    decoder = os.environ.get('FBWB_DECODER', 'population')
    duration = float(os.environ.get('FBWB_DURATION', '30'))
    out = Path(os.environ.get('FBWB_OUTPUT', str(ROOT / 'results' / mode)))
    out.mkdir(parents=True, exist_ok=True)
    robot = Supervisor()
    simulator = robot
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000
    pid = pid_velocity_fixed_height_controller()
    motors = [robot.getDevice(f'm{i}_motor') for i in range(1, 5)]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0)
    imu, gps, gyro, camera = [robot.getDevice(name) for name in ['inertial_unit', 'gps', 'gyro', 'camera']]
    for sensor in [imu, gps, gyro, camera]:
        sensor.enable(timestep)
    scene = os.environ.get('FBWB_SCENE','corridor')
    experiment = VisualExperiment(robot,motors) if scene == 'explain' else None
    probe_camera = robot.getDevice('probe_camera') if experiment else None
    if probe_camera:
        probe_camera.enable(timestep)
    publisher = Publisher(out,os.environ.get('FBWB_RUN_ID','direct')) if os.environ.get('FBWB_RECORD') == '1' or experiment else None
    brain = None
    if mode in ('upstream', 'stable', 'corrected'):
        from fbwb.brain import load_brain
        print(f'Loading full connectome ({mode})...', flush=True)
        brain = load_brain(mode, timestep, compare_decoders=mode=='corrected', decoder=decoder,
                           dna02_gain=float(os.environ.get('FBWB_DNA02_GAIN','2')))
    elif mode not in ('hover', 'scripted'):
        raise ValueError(f'Unknown mode {mode}')
    vision = OpticFlow()
    flyvis = eye_cameras = None
    if vision_mode == 'flyvis':
        if mode != 'corrected':
            raise ValueError('FlyVis input requires corrected mode')
        from fbwb.flyvis_input import FlyVisInput, WebotsFlyVisCameras
        print('Loading pretrained FlyVis on CPU...',flush=True)
        flyvis = FlyVisInput(brain.neurons_df,timestep)
        eye_cameras = WebotsFlyVisCameras(robot,timestep)
    compare_stream = (out/'decoder_comparison.csv').open('w',newline='')
    compare_fields = ['t','phase','vision','selected_decoder','population_yaw_rad_s','dna02_yaw_rad_s','dna02_left_hz','dna02_right_hz','applied_yaw_rad_s','vision_ms','brain_ms']
    comparison = csv.DictWriter(compare_stream,fieldnames=compare_fields)
    comparison.writeheader()
    command_filter = CommandFilter()
    past_position = None
    started = time.monotonic()
    samples = []
    fields = ['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vx_target', 'yaw_target', 'vx_raw', 'yaw_raw', 'flow_forward', 'flow_left', 'flow_right', 'flow_vertical', 'spikes', 'dn_activity', 'brain_ms']
    saved_frame = False
    initial_position = None
    contact_steps = 0
    doorway_passed = False
    previous_phase = None
    print(f'FBWB start: {mode}, {duration}s simulation, {timestep}ms steps', flush=True)
    with (out / 'telemetry.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while True:
            try:
                request = json.loads((out/'control.json').read_text())
                if request.get('run_id') == os.environ.get('FBWB_RUN_ID') and request.get('action') == 'stop':
                    print('Stop requested from visualization panel.',flush=True)
                    break
            except (OSError,ValueError):
                pass
            next_time = robot.getTime()+dt
            phase = experiment.before_step(next_time) if experiment else (PHASES[0] if next_time < 6 else Phase('FREE FLIGHT',6,False,(0,0),(0,0),'Live camera drives the neural input; filtered forward/yaw commands drive the drone.'))
            if eye_cameras:
                orientation = robot.getSelf().getOrientation()
                eye_cameras.stabilize(math.atan2(orientation[3],orientation[0]),phase.held or phase.turn_trial)
            if robot.step(timestep) == -1:
                break
            t = robot.getTime()
            if phase.name != previous_phase:
                if phase.held or phase.turn_trial or (experiment and phase.name == 'FREE FLIGHT'):
                    if brain is not None:
                        brain.reset_state()
                    vision = OpticFlow()
                    if flyvis:
                        flyvis.reset()
                if experiment:
                    command_filter = CommandFilter()
                    past_position = None
                previous_phase = phase.name
            position = np.array(gps.getValues())
            if initial_position is None:
                initial_position = position.copy()
            if t >= 6:
                contact_steps += int(len(robot.getSelf().getContactPoints()) > 0)
                if os.environ.get('FBWB_SCENE') == 'corridor':
                    doorway_passed |= bool(position[0] > 2.3 and abs(position[1]) < 0.65 and 0.1 < position[2] < 1.75)
            roll, pitch, yaw = imu.getRollPitchYaw()
            yaw_rate = gyro.getValues()[2]
            velocity = np.zeros(3) if past_position is None else (position - past_position) / dt
            past_position = position.copy()
            vx = velocity[0]*math.cos(yaw) + velocity[1]*math.sin(yaw)
            vy = -velocity[0]*math.sin(yaw) + velocity[1]*math.cos(yaw)
            active_camera = probe_camera if experiment and (phase.held or phase.turn_trial) else camera
            frame = np.frombuffer(active_camera.getImage(), dtype=np.uint8).reshape(active_camera.getHeight(), active_camera.getWidth(), 4)
            flow = vision.compute(frame)
            desired_vx = desired_vy = desired_yaw = 0.0
            metrics = {'total_spikes': 0.0, 'dn_activity': 0.0}
            brain_ms = 0.0
            vision_ms = 0.0
            motor_command = {'forward':0.0,'yaw':0.0,'vertical':0.0,'roll':0.0}
            # A reproducible ascent supplies camera motion, then hand off at 6s.
            height = min(1.0, 0.05 + 0.25*t)
            if brain is not None and t >= 6:
                if flyvis:
                    visual_tick = time.monotonic()
                    visual_rates = flyvis.compute(eye_cameras.frames())
                    brain.set_visual_rates(flyvis.indices,visual_rates)
                    vision_ms = (time.monotonic()-visual_tick)*1000
                tick = time.monotonic()
                with __import__('torch').no_grad():
                    motor_command, metrics = brain.compute(np.zeros(4) if flyvis else flow)
                brain_ms = (time.monotonic()-tick)*1000
                if t >= 6:
                    desired_vx, desired_vy, desired_yaw = map_commands(motor_command)
            if mode == 'scripted' and 6 <= t < 10:
                desired_vx = 0.2
            if mode == 'scripted' and 10 <= t < 12:
                desired_yaw = 0.5
            raw_vx, raw_yaw = desired_vx, desired_yaw
            if brain is not None:
                desired_vx, desired_yaw = command_filter.update(raw_vx, raw_yaw, dt)
            shadow_vx, shadow_yaw = desired_vx, desired_yaw
            if phase.held:
                desired_vx = desired_vy = desired_yaw = 0.0
            elif phase.turn_trial:
                desired_vx = desired_vy = 0.0
            power = pid.pid(dt, desired_vx, desired_vy, desired_yaw, height, roll, pitch, yaw_rate, position[2], vx, vy)
            state = [*position, roll, pitch, yaw, *power]
            if not np.isfinite(state).all():
                raise RuntimeError('Non-finite simulator or controller state')
            for motor, sign, value in zip(motors, [-1, 1, -1, 1], power):
                motor.setVelocity(sign*float(value))
            row = dict(zip(fields, map(float,[t, *position, roll, pitch, yaw, vx, vy, desired_vx, desired_yaw, raw_vx, raw_yaw, *flow, metrics['total_spikes'], metrics['dn_activity'], brain_ms])))
            writer.writerow(row)
            samples.append(row)
            observation = getattr(brain,'observation',None) if brain is not None and t >= 6 else None
            if observation and 'selected_decoder' in observation.get('decode',{}):
                d = observation['decode']
                comparison.writerow(dict(t=t,phase=phase.name,vision=vision_mode,selected_decoder=decoder,
                    **{k:d[k] for k in ['population_yaw_rad_s','dna02_yaw_rad_s','dna02_left_hz','dna02_right_hz']},
                    applied_yaw_rad_s=desired_yaw,vision_ms=vision_ms,brain_ms=brain_ms))
                compare_stream.flush()
            if publisher and (t >= 6 or len(samples)%5 == 0):
                observation = getattr(brain,'observation',None) if brain is not None and t >= 6 else None
                applied = {'vx':desired_vx,'vy':desired_vy,'yaw':desired_yaw,'height':height,'shadow_vx':shadow_vx,'shadow_yaw':shadow_yaw,'motors':[float(p)*s for p,s in zip(power,[-1,1,-1,1])]}
                narrative = explain_step(phase,row,flow,motor_command,applied,observation['decode'] if observation else None,vision_mode)
                if observation and 'selected_decoder' in observation['decode']:
                    d = observation['decode']
                    dn_text = f"{d['dna02_yaw_rad_s']:+.3f}" if d['dna02_available'] else 'unavailable'
                    narrative = (f"Vision: {vision_mode}; applied decoder: {decoder}. "
                        f"Same-spike yaw proposals: population {d['population_yaw_rad_s']:+.3f}, DNa02 {dn_text} rad/s. "+narrative)
                publisher.publish({'state':row,'phase':{'name':phase.name,'held':phase.held,'turn_trial':phase.turn_trial,'camera':('panoramic FlyVis rig; front preview' if flyvis else 'stabilized test camera' if active_camera is probe_camera else 'onboard camera')},'mode':mode,'scene':scene,'vision_mode':vision_mode,'visual':flyvis.observation if flyvis else None,'inputs':flow.tolist(),'brain':observation,'decoded':motor_command,'applied':applied,'narrative':narrative,'status':'running'},frame)
            if len(samples) % 50 == 0:
                stream.flush()
                print(f't={t:.1f} z={position[2]:.3f} vx={vx:.3f} spikes={metrics["total_spikes"]:.1f} brain_ms={brain_ms:.1f}', flush=True)
            if t >= 6 and not saved_frame:
                cv2.imwrite(str(out / 'camera.png'), frame)
                robot.exportImage(str(out / 'scene.png'), 90)
                saved_frame = True
            if t >= duration:
                break
    if samples:
        tail = [s for s in samples if s['t'] >= 6]
        summary = {'mode': mode, 'steps': len(samples), 'simulation_seconds': samples[-1]['t'], 'wall_seconds': time.monotonic()-started, 'final_position': [samples[-1][a] for a in ['x', 'y', 'z']], 'max_abs_roll_pitch': max(max(abs(s['roll']), abs(s['pitch'])) for s in samples), 'max_altitude_error_after_6s': max((abs(s['z']-1) for s in tail), default=None), 'mean_brain_ms': float(np.mean([s['brain_ms'] for s in samples])), 'max_mean_spikes_per_neural_step': max(s['spikes'] for s in samples), 'max_dn_activity': max(s['dn_activity'] for s in samples)}
        summary['run_id'] = os.environ.get('FBWB_RUN_ID', 'direct-world-open')
        summary['scene'] = os.environ.get('FBWB_SCENE', 'direct-world-open')
        summary['vision'] = vision_mode
        summary['selected_decoder'] = decoder
        summary['decoder_comparison'] = 'decoder_comparison.csv'
        if flyvis:
            summary['flyvis'] = flyvis.metadata
        summary['recording'] = 'live.ndjson' if publisher else None
        summary['forward_progress_m'] = float(samples[-1]['x']-initial_position[0])
        summary['airborne_contact_steps'] = contact_steps
        summary['doorway_passed'] = doorway_passed
        summary['mean_active_brain_ms'] = float(np.mean([s['brain_ms'] for s in tail])) if tail else None
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('FBWB_RESULT ' + json.dumps(summary), flush=True)
    if publisher:
        publisher.close()
    compare_stream.close()
    robot.simulationQuit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        error_out = Path(os.environ.get('FBWB_OUTPUT',str(ROOT/'results')))
        error_out.mkdir(parents=True,exist_ok=True)
        (error_out/'controller-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        traceback.print_exc()
        sys.stderr.flush()
        # Exit Webots as well, making failed batch runs unambiguous.
        if simulator is not None:
            simulator.simulationQuit(1)
        raise
