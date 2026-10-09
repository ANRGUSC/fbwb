"""Controlled, visibly labeled interventions before free flight."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Phase:
    name: str
    start: float
    held: bool
    left_motion: tuple
    right_motion: tuple
    explanation: str
    turn_trial: bool = False


PHASES = [
    Phase('TAKEOFF',0,False,(0,0),(0,0),'The altitude PID raises the drone. The brain is not controlling takeoff.'),
    Phase('STILL BASELINE',6,True,(0,0),(0,0),'Pose held; static panels. Brain and frame history reset. Watch for zero visual drive.'),
    Phase('LEFT SCREEN',6.4,True,(0.65,0),(0,0),'Only the left screen moves. Compare the measured flow, receptor groups, and decoded yaw; pose remains held.'),
    Phase('RIGHT SCREEN',6.8,True,(0,0),(-0.65,0),'Only the right screen moves. Neural state is reset; compare anatomically labelled receptor and motion-cell sides.'),
    Phase('OUTWARD MOTION',7.2,True,(0.65,0),(-0.65,0),'Both screens move outward. Watch the radial-flow input and descending-neuron forward readout.'),
    Phase('VERTICAL MOTION',7.6,True,(0,0.65),(0,0.65),'Both screens move vertically. Watch the vertical-input receptor group. Neural climb is displayed but not applied.'),
    Phase('LEFT TURN',8,False,(0.65,0),(0,0),'Yaw-only rig: left screen drives the stabilized test camera. Neural yaw reaches the motors; position is held. This is not scripted yaw.',True),
    Phase('RIGHT TURN',10,False,(0,0),(-0.65,0),'Yaw-only rig: right screen drives the stabilized test camera. Pose and brain reset for comparison; neural yaw reaches the motors.',True),
    Phase('FREE FLIGHT',12,False,(0.65,0),(0,0),'Rig released from a centred pose; brain reset. Onboard camera now drives the brain. Neural forward/yaw commands control unconstrained flight.'),
]


def phase_at(t):
    return next(p for p in reversed(PHASES) if t >= p.start-1e-9)


def explain_step(phase, state, inputs, motor, applied, decode=None):
    """Mechanistic statements only; no invented neuron-level causal story."""
    pieces = [phase.explanation]
    if state['t'] < 6:
        pieces.append(f"Height target {applied['height']:.2f} m; measured height {state['z']:.2f} m. Vertical motion comes from the altitude controller.")
        return ' '.join(pieces)
    if decode:
        if 'motion_left_hz' in decode:
            pieces.append(f"Motion L/R: {decode['motion_left_hz']:.1f}/{decode['motion_right_hz']:.1f} Hz/cell; gain-balanced {decode['motion_left_relative']:.2f}/{decode['motion_right_relative']:.2f}; yaw {motor['yaw']:+.1f} RC.")
        else:
            pieces.append(f"Motion pools A/B: {decode['motion_a_sum_per_ms']:.2f}/{decode['motion_b_sum_per_ms']:.2f} spikes/ms; their difference gives yaw {motor['yaw']:+.1f} RC units.")
        pieces.append(f"Descending activity {decode['dn_sum_per_ms']:.2f} spikes/ms gives forward {motor['forward']:.1f} RC units.")
    if phase.held:
        pieces.append('These are shadow commands: the fixture prevents motion, regardless of neural output.')
    else:
        direction = 'left / CCW' if applied['yaw'] > 0.005 else 'right / CW' if applied['yaw'] < -0.005 else 'approximately straight'
        pieces.append(f"Applied forward target {applied['vx']:.3f} m/s; yaw {applied['yaw']:+.3f} rad/s ({direction}). Measured forward speed {state['vx']:.3f} m/s. Smoothing and inertia cause lag.")
    pieces.append(f"Height target is 1 m (PID); measured {state['z']:.2f} m. Neural climb is unused. Sideways is disabled; this forward decoder cannot reverse.")
    return ' '.join(pieces)


class VisualExperiment:
    def __init__(self, robot, motors):
        self.robot = robot
        self.motors = motors
        self.body = robot.getSelf()
        self.position = self.body.getField('translation')
        self.rotation = self.body.getField('rotation')
        self.left = robot.getFromDef('LEFT_PATTERN').getField('translation')
        self.right = robot.getFromDef('RIGHT_PATTERN').getField('translation')
        self.probe = robot.getFromDef('PROBE_CAMERA_POSE')
        self.previous_phase = None

    def before_step(self, next_time):
        phase = phase_at(next_time)
        changed = phase.name != self.previous_phase
        yaw = math.atan2(self.body.getOrientation()[3],self.body.getOrientation()[0])
        if phase.held or phase.turn_trial or (changed and phase.name == 'FREE FLIGHT'):
            yaw = 0 if phase.held or changed else yaw
            wz = self.body.getVelocity()[5] if phase.turn_trial and not changed else 0
            self.position.setSFVec3f([-0.8,0,1])
            self.rotation.setSFRotation([0,0,1,yaw])
            self.body.setVelocity([0,0,0,0,0,wz])
        # Test camera compensates rig yaw; free flight uses the onboard camera.
        # BODY extension origin is 15 mm below the robot origin.
        self.probe.getField('rotation').setSFRotation([0,0,1,-yaw])
        self.probe.getField('translation').setSFVec3f([.03*math.cos(yaw),-.03*math.sin(yaw),.025])
        self.previous_phase = phase.name
        elapsed = max(0,next_time-phase.start)
        for field,motion in [(self.left,phase.left_motion),(self.right,phase.right_motion)]:
            # Texture coordinates: local X is vertical, local Y is horizontal.
            field.setSFVec2f([-motion[1]*elapsed,-motion[0]*elapsed])
        return phase
