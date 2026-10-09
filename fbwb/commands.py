"""Explicit Tello-style normalized commands to Crazyflie SI velocity targets."""
import math


def map_commands(motor, yaw_sign=-1.0):
    forward, yaw = float(motor['forward']), float(motor['yaw'])
    if not all(math.isfinite(x) for x in (forward, yaw)):
        raise ValueError('Non-finite neural command')
    # Tello positive yaw is clockwise; Crazyflie positive yaw is counterclockwise.
    vx = 0.3 * max(-100, min(100, forward)) / 100
    wz = yaw_sign * max(-100, min(100, yaw)) / 100
    return vx, 0.0, wz


class CommandFilter:
    """Avoid differentiating neural-window jitter in the flight PID."""
    def __init__(self, tau=0.2):
        self.tau = tau
        self.vx = 0.0
        self.yaw = 0.0

    def update(self, vx, yaw, dt):
        if dt <= 0 or not all(math.isfinite(x) for x in (vx, yaw, dt)):
            raise ValueError('Invalid command/filter timestep')
        alpha = 1-math.exp(-dt/self.tau)
        self.vx += max(-0.3*dt, min(0.3*dt, alpha*(vx-self.vx)))
        self.yaw += max(-1.0*dt, min(1.0*dt, alpha*(yaw-self.yaw)))
        return self.vx, self.yaw
