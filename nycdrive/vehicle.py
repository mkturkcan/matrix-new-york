"""Vehicle dynamics and their mapping onto world-model action spaces.

Driver inputs are per-frame (throttle, steer) in [-1, 1]. A kinematic bicycle
model turns them into vehicle speed and heading; a chase camera follows the
heading with a first-order lag. The resulting trajectory is then expressed in
the action space of each world model:

* Matrix-Game-2.0 GTA-drive: keyboard [W, S] + mouse [0, steer]; the model was
  trained on third-person GTA V driving and renders the vehicle itself.
* Matrix-Game-3.0 (first-person): keyboard W + mouse yaw = camera yaw rate.
  The vehicle is drawn by us, rotated by its yaw relative to the camera.
"""
from dataclasses import dataclass

import numpy as np
import torch

FPS = 17


@dataclass
class VehicleParams:
    wheelbase: float = 2.7          # m
    max_steer: float = np.deg2rad(32)
    steer_speed_ref: float = 6.0    # m/s; steering authority falls off as 1 / (1 + v / ref)
    max_speed: float = 25.0         # m/s  (~56 mph)
    accel: float = 3.0              # m/s^2 at full throttle
    brake: float = 7.0              # m/s^2 at full brake
    drag: float = 0.12              # 1/s, linearised drag; steady state v = accel * throttle / drag
    cam_lag: float = 0.35           # s, chase-camera heading time constant


def simulate(controls: np.ndarray, p: VehicleParams = VehicleParams(), fps: int = FPS, v0: float = 0.0):
    """controls: (T, 2) [throttle, steer]. Returns dict of per-frame arrays (SI units, radians)."""
    dt = 1.0 / fps
    T = len(controls)
    v = np.zeros(T); psi = np.zeros(T); psi_cam = np.zeros(T); delta = np.zeros(T)
    x = np.zeros(T); y = np.zeros(T)
    speed, heading, cam, px, py = v0, 0.0, 0.0, 0.0, 0.0
    alpha = 1.0 - np.exp(-dt / p.cam_lag)
    for t, (thr, st) in enumerate(controls):
        a = thr * p.accel if thr >= 0 else thr * p.brake
        speed = float(np.clip(speed + (a - p.drag * speed) * dt, 0.0, p.max_speed))
        d = st * p.max_steer / (1.0 + speed / p.steer_speed_ref)
        heading += speed / p.wheelbase * np.tan(d) * dt
        cam += alpha * (heading - cam)
        px += speed * np.cos(heading) * dt
        py += speed * np.sin(heading) * dt
        v[t], psi[t], psi_cam[t], delta[t], x[t], y[t] = speed, heading, cam, d, px, py
    return {"v": v, "psi": psi, "psi_cam": psi_cam, "delta": delta, "x": x, "y": y,
            "rel_yaw": psi - psi_cam, "controls": np.asarray(controls, dtype=np.float32)}


# --------------------------------------------------------------------------- routes
# A route is a list of (seconds, throttle, steer). Lane changes are steering
# S-curves (non-holonomic), not lateral translations.
def lane_change(direction: int, steer: float = 0.7, seconds: float = 0.5):
    return [(seconds, 1.0, direction * steer), (seconds, 1.0, -direction * steer)]


ROUTES = {
    "cruise": [(3.0, 1.0, 0.0), (60.0, 0.7, 0.0)],
    "highway": [(5.0, 1.0, 0.0), *lane_change(-1), (3.0, 1.0, 0.0), (3.0, 1.0, 0.08), (3.0, 0.8, 0.0),
                *lane_change(+1), (3.0, 1.0, -0.06), (60.0, 0.8, 0.0)],
    "downtown": [(4.0, 0.6, 0.0), *lane_change(+1, 0.6), (3.0, 0.5, 0.0), (1.5, -0.6, 0.0),
                 (2.0, 0.8, 0.0), *lane_change(-1, 0.6), (60.0, 0.5, 0.0)],
    "turn_right": [(3.0, 0.7, 0.0), (1.0, -0.4, 0.0), (3.0, 0.5, 0.8), (60.0, 0.7, 0.0)],
    "turn_left": [(3.0, 0.7, 0.0), (1.0, -0.4, 0.0), (3.0, 0.5, -0.8), (60.0, 0.7, 0.0)],
    "slalom": [(3.0, 1.0, 0.0), *lane_change(-1, 0.8, 0.5), *lane_change(+1, 0.8, 0.5),
               *lane_change(-1, 0.8, 0.5), *lane_change(+1, 0.8, 0.5), (60.0, 0.8, 0.0)],
}


def route_controls(route, num_frames: int, fps: int = FPS) -> np.ndarray:
    """Expand a route (preset name or list of segments) to (num_frames, 2) controls."""
    if isinstance(route, str):
        route = ROUTES[route]
    out = np.zeros((num_frames, 2), dtype=np.float32)
    t = 0
    for seconds, thr, st in route:
        n = max(1, int(round(seconds * fps)))
        out[t:t + n] = (thr, st)
        t += n
        if t >= num_frames:
            break
    if t < num_frames:
        out[t:] = out[t - 1]
    # first-order smoothing of the steering wheel (~0.15 s) to avoid step inputs
    k = 1.0 - np.exp(-1.0 / (0.15 * fps))
    for i in range(1, num_frames):
        out[i, 1] = out[i - 1, 1] + k * (out[i, 1] - out[i - 1, 1])
    return out


# --------------------------------------------------------------------------- action adapters
def pwm(u: np.ndarray, fps: int, period_s: float = 1.0) -> np.ndarray:
    """Binary actuation of a continuous command in [0, 1] (duty cycle = u)."""
    n = max(2, int(round(period_s * fps)))
    phase = (np.arange(len(u)) % n + 0.5) / n
    return (u > phase).astype(np.float32)


def to_mg2_gta(sim, steer_scale: float = 0.1, fps: int = 12):
    """Matrix-Game-2.0 GTA-drive: keyboard (T, 2) = [W, S] (PWM of throttle / brake),
    mouse (T, 2) = [0, steer * 0.1] (the training data uses +-0.1 for A / D)."""
    c = sim["controls"]
    keyboard = np.stack([pwm(np.clip(c[:, 0], 0, 1), fps), pwm(np.clip(-c[:, 0], 0, 1), fps)], 1)
    mouse = np.stack([np.zeros(len(c)), np.clip(c[:, 1], -1, 1) * steer_scale], 1).astype(np.float32)
    return {"keyboard_condition": torch.from_numpy(keyboard), "mouse_condition": torch.from_numpy(mouse)}


MG3_YAW_DEG_PER_UNIT = 15.0   # utils.utils.MOUSE_YAW_SENSITIVITY in Matrix-Game-3
MG3_DEADZONE = 0.02


def to_mg3(sim, max_mouse: float = 0.12, fps: int = FPS, v_ref: float = 10.0):
    """Matrix-Game-3.0: keyboard (T, 6) with W duty cycle = min(v / v_ref, 1), mouse yaw = camera yaw rate.

    MG3's forward speed while W is held is fixed by the model (scaling the camera translation
    used for its Plucker conditioning by 3x left the output unchanged), so W saturates above v_ref.
    """
    T = len(sim["v"])
    keyboard = np.zeros((T, 6), dtype=np.float32)
    keyboard[:, 0] = pwm(np.clip(sim["v"] / v_ref, 0, 1), fps, period_s=0.5)
    dpsi = np.diff(np.rad2deg(sim["psi_cam"]), prepend=0.0)
    yaw = np.clip(dpsi / MG3_YAW_DEG_PER_UNIT, -max_mouse, max_mouse)
    # MG3 ignores |yaw| < deadzone; accumulate sub-threshold rotation and release it in steps
    acc, out = 0.0, np.zeros(T, dtype=np.float32)
    for i, y in enumerate(yaw):
        acc += y
        if abs(acc) >= MG3_DEADZONE:
            out[i], acc = acc, 0.0
    mouse = np.stack([np.zeros(T), out], 1).astype(np.float32)
    return {"keyboard_condition": torch.from_numpy(keyboard), "mouse_condition": torch.from_numpy(mouse)}
