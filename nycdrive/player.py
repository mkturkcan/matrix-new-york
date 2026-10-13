"""Player-car layer re-composited on every generated frame.

Matrix-Game-3 is a first-person world model: it treats the car composited into
the first frame as part of the (static) world and drives through it.  Like a
game engine drawing the player model over the world, we keep the same car
sprite in place on every frame and animate it slightly with the steering input.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_car_layer(scene_dir: Path):
    """Return (rgb float32 HxWx3, alpha float32 HxW) or None.

    Prefers `car_layer.png` (exact RGBA written by prepare_inputs).  Falls back to
    the difference between `image.png` and `inpainted.png`.
    """
    scene_dir = Path(scene_dir)
    if (scene_dir / "car_layer.png").exists():
        rgba = np.asarray(Image.open(scene_dir / "car_layer.png").convert("RGBA")).astype(np.float32) / 255
        return rgba[..., :3], rgba[..., 3]
    if (scene_dir / "inpainted.png").exists():
        img = np.asarray(Image.open(scene_dir / "image.png").convert("RGB")).astype(np.float32) / 255
        bg = np.asarray(Image.open(scene_dir / "inpainted.png").convert("RGB")).astype(np.float32) / 255
        if bg.shape != img.shape:
            return None
        d = np.abs(img - bg).max(-1)
        alpha = np.clip((d - 0.02) / 0.10, 0, 1)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        return img, alpha
    return None


def composite_player(frames, keyboard, mouse, layer):
    """Overlay the car layer on each frame; steering yaws/shifts the car a little, plus a road bob."""
    rgb, alpha = layer
    H, W = alpha.shape
    ys, xs = np.nonzero(alpha > 0.5)
    if len(xs) == 0:
        return frames
    pivot = (float(xs.mean()), float(ys.max()))  # rotate about the rear axle contact point
    out = np.empty_like(frames)
    steer, lane = 0.0, 0.0
    for i, f in enumerate(frames):
        k = keyboard[min(i, len(keyboard) - 1)]
        yaw = float(mouse[min(i, len(mouse) - 1)][1])
        steer += (np.clip(yaw / 0.1, -1, 1) - steer) * 0.15
        lane_cmd = (k[3] > 0.5) - (k[2] > 0.5)
        lane += (lane_cmd - lane) * 0.12
        moving = k[0] > 0.5 or k[1] > 0.5
        bob = 1.2 * np.sin(i * 0.9) if moving else 0.0
        M = cv2.getRotationMatrix2D(pivot, -3.0 * steer - 2.0 * lane, 1.0)
        M[0, 2] += 18 * steer + 26 * lane
        M[1, 2] += bob
        rgb_w = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        a_w = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        fr = f.astype(np.float32) / 255
        if fr.shape[:2] != (H, W):
            fr = cv2.resize(fr, (W, H))
        out[i] = np.clip((fr * (1 - a_w) + rgb_w * a_w) * 255, 0, 255).astype(np.uint8)
    return out
