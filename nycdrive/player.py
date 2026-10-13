"""Player-vehicle layer for the first-person backend (Matrix-Game-3.0).

MG3 is conditioned on the vehicle-free frame (inpainted.png); the vehicle is
drawn on every generated frame from a bank of renders indexed by its yaw
relative to the chase camera (car_bank.npz), as produced by the vehicle model
in nycdrive/vehicle.py. Without a bank, the static car_layer.png is used.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def load_car_sprites(scene_dir: Path):
    """Return (yaw_deg (K,), rgba float32 (K,H,W,4)) or None."""
    scene_dir = Path(scene_dir)
    if (scene_dir / "car_bank.npz").exists():
        d = np.load(scene_dir / "car_bank.npz")
        return d["yaw_deg"].astype(np.float32), d["rgba"]
    if (scene_dir / "car_layer.png").exists():
        rgba = np.asarray(Image.open(scene_dir / "car_layer.png").convert("RGBA"))
        return np.zeros(1, np.float32), rgba[None]
    return None


def composite_player(frames, rel_yaw_deg, sprites, moving=None):
    """Alpha-composite the sprite whose yaw is closest to rel_yaw_deg[t] onto frames[t]."""
    yaws, rgba = sprites
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        k = int(np.argmin(np.abs(yaws - rel_yaw_deg[min(t, len(rel_yaw_deg) - 1)])))
        s = rgba[k]
        if s.shape[:2] != f.shape[:2]:
            s = np.asarray(Image.fromarray(s).resize((f.shape[1], f.shape[0]), Image.LANCZOS))
        a = s[..., 3:4].astype(np.float32) / 255
        # sub-pixel suspension bob while moving
        dy = 1 if (moving is not None and moving[min(t, len(moving) - 1)] and t % 6 < 3) else 0
        if dy:
            s = np.roll(s, dy, axis=0)
            a = np.roll(a, dy, axis=0)
        out[t] = (f.astype(np.float32) * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)
    return out
