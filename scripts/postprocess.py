"""Re-draw the player vehicle and HUD on saved Matrix-Game-3.0 rollouts (<name>_world.mp4 + _actions.npz).

    python scripts/postprocess.py configs/comparison.json
"""
import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nycdrive.drive import rel_yaw_from_mouse  # noqa: E402
from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_sprites  # noqa: E402
from nycdrive.vehicle import FPS, route_controls  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    for j in json.loads(Path(args.config).read_text()):
        name = j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"
        base = ROOT / "outputs" / "videos" / name
        if not Path(f"{base}_world.mp4").exists():
            print("missing", base)
            continue
        frames = np.stack(imageio.mimread(f"{base}_world.mp4", memtest=False))
        a = np.load(f"{base}_actions.npz")
        keyboard, mouse = a["keyboard"], a["mouse"]
        sprites = load_car_sprites(ROOT / j["scene"])
        frames = composite_player(frames, rel_yaw_from_mouse(mouse), sprites, moving=keyboard[:, 0] > 0.5)
        imageio.mimwrite(f"{base}_clean.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        controls = route_controls(j["route"], len(frames), fps=FPS)
        label = (j.get("label") or Path(j["scene"]).name).upper()
        imageio.mimwrite(f"{base}_hud.mp4", draw_hud(frames, controls, None, label), fps=FPS, quality=8, macro_block_size=None)
        print("done", name)


if __name__ == "__main__":
    main()
