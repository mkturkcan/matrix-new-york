"""Re-apply the player-car layer and HUD to saved world-model rollouts (<name>_world.mp4).

    python scripts/postprocess.py configs/showcase.json
"""
import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_layer  # noqa: E402
from nycdrive.routes import FPS, build_actions  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    for j in json.loads(Path(args.config).read_text()):
        name = j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"
        base = ROOT / "outputs" / "videos" / name
        world = Path(f"{base}_world.mp4")
        if not world.exists():
            print("missing", world)
            continue
        frames = np.stack(imageio.mimread(world, memtest=False))
        if Path(f"{base}_actions.npz").exists():
            a = np.load(f"{base}_actions.npz")
            keyboard, mouse = a["keyboard"], a["mouse"]
        else:
            acts = build_actions(j["route"], len(frames))
            keyboard, mouse = acts["keyboard_condition"].numpy(), acts["mouse_condition"].numpy()
            np.savez(f"{base}_actions.npz", keyboard=keyboard, mouse=mouse)
        scene = ROOT / j["scene"]
        meta = json.loads((scene / "meta.json").read_text()) if (scene / "meta.json").exists() else {}
        layer = load_car_layer(scene)
        if layer is not None:
            frames = composite_player(frames, keyboard, mouse, layer)
        imageio.mimwrite(f"{base}_clean.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        top = j.get("top_speed") or (55 if meta.get("type") == "highway" or j["route"] in ("highway", "weave") else 25)
        label = (j.get("label") or meta.get("label") or scene.name).upper()
        hud = draw_hud(frames, keyboard, mouse, label=label, top_speed=top)
        imageio.mimwrite(f"{base}_hud.mp4", hud, fps=FPS, quality=8, macro_block_size=None)
        print("done", name, len(frames), "frames")


if __name__ == "__main__":
    main()
