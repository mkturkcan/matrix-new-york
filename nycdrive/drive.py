"""Drive through New York City with the Matrix-Game-3.0 world model.

Batch mode (scripted route -> video):
    python -m nycdrive.drive --scene data/inputs/fdr_drive --route highway

Interactive mode (type driving commands between generated chunks):
    python -m nycdrive.drive --scene data/inputs/times_square --interactive

Everything after `--` is forwarded verbatim to Matrix-Game-3's generate.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MG3_DIR = ROOT / "third_party" / "Matrix-Game" / "Matrix-Game-3"
sys.path.insert(0, str(MG3_DIR))

from nycdrive.routes import FPS, STEER, build_actions, num_frames_for  # noqa: E402
from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_layer  # noqa: E402


def driving_action():
    """Car-style controls for Matrix-Game-3's interactive mode (replaces its WASD/IJKL prompt)."""
    print("\n" + "-" * 48)
    print(" w = gas   s = brake/reverse   a/d = steer   q = coast")
    print(" z/c = lane change left/right   combine e.g. 'wa', 'wd'")
    print("-" * 48)
    while True:
        cmd = input("drive> ").strip().lower() or "w"
        if set(cmd) <= set("wsadqzc"):
            break
    keyboard = [0] * 6
    yaw = 0.0
    if "w" in cmd or "z" in cmd or "c" in cmd:
        keyboard[0] = 1
    if "s" in cmd:
        keyboard[1] = 1
    if "a" in cmd:
        yaw -= STEER
    if "d" in cmd:
        yaw += STEER
    if "z" in cmd:
        keyboard[2] = 1
    if "c" in cmd:
        keyboard[3] = 1
    return {"mouse": torch.tensor([0.0, yaw]), "keyboard": torch.tensor(keyboard)}


def make_saver(label, top_speed, units, car_layer=None):
    """Replacement for Matrix-Game-3's process_video.

    Writes <name>_world.mp4 (raw world-model output), <name>_clean.mp4 (with the
    player car layer) and <name>_hud.mp4 (car + game HUD).
    """
    def save(input_video, output_video, config, *_, **__):
        keyboard, mouse = config
        frames = np.asarray(input_video).astype(np.uint8)
        base = os.path.splitext(output_video)[0]
        np.savez(base + "_actions.npz", keyboard=keyboard, mouse=mouse)
        imageio.mimwrite(base + "_world.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        if car_layer is not None:
            frames = composite_player(frames, keyboard, mouse, car_layer)
        imageio.mimwrite(base + "_clean.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        hud = draw_hud(frames, keyboard, mouse, label=label, top_speed=top_speed, units=units)
        imageio.mimwrite(base + "_hud.mp4", hud, fps=FPS, quality=8, macro_block_size=None)
        print(f"[nycdrive] wrote {base}_clean.mp4 and {base}_hud.mp4 ({len(frames)} frames)")
    return save


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="dir with image.png, prompt.txt and optional meta.json")
    p.add_argument("--route", default="cruise", help="route preset from nycdrive/routes.py or a JSON file")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=None, help="override prompt.txt")
    p.add_argument("--label", default=None, help="HUD location label")
    p.add_argument("--top_speed", type=float, default=None)
    p.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "Matrix-Game-3.0"))
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "videos"))
    p.add_argument("--save_name", default=None)
    p.add_argument("--no_player_layer", action="store_true", help="do not re-composite the player car on generated frames")
    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    scene = Path(args.scene).resolve()
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.route.endswith(".json"):
        args.route = str(Path(args.route).resolve())
    meta = json.loads((scene / "meta.json").read_text()) if (scene / "meta.json").exists() else {}
    prompt = args.prompt or (scene / "prompt.txt").read_text().strip()
    label = (args.label or meta.get("label") or scene.name.replace("_", " ")).upper()
    is_highway = args.route in ("highway", "weave") or meta.get("type") == "highway"
    top_speed = args.top_speed or (55 if is_highway else 25)
    save_name = args.save_name or f"{scene.name}_{Path(args.route).stem}"

    # Matrix-Game-3 resolves some assets relative to its own directory.
    os.chdir(MG3_DIR)
    import utils.utils as mg_utils
    import pipeline.inference_pipeline as mg_pipe
    import pipeline.inference_interactive_pipeline as mg_ipipe
    import generate as mg_generate

    if args.route.endswith(".json"):
        route = json.loads(Path(args.route).read_text())
    else:
        route = args.route
    # get_data() calls Bench_actions_universal(num_frames) for the action track: swap in our route.
    mg_utils.Bench_actions_universal = lambda num_frames, **_: build_actions(route, num_frames)
    mg_ipipe.get_current_action = driving_action
    car_layer = None if args.no_player_layer else load_car_layer(scene)
    saver = make_saver(label, top_speed, "MPH", car_layer)
    mg_pipe.process_video = saver
    mg_ipipe.process_video = saver

    sys.argv = [
        "generate.py",
        "--size", "704*1280",
        "--ckpt_dir", args.ckpt_dir,
        "--image", str((scene / "image.png").resolve()),
        "--prompt", prompt,
        "--num_iterations", str(args.num_iterations),
        "--num_inference_steps", str(args.steps),
        "--seed", str(args.seed),
        "--save_name", save_name,
        "--output_dir", args.output_dir,
        "--vae_type", "mg_lightvae",
        "--lightvae_pruning_rate", "0.5",
        "--fa_version", "2",
    ] + (["--interactive"] if args.interactive else []) + passthrough
    print(f"[nycdrive] scene={scene} route={args.route} frames={num_frames_for(args.num_iterations)}")
    mg_generate.generate(mg_generate._parse_args())


if __name__ == "__main__":
    main()
