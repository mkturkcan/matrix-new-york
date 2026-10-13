"""Matrix-Game-3.0 backend (first-person, 704x1280).

MG3 is conditioned on the vehicle-free frame (inpainted.png). Driver inputs are
integrated by the kinematic bicycle model in nycdrive/vehicle.py; the chase
camera's yaw rate drives MG3's mouse-yaw channel and the player vehicle is drawn
on each generated frame with its yaw relative to the camera.

Batch:        python -m nycdrive.drive --scene data/inputs/fdr_drive --route highway
Interactive:  python -m nycdrive.drive --scene data/inputs/times_square --interactive

Arguments after `--` are forwarded to Matrix-Game-3's generate.py.
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

from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_sprites  # noqa: E402
from nycdrive.vehicle import (FPS, MG3_YAW_DEG_PER_UNIT, VehicleParams, route_controls,  # noqa: E402
                              simulate, to_mg3)

def num_frames_for(num_iterations: int) -> int:
    return 57 + (num_iterations - 1) * 40


def rel_yaw_from_mouse(mouse: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Vehicle yaw relative to the chase camera (deg). With psi_cam[t+1] - psi_cam[t] = alpha * (psi - psi_cam),
    psi - psi_cam = (per-frame camera yaw) / alpha; the per-frame yaw is the (low-passed) MG3 yaw command."""
    alpha = 1.0 - np.exp(-1.0 / (fps * VehicleParams.cam_lag))
    rate = mouse[:, 1] * MG3_YAW_DEG_PER_UNIT
    k = 1.0 - np.exp(-1.0 / (0.2 * fps))
    smooth = np.zeros_like(rate)
    for i in range(1, len(rate)):
        smooth[i] = smooth[i - 1] + k * (rate[i] - smooth[i - 1])
    return smooth / alpha


def driving_action():
    """Interactive controls: throttle on W, steering on the yaw channel (no lateral translation)."""
    print("\n w = throttle   s = brake   a / d = steer   q = coast   (combine, e.g. 'wa')")
    while True:
        cmd = input("drive> ").strip().lower() or "w"
        if set(cmd) <= set("wsadq"):
            break
    keyboard = [0] * 6
    keyboard[0] = int("w" in cmd)
    keyboard[1] = int("s" in cmd)
    yaw = 0.06 * (("d" in cmd) - ("a" in cmd))
    return {"mouse": torch.tensor([0.0, yaw]), "keyboard": torch.tensor(keyboard)}


def make_saver(label, sprites, sim=None, stride=1):
    """Replacement for MG3's process_video. With stride k, every k-th generated frame is kept
    (the controller runs at k * FPS, so on-screen timing of manoeuvres is preserved)."""
    def save(input_video, output_video, config, *_, **__):
        keyboard, mouse = (np.asarray(c) for c in config)
        frames = np.asarray(input_video).astype(np.uint8)
        base = os.path.splitext(output_video)[0]
        rel_yaw = rel_yaw_from_mouse(mouse, fps=FPS * stride)
        np.savez(base + "_actions.npz", keyboard=keyboard, mouse=mouse, rel_yaw=rel_yaw, stride=stride,
                 controls=sim["controls"] if sim is not None else np.zeros((0, 2)))
        frames, keyboard, rel_yaw = frames[::stride], keyboard[::stride], rel_yaw[::stride]
        imageio.mimwrite(base + "_world.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        if sprites is not None:
            frames = composite_player(frames, rel_yaw, sprites, moving=keyboard[:, 0] > 0.5)
        imageio.mimwrite(base + "_clean.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
        if sim is not None:
            controls = sim["controls"][::stride]
        else:  # interactive: reconstruct inputs from the issued actions
            controls = np.stack([keyboard[:, 0] - keyboard[:, 1], np.clip(mouse[:, 1] / 0.06, -1, 1)], 1)
        # no speed readout: MG3's realised speed is fixed by the model, not by the controller
        hud = draw_hud(frames, controls, None, label)
        imageio.mimwrite(base + "_hud.mp4", hud, fps=FPS, quality=8, macro_block_size=None)
        print(f"[nycdrive] wrote {base}_{{world,clean,hud}}.mp4 ({len(frames)} frames)")
    return save


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--route", default="cruise", help="preset in nycdrive/vehicle.py or JSON list of [s, throttle, steer]")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--condition", default="inpainted.png", help="conditioning frame in the scene dir")
    p.add_argument("--no_player", action="store_true")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="keep every k-th generated frame (k x apparent speed); controller runs at k * 17 Hz")
    p.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "Matrix-Game-3.0"))
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "videos"))
    p.add_argument("--save_name", default=None)
    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    scene = Path(args.scene).resolve()
    output_dir = str(Path(args.output_dir).resolve())
    route = json.loads(Path(args.route).read_text()) if args.route.endswith(".json") else args.route
    meta = json.loads((scene / "meta.json").read_text()) if (scene / "meta.json").exists() else {}
    prompt = args.prompt or (scene / "prompt.txt").read_text().strip()
    prompt = prompt.replace("Third-person view of a car driving", "First-person view driving")
    label = (args.label or meta.get("label") or scene.name).upper()
    save_name = args.save_name or f"{scene.name}_{Path(args.route).stem}"
    sprites = None if args.no_player else load_car_sprites(scene)
    cond = scene / args.condition if (scene / args.condition).exists() else scene / "image.png"

    n = num_frames_for(args.num_iterations)
    k = 1 if args.interactive else args.frame_stride
    sim = None if args.interactive else simulate(route_controls(route, n, fps=FPS * k), fps=FPS * k)

    os.chdir(MG3_DIR)
    import utils.utils as mg_utils
    import pipeline.inference_pipeline as mg_pipe
    import pipeline.inference_interactive_pipeline as mg_ipipe
    import generate as mg_generate

    if sim is not None:
        actions = to_mg3(sim, fps=FPS * k)
        mg_utils.Bench_actions_universal = lambda num_frames, **_: {k: v[:num_frames] for k, v in actions.items()}
    mg_ipipe.get_current_action = driving_action
    saver = make_saver(label, sprites, sim, stride=k)
    mg_pipe.process_video = saver
    mg_ipipe.process_video = saver

    sys.argv = [
        "generate.py", "--size", "704*1280", "--ckpt_dir", args.ckpt_dir, "--image", str(cond),
        "--prompt", prompt, "--num_iterations", str(args.num_iterations),
        "--num_inference_steps", str(args.steps), "--seed", str(args.seed), "--save_name", save_name,
        "--output_dir", output_dir, "--vae_type", "mg_lightvae", "--lightvae_pruning_rate", "0.5",
        "--fa_version", "2",
    ] + (["--interactive"] if args.interactive else []) + passthrough
    print(f"[nycdrive] scene={scene.name} route={args.route} frames={n} condition={cond.name}")
    mg_generate.generate(mg_generate._parse_args())


if __name__ == "__main__":
    main()
