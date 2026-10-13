"""Matrix-Game-2.0 GTA-drive backend.

The GTA-drive checkpoint (Skywork/Matrix-Game-2.0, revision f1729d9, later
removed from the hub) is trained on third-person GTA V driving with a 2-dim
keyboard [W, S] and steering on the mouse-yaw channel. It renders the player
vehicle itself, so the vehicle in the first frame is propagated by the model.

    python -m nycdrive.drive_gta --scene data/inputs/fdr_drive --route highway
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MG2_DIR = ROOT / "third_party" / "mg2_gta" / "Matrix-Game-2"
MG2_FPS = 12  # MG2 output frame rate (utils/visualize.py)

from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.vehicle import route_controls, simulate, to_mg2_gta  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--route", default="cruise", help="preset in nycdrive/vehicle.py or JSON list of [s, throttle, steer]")
    p.add_argument("--latent_frames", type=int, default=99, help="multiple of 3; video frames = 4 * (n - 1) + 1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image", default="image.png", help="input frame inside the scene dir")
    p.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "Matrix-Game-2.0"))
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "videos_gta"))
    p.add_argument("--save_name", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    scene = Path(args.scene).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((scene / "meta.json").read_text()) if (scene / "meta.json").exists() else {}
    label = (args.label or meta.get("label") or scene.name).upper()
    route = json.loads(Path(args.route).read_text()) if args.route.endswith(".json") else args.route
    name = args.save_name or f"{scene.name}_{Path(args.route).stem}"
    ckpt = Path(args.ckpt_dir).resolve()

    num_frames = 4 * (args.latent_frames - 1) + 1
    controls = route_controls(route, num_frames, fps=MG2_FPS)
    sim = simulate(controls, fps=MG2_FPS)
    actions = to_mg2_gta(sim, fps=MG2_FPS)

    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # skip max-autotune compile of the VAE decoder
    os.chdir(MG2_DIR)
    sys.path.insert(0, str(MG2_DIR))
    import inference as mg2

    mg2.Bench_actions_gta_drive = lambda n, **_: {k: v[:n] for k, v in actions.items()}

    def save(video, output_video, config, *_, **__):
        if output_video.endswith("_icon.mp4"):
            return
        frames = np.asarray(video).astype(np.uint8)
        base = out_dir / name
        imageio.mimwrite(f"{base}_world.mp4", frames, fps=MG2_FPS, quality=8, macro_block_size=None)
        # 2x Lanczos upsampling for display (native output is 352x640)
        up = np.stack([cv2.resize(f, (2 * f.shape[1], 2 * f.shape[0]), interpolation=cv2.INTER_LANCZOS4) for f in frames])
        hud = draw_hud(up, sim["controls"], sim["v"], label)
        imageio.mimwrite(f"{base}_hud.mp4", hud, fps=MG2_FPS, quality=8, macro_block_size=None)
        np.savez(f"{base}_actions.npz", controls=sim["controls"], v=sim["v"], psi=sim["psi"],
                 keyboard=actions["keyboard_condition"].numpy(), mouse=actions["mouse_condition"].numpy())
        print(f"[nycdrive] wrote {base}_world.mp4 / _hud.mp4 ({len(frames)} frames @ {MG2_FPS} fps)")

    mg2.process_video = save
    mg2.set_seed(args.seed)
    mg2_args = SimpleNamespace(
        config_path="configs/inference_yaml/inference_gta_drive.yaml",
        checkpoint_path=str(ckpt / "gta_distilled_model" / "gta_keyboard2dim.safetensors"),
        img_path=str(scene / args.image), output_folder=str(out_dir), num_output_frames=args.latent_frames,
        seed=args.seed, pretrained_model_path=str(ckpt))
    mg2.InteractiveGameInference(mg2_args).generate_videos()


if __name__ == "__main__":
    main()
