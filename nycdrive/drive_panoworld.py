"""PanoWorld backend (360-degree panorama -> panoramic video, 960x480 @ 16 fps).

PanoWorld (Insta360 Research, arXiv 2607.09661) is a Wan2.2-TI2V-5B based
world model that turns a 2:1 equirectangular panorama + text prompt + camera
*translation* trajectory into a panoramic video. Camera heading is fixed (the
panorama stays world-aligned), so the controller only has to move the camera
along the vehicle path; steering is applied afterwards by rendering the
perspective view at the chase-camera heading.

We use the released Causal-Forcing / Rolling-Forcing checkpoint
(`rolling_dmd_480p_161`, 4 denoising steps, trained for 41 latent = 161 frames).
The upstream Action Model (bidirectional, 81 frames, 30 steps + CFG) cannot run
from the public repo: `diffsynth/models/` is missing (upstream issue #1).

Pipeline per (scene, route):
  1. re-fetch the scene's Street View panorama by `pano_id`, rotate it so the
     image center looks along the scene's view heading, mask the nadir (capture
     car / rig) and nearby vehicles (Mask2Former on the equirect) and inpaint
     them with LaMa -> data/panoworld/<scene>/;
  2. integrate the route with the bicycle model (nycdrive/vehicle.py) at 16 fps
     and convert the path (x forward, y right, metres) to PanoWorld's world frame
     (-x forward, +y left, -z up):  t = (-s * x, -s * y, 0), s = --scale units/m
     (default: s such that the route's fastest frame moves MAX_STEP = 0.10 units);
  3. generate the panoramic video with Rolling Forcing (one pass, any multiple of
     4 latent frames) or as chained segments (`--segment_frames`), each segment
     restarting from the previous segment's last frame -- or, with `--anchor`,
     from the next real Street View panorama along the road;
  4. render 1280x704 perspective crops at yaw = psi_cam (pitch -3, HFOV 90),
     composite the player car with its yaw relative to the camera, draw the HUD.

Outputs (outputs/videos_panoworld/<scene>_<route>_*):
  _pano.mp4 (equirect model output), _world.mp4, _clean.mp4, _hud.mp4, _actions.npz

Usage:
  python -m nycdrive.drive_panoworld --scene data/inputs/fdr_drive --route highway [--seconds 20] [--seed 0]
  python -m nycdrive.drive_panoworld --scene data/inputs/fdr_drive --route highway --more times_square:downtown
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PW_DIR = ROOT / "third_party" / "PanoWorld"
CF_DIR = PW_DIR / "Casual-forcing"
MODELS = ROOT / "checkpoints" / "PanoWorld" / "models"
os.environ.setdefault("PANOWORLD_WAN22_DIR", str(MODELS / "Wan-AI" / "Wan2.2-TI2V-5B"))

from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_sprites  # noqa: E402
from nycdrive.vehicle import route_controls, simulate  # noqa: E402

FPS = 16                    # PanoWorld causal output frame rate (inference.py writes 16 fps)
PANO_W, PANO_H = 960, 480   # 480p model; the 720p checkpoints are not released (upstream issue #3)
VIEW_W, VIEW_H = 1280, 704
MAX_STEP = 0.10             # PanoWorld units / frame at the route's top speed (see choose_scale)
NADIR_DEG = -62.0           # rows below this latitude (capture car / rig) are inpainted


# ----------------------------------------------------------------------------- panorama
def pano_prompt(meta: dict) -> str:
    """PanoWorld prompts start with 'This is a panoramic video.'; reuse the scene description."""
    p = meta.get("prompt", "a street in New York City, daytime, realistic.")
    p = p.replace("Third-person view of a car driving", "The camera is mounted on a car driving")
    return "This is a panoramic video. " + p


def yaw_roll(equi: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Re-center an equirect so column W/2 looks yaw_deg to the right of the old center."""
    return np.roll(equi, -int(round(yaw_deg / 360.0 * equi.shape[1])), axis=1)


def vehicle_mask_equirect(seg: np.ndarray, near_deg: float = -6.0, dilate_px: int = 9) -> np.ndarray:
    """Vehicles whose lowest pixel is more than |near_deg| below the horizon (i.e. within
    ~25 m of a 2.5 m high camera) or that are large; far traffic is kept."""
    from nycdrive.prepare_inputs import CS_VEHICLES
    H, W = seg.shape
    veh = np.isin(seg, CS_VEHICLES).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(veh, connectivity=8)
    keep = np.zeros_like(veh)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        bottom_lat = 90.0 - (y + h) / H * 180.0
        if area >= 20 and (bottom_lat < near_deg or area > 0.004 * H * W):
            keep[lab == i] = 1
    if dilate_px:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return keep * 255


def clean_equirect(equi: np.ndarray, segmenter, inpainter) -> tuple:
    """Mask nadir + no-data + nearby vehicles and inpaint. Works at 2048x1024; the
    horizontal wrap is handled by padding a quarter turn on each side."""
    from nycdrive.prepare_inputs import nodata_mask
    img = cv2.resize(equi, (2048, 1024), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    pad = W // 4
    wide = np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)
    seg = segmenter(wide)[:, pad:pad + W]
    mask = np.maximum(vehicle_mask_equirect(seg), nodata_mask(img))
    mask[int((90 - NADIR_DEG) / 180 * H):] = 255
    m_wide = np.concatenate([mask[:, -pad:], mask, mask[:, :pad]], 1)
    out = inpainter(wide, m_wide)[:, pad:pad + W]
    return out, mask


def prepare_panorama(scene: Path, meta: dict, pano_id: str = None, tag: str = "", zoom: int = 3,
                     tools=None) -> np.ndarray:
    """Fetch + clean (cached) the panorama; returns the cleaned 960x480 equirect, centred on the
    scene's view heading. `pano_id` overrides the scene panorama (multi-panorama anchoring)."""
    cache = ROOT / "data" / "panoworld" / scene.name
    cache.mkdir(parents=True, exist_ok=True)
    pano_id = pano_id or meta["pano_id"]
    clean_path = cache / f"equirect{tag}_clean.png"
    if clean_path.exists():
        return np.asarray(Image.open(clean_path).convert("RGB"))
    from streetlevel import streetview
    from nycdrive.prepare_inputs import fetch_equirect
    pano = streetview.find_panorama_by_id(pano_id)
    equi = fetch_equirect(pano, zoom)
    pano_heading = math.degrees(pano.heading or 0.0) % 360
    equi = yaw_roll(equi, meta["view_heading_deg"] - pano_heading)
    segmenter, inpainter = tools if tools is not None else make_cleaning_tools()
    clean, mask = clean_equirect(equi, segmenter, inpainter)
    Image.fromarray(cv2.resize(equi, (2048, 1024), interpolation=cv2.INTER_AREA)).save(cache / f"equirect{tag}_raw.jpg", quality=92)
    Image.fromarray(mask).save(cache / f"equirect{tag}_mask.png")
    Image.fromarray(cv2.resize(clean, (PANO_W, PANO_H), interpolation=cv2.INTER_AREA)).save(clean_path)
    json.dump({"pano_id": pano.id, "lat": pano.lat, "lon": pano.lon, "date": str(pano.date),
               "pano_heading_deg": pano_heading, "view_heading_deg": meta["view_heading_deg"]},
              open(cache / f"equirect{tag}.json", "w"), indent=2)
    return np.asarray(Image.open(clean_path).convert("RGB"))


def make_cleaning_tools():
    from nycdrive.prepare_inputs import Inpainter, Segmenter
    return Segmenter(), Inpainter()


# ----------------------------------------------------------------------------- trajectory
def choose_scale(sim: dict, max_step: float = MAX_STEP) -> float:
    """Metres -> PanoWorld units. The authors recommend ~0.05 units / frame (their drone data);
    on Street View panoramas 0.05 / frame reads as walking pace, 0.1-0.2 / frame as slow driving,
    and >= 0.4 / frame the video dissolves within ~2 s (sweep on fdr_drive / times_square).
    Apparent speed saturates while radial streaking grows with the step, so 0.10 is the default.
    The route's top speed is mapped to max_step and slower frames scale proportionally."""
    return max_step / max(float(np.hypot(np.diff(sim["x"]), np.diff(sim["y"])).max()), 1e-3)


def model_translation(sim: dict, scale: float) -> np.ndarray:
    """Vehicle path (x forward, y right; m) -> PanoWorld world frame (-x forward, +y left, -z up).

    Positive steer turns right in nycdrive.vehicle (psi grows clockwise, y grows to the right),
    so both axes flip sign. Heading never enters: PanoWorld keeps the panorama world-aligned."""
    return np.stack([-scale * sim["x"], -scale * sim["y"], np.zeros_like(sim["x"])], 1).astype(np.float32)


def viewmats(rel_xyz_latent: np.ndarray, device) -> dict:
    """(L, 3) per-latent-frame translations -> PanoWorld relray conditioning."""
    from action_control import control_model
    L = len(rel_xyz_latent)
    c2w = torch.eye(4, device=device).repeat(1, L, 1, 1)
    c2w[0, :, :3, 3] = torch.from_numpy(rel_xyz_latent).to(device)
    return {"viewmats": control_model.viewmats_panorama(pose=c2w, height=PANO_H // 32, width=PANO_W // 32,
                                                        device=device, dtype=torch.float32)}


# ----------------------------------------------------------------------------- model
class PanoWorldCausal:
    """Rolling-Forcing PanoWorld (Casual-forcing/inference.py, in-process)."""

    def __init__(self, device="cuda", ckpt="rolling_dmd_480p_161"):
        for p in (str(PW_DIR), str(CF_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline
        from utils.wan_wrapper import WanDiffusionWrapper, WanVAE22Wrapper

        self.device = torch.device(device)
        cfg = OmegaConf.merge(OmegaConf.load(CF_DIR / "configs" / "default_config.yaml"),
                              OmegaConf.load(CF_DIR / "configs" / "causal_forcing_rolling_dmd_480p_161.yaml"))
        self.cfg = cfg
        # the wrapper loads the base LoRA + action module before the full generator checkpoint
        # (its defaults point at the authors' NFS); the checkpoint then overwrites everything.
        gen = WanDiffusionWrapper(**cfg.model_kwargs, is_causal=True,
                                  lora_path=str(MODELS / "lora/480p/480p_lora.safetensors"),
                                  action_path=str(MODELS / "action/480p/480p_action.safetensors"))
        self.pipe = CausalInferencePipeline(cfg, device=self.device, generator=gen, vae=WanVAE22Wrapper())
        sd = torch.load(MODELS / "casual" / ckpt / "model.pt", map_location="cpu", mmap=True)
        key = next(k for k in ("generator_ema", "generator", "model") if k in sd)
        gsd = sd[key]["shadow"] if "shadow" in sd[key] else sd[key]   # EMA: {shadow, decay, is_fsdp}
        gsd = {k.replace("model._fsdp_wrapped_module.", "model.", 1).replace("._fsdp_wrapped_module.", ".")
               .removeprefix("_fsdp_wrapped_module.").removeprefix("module."): v for k, v in gsd.items()}
        missing, unexpected = self.pipe.generator.load_state_dict(gsd, strict=False)
        print(f"[panoworld] loaded {ckpt}:{key} missing={len(missing)} unexpected={len(unexpected)}")
        del sd, gsd
        self.pipe = self.pipe.to(dtype=torch.bfloat16)
        self.pipe.text_encoder.to(self.device)
        self.pipe.generator.to(self.device)
        self.pipe.vae.to(self.device, dtype=torch.float32)
        self.frame_seq_length = cfg.frame_seq_length

    @torch.no_grad()
    def generate(self, pano: np.ndarray, prompt: str, rel_xyz: np.ndarray, seed: int = 0) -> np.ndarray:
        """pano (480,960,3) uint8; rel_xyz (T,3) per pixel frame with T = 1 + 4k, rel_xyz[0] = 0.
        Returns (T, 480, 960, 3) uint8."""
        T = len(rel_xyz)
        L = (T - 1) // 4 + 1
        assert (T - 1) % 4 == 0 and (L - 1) % self.cfg.num_frame_per_block == 0, T
        torch.manual_seed(seed)
        p, g = self.pipe, self.pipe.generator
        p.frame_seq_length = g.frame_seq_length = self.frame_seq_length
        g.seq_len = self.frame_seq_length * L
        if hasattr(g.model, "configure_frame_seqlen"):
            g.model.configure_frame_seqlen(self.frame_seq_length, L)
        p.kv_cache1 = None   # caches are sized from the sequence length; rebuild per call
        img = torch.from_numpy(np.ascontiguousarray(pano).copy()).float().div(255).permute(2, 0, 1)[None, :, None]
        img = (img * 2 - 1).to(self.device, torch.bfloat16)
        init = p.vae.encode_to_latent(img).to(self.device, torch.bfloat16)
        _, _, C, h, w = init.shape
        noise = torch.randn([1, L - 1, C, h, w], device=self.device, dtype=torch.bfloat16)
        ctrl = viewmats(rel_xyz[::4][:L], self.device)
        video, _ = p.inference_rolling_forcing(noise=noise, text_prompts=[prompt], return_latents=True,
                                               initial_latent=init, control_camera_dit_input=ctrl)
        if hasattr(p.vae, "model") and hasattr(p.vae.model, "clear_cache"):
            p.vae.model.clear_cache()
        return (video[0].permute(0, 2, 3, 1).float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def generate_long(model, pano, prompt, rel_xyz, seed=0, segment_frames=None, anchors=None):
    """Generate len(rel_xyz) frames. With segment_frames=S (1 + 4k), chain segments that each start
    from the previous last frame (or from anchors[i] = cleaned real panorama for segment i>0,
    re-centred at the position where that panorama was taken)."""
    T = len(rel_xyz)
    S = segment_frames or T
    frames, start, i = [], 0, 0
    while start < T - 1:
        n = min(S, T - start)
        n = 1 + -(-(n - 1) // 16) * 16       # 1 + 4 * (4-latent-frame blocks); extra frames are dropped
        idx = np.minimum(np.arange(start, start + n), T - 1)
        seg_xyz = rel_xyz[idx] - rel_xyz[start]
        if i > 0 and anchors is not None and anchors[i] is not None:
            first = anchors[i]
        else:
            first = pano if i == 0 else frames[-1][-1]
        print(f"[panoworld] segment {i}: frames {start}..{start + n - 1} ({n})")
        out = model.generate(first, prompt, seg_xyz, seed=seed + i)
        out = out[: min(n, T - start)]
        frames.append(out if i == 0 else out[1:])
        start += len(out) - 1
        i += 1
    return np.concatenate(frames)[:T]


# ----------------------------------------------------------------------------- rendering
def perspective(frames: np.ndarray, yaw_deg: np.ndarray, pitch=-3.0, hfov=90.0) -> np.ndarray:
    from nycdrive.prepare_inputs import equirect_to_perspective
    return np.stack([equirect_to_perspective(f, float(y), pitch, hfov, VIEW_W, VIEW_H)
                     for f, y in zip(frames, yaw_deg)])


def write_outputs(base: str, frames, sim, rel_xyz, label, sprites, scale, extra=None):
    imageio.mimwrite(base + "_pano.mp4", frames, fps=FPS, quality=8, macro_block_size=None)
    psi_cam_deg = np.rad2deg(sim["psi_cam"][: len(frames)])
    rel_yaw = np.rad2deg(sim["rel_yaw"][: len(frames)])
    world = perspective(frames, psi_cam_deg)
    imageio.mimwrite(base + "_world.mp4", world, fps=FPS, quality=8, macro_block_size=None)
    clean = composite_player(world, rel_yaw, sprites, moving=sim["v"] > 0.5) if sprites is not None else world
    imageio.mimwrite(base + "_clean.mp4", clean, fps=FPS, quality=8, macro_block_size=None)
    controls = sim["controls"][: len(frames)]
    imageio.mimwrite(base + "_hud.mp4", draw_hud(clean, controls, None, label), fps=FPS, quality=8,
                     macro_block_size=None)
    np.savez(base + "_actions.npz", controls=controls, stride=1, fps=FPS, scale=scale,
             v=sim["v"], psi=sim["psi"], psi_cam=sim["psi_cam"], x=sim["x"], y=sim["y"],
             rel_yaw=rel_yaw, translation=rel_xyz, crop_yaw_deg=psi_cam_deg, **(extra or {}))
    print(f"[nycdrive] wrote {base}_{{pano,world,clean,hud}}.mp4 ({len(frames)} frames @ {FPS} fps)")


# ----------------------------------------------------------------------------- anchoring
def anchor_panoramas(scene: Path, meta: dict, sim: dict, T: int, every: int, tools) -> list:
    """For segment starts k*every, pick the real Street View panorama closest to where the vehicle
    is then (projected along the initial heading), clean it and return it, else None."""
    from streetlevel import streetview
    p0 = streetview.find_panorama_by_id(meta["pano_id"])
    h = math.radians(meta["view_heading_deg"])
    cands = [(n.id, n.lat, n.lon) for n in (p0.neighbors or [])]
    R = 6371000.0
    out = [None]
    used = {meta["pano_id"]}
    for k in range(every, T - 1, every):
        x, y = sim["x"][k], sim["y"][k]             # forward / right of the initial view heading
        east, north = x * math.sin(h) + y * math.cos(h), x * math.cos(h) - y * math.sin(h)
        lat = meta["lat"] + math.degrees(north / R)
        lon = meta["lon"] + math.degrees(east / (R * math.cos(math.radians(meta["lat"]))))
        def dist(c):
            return math.hypot((c[1] - lat) * R * math.pi / 180,
                              (c[2] - lon) * R * math.pi / 180 * math.cos(math.radians(lat)))
        near = [c for c in sorted(cands, key=dist) if dist(c) <= 8.0 and c[0] not in used][:6]
        if not near:
            print(f"[anchor] frame {k}: no unused pano within 8 m -> no anchor")
            out.append(None)
            continue
        # prefer a panorama from the same capture as the scene's (same light / season / traffic)
        dates = [str(streetview.find_panorama_by_id(c[0]).date) for c in near]
        j = next((i for i, d in enumerate(dates) if d == str(meta.get("date"))), 0)
        best = near[j]
        used.add(best[0])
        print(f"[anchor] frame {k}: pano {best[0]} ({dates[j]}) at {dist(best):.1f} m from the vehicle")
        out.append(prepare_panorama(scene, meta, best[0], tag=f"_{best[0]}", tools=tools))
    return out


# ----------------------------------------------------------------------------- main
def run(model, scene: Path, route: str, args, tools):
    meta = json.loads((scene / "meta.json").read_text())
    label = (meta.get("label") or scene.name).upper()
    T = 1 + 16 * int(round((args.seconds * FPS - 1) / 16))   # 1 + 4k latent frames, k % 4 == 0
    sim = simulate(route_controls(route, T, fps=FPS), fps=FPS)
    scale = args.scale or choose_scale(sim)
    rel_xyz = model_translation(sim, scale)
    pano = prepare_panorama(scene, meta, tools=tools)
    anchors = None
    if args.anchor:
        anchors = anchor_panoramas(scene, meta, sim, T, args.segment_frames - 1, tools)
    prompt = args.prompt or pano_prompt(meta)
    print(f"[nycdrive] scene={scene.name} route={route} frames={T} ({T / FPS:.1f} s) "
          f"path={sim['x'][-1]:.0f} m fwd / {sim['y'][-1]:.1f} m right, scale {scale:.3f} units/m")
    frames = generate_long(model, pano, prompt, rel_xyz, seed=args.seed,
                           segment_frames=args.segment_frames, anchors=anchors)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.save_name or f"{scene.name}_{Path(route).stem}"
    sprites = None if args.no_player else load_car_sprites(scene)
    write_outputs(str(out / name), frames, sim, rel_xyz, label, sprites, scale,
                  extra={"segment_frames": args.segment_frames or 0, "anchored": bool(args.anchor),
                         "prompt": prompt})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--route", default="cruise", help="preset in nycdrive/vehicle.py")
    p.add_argument("--more", nargs="*", default=[], help="extra scene:route jobs run with the same loaded model")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=None,
                   help="PanoWorld units per metre of vehicle travel (default: top speed -> 0.10 units / frame)")
    p.add_argument("--segment_frames", type=int, default=161,
                   help="frames per generated segment (161 = trained Rolling-Forcing length); 0 = one pass")
    p.add_argument("--anchor", action="store_true", help="restart each segment from the next real panorama")
    p.add_argument("--prompt", default=None)
    p.add_argument("--no_player", action="store_true")
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "videos_panoworld"))
    p.add_argument("--save_name", default=None)
    args = p.parse_args()
    args.segment_frames = args.segment_frames or None

    jobs = [(Path(args.scene).resolve(), args.route)]
    for j in args.more:
        s, r = j.split(":")
        jobs.append(((ROOT / "data" / "inputs" / s).resolve(), r))
    tools = make_cleaning_tools()
    for scene, _ in jobs:       # clean all panoramas first, then free the segmenter / LaMa
        prepare_panorama(scene, json.loads((scene / "meta.json").read_text()), tools=tools)
    if not args.anchor:
        del tools
        tools = None
        torch.cuda.empty_cache()
    model = PanoWorldCausal()
    for scene, route in jobs:
        if len(jobs) > 1:
            args.save_name = None
        run(model, scene, route, args, tools)


if __name__ == "__main__":
    main()
