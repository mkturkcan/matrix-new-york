"""Matrix-Game-3.5 backend (first-person, 704x1280, camera-trajectory controlled).

MG3.5 (Riemann Dynamics; Wan2.2-TI2V-5B + Patch Memory + PRoPE) takes an anchor
image, a text prompt and an explicit camera trajectory (c2w, metric units) rather
than keyboard/mouse actions. We therefore integrate the driver inputs with the
same kinematic bicycle model as the MG3 backend (nycdrive/vehicle.py) and hand
MG3.5 the chase-camera trajectory directly:

* camera position = vehicle position minus ``--cam_back`` metres along the
  chase-camera heading psi_cam (the camera trails the car), constant height;
* camera yaw = psi_cam, constant pitch = meta.json view_pitch_deg;
* OpenCV convention (x right, y down, z forward; positive yaw about +y turns
  right, matching psi), poses relative to the first frame (identity),
  translations in metres times ``--scale`` (default 1: MG3.5 is metric, its
  Patch Memory lifts frames to 3D with Depth-Anything-3 metric depth);
* pinhole intrinsics from view_hfov_deg (90 deg -> fx = fy = 640 px at 1280 px).

Road following. MG3.5 moves the camera exactly along the commanded path and,
unlike MG3, does not steer along the road by itself, so an open-loop route
leaves the road wherever the real road bends (Times Square: ~3 deg off the view
heading -> on the sidewalk after ~150 m). By default the route is driven
relative to the Street View capture path ahead of the anchor panorama (chained
pano links = a lane centerline, cached in outputs/mg35_work/road_<scene>.json):
s = distance along the road, d = lateral offset. Route segments with
|steer| < 0.2 (MG3 presets use them to follow gentle curves) are zeroed, since
the curvature now comes from the real road; lane changes and turns are kept.
``--no_road`` / ``--keep_small_steer`` restore the open-loop behaviour.

We use the released *distilled* three-step causal checkpoint (infer_distilled.py),
the practical path; the bidirectional base model needs 25 denoising steps + CFG (vs 3) and was not used.
MG3.5 writes 16 fps video; each block is 21 latents = 84 frames and consumes 84
poses, so a B-block run yields 1 + 84 * B frames with frame i rendered at pose i.
Measured on an RTX PRO 6000 (shared): ~45 GB VRAM, ~1.5 s per generated frame
with memory on (~8 min for a 21 s clip at stride 1, incl. ~1 min model load).

With ``--frame_stride k`` the vehicle is simulated at 16 * k Hz (k x denser poses,
so smaller per-frame motion for the model) and every k-th frame is kept for
display at 16 fps, which preserves real-time speed and manoeuvre timing. k = 1
(the default) already gives up to ~1.3 m/frame at highway speed, which MG3.5
renders as plausible fast forward motion; k = 2 doubles the cost.

As with MG3, generation is conditioned on the vehicle-free frame (inpainted.png)
and the player car is composited afterwards from car_bank.npz using its yaw
relative to the chase camera.

Run inside .venv-mg35 (torch 2.12 cu130 from the base env + third_party/Matrix-Game-3.5 deps):
  CUDA_VISIBLE_DEVICES=0 .venv-mg35/bin/python -m nycdrive.drive_mg35 \
      --scene data/inputs/fdr_drive --route highway --seconds 21
Writes outputs/videos_mg35/<scene>_<route>_{world,clean,hud}.mp4 and _actions.npz
(controls at the generation rate, stride, fps, c2w, intrinsics, vehicle state).
Weights: checkpoints/mg35/{Matrix-Game-3.5-Distilled, Wan2.2-TI2V-5B, DA3NESTED-GIANT-LARGE-1.1}.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MG35_DIR = ROOT / "third_party" / "Matrix-Game-3.5"
CKPT = ROOT / "checkpoints" / "mg35"

from nycdrive.hud import draw_hud  # noqa: E402
from nycdrive.player import composite_player, load_car_sprites  # noqa: E402
from nycdrive.vehicle import ROUTES, route_controls, simulate  # noqa: E402

FPS_OUT = 16            # MG3.5 output frame rate (video_io._write_video default; training clips)
POSES_PER_BLOCK = 84    # 21 latents x 4 frames per causal block
WIDTH, HEIGHT = 1280, 704


# --------------------------------------------------------------------------- poses
def _rot_y(a):
    """Rotation about the camera/world y axis (pointing down): positive a turns +z towards +x (right)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    """Rotation about x (pointing right): negative a tilts +z towards +y, i.e. pitches the view down."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def chase_camera_c2w(sim, cam_back: float = 6.0, pitch_deg: float = -3.0, scale: float = 1.0,
                     prepend_rest: bool = True) -> np.ndarray:
    """Chase-camera trajectory of a vehicle simulation as OpenCV c2w matrices (N, 4, 4).

    Vehicle frame of nycdrive.vehicle: x forward, y right (positive psi = right turn), metres.
    MG3.5 / OpenCV world: X right (= y), Y down (height constant), Z forward (= x).
    The camera sits ``cam_back`` m behind the car along psi_cam, looks along psi_cam and is
    pitched by ``pitch_deg``. Poses are expressed relative to the first one (identity) and the
    translation is multiplied by ``scale`` (model units per metre).
    With ``prepend_rest`` the first state is duplicated as pose 0, the pose of the anchor image.
    """
    x, y, pc = sim["x"], sim["y"], sim["psi_cam"]
    if prepend_rest:                                     # anchor = first state, car not yet moving
        x, y, pc = (np.concatenate([a[:1], a]) for a in (x, y, pc))
    cx = x - cam_back * np.cos(pc)
    cy = y - cam_back * np.sin(pc)
    pitch = _rot_x(np.deg2rad(pitch_deg))
    E = np.tile(np.eye(4), (len(x), 1, 1))
    for i in range(len(x)):
        E[i, :3, :3] = _rot_y(pc[i]) @ pitch
        E[i, :3, 3] = (cy[i], 0.0, cx[i])
    E = np.linalg.inv(E[0]) @ E                     # relative to the anchor camera
    E[:, :3, 3] *= scale
    return E.astype(np.float32)


# --------------------------------------------------------------------------- road following
# MG3.5 moves the camera exactly along the commanded path; unlike MG3 it does not steer along
# the road by itself. An open-loop route therefore leaves the road wherever the real road bends
# (FDR Drive hits the river fence after ~30 m). We recover the road from Street View: chaining
# each panorama's forward link gives the capture car's path, i.e. a lane centerline, and the
# route is driven *relative* to it (s = distance along the road, d = lateral offset).
def road_centerline(pano_id: str, heading_deg: float, length_m: float = 600.0, cache: Path = None):
    """Polyline (N, 2) of the Street View capture path ahead of `pano_id`, in the vehicle frame
    (x forward along `heading_deg`, y right, metres; the anchor pano at the origin)."""
    if cache is not None and cache.exists():
        return np.asarray(json.loads(cache.read_text())["xy"], dtype=np.float64)
    from streetlevel import streetview
    p = streetview.find_panorama_by_id(pano_id)
    lat0, lon0, h = p.lat, p.lon, math.radians(heading_deg)

    def local(lat, lon):
        e = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
        n = (lat - lat0) * 110540.0
        return e * math.sin(h) + n * math.cos(h), e * math.cos(h) - n * math.sin(h)

    pts, seen, bearing, total = [(0.0, 0.0)], {p.id}, heading_deg, 0.0
    while total < length_m and len(pts) < 150:
        links = [l for l in (p.links or []) if l.pano.id not in seen]
        if not links:
            break
        diff = lambda l: abs((math.degrees(l.direction) - bearing + 180) % 360 - 180)  # noqa: E731
        link = min(links, key=diff)
        if diff(link) > 50:
            break
        q = link.pano
        xy = local(q.lat, q.lon)
        step = math.hypot(xy[0] - pts[-1][0], xy[1] - pts[-1][1])
        if step > 0.5:
            bearing = (math.degrees(math.atan2(xy[1] - pts[-1][1], xy[0] - pts[-1][0])) + heading_deg) % 360
            pts.append(xy)
            total += step
        seen.add(q.id)
        try:
            p = streetview.find_panorama_by_id(q.id)
        except Exception:
            break
        if p is None:
            break
    xy = np.asarray(pts)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"pano_id": pano_id, "heading_deg": heading_deg, "xy": xy.tolist()}))
    return xy


def _road_frame(xy: np.ndarray, smooth_m: float = 25.0):
    """Arc-length parametrisation of a polyline: s grid (1 m), positions P(s), headings theta(s).
    Extended straight beyond both ends."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.arange(-50.0, s[-1] + 2000.0, 1.0)
    inside = np.clip(grid, 0, s[-1])
    P = np.stack([np.interp(inside, s, xy[:, 0]), np.interp(inside, s, xy[:, 1])], 1)
    k = int(smooth_m)                                    # moving-average smoothing of GPS jitter
    if k > 1 and len(P) > k:
        pad = np.pad(P, ((k, k), (0, 0)), mode="edge")
        P = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="same")[k:-k] for i in range(2)], 1)
    d = np.gradient(P, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # straight extrapolation past the ends of the captured path
    for mask, ref in ((grid > s[-1], np.searchsorted(grid, s[-1])), (grid < 0, np.searchsorted(grid, 0.0))):
        ref = min(ref, len(grid) - 1)
        theta[mask] = theta[ref]
        P[mask] = P[ref] + (grid[mask] - grid[ref])[:, None] * np.stack([np.cos(theta[ref]), np.sin(theta[ref])])
    return grid, P, theta


def washout(sim, fps: float, tau: float = 3.0, cam_lag: float = 0.35):
    """Road-relative trajectory of a simulation with sustained heading washed out (time constant
    `tau`), so the car keeps returning parallel to the road. Being a high-pass on heading, this
    also slowly undoes lateral offsets (lane changes), hence it is off by default.
    Returns s (along road), d (lateral, right +), psi (rel. heading), psi_cam, rel_yaw."""
    dt = 1.0 / fps
    a = 1.0 - math.exp(-dt / tau)
    c = 1.0 - math.exp(-dt / cam_lag)
    psi_in = sim["psi"]
    lp, cam, s, d = 0.0, 0.0, 0.0, 0.0
    out = {k: np.zeros(len(psi_in)) for k in ("s", "d", "psi", "psi_cam")}
    for t, (v, p) in enumerate(zip(sim["v"], psi_in)):
        lp += a * (p - lp)
        psi = p - lp
        cam += c * (psi - cam)
        s += v * math.cos(psi) * dt
        d += v * math.sin(psi) * dt
        out["s"][t], out["d"][t], out["psi"][t], out["psi_cam"][t] = s, d, psi, cam
    out["rel_yaw"] = out["psi"] - out["psi_cam"]
    return out


def follow_road(rel, road_xy: np.ndarray, cam_back: float = 6.0):
    """Place a road-relative trajectory (from `washout`) on the road centerline. The road is
    rotated so its tangent at the anchor pano matches the view heading (the capture car's
    heading), and the car starts `cam_back` m ahead of the anchor camera. Returns x, y, psi,
    psi_cam (vehicle frame) as consumed by `chase_camera_c2w`."""
    grid, P, theta = _road_frame(road_xy)
    i0 = int(np.searchsorted(grid, 0.0))
    c, s_ = math.cos(-theta[i0]), math.sin(-theta[i0])
    P = (P - P[i0]) @ np.array([[c, s_], [-s_, c]])        # rotate by -theta0 about the anchor
    theta = theta - theta[i0]
    sv = rel["s"] + cam_back                                # vehicle arc length
    th_v = np.interp(sv, grid, theta)
    x = np.interp(sv, grid, P[:, 0]) - np.sin(th_v) * rel["d"]
    y = np.interp(sv, grid, P[:, 1]) + np.cos(th_v) * rel["d"]
    psi = th_v + rel["psi"]
    psi_cam = np.interp(rel["s"], grid, theta) + rel["psi_cam"]   # camera trails at arc length s
    return {"x": x, "y": y, "psi": psi, "psi_cam": psi_cam, "rel_yaw": psi - psi_cam}


# --------------------------------------------------------------------------- metric scale
CAM_HEIGHT_M = 2.5      # Street View camera height above the road (approx.)


def da3_scale(image_path, pitch_deg: float = -3.0, hfov_deg: float = 90.0, cam_height: float = CAM_HEIGHT_M) -> float:
    """Model units per metre for this anchor. MG3.5's Patch Memory lifts frames to 3D with
    Depth-Anything-3 metric depth, so the model's world has DA3's scale, which on our Street View
    frames is 0.6-1.0x the true scale (highways worst). We compare DA3 depth on the road just
    ahead (rows 64-99 % of the height, central 20 %) with the flat-road depth implied by the
    camera height and pitch, and return the median ratio."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(MG35_DIR / "third_party" / "depth-anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(str(CKPT / "DA3NESTED-GIANT-LARGE-1.1")).to("cuda").eval()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        depth = model.inference([img], process_res=448).depth[0]
    del model
    torch.cuda.empty_cache()
    H, W = depth.shape
    sy = H / img.height
    f = (img.width / 2) / math.tan(math.radians(hfov_deg) / 2)
    ratios = []
    for v in np.arange(0.64, 1.0, 0.035) * img.height:
        a = math.atan((v - img.height / 2) / f)                     # below the optical axis
        ground = cam_height / math.tan(a - math.radians(pitch_deg)) * math.cos(a)   # z-depth of the road
        patch = depth[int((v - 4) * sy):int((v + 4) * sy), int(W * 0.4):int(W * 0.6)]
        ratios.append(float(np.median(patch)) / ground)
    return float(np.median(ratios))


def intrinsics(hfov_deg: float = 90.0, w: int = WIDTH, h: int = HEIGHT) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([f, f, w / 2, h / 2], dtype=np.float32)


# --------------------------------------------------------------------------- generation
def run_mg35(image, c2w, K, prompt, out_mp4, *, num_blocks, seed, work, config, checkpoint, overrides=()):
    """Call the upstream distilled entrypoint (infer_distilled.py) in a subprocess."""
    work.mkdir(parents=True, exist_ok=True)
    cam = work / "camera.npz"
    np.savez(cam, extrinsics_c2w=c2w, intrinsics=np.tile(K[None], (len(c2w), 1)))
    cfg = yaml.safe_load(Path(config).read_text())
    cfg.update(num_blocks=int(num_blocks), seed=int(seed))
    for kv in overrides:                              # e.g. memory_mode=empty
        key, val = kv.split("=", 1)
        cfg[key] = yaml.safe_load(val)
    cfg_path = work / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    cmd = [sys.executable, str(MG35_DIR / "infer_distilled.py"),
           "--config", str(cfg_path), "--checkpoint", str(checkpoint),
           "--image", str(image), "--camera", str(cam), "--prompt", prompt,
           "--wan-dir", str(CKPT / "Wan2.2-TI2V-5B"),
           "--tokenizer-dir", str(CKPT / "Wan2.2-TI2V-5B" / "google" / "umt5-xxl"),
           "--da3-dir", str(CKPT / "DA3NESTED-GIANT-LARGE-1.1"),
           "--output", str(out_mp4)]
    t0 = time.time()
    subprocess.run(cmd, cwd=MG35_DIR, check=True)
    return time.time() - t0


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--route", default="cruise", help="preset in nycdrive/vehicle.py or JSON list of [s, throttle, steer]")
    p.add_argument("--seconds", type=float, default=21.0, help="displayed duration (s); rounded up to whole 84-frame blocks")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="keep every k-th generated frame; the vehicle is simulated at k * 16 Hz")
    p.add_argument("--scale", default="1.0",
                   help="model translation units per metre, or 'auto' = DA3 depth / flat-road depth of the "
                        "anchor (measured: fdr 0.69, gowanus 0.59, times_square 0.85, park 0.73, herald 0.72, fifth 1.02)")
    p.add_argument("--cam_back", type=float, default=6.0, help="chase-camera distance behind the car (m)")
    p.add_argument("--washout", type=float, default=0.0,
                   help="time constant (s) washing out sustained heading relative to the road; 0 = off "
                        "(note: a heading washout also slowly undoes lane changes)")
    p.add_argument("--keep_small_steer", action="store_true",
                   help="keep route segments with |steer| < 0.2 (MG3 presets use them to follow gentle "
                        "curves; with --road the real curvature comes from Street View, so they are zeroed)")
    p.add_argument("--no_road", action="store_true",
                   help="do not follow the Street View capture path (assume a straight road)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--prompt", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--condition", default="inpainted.png", help="conditioning frame in the scene dir")
    p.add_argument("--no_player", action="store_true")
    p.add_argument("--config", default=str(MG35_DIR / "configs" / "infer_distilled.yaml"))
    p.add_argument("--checkpoint", default=str(CKPT / "Matrix-Game-3.5-Distilled" / "first-person.safetensors"))
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "videos_mg35"))
    p.add_argument("--save_name", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a key of the MG3.5 distilled config (e.g. memory_mode=empty)")
    p.add_argument("--reuse_world", action="store_true", help="skip generation if outputs/mg35_work/<name>/result.mp4 exists")
    args = p.parse_args()

    scene = Path(args.scene).resolve()
    route = json.loads(Path(args.route).read_text()) if args.route.endswith(".json") else args.route
    meta = json.loads((scene / "meta.json").read_text()) if (scene / "meta.json").exists() else {}
    prompt = args.prompt or (scene / "prompt.txt").read_text().strip()
    prompt = prompt.replace("Third-person view of a car driving", "First-person view driving")
    label = (args.label or meta.get("label") or scene.name).upper()
    save_name = args.save_name or f"{scene.name}_{Path(args.route).stem}"
    cond = scene / args.condition if (scene / args.condition).exists() else scene / "image.png"
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / save_name

    if str(args.scale).lower() == "auto":
        args.scale = round(da3_scale(cond, meta.get("view_pitch_deg", -3.0), meta.get("view_hfov_deg", 90.0)), 3)
        print(f"[nycdrive-mg35] auto scale (DA3 / flat-road depth) = {args.scale}", flush=True)
    args.scale = float(args.scale)

    # --- trajectory: vehicle model at the generation rate -> chase-camera c2w
    k = max(1, args.frame_stride)
    fps_gen = FPS_OUT * k
    num_blocks = max(1, math.ceil(args.seconds * fps_gen / POSES_PER_BLOCK))
    n = 1 + POSES_PER_BLOCK * num_blocks                    # poses == generated frames (incl. anchor)
    segments = ROUTES[route] if isinstance(route, str) else route
    if not args.keep_small_steer:
        segments = [(sec, thr, st if abs(st) >= 0.2 else 0.0) for sec, thr, st in segments]
    sim = simulate(route_controls(segments, n - 1, fps=fps_gen), fps=fps_gen)
    # road-relative trajectory (sustained heading washed out), then placed on the real road
    if args.washout > 0:
        rel = washout(sim, fps_gen, tau=args.washout)
    else:
        rel = {"s": sim["x"], "d": sim["y"], "psi": sim["psi"], "psi_cam": sim["psi_cam"], "rel_yaw": sim["rel_yaw"]}
    road_xy = None
    if not args.no_road and meta.get("pano_id"):
        try:
            road_xy = road_centerline(meta["pano_id"], meta.get("view_heading_deg", meta.get("pano_heading_deg", 0.0)),
                                      cache=ROOT / "outputs" / "mg35_work" / f"road_{scene.name}.json")
            print(f"[nycdrive-mg35] road centerline: {len(road_xy)} panos, "
                  f"{np.hypot(*np.diff(road_xy, axis=0).T).sum():.0f} m", flush=True)
        except Exception as e:  # network / coverage problems: fall back to a straight road
            print(f"[nycdrive-mg35] warning: road centerline unavailable ({e}); assuming a straight road")
    if road_xy is not None and len(road_xy) >= 3:
        traj = follow_road(rel, road_xy, cam_back=args.cam_back)
    else:
        traj = {"x": rel["s"], "y": rel["d"], "psi": rel["psi"], "psi_cam": rel["psi_cam"], "rel_yaw": rel["rel_yaw"]}
    c2w = chase_camera_c2w(traj, cam_back=args.cam_back, pitch_deg=meta.get("view_pitch_deg", -3.0),
                           scale=args.scale)
    K = intrinsics(meta.get("view_hfov_deg", 90.0))
    # per-frame vehicle state aligned with poses (pose 0 = anchor, car at rest)
    pad = lambda a: np.concatenate([a[:1], a])  # noqa: E731
    state = {key: pad(traj[key]) for key in ("psi", "psi_cam", "x", "y", "rel_yaw")}
    state["v"] = np.concatenate([[0.0], sim["v"]])
    state["rel_yaw"][0] = 0.0
    controls = np.concatenate([sim["controls"][:1], sim["controls"]]).astype(np.float32)
    print(f"[nycdrive-mg35] scene={scene.name} route={args.route} blocks={num_blocks} frames={n} "
          f"stride={k} gen_fps={fps_gen} scale={args.scale} v_max={state['v'].max():.1f} m/s "
          f"dist={np.hypot(np.diff(state['x']), np.diff(state['y'])).sum():.0f} m", flush=True)

    work = ROOT / "outputs" / "mg35_work" / save_name
    raw = work / "result.mp4"                                # upstream output, every generated frame
    gen_s = float("nan")
    if not (args.reuse_world and raw.exists()):
        gen_s = run_mg35(cond, c2w, K, prompt, raw, num_blocks=num_blocks, seed=args.seed, work=work,
                         config=args.config, checkpoint=args.checkpoint, overrides=args.set)
        print(f"[nycdrive-mg35] generation took {gen_s:.0f} s ({gen_s / n:.2f} s/frame)", flush=True)
    frames = np.stack(imageio.mimread(raw, memtest=False))
    if len(frames) != n:
        print(f"[nycdrive-mg35] warning: got {len(frames)} frames for {n} poses", flush=True)
    T = min(len(frames), n)
    frames = frames[:T]

    np.savez(base.with_name(base.name + "_actions.npz"), controls=controls[:T], stride=k, fps=FPS_OUT,
             gen_fps=fps_gen, c2w=c2w[:T], intrinsics=K, scale=args.scale, cam_back=args.cam_back,
             rel_yaw=np.rad2deg(state["rel_yaw"][:T]), **{key: state[key][:T] for key in ("v", "psi", "psi_cam", "x", "y")},
             prompt=prompt, gen_seconds=gen_s, num_blocks=num_blocks)

    shown = slice(0, T, k)
    frames, rel_yaw, v = frames[shown], np.rad2deg(state["rel_yaw"][shown]), state["v"][shown]
    imageio.mimwrite(base.with_name(base.name + "_world.mp4"), frames, fps=FPS_OUT, quality=8, macro_block_size=None)
    sprites = None if args.no_player else load_car_sprites(scene)
    if sprites is not None:
        frames = composite_player(frames, rel_yaw, sprites, moving=v > 0.5)
    imageio.mimwrite(base.with_name(base.name + "_clean.mp4"), frames, fps=FPS_OUT, quality=8, macro_block_size=None)
    hud = draw_hud(frames, controls[shown], None, label)
    imageio.mimwrite(base.with_name(base.name + "_hud.mp4"), hud, fps=FPS_OUT, quality=8, macro_block_size=None)
    print(f"[nycdrive-mg35] wrote {base}_{{world,clean,hud}}.mp4 ({len(frames)} frames @ {FPS_OUT} fps)")


if __name__ == "__main__":
    main()
