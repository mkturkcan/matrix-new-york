"""Pre-render the player car at a range of yaw angles ("car bank") per scene.

For the chase-camera overlay the player car should *rotate* when steering
(yaw relative to the camera) instead of sliding sideways in 2D.  This script
re-renders, for each scene in data/inputs/, the same MeshFleet car with the
same chase camera, scale, screen placement, lighting and color matching that
`prepare_inputs.py` used for that scene's car_layer.png, but with the car
yawed about a vertical axis through its rear axle, and writes:

  data/inputs/<scene>/car_bank.npz
      yaw_deg  (N,)            float32  relative yaw in degrees
      rgba     (N, 704, 1280, 4) uint8  straight-alpha car + contact shadow
  data/inputs/<scene>/car_bank_preview.jpg   (contact sheet for eyeballing)

Sign convention: positive yaw = the car's nose turns to the RIGHT of the image
(we then see more of its left flank), i.e. a right turn / right steer.  The
yaw-0 entry reproduces car_layer.png (same placement, up to resampling).

How the placement is reproduced: car_layer.png was made by rendering the car
in a square `CHASE_CAM` frame, tight-cropping it to its alpha bbox and resizing
the crop to `car_placement(...)`.  That crop+resize is an axis-aligned affine
map, so it folds into pinhole intrinsics (fx, fy, cx, cy) for the 1280x704
output frame; we render every yaw directly with those intrinsics (2x
supersampled).  The contact shadow lies on the ground plane, so rotating it
with the car is a homography of the yaw-0 shadow: H_ground @ R_yaw @ H_ground^-1.

Usage:
  python -m nycdrive.car_bank                          # every scene with a car_layer.png
  python -m nycdrive.car_bank --scenes herald_square times_square
  python -m nycdrive.car_bank --max-yaw 24 --step 3 --pivot 0.30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python nycdrive/car_bank.py`
from nycdrive.car_render import (PLAYER_CARS, _load_normalized_scene, alpha_bbox, build_car_scene,
                                 chase_camera_pose, mesh_path, render_car_frame)
from nycdrive.prepare_inputs import (CHASE_CAM, OUT_H, OUT_W, ROOT, apply_car_color, car_color_match,
                                     car_placement, contact_shadow, player_layer)

# Pivot of the yaw rotation, in car lengths behind the car's center (the car
# spans z in [-0.5, 0.5], rear at +0.5).  0.30 is roughly the rear axle (rear
# overhang ~20% of the length): a front-steered car turns about a point on its
# rear axle line, so the rear stays planted and the nose swings.  Use 0.0 to
# spin about the car's center instead.
REAR_AXLE_PIVOT = 0.30
SUPERSAMPLE = 2


@dataclass
class YawRenders:
    """Car-only renders (no color matching / shadow) on the output frame."""
    yaw_deg: np.ndarray        # (N,) float32
    frames: np.ndarray         # (N, H, W, 4) uint8, straight alpha
    shadow_warps: np.ndarray   # (N, 3, 3) image homographies moving the yaw-0 shadow
    placement: tuple           # (x0, y0, cw, ch) of the yaw-0 sprite, as in prepare_inputs


def yaw_angles(max_yaw: float = 24.0, step: float = 3.0) -> np.ndarray:
    n = int(round(max_yaw / step))
    return (np.arange(-n, n + 1) * step).astype(np.float32)


def _yaw_about(yaw_deg: float, pivot_z: float) -> np.ndarray:
    """4x4 rotation about the vertical axis through (0, 0, pivot_z).

    The camera sits at +Z looking towards -Z (car front), image right = +X.
    A rotation by +phi about +Y moves the nose (-Z) towards -X, so positive
    yaw (nose to the image right) is a rotation by -yaw."""
    phi = np.radians(-yaw_deg)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)
    T = np.eye(4)
    T[2, 3] = pivot_z
    return T @ R @ np.linalg.inv(T)


def render_yaws(car_key: str, yaws: np.ndarray, pivot_z: float = REAR_AXLE_PIVOT,
                ss: int = SUPERSAMPLE) -> YawRenders:
    import pyrender

    car = PLAYER_CARS[car_key]
    path = mesh_path(car)

    # 1) Reference render exactly as prepare_inputs.car_sprite did it, to get
    #    the alpha bbox -> where the crop landed on the output frame.
    S, fov = CHASE_CAM["size"], np.radians(CHASE_CAM["fov_deg"])
    ref = Image.fromarray(render_car_frame(path, car.yaw_deg, 0.0, **CHASE_CAM), "RGBA")
    bx0, by0, bx1, by1 = alpha_bbox(ref)
    x0, y0, cw, ch = car_placement(bx1 - bx0, by1 - by0)
    sx, sy = cw / (bx1 - bx0), ch / (by1 - by0)

    # 2) Fold "crop + resize" into output-frame intrinsics: u_out = x0 + (u_ref - bx0) * sx.
    f = (S / 2) / np.tan(fov / 2)
    fx, fy = f * sx, f * sy
    cx, cy = x0 + (S / 2 - bx0) * sx, y0 + (S / 2 - by0) * sy

    meshes, ext = _load_normalized_scene(path, car.yaw_deg)
    scene, car_node = build_car_scene(meshes)
    cam_pose = chase_camera_pose(ext, CHASE_CAM["cam_height"], CHASE_CAM["cam_dist"])
    cam = pyrender.IntrinsicsCamera(fx * ss, fy * ss, cx * ss, cy * ss)
    scene.add(cam, pose=cam_pose)

    # Ground plane (y = 0) -> output pixels, for warping the contact shadow.
    # pyrender camera looks down -Z: u = cx + fx * X / -Z, v = cy - fy * Y / -Z.
    Kp = np.array([[fx, 0, -cx], [0, -fy, -cy], [0, 0, -1]])
    P = Kp @ np.linalg.inv(cam_pose)[:3]
    H_ground = P[:, [0, 2, 3]]                     # columns for world (x, z, 1)
    H_ground_inv = np.linalg.inv(H_ground)

    r = pyrender.OffscreenRenderer(OUT_W * ss, OUT_H * ss)
    frames, warps = [], []
    for yaw in yaws:
        M = _yaw_about(float(yaw), pivot_z)
        scene.set_pose(car_node, M)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        # PIL resizes RGBA in premultiplied space, like the original sprite resize.
        frames.append(np.asarray(Image.fromarray(color, "RGBA").resize((OUT_W, OUT_H), Image.LANCZOS)))
        Rg = M[np.ix_([0, 2, 3], [0, 2, 3])]       # same rotation acting on ground (x, z, 1)
        warps.append(H_ground @ Rg @ H_ground_inv)
    r.delete()
    return YawRenders(np.asarray(yaws, np.float32), np.stack(frames), np.stack(warps), (x0, y0, cw, ch))


def build_car_bank(scene_dir: str, renders: YawRenders) -> np.ndarray:
    """Color-match the yaw renders to this scene and add the rotated contact
    shadow.  Returns (N, H, W, 4) uint8 straight-alpha layers."""
    scene_img = np.asarray(Image.open(os.path.join(scene_dir, "inpainted.png")).convert("RGB"))
    x0, y0, cw, ch = renders.placement
    # Color matching parameters come from the yaw-0 car (as in car_layer.png)
    # and are shared by all angles so brightness does not flicker with steering.
    i0 = int(np.argmin(np.abs(renders.yaw_deg)))
    ref = renders.frames[i0].astype(np.float32)
    gain, tint = car_color_match(scene_img, ref[..., :3], ref[..., 3] / 255.0, x0, y0, cw, ch)
    shadow0 = np.asarray(contact_shadow(x0, y0, cw, ch))

    out = np.empty_like(renders.frames)
    for i, (frame, Hw) in enumerate(zip(renders.frames, renders.shadow_warps)):
        rgb = apply_car_color(frame[..., :3].astype(np.float32), gain, tint)
        car = Image.fromarray(np.dstack([rgb, frame[..., 3]]).astype(np.uint8), "RGBA")
        shadow = cv2.warpPerspective(shadow0, Hw, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR)
        out[i] = np.asarray(player_layer(car, Image.fromarray(shadow)))
    return out


def preview(scene_dir: str, yaw_deg: np.ndarray, rgba: np.ndarray,
            show=(-24, -12, 0, 12, 24), tile_w: int = 640) -> Image.Image:
    """Contact sheet: original car_layer.png, then the bank at `show` angles,
    each composited over inpainted.png (bottom half of the frame, zoomed)."""
    base = Image.open(os.path.join(scene_dir, "inpainted.png")).convert("RGBA")
    layers = [("car_layer.png", Image.open(os.path.join(scene_dir, "car_layer.png")).convert("RGBA"))]
    for a in show:
        i = int(np.argmin(np.abs(yaw_deg - a)))
        layers.append((f"yaw {yaw_deg[i]:+.0f}", Image.fromarray(rgba[i], "RGBA")))
    crop = (OUT_W // 8, OUT_H * 3 // 8, OUT_W * 7 // 8, OUT_H)   # where the car lives
    tile_h = int(tile_w * (crop[3] - crop[1]) / (crop[2] - crop[0]))
    cols = 3
    rows = (len(layers) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h))
    for k, (label, layer) in enumerate(layers):
        im = base.copy()
        im.alpha_composite(layer)
        im = np.array(im.convert("RGB").crop(crop).resize((tile_w, tile_h), Image.LANCZOS))
        cv2.putText(im, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        sheet.paste(Image.fromarray(im), ((k % cols) * tile_w, (k // cols) * tile_h))
    return sheet


def yaw0_check(scene_dir: str, yaw_deg: np.ndarray, rgba: np.ndarray) -> tuple[float, float]:
    """(car-alpha IoU, mean |RGB diff| on the car) of the yaw-0 entry vs car_layer.png."""
    orig = np.asarray(Image.open(os.path.join(scene_dir, "car_layer.png")).convert("RGBA")).astype(np.float32)
    new = rgba[int(np.argmin(np.abs(yaw_deg)))].astype(np.float32)
    a, b = orig[..., 3] > 240, new[..., 3] > 240   # the car body (the shadow is < 200)
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    diff = float(np.abs(orig[..., :3] - new[..., :3])[a & b].mean())
    return iou, diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--scenes", nargs="*", help="scene names (default: every dir with a car_layer.png)")
    ap.add_argument("--max-yaw", type=float, default=24.0)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--pivot", type=float, default=REAR_AXLE_PIVOT,
                    help="yaw pivot in car lengths behind the car center (0.30 ~ rear axle, 0 = center)")
    args = ap.parse_args(argv)

    names = args.scenes or sorted(d for d in os.listdir(args.inputs)
                                  if os.path.isfile(os.path.join(args.inputs, d, "car_layer.png")))
    yaws = yaw_angles(args.max_yaw, args.step)
    cache: dict[str, YawRenders] = {}   # renders only depend on the car model
    failed = []
    for name in names:
        d = os.path.join(args.inputs, name)
        try:
            with open(os.path.join(d, "meta.json")) as f:
                key = json.load(f)["player_car"]["key"]
            if key not in cache:
                cache[key] = render_yaws(key, yaws, args.pivot)
            rgba = build_car_bank(d, cache[key])
            np.savez_compressed(os.path.join(d, "car_bank.npz"), yaw_deg=yaws, rgba=rgba)
            preview(d, yaws, rgba).save(os.path.join(d, "car_bank_preview.jpg"), quality=88)
            iou, diff = yaw0_check(d, yaws, rgba)
            print(f"[{name}] {key}: {len(yaws)} yaws -> car_bank.npz  (yaw-0 vs car_layer: IoU {iou:.3f}, "
                  f"mean |dRGB| {diff:.1f})")
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print("failed:", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
