"""Render MeshFleet car meshes as a third-person "chase camera" sprite (RGBA).

The player car is rendered offscreen with pyrender (EGL), from directly behind
and slightly above, with a transparent background, so it can be composited onto
a Street View frame.

Car models come from the MeshFleet dataset (Boborzi et al., 2025,
https://huggingface.co/datasets/DamianBoborzi/MeshFleet), which curates car
meshes from Objaverse-XL / Sketchfab.  Every model has its own license (mostly
CC-BY); see `PLAYER_CARS` below for the per-model source and license.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
from PIL import Image

MESHFLEET_REPO = "DamianBoborzi/MeshFleet"


@dataclass
class CarModel:
    key: str
    sha256: str          # MeshFleet / Objaverse-XL sha256
    mesh_filename: str   # file name inside MeshFleet `meshes/`
    description: str
    source_url: str
    license: str
    # Rotation (degrees, around the up axis) that makes the car's FRONT point
    # towards -Z after the default glTF (+Y up) loading.  Verified visually
    # with a 4-view contact sheet (see `contact_sheet`).
    yaw_deg: float = 0.0


PLAYER_CARS = {
    "subaru": CarModel(
        "subaru", "8c63a6e3ae36c780f9ea5de58d8d746f59f1468778f30ecbc376d92feda6d0ba",
        "36f4a47b619a49bda4c1870e7833b25b.glb", "blue Subaru WRX STI sedan",
        "https://sketchfab.com/3d-models/36f4a47b619a49bda4c1870e7833b25b", "CC-BY", yaw_deg=90),
    "vitara": CarModel(
        "vitara", "78e5a54121d32aee2e1850aaff8fabe58bd1db4f14bfbe946326fbc35a576421",
        "6c2f647e8874401abfee51dbe6a0bee9.glb", "red Suzuki Vitara SUV",
        "https://sketchfab.com/3d-models/6c2f647e8874401abfee51dbe6a0bee9", "CC-BY", yaw_deg=180),
    "mustang": CarModel(
        "mustang", "89f6d9c858a0021d608030c8b2f144799a11639ff360873c574b37db97e5ea61",
        "f8e521c3315e445ea69f630c79b682f0.glb", "yellow Ford Mustang GT (taxi-yellow)",
        "https://sketchfab.com/3d-models/f8e521c3315e445ea69f630c79b682f0", "CC-BY", yaw_deg=180),
    "ferrari": CarModel(
        "ferrari", "e7d2897cbdcfe20049dda66d04f55d9d5b29cbf4c83648c1969d117e62edad1a",
        "0045292de2114f0c83a9e5413eb043d4.glb", "red Ferrari 488 GTB sports car",
        "https://sketchfab.com/3d-models/0045292de2114f0c83a9e5413eb043d4", "CC-BY", yaw_deg=180),
    "police": CarModel(
        "police", "78a24ccee56811b22af2f167b9f7eea5f2eb4e2873cc1f36cb5f49947dd16866",
        "b1694c78e3184150b8fd518eeee5af06.glb", "white Dodge Charger police car",
        "https://sketchfab.com/3d-models/b1694c78e3184150b8fd518eeee5af06", "CC-BY"),
}


def mesh_path(car: CarModel) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MESHFLEET_REPO, f"meshes/{car.mesh_filename}", repo_type="dataset")


def _load_normalized_scene(path: str, yaw_deg: float):
    """Load a glTF scene, rotate so front -> -Z, scale to unit length, ground at y=0."""
    import pyrender

    tm_scene = trimesh.load(path, force="scene")
    # Bake node transforms into geometry so bounds are correct.
    meshes = []
    for node_name in tm_scene.graph.nodes_geometry:
        T, geom_name = tm_scene.graph[node_name]
        g = tm_scene.geometry[geom_name]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        g = g.copy()
        g.apply_transform(T)
        meshes.append(g)
    R = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])
    for g in meshes:
        g.apply_transform(R)
    lo = np.min([g.bounds[0] for g in meshes], axis=0)
    hi = np.max([g.bounds[1] for g in meshes], axis=0)
    ext = hi - lo
    length = max(ext[0], ext[2])
    center = (lo + hi) / 2
    S = np.eye(4)
    S[:3, :3] /= length
    S[:3, 3] = -np.array([center[0], lo[1], center[2]]) / length
    pr_meshes = []
    for g in meshes:
        g.apply_transform(S)
        m = pyrender.Mesh.from_trimesh(g, smooth=False)
        for p in m.primitives:
            mat = p.material
            # pyrender has no environment lighting; fully metallic paint
            # renders nearly black, so tone metalness down.
            if hasattr(mat, "metallicFactor") and mat.metallicFactor is not None:
                mat.metallicFactor = min(float(mat.metallicFactor), 0.25)
            if hasattr(mat, "roughnessFactor") and mat.roughnessFactor is not None:
                mat.roughnessFactor = max(float(mat.roughnessFactor), 0.35)
        pr_meshes.append(m)
    return pr_meshes, ext / length


def _look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, eye
    return M


def chase_camera_pose(ext, cam_height: float = 0.55, cam_dist: float = 1.6,
                      view_yaw_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world pose orbiting the (unit length, front -> -Z) car at
    `view_yaw_deg` (0 = directly behind), aimed a bit above the ground."""
    a = np.radians(view_yaw_deg)
    eye = np.array([np.sin(a) * cam_dist, cam_height, np.cos(a) * cam_dist])
    target = np.array([0.0, ext[1] * 0.35, 0.0])
    return _look_at(eye, target)


def build_car_scene(meshes):
    """pyrender scene with the car meshes under a single "car" node (so the
    whole car can be re-posed, e.g. yawed, with `scene.set_pose`) and
    world-fixed lights.  Returns (scene, car_node); no camera yet."""
    import pyrender

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
    car_node = pyrender.Node(name="car", matrix=np.eye(4))
    scene.add_node(car_node)
    for m in meshes:
        scene.add(m, parent_node=car_node)
    # Key light from above/behind the camera, fill lights from the sides.
    for d, inten in [((0.3, 1.0, 0.8), 3.5), ((-0.8, 0.6, -0.3), 1.5),
                     ((0.8, 0.5, -0.4), 1.2), ((0, -0.2, 1.0), 0.8)]:
        d = np.asarray(d, float)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=inten),
                  pose=_look_at(d * 5, np.zeros(3)))
    return scene, car_node


def render_car_frame(path: str, yaw_deg: float = 0.0, view_yaw_deg: float = 0.0,
                     size: int = 1024, cam_height: float = 0.55, cam_dist: float = 1.6,
                     fov_deg: float = 40.0) -> np.ndarray:
    """Full (uncropped) size x size RGBA render; see `render_car`."""
    import pyrender

    meshes, ext = _load_normalized_scene(path, yaw_deg)
    scene, _ = build_car_scene(meshes)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg))
    scene.add(cam, pose=chase_camera_pose(ext, cam_height, cam_dist, view_yaw_deg))
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return color


def alpha_bbox(img: Image.Image, thresh: int = 8):
    """(left, top, right, bottom) of the pixels with alpha > thresh, or None."""
    return img.getchannel("A").point(lambda v: 255 if v > thresh else 0).getbbox()


def render_car(path: str, yaw_deg: float = 0.0, view_yaw_deg: float = 0.0,
               size: int = 1024, cam_height: float = 0.55, cam_dist: float = 1.6,
               fov_deg: float = 40.0) -> Image.Image:
    """Render the car (unit length, front -> -Z) from a camera orbiting at
    `view_yaw_deg` (0 = directly behind).  Returns a tight-cropped RGBA image."""
    img = Image.fromarray(render_car_frame(path, yaw_deg, view_yaw_deg, size, cam_height,
                                           cam_dist, fov_deg), "RGBA")
    bbox = alpha_bbox(img)
    return img.crop(bbox) if bbox else img


def contact_sheet(path: str, yaw_deg: float = 0.0, out: str | None = None) -> Image.Image:
    """4 views (behind, right, front, left) on grey, for orientation checks."""
    tiles = []
    for vy in (0, 90, 180, 270):
        im = render_car(path, yaw_deg, view_yaw_deg=vy, size=384)
        bg = Image.new("RGBA", (384, 384), (128, 128, 128, 255))
        im.thumbnail((380, 380))
        bg.alpha_composite(im, ((384 - im.width) // 2, (384 - im.height) // 2))
        tiles.append(bg)
    sheet = Image.new("RGBA", (384 * 4, 384))
    for i, t in enumerate(tiles):
        sheet.paste(t, (384 * i, 0))
    if out:
        sheet.convert("RGB").save(out)
    return sheet


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write orientation contact sheets for the player cars")
    ap.add_argument("--out", default="outputs/car_sheets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key, car in PLAYER_CARS.items():
        contact_sheet(mesh_path(car), car.yaw_deg, os.path.join(args.out, f"{key}.jpg"))
        print("wrote", key)
