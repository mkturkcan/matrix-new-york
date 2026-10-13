"""Prepare first-frame inputs for the NYC driving demo (Matrix-Game-3.0).

For every scene this script:
  1. finds an official Google Street View car panorama near (lat, lon) via
     `streetlevel`, and downloads it as an equirectangular image;
  2. renders a 1280x704 perspective view looking along the capture car's
     driving direction (or a user-given heading), slightly pitched down;
  3. segments vehicles (Mask2Former, Cityscapes) and masks the ones close to
     the camera (keeps distant traffic so the scene stays alive);
  4. inpaints the masked vehicles with LaMa (falls back to OpenCV Telea);
  5. composites a rendered MeshFleet car seen from a third-person chase camera
     at the bottom-center as the player's vehicle, with a soft contact shadow;
  6. writes data/inputs/<scene>/{image.png, prompt.txt, meta.json} plus debug
     images {raw.png, mask.png, inpainted.png}.

Usage:
  python -m nycdrive.prepare_inputs                       # all curated scenes
  python -m nycdrive.prepare_inputs --scenes times_square fdr_drive
  python -m nycdrive.prepare_inputs --random [--count 3]  # random Manhattan/Brooklyn/Queens spot
  python -m nycdrive.prepare_inputs --lat 40.758 --lon -73.9855 [--heading 200] [--name my_scene]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image, ImageFilter

from nycdrive.car_render import PLAYER_CARS, mesh_path, render_car

OUT_W, OUT_H = 1280, 704
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cityscapes train ids (Mask2Former cityscapes-semantic label space)
CS_ROAD = 0
CS_VEHICLES = [12, 13, 14, 15, 16, 17, 18]  # rider, car, truck, bus, train, motorcycle, bicycle

HIGHWAY_PROMPT = ("Third-person view of a car driving on {where} in New York City, {details}, "
                  "daytime, realistic.")
STREET_PROMPT = ("Third-person view of a car driving down {where} in New York City, {details}, "
                 "daytime, realistic.")


@dataclass
class Scene:
    name: str
    lat: float
    lon: float
    kind: str = "street"           # "street" or "highway"
    where: str = "a busy street in Manhattan"
    details: str = "with tall buildings, yellow taxis, crosswalks and traffic lights"
    heading: float | None = None   # None -> use the capture car's heading
    heading_offset: float = 0.0    # added to the capture car's heading (e.g. 180 to flip)
    label: str | None = None       # short HUD name, e.g. "FDR DRIVE"
    car: str | None = None         # key into PLAYER_CARS
    street: tuple = ()             # pano address must contain one of these (e.g. "FDR")

    @property
    def prompt(self) -> str:
        tpl = HIGHWAY_PROMPT if self.kind == "highway" else STREET_PROMPT
        return tpl.format(where=self.where, details=self.details)


# Curated scenes. Each was checked visually to return an official pano on the
# intended road; headings follow the capture car unless overridden.
# (Tried and rejected for now: FDR at E 57th St, West Side Hwy at W 23rd St,
# Brooklyn Bridge -- resolves to the pedestrian promenade --, LIE in Elmhurst,
# Broadway/Wall St and E 42nd St: the nearest pano was a side street,
# sidewalk or the Park Ave viaduct.)
CURATED = [
    # --- highways -------------------------------------------------------------
    Scene("fdr_drive", 40.7425, -73.9715, "highway", "the FDR Drive along the East River",
          "with the Manhattan skyline, guardrails and highway traffic", label="FDR DRIVE", car="mustang", street=("FDR",)),
    Scene("henry_hudson_pkwy", 40.8000, -73.9760, "highway", "the Henry Hudson Parkway",
          "with trees, the Hudson River and highway traffic", label="HENRY HUDSON PKWY", car="vitara", street=("Henry Hudson",)),
    Scene("gowanus_expressway", 40.6640, -74.0000, "highway", "the elevated Gowanus Expressway in Brooklyn",
          "with concrete barriers, overhead signs and highway traffic", label="GOWANUS EXPWY", car="subaru",
          street=("Gowanus", "Brooklyn-Queens", "I-278", "BQE")),
    # --- downtown / midtown streets ----------------------------------------------
    Scene("times_square", 40.7580, -73.9855, "street", "Seventh Avenue through Times Square",
          "with giant illuminated billboards, tall buildings, yellow taxis and crowds", label="TIMES SQUARE", car="mustang",
          street=("7th Ave", "Broadway")),
    Scene("fifth_avenue", 40.7540, -73.9810, "street", "Fifth Avenue in Midtown Manhattan",
          "with tall buildings, luxury storefronts, yellow taxis and traffic lights", label="FIFTH AVENUE", car="ferrari",
          street=("5th Ave",)),
    Scene("park_avenue", 40.7560, -73.9740, "street", "Park Avenue in Midtown Manhattan",
          "with glass office towers, a planted median, yellow taxis and traffic lights", label="PARK AVENUE", car="subaru",
          street=("Park Ave",)),
    Scene("canal_street", 40.7180, -73.9990, "street", "Canal Street in Chinatown, Manhattan",
          "with colorful shop signs, fire escapes, delivery trucks and crosswalks", label="CANAL STREET - CHINATOWN", car="mustang",
          street=("Canal St",)),
    Scene("soho_broadway", 40.7235, -73.9985, "street", "Broadway in SoHo, Manhattan",
          "with cast-iron buildings, storefronts, yellow taxis and crosswalks", label="SOHO", car="ferrari",
          street=("Broadway",)),
    Scene("dumbo", 40.7033, -73.9897, "street", "a cobblestone street in DUMBO, Brooklyn",
          "with red-brick warehouses and the Manhattan Bridge overhead", label="DUMBO", car="police",
          street=("Washington St",)),
]

# Rough bounding boxes (lat_min, lat_max, lon_min, lon_max) for --random
RANDOM_BOXES = {
    "Manhattan": (40.700, 40.800, -74.015, -73.935),
    "Brooklyn": (40.635, 40.705, -74.010, -73.920),
    "Queens": (40.705, 40.765, -73.940, -73.830),
}


# ----------------------------------------------------------------------------
# Street View
# ----------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _pano_streets(pano) -> str:
    names = [str(a.value) for a in (pano.address or [])]
    names += [str(sl.name.value) for sl in (getattr(pano, "street_names", None) or [])]
    return " | ".join(names)


def find_official_pano(lat: float, lon: float, max_dist: float = 60.0, street: tuple = ()):
    """Nearest official (Google car) panorama to (lat, lon), or None.

    Google decommissioned the `SingleImageSearch` endpoint that
    `streetview.find_panorama` relies on, so we query the coverage tiles
    (which only list official car/trekker panoramas) around the point and
    pick the closest one, then fetch its full metadata by id.  If `street`
    keywords are given (e.g. ("FDR",)), the closest pano whose address /
    street label matches one of them is preferred, so that a highway is not
    confused with the service road next to it.
    """
    from streetlevel import streetview
    from streetlevel.geo import wgs84_to_tile_coord

    if street:
        max_dist = max(max_dist, 250.0)
    x, y = wgs84_to_tile_coord(lat, lon, 17)
    cands = {}
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        try:
            for p in streetview.get_coverage_tile(x + dx, y + dy):
                cands[p.id] = p
        except Exception as e:  # network hiccups
            print(f"  coverage tile error: {e}")
        if not street and cands and dx == 0 and dy == 0:
            if min(_haversine(lat, lon, p.lat, p.lon) for p in cands.values()) < max_dist / 2:
                break
    ranked = sorted(cands.values(), key=lambda p: _haversine(lat, lon, p.lat, p.lon))
    ranked = [p for p in ranked if _haversine(lat, lon, p.lat, p.lon) <= max_dist]
    fallback = None
    for p in ranked[: 40 if street else 5]:
        full = streetview.find_panorama_by_id(p.id)
        if full is None or getattr(full, "is_third_party", False):
            continue
        if not street:
            return full
        fallback = fallback or full
        names = _pano_streets(full).lower()
        if any(k.lower() in names for k in street):
            return full
    if street and fallback is not None:
        print(f"  [warn] no pano on {street} near {lat:.5f},{lon:.5f}; using nearest")
    return fallback


def fetch_equirect(pano, zoom: int = 4) -> np.ndarray:
    """Download the equirect panorama.  Some (newer) panoramas have missing
    tiles at high zoom levels which come back black; drop to a lower zoom
    when that happens."""
    from streetlevel import streetview
    for z in range(zoom, 1, -1):
        img = np.asarray(streetview.get_panorama(pano, zoom=z).convert("RGB"))
        # Ignore the nadir: many panoramas are black below the car by design;
        # that part is handled by `nodata_mask` + inpainting.
        black = (img.max(-1) == 0)[: int(img.shape[0] * 0.75)]
        if black.mean() < 0.002:
            return img
        print(f"  zoom {z}: {black.mean():.1%} missing tiles, retrying at zoom {z - 1}")
    return img


def equirect_to_perspective(equi: np.ndarray, yaw_deg: float, pitch_deg: float,
                            hfov_deg: float = 90.0, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    """Pinhole view from an equirect image.

    yaw_deg is relative to the panorama center (positive = to the right /
    clockwise), pitch_deg is positive up.
    """
    He, We = equi.shape[:2]
    f = (out_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    jj, ii = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (jj - out_w / 2 + 0.5) / f
    y = -(ii - out_h / 2 + 0.5) / f   # up
    z = np.ones_like(x)
    p = math.radians(pitch_deg)
    y2 = y * math.cos(p) + z * math.sin(p)
    z2 = -y * math.sin(p) + z * math.cos(p)
    lon = np.arctan2(x, z2) + math.radians(yaw_deg)
    lat = np.arctan2(y2, np.sqrt(x ** 2 + z2 ** 2))
    map_x = ((lon / (2 * math.pi) + 0.5) % 1.0) * We - 0.5
    map_y = (0.5 - lat / math.pi) * He - 0.5
    return cv2.remap(equi, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


# ----------------------------------------------------------------------------
# Vehicle segmentation + inpainting
# ----------------------------------------------------------------------------
class Segmenter:
    MODEL = "facebook/mask2former-swin-large-cityscapes-semantic"

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(self.MODEL)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL).to(device).eval()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        inputs = self.proc(images=Image.fromarray(img), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0]
        return seg.cpu().numpy().astype(np.uint8)


def nodata_mask(img: np.ndarray, thresh: int = 8, dilate_px: int = 6) -> np.ndarray:
    """Near-black "no data" regions (missing pano tiles / nadir holes) that
    touch the image border, so they can be inpainted as well."""
    dark = (img.max(-1) < thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    H, W = dark.shape
    out = np.zeros_like(dark)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        if touches and area > 50:
            out[lab == i] = 1
    if dilate_px > 0 and out.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        out = cv2.dilate(out, k)
    return out * 255


def near_vehicle_mask(seg: np.ndarray, dilate_px: int = 18) -> np.ndarray:
    """Mask of vehicles that are close to the camera (or where the player car goes).

    Far traffic is kept so the scene still looks alive.
    """
    H, W = seg.shape
    veh = np.isin(seg, CS_VEHICLES).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(veh, connectivity=8)
    keep = np.zeros_like(veh)
    # Corridor in front of the camera (our lane) and the region where the
    # player's car will sit (bottom-center).
    cx0, cx1, cy0 = int(0.30 * W), int(0.70 * W), int(0.45 * H)
    px0, px1, py0 = int(0.22 * W), int(0.78 * W), int(0.62 * H)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        bottom = y + h
        comp = lab == i
        if area < 30:
            continue
        # Large / tall / very close vehicles (e.g. a bus right next to us).
        near = (area > 0.04 * H * W) or (h > 0.30 * H) or (bottom > 0.85 * H)
        # Vehicles ahead in our lane that the player car would drive into.
        ahead = comp[cy0:, cx0:cx1].any() and bottom > 0.55 * H
        in_player_zone = comp[py0:, px0:px1].any()
        if near or ahead or in_player_zone:
            keep[comp] = 1
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        keep = cv2.dilate(keep, k)
        # Extend downward to also cover the vehicle's shadow on the road.
        shifted = np.zeros_like(keep)
        shifted[dilate_px:] = keep[:-dilate_px]
        keep = np.maximum(keep, shifted)
    return keep * 255


class Inpainter:
    def __init__(self):
        self.lama = None
        try:
            from simple_lama_inpainting import SimpleLama
            self.lama = SimpleLama()
        except Exception as e:
            print(f"  [warn] LaMa unavailable ({e}); falling back to cv2.inpaint")

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return img.copy()
        if self.lama is not None:
            try:
                out = np.asarray(self.lama(Image.fromarray(img), Image.fromarray(mask).convert("L")))
                return out[: img.shape[0], : img.shape[1]].copy()
            except Exception as e:
                print(f"  [warn] LaMa failed ({e}); falling back to cv2.inpaint")
        return cv2.inpaint(img, (mask > 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)


# ----------------------------------------------------------------------------
# Player car compositing
# ----------------------------------------------------------------------------
_SPRITE_CACHE: dict[str, Image.Image] = {}


def car_sprite(key: str) -> Image.Image:
    if key not in _SPRITE_CACHE:
        car = PLAYER_CARS[key]
        # Chase cam: ~2 car lengths behind, ~1.5 car heights above the ground.
        _SPRITE_CACHE[key] = render_car(mesh_path(car), car.yaw_deg, view_yaw_deg=0.0,
                                        size=1400, cam_height=0.48, cam_dist=2.0, fov_deg=30.0)
    return _SPRITE_CACHE[key]


def composite_player_car(scene_img: np.ndarray, sprite: Image.Image, width_frac: float = 0.31,
                         bottom_margin: int = 34) -> tuple[np.ndarray, Image.Image]:
    """Paste the car sprite at the bottom-center. Returns (final RGB, RGBA car+shadow layer)."""
    H, W = scene_img.shape[:2]
    cw = int(W * width_frac)
    ch = int(sprite.height * cw / sprite.width)
    car = sprite.resize((cw, ch), Image.LANCZOS)
    x0 = (W - cw) // 2
    y0 = H - bottom_margin - ch

    # Light color matching: pull car brightness/contrast toward the road below it.
    rgb = np.asarray(car.convert("RGB")).astype(np.float32)
    a = np.asarray(car.getchannel("A")).astype(np.float32) / 255.0
    region = scene_img[max(0, y0 - ch // 2):H, x0:x0 + cw].astype(np.float32)
    scene_lum = region.mean()
    car_lum = (rgb.mean(-1) * a).sum() / max(a.sum(), 1)
    gain = float(np.clip((scene_lum / max(car_lum, 1)) ** 0.25, 0.85, 1.12))
    rgb = np.clip((rgb - 128) * 0.95 + 128, 0, 255) * gain
    # Slight tint toward the scene's average color (ambient bounce).
    tint = region.reshape(-1, 3).mean(0) / max(region.mean(), 1)
    rgb = np.clip(rgb * (0.9 + 0.1 * tint), 0, 255)
    car = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA")

    base = Image.fromarray(scene_img).convert("RGBA")
    # Soft contact shadow: a dark blurred ellipse under the wheels plus a
    # tighter, darker core right below the body.
    shadow = Image.new("L", (W, H), 0)
    sh = np.zeros((H, W), np.uint8)
    cx, cy = W // 2, y0 + ch - int(0.04 * ch)
    cv2.ellipse(sh, (cx, cy), (int(cw * 0.56), int(ch * 0.14)), 0, 0, 360, 150, -1)
    cv2.ellipse(sh, (cx, cy), (int(cw * 0.47), int(ch * 0.07)), 0, 0, 360, 220, -1)
    shadow = Image.fromarray(sh).filter(ImageFilter.GaussianBlur(radius=max(6, cw // 40)))
    shadow_alpha = shadow.point(lambda v: int(v * 0.75))
    # Player layer (shadow + car) as straight-alpha RGBA, so downstream code can
    # re-composite the car on every generated video frame.
    layer = Image.new("RGBA", (W, H), (10, 10, 12, 0))
    layer.putalpha(shadow_alpha)
    layer.alpha_composite(car, (x0, y0))
    base.alpha_composite(layer)
    return np.asarray(base.convert("RGB")), layer


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def road_fraction(seg: np.ndarray) -> float:
    """Fraction of road pixels in the lower-center part of the frame."""
    H, W = seg.shape
    roi = seg[int(0.6 * H):, int(0.3 * W):int(0.7 * W)]
    return float((roi == CS_ROAD).mean())


class Pipeline:
    def __init__(self, out_root: str, zoom: int = 4, pitch: float = -3.0, hfov: float = 90.0):
        self.out_root = out_root
        self.zoom, self.pitch, self.hfov = zoom, pitch, hfov
        self.seg = Segmenter()
        self.inpaint = Inpainter()

    def view_for(self, scene: Scene, pano=None):
        """Return (pano, raw_view, heading_deg)."""
        pano = pano or find_official_pano(scene.lat, scene.lon, street=scene.street)
        if pano is None:
            return None, None, None
        pano_heading = math.degrees(pano.heading or 0.0) % 360
        heading = scene.heading if scene.heading is not None else pano_heading + scene.heading_offset
        equi = fetch_equirect(pano, self.zoom)
        raw = equirect_to_perspective(equi, heading - pano_heading, self.pitch, self.hfov)
        return pano, raw, heading % 360

    def process(self, scene: Scene, car_key: str, pano=None, raw=None, heading=None, seg=None) -> dict | None:
        if raw is None:
            pano, raw, heading = self.view_for(scene, pano)
            if pano is None:
                print(f"[{scene.name}] no official panorama near {scene.lat:.5f},{scene.lon:.5f}")
                return None
        if seg is None:
            seg = self.seg(raw)
        mask = np.maximum(near_vehicle_mask(seg), nodata_mask(raw))
        inpainted = self.inpaint(raw, mask)
        final, car_layer = composite_player_car(inpainted, car_sprite(car_key))
        assert final.shape == (OUT_H, OUT_W, 3)

        d = os.path.join(self.out_root, scene.name)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(raw).save(os.path.join(d, "raw.png"))
        Image.fromarray(mask).save(os.path.join(d, "mask.png"))
        Image.fromarray(inpainted).save(os.path.join(d, "inpainted.png"))
        Image.fromarray(final).save(os.path.join(d, "image.png"))
        car_layer.save(os.path.join(d, "car_layer.png"))
        with open(os.path.join(d, "prompt.txt"), "w") as f:
            f.write(scene.prompt + "\n")
        car = PLAYER_CARS[car_key]
        date = pano.date
        meta = {
            "scene": scene.name,
            "kind": scene.kind,
            "type": "highway" if scene.kind == "highway" else "downtown",
            "label": scene.label or scene.name.replace("_", " ").upper(),
            "query_lat": scene.lat, "query_lon": scene.lon,
            "lat": pano.lat, "lon": pano.lon,
            "pano_id": pano.id,
            "pano_heading_deg": round(math.degrees(pano.heading or 0.0) % 360, 2),
            "view_heading_deg": round(heading, 2),
            "view_pitch_deg": self.pitch, "view_hfov_deg": self.hfov,
            "date": str(date) if date is not None else None,
            "address": _pano_streets(pano),
            "source": "Google Street View (official car panorama) via streetlevel",
            "permalink": pano.permalink(heading=heading, pitch=90 + self.pitch, fov=self.hfov),
            "road_fraction": round(road_fraction(seg), 3),
            "masked_fraction": round(float((mask > 0).mean()), 4),
            "player_car": {"key": car_key, "description": car.description,
                           "meshfleet_sha256": car.sha256, "source": car.source_url,
                           "license": car.license},
            "prompt": scene.prompt,
        }
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{scene.name}] pano {pano.id} ({pano.lat:.5f},{pano.lon:.5f}) {meta['date']} "
              f"heading {heading:.1f} road={meta['road_fraction']:.2f} masked={meta['masked_fraction']:.3f} -> {d}")
        return meta


def borough_of(lat, lon) -> str:
    for name, (a, b, c, d) in RANDOM_BOXES.items():
        if a <= lat <= b and c <= lon <= d:
            return name
    return "New York City"


def generic_scene(name, lat, lon, heading=None) -> Scene:
    b = borough_of(lat, lon)
    where = f"a street in {b}" if b != "New York City" else "a street"
    return Scene(name, lat, lon, "street", where,
                 "with buildings, parked cars, crosswalks and traffic lights", heading=heading)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "inputs"))
    ap.add_argument("--scenes", nargs="*", help="subset of curated scene names")
    ap.add_argument("--random", action="store_true", help="random location in Manhattan/Brooklyn/Queens")
    ap.add_argument("--count", type=int, default=1, help="number of random scenes")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--heading", type=float, default=None,
                    help="view heading in degrees (0=N, 90=E); default: capture car direction")
    ap.add_argument("--name", default=None, help="scene name for --lat/--lon or --random")
    ap.add_argument("--prompt-where", default=None, help="override the place phrase in the prompt")
    ap.add_argument("--highway", action="store_true", help="use the highway prompt template")
    ap.add_argument("--car", choices=sorted(PLAYER_CARS), default=None, help="player car model")
    ap.add_argument("--zoom", type=int, default=4, help="Street View zoom level (4 = 8192x4096)")
    ap.add_argument("--pitch", type=float, default=-3.0)
    ap.add_argument("--hfov", type=float, default=90.0)
    args = ap.parse_args(argv)

    pipe = Pipeline(args.out, args.zoom, args.pitch, args.hfov)
    cars = sorted(PLAYER_CARS)
    rng = random.Random(args.seed)

    if args.lat is not None and args.lon is not None:
        sc = generic_scene(args.name or f"custom_{args.lat:.5f}_{args.lon:.5f}", args.lat, args.lon, args.heading)
        if args.highway:
            sc.kind, sc.details = "highway", "with multiple lanes, guardrails and highway traffic"
            sc.where = "a highway"
        if args.prompt_where:
            sc.where = args.prompt_where
        if pipe.process(sc, args.car or rng.choice(cars)) is None:
            sys.exit(1)
        return

    if args.random:
        made = 0
        tries = 0
        while made < args.count and tries < 60:
            tries += 1
            boro = rng.choice(list(RANDOM_BOXES))
            a, b, c, d = RANDOM_BOXES[boro]
            lat, lon = rng.uniform(a, b), rng.uniform(c, d)
            pano = find_official_pano(lat, lon, max_dist=40)
            if pano is None:
                print(f"  no pano at {lat:.5f},{lon:.5f}, retrying")
                continue
            name = args.name if (args.name and args.count == 1) else f"random_{pano.lat:.5f}_{pano.lon:.5f}"
            sc = generic_scene(name, pano.lat, pano.lon, args.heading)
            pano, raw, heading = pipe.view_for(sc, pano)
            seg = pipe.seg(raw)
            if road_fraction(seg) < 0.35:  # e.g. looking at a wall / plaza / park path
                print(f"  {name}: not much road in view ({road_fraction(seg):.2f}), retrying")
                continue
            pipe.process(sc, args.car or rng.choice(cars), pano, raw, heading, seg)
            made += 1
        return

    scenes = CURATED
    if args.scenes:
        scenes = [s for s in CURATED if s.name in set(args.scenes)]
    for s in scenes:
        try:
            pipe.process(s, args.car or s.car or rng.choice(cars))
        except Exception as e:
            print(f"[{s.name}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
