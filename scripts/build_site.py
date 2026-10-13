"""Encode rollouts into docs/ and generate docs/index.html (GitHub Pages).

    python scripts/build_site.py --config configs/showcase.json --pipeline_scene times_square --hero fdr_drive_highway
"""
import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VIDS = ROOT / "outputs" / "videos"
SITE_URL = "https://mehmetkeremturkcan.com/matrix-new-york"
FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
FPS = 17


def encode(src: Path, dst: Path, crf=27):
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(dst), "-frames:v", "1", "-q:v", "3",
                    str(dst.with_suffix(".jpg"))], check=True)


def name_of(j):
    return j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"


def frames_at(path: Path, idx):
    r = imageio.get_reader(path)
    out = {}
    for i, f in enumerate(r):
        if i in idx:
            out[i] = f
        if i >= max(idx):
            break
    return [out[i] for i in idx]


def mean_flow(path: Path, start=34, n=136):
    """Mean Farneback flow magnitude (px/frame) on a mid-image band at 320 px width."""
    r = imageio.get_reader(path)
    prev, mags = None, []
    for i, f in enumerate(r):
        if i < start:
            continue
        if i >= start + n:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, 176)), cv2.COLOR_RGB2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(np.linalg.norm(flow[60:120], axis=-1).mean())
        prev = g
    return float(np.mean(mags))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/showcase.json")
    p.add_argument("--pipeline_scene", default="times_square")
    p.add_argument("--hero", default="fdr_drive_highway")
    p.add_argument("--ablation", default="ablation_vehicle_in_frame")
    p.add_argument("--ablation_ref", default="times_square_downtown")
    args = p.parse_args()

    jobs = [j for j in json.loads((ROOT / args.config).read_text()) if j.get("group") != "ablation"]
    clips = []
    for j in jobs:
        n = name_of(j)
        src = VIDS / f"{n}_clean.mp4"
        if not src.exists():
            print("missing", src)
            continue
        a = np.load(VIDS / f"{n}_actions.npz")
        if "controls" not in a or len(a["controls"]) == 0:
            print("stale (rendered without controller state), skipping", n)
            continue
        dst = DOCS / "videos" / f"{n}.mp4"
        encode(src, dst)
        stride = int(a["stride"]) if "stride" in a else 1
        ctrl = a["controls"][::stride]
        nframes = imageio.get_reader(dst).count_frames()
        ctrl = ctrl[:nframes]
        m = json.loads((ROOT / j["scene"] / "meta.json").read_text())
        clips.append({"id": n, "scene": Path(j["scene"]).name, "title": j["title"], "group": j["group"], "route": j["route"],
                      "lat": m["lat"], "lon": m["lon"], "date": m.get("date", ""), "src": f"videos/{n}.mp4",
                      "poster": f"videos/{n}.jpg", "frames": nframes, "fps": FPS,
                      "steer": [round(float(x), 2) for x in ctrl[:, 1]], "thr": [round(float(x), 2) for x in ctrl[:, 0]]})
        print("clip", n, nframes, "frames", round(dst.stat().st_size / 1e6, 1), "MB")
    hero = next((c for c in clips if c["id"] == args.hero), clips[0])
    clips.sort(key=lambda c: c["id"] != hero["id"])  # hero first in the explorer

    # pipeline figure
    sdir = ROOT / "data" / "inputs" / args.pipeline_scene
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    steps = [("raw.png", "(a) Street View", "Perspective view along the capture heading."),
             ("mask.png", "(b) Near-field mask", "Mask2Former vehicle components selected for removal."),
             ("inpainted.png", "(c) Inpainted", "LaMa fill; this frame conditions the world model."),
             ("image.png", "(d) Ego vehicle", "MeshFleet render under the chase camera.")]
    step_html = ""
    for fn, title, cap in steps:
        im = Image.open(sdir / fn).convert("RGB")
        if fn == "mask.png":
            raw = Image.open(sdir / "raw.png").convert("RGB")
            im = Image.composite(Image.blend(raw, Image.new("RGB", raw.size, (255, 70, 70)), 0.65), raw, im.convert("L"))
        out = DOCS / "assets" / f"step_{Path(fn).stem}.jpg"
        im.save(out, quality=88)
        step_html += (f'<div class="step"><img src="assets/{out.name}" alt="{html.escape(title)}" loading="lazy">'
                      f'<div class="cap"><b>{title.upper()}</b>{cap}</div></div>')
    m = json.loads((sdir / "meta.json").read_text())
    cmp_caption = (f"{html.escape(m.get('label', '').title())}, panorama {m.get('date')}, "
                   f"({m['lat']:.4f}, {m['lon']:.4f}).")

    # ablation strip: vehicle in conditioning frame vs vehicle-free conditioning + composite
    strip = ""
    abl, ref = VIDS / f"{args.ablation}_world.mp4", VIDS / f"{args.ablation_ref}_clean.mp4"
    if abl.exists() and ref.exists():
        times = [0.0, 0.5, 1.0, 2.0]
        idx = [int(round(t * FPS)) for t in times]
        strip += '<div class="corner"></div>' + "".join(f'<div class="ch">t = {t:.1f} s</div>' for t in times)
        for label, sub, path, key in [("Vehicle in frame", "conditioned on (d)", abl, "a"), ("Ours", "conditioned on (c) + composite", ref, "b")]:
            strip += f'<div class="rl"><b>{label}</b>{sub}</div>'
            for t, f in zip(times, frames_at(path, idx)):
                out = DOCS / "assets" / f"ablation_{key}_{t:.1f}.jpg"
                Image.fromarray(f).resize((640, 352), Image.LANCZOS).save(out, quality=86)
                strip += f'<img src="assets/{out.name}" alt="{label}, t={t:.1f}s" loading="lazy">'
    else:
        print("ablation videos missing")

    flow = VIDS / f"{args.hero}_world.mp4"
    flow_txt = f"{mean_flow(flow):.2f}" if flow.exists() else "n/a"
    print("flow stride-2", flow_txt)

    # hero: first 12 s (displayed) of the hero rollout, stated in the caption
    hero_src = DOCS / "videos" / "hero.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(VIDS / f"{hero['id']}_clean.mp4"), "-t", "12", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(hero_src)], check=True)
    hero_cap = (f"Matrix-Game-3.0 rollout from one Street View frame · {hero['title']} · "
                f"first 12 s of {hero['frames'] / FPS:.0f} s, 2× playback")
    tpl = (DOCS / "template.html").read_text()
    out = (tpl.replace("{{HERO_SRC}}", "videos/hero.mp4").replace("{{HERO_POSTER}}", hero["poster"])
              .replace("{{HERO_CAPTION}}", html.escape(hero_cap)).replace("{{OG_IMAGE}}", f"{SITE_URL}/{hero['poster']}")
              .replace("{{STEPS}}", step_html).replace("{{CMP_A}}", "assets/step_raw.jpg").replace("{{CMP_B}}", "assets/step_image.jpg")
              .replace("{{CMP_CAPTION}}", cmp_caption).replace("{{STRIP}}", strip).replace("{{FLOW_STRIDE}}", flow_txt))
    s, e = out.index("/*DATA*/") + len("/*DATA*/"), out.index("/*END*/")
    out = out[:s] + json.dumps({"clips": clips}, separators=(",", ":")) + out[e:]
    (DOCS / "index.html").write_text(out)
    print("wrote docs/index.html with", len(clips), "clips")


if __name__ == "__main__":
    main()
