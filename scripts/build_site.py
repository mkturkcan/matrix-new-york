"""Compress rendered videos into docs/ and generate docs/index.html for GitHub Pages.

    python scripts/build_site.py configs/showcase.json --pipeline_scene times_square
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def encode(src: Path, dst: Path, crf=27):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def poster(src: Path, dst: Path):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "3", "-i", str(src), "-frames:v", "1",
                    "-q:v", "4", str(dst)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--pipeline_scene", default=None)
    p.add_argument("--hero", default=None, help="save_name of the hero video")
    args = p.parse_args()

    jobs = json.loads(Path(args.config).read_text())
    vids_dir = DOCS / "videos"
    videos = []
    for j in jobs:
        name = j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"
        src = ROOT / "outputs" / "videos" / f"{name}_hud.mp4"
        if not src.exists():
            print("missing", src)
            continue
        encode(src, vids_dir / f"{name}.mp4")
        poster(vids_dir / f"{name}.mp4", vids_dir / f"{name}.jpg")
        videos.append({"src": f"videos/{name}.mp4", "poster": f"videos/{name}.jpg", "group": j["group"],
                       "title": j["title"], "sub": j.get("sub", f"route: {j['route']}")})
        print("encoded", name, round((vids_dir / f"{name}.mp4").stat().st_size / 1e6, 1), "MB")

    hero = ""
    if videos:
        hero_name = args.hero or Path(videos[0]["src"]).stem
        clean = ROOT / "outputs" / "videos" / f"{hero_name}_clean.mp4"
        if clean.exists():
            encode(clean, vids_dir / "hero.mp4", crf=28)
            hero = "videos/hero.mp4"
        else:
            hero = f"videos/{hero_name}.mp4"

    pipeline = []
    if args.pipeline_scene:
        sdir = ROOT / "data" / "inputs" / args.pipeline_scene
        steps = [("raw.png", "Street View panorama, reprojected to look down the road"),
                 ("mask.png", "Vehicles near the camera segmented (Mask2Former)"),
                 ("inpainted.png", "Nearby cars erased with LaMa inpainting"),
                 ("image.png", "MeshFleet car rendered in chase-cam view and composited")]
        (DOCS / "assets").mkdir(parents=True, exist_ok=True)
        for fn, title in steps:
            if (sdir / fn).exists():
                out = DOCS / "assets" / f"step_{Path(fn).stem}.jpg"
                im = Image.open(sdir / fn).convert("RGB")
                if fn == "mask.png":  # show the mask as a red overlay on the panorama crop
                    raw = Image.open(sdir / "raw.png").convert("RGB")
                    red = Image.new("RGB", raw.size, (255, 40, 40))
                    im = Image.composite(Image.blend(raw, red, 0.6), raw, im.convert("L"))
                im.save(out, quality=85)
                pipeline.append({"src": f"assets/{out.name}", "title": title})

    data = json.dumps({"videos": videos, "pipeline": pipeline, "hero": hero})
    tpl = (DOCS / "template.html").read_text()
    start, end = tpl.index("/*DATA*/") + len("/*DATA*/"), tpl.index("/*END*/")
    (DOCS / "index.html").write_text(tpl[:start] + data + tpl[end:])
    print("wrote docs/index.html with", len(videos), "videos")


if __name__ == "__main__":
    main()
