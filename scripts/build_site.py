"""Encode rendered clips into docs/ and generate docs/index.html (GitHub Pages).

    python scripts/build_site.py --gta configs/showcase_gta.json --comparison configs/comparison.json \
        --pipeline_scene times_square --hero fdr_drive_highway
"""
import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def encode(src: Path, dst: Path, crf=28):
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(dst), "-frames:v", "1", "-q:v", "4",
                    str(dst.with_suffix(".jpg"))], check=True)


def n_frames(path: Path) -> int:
    import imageio
    return imageio.get_reader(path).count_frames()


def name_of(j):
    return j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"


def caption(j, frames, fps):
    m = json.loads((ROOT / j["scene"] / "meta.json").read_text())
    return (f"<b>{html.escape(j['title'])}</b>. Route <code>{j['route']}</code>; panorama {m.get('date', '?')}, "
            f"({m['lat']:.4f}, {m['lon']:.4f}); {frames} frames at {fps} fps.")


def video_fig(src, cap, autoplay=False):
    attrs = "autoplay muted loop playsinline controls" if autoplay else 'muted loop playsinline controls preload="none"'
    return (f'<figure><video src="{src}" poster="{Path(src).with_suffix(".jpg")}" {attrs}></video>'
            f"<figcaption>{cap}</figcaption></figure>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gta", default="configs/showcase_gta.json")
    p.add_argument("--comparison", default="configs/comparison.json")
    p.add_argument("--pipeline_scene", default="times_square")
    p.add_argument("--hero", default="fdr_drive_highway")
    args = p.parse_args()

    # Matrix-Game-2.0 GTA-drive results, grouped
    gta_jobs = json.loads((ROOT / args.gta).read_text())
    groups, hero = {}, None
    for j in gta_jobs:
        src = ROOT / "outputs" / "videos_gta" / f"{name_of(j)}_hud.mp4"
        if not src.exists():
            print("missing", src)
            continue
        dst = DOCS / "videos" / "gta" / f"{name_of(j)}.mp4"
        encode(src, dst)
        cap = caption(j, n_frames(dst), 12)
        groups.setdefault(j["group"], []).append(video_fig(f"videos/gta/{dst.name}", cap))
        if name_of(j) == args.hero:
            hero = (f"videos/gta/{dst.name}", "Matrix-Game-2.0 GTA-drive. " + cap)
        print("gta", dst.name, round(dst.stat().st_size / 1e6, 1), "MB")
    gta_html = "".join(f'<h3>{html.escape(g)}</h3><div class="grid2">{"".join(v)}</div>' for g, v in groups.items())

    # Backend comparison: same scene and route
    comp_html = ""
    cpath = ROOT / args.comparison
    for j in (json.loads(cpath.read_text()) if cpath.exists() else []):
        n = name_of(j)
        a = ROOT / "outputs" / "videos_gta" / f"{n}_hud.mp4"
        b = ROOT / "outputs" / "videos" / f"{n}_hud.mp4"
        if not (a.exists() and b.exists()):
            print("missing comparison", n)
            continue
        da, db = DOCS / "videos" / "gta" / f"{n}.mp4", DOCS / "videos" / "mg3" / f"{n}.mp4"
        if not da.exists():
            encode(a, da)
        encode(b, db)
        m = json.loads((ROOT / j["scene"] / "meta.json").read_text())
        comp_html += (f'<div class="pair">{video_fig(f"videos/gta/{da.name}", "<b>Matrix-Game-2.0 GTA-drive</b>, 352×640 (2× upsampled), 12 fps.")}'
                      f'{video_fig(f"videos/mg3/{db.name}", "<b>Matrix-Game-3.0</b>, 704×1280, 17 fps.")}</div>'
                      f'<p class="note">{html.escape(j["title"])}, route <code>{j["route"]}</code>; panorama {m.get("date")}, '
                      f'({m["lat"]:.4f}, {m["lon"]:.4f}).</p>')
        print("comparison", n)

    # Pipeline figure
    sdir = ROOT / "data" / "inputs" / args.pipeline_scene
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    steps = [("raw.png", "(a) Perspective view along the capture heading."),
             ("mask.png", "(b) Near-field vehicle mask (red)."),
             ("inpainted.png", "(c) LaMa inpainting; Matrix-Game-3.0 conditioning frame."),
             ("image.png", "(d) Inserted MeshFleet vehicle; Matrix-Game-2.0 GTA-drive conditioning frame.")]
    figs = []
    for fn, title in steps:
        im = Image.open(sdir / fn).convert("RGB")
        if fn == "mask.png":
            raw = Image.open(sdir / "raw.png").convert("RGB")
            im = Image.composite(Image.blend(raw, Image.new("RGB", raw.size, (230, 30, 30)), 0.6), raw, im.convert("L"))
        out = DOCS / "assets" / f"step_{Path(fn).stem}.jpg"
        im.save(out, quality=88)
        figs.append(f'<figure><img src="assets/{out.name}" alt="{html.escape(title)}" loading="lazy"><figcaption>{title}</figcaption></figure>')
    m = json.loads((sdir / "meta.json").read_text())
    pipeline_html = (f'<div class="grid4">{"".join(figs)}</div><p class="note">Example: {html.escape(m.get("label", ""))}, '
                     f'panorama {m.get("date")}, ({m["lat"]:.4f}, {m["lon"]:.4f}).</p>')

    tpl = (DOCS / "template.html").read_text()
    out = (tpl.replace("{{HERO_SRC}}", hero[0] if hero else "")
              .replace("{{HERO_CAPTION}}", hero[1] if hero else "")
              .replace("{{GTA_RESULTS}}", gta_html)
              .replace("{{COMPARISON}}", comp_html or '<p class="note">(not rendered)</p>')
              .replace("{{PIPELINE}}", pipeline_html))
    (DOCS / "index.html").write_text(out)
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
