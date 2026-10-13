"""Encode rollouts of every backend into docs/ and generate docs/index.html (GitHub Pages).

Inputs:
    configs/models.json   model registry (name, colour, facts, output dir, job config); entries with
                          "pending": true are skipped
    configs/page.json     page text (abstract, stats, per-model mapping table, analysis, limitations)
    configs/showcase*.json  jobs per backend (scene, route, title, group); outputs without a job entry
                          are picked up automatically and titled from the MG3 jobs or meta.json

    python scripts/build_site.py --pipeline_scene times_square --hero mg3/fdr_drive_highway
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
SITE_URL = "https://mehmetkeremturkcan.com/matrix-new-york"
FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def encode(src: Path, dst: Path, crf=27, poster=True):
    if not (dst.exists() and dst.stat().st_mtime > src.stat().st_mtime):
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)
    if poster and not dst.with_suffix(".jpg").exists():
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(dst), "-frames:v", "1", "-q:v", "3",
                        str(dst.with_suffix(".jpg"))], check=True)


def name_of(j):
    return j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"


def frames_at(path: Path, idx):
    out = {}
    for i, f in enumerate(imageio.get_reader(path)):
        if i in idx:
            out[i] = f
        if i >= max(idx):
            break
    return [out[i] for i in idx]


def mean_flow(path: Path, start=34, n=136):
    """Mean Farneback flow magnitude (px/frame) on a mid-image band at 320 px width."""
    prev, mags = None, []
    for i, f in enumerate(imageio.get_reader(path)):
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


def discover_jobs(model, ref_jobs):
    """Jobs from the model's config, plus any finished outputs not listed there."""
    cfg = ROOT / model["config"]
    jobs = [j for j in json.loads(cfg.read_text()) if j.get("group") != "ablation"] if cfg.exists() else []
    known = {name_of(j) for j in jobs}
    ref = {name_of(j): j for j in ref_jobs}
    scenes = sorted((p.name for p in (ROOT / "data" / "inputs").iterdir() if p.is_dir()), key=len, reverse=True)
    for f in sorted((ROOT / model["videos_dir"]).glob("*_clean.mp4")):
        n = f.name[: -len("_clean.mp4")]
        if n in known:
            continue
        scene = next((s for s in scenes if n.startswith(s + "_")), None)
        if scene is None:
            continue
        route = n[len(scene) + 1:]
        base = ref.get(n) or next((j for j in ref_jobs if Path(j["scene"]).name == scene), None)
        meta = json.loads((ROOT / "data" / "inputs" / scene / "meta.json").read_text())
        jobs.append({"scene": f"data/inputs/{scene}", "route": route,
                     "title": base["title"] if base else meta.get("label", scene).title(),
                     "group": base["group"] if base else ("Highway" if meta.get("type") == "highway" else "Urban")})
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline_scene", default="times_square")
    p.add_argument("--hero", default="mg3/fdr_drive_highway")
    p.add_argument("--ablation", default="ablation_vehicle_in_frame")
    p.add_argument("--ablation_ref", default="times_square_downtown")
    args = p.parse_args()

    models = [m for m in json.loads((ROOT / "configs" / "models.json").read_text()) if not m.get("pending")]
    page = json.loads((ROOT / "configs" / "page.json").read_text())
    ref_jobs = json.loads((ROOT / "configs" / "showcase.json").read_text())

    clips, panos = [], []
    for m in models:
        vdir = ROOT / m["videos_dir"]
        for j in discover_jobs(m, ref_jobs):
            n = name_of(j)
            src, act = vdir / f"{n}_clean.mp4", vdir / f"{n}_actions.npz"
            if not (src.exists() and act.exists()):
                print(f"[{m['id']}] missing {n}")
                continue
            a = np.load(act)
            if "controls" not in a or len(a["controls"]) == 0:
                print(f"[{m['id']}] stale (no controller state), skipping {n}")
                continue
            stride = int(a["stride"]) if "stride" in a else 1
            dst = DOCS / "videos" / m["id"] / f"{n}.mp4"
            encode(src, dst)
            reader = imageio.get_reader(dst)
            nframes, fps = reader.count_frames(), float(reader.get_meta_data().get("fps", 17))
            ctrl = a["controls"][::stride][:nframes]
            meta = json.loads((ROOT / j["scene"] / "meta.json").read_text())
            clips.append({"id": f"{m['id']}/{n}", "model": m["id"], "scene": Path(j["scene"]).name, "title": j["title"],
                          "group": j["group"], "route": j["route"], "lat": meta["lat"], "lon": meta["lon"],
                          "date": meta.get("date", ""), "src": f"videos/{m['id']}/{n}.mp4", "poster": f"videos/{m['id']}/{n}.jpg",
                          "frames": nframes, "fps": round(fps, 3),
                          "playback": f"{stride}× playback" if stride > 1 else "real time",
                          "steer": [round(float(x), 2) for x in ctrl[:, 1]], "thr": [round(float(x), 2) for x in ctrl[:, 0]]})
            pano = vdir / f"{n}_pano.mp4"
            if pano.exists():
                pdst = DOCS / "videos" / m["id"] / f"{n}_pano.mp4"
                encode(pano, pdst, crf=24, poster=False)
                panos.append({"title": j["title"], "src": f"videos/{m['id']}/{pdst.name}"})
            print(f"[{m['id']}] {n}: {nframes} frames, {dst.stat().st_size / 1e6:.1f} MB")

    # comparison sets: same scene + route rendered by >= 2 models
    order = {m["id"]: k for k, m in enumerate(models)}
    groups = {}
    for c in clips:
        groups.setdefault((c["scene"], c["route"]), []).append(c)
    compare = [{"title": f"{v[0]['title']}, {v[0]['route']} route",
                "items": [{"model": c["model"], "src": c["src"], "poster": c["poster"], "playback": c["playback"]}
                          for c in sorted(v, key=lambda c: order[c["model"]])]}
               for v in groups.values() if len({c["model"] for c in v}) >= 2]

    hero = next((c for c in clips if c["id"] == args.hero), clips[0])
    clips.sort(key=lambda c: (order[c["model"]], c["id"] != hero["id"]))

    # pipeline figure
    sdir = ROOT / "data" / "inputs" / args.pipeline_scene
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    steps = [("raw.png", "(a) Perspective view along the capture heading."),
             ("mask.png", "(b) Near-field vehicles selected for removal (red)."),
             ("inpainted.png", "(c) LaMa inpainting. Conditioning frame for first-person models."),
             ("image.png", "(d) MeshFleet vehicle rendered under the chase camera.")]
    step_html = ""
    for fn, cap in steps:
        im = Image.open(sdir / fn).convert("RGB")
        if fn == "mask.png":
            raw = Image.open(sdir / "raw.png").convert("RGB")
            im = Image.composite(Image.blend(raw, Image.new("RGB", raw.size, (255, 70, 70)), 0.65), raw, im.convert("L"))
        out = DOCS / "assets" / f"step_{Path(fn).stem}.jpg"
        im.save(out, quality=88)
        step_html += f'<div><img src="assets/{out.name}" alt="{html.escape(cap)}" loading="lazy"><p>{cap}</p></div>'
    meta = json.loads((sdir / "meta.json").read_text())
    cmp_caption = f"{html.escape(meta.get('label', '').title())}, panorama {meta.get('date')}, ({meta['lat']:.4f}, {meta['lon']:.4f})."

    # MG3 ablation strip: vehicle in the conditioning frame vs vehicle-free conditioning + composite
    strip, vids = "", ROOT / "outputs" / "videos"
    abl, ref = vids / f"{args.ablation}_world.mp4", vids / f"{args.ablation_ref}_clean.mp4"
    if abl.exists() and ref.exists():
        times = [0.0, 0.5, 1.0, 2.0]
        idx = [int(round(t * 17)) for t in times]
        strip += '<div class="corner"></div>' + "".join(f'<div class="ch">t = {t:.1f} s</div>' for t in times)
        for label, sub, path, key in [("Vehicle in frame", "conditioned on (d)", abl, "a"), ("Ours", "conditioned on (c) + composite", ref, "b")]:
            strip += f'<div class="rl"><b>{label}</b>{sub}</div>'
            for t, f in zip(times, frames_at(path, idx)):
                out = DOCS / "assets" / f"ablation_{key}_{t:.1f}.jpg"
                Image.fromarray(f).resize((640, 352), Image.LANCZOS).save(out, quality=86)
                strip += f'<img src="assets/{out.name}" alt="{label}, t={t:.1f}s" loading="lazy">'
    flow = vids / "fdr_drive_highway_world.mp4"
    flow_txt = f"{mean_flow(flow):.2f}" if flow.exists() else "n/a"

    # hero: crossfaded montage of rollout openings (raw world-model output), or the first 12 s of one rollout
    hero_src = DOCS / "videos" / "hero.mp4"
    vdirs = {m["id"]: ROOT / m["videos_dir"] for m in models}
    segs = [s for s in page.get("hero_montage", []) if (vdirs.get(s[0], ROOT) / f"{s[1]}_world.mp4").exists()]
    if segs:
        xf, cmd, chain = 0.6, [FFMPEG, "-y", "-loglevel", "error"], ""
        for mid, n, start, dur in segs:
            cmd += ["-ss", str(start), "-t", str(dur), "-i", str(vdirs[mid] / f"{n}_world.mp4")]
        prev, offset = "[0:v]", 0.0
        for k in range(1, len(segs)):
            offset += segs[k - 1][3] - xf
            chain += f"{prev}[{k}:v]xfade=transition=fade:duration={xf}:offset={offset:.2f}[x{k}];"
            prev = f"[x{k}]"
        total = sum(sg[3] for sg in segs) - xf * (len(segs) - 1)
        chain += f"{prev}fade=t=in:st=0:d=0.4,fade=t=out:st={total - 0.4:.2f}:d=0.4,format=yuv420p[v]"
        subprocess.run(cmd + ["-filter_complex", chain, "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                              "-movflags", "+faststart", "-an", str(hero_src)], check=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "1.0", "-i", str(hero_src), "-frames:v", "1", "-q:v", "3",
                        str(DOCS / "videos" / "hero.jpg")], check=True)
        hero_poster, hero_cap = "videos/hero.jpg", page["hero_caption"]
    else:
        hm = next(m for m in models if m["id"] == hero["model"])
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(vdirs[hm["id"]] / f"{hero['id'].split('/')[1]}_clean.mp4"),
                        "-t", "12", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(hero_src)], check=True)
        hero_poster = hero["poster"]
        hero_cap = (f"{hm['name']} rollout from a single Street View frame ({hero['title']}). "
                    f"First 12 s of {hero['frames'] / hero['fps']:.0f} s, {hero['playback']}.")

    lims = "".join(f"<li>{t}</li>" for t in page["limitations"])
    mapping = ("<thead><tr><th></th>" + "".join(f'<th>{m["name"]}</th>' for m in models) + "</tr></thead><tbody>"
               + "".join("<tr><td>" + row[0] + "</td>" + "".join(f"<td>{row[1].get(m['id'], '—')}</td>" for m in models) + "</tr>"
                         for row in page["mapping"]) + "</tbody>")
    subs = {"{{HERO_SRC}}": "videos/hero.mp4", "{{HERO_POSTER}}": hero_poster, "{{HERO_CAPTION}}": html.escape(hero_cap),
            "{{OG_IMAGE}}": f"{SITE_URL}/{hero_poster}", "{{STEPS}}": step_html, "{{CMP_A}}": "assets/step_raw.jpg",
            "{{CMP_B}}": "assets/step_image.jpg", "{{CMP_CAPTION}}": cmp_caption, "{{STRIP}}": strip, "{{FLOW_STRIDE}}": flow_txt,
            "{{ABSTRACT}}": page["abstract"], "{{MODELS_LEDE}}": page["models_lede"],
            "{{PANO_LEDE}}": page["pano_lede"], "{{MAPPING_TABLE}}": mapping, "{{RESULTS_LEDE}}": page["results_lede"],
            "{{ANALYSIS_EXTRA}}": page["analysis_extra"], "{{LIMITATIONS}}": lims, "{{REFS_EXTRA}}": page["refs_extra"]}
    out = (DOCS / "template.html").read_text()
    for k, v in subs.items():
        out = out.replace(k, v)
    assert "{{" not in out, "unfilled placeholder"
    s, e = out.index("/*DATA*/") + len("/*DATA*/"), out.index("/*END*/")
    site = {"models": [{k: m[k] for k in ("id", "name", "org", "color", "link", "facts", "summary")} for m in models],
            "clips": clips, "compare": compare, "pano": panos}
    out = out[:s] + json.dumps(site, separators=(",", ":"), ensure_ascii=False) + out[e:]
    (DOCS / "index.html").write_text(out)
    print(f"wrote docs/index.html: {len(models)} models, {len(clips)} clips, {len(compare)} comparisons, {len(panos)} panoramas")


if __name__ == "__main__":
    main()
