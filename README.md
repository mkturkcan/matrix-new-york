# Matrix New York

**Driving through New York City inside a world model.**

Pick a spot in NYC, either a curated one, a random one, or your own lat/lon. We grab the real Google Street View panorama with [streetlevel](https://github.com/sk-zk/streetlevel), erase the vehicles right around the camera, render a 3D car from [MeshFleet](https://huggingface.co/datasets/DamianBoborzi/MeshFleet) in a third-person chase-cam view, and composite it in. That single frame goes to [Matrix-Game-3.0](https://huggingface.co/Skywork/Matrix-Game-3.0), an action-conditioned interactive world model, which generates every frame after it. You drive with W (throttle), steering (camera yaw), and short strafes for lane changes.

**Project page with videos: https://mkturkcan.github.io/matrix-new-york/**

<!--RESULTS-->

## How it works

```
lat/lon ──► streetlevel ──► equirect pano ──► perspective crop (heading = road direction, 1280×704)
                                                   │
                                Mask2Former (Cityscapes) vehicle masks, near-camera only
                                                   │
                                          LaMa inpainting (erase nearby cars)
                                                   │
          MeshFleet car ──► pyrender chase-cam RGBA sprite ──► composite + contact shadow
                                                   │
                                     first frame + text prompt
                                                   │
             driving actions (W / steer / lane change) ──► Matrix-Game-3.0 ──► video
```

| Stage | File |
|---|---|
| Street View fetch, reprojection, masking, inpainting, compositing | [`nycdrive/prepare_inputs.py`](nycdrive/prepare_inputs.py) |
| MeshFleet chase-cam car renderer | [`nycdrive/car_render.py`](nycdrive/car_render.py) |
| Driving routes → Matrix-Game-3 keyboard/mouse action tracks | [`nycdrive/routes.py`](nycdrive/routes.py) |
| Matrix-Game-3 wrapper (batch and interactive car controls) | [`nycdrive/drive.py`](nycdrive/drive.py) |
| Game HUD (location, keys, speedometer, steering wheel) | [`nycdrive/hud.py`](nycdrive/hud.py) |
| Multi-GPU batch rendering, site builder | [`scripts/`](scripts/) |

### Actions as car controls

Matrix-Game-3 takes a 6-dim keyboard vector (`W S A D …`) and a 2-dim mouse vector (pitch, yaw; 15°/unit/frame) for every frame. We map them like this:

| Car | Matrix-Game-3 action |
|---|---|
| throttle | `W` |
| brake / reverse | `S` |
| steering wheel | mouse yaw (±0.03 gentle curve, ±0.06 steer, ±0.10 for a 90° intersection turn) |
| lane change | `W` + short `A`/`D` strafe |

Route presets (`cruise`, `highway`, `downtown`, `weave`, `sightseeing`) live in `nycdrive/routes.py`. You can also pass a JSON list of `[seconds, keys, yaw]` segments.

## Setup

```bash
git clone https://github.com/mkturkcan/matrix-new-york && cd matrix-new-york
# Matrix-Game-3 code (only the Matrix-Game-3 folder is needed)
git clone --depth 1 --filter=blob:none --sparse https://github.com/SkyworkAI/Matrix-Game.git third_party/Matrix-Game
git -C third_party/Matrix-Game sparse-checkout set Matrix-Game-3
# weights (~43 GB without base_model/)
hf download Skywork/Matrix-Game-3.0 --local-dir checkpoints/Matrix-Game-3.0 --exclude "base_model/*"
# python deps (install torch + flash-attn for your CUDA first)
pip install -r requirements.txt
```

Tested on 4× RTX PRO 6000 Blackwell (96 GB) with torch 2.12 + CUDA 13.0 and flash-attn 2. One scene fits comfortably on a single GPU, so `scripts/render_showcase.py` runs one scene per GPU in parallel.

## Usage

```bash
# 1. Build first frames
python -m nycdrive.prepare_inputs                                   # curated highway + downtown scenes
python -m nycdrive.prepare_inputs --random                          # random spot in NYC
python -m nycdrive.prepare_inputs --lat 40.7580 --lon -73.9855      # your own spot

# 2a. Drive a scripted route → outputs/videos/<name>_{clean,hud}.mp4
python -m nycdrive.drive --scene data/inputs/<scene> --route highway --num_iterations 10

# 2b. Play interactively: the model pauses after each chunk for your next command
python -m nycdrive.drive --scene data/inputs/<scene> --interactive
#   w gas · s brake · a/d steer · wa/wd turn while accelerating · z/c lane change · q coast

# 3. Batch render + build the project page
python scripts/render_showcase.py configs/showcase.json --gpus 0,1,2,3
python scripts/build_site.py configs/showcase.json --pipeline_scene <scene>
```

Extra flags after `--` are passed straight to Matrix-Game-3's `generate.py`. For example, `-- --use_int8` turns on int8 DiT quantization.

## Limitations

- Every frame after the first is generated. Buildings, signs, and traffic are plausible, not a faithful map of NYC, and the model drifts further from reality the longer you drive.
- Matrix-Game-3 was trained mostly on walking and game-camera motion, not cars. The chase-cam car is usually kept in frame, but it can morph or fade on long rollouts and sharp turns.
- Street View coverage, capture date, and weather depend on the panorama found nearest the requested point.

## Credits & licenses

- **Matrix-Game-3.0** by Skywork AI: [model](https://huggingface.co/Skywork/Matrix-Game-3.0), [code](https://github.com/SkyworkAI/Matrix-Game), Apache-2.0. Built on Wan2.2.
- **streetlevel** by sk-zk: [repo](https://github.com/sk-zk/streetlevel), MIT. Street View imagery © Google; use it in line with Google's terms of service.
- **MeshFleet** by Boborzi et al.: [dataset](https://huggingface.co/datasets/DamianBoborzi/MeshFleet). The car meshes come from Objaverse-XL / Sketchfab and keep their per-model licenses; see `nycdrive/car_render.py`.
- **Mask2Former** (Meta, Cityscapes semantic checkpoint) and **LaMa** (Samsung AI, via `simple-lama-inpainting`) for masking and inpainting.

The code in this repository is released under Apache-2.0. This is an independent research demo, not affiliated with Skywork, Google, or the dataset authors.
