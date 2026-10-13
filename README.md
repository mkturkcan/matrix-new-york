# Matrix New York

Street View initial observations and vehicle control for interactive video world models, applied to New York City.

Project page: https://mehmetkeremturkcan.com/matrix-new-york/

## Summary

For a query location we:

1. retrieve the nearest Google Street View panorama captured by a Street View car (via [streetlevel](https://github.com/sk-zk/streetlevel));
2. remove near-field vehicles (Mask2Former, Cityscapes classes; LaMa inpainting);
3. insert a vehicle mesh from [MeshFleet](https://huggingface.co/datasets/DamianBoborzi/MeshFleet) rendered under a chase camera.

A kinematic bicycle model converts throttle and steering into the action space of three pretrained world models. The ego vehicle is composited from 17 renders indexed by the controller's yaw relative to the camera. No model is fine-tuned, and each rollout runs on one GPU.

| | [Matrix-Game-3.0](https://huggingface.co/Skywork/Matrix-Game-3.0) | [Matrix-Game-3.5](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Distilled) | [PanoWorld](https://huggingface.co/Insta360-Research/PanoWorld) |
|---|---|---|---|
| Conditioning | vehicle-free perspective frame, text | vehicle-free perspective frame, text | 360° panorama (nearby vehicles inpainted), text |
| Control | keyboard `W` + camera yaw per frame | camera-to-world trajectory (OpenCV, metric) | camera translation, fixed orientation |
| Memory | camera-aware latent retrieval | Patch Memory (Depth Anything 3) | none released |
| Output | 704×1280, 17 fps | 704×1280, 16 fps | 960×480 equirectangular, 16 fps |
| Adapter | `W` duty cycle `min(v / 10 m/s, 1)`, yaw `Δψ_c / 15°` per frame | chase-camera pose along the Street View capture path | vehicle path as translation; perspective crop at `ψ_c` |
| Time per clip (1 shared RTX PRO 6000) | 4.8 min for 817 frames | 8 min for 337 frames | 2 min for 321 frames |
| Entry point | `nycdrive/drive.py` | `nycdrive/drive_mg35.py` | `nycdrive/drive_panoworld.py` |

## Observations

- **Ego vehicle and conditioning.** Matrix-Game-3.0 has no ego-vehicle prior. A vehicle composited into the conditioning frame is treated as static scene content and smears across the lower image within about a second. All perspective backends are therefore conditioned on the inpainted frame, and the vehicle is composited afterwards.
- **Matrix-Game-3.0 forward speed.** The model advances a roughly constant distance per frame while `W` is held. Scaling the camera translation used for its Plücker-ray conditioning by 3× left mean Farnebäck flow unchanged (0.65 vs 0.66 px/frame at 320 px width). We therefore set apparent speed by temporal subsampling: the controller runs at 34 Hz and every second generated frame is kept (`--frame_stride 2`).
- **Matrix-Game-3.5 road following.** The model follows the commanded camera path closely, but it does not follow the road. Driving the route relative to the Street View capture path reduces the drift without removing it: in all six scenes the camera leaves the roadway after 6 to 15 s, with a consistent bias to the right. For the first 5 to 10 s it is the sharpest of the three models. Patch Memory is required; without it, a straight path reaches the barrier within 5 s.
- **PanoWorld.** It keeps the real scene layout recognisable for the whole 20 s clip, but the released model is 480p only:
  - A 90° crop covers about 240 of the 960 panorama columns and is upsampled about 5×, so the driving view is soft.
  - Apparent speed saturates at a slow roll. Steps above about 0.2 units per frame streak, and the scene dissolves above 0.4.
  - With `--anchor`, each 81-frame segment restarts from the real panorama nearest the vehicle's position. This keeps the rollout tied to real imagery, with cuts at segment boundaries.
- **Turns.** Large heading changes away from intersections steer the camera into facades.

## Vehicle model

Inputs: throttle/brake `u ∈ [-1, 1]` and steering `s ∈ [-1, 1]` (low-pass filtered, τ = 0.15 s).

$$
\dot v = a^{+}u^{+} - a^{-}u^{-} - c\,v,\qquad
\delta = \frac{s\,\delta_{\max}}{1 + v/v_{\mathrm{ref}}},\qquad
\dot\psi = \frac{v}{L}\tan\delta,\qquad
\dot\psi_c = \frac{\psi-\psi_c}{\tau_c}
$$

| Parameter | Value |
|---|---|
| Wheelbase `L` | 2.7 m |
| `δ_max`, `v_ref` | 32°, 6 m/s |
| Acceleration `a⁺`, braking `a⁻` | 3 m/s², 7 m/s² |
| Drag `c`, top speed | 0.12 s⁻¹, 25 m/s |
| Chase-camera lag `τ_c` | 0.35 s |

Lane changes are steering doublets (`s = ±0.7` for 0.5 s each).

Route presets, in [`nycdrive/vehicle.py`](nycdrive/vehicle.py): `cruise`, `highway`, `downtown`, `slalom`, `turn_left`, `turn_right`. Custom routes are JSON lists of `[seconds, throttle, steer]`.

## Setup

Common (tested with torch 2.12 + CUDA 13.0 and flash-attn 2.8 on RTX PRO 6000 Blackwell GPUs):

```bash
git clone https://github.com/mkturkcan/matrix-new-york && cd matrix-new-york
pip install -r requirements.txt          # install torch + flash-attn for your CUDA first
```

Matrix-Game-3.0 (~43 GB):

```bash
git clone --filter=blob:none --sparse https://github.com/SkyworkAI/Matrix-Game.git third_party/Matrix-Game
git -C third_party/Matrix-Game sparse-checkout set Matrix-Game-3
hf download Skywork/Matrix-Game-3.0 --local-dir checkpoints/Matrix-Game-3.0 --exclude "base_model/*"
```

Matrix-Game-3.5 (~35 GB). We used a separate virtual environment with the upstream requirements:

```bash
git clone https://github.com/Riemann-Dynamics/Matrix-Game-3.5 third_party/Matrix-Game-3.5
hf download RiemannDynamics/Matrix-Game-3.5-Distilled --local-dir checkpoints/mg35/Matrix-Game-3.5-Distilled
hf download Wan-AI/Wan2.2-TI2V-5B --exclude "assets/*" "examples/*" --local-dir checkpoints/mg35/Wan2.2-TI2V-5B
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/mg35/DA3NESTED-GIANT-LARGE-1.1   # CC-BY-NC-4.0
```

PanoWorld (480p Rolling Forcing checkpoint plus the Wan2.2-TI2V-5B base). The adapter reads `third_party/PanoWorld/models`:

```bash
git clone https://github.com/Insta360-Research-Team/PanoWorld third_party/PanoWorld
hf download Insta360-Research/PanoWorld --include "models/casual/rolling_dmd_480p_161/*" "models/lora/480p/*" "models/action/480p/*" \
  --local-dir checkpoints/PanoWorld
hf download Wan-AI/Wan2.2-TI2V-5B --exclude "assets/*" "examples/*" --local-dir checkpoints/PanoWorld/models/Wan-AI/Wan2.2-TI2V-5B
ln -s "$PWD/checkpoints/PanoWorld/models" third_party/PanoWorld/models
```

## Usage

```bash
# initial observations
python -m nycdrive.prepare_inputs                                   # curated scenes
python -m nycdrive.prepare_inputs --random --seed 7                 # random location (Manhattan / Brooklyn / Queens)
python -m nycdrive.prepare_inputs --lat 40.7497 --lon -73.9877 --name herald_square [--heading DEG]
python -m nycdrive.car_bank                                         # ego-vehicle yaw renders for every scene

# rollouts
python -m nycdrive.drive           --scene data/inputs/fdr_drive --route highway --num_iterations 20 --frame_stride 2
python -m nycdrive.drive           --scene data/inputs/fdr_drive --interactive          # Matrix-Game-3.0, step-wise input
python -m nycdrive.drive_mg35      --scene data/inputs/fdr_drive --route highway --seconds 21
python -m nycdrive.drive_panoworld --scene data/inputs/fdr_drive --route highway --seconds 20
python -m nycdrive.drive_panoworld --scene data/inputs/park_avenue --route cruise --seconds 15 --segment_frames 81 --anchor

# batch rendering (Matrix-Game-3.0) and project page
python scripts/render_showcase.py configs/showcase.json --gpus 0,1,2,3
python scripts/build_site.py
```

Each rollout writes:

- `<name>_world.mp4`: raw model output (PanoWorld: a perspective crop; the panorama is `<name>_pano.mp4`)
- `<name>_clean.mp4`: with the ego vehicle composited
- `<name>_hud.mp4`: with the ego vehicle and an input overlay
- `<name>_actions.npz`: controller inputs, issued actions and camera trajectory

Rollouts go to `outputs/videos/` (Matrix-Game-3.0), `outputs/videos_mg35/` or `outputs/videos_panoworld/`.

## Repository layout

| Path | Contents |
|---|---|
| [`nycdrive/prepare_inputs.py`](nycdrive/prepare_inputs.py) | panorama retrieval, reprojection, vehicle masking, inpainting, vehicle insertion |
| [`nycdrive/car_render.py`](nycdrive/car_render.py), [`nycdrive/car_bank.py`](nycdrive/car_bank.py) | MeshFleet chase-camera renders; per-scene yaw banks |
| [`nycdrive/vehicle.py`](nycdrive/vehicle.py) | bicycle model, chase camera, routes, action adapters |
| [`nycdrive/drive.py`](nycdrive/drive.py), [`nycdrive/drive_mg35.py`](nycdrive/drive_mg35.py), [`nycdrive/drive_panoworld.py`](nycdrive/drive_panoworld.py) | backends |
| [`nycdrive/player.py`](nycdrive/player.py), [`nycdrive/hud.py`](nycdrive/hud.py) | ego-vehicle compositing, input overlay |
| [`configs/`](configs/) | showcase jobs per backend, model registry and page text for the project page |
| [`scripts/`](scripts/) | multi-GPU batch rendering, post-processing, project page |

## Limitations

- **Single observation.** Only the first frame or panorama is observed, except in the re-anchored PanoWorld clips. Content entering the view later is sampled from the model's prior and does not match the real street network.
- **Long-horizon degradation.**
  - Matrix-Game-3.0 quality is highest in roughly the first 15 s of generated time. Afterwards facades simplify, road texture streaks, and color shifts.
  - Matrix-Game-3.5 leaves the roadway after 6 to 15 s.
  - PanoWorld stays recognisable but is soft and slow.
- **2D ego vehicle.** The ego vehicle is a composited layer. It is not occluded by generated content, casts no shadows onto it, and does not interact with it.
- **Panorama retrieval.** `streetlevel.streetview.find_panorama` no longer works because Google decommissioned the endpoint behind it. `prepare_inputs.py` queries Street View coverage tiles instead.

## References and licenses

- **Matrix-Game-3.0**, Skywork AI ([weights](https://huggingface.co/Skywork/Matrix-Game-3.0), [code](https://github.com/SkyworkAI/Matrix-Game)). Apache-2.0.
- **Matrix-Game-3.5**, Qian et al., [arXiv:2608.29910](https://arxiv.org/abs/2608.29910) ([code](https://github.com/Riemann-Dynamics/Matrix-Game-3.5)). Code and weights Apache-2.0. Its Patch Memory uses [Depth Anything 3](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) weights, which are licensed **CC-BY-NC-4.0**.
- **PanoWorld**, Li et al., [arXiv:2607.09661](https://arxiv.org/abs/2607.09661) ([code](https://github.com/Insta360-Research-Team/PanoWorld)). The code is Apache-2.0; the weights on Hugging Face carry no license tag.
- **MeshFleet**, Boborzi et al. 2025 ([dataset](https://huggingface.co/datasets/DamianBoborzi/MeshFleet), [arXiv:2503.14002](https://arxiv.org/abs/2503.14002)). Vehicle meshes used (CC-BY, Sketchfab):
  - "Subaru WRX STI" by Skyborg ([model](https://sketchfab.com/3d-models/36f4a47b619a49bda4c1870e7833b25b))
  - "SuzukiVitara3DModel" by TuMankito77 ([model](https://sketchfab.com/3d-models/6c2f647e8874401abfee51dbe6a0bee9))
  - "Dodge-charger-pov" by lukas.ohmer ([model](https://sketchfab.com/3d-models/b1694c78e3184150b8fd518eeee5af06))
  - Ford Mustang GT (MeshFleet sha256 `89f6d9c8…`; Sketchfab [f8e521c3…](https://sketchfab.com/3d-models/f8e521c3315e445ea69f630c79b682f0), page no longer online)
  - Ferrari 488 (MeshFleet sha256 `e7d2897c…`; Sketchfab [0045292d…](https://sketchfab.com/3d-models/0045292de2114f0c83a9e5413eb043d4), page no longer online)
- **Mask2Former**, Cheng et al., CVPR 2022 (`facebook/mask2former-swin-large-cityscapes-semantic`).
- **LaMa**, Suvorov et al., WACV 2022, via [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting).
- **[streetlevel](https://github.com/sk-zk/streetlevel)** (MIT). Street View imagery © Google; scene images are not redistributed in this repository.
- **Project page:** built with [Leaflet](https://leafletjs.com/), [three.js](https://threejs.org/) and [KaTeX](https://katex.org/). Map tiles © Esri, HERE, Garmin and OpenStreetMap contributors.

Code in this repository: Apache-2.0.
