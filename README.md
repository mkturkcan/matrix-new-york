# Matrix New York

Street View initial observations and vehicle control for interactive video world models, applied to New York City.

**Project page:** https://mehmetkeremturkcan.com/matrix-new-york/

## Summary

For a query location we retrieve the nearest Google Street View panorama captured by a Street View car (via [streetlevel](https://github.com/sk-zk/streetlevel)), then:

1. reproject it to a 1280×704 perspective view along the capture heading;
2. remove near-field vehicles (Mask2Former, Cityscapes classes; LaMa inpainting);
3. insert a vehicle mesh from [MeshFleet](https://huggingface.co/datasets/DamianBoborzi/MeshFleet) rendered under a chase camera.

A kinematic bicycle model converts throttle and steering into the keyboard and camera-yaw actions of [Matrix-Game-3.0](https://huggingface.co/Skywork/Matrix-Game-3.0). Because Matrix-Game-3.0 is first-person:

- the model is conditioned on the vehicle-free frame;
- the ego vehicle is composited from 17 renders indexed by the controller's yaw relative to the camera.

No model is fine-tuned. Each rollout runs on one GPU.

| | |
|---|---|
| Output | 704×1280, 17 fps; 20 autoregressive chunks (817 frames ≈ 48 s), shown with frame stride 2 |
| Sampling | 3 steps (distilled DiT), bf16 |
| Throughput | ≈3.6 generated frames/s on one RTX PRO 6000 Blackwell (no int8, synchronous VAE) |

## Observations

- **The ego vehicle must not be in the conditioning frame.** Matrix-Game-3.0 has no ego-vehicle prior. A vehicle composited into the conditioning frame is treated as static scene content and smears across the lower image within about a second. We therefore condition on the inpainted frame and composite the vehicle afterwards.
- **Forward speed is fixed by the model.** Matrix-Game-3.0 advances a roughly constant distance per frame while `W` is held, and the action interface cannot make it faster:
  - Scaling the camera translation used for its Plücker-ray conditioning by 3× left mean Farnebäck flow unchanged: 0.65 vs 0.66 px/frame at 320 px width.
  - A lower `W` duty cycle does slow it (0.04 px/frame at 12% duty).
  - Apparent speed is therefore set by temporal subsampling (`--frame_stride`). The controller runs at `stride × 17` Hz, so manoeuvres keep their real-time duration on screen.
- **Turns.** Large heading changes away from intersections steer the camera into facades. Turn presets are only meaningful when the first frame is at an intersection.

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
| Acceleration `a⁺` / braking `a⁻` | 3 / 7 m/s² |
| Drag `c`, top speed | 0.12 s⁻¹, 25 m/s |
| Chase-camera lag `τ_c` | 0.35 s |

**Action mapping to Matrix-Game-3.0:**

- `W` is pulse-width modulated with duty `min(v / 10 m/s, 1)`.
- Mouse yaw is `Δψ_c / 15°` per frame (MG3 yaw sensitivity). Sub-deadzone values are accumulated.
- Lane changes are steering doublets (`s = ±0.7` for 0.5 s each), so the vehicle yaws instead of translating.
- The ego vehicle is drawn at relative yaw `ψ − ψ_c` from a bank of 17 renders (−24° to +24°, pivot at the rear axle).

Route presets in [`nycdrive/vehicle.py`](nycdrive/vehicle.py): `cruise`, `highway`, `downtown`, `slalom`, `turn_left`, `turn_right`. Custom routes are JSON lists of `[seconds, throttle, steer]`.

## Setup

```bash
git clone https://github.com/mkturkcan/matrix-new-york && cd matrix-new-york
pip install -r requirements.txt          # install torch + flash-attn for your CUDA first
git clone --filter=blob:none --sparse https://github.com/SkyworkAI/Matrix-Game.git third_party/Matrix-Game
git -C third_party/Matrix-Game sparse-checkout set Matrix-Game-3
hf download Skywork/Matrix-Game-3.0 --local-dir checkpoints/Matrix-Game-3.0 --exclude "base_model/*"   # ~43 GB
```

Tested with torch 2.12 + CUDA 13.0 and flash-attn 2.8 on RTX PRO 6000 Blackwell GPUs.

## Usage

```bash
# initial observations
python -m nycdrive.prepare_inputs                                   # curated scenes
python -m nycdrive.prepare_inputs --random --seed 7                 # random location (Manhattan / Brooklyn / Queens)
python -m nycdrive.prepare_inputs --lat 40.7497 --lon -73.9877 --name herald_square [--heading DEG]
python -m nycdrive.car_bank                                         # ego-vehicle yaw renders for every scene

# rollouts
python -m nycdrive.drive --scene data/inputs/fdr_drive --route highway --num_iterations 20 --frame_stride 2
python -m nycdrive.drive --scene data/inputs/fdr_drive --interactive        # step-wise keyboard input (w/s/a/d/q)

# batch rendering and project page
python scripts/render_showcase.py configs/showcase.json --gpus 0,1,2,3
python scripts/build_site.py
```

Each rollout writes four files:

- `<name>_world.mp4`: raw model output
- `<name>_clean.mp4`: with the ego vehicle
- `<name>_hud.mp4`: with the ego vehicle and an input overlay
- `<name>_actions.npz`: issued actions, controller inputs and relative yaw

## Repository layout

| Path | Contents |
|---|---|
| [`nycdrive/prepare_inputs.py`](nycdrive/prepare_inputs.py) | panorama retrieval, reprojection, vehicle masking, inpainting, vehicle insertion |
| [`nycdrive/car_render.py`](nycdrive/car_render.py), [`nycdrive/car_bank.py`](nycdrive/car_bank.py) | MeshFleet chase-camera renders; per-scene yaw banks |
| [`nycdrive/vehicle.py`](nycdrive/vehicle.py) | bicycle model, chase camera, routes, action adapter |
| [`nycdrive/drive.py`](nycdrive/drive.py), [`nycdrive/player.py`](nycdrive/player.py) | Matrix-Game-3.0 rollout and ego-vehicle compositing |
| [`nycdrive/hud.py`](nycdrive/hud.py) | input overlay |
| [`scripts/`](scripts/) | multi-GPU batch rendering, post-processing, project page |

## Limitations

- **Single observation.** Only the first frame is observed. Content entering the view later is sampled from the model's prior and does not match the real street network.
- **Long-horizon degradation.** Quality is highest in roughly the first 15 s of generated time. Afterwards facades simplify into flat geometry, road texture streaks, and color shifts. Forward driving rarely revisits earlier viewpoints, so camera-aware memory does not counteract this.
- **2D ego vehicle.** The ego vehicle is a composited layer. It is not occluded by generated content, casts no shadows onto it, and does not interact with it.
- **Panorama retrieval.** `streetlevel.streetview.find_panorama` no longer works because Google decommissioned the endpoint behind it. `prepare_inputs.py` queries Street View coverage tiles instead.

## References and licenses

- Matrix-Game-3.0, Skywork AI ([weights](https://huggingface.co/Skywork/Matrix-Game-3.0), [code](https://github.com/SkyworkAI/Matrix-Game)).
- MeshFleet, Boborzi et al. 2025 ([dataset](https://huggingface.co/datasets/DamianBoborzi/MeshFleet), [arXiv:2503.14002](https://arxiv.org/abs/2503.14002)). Vehicle meshes used (CC-BY, Sketchfab):
  - "Subaru WRX STI" by Skyborg ([model](https://sketchfab.com/3d-models/36f4a47b619a49bda4c1870e7833b25b))
  - "SuzukiVitara3DModel" by TuMankito77 ([model](https://sketchfab.com/3d-models/6c2f647e8874401abfee51dbe6a0bee9))
  - "Dodge-charger-pov" by lukas.ohmer ([model](https://sketchfab.com/3d-models/b1694c78e3184150b8fd518eeee5af06))
  - Ford Mustang GT (MeshFleet sha256 `89f6d9c8…`; Sketchfab [f8e521c3…](https://sketchfab.com/3d-models/f8e521c3315e445ea69f630c79b682f0), page no longer online)
  - Ferrari 488 (MeshFleet sha256 `e7d2897c…`; Sketchfab [0045292d…](https://sketchfab.com/3d-models/0045292de2114f0c83a9e5413eb043d4), page no longer online)
- Mask2Former (Cheng et al., CVPR 2022): `facebook/mask2former-swin-large-cityscapes-semantic`.
- LaMa (Suvorov et al., WACV 2022), via [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting).
- [streetlevel](https://github.com/sk-zk/streetlevel) (MIT). Street View imagery © Google; scene images are not redistributed in this repository.
- Project page: [Leaflet](https://leafletjs.com/), map tiles © Esri, HERE, Garmin, © OpenStreetMap contributors, [KaTeX](https://katex.org/).

Code in this repository: Apache-2.0.
