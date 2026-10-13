"""Render a batch of (scene, route) jobs, one job per GPU at a time.

    python scripts/render_showcase.py configs/showcase.json --gpus 0,1,2,3
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    jobs = json.loads(Path(args.config).read_text())
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            name = j.get("save_name") or f"{Path(j['scene']).name}_{j['route']}"
            if args.skip_existing and (ROOT / "outputs" / "videos" / f"{name}_hud.mp4").exists():
                continue
            cmd = [sys.executable, "-m", "nycdrive.drive", "--scene", j["scene"], "--route", j["route"],
                   "--num_iterations", str(j.get("num_iterations", args.num_iterations)),
                   "--seed", str(j.get("seed", 42)), "--save_name", name]
            if j.get("label"):
                cmd += ["--label", j["label"]]
            if j.get("prompt"):
                cmd += ["--prompt", j["prompt"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            print(f"[gpu {gpu}] start {name}", flush=True)
            with open(log_dir / f"{name}.log", "w") as f:
                rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            print(f"[gpu {gpu}] done  {name} rc={rc}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
