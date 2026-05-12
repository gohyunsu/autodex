#!/usr/bin/env python3
"""Silhouette refinement resolution sweep.

For each (obj, episode), runs SilhouetteOptimizer.optimize() at multiple
downsample scales (default 1, 2, 4, 8) and records:
  - wall time
  - final sil_loss
  - pose error vs reference (pose_world.npy)
  - pose error vs scale=1 result (intrinsic accuracy loss from downsampling)

Inputs from cached pipeline outputs (no FoundPose / SAM3 run needed):
  ~/shared_data/AutoDex/experiment/selected_100/allegro/{obj}/{ep}/
    cam_param/{intrinsics,extrinsics}.json
    _pipeline_tmp/masks/{serial}.png
    pose_world.npy

Initial pose: pose_world + Gaussian noise (deterministic seed per (obj, ep)).

Output:
  {out-dir}/{ts}/results.json   per-trial records
  {out-dir}/{ts}/summary.csv    per-scale aggregates

Usage:
  conda activate autodex
  python src/validation/perception/sil_resolution_exp.py
  python src/validation/perception/sil_resolution_exp.py \
      --objs blue_alarm,book --n-eps 5 --scales 1 2 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_FP_ROOT = REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))


EXP_ROOT = Path.home() / "shared_data/AutoDex/experiment/selected_100/allegro"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"


def list_objects_with_masks(min_eps: int) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for obj_dir in sorted(EXP_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue
        eps: List[Path] = []
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            mdir = ep / "_pipeline_tmp/masks"
            try:
                if mdir.exists() and len(list(mdir.iterdir())) >= 20:
                    eps.append(ep)
            except OSError:
                continue
        if len(eps) >= min_eps:
            out[obj_dir.name] = eps
    return out


def load_calib(ep: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int]:
    with open(ep / "cam_param/intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(ep / "cam_param/extrinsics.json") as f:
        extr_raw = json.load(f)
    intr_K, extr = {}, {}
    H = W = None
    for s, d in intr_raw.items():
        intr_K[s] = np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        H, W = int(d["height"]), int(d["width"])
    for s, e in extr_raw.items():
        a = np.asarray(e, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extr[s] = a
    return intr_K, extr, H, W


def load_masks(ep: Path) -> Dict[str, np.ndarray]:
    mdir = ep / "_pipeline_tmp/masks"
    out: Dict[str, np.ndarray] = {}
    for p in sorted(mdir.iterdir()):
        if p.suffix != ".png":
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out[p.stem] = m
    return out


def downsample_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask
    H, W = mask.shape
    m = cv2.resize(mask, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    return ((m > 127).astype(np.uint8) * 255)


def scale_K(K: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return K.astype(np.float64).copy()
    K = K.astype(np.float64).copy()
    K[0, 0] /= scale
    K[1, 1] /= scale
    K[0, 2] = (K[0, 2] + 0.5) / scale - 0.5
    K[1, 2] = (K[1, 2] + 0.5) / scale - 0.5
    return K


def add_pose_noise(pose: np.ndarray, trans_std: float, rot_std_deg: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    angle = rng.normal(scale=np.deg2rad(rot_std_deg))
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R_noise = np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * (Kx @ Kx)
    t_noise = rng.normal(scale=trans_std, size=3)
    P = pose.copy()
    P[:3, :3] = R_noise @ pose[:3, :3]
    P[:3, 3] = pose[:3, 3] + t_noise
    return P


def pose_err(p_opt: np.ndarray, p_ref: np.ndarray) -> Tuple[float, float]:
    trans_err = float(np.linalg.norm(p_opt[:3, 3] - p_ref[:3, 3])) * 1000.0
    R = p_opt[:3, :3] @ p_ref[:3, :3].T
    cos_a = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    rot_err = float(np.degrees(np.arccos(cos_a)))
    return trans_err, rot_err


def torch_cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(optimizer, init_pose, masks, intr_K, extr, scale, iters, lr):
    views = []
    for s, mask in masks.items():
        if s not in intr_K or s not in extr:
            continue
        views.append({
            "mask": downsample_mask(mask, scale),
            "K": scale_K(intr_K[s], scale),
            "extrinsic": extr[s],
        })
    torch_cuda_sync()
    t0 = time.perf_counter()
    refined, sil_loss = optimizer.optimize(
        initial_pose_world=init_pose,
        views=views,
        iters=iters, lr=lr,
        antialias=True,
    )
    torch_cuda_sync()
    dt = time.perf_counter() - t0
    return np.asarray(refined, dtype=np.float64).reshape(4, 4), float(sil_loss), dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objs", type=str, default=None,
                        help="comma-separated obj names; default: top --n-objs by ep count")
    parser.add_argument("--n-objs", type=int, default=10)
    parser.add_argument("--n-eps", type=int, default=10)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--noise-trans", type=float, default=0.005,
                        help="translation noise std (m), per-axis")
    parser.add_argument("--noise-rot", type=float, default=3.0,
                        help="rotation noise std (deg)")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path.home() / "shared_data/AutoDex/experiment/_sil_resolution_exp"))
    args = parser.parse_args()

    inventory = list_objects_with_masks(min_eps=args.n_eps)
    if not inventory:
        sys.exit(f"No objects with >= {args.n_eps} cached-mask episodes under {EXP_ROOT}")

    if args.objs:
        obj_list = [s.strip() for s in args.objs.split(",")]
        missing = [o for o in obj_list if o not in inventory]
        if missing:
            print(f"[warn] no cached masks for: {missing}")
        obj_list = [o for o in obj_list if o in inventory]
    else:
        obj_list = sorted(inventory.keys(), key=lambda k: -len(inventory[k]))[:args.n_objs]

    print(f"Objects ({len(obj_list)}): {obj_list}")
    print(f"Scales: {args.scales}  iters={args.iters}  lr={args.lr}  "
          f"noise=trans{args.noise_trans*1000:.0f}mm/rot{args.noise_rot:.1f}deg")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    from autodex.perception.silhouette import SilhouetteOptimizer

    optimizer = None
    records: List[dict] = []

    for obj in obj_list:
        if obj not in inventory:
            continue
        mesh_path = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"
        if not mesh_path.exists():
            print(f"[skip] {obj}: mesh missing at {mesh_path}")
            continue

        print(f"\n=== {obj} ===")
        if optimizer is None:
            optimizer = SilhouetteOptimizer(mesh_path=str(mesh_path))
        else:
            optimizer.reset_mesh(str(mesh_path))

        eps = inventory[obj][:args.n_eps]
        for ep in eps:
            try:
                intr_K, extr, _, _ = load_calib(ep)
                masks = load_masks(ep)
                pose_world_path = ep / "pose_world.npy"
                if not pose_world_path.exists():
                    print(f"  [skip] {ep.name}: no pose_world.npy")
                    continue
                pose_ref = np.load(pose_world_path).astype(np.float64).reshape(4, 4)
            except Exception as e:
                print(f"  [skip] {ep.name}: load error {e}")
                continue

            seed = abs(hash((obj, ep.name))) % (2**31)
            init_pose = add_pose_noise(pose_ref, args.noise_trans, args.noise_rot, seed)

            scale1_pose = None
            for scale in args.scales:
                rec = {"obj": obj, "ep": ep.name, "scale": scale}
                try:
                    refined, sil_loss, dt = run_one(
                        optimizer, init_pose, masks, intr_K, extr,
                        scale=scale, iters=args.iters, lr=args.lr,
                    )
                except Exception as e:
                    rec.update({"ok": False, "error": str(e)})
                    records.append(rec)
                    print(f"  [{ep.name}] scale={scale}: FAILED {e}")
                    continue

                tref, rref = pose_err(refined, pose_ref)
                if scale == args.scales[0]:
                    scale1_pose = refined
                    ts1, rs1 = 0.0, 0.0
                elif scale1_pose is not None:
                    ts1, rs1 = pose_err(refined, scale1_pose)
                else:
                    ts1, rs1 = float("nan"), float("nan")

                rec.update({
                    "ok": True,
                    "wall_s": dt,
                    "sil_loss": sil_loss,
                    "trans_err_ref_mm": tref,
                    "rot_err_ref_deg": rref,
                    "trans_err_s1_mm": ts1,
                    "rot_err_s1_deg": rs1,
                })
                records.append(rec)
                print(f"  [{ep.name}] scale={scale}: time={dt:5.2f}s  "
                      f"loss={sil_loss:.4f}  "
                      f"vs_ref t={tref:5.1f}mm r={rref:4.2f}°  "
                      f"vs_s1 t={ts1:5.2f}mm r={rs1:4.2f}°")

            with open(out_dir / "results.json", "w") as f:
                json.dump(records, f, indent=2, default=str)

    print("\n=== Per-scale aggregates (median) ===")
    cols = ("scale", "n", "time_med", "time_mean", "sil_loss", "tref_mm", "rref_deg",
            "ts1_mm", "rs1_deg")
    print(f"{cols[0]:>5} {cols[1]:>4} " + " ".join(f"{c:>10}" for c in cols[2:]))
    summary_rows = []
    for scale in args.scales:
        rs = [r for r in records if r.get("ok") and r["scale"] == scale]
        if not rs:
            print(f"{scale:>5}  no successful trials")
            continue
        times = [r["wall_s"] for r in rs]
        row = {
            "scale": scale, "n": len(rs),
            "time_median": float(np.median(times)),
            "time_mean": float(np.mean(times)),
            "sil_loss_median": float(np.median([r["sil_loss"] for r in rs])),
            "trans_err_ref_median_mm": float(np.median([r["trans_err_ref_mm"] for r in rs])),
            "rot_err_ref_median_deg": float(np.median([r["rot_err_ref_deg"] for r in rs])),
            "trans_err_s1_median_mm": float(np.median([r["trans_err_s1_mm"] for r in rs])),
            "rot_err_s1_median_deg": float(np.median([r["rot_err_s1_deg"] for r in rs])),
        }
        summary_rows.append(row)
        print(f"{scale:>5} {len(rs):>4} "
              f"{row['time_median']:>10.2f} {row['time_mean']:>10.2f} "
              f"{row['sil_loss_median']:>10.4f} "
              f"{row['trans_err_ref_median_mm']:>10.2f} "
              f"{row['rot_err_ref_median_deg']:>10.2f} "
              f"{row['trans_err_s1_median_mm']:>10.2f} "
              f"{row['rot_err_s1_median_deg']:>10.2f}")

    with open(out_dir / "summary.csv", "w") as f:
        f.write("scale,n,time_median_s,time_mean_s,sil_loss_median,"
                "trans_err_ref_median_mm,rot_err_ref_median_deg,"
                "trans_err_s1_median_mm,rot_err_s1_median_deg\n")
        for r in summary_rows:
            f.write(f"{r['scale']},{r['n']},{r['time_median']:.4f},{r['time_mean']:.4f},"
                    f"{r['sil_loss_median']:.6f},{r['trans_err_ref_median_mm']:.4f},"
                    f"{r['rot_err_ref_median_deg']:.4f},{r['trans_err_s1_median_mm']:.4f},"
                    f"{r['rot_err_s1_median_deg']:.4f}\n")
    print(f"\nResults: {out_dir}/")


if __name__ == "__main__":
    main()