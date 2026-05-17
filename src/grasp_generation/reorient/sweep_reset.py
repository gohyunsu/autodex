"""
Sweep reset planner over a (pickup_x, pickup_theta_z) grid.

The placement location is fixed (canonical drop zone) — the reset task only
cares about reorientation, so place is decoupled from pickup. For each cell on
the pickup grid we run the full 8-phase plan and cache the successful
trajectory.

Output layout (one dir per cell):
    outputs/reset_cache/{hand}/{obj}/reorient_{h_cm}/{i}_{j}/
        sweep_summary.json
        x{xx}_tz{zz}/{seed_id}/trajectory.npz
        x{xx}_tz{zz}/{seed_id}/meta.json

`sweep_summary.json` lists every cell with status (ok / fail reason), the
chosen seed (on success), and per-cell elapsed time. Useful as a heatmap.

Usage:
    python src/grasp_generation/reorient/sweep_reset.py \
        --obj attached_container --i 0 --j 16 --h_cm 0 \
        --x_min 0.30 --x_max 0.55 --x_step 0.05 \
        --tz_min 0 --tz_max 330 --tz_step 30 \
        --hand inspire_left --max_seeds 15
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from autodex.utils.path import repo_dir

# plan_reset.py lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_reset import (  # noqa: E402
    DEFAULT_PLACE_XY, DEFAULT_PLACE_TZ, APEX_Z, HAND_Z_MIN, FLOOR_Z, PHASE_NAMES,
    init_planner, load_tabletop_pose, make_obj_pose, plan_one_cell, save_plan,
    load_fk_urdf, load_object_vertices,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--j", type=int, required=True)
    p.add_argument("--h_cm", type=int, default=0)
    p.add_argument("--hand", default="inspire_left", choices=["inspire_left", "inspire", "allegro"])
    p.add_argument("--x_min", type=float, default=0.30)
    p.add_argument("--x_max", type=float, default=0.55)
    p.add_argument("--x_step", type=float, default=0.05)
    p.add_argument("--tz_min", type=float, default=0.0)
    p.add_argument("--tz_max", type=float, default=330.0)
    p.add_argument("--tz_step", type=float, default=30.0)
    p.add_argument("--place_x", type=float, default=DEFAULT_PLACE_XY[0])
    p.add_argument("--place_y", type=float, default=DEFAULT_PLACE_XY[1])
    p.add_argument("--place_tz", type=float, default=DEFAULT_PLACE_TZ)
    p.add_argument("--max_seeds", type=int, default=20)
    p.add_argument("--out", default=None, help="override sweep root dir")
    p.add_argument("--skip_done", action="store_true", help="skip cells with existing trajectory")
    args = p.parse_args()

    xs = np.arange(args.x_min, args.x_max + 1e-6, args.x_step).round(3)
    tzs = np.arange(args.tz_min, args.tz_max + 1e-6, args.tz_step).round(3)
    print(f"[sweep] obj={args.obj} i={args.i} j={args.j} h={args.h_cm}cm hand={args.hand}")
    print(f"[sweep] x: {xs.tolist()}")
    print(f"[sweep] tz: {tzs.tolist()}")
    print(f"[sweep] place=({args.place_x:.2f}, {args.place_y:.2f}, tz={args.place_tz:.0f}°)")
    print(f"[sweep] total cells: {len(xs) * len(tzs)}")

    h_m = args.h_cm / 100.0
    Ti = load_tabletop_pose(args.obj, args.i)
    Tj = load_tabletop_pose(args.obj, args.j)
    T_obj_end = make_obj_pose(Tj, np.array([args.place_x, args.place_y, Tj[2, 3] + h_m]),
                              args.place_tz)

    sweep_root = Path(args.out) if args.out else (
        Path(repo_dir) / "outputs" / "reset_cache" / args.hand / args.obj
        / f"reorient_{args.h_cm}" / f"{args.i}_{args.j}"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] init planner ...")
    t0 = time.time()
    planner, base_world = init_planner(args.hand)
    urdf_fk, ee_link = load_fk_urdf(args.hand)
    obj_verts = load_object_vertices(args.obj)
    print(f"[sweep] planner warmup: {time.time() - t0:.1f}s ({len(obj_verts)} mesh verts)")

    summary = {
        "obj_name": args.obj, "hand": args.hand,
        "i": args.i, "j": args.j, "h_cm": args.h_cm,
        "x_values": xs.tolist(), "tz_values": tzs.tolist(),
        "place_x": args.place_x, "place_y": args.place_y, "place_tz": args.place_tz,
        "T_obj_end": T_obj_end.tolist(),
        "phase_names": PHASE_NAMES, "apex_z": APEX_Z, "hand_z_min": HAND_Z_MIN,
        "max_seeds": args.max_seeds,
        "cells": [],
    }

    n_ok = 0
    overall_t0 = time.time()
    for x in xs:
        for tz in tzs:
            cell_name = f"x{x:.2f}_tz{int(round(tz)):03d}"
            cell_dir = sweep_root / cell_name
            # skip_done: if any seed subdir has trajectory.npz, treat as cached
            cached = None
            if args.skip_done and cell_dir.exists():
                for sd in cell_dir.iterdir():
                    if (sd / "trajectory.npz").exists() and (sd / "meta.json").exists():
                        cached = sd
                        break
            if cached is not None:
                print(f"[sweep] {cell_name}: cached -> {cached.name}")
                summary["cells"].append({
                    "x": float(x), "tz": float(tz), "status": "ok",
                    "seed_id": cached.name, "elapsed_s": 0.0, "cached": True,
                })
                n_ok += 1
                continue

            T_obj_start = make_obj_pose(
                Ti, np.array([float(x), 0.0, Ti[2, 3]]), float(tz),
            )

            t1 = time.time()
            print(f"[sweep] {cell_name} ...", flush=True)
            result = plan_one_cell(
                planner, obj_name=args.obj, hand=args.hand,
                h_cm=args.h_cm, i=args.i, j=args.j,
                T_obj_start=T_obj_start, T_obj_end=T_obj_end,
                base_world=base_world, max_seeds=args.max_seeds, verbose=False,
                urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
            )
            elapsed = time.time() - t1

            if result is None:
                print(f"[sweep] {cell_name}: FAIL ({elapsed:.1f}s)")
                summary["cells"].append({
                    "x": float(x), "tz": float(tz), "status": "fail",
                    "elapsed_s": round(elapsed, 2),
                })
                continue

            n_ok += 1
            print(f"[sweep] {cell_name}: ok seed={result['seed_id']} ({elapsed:.1f}s)")
            out_dir = cell_dir / result["seed_id"]
            meta = {
                "obj_name": args.obj, "hand": args.hand,
                "i": args.i, "j": args.j, "h_cm": args.h_cm,
                "pickup_x": float(x), "pickup_tz": float(tz),
                "place_x": args.place_x, "place_y": args.place_y, "place_tz": args.place_tz,
                "seed_id": result["seed_id"], "phase_names": PHASE_NAMES,
                "wrist_se3_obj": result["wrist_se3_obj"].tolist(),
                "T_obj_start": T_obj_start.tolist(),
                "T_obj_apex_i": result["T_obj_apex_i"].tolist(),
                "T_obj_apex_j": result["T_obj_apex_j"].tolist(),
                "T_obj_end": T_obj_end.tolist(),
                "apex_z": APEX_Z, "hand_z_min": HAND_Z_MIN,
            }
            save_plan(out_dir, result["trajs"], meta)
            summary["cells"].append({
                "x": float(x), "tz": float(tz), "status": "ok",
                "seed_id": result["seed_id"], "elapsed_s": round(elapsed, 2),
            })

            # Save summary incrementally so partial progress is recoverable
            with open(sweep_root / "sweep_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    total = time.time() - overall_t0
    print(f"[sweep] done: {n_ok}/{len(summary['cells'])} ok  total={total:.1f}s")
    with open(sweep_root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sweep] summary -> {sweep_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
