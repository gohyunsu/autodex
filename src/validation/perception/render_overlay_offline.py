#!/usr/bin/env python3
"""Offline overlay renderer for a saved init_interactive trial.

Reads pose_world.npy + capture/images/*.png from a trial dir, renders the
object mesh at the saved pose into every camera, and saves per-cam overlays
plus a grid.png.

Run in the same env as init_interactive (`gotrack_cu128`):

    python src/validation/perception/render_overlay_offline.py \\
        --trial-dir ~/shared_data/AutoDex/experiment/object6d_test_foundpose/attached_container/20260512_233200

Output:
    {trial_dir}/overlay/{serial}.png  (per-cam)
    {trial_dir}/overlay/grid.png      (4-col grid)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_FP_ROOT = REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))

MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


def _load_calib(calib_dir: Path):
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)
    intr_undist, extr, H, W = {}, {}, None, None
    for s, d in intr_raw.items():
        intr_undist[s] = np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        H = int(d["height"]); W = int(d["width"])
    for s, e in extr_raw.items():
        a = np.asarray(e, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extr[s] = a
    return intr_undist, extr, H, W


def _resolve_calib_dir(arg: str | None, trial_dir: Path) -> Path:
    if arg:
        return Path(arg).expanduser()
    # try trial_dir/cam_param first (if we ever save one), else latest under ~/shared_data/cam_param/
    local = trial_dir / "cam_param"
    if (local / "intrinsics.json").exists():
        return local
    candidates = sorted([p for p in CAM_PARAM_ROOT.iterdir() if p.is_dir()]) if CAM_PARAM_ROOT.exists() else []
    if not candidates:
        sys.exit(f"No calib found at {trial_dir/'cam_param'} or {CAM_PARAM_ROOT}")
    return candidates[-1]


def _resolve_mesh(obj: str) -> Path:
    for sub in [
        MESH_BASE / obj / "raw_mesh" / f"{obj}.obj",
        MESH_BASE / obj / "processed_data" / "mesh" / "raw.obj",
        MESH_BASE / obj / "processed_data" / "mesh" / "simplified.obj",
    ]:
        if sub.exists():
            return sub
    sys.exit(f"No mesh found for obj={obj} under {MESH_BASE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", required=True,
                        help="Path to a single trial dir containing pose_world.npy + capture/images/")
    parser.add_argument("--obj", default=None,
                        help="Object name. Default: parent dir name of trial_dir.")
    parser.add_argument("--calib-dir", default=None,
                        help="cam_param dir. Default: trial_dir/cam_param if exists, "
                             "else latest under ~/shared_data/cam_param/.")
    parser.add_argument("--images-dir", default=None,
                        help="Override images dir. Default: trial_dir/capture/images/.")
    parser.add_argument("--out", default=None,
                        help="Output dir. Default: trial_dir/overlay/.")
    parser.add_argument("--color", type=int, nargs=3, default=[0, 200, 0],
                        help="Overlay color BGR. Default green.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n-cols", type=int, default=4)
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir).expanduser().resolve()
    if not trial_dir.is_dir():
        sys.exit(f"Not a directory: {trial_dir}")
    pose_path = trial_dir / "pose_world.npy"
    if not pose_path.exists():
        sys.exit(f"pose_world.npy missing: {pose_path}")

    obj = args.obj or trial_dir.parent.name
    images_dir = Path(args.images_dir).expanduser() if args.images_dir else trial_dir / "capture" / "images"
    if not images_dir.exists():
        sys.exit(f"images dir missing: {images_dir} (live mode trial?)")
    out_dir = Path(args.out).expanduser() if args.out else trial_dir / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = _resolve_calib_dir(args.calib_dir, trial_dir)
    mesh_path = _resolve_mesh(obj)
    print(f"[load] obj={obj}")
    print(f"[load] pose={pose_path}")
    print(f"[load] images={images_dir}")
    print(f"[load] calib={calib_dir}")
    print(f"[load] mesh={mesh_path}")

    pose_world = np.load(pose_path).astype(np.float64).reshape(4, 4)
    intr_undist, extrinsics, H, W = _load_calib(calib_dir)

    import torch
    import nvdiffrast.torch as dr
    import trimesh
    from Utils import nvdiffrast_render, make_mesh_tensors

    mesh = trimesh.load(str(mesh_path), process=False)
    glctx = dr.RasterizeCudaContext()
    mesh_tensors = make_mesh_tensors(mesh, device="cuda")
    color = np.array(args.color, dtype=np.float32)

    overlays: dict[str, np.ndarray] = {}
    for img_path in sorted(images_dir.glob("*.png")):
        s = img_path.stem
        if s not in intr_undist or s not in extrinsics:
            print(f"  skip {s}: no calib")
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  skip {s}: bad image")
            continue
        K = intr_undist[s].astype(np.float32)
        pose_cam = extrinsics[s] @ pose_world
        pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
        rc, _, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pt,
                                     glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
        render = rc[0].detach().cpu().numpy()
        mask = render.sum(axis=2) > 0
        overlay = bgr.copy()
        overlay[mask] = (overlay[mask] * (1 - args.alpha) + color * args.alpha).astype(np.uint8)
        cv2.putText(overlay, s, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(str(out_dir / f"{s}.png"), overlay)
        overlays[s] = overlay
        print(f"  ok {s}")

    if not overlays:
        sys.exit("no overlays rendered")

    # grid
    keys = sorted(overlays.keys())
    scale = 0.5
    th, tw = int(H * scale), int(W * scale)
    n_cols = max(1, args.n_cols)
    n_rows = (len(keys) + n_cols - 1) // n_cols
    grid = np.zeros((n_rows * th, n_cols * tw, 3), dtype=np.uint8)
    for i, k in enumerate(keys):
        r, c = divmod(i, n_cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(overlays[k], (tw, th))
    cv2.imwrite(str(out_dir / "grid.png"), grid)
    print(f"[done] {len(overlays)} cams -> {out_dir}/grid.png")


if __name__ == "__main__":
    main()
