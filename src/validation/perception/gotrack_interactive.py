#!/usr/bin/env python3
"""Interactive GoTrack tracking pipeline (init + continuous tracking).

End-to-end runner that:
  1. Starts camera stream via paradex remote_camera_controller
  2. Runs InitOrchestrator to get an initial pose_world (one trigger)
  3. Sends `init` + `start` to gotrack_daemon on each capture PC
  4. Spins up GoTrackTracker (robot PC stages 5-6) and streams poses

Output per session (timestamp dir):
  ~/shared_data/AutoDex/experiment/object6d_tracking/{obj}/{YYYYMMDD_HHMMSS}/
    ├── init_pose_world.npy            # initial pose from FoundPose init
    ├── poses/pose_{frame_id:06d}.npy  # tracked poses per frame
    └── tracking_summary.json          # frame count, fps, fit stats

Daemons that must be running:
  - init_daemon.py (capture1-6, port 5006/5007/6893) → FoundPose init phase
  - gotrack_daemon.py (capture1-6, port 1235/1236/6892) → tracking phase
  - paradex camera stream (started by this script)

Both daemons live in `gotrack_cu128` env on capture PCs.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
ANCHOR_BANK_DIR = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack/anchor_banks"
EXP_OUT = Path.home() / "shared_data/AutoDex/experiment/object6d_tracking"
DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]  # capture4 out


def _load_calib(calib_dir: Path):
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)
    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a
    H = next(iter(intrinsics_full.values()))["height"]
    W = next(iter(intrinsics_full.values()))["width"]
    return intrinsics_full, extrinsics_full, H, W


def _to_home_relative(p) -> str:
    p = str(p)
    home = str(Path.home())
    if p.startswith(home + "/"):
        return "~/" + p[len(home) + 1:]
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--pc-list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="cam_param dir. Default: latest under ~/shared_data/cam_param/.")
    parser.add_argument("--anchor-bank", type=str, default=None,
                        help="Path to anchor_bank .npz. Default: "
                             "autodex/perception/thirdparty/MV-GoTrack/anchor_banks/{obj}.npz")
    # Stream
    parser.add_argument("--stream-fps", type=int, default=20)
    parser.add_argument("--stream-warmup-s", type=float, default=2.0)
    # Init ports (must match init_daemon.py defaults)
    parser.add_argument("--init-port-mask", type=int, default=5006)
    parser.add_argument("--init-port-pose", type=int, default=5007)
    parser.add_argument("--init-port-cmd", type=int, default=6893)
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--init-timeout-s", type=float, default=120.0)
    # Tracking ports (must match gotrack_daemon.py defaults)
    parser.add_argument("--gotrack-port-obs", type=int, default=1235)
    parser.add_argument("--gotrack-port-prior", type=int, default=1236)
    parser.add_argument("--gotrack-port-cmd", type=int, default=6892)
    parser.add_argument("--min-cams-per-frame", type=int, default=None,
                        help="Override tracker min_cams_per_frame. Default: total active cams.")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Stop after this many tracked frames. -1 = unlimited.")
    parser.add_argument("--duration-s", type=float, default=-1.0,
                        help="Stop after this many seconds of tracking. -1 = unlimited.")
    parser.add_argument("--dashboard-port", type=int, default=0,
                        help="Tracker dashboard port (0 = disabled).")
    parser.add_argument("--out", type=str, default=str(EXP_OUT))
    args = parser.parse_args()

    from paradex.utils.system import get_pc_ip, get_camera_list
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.io.capture_pc.command_sender import CommandSender
    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.perception.gotrack_tracker import GoTrackTracker

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")

    init_assets_root = ASSETS_BASE / args.obj
    if not (init_assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"FoundPose repre.pth missing for {args.obj} under {init_assets_root}")

    anchor_bank = Path(args.anchor_bank).expanduser() if args.anchor_bank \
                  else ANCHOR_BANK_DIR / f"{args.obj}.npz"
    if not anchor_bank.exists():
        sys.exit(f"anchor_bank missing for {args.obj}: {anchor_bank}")

    # Calibration
    if args.calib_dir:
        calib_dir = Path(args.calib_dir).expanduser()
    else:
        cam_root = Path.home() / "shared_data/cam_param"
        calib_dir = sorted(cam_root.iterdir())[-1]
    print(f"[calib] {calib_dir}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"[calib] {len(intrinsics_full)} cams active ({len(args.pc_list)} PCs)  {H}x{W}")

    if args.min_cams_per_frame is None:
        args.min_cams_per_frame = max(1, len(intrinsics_full) - 4)  # tolerate few drops

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).expanduser() / args.obj / session_ts
    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    print(f"[session] {out_dir}")

    # ── Start camera stream (one rcc for entire run, init + tracking) ──
    rcc = None
    init_orch = None
    cmd = None
    tracker = None
    try:
        print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
        rcc = remote_camera_controller("gotrack_interactive", pc_list=args.pc_list)
        rcc.start("stream", False, fps=args.stream_fps)
        if args.stream_warmup_s > 0:
            time.sleep(args.stream_warmup_s)
        print("[stream] ready")

        # ── INIT phase: get initial pose_world via FoundPose distributed init ──
        print("\n[init] dispatching FoundPose init to capture PCs...")
        init_orch = InitOrchestrator(
            pc_list=args.pc_list, capture_ips=pc_ips,
            port_mask=args.init_port_mask, port_pose=args.init_port_pose,
            port_cmd=args.init_port_cmd,
        )
        init_orch.init_object(
            obj_name=args.obj,
            mesh_path=str(mesh_path),
            assets_root=str(init_assets_root),
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full,
            image_hw=(H, W),
            mode="live",
        )

        init_pose: Optional[np.ndarray] = None
        attempt = 0
        while init_pose is None:
            attempt += 1
            ans = input(f"[init attempt {attempt}] Press Enter to run, 'q' to abort: ").strip().lower()
            if ans == "q":
                sys.exit("aborted by user")
            init_pose, timing = init_orch.trigger_init(
                prompt=args.prompt,
                sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                timeout_s=args.init_timeout_s,
            )
            if init_pose is None:
                print(f"[init] FAILED: {timing.get('reason')}  (retry)")
            else:
                print(f"[init] OK  iou={timing.get('best_iou', 0):.3f}  "
                      f"sil_loss={timing.get('sil_loss', 0):.6f}  "
                      f"total {timing.get('total_s', 0):.2f}s")

        init_pose = np.asarray(init_pose, dtype=np.float64).reshape(4, 4)
        np.save(out_dir / "init_pose_world.npy", init_pose)
        print(f"[init] pose saved -> {out_dir/'init_pose_world.npy'}")

        # Close init orchestrator's SUB threads — its job is done. Stream stays up.
        init_orch.close()
        init_orch = None

        # ── TRACKING phase: send init+start to gotrack daemons, run tracker loop ──
        print("\n[tracking] init gotrack daemons...")
        intr_jsonable = {
            s: {
                "K": np.asarray(v["K_undist"], dtype=np.float64).reshape(3, 3).tolist(),
                "width": int(v["width"]), "height": int(v["height"]),
            }
            for s, v in intrinsics_full.items()
        }
        extr_jsonable = {
            s: np.asarray(v, dtype=np.float64).reshape(4, 4).tolist()
            for s, v in extrinsics_full.items()
        }
        gotrack_init_info = {
            "mesh_path": _to_home_relative(mesh_path),
            "anchor_bank_path": _to_home_relative(anchor_bank),
            "object_id": 1,
            "object_name": args.obj,
            "intrinsics": intr_jsonable,
            "extrinsics": extr_jsonable,
        }
        cmd = CommandSender(pc_list=args.pc_list, port=args.gotrack_port_cmd)
        with contextlib.redirect_stdout(io.StringIO()):
            cmd.send_command("init", wait=True, cmd_info=gotrack_init_info)
        print("[tracking] gotrack daemons initialized")

        tracker = GoTrackTracker(
            capture_pc_ips=pc_ips,
            port_obs=args.gotrack_port_obs,
            port_prior=args.gotrack_port_prior,
            min_cams_per_frame=args.min_cams_per_frame,
        )
        with tracker._status_lock:
            tracker.status["obj_name"] = args.obj
        if args.dashboard_port > 0:
            tracker.start_dashboard(args.dashboard_port)

        with contextlib.redirect_stdout(io.StringIO()):
            cmd.send_command("start", wait=False)
        print("[tracking] daemons running, entering track loop")
        print("[tracking] Ctrl+C to stop")

        n_ok = 0
        n_fail = 0
        t0 = time.perf_counter()
        try:
            for frame_id, pose, info in tracker.track(init_pose):
                n_ok += 1
                np.save(poses_dir / f"pose_{frame_id:06d}.npy", pose)
                if n_ok % 10 == 0:
                    fps = tracker.status.get("fps", 0.0)
                    print(f"  [{n_ok:5d}] frame {frame_id}  "
                          f"t={pose[:3, 3].round(3).tolist()}  "
                          f"n_inl={info.get('n_inliers')}  "
                          f"resid={info.get('mean_residual_mm', -1):.2f}mm  "
                          f"fps={fps:.1f}")
                if args.max_frames > 0 and n_ok >= args.max_frames:
                    break
                if args.duration_s > 0 and (time.perf_counter() - t0) >= args.duration_s:
                    break
        except KeyboardInterrupt:
            print("\n[tracking] interrupted by user")

        elapsed = time.perf_counter() - t0
        summary = {
            "obj": args.obj,
            "session_ts": session_ts,
            "n_tracked_frames": n_ok,
            "n_failed_frames": n_fail,
            "elapsed_s": elapsed,
            "fps_avg": (n_ok / elapsed) if elapsed > 0 else 0.0,
            "min_cams_per_frame": args.min_cams_per_frame,
            "active_serials": sorted(active_serials),
        }
        with open(out_dir / "tracking_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[done] {n_ok} frames in {elapsed:.1f}s ({summary['fps_avg']:.1f} fps avg)")
        print(f"[done] summary -> {out_dir/'tracking_summary.json'}")

    finally:
        if cmd is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cmd.send_command("stop", wait=False)
            except Exception:
                pass
            try:
                cmd.end()
            except Exception:
                pass
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        if init_orch is not None:
            try:
                init_orch.close()
            except Exception:
                pass
        if rcc is not None:
            try:
                print("[stream] stopping camera stream...")
                rcc.stop()
            except Exception:
                pass
            try:
                rcc.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
