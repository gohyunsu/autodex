"""
Plan reset (reorient) trajectories.

Pickup is parameterized by `(pickup_x, pickup_theta_z)` — object on the table at
world `(pickup_x, 0, Ti.z)` with orientation `Rz(theta_z) @ Ri`. Place location
is free (the user choses a canonical "drop zone") so reset is decoupled from
pickup geometry: object lands at `(place_x, place_y, Tj.z + h_m)` with
orientation `Rz(place_theta_z) @ Rj`.

Full 8-phase pipeline per candidate seed:
  approach → grasp_close → lift → rotate → place → release → depart → retract

Phases 1–7 run with table-only world. Retract switches to a world that also
contains the placed object as a mesh obstacle (so the arm clears it on the way
back to INIT).

The library exposes:
  - `init_planner(hand)` — build GraspPlanner + warmup
  - `load_candidates_object_frame(obj_name, hand, h_cm, i, j)` — candidate seeds
  - `make_obj_pose(T_can, xyz, theta_z_deg)` — Rz(theta_z) @ Ri object pose
  - `plan_one_cell(planner, ...)` — try seeds for one (start, end) configuration

CLI (single cell):
    python src/grasp_generation/reorient/plan_reset.py \
        --obj attached_container --i 0 --j 16 --h_cm 0 \
        --pickup_x 0.40 --pickup_tz 0 --hand inspire_left

Output: outputs/reset_plans/{hand}/{obj}/reorient_{h_cm}/{i}_{j}/
        x{pickup_x:.2f}_tz{pickup_tz:03d}/{seed}/
  trajectory.npz  — phase-named joint trajectories
  meta.json       — config + T_obj keyframes
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from curobo.types.robot import JointState

from autodex.planner.planner import GraspPlanner, _to_curobo_pose
from autodex.utils.path import obj_path, repo_dir
from autodex.utils.robot_config import INSPIRE_INIT


# ── constants ────────────────────────────────────────────────────────────────

HAND_Z_MIN = 0.30
APEX_Z = 0.40
DEPART_DZ = 0.15
TABLE_DIMS = [2.0, 3.0, 0.2]
TABLE_POSE_XYZ = [1.1, 0.0, -0.1]
DEFAULT_PLACE_XY = (0.55, 0.0)
DEFAULT_PLACE_TZ = 0.0
PHASE_NAMES = [
    "approach", "grasp_close", "lift", "rotate", "place",
    "release", "depart", "retract",
]


def _reset_candidate_path(hand: str) -> Path:
    return Path(repo_dir) / "candidates" / hand


# ── data loading ─────────────────────────────────────────────────────────────

def load_tabletop_pose(obj_name: str, pose_idx: int) -> np.ndarray:
    p = Path(obj_path) / obj_name / "processed_data" / "info" / "tabletop" / f"{pose_idx:03d}.npy"
    return np.load(p)


def load_candidates_object_frame(obj_name: str, hand: str, h_cm: int, i: int, j: int):
    """Returns (wrist_se3_obj, pregrasp, grasp, seed_ids). All in object frame.

    Path layout: candidates/{hand}/reset/{obj}/{h_cm}/{i}_{j}/{seed}/
    """
    base = _reset_candidate_path(hand) / "reset" / obj_name / str(h_cm) / f"{i}_{j}"
    if not base.exists():
        raise FileNotFoundError(f"No candidates at {base}")
    wrist_o, preg, grasp_f, seeds = [], [], [], []
    for sd in sorted(base.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        wrist_o.append(np.load(sd / "wrist_se3.npy"))
        pg = np.load(sd / "pregrasp_pose.npy")
        preg.append(pg)
        gp_file = sd / "grasp_pose.npy"
        grasp_f.append(np.load(gp_file) if gp_file.exists() else pg)
        seeds.append(sd.name)
    return np.stack(wrist_o), np.stack(preg), np.stack(grasp_f), seeds


# ── geometry ─────────────────────────────────────────────────────────────────

def make_obj_pose(T_canonical: np.ndarray, xyz: np.ndarray, theta_z_deg: float) -> np.ndarray:
    """Apply Rz(theta_z) ∘ T_canonical, then override translation with xyz.

    Equivalent to: object placed at xyz with orientation Rz(theta_z) @ R_canonical.
    """
    th = np.radians(theta_z_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th),  np.cos(th), 0.0],
                   [0.0,         0.0,        1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ T_canonical[:3, :3]
    T[:3, 3] = xyz
    return T


def compute_apex_pair(T_obj_start, T_obj_end, wrist_se3_obj, apex_z=APEX_Z):
    """Apex frames: same orientation & xy as start/end, z chosen so wrist z == apex_z."""
    z_off_i = (T_obj_start[:3, :3] @ wrist_se3_obj[:3, 3])[2]
    z_off_j = (T_obj_end[:3, :3]   @ wrist_se3_obj[:3, 3])[2]
    T_apex_i = T_obj_start.copy(); T_apex_i[2, 3] = apex_z - z_off_i
    T_apex_j = T_obj_end.copy();   T_apex_j[2, 3] = apex_z - z_off_j
    return T_apex_i, T_apex_j


# ── world ────────────────────────────────────────────────────────────────────

def build_world_cfg() -> dict:
    return {
        "cuboid": {"table": {"dims": list(TABLE_DIMS),
                             "pose": [*TABLE_POSE_XYZ, 1.0, 0.0, 0.0, 0.0]}},
        "mesh": {},
    }


def _se3_to_cart7(T: np.ndarray):
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def world_with_object(base_world: dict, obj_name: str, T_obj_world: np.ndarray) -> dict:
    out = {"cuboid": dict(base_world["cuboid"]), "mesh": dict(base_world.get("mesh", {}))}
    mesh_path = Path(obj_path) / obj_name / "processed_data" / "mesh" / "simplified.obj"
    out["mesh"]["placed_object"] = {
        "pose": _se3_to_cart7(T_obj_world),
        "file_path": str(mesh_path),
    }
    return out


# ── planner ──────────────────────────────────────────────────────────────────

def init_planner(hand: str):
    """Build GraspPlanner with motion_gen + ik_solver warmed up. Returns (planner, base_world)."""
    base_world = build_world_cfg()
    planner = GraspPlanner(hand=hand)
    planner._init_motion_gen(base_world)
    planner._init_ik_solver({"cuboid": dict(base_world["cuboid"]), "mesh": {}})
    return planner, base_world


# cuRobo ee_link per hand — wrist frame target. Mirrors view_reset.EE_LINK_BY_HAND.
EE_LINK_BY_HAND = {
    "inspire_left": "base_link",
    "inspire":      "base_link",
    "allegro":      "base_link",
}

URDF_BY_HAND = {
    "inspire_left": ("inspire_left_description", "xarm_inspire_left.urdf"),
    "inspire":      ("inspire_description",      "xarm_inspire.urdf"),
    "allegro":      ("allegro_description",      "xarm_allegro.urdf"),
}


def load_fk_urdf(hand: str):
    """yourdfpy URDF for post-hoc FK checks. Returns (urdf, ee_link)."""
    import os
    import yourdfpy
    from autodex.utils.path import project_dir
    sub, name = URDF_BY_HAND[hand]
    path = os.path.join(project_dir, "content", "assets", "robot", sub, name)
    return yourdfpy.URDF.load(path, build_scene_graph=True), EE_LINK_BY_HAND[hand]


def load_object_vertices(obj_name: str) -> np.ndarray:
    """Object mesh vertices (V,3) — simplified for fast post-hoc checks."""
    import trimesh
    mesh_path = Path(obj_path) / obj_name / "processed_data" / "mesh" / "simplified.obj"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=np.float32)


def _ik_solve(planner: GraspPlanner, wrist_se3: np.ndarray, finger_q: np.ndarray, retract_q=None):
    poses = wrist_se3[None]
    goal = _to_curobo_pose(poses, planner._tensor_args.device)
    kwargs = {}
    if retract_q is not None:
        kwargs["retract_config"] = torch.tensor(
            retract_q, dtype=torch.float32, device=planner._tensor_args.device,
        ).unsqueeze(0)
    result = planner._ik_solver.solve_batch(goal, **kwargs)
    if not bool(result.success.cpu().numpy()[0]):
        return None
    q_sol = result.solution.cpu().numpy()[0]
    if q_sol.ndim == 2:
        q_sol = q_sol[0]
    arm = q_sol[:6].copy()
    # Snap joint 6 to nearest 2π-equivalent of the previous qpos so consecutive
    # keyframes don't end up on opposite sides of the joint range (which makes
    # the planner take a long way around). Fall back to INIT if no retract.
    ref = retract_q[5] if retract_q is not None else planner._init_state[5]
    diff = arm[5] - ref
    arm[5] -= np.round(diff / (2 * np.pi)) * 2 * np.pi
    # xarm URDF says joint 6 has ±2π range but cuRobo clamps to ±π. If the
    # nearest-to-ref equivalent lands outside cuRobo's limit, wrap again into
    # [-π, π] (the physical kinematics are identical; motion_gen / trajopt /
    # PRM mask would reject it as joint-limit violation otherwise).
    if arm[5] > np.pi:
        arm[5] -= 2 * np.pi
    elif arm[5] < -np.pi:
        arm[5] += 2 * np.pi
    return np.concatenate([arm, finger_q])


def _plan_js(planner: GraspPlanner, q_start: np.ndarray, q_goal: np.ndarray):
    start = JointState.from_position(
        torch.tensor(q_start, dtype=torch.float32, device=planner._tensor_args.device).unsqueeze(0)
    )
    goal = JointState.from_position(
        torch.tensor(q_goal, dtype=torch.float32, device=planner._tensor_args.device).unsqueeze(0)
    )
    result = planner._motion_gen.plan_single_js(
        start_state=start, goal_state=goal, plan_config=planner._plan_cfg,
    )
    if not result.success.item():
        return None
    return result.get_interpolated_plan().position.cpu().numpy()


FLOOR_Z = 0.0  # table top — object mesh vertices must stay >= this during carry


def _fk_ee_traj(urdf: "yourdfpy.URDF", joint_traj: np.ndarray, ee_link: str) -> np.ndarray:
    """Per-frame ee_link world SE3. (T,4,4)."""
    out = np.tile(np.eye(4), (len(joint_traj), 1, 1))
    base = urdf.base_link
    for t, q in enumerate(joint_traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(ee_link, base)
    return out


def _carry_object_min_z(ee_traj: np.ndarray, wrist_se3_obj: np.ndarray,
                         obj_verts: np.ndarray) -> float:
    """Min z over all transformed object vertices across the carry trajectory.

    T_obj(t) = ee(t) @ inv(wrist_se3_obj)
    vert_world(t) = T_obj(t).R @ vert + T_obj(t).t
    """
    inv_w = np.linalg.inv(wrist_se3_obj)
    obj_T = ee_traj @ inv_w  # (T,4,4)
    R = obj_T[:, :3, :3]      # (T,3,3)
    t = obj_T[:, :3, 3]       # (T,3)
    # (T, V, 3) = (T,3,3) @ (V,3).T  reshaped — use einsum
    v_world_z = np.einsum("tij,vj->tvi", R, obj_verts)[..., 2] + t[:, None, 2]
    return float(v_world_z.min())


def plan_one_seed(planner: GraspPlanner, *,
                  obj_name: str, base_world: dict,
                  T_obj_start, T_obj_apex_i, T_obj_apex_j, T_obj_end,
                  wrist_se3_obj, pregrasp_q, grasp_q,
                  urdf_fk=None, ee_link: str = "base_link",
                  obj_verts: np.ndarray = None):
    """Full 8-phase plan for one candidate seed. Returns (trajs_dict, status).

    World swaps:
      - approach: base + object @ T_obj_start (pregrasp is open-finger, no contact)
      - grasp_close..release: base only (fingers contacting object)
      - depart: base + object @ T_obj_end (open-finger, no contact)
      - retract: base + object @ T_obj_end

    Post-hoc checks (if urdf_fk provided):
      - lift/rotate/place: wrist z >= HAND_Z_MIN
      - lift/rotate/place: object vertex min z >= FLOOR_Z (carried object vs floor)
    """
    init = planner._init_state.copy()
    open_q = INSPIRE_INIT.astype(np.float32)

    def wrist_world(T_obj):
        return T_obj @ wrist_se3_obj

    q_pregrasp = _ik_solve(planner, wrist_world(T_obj_start), pregrasp_q, retract_q=init)
    if q_pregrasp is None: return None, "ik_pregrasp"
    q_grasped = q_pregrasp.copy(); q_grasped[6:] = grasp_q
    q_apex_i = _ik_solve(planner, wrist_world(T_obj_apex_i), grasp_q, retract_q=q_grasped)
    if q_apex_i is None: return None, "ik_apex_i"
    q_apex_j = _ik_solve(planner, wrist_world(T_obj_apex_j), grasp_q, retract_q=q_apex_i)
    if q_apex_j is None: return None, "ik_apex_j"
    q_placed = _ik_solve(planner, wrist_world(T_obj_end), grasp_q, retract_q=q_apex_j)
    if q_placed is None: return None, "ik_placed"
    q_released = q_placed.copy(); q_released[6:] = open_q

    T_wrist_depart = wrist_world(T_obj_end).copy()
    T_wrist_depart[2, 3] += DEPART_DZ
    q_depart = _ik_solve(planner, T_wrist_depart, open_q, retract_q=q_released)
    if q_depart is None: return None, "ik_depart"

    world_start = world_with_object(base_world, obj_name, T_obj_start)
    world_end   = world_with_object(base_world, obj_name, T_obj_end)

    trajs = {}
    # Phase-by-phase with world swaps as needed. Sequence:
    #   approach (world_start) → restore (base) → grasp..release → world_end → depart → retract
    try:
        # approach: avoid pickup-side object
        planner._update_world(world_start)
        t = _plan_js(planner, init, q_pregrasp)
        if t is None: return None, "plan_approach"
        trajs["approach"] = t

        # grasp_close..release: base only (fingers touching object)
        planner._update_world(base_world)
        contact_pairs = [
            ("grasp_close", q_pregrasp,  q_grasped),
            ("lift",        q_grasped,   q_apex_i),
            ("rotate",      q_apex_i,    q_apex_j),
            ("place",       q_apex_j,    q_placed),
            ("release",     q_placed,    q_released),
        ]
        for phase, qs, qg in contact_pairs:
            t = _plan_js(planner, qs, qg)
            if t is None: return None, f"plan_{phase}"
            trajs[phase] = t

        # Post-hoc checks on the rotate phase only — wrist z >= HAND_Z_MIN and
        # carried-object vertex min z >= FLOOR_Z. Lift starts at q_grasped
        # (wrist near object, low z) and place ends at q_placed (placing on
        # table); checking them is meaningless because they intrinsically have
        # endpoints below the thresholds. Only rotate is "high all the way".
        if urdf_fk is not None:
            ee_tr = _fk_ee_traj(urdf_fk, trajs["rotate"], ee_link)
            if ee_tr[:, 2, 3].min() < HAND_Z_MIN:
                return None, "wrist_below_min_rotate"
            if obj_verts is not None:
                obj_z_min = _carry_object_min_z(ee_tr, wrist_se3_obj, obj_verts)
                if obj_z_min < FLOOR_Z:
                    return None, "object_below_floor_rotate"

        # depart: avoid place-side object (open-finger, no contact)
        planner._update_world(world_end)
        t = _plan_js(planner, q_released, q_depart)
        if t is None: return None, "plan_depart"
        trajs["depart"] = t

        # retract: object stays as obstacle
        t = _plan_js(planner, q_depart, init)
        if t is None: return None, "plan_retract"
        trajs["retract"] = t
    finally:
        planner._update_world(base_world)

    return trajs, "ok"


def plan_one_cell(planner: GraspPlanner, *,
                  obj_name: str, hand: str, h_cm: int, i: int, j: int,
                  T_obj_start: np.ndarray, T_obj_end: np.ndarray,
                  base_world: dict, max_seeds: int = 20, verbose: bool = True,
                  urdf_fk=None, ee_link: str = "base_link",
                  obj_verts: np.ndarray = None):
    """Try seeds until first successful 8-phase plan. Returns dict or None.

    Result dict keys: trajs, seed_id, seed_idx, T_obj_apex_i, T_obj_apex_j,
                      wrist_se3_obj, fail_counts.

    If urdf_fk + obj_verts are provided, post-hoc wrist-z and carried-object
    floor checks are enforced on each seed before accepting.
    """
    wrist_o_all, preg_all, grasp_all, seeds = load_candidates_object_frame(
        obj_name, hand, h_cm, i, j,
    )
    n_try = min(max_seeds, len(seeds))
    if verbose:
        print(f"[cell] candidates={len(seeds)} attempting={n_try}")

    fail = {}
    for k in range(n_try):
        wso = wrist_o_all[k]
        T_ai, T_aj = compute_apex_pair(T_obj_start, T_obj_end, wso, APEX_Z)
        t0 = time.time()
        result, status = plan_one_seed(
            planner, obj_name=obj_name, base_world=base_world,
            T_obj_start=T_obj_start, T_obj_apex_i=T_ai,
            T_obj_apex_j=T_aj, T_obj_end=T_obj_end,
            wrist_se3_obj=wso, pregrasp_q=preg_all[k], grasp_q=grasp_all[k],
            urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"[cell] seed {seeds[k]}: {status} ({elapsed:.1f}s)")
        if status == "ok":
            return {
                "trajs": result, "seed_id": seeds[k], "seed_idx": k,
                "T_obj_apex_i": T_ai, "T_obj_apex_j": T_aj,
                "wrist_se3_obj": wso, "fail_counts": fail,
            }
        fail[status] = fail.get(status, 0) + 1
    return None


def save_plan(out_dir: Path, trajs: dict, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "trajectory.npz", **trajs)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--j", type=int, required=True)
    p.add_argument("--h_cm", type=int, default=0)
    p.add_argument("--pickup_x", type=float, default=0.40)
    p.add_argument("--pickup_tz", type=float, default=0.0, help="object z-rotation at pickup (deg)")
    p.add_argument("--place_x", type=float, default=DEFAULT_PLACE_XY[0])
    p.add_argument("--place_y", type=float, default=DEFAULT_PLACE_XY[1])
    p.add_argument("--place_tz", type=float, default=DEFAULT_PLACE_TZ)
    p.add_argument("--hand", default="inspire_left", choices=["inspire_left", "inspire", "allegro"])
    p.add_argument("--max_seeds", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    h_m = args.h_cm / 100.0
    print(f"[reset] obj={args.obj} i={args.i} j={args.j} h={args.h_cm}cm "
          f"pickup=({args.pickup_x:.2f}, 0, tz={args.pickup_tz:.0f}°) "
          f"place=({args.place_x:.2f}, {args.place_y:.2f}, tz={args.place_tz:.0f}°) "
          f"hand={args.hand}")

    Ti = load_tabletop_pose(args.obj, args.i)
    Tj = load_tabletop_pose(args.obj, args.j)
    T_obj_start = make_obj_pose(Ti, np.array([args.pickup_x, 0.0, Ti[2, 3]]),
                                args.pickup_tz)
    T_obj_end = make_obj_pose(Tj, np.array([args.place_x, args.place_y, Tj[2, 3] + h_m]),
                              args.place_tz)

    print(f"[reset] init planner (hand={args.hand}) ...")
    t0 = time.time()
    planner, base_world = init_planner(args.hand)
    urdf_fk, ee_link = load_fk_urdf(args.hand)
    obj_verts = load_object_vertices(args.obj)
    print(f"[reset] planner warmup: {time.time() - t0:.1f}s ({len(obj_verts)} mesh verts)")

    result = plan_one_cell(
        planner, obj_name=args.obj, hand=args.hand,
        h_cm=args.h_cm, i=args.i, j=args.j,
        T_obj_start=T_obj_start, T_obj_end=T_obj_end,
        base_world=base_world, max_seeds=args.max_seeds,
        urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
    )

    if result is None:
        print("[reset] NO success")
        return

    print(f"[reset] SUCCESS — seed={result['seed_id']} (idx={result['seed_idx']})")
    for ph in PHASE_NAMES:
        print(f"  {ph}: {result['trajs'][ph].shape}")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[3]
        / "outputs" / "reset_plans" / args.hand / args.obj
        / f"reorient_{args.h_cm}" / f"{args.i}_{args.j}"
        / f"x{args.pickup_x:.2f}_tz{int(round(args.pickup_tz)):03d}"
        / result["seed_id"]
    )
    meta = {
        "obj_name": args.obj, "hand": args.hand,
        "i": args.i, "j": args.j, "h_cm": args.h_cm,
        "pickup_x": args.pickup_x, "pickup_tz": args.pickup_tz,
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
    print(f"[reset] saved -> {out_dir}")
    print(f"[reset] view: python src/grasp_generation/reorient/view_reset.py --plan_dir {out_dir}")


if __name__ == "__main__":
    main()
