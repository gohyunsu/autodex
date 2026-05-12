"""
Real-world grasp executor for xArm + Allegro hand.

Autonomous (no GUI) trajectory execution.

Execution sequence (matches RSS2026 reference: planner/inference/train/run_auto_v2.py):
    execute:  init(joint0) -> approach(traj) -> pregrasp -> grasp -> squeeze -> lift -> place
    release:  reverse_squeeze -> grasp -> pregrasp -> hand_init -> arm_return

Usage:
    executor = RealExecutor()
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
import datetime
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner import PlanResult
from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT,
    ALLEGRO_INIT, ALLEGRO_LINK6_TO_WRIST,
    INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
)

# Per-hand config: (init_joints, link6_to_wrist, convert_fn)
def _convert_allegro(hand_pose: np.ndarray) -> np.ndarray:
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    if hand_pose.ndim == 1:
        out = hand_pose.copy()
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out = hand_pose.copy()
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out

def _convert_inspire(hand_pose: np.ndarray) -> np.ndarray:
    """Convert inspire qpos (radians) to controller action (0-1000).

    qpos order:   [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    action order:  [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    """
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
    if hand_pose.ndim == 1:
        q = hand_pose[:6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros(6, dtype=np.float64)
        action[0] = np.clip(action_float[5], 0, 1000)  # pinky
        action[1] = np.clip(action_float[4], 0, 1000)  # ring
        action[2] = np.clip(action_float[3], 0, 1000)  # middle
        action[3] = np.clip(action_float[2], 0, 1000)  # index
        action[4] = np.clip(action_float[1], 0, 1000)  # thumb_pitch
        action[5] = np.clip(action_float[0], 0, 1000)  # thumb_yaw
    else:
        q = hand_pose[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros_like(hand_pose)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000)
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000)
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000)
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000)
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000)
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000)
    return action

HAND_CONFIG = {
    "allegro": {
        "init": ALLEGRO_INIT,
        "link6_to_wrist": ALLEGRO_LINK6_TO_WRIST,
        "convert": _convert_allegro,
        "xarm_init": XARM_INIT,
    },
    "inspire": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
    "inspire_left": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LEFT_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
}


class RealExecutor:
    def __init__(
        self,
        arm_name: str = "xarm",
        hand_name: str = "allegro",
        dt: float = 0.01,
        squeeze_level: int = 10,
    ):
        if hand_name not in HAND_CONFIG:
            raise ValueError(f"Unknown hand: {hand_name}. Choose from {list(HAND_CONFIG)}")
        self.dt = dt
        self.squeeze_level = squeeze_level
        self.hand_name = hand_name

        hcfg = HAND_CONFIG[hand_name]
        self._convert = hcfg["convert"]
        self._hand_init = hcfg["init"]
        self._link6_to_wrist = hcfg["link6_to_wrist"]
        self._xarm_init = hcfg["xarm_init"]

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Safety velocity limits
        self.joint_vel_limit = 0.05
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    # ── low-level motion primitives ──────────────────────────────────────

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02):
        for i in range(len(arm_traj)):
            target_arm = arm_traj[i]
            target_hand = hand_traj[i] if hand_traj is not None else None
            if target_hand is not None:
                self.hand.move(target_hand)
            stall_count = 0
            prev_qpos = None
            recovered = False
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print("[executor] stall detected, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print("[executor] stall after recovery, aborting")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, target_arm)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.linalg.norm(self.arm.get_data()["qpos"] - target_arm) < threshold:
                    break

    def _move_hand(self, target):
        self.hand.move(target)
        time.sleep(self.dt)

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02,
                        vel_scale=1.0, stop_on_stall=False,
                        stall_window=30, stall_progress_ratio=0.3):
        """Stall detection is window-based + ratio to commanded velocity:
        over the last `stall_window` ticks the arm should advance roughly
        (cart_vel_limit * vel_scale * stall_window) meters in free motion.
        If actual progress < `stall_progress_ratio` of that expected, we count
        as stalled — robust to both reading latency and xarm yielding on contact.

        stop_on_stall=True breaks immediately on stall (placing mode — stop on
        contact, don't clear_error or retry).
        """
        from collections import deque

        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        pos_history = deque(maxlen=stall_window)
        expected_progress = self.cart_vel_limit * vel_scale * stall_window
        stall_thresh = expected_progress * stall_progress_ratio
        stalled = False
        recovered = False
        recover_count = 0
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            cur_pos = cur[:3, 3].copy()
            pos_history.append(cur_pos)
            # Stall = full window collected and progress < expected*ratio.
            if len(pos_history) == stall_window:
                progress = np.linalg.norm(pos_history[-1] - pos_history[0])
                stalled = (progress < stall_thresh)
            if stalled:
                if stop_on_stall:
                    print(f"[executor] stall detected (window {stall_window} ticks, "
                          f"progress {progress*1000:.2f}mm) — stopping (placing mode)")
                    break
                if not recovered:
                    print("[executor] stall detected, clearing error...")
                    self.arm.clear_error()
                    recovered = True
                    pos_history.clear()
                    stalled = False
                else:
                    recover_count += 1
                    if recover_count >= stall_window:
                        print("[executor] stall after recovery, aborting")
                        break
            prev_pos = cur_pos
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            vel = self.cart_vel_limit * vel_scale
            if t_dist > vel:
                t_delta = t_delta / t_dist * vel
            cur[:3, 3] += t_delta
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            if r_dist > self.rot_vel_limit:
                r_delta = r_delta / r_dist * self.rot_vel_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.06):
        current_target = self.arm.get_data()["qpos"].copy()
        for j in joint_order:
            current_target[j] = target_qpos[j]
            stall_count = 0
            prev_qpos = None
            recovered = False
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print(f"[executor] joint {j} stall, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print(f"[executor] joint {j} stall after recovery, skipping")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, current_target, vel_limit=0.06)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if np.abs(self.arm.get_data()["qpos"][j] - target_qpos[j]) < threshold:
                    break

    # ── public API ────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        self.arm.stop()
        self.hand.stop()

    def _log_state(self, state):
        ts = datetime.datetime.now().isoformat()
        self.state_timestamps.append({"state": state, "time": ts})

    def execute(self, plan_result: PlanResult, lift_height: float = 0.10):
        """
        Execute: init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        State timestamps stored in self.state_timestamps.
        Returns the squeezed hand pose.

        Place (descend) is now a separate `place(plan_result, ...)` call so
        callers can do work (e.g. capture label image) while the object is
        held up.
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None

        self.state_timestamps = []
        traj = plan_result.traj
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        wrist_ee = plan_result.wrist_se3 @ np.linalg.inv(self._link6_to_wrist)

        sl = self.squeeze_level

        # 1. Return to init pose (joint 0 first)
        self._log_state("init")
        self._move_joint_sequential(self._xarm_init[:6], [0])

        # 2. Approach trajectory
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        self._move_joints(traj[:, :6], hand_traj)

        # 3. Pregrasp
        self._log_state("pregrasp")
        self._move_hand(pg_hand)

        # 4. Grasp
        self._log_state("grasp")
        self._move_hand(g_hand)

        # 5. Squeeze
        self._log_state("squeeze")
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        # 6. Lift
        self._log_state("lift")
        lift_pose = wrist_ee.copy()
        lift_pose[2, 3] += lift_height
        self._move_cartesian(lift_pose, vel_scale=1/1.5)

        self._log_state("lift_done")
        return s_hand

    def place(self, plan_result: PlanResult, lift_height: float = 0.10,
              overshoot: float = 0.05,
              mcc_model_path: str = None,
              descend_time_s: float = 8.0,
              total_time_s: float = 12.0,
              log_path: str = None) -> dict:
        """Descend with mcc_minimal admittance control. Target = lift_height +
        overshoot below current z; arm yields naturally on table/object contact
        via learned tau-model admittance loop.

        Hands the arm off from paradex's XArmController to a fresh XArmAPI for
        the mcc loop, then re-inits paradex after."""
        import sys
        from pathlib import Path

        if not plan_result.success:
            return {"descended": 0.0, "stopped_on_contact": False, "target": 0.0}

        if mcc_model_path is None:
            mcc_model_path = str(Path.home() / "mcc_minimal" / "results"
                                 / "tau_model_inspire_left.pt")

        from paradex.io.robot_controller.xarm_controller import homo2cart

        self._log_state("place")
        target_descend = lift_height + overshoot
        start_q = self.arm.get_data()["qpos"][:6].copy().astype(np.float64)
        current_pos = self.arm.get_data()["position"].copy()

        # Compute target cart + IK via xarm SDK before disconnecting.
        target_pos = current_pos.copy()
        target_pos[2, 3] -= target_descend
        target_cart = homo2cart(target_pos)  # [x_mm, y_mm, z_mm, r_rad, p_rad, y_rad]
        code, target_q_deg = self.arm.arm.get_inverse_kinematics(
            target_cart.tolist(), input_is_radian=True, return_is_radian=False)
        if code != 0:
            print(f"[place] IK failed for descent target (code={code}). aborting place.")
            self._log_state("place_done")
            return {"descended": 0.0, "stopped_on_contact": False, "target": target_descend,
                    "reason": "ik_failed"}
        target_q = np.deg2rad(np.asarray(target_q_deg[:6], dtype=np.float64))

        # Adapter: paradex's control thread keeps running (so its recording
        # continues), but mcc's writes are redirected to xarm_ctrl.action and
        # paradex sends them. Reads delegate to the raw XArmAPI handle.
        xarm_ctrl = self.arm
        xarm_handle = xarm_ctrl.arm   # raw XArmAPI

        # mcc-needed handle setup (one-shot, harmless to paradex).
        xarm_handle.set_report_tau_or_i(1)
        xarm_handle.set_collision_sensitivity(0)

        # Import mcc_minimal (only tau model loading + input encoding).
        mcc_dir = str(Path.home() / "mcc_minimal")
        if mcc_dir not in sys.path:
            sys.path.insert(0, mcc_dir)
        import torch  # noqa: E402
        from fit_tau_model import load_model, build_input  # noqa: E402

        print(f"[place] loading mcc model: {mcc_model_path}")
        model = load_model(mcc_model_path)

        # Contact-stop loop: use the learned tau model only to estimate tau_ext;
        # on contact (tau_ext > threshold), freeze q_des at current pose (paradex
        # holds it) and break. No yield, no bounce.
        DT = 0.01
        LOOP_HZ = 100
        FILTER_ALPHA = 0.1
        QDOT_SMOOTH_ALPHA = 0.1
        WARMUP_SEC = 1.0
        # baseline noise per joint (Nm) — from mcc DEADBAND_J. Contact threshold
        # is some multiplier above this.
        DEADBAND_J = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 0.5])
        CONTACT_MULT = 3.0
        CONTACT_THRESH = DEADBAND_J * CONTACT_MULT

        def _read():
            _, q_deg = xarm_handle.get_servo_angle()
            q = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64))
            # tau_motor from stream (paradex has report_type='real' on the XArmAPI)
            KT_GEAR = 1.0  # mcc applies this internally; for raw stream both sides cancel in tau_ext
            tau = np.asarray(xarm_handle._arm._joints_torque[:6], dtype=np.float64)
            return q, tau

        def _push_action(q_cmd):
            with xarm_ctrl.lock:
                xarm_ctrl.action = q_cmd.astype(np.float64)
                xarm_ctrl.is_servo = True

        # Make sure tau reporting is on and start with paradex holding start_q.
        _push_action(start_q)

        # Warmup: prime tau_filt at hold pose.
        tau_filt = np.zeros(6)
        qdot_smooth = np.zeros(6)
        q_last, t_last = None, None
        t_warm0 = time.time()
        while time.time() - t_warm0 < WARMUP_SEC:
            q, tau_motor = _read()
            t_now = time.time()
            if q_last is not None and t_last is not None:
                dt = max(t_now - t_last, 1e-4)
                qdot = (q - q_last) / dt
            else:
                qdot = np.zeros(6)
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth
            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt
            _push_action(start_q)
            time.sleep(DT)
        print(f"[place] warmup done. baseline tau_filt = {tau_filt.round(2)}")
        print(f"[place] contact threshold per joint = {CONTACT_THRESH.round(2)}")

        # Descend loop with contact stop.
        log = [] if log_path else None
        contact = False
        contact_t = None
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < total_time_s:
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += DT
            t = time.time() - t0

            q, tau_motor = _read()
            t_now = time.time()
            dt = max(t_now - t_last, 1e-4)
            qdot = (q - q_last) / dt
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth

            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt

            # Check contact.
            if not contact and np.any(np.abs(tau_filt) > CONTACT_THRESH):
                contact = True
                contact_t = t
                worst = int(np.argmax(np.abs(tau_filt)))
                print(f"[place] CONTACT detected at t={t:.2f}s — "
                      f"joint {worst+1} tau_ext={tau_filt[worst]:.2f}Nm "
                      f"(thresh {CONTACT_THRESH[worst]:.2f})")
                # Freeze at current pose; break.
                _push_action(q)
                if log is not None:
                    log.append((t, *q, *tau_filt, 1))
                break

            # Lerp q_des down toward target. After contact this wouldn't run.
            alpha = min(1.0, max(0.0, t / descend_time_s))
            q_des = (1 - alpha) * start_q + alpha * target_q
            _push_action(q_des)

            if log is not None:
                log.append((t, *q, *tau_filt, 0))

        if not contact:
            print(f"[place] no contact within {total_time_s}s — reached target. final pose held.")

        # Optional CSV log.
        if log_path and log:
            import csv as _csv
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["t"] + [f"q{i}" for i in range(6)] +
                           [f"tau_ext{i}" for i in range(6)] + ["contact"])
                w.writerows(log)
            print(f"[place] log -> {log_path}")

        # Read final pose for descended-distance reporting.
        try:
            _, final_pos_xarm = xarm_handle.get_position(is_radian=True)
        except Exception:
            final_pos_xarm = None

        # Compute descended distance from final pose.
        if final_pos_xarm is not None:
            final_z = final_pos_xarm[2] / 1000.0  # mm -> m
            descended = current_pos[2, 3] - final_z
        else:
            descended = float("nan")
        stopped = descended < target_descend - 0.005 if descended == descended else False
        print(f"[place] descended {descended*1000:.1f}mm of target {target_descend*1000:.0f}mm "
              f"({'yielded on contact' if stopped else 'reached target'})")

        # paradex thread never stopped; it's been forwarding mcc's q_ref the
        # whole time, so no re-init needed. Just leave its action at last q_ref.

        self._log_state("place_done")
        return {"descended": float(descended), "stopped_on_contact": bool(stopped),
                "target": float(target_descend)}

    def release(self, plan_result: PlanResult):
        """Release object and return arm to init pose."""
        if not plan_result.success:
            return

        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        self._release_auto(pg_hand, g_hand)

    def _release_auto(self, pg_hand, g_hand):
        """Reverse squeeze -> grasp -> pregrasp, then STOP.
        Hand opening to hand_init and arm retract back to init are intentionally
        skipped — user resets those manually after inspecting the placed object."""
        sl = self.squeeze_level

        # Reverse squeeze
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.01)

        self._move_hand(g_hand)
        time.sleep(0.01)
        self._move_hand(pg_hand)

    def shutdown(self):
        self.arm.end()
        self.hand.end()
