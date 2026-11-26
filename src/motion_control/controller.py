"""Panda 机械臂控制器：全部动作通过 actuator + mj_step 实现。

控制通道
--------
* 关节 1-7 : 官方位置伺服执行器 actuator1..actuator7  (ctrl = 目标关节角 rad)
* 夹爪     : 场景扩展的位置伺服 gripper_left / gripper_right (ctrl = 单指开度 m)
* 官方 actuator8 (tendon) 保持 ctrl=0，不参与控制

不接受任何"直接写 qpos 实现运动"的调用；qpos 只在 IK 的 scratch 数据中
用于求解，求解结果经 actuator 执行。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import SimEnv  # noqa: E402
from motion_control.kinematics import (  # noqa: E402
    PandaKinematics, TCP_OFFSET, down_grasp_rotation)

TENDON_ACT = 7
GRIPPER_LEFT = 8
GRIPPER_RIGHT = 9
OPEN_WIDE = 0.038          # m，单指
OPEN_GRASP = 0.030
CLOSED = 0.0


@dataclass
class MoveResult:
    success: bool
    reason: str = ""
    pos_err: float = 0.0
    rot_err: float = 0.0
    duration: float = 0.0
    steps: int = 0


class PandaController:
    def __init__(self, env: SimEnv):
        self.env = env
        self.kin = PandaKinematics(env)
        self.q_home = env.model.key_qpos[env._key_id][:7].copy()
        self.traj: List[np.ndarray] = []      # 实际执行过的关节轨迹
        self.tcp_traj: List[np.ndarray] = []  # 实际 TCP 轨迹
        self.ctrl_rate = 0.08                 # rad / 2 ms，单步最大关节增量
        self.env.data.ctrl[TENDON_ACT] = 0.0
        # 评估钩子：逐物理步统计接触力峰值/碰撞事件；frame_hook 用于录制视频
        self.monitor: Optional[dict] = None
        self.frame_hook: Optional[Callable] = None

    # ---------------- 基础 ----------------
    def reset(self):
        self.env.reset()
        self.env.data.ctrl[:7] = self.q_home
        self.env.data.ctrl[GRIPPER_LEFT] = self.env.data.ctrl[GRIPPER_RIGHT] = OPEN_GRASP
        self.traj.clear()
        self.tcp_traj.clear()

    def q(self) -> np.ndarray:
        return self.kin.q()

    def tcp(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kin.tcp_pose()

    def finger_opening(self) -> float:
        f = self.kin.finger_opening()
        return float(np.mean(f))

    def _record(self):
        self.traj.append(self.q())
        self.tcp_traj.append(self.tcp()[0])

    def _step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.env.model, self.env.data)
            self._record()
            if self.monitor is not None:
                self._sample_monitor()
            if self.frame_hook is not None:
                self.frame_hook(self.env)

    # ---------------- 评估监视 ----------------
    def enable_monitor(self, target_body: str, sample_every: int = 5):
        """开启逐步接触统计：夹持力峰值 + 非预期接触（碰撞）事件计数。"""
        self.monitor = {"target": target_body, "sample_every": int(sample_every),
                        "peak_grip_force": 0.0, "collision_events": 0,
                        "collision_bodies": set(), "table_contacts": 0,
                        "in_contact": set(), "n_samples": 0}

    def disable_monitor(self) -> dict:
        m = self.monitor or {}
        self.monitor = None
        return m

    def _sample_monitor(self):
        m = self.monitor
        m["n_samples"] += 1
        if m["n_samples"] % m["sample_every"] != 0:
            return
        touching = set()
        for c in self.env.robot_object_contacts():
            other = c.body2 if c.body1.startswith(("link", "hand", "left_finger",
                                                   "right_finger")) else c.body1
            touching.add(other)
            if other == m["target"]:
                if "finger" in c.body1 or "finger" in c.body2:
                    m["peak_grip_force"] = max(m["peak_grip_force"],
                                               c.force_magnitude)
            elif other in ("bench", "world"):
                # 指腹在抓取/放置时蹭到台面：单独统计，不计为碰撞
                if other not in m["in_contact"]:
                    m["table_contacts"] += 1
            else:
                if other not in m["in_contact"]:
                    m["collision_events"] += 1
                    m["collision_bodies"].add(other)
        m["in_contact"] = touching

    # ---------------- 关节运动 ----------------
    def hold(self, steps: int):
        self._step(steps)

    def move_joints(self, q_goal: np.ndarray, duration: float = 1.5,
                    tol: float = 0.02, gripper: Optional[float] = None) -> MoveResult:
        """最小加加速度（min-jerk）关节空间插值，逐物理步下发 ctrl。"""
        q_goal = np.asarray(q_goal, float)
        if not self.kin.within_limits(q_goal):
            return MoveResult(False, f"目标关节角超限 (violation={self.kin.limit_violation(q_goal):.4f} rad)")
        q0 = self.q()
        n = max(1, int(duration / self.env.model.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5   # min-jerk
            self.env.data.ctrl[:7] = q0 + (q_goal - q0) * s
            if gripper is not None:
                self.env.data.ctrl[GRIPPER_LEFT] = self.env.data.ctrl[GRIPPER_RIGHT] = gripper
            self._step(1)
        err = float(np.max(np.abs(self.q() - q_goal)))
        return MoveResult(err < tol, "" if err < tol else f"关节跟踪误差 {err:.4f} rad",
                          duration=duration, steps=n)

    # ---------------- 夹爪 ----------------
    def set_gripper(self, opening: float, steps: int = 250) -> float:
        opening = float(np.clip(opening, 0.0, 0.04))
        self.env.data.ctrl[GRIPPER_LEFT] = self.env.data.ctrl[GRIPPER_RIGHT] = opening
        self._step(steps)
        return self.finger_opening()

    def open_gripper(self, opening: float = OPEN_WIDE, steps: int = 250) -> float:
        return self.set_gripper(opening, steps)

    def close_gripper(self, steps: int = 350, stop_on_contact: bool = True,
                      object_body: Optional[str] = None,
                      force_threshold: float = 0.8,
                      squeeze_steps: int = 300, force_cap: float = 8.0,
                      hold_steps: int = 150) -> dict:
        """闭合夹爪：先扫到首次接触，再做小幅挤压，使两侧指面都压紧。

        实测问题（本机 Panda 模型）：夹爪中心与被抓物体中心通常有零点几毫米
        偏差，一检测到单侧接触就停会导致只有一根手指接触、抓取不稳定。
        因此分两阶段：
            阶段 1  由当前开度线性收到 0，遇到 >force_threshold 的接触即停；
            阶段 2  在首次接触开度基础上再收 squeeze(默认 3 mm)，让另一侧也压紧。
        全程只用 gripper_left/right 位置伺服执行，属于真实物理过程。
        """
        ctrl = self.env.data.ctrl
        info = {"contact": False, "peak_force": 0.0, "steps": 0,
                "opening": None, "first_contact_opening": None,
                "left_contact": False, "right_contact": False}
        start_open = float(np.mean(self.kin.finger_opening()))
        n1 = max(1, steps)
        first = None
        for i in range(n1):
            s = i / n1
            ctrl[GRIPPER_LEFT] = ctrl[GRIPPER_RIGHT] = start_open * (1.0 - s)
            self._step(1)
            cs = self.env.robot_object_contacts()
            if object_body is not None:
                cs = [c for c in cs if object_body in (c.body1, c.body2)]
            if cs:
                f = max(c.force_magnitude for c in cs)
                info["peak_force"] = max(info["peak_force"], f)
                if f > force_threshold:
                    first = (i + 1, float(np.mean(self.kin.finger_opening())))
                    break
        info["steps"] = n1 if first is None else first[0]
        info["first_contact_opening"] = None if first is None else first[1]

        if first is not None:
            # 阶段 2：每步只多收 0.02 mm，直到两侧都接触或力达到上限
            target = first[1]
            n2 = max(1, int(squeeze_steps))
            for j in range(n2):
                target = max(0.0, target - 0.00002)
                ctrl[GRIPPER_LEFT] = ctrl[GRIPPER_RIGHT] = target
                self._step(1)
                cs = self.env.robot_object_contacts()
                if object_body is not None:
                    cs = [c for c in cs if object_body in (c.body1, c.body2)]
                if not cs:
                    continue
                info["peak_force"] = max(
                    info["peak_force"], max(c.force_magnitude for c in cs))
                left = any("left_finger" in (c.body1, c.body2) for c in cs)
                right = any("right_finger" in (c.body1, c.body2) for c in cs)
                if (left and right and info["peak_force"] > force_threshold) \
                        or info["peak_force"] > force_cap:
                    break
                info["steps"] += 1
            # 阶段 3：保持该开度，让接触稳定下来
            ctrl[GRIPPER_LEFT] = ctrl[GRIPPER_RIGHT] = target
            self._step(int(hold_steps))
            info["steps"] += int(hold_steps)

        # 最终接触状态（两侧分别判定）
        cs = self.env.robot_object_contacts()
        if object_body is not None:
            cs = [c for c in cs if object_body in (c.body1, c.body2)]
        info["left_contact"] = any(
            "left_finger" in (c.body1, c.body2) for c in cs)
        info["right_contact"] = any(
            "right_finger" in (c.body1, c.body2) for c in cs)
        if cs:
            info["peak_force"] = max(info["peak_force"],
                                     max(c.force_magnitude for c in cs))
            info["contact"] = info["peak_force"] > force_threshold
        info["opening"] = self.finger_opening()
        info["both_fingers"] = info["left_contact"] and info["right_contact"]
        return info

    # ---------------- 笛卡尔运动 ----------------
    def move_to_pose(self, pos: np.ndarray, rot: Optional[np.ndarray] = None,
                     max_duration: float = 4.0, tol_pos: float = 0.005,
                     tol_rot: float = 0.05, gripper: Optional[float] = None,
                     check_collision: bool = True, allow_body: Optional[str] = None,
                     max_step: float = 0.6) -> MoveResult:
        """闭环笛卡尔伺服：每步求解 IK -> 限速 -> 经 actuator 执行。"""
        dt = self.env.model.opt.timestep
        n_max = int(max_duration / dt)
        start = self.env.time
        p_err = r_err = np.inf
        # 外环积分：抵消位置/姿态伺服在重力与负载下的稳态误差（不修改任何模型参数）
        cmd = np.asarray(pos, float).copy()
        accum = np.zeros(3)
        r_accum = np.zeros(3)          # 世界系轴角修正，用于姿态积分
        cmd_rot = None if rot is None else np.asarray(rot, float).copy()
        for i in range(n_max):
            q_cur = self.q()
            seed = q_cur
            res = self.kin.solve_ik(cmd, cmd_rot, q_seed=seed, max_iter=60,
                                    tol_pos=tol_pos * 0.25,
                                    tol_rot=(tol_rot * 0.25 if rot is not None else 1e-6))
            q_cmd = res.q
            dq = q_cmd - q_cur
            n = np.linalg.norm(dq)
            if n > max_step:
                q_cmd = q_cur + dq * (max_step / n)
            self.env.data.ctrl[:7] = q_cmd
            if gripper is not None:
                self.env.data.ctrl[GRIPPER_LEFT] = self.env.data.ctrl[GRIPPER_RIGHT] = gripper
            self._step(1)

            p, R = self.tcp()
            p_err = float(np.linalg.norm(p - pos))
            r_err = 0.0 if rot is None else float(
                np.arccos(np.clip((np.trace(rot @ R.T) - 1) / 2, -1, 1)))
            if p_err < tol_pos and r_err < tol_rot:
                return MoveResult(True, "", p_err, r_err, self.env.time - start, i + 1)
            if check_collision:
                bad = [c for c in self.env.robot_object_contacts()
                       if allow_body not in (c.body1, c.body2)]
                if bad and i > 20:
                    return MoveResult(False, f"运动过程中发生非预期接触: "
                                             f"{bad[0].body1}<->{bad[0].body2} "
                                             f"d={bad[0].dist:.4f}",
                                      p_err, r_err, self.env.time - start, i + 1)

            # 积分项：只修正"已接近目标但存在稳态误差"的情况，防止积分饱和
            # 把末端压进物体（小误差才积分，且限幅 ±4 cm）
            if i > 10 and p_err < 0.05:
                accum = np.clip(accum + (pos - p) * 0.15, -0.04, 0.04)
                cmd = np.asarray(pos, float) + accum
            # 姿态积分：只在小角度偏差时启用，限幅 ±6°（同样只改指令，不改模型）
            if rot is not None and i > 10 and r_err < 0.25:
                err_vec = Rotation.from_matrix(rot @ R.T).as_rotvec()
                r_accum = np.clip(r_accum + err_vec * 0.10, -0.105, 0.105)
                cmd_rot = Rotation.from_rotvec(r_accum).as_matrix() @ rot
        return MoveResult(False, f"超时未收敛 (pos_err={p_err*1000:.1f} mm, "
                                 f"rot_err={np.degrees(r_err):.1f} deg)",
                          p_err, r_err, self.env.time - start, n_max)

    def move_linear(self, pos_target: np.ndarray, rot: Optional[np.ndarray] = None,
                    speed: float = 0.08, gripper: Optional[float] = None,
                    **kw) -> MoveResult:
        """笛卡尔直线插值（分小段调用 move_to_pose），用于接近/下压/抬升。"""
        p0, R0 = self.tcp()
        dist = float(np.linalg.norm(pos_target - p0))
        n_seg = max(1, int(np.ceil(dist / 0.01)))
        last = MoveResult(True)
        for k in range(1, n_seg + 1):
            a = k / n_seg
            p = p0 + (pos_target - p0) * a
            r = rot if rot is not None else R0
            last = self.move_to_pose(p, r, max_duration=max(0.5, dist / max(speed, 1e-3) / n_seg * 5),
                                     gripper=gripper, **kw)
            if not last.success:
                return last
        return last

    # ---------------- 碰撞 ----------------
    def collision_report(self, allowed_objects: Sequence[str] = ()) -> dict:
        cs = self.env.robot_object_contacts()
        bad = [c for c in cs if c.body1 not in allowed_objects and c.body2 not in allowed_objects]
        return {
            "collisions": [
                {"a": c.body1, "b": c.body2, "dist": c.dist,
                 "force": c.force_magnitude, "pos": c.position.tolist()}
                for c in bad],
            "count": len(bad),
        }

    def joint_effort(self) -> np.ndarray:
        return self.env.data.actuator_force[:7].copy()
