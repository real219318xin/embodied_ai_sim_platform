"""运动学与逆运动学（阻尼最小二乘 DLS）。

设计要点
--------
* 正运动学：TCP 点 = hand body 原点 + R_hand · (0, 0, TCP_OFFSET)
* 逆运动学在**独立的 scratch MjData** 上迭代求解，不触碰真实仿真状态，
  求解结果再通过 actuator 位置指令下发给物理仿真执行 —— 不存在 teleport。
* 关节限位：求解时把 q 裁剪在 jnt_range 内，并对超限残差做惩罚。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import SimEnv, body_id, TCP_OFFSET  # noqa: E402

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    success: bool
    iterations: int


class PandaKinematics:
    def __init__(self, env: SimEnv):
        self.env = env
        self.model = env.model
        self.hand_id = body_id(self.model, "hand")
        self.arm_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES])
        self.arm_dadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES])
        self.jnt_lo = self.model.jnt_range[
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in ARM_JOINT_NAMES], 0].copy()
        self.jnt_hi = self.model.jnt_range[
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in ARM_JOINT_NAMES], 1].copy()
        # 逆解专用 scratch 状态
        self.scratch = mujoco.MjData(self.model)

    # ---------------- 状态 ----------------
    def q(self) -> np.ndarray:
        return self.env.data.qpos[self.arm_qadr].copy()

    def qdot(self) -> np.ndarray:
        return self.env.data.qvel[self.arm_dadr].copy()

    def finger_opening(self) -> np.ndarray:
        d = self.env.data
        return np.array([d.qpos[7], d.qpos[8]])

    def tcp_pose(self, data=None) -> Tuple[np.ndarray, np.ndarray]:
        d = self.env.data if data is None else data
        R = d.xmat[self.hand_id].reshape(3, 3)
        p = d.xpos[self.hand_id] + R @ np.array([0.0, 0.0, TCP_OFFSET])
        return p.copy(), R.copy()

    # ---------------- 逆解 ----------------
    def solve_ik(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None,
                 q_seed: Optional[np.ndarray] = None, max_iter: int = 200,
                 tol_pos: float = 1e-3, tol_rot: float = 0.02,
                 damping: float = 0.02, max_step: float = 0.08) -> IKResult:
        m = self.model
        d = self.scratch
        # 从当前真实状态复制（不改动真实 data）
        d.qpos[:] = self.env.data.qpos
        d.qvel[:] = 0.0
        if q_seed is not None:
            d.qpos[self.arm_qadr] = np.clip(q_seed, self.jnt_lo, self.jnt_hi)
        mujoco.mj_forward(m, d)

        target_pos = np.asarray(target_pos, float)
        point = np.zeros(3)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        pos_err = rot_err = np.inf
        for it in range(max_iter):
            R = d.xmat[self.hand_id].reshape(3, 3)
            p = d.xpos[self.hand_id] + R @ np.array([0, 0, TCP_OFFSET])
            e_pos = target_pos - p
            pos_err = float(np.linalg.norm(e_pos))

            if target_rot is None:
                e = e_pos
                J = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacp, jacr, p, self.hand_id)
                J[:] = jacp
                rot_err = 0.0
            else:
                # 旋转误差：把 R_target 与 R 的差转成小角度旋转向量
                dR = target_rot @ R.T
                ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                if ang < 1e-8:
                    e_rot = np.zeros(3)
                else:
                    axis = np.array([dR[2, 1] - dR[1, 2],
                                     dR[0, 2] - dR[2, 0],
                                     dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
                    e_rot = axis * ang
                rot_err = float(ang)
                mujoco.mj_jac(m, d, jacp, jacr, p, self.hand_id)
                e = np.concatenate([e_pos, e_rot])
                J = np.vstack([jacp, jacr])

            if pos_err < tol_pos and rot_err < tol_rot:
                return IKResult(d.qpos[self.arm_qadr].copy(), pos_err, rot_err, True, it)

            Jarm = J[:, self.arm_dadr]
            lam = damping ** 2
            JJt = Jarm @ Jarm.T + lam * np.eye(J.shape[0])
            dq = Jarm.T @ np.linalg.solve(JJt, e)
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq *= max_step / nrm
            q = d.qpos[self.arm_qadr] + dq
            q = np.clip(q, self.jnt_lo, self.jnt_hi)
            d.qpos[self.arm_qadr] = q
            mujoco.mj_forward(m, d)

        return IKResult(d.qpos[self.arm_qadr].copy(), pos_err, rot_err, False, max_iter)

    # ---------------- 工具 ----------------
    def within_limits(self, q: np.ndarray, margin: float = 1e-6) -> bool:
        return bool(np.all(q >= self.jnt_lo - margin) and np.all(q <= self.jnt_hi + margin))

    def limit_violation(self, q: np.ndarray) -> float:
        return float(max(0.0, np.max(self.jnt_lo - q), np.max(q - self.jnt_hi)))


def down_grasp_rotation(yaw: float = 0.0) -> np.ndarray:
    """构造"夹爪朝下、手指沿 y 方向闭合"的目标姿态（可绕世界 z 旋转 yaw）。"""
    z = np.array([0.0, 0.0, -1.0])       # 手指指向（hand z 轴）
    y = np.array([0.0, 1.0, 0.0])        # 手指开合方向
    R = np.column_stack([np.cross(y, z), y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ R
