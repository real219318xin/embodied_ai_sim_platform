"""本体感知：关节状态、末端位姿、夹爪开度、运动状态。

数据来源全部是 MuJoCo 的传感器量：
    qpos / qvel          -> 关节角度与角速度
    actuator_force       -> 关节实际出力
    hand body 位姿       -> 末端执行器位姿（TCP 由固定偏移换算）
    finger joint qpos    -> 夹爪开度
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import TCP_OFFSET, body_id  # noqa: E402
from perception.types import ProprioState  # noqa: E402


class ProprioceptionSensor:
    def __init__(self, env, steady_speed: float = 0.02):
        self.env = env
        self.model = env.model
        self.steady_speed = float(steady_speed)
        self.hand_id = body_id(self.model, "hand")
        self.arm_joints = [f"joint{i}" for i in range(1, 8)]
        self.arm_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.arm_joints])
        self.arm_dadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.arm_joints])
        self.finger_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("finger_joint1", "finger_joint2")])

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        d = self.env.data
        R = d.xmat[self.hand_id].reshape(3, 3)
        p = d.xpos[self.hand_id] + R @ np.array([0.0, 0.0, TCP_OFFSET])
        return p.copy(), R.copy()

    def read(self) -> ProprioState:
        d = self.env.data
        q = d.qpos[self.arm_qadr].copy()
        qd = d.qvel[self.arm_dadr].copy()
        eff = d.actuator_force[:7].copy()
        p, R = self.tcp_pose()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.reshape(-1))
        fingers = d.qpos[self.finger_qadr].copy()
        max_speed = float(np.max(np.abs(qd))) if len(qd) else 0.0
        return ProprioState(
            timestamp=float(d.time),
            joint_positions=q, joint_velocities=qd, joint_efforts=eff,
            tcp_position=p, tcp_rotation=R, tcp_quaternion=quat,
            gripper_opening=float(np.sum(fingers)),
            gripper_finger_positions=fingers,
            motion_state="moving" if max_speed > self.steady_speed else "steady",
            max_joint_speed=max_speed,
        )
