"""触觉/接触感知（仿真接触数据）。

重要声明
--------
本模块输出的是 **MuJoCo 物理求解器计算出的仿真接触力**，
不是真实触觉传感器（如 ATI 力/力矩传感器、GelSight）的数据。
`GraspContactReport.source` 字段始终携带该声明，导出 JSON 时也会保留。

抓取稳定性判据（规则化，非学习模型）
------------------------------------
    stable = 两指都有接触
            且 峰值接触力 > force_min
            且 物体速度 < speed_max
            且 接触数 >= min_contacts
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.types import ContactSample, GraspContactReport  # noqa: E402

FINGER_BODIES = ("left_finger", "right_finger")


class ContactSensor:
    def __init__(self, env, force_min: float = 0.3, speed_max: float = 0.05,
                 min_contacts: int = 2):
        self.env = env
        self.force_min = float(force_min)
        self.speed_max = float(speed_max)
        self.min_contacts = int(min_contacts)

    # ---------- 原始接触 ----------
    def samples(self, body_filter: Optional[Iterable[str]] = None):
        out = []
        for c in self.env.contacts():
            if body_filter is not None and not (
                    c.body1 in body_filter or c.body2 in body_filter):
                continue
            out.append(ContactSample(
                body_a=c.body1, body_b=c.body2,
                geom_a=self.env.geom_name(c.geom1),
                geom_b=self.env.geom_name(c.geom2),
                force_n=c.force_magnitude,
                force_world=c.force[3:].copy(),
                position=c.position.copy(), normal=c.normal.copy(),
                penetration=float(-c.dist) if c.dist < 0 else 0.0,
            ))
        return out

    def object_speed(self, object_name: str) -> float:
        """物体的线速度大小（自由关节 qvel 的前 3 项）。"""
        import mujoco
        m = self.env.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_free")
        if jid < 0:
            return 0.0
        adr = int(m.jnt_dofadr[jid])
        return float(np.linalg.norm(self.env.data.qvel[adr:adr + 3]))

    # ---------- 汇总 ----------
    def grasp_report(self, object_name: str = "red_cube") -> GraspContactReport:
        allc = self.samples()
        rel = [c for c in allc
               if object_name in (c.body_a, c.body_b)
               and ({c.body_a, c.body_b} & set(FINGER_BODIES))]
        forces = [c.force_n for c in rel]
        left_force = float(sum(c.force_n for c in rel
                               if "left_finger" in (c.body_a, c.body_b)))
        right_force = float(sum(c.force_n for c in rel
                                if "right_finger" in (c.body_a, c.body_b)))
        left = left_force > 0.0
        right = right_force > 0.0
        speed = self.object_speed(object_name)
        peak = max(forces) if forces else 0.0
        # 夹爪由多个小接触面构成，单点峰值偏小；稳定性用"每侧合力"判定
        stable = (left_force > self.force_min and right_force > self.force_min
                  and speed < self.speed_max and len(rel) >= self.min_contacts)
        return GraspContactReport(
            in_contact=bool(rel), n_contacts=len(rel), peak_force_n=peak,
            total_force_n=float(np.sum(forces)),
            contact_positions=[c.position.copy() for c in rel],
            left_finger_contact=bool(left), right_finger_contact=bool(right),
            object_speed=speed, grasp_stable=bool(stable),
            left_force_n=left_force, right_force_n=right_force,
            samples=rel, object_name=object_name,
        )

    def obstacle_contacts(self, robot_prefixes=("link", "hand", "finger")) -> list:
        """机器人与非目标物体的接触（用于碰撞统计）。"""
        out = []
        for c in self.samples():
            is_robot = (c.body_a.startswith(robot_prefixes)
                        or c.body_b.startswith(robot_prefixes))
            if is_robot:
                out.append(c)
        return out
