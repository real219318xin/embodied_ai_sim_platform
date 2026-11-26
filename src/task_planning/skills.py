"""可复用技能库（真实物理执行）。

每个技能只做一件事，全部通过 `PandaController`（actuator + mj_step）执行：
    open_gripper / move_above / descend_to / close_gripper / lift_by /
    transport_xy / place_at / release_and_retreat

技能返回 `SkillResult`，失败时带明确原因，便于上层做局部重试或重新规划。
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_control.controller import PandaController, OPEN_WIDE  # noqa: E402
from motion_control.kinematics import down_grasp_rotation  # noqa: E402

PREGRASP_DZ = 0.14
GRASP_DZ = 0.004      # 抓取点高于物体中心，避免手指下缘插入台面
SAFE_Z = 0.60         # 搬运安全高度（高于黄色障碍物顶部 0.52，且远离臂展边界）
PLACE_CLEARANCE = 0.004


@dataclass
class SkillResult:
    name: str
    ok: bool
    reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {"skill": self.name, "ok": bool(self.ok), "reason": self.reason,
                "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()},
                "duration": round(float(self.duration), 3)}


class SkillLibrary:
    def __init__(self, env, controller: PandaController, verbose: bool = True):
        self.env = env
        self.ctrl = controller
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"    {msg}", flush=True)

    # ---------------- 基础 ----------------
    def open_gripper(self, width: float = OPEN_WIDE, steps: int = 220) -> SkillResult:
        t0 = self.env.time
        opening = self.ctrl.open_gripper(width, steps=steps)
        ok = opening > 0.02
        return SkillResult("open_gripper", ok,
                           "" if ok else f"开度仅 {opening*1000:.1f} mm",
                           {"opening": opening}, self.env.time - t0)

    def move_above(self, pos, yaw: float = 0.0, dz: float = PREGRASP_DZ,
                   gripper: float = OPEN_WIDE, allowed: Optional[str] = None,
                   max_duration: float = 6.0) -> SkillResult:
        t0 = self.env.time
        target = np.array([pos[0], pos[1], pos[2] + dz])
        mr = self.ctrl.move_to_pose(target, down_grasp_rotation(yaw),
                                   max_duration=max_duration, gripper=gripper,
                                   allow_body=allowed, tol_pos=0.003)
        tcp, _ = self.ctrl.tcp()
        return SkillResult("move_above", mr.success, mr.reason,
                           {"pos_err": float(np.linalg.norm(tcp - target))},
                           self.env.time - t0)

    def descend_to(self, pos, yaw: float = 0.0, gripper: float = OPEN_WIDE,
                   allowed: Optional[str] = None, tol: float = 0.0015) -> SkillResult:
        t0 = self.env.time
        mr = self.ctrl.move_linear(np.asarray(pos, float), down_grasp_rotation(yaw),
                                   speed=0.05, gripper=gripper,
                                   allow_body=allowed, tol_pos=tol)
        tcp, _ = self.ctrl.tcp()
        return SkillResult("descend_to", mr.success, mr.reason,
                           {"pos_err": float(np.linalg.norm(tcp - np.asarray(pos)))},
                           self.env.time - t0)

    def close_gripper(self, object_name: str, force_threshold: float = 0.3) -> SkillResult:
        t0 = self.env.time
        info = self.ctrl.close_gripper(steps=500, object_body=object_name,
                                       force_threshold=force_threshold)
        ok = bool(info["contact"])
        detail = "" if ok else "闭合过程中未检测到与目标的接触"
        return SkillResult("close_gripper", ok, detail,
                           {"peak_force": info["peak_force"],
                            "opening": info["opening"] or 0.0,
                            "both_fingers": float(bool(info.get("both_fingers")))},
                           self.env.time - t0)

    def lift_by(self, object_name: str, dz: float = 0.12,
                gripper: float = 0.0) -> SkillResult:
        t0 = self.env.time
        p0 = self.env.body_pos(object_name).copy()
        target = np.array([p0[0], p0[1], p0[2] + dz])
        mr = self.ctrl.move_linear(target, down_grasp_rotation(0.0), speed=0.04,
                                   gripper=gripper, allow_body=object_name)
        p1 = self.env.body_pos(object_name)
        moved = float(p1[2] - p0[2])
        ok = mr.success and moved > 0.05
        reason = mr.reason if not mr.success else (
            "" if moved > 0.05 else f"物体仅上升 {moved*1000:.1f} mm，未被夹住")
        return SkillResult("lift_by", ok, reason,
                           {"object_dz": moved,
                            "gripper_opening": self.ctrl.finger_opening()},
                           self.env.time - t0)

    def transport_xy(self, xy, safe_z: float = SAFE_Z, gripper: float = 0.0,
                     object_name: Optional[str] = None) -> SkillResult:
        """先升到安全高度，再平移到目标 xy（两段直线，避开障碍物顶部）。"""
        t0 = self.env.time
        p0 = self.env.body_pos(object_name).copy() if object_name else self.ctrl.tcp()[0]
        up = np.array([p0[0], p0[1], safe_z])
        mr1 = self.ctrl.move_linear(up, down_grasp_rotation(0.0), speed=0.05,
                                    gripper=gripper, allow_body=object_name,
                                    tol_pos=0.010, tol_rot=0.08)
        if not mr1.success:
            return SkillResult("transport_xy", False, f"抬升段失败: {mr1.reason}",
                               {}, self.env.time - t0)
        goal = np.array([xy[0], xy[1], safe_z])
        mr2 = self.ctrl.move_linear(goal, down_grasp_rotation(0.0), speed=0.07,
                                    gripper=gripper, allow_body=object_name,
                                    tol_pos=0.010, tol_rot=0.08)
        dist = float(np.linalg.norm(goal[:2] - p0[:2]))
        return SkillResult("transport_xy", mr2.success, mr2.reason,
                           {"xy_distance": dist}, self.env.time - t0)

    def place_at(self, xy, height: float, gripper: float = 0.0,
                 object_name: Optional[str] = None) -> SkillResult:
        t0 = self.env.time
        target = np.array([xy[0], xy[1], height])
        mr = self.ctrl.move_linear(target, down_grasp_rotation(0.0), speed=0.04,
                                   gripper=gripper, allow_body=object_name,
                                   tol_pos=0.003)
        return SkillResult("place_at", mr.success, mr.reason, {}, self.env.time - t0)

    def release_and_retreat(self, retreat_dz: float = 0.10,
                            object_name: Optional[str] = None) -> SkillResult:
        t0 = self.env.time
        opening = self.ctrl.open_gripper(OPEN_WIDE, steps=250)
        p, _ = self.ctrl.tcp()
        up = np.array([p[0], p[1], p[2] + retreat_dz])
        mr = self.ctrl.move_linear(up, down_grasp_rotation(0.0), speed=0.06,
                                   gripper=OPEN_WIDE, allow_body=object_name)
        self.ctrl.hold(150)
        ok = opening > 0.02 and mr.success
        return SkillResult("release_and_retreat", ok,
                           "" if ok else f"开度 {opening*1000:.1f} mm / {mr.reason}",
                           {"opening": opening}, self.env.time - t0)

    def go_home(self, duration: float = 2.0) -> SkillResult:
        t0 = self.env.time
        mr = self.ctrl.move_joints(self.ctrl.q_home, duration=duration,
                                   gripper=OPEN_WIDE)
        return SkillResult("go_home", mr.success, mr.reason, {}, self.env.time - t0)
