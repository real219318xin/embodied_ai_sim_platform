"""抓取-放置技能（阶段五）：全部通过 actuator + mj_step 完成。

流程（每一步都写进 trace，失败保留原因）
------------------------------------------
1  回到初始姿态              -> 关节伺服
2  读取目标物体位置          -> MjData 真实位姿
3  张开夹爪
4  移动到上方预抓取位（避障高度）
5  垂直下降到抓取位
6  闭合夹爪（接触力触发）
7  抬升并校验物体是否随夹爪上升
8  抬升到安全高度后平移至目标区上方
9  下降放置
10 松开夹爪并撤离
11 校验放置结果（位置 / 稳定性）
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import SimEnv  # noqa: E402
from motion_control.controller import PandaController, OPEN_WIDE, OPEN_GRASP  # noqa: E402
from motion_control.kinematics import down_grasp_rotation  # noqa: E402

CUBE_HALF = 0.025
SAFE_Z = 0.62          # 高于障碍物顶端(0.52)的搬运高度
PREGRASP_DZ = 0.14
GRASP_DZ = 0.004       # 抓取点相对方块中心抬高，避免手指下缘插入台面


@dataclass
class StepRecord:
    name: str
    ok: bool
    detail: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def as_dict(self):
        return {"step": self.name, "ok": self.ok, "detail": self.detail,
                "t_start": round(self.t_start, 4), "t_end": round(self.t_end, 4),
                "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()}}


class PickPlaceTask:
    def __init__(self, env: SimEnv, controller: PandaController,
                 object_name: str = "red_cube", target_name: str = "target_zone",
                 obstacle_name: str = "yellow_obstacle", verbose: bool = True,
                 on_step=None):
        self.env = env
        self.ctrl = controller
        self.object = object_name
        self.target = target_name
        self.obstacle = obstacle_name
        self.verbose = verbose
        self.on_step = on_step      # 回调: (step_name, ok) -> None，用于截图/记录
        self.trace: List[StepRecord] = []
        self.peak_contact_force = 0.0
        self.contact_steps = 0
        self.collision_count = 0
        self.retries = 0

    # ---------------- 工具 ----------------
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _step(self, name: str, ok: bool, detail: str = "", t0: float = 0.0,
              metrics=None):
        rec = StepRecord(name, ok, detail, t0, self.env.time, metrics or {})
        self.trace.append(rec)
        self._log(f"  [{'OK ' if ok else 'FAIL'}] {name}"
                  f"{(' :: ' + detail) if detail else ''}")
        if self.on_step is not None:
            try:
                self.on_step(name, ok)
            except Exception as exc:      # 回调失败不应中断物理任务
                self._log(f"  [warn] on_step({name}) failed: {exc!r}")
        return ok

    def _object_pose(self):
        return self.env.body_pos(self.object), self.env.body_quat(self.object)

    def _grasp_rot(self):
        # 让手指闭合方向对准方块的某一对面：这里用世界 y 方向开合
        return down_grasp_rotation(0.0)

    # ---------------- 主流程 ----------------
    def run(self) -> Dict:
        t_task = self.env.time
        ctrl = self.ctrl
        R = self._grasp_rot()

        # 1. 回到初始姿态
        t0 = self.env.time
        mr = ctrl.move_joints(ctrl.q_home, duration=1.5)
        self._step("home_pose", mr.success, mr.reason, t0)

        # 2. 读取物体位置
        t0 = self.env.time
        obj_p, obj_q = self._object_pose()
        self._log(f"  物体 {self.object} 位置 = {np.round(obj_p, 4)}")
        self._step("locate_object", True, f"pos={np.round(obj_p,4).tolist()}", t0,
                   {"x": obj_p[0], "y": obj_p[1], "z": obj_p[2]})

        # 3. 张开夹爪
        t0 = self.env.time
        opening = ctrl.open_gripper(OPEN_WIDE, steps=200)
        self._step("open_gripper", opening > 0.02, f"opening={opening*1000:.1f} mm", t0)

        # 4. 预抓取位（物体正上方）
        t0 = self.env.time
        pre = np.array([obj_p[0], obj_p[1], obj_p[2] + PREGRASP_DZ])
        mr = ctrl.move_to_pose(pre, R, max_duration=5.0, gripper=OPEN_WIDE)
        tcp, _ = ctrl.tcp()
        self._step("pregrasp", mr.success,
                   mr.reason or f"tcp_err={np.linalg.norm(tcp-pre)*1000:.1f} mm", t0,
                   {"pos_err": float(np.linalg.norm(tcp - pre))})
        if not mr.success:
            return self._finish(False, "pregrasp_failed", t_task)

        # 5. 垂直下降到抓取位
        t0 = self.env.time
        grasp_p = np.array([obj_p[0], obj_p[1], obj_p[2] + GRASP_DZ])
        mr = ctrl.move_linear(grasp_p, R, speed=0.05, gripper=OPEN_WIDE,
                              allow_body=self.object, tol_pos=0.0015)
        tcp, _ = ctrl.tcp()
        self._step("descend_to_grasp", mr.success,
                   mr.reason or f"tcp_err={np.linalg.norm(tcp-grasp_p)*1000:.1f} mm", t0,
                   {"pos_err": float(np.linalg.norm(tcp - grasp_p))})
        if not mr.success:
            return self._finish(False, "descend_failed", t_task)

        # 6. 闭合夹爪（真实接触触发）
        t0 = self.env.time
        ginfo = ctrl.close_gripper(steps=500, stop_on_contact=True,
                                   object_body=self.object, force_threshold=0.3)
        self.peak_contact_force = max(self.peak_contact_force, ginfo["peak_force"])
        self.contact_steps = int(ginfo["steps"])
        self._step("close_gripper", ginfo["contact"],
                   f"contact={ginfo['contact']} opening={ginfo['opening']*1000:.1f} mm "
                   f"peak_force={ginfo['peak_force']:.2f} N "
                   f"both_fingers={ginfo.get('both_fingers')}", t0,
                   {"peak_force": ginfo["peak_force"]})
        if not ginfo["contact"]:
            return self._finish(False, "no_grasp_contact", t_task)

        # 7. 抬升并验证
        t0 = self.env.time
        lift_p = np.array([obj_p[0], obj_p[1], obj_p[2] + 0.12])
        mr = ctrl.move_linear(lift_p, R, speed=0.04, gripper=0.0,
                              allow_body=self.object)
        p_after = self.env.body_pos(self.object)
        lifted = (p_after[2] - obj_p[2]) > 0.05
        fingers = ctrl.finger_opening()
        self._step("lift", mr.success and lifted,
                   f"dz={p_after[2]-obj_p[2]:.3f} m, finger_opening={fingers*1000:.1f} mm",
                   t0, {"dz": float(p_after[2] - obj_p[2])})
        if not lifted:
            return self._finish(False, "object_not_lifted", t_task)

        # 8. 搬运到目标区上方（先到安全高度，避开黄色障碍物）
        t0 = self.env.time
        tgt_p = self.env.body_pos(self.target)
        above = np.array([tgt_p[0], tgt_p[1], SAFE_Z + 0.05])
        mr1 = ctrl.move_linear(np.array([obj_p[0], obj_p[1], SAFE_Z + 0.05]), R,
                               speed=0.06, gripper=0.0, allow_body=self.object)
        mr2 = ctrl.move_linear(above, R, speed=0.08, gripper=0.0,
                               allow_body=self.object) if mr1.success else mr1
        travel = float(np.linalg.norm(self.env.body_pos(self.object) - p_after))
        self._step("transport", mr1.success and mr2.success,
                   (mr1.reason or mr2.reason or
                    f"object_travel={travel*1000:.0f} mm"), t0,
                   {"travel": travel})
        if not (mr1.success and mr2.success):
            return self._finish(False, "transport_failed", t_task)

        # 9. 下降放置
        t0 = self.env.time
        place_p = np.array([tgt_p[0], tgt_p[1], tgt_p[2] + CUBE_HALF + 0.004])
        mr = ctrl.move_linear(place_p, R, speed=0.04, gripper=0.0,
                              allow_body=self.object)
        self._step("descend_to_place", mr.success, mr.reason, t0)
        if not mr.success:
            return self._finish(False, "place_descend_failed", t_task)

        # 10. 松开夹爪并撤离
        t0 = self.env.time
        op = ctrl.open_gripper(OPEN_WIDE, steps=250)
        mr = ctrl.move_linear(np.array([place_p[0], place_p[1], place_p[2] + 0.10]),
                              R, speed=0.06, gripper=OPEN_WIDE,
                              allow_body=self.object)
        ctrl.hold(150)   # 让物体稳定落地
        self._step("release", op > 0.02, f"opening={op*1000:.1f} mm", t0)

        # 11. 校验放置
        t0 = self.env.time
        fin = self.env.body_pos(self.object)
        vel = self.env.data.qvel[self._object_dofadr():self._object_dofadr() + 3]
        dist_xy = float(np.linalg.norm(fin[:2] - tgt_p[:2]))
        z_err = abs(float(fin[2] - (tgt_p[2] + CUBE_HALF)))
        stable = float(np.linalg.norm(vel)) < 0.01
        ok_place = dist_xy < 0.07 and z_err < 0.015 and stable
        self._step("check_placement", ok_place,
                   f"xy_dist={dist_xy*1000:.1f} mm, z_err={z_err*1000:.1f} mm, "
                   f"speed={np.linalg.norm(vel):.4f} m/s", t0,
                   {"xy_dist": dist_xy, "z_err": z_err})

        return self._finish(ok_place, "" if ok_place else "placement_out_of_tolerance",
                            t_task)

    def _object_dofadr(self) -> int:
        import mujoco
        jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{self.object}_free")
        if jid < 0:
            raise KeyError(f"joint {self.object}_free not found")
        return int(self.env.model.jnt_dofadr[jid])

    def _finish(self, success: bool, reason: str, t_task: float) -> Dict:
        # 统计整段任务中的非预期碰撞与接触峰值
        allowed = {self.object}
        coll = self.ctrl.collision_report(allowed_objects=allowed)
        self.collision_count = coll["count"]
        for c in self.env.robot_object_contacts():
            self.peak_contact_force = max(self.peak_contact_force, c.force_magnitude)
        res = {
            "success": bool(success),
            "reason": reason,
            "duration": float(self.env.time - t_task),
            "steps": [s.as_dict() for s in self.trace],
            "peak_contact_force": float(self.peak_contact_force),
            "contact_steps": int(self.contact_steps),
            "collisions": coll["collisions"][:10],
            "collision_count": int(coll["count"]),
            "final_object_pos": self.env.body_pos(self.object).tolist(),
        }
        self._log(f"  任务结果: {'成功' if success else '失败'} {reason}")
        return res
