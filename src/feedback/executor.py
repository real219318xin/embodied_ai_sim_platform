"""执行反馈与失败恢复。

职责
----
* 按规划好的技能序列驱动仿真，逐步记录结果；
* 失败时按策略恢复：
    1) 局部重试：抓取链（接近→下降→闭合→抬升）失败时，换一个夹爪朝向重试；
    2) 重新感知：任何失败后先刷新场景图，再决定是否重试；
    3) 返回失败节点：超过重试上限则返回 failure_node + 原因，不伪造成功。
* 在每个关键节点输出多模态融合快照（视觉 + 本体 + 接触），作为执行反馈。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_fusion.fusion import RuleBasedFusion  # noqa: E402
from perception.contact import ContactSensor  # noqa: E402
from perception.proprioception import ProprioceptionSensor  # noqa: E402
from perception.vision import VisionSensor  # noqa: E402
from scene_graph.graph import SceneGraph  # noqa: E402
from task_planning.language import TaskSpec  # noqa: E402
from task_planning.planner import Plan, TaskPlanner  # noqa: E402
from task_planning.skills import SkillLibrary, SkillResult  # noqa: E402

GRASP_CHAIN = ("move_above_target", "descend_to_grasp", "close_gripper", "lift_object")


@dataclass
class NodeRecord:
    node: str
    skill: str
    ok: bool
    detail: str = ""
    attempts: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)
    t: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {"node": self.node, "skill": self.skill, "ok": bool(self.ok),
                "detail": self.detail, "attempts": int(self.attempts),
                "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()},
                "t": round(float(self.t), 3)}


@dataclass
class ExecutionResult:
    success: bool
    reason: str = ""
    failed_node: Optional[str] = None
    retries: int = 0
    duration: float = 0.0
    records: List[NodeRecord] = field(default_factory=list)
    recovery_log: List[str] = field(default_factory=list)
    fusion: List[Dict[str, object]] = field(default_factory=list)
    plan: Optional[Dict[str, object]] = None
    spec: Optional[Dict[str, object]] = None
    scene_graph: Optional[Dict[str, object]] = None
    perception: Optional[Dict[str, object]] = None
    verify: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "success": bool(self.success), "reason": self.reason,
            "failed_node": self.failed_node, "retries": int(self.retries),
            "duration": round(float(self.duration), 3),
            "records": [r.as_dict() for r in self.records],
            "recovery_log": list(self.recovery_log),
            "fusion_snapshots": list(self.fusion),
            "plan": self.plan, "spec": self.spec,
            "scene_graph": self.scene_graph,
            "perception": self.perception,
            "verify": dict(self.verify),
        }


class TaskExecutor:
    def __init__(self, env, controller, graph: SceneGraph, vision: VisionSensor,
                 verbose: bool = True,
                 on_event: Optional[Callable[[str, Dict], None]] = None):
        self.env = env
        self.ctrl = controller
        self.graph = graph
        self.vision = vision
        self.proprio = ProprioceptionSensor(env)
        self.contact = ContactSensor(env)
        self.skills = SkillLibrary(env, controller, verbose=verbose)
        self.fusion = RuleBasedFusion()
        self.planner = TaskPlanner()
        self.verbose = verbose
        self.on_event = on_event
        self._fusion_log: List[Dict[str, object]] = []
        self._perception_log: List[Dict[str, object]] = []

    # ---------------- 工具 ----------------
    def _emit(self, name: str, payload: Dict):
        if self.on_event:
            try:
                self.on_event(name, payload)
            except Exception:
                pass

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def perceive(self) -> list:
        """采集一帧并刷新场景图（视觉为准）。"""
        frame = self.vision.capture()
        dets = self.vision.detect(frame)
        changes = self.graph.update_from_detections(dets, frame.timestamp)
        keys = [getattr(d, "body_name", "") or d.geom_name for d in dets]
        self.graph.mark_missing(keys, frame.timestamp)
        self._perception_log.append({
            "t": round(float(frame.timestamp), 3),
            "frame": frame.as_dict(),
            "detections": [d.as_dict() for d in dets],
            "graph_changes": changes,
        })
        self._emit("perception", {"t": frame.timestamp,
                                  "detections": [d.as_dict() for d in dets]})
        return dets

    def fused_snapshot(self, tag: str, vision_ts: float,
                       object_name: Optional[str] = None) -> None:
        """写入一条多模态融合快照（视觉 + 本体 + 仿真接触）。"""
        self.env.forward()
        prop = self.proprio.read()
        contact = self.contact.grasp_report(object_name) if object_name \
            else self.contact.grasp_report("")
        dets = self.vision.detect(self.vision.capture())
        snap = self.fusion.fuse(dets, prop, contact, self.graph,
                               timestamp=prop.timestamp,
                               vision_timestamp=vision_ts)
        d = snap.as_dict()
        d["tag"] = tag
        self._fusion_log.append(d)
        self._emit("fusion", d)

    # ---------------- 执行 ----------------
    def run(self, spec: TaskSpec, max_retries: int = 2) -> ExecutionResult:
        t0 = self.env.time
        res = ExecutionResult(success=False, spec=spec.as_dict())

        dets = self.perceive()
        plan = self.planner.plan(spec, self.graph,
                                 gripper_opening=self.proprio.read().gripper_opening)
        res.plan = plan.as_dict()
        if not plan.ok:
            res.reason = plan.reason
            res.failed_node = plan.failure_node
            res.duration = self.env.time - t0
            res.scene_graph = self.graph.to_dict()
            res.perception = {"frames": self._perception_log}
            return res

        self._log(f"  规划完成：{len(plan.nodes)} 个技能节点，目标 "
                  f"{plan.target.label} @ {np.round(plan.target.position,3)}")

        unmet = self.planner.check_preconditions(
            plan, self.graph, self.proprio.read().gripper_opening)
        if unmet:
            res.reason = f"前置条件不满足: {unmet}"
            res.failed_node = unmet[0].split(":")[0]
            res.duration = self.env.time - t0
            res.scene_graph = self.graph.to_dict()
            return res

        target = plan.target
        destination = plan.destination
        attempts = 0
        yaw = 0.0
        picked = False

        # ---- 抓取链（可局部重试） ----
        while attempts <= max_retries:
            attempts += 1
            self._emit("progress", {"stage": "pick", "attempt": attempts,
                                    "yaw": yaw})
            chain = self._run_grasp_chain(target.position, target.name, yaw, res)
            if chain.ok:
                picked = True
                break
            res.retries += 1
            res.recovery_log.append(
                f"抓取链在 '{chain.node}' 失败（{chain.detail}），第 {attempts} 次尝试")
            if attempts <= max_retries:
                # 重新感知后再试：换 90° 夹爪朝向，绕开单侧碰撞
                dets = self.perceive()
                node = self.graph.nodes.get(target.name)
                if node is None:
                    res.recovered_by = "reperceive_failed"
                    res.reason = "重试前重新感知失败：目标不在视野内"
                    res.failed_node = "perceive"
                    res.duration = self.env.time - t0
                    res.scene_graph = self.graph.to_dict()
                    res.perception = {"frames": self._perception_log}
                    return res
                target = node
                yaw = 90.0 if yaw == 0.0 else 0.0
                self.skills.go_home(duration=1.2)

        if not picked:
            res.reason = f"抓取失败：连续 {attempts} 次未能夹起目标"
            res.failed_node = "lift_object"
            res.duration = self.env.time - t0
            res.scene_graph = self.graph.to_dict()
            res.perception = {"frames": self._perception_log}
            return res

        self.graph.set_state(target.name, "held", self.env.time)
        self.fused_snapshot("after_grasp", self.env.time, target.name)

        # ---- 搬运 / 放置 ----
        if destination is not None:
            r = self.skills.transport_xy([destination.position[0],
                                          destination.position[1]],
                                         object_name=target.name)
            res.records.append(NodeRecord("transport_to_region", "transport_xy",
                                          r.ok, r.reason, 1, r.metrics, self.env.time))
            if not r.ok:
                return self._fail(res, "transport_to_region", r.reason, t0)
            self._emit("progress", {"stage": "transport", "ok": True})

            place_h = float(destination.position[2] + target.size[2] / 2 + 0.004)
            r = self.skills.place_at([destination.position[0],
                                      destination.position[1]], place_h,
                                     object_name=target.name)
            res.records.append(NodeRecord("descend_to_place", "place_at",
                                          r.ok, r.reason, 1, r.metrics, self.env.time))
            if not r.ok:
                return self._fail(res, "descend_to_place", r.reason, t0)

            r = self.skills.release_and_retreat(object_name=target.name)
            res.records.append(NodeRecord("release", "release_and_retreat",
                                          r.ok, r.reason, 1, r.metrics, self.env.time))
            if not r.ok:
                return self._fail(res, "release", r.reason, t0)
        else:
            # 只抓取任务：保持夹持状态，不松手（松手会让物体落回台面，
            # 那样"抓起"这一目标就无法被真实校验）
            res.records.append(NodeRecord("hold_object", "hold", True,
                                          "保持夹持，未释放", 1, {}, self.env.time))

        # ---- 校验 + 场景图更新 ----
        self.ctrl.hold(200)          # 静止 0.4 s 再判定速度，避免刚停下的残余速度误判
        verify = self._verify(target, destination)
        res.verify = verify
        self._emit("progress", {"stage": "verify", "ok": verify["ok"]})
        if destination is not None:
            self.skills.go_home(duration=1.5)
            self.perceive()
            self.graph.set_state(target.name,
                                 "placed" if verification_ok(verify) else "free",
                                 self.env.time)
            self.fused_snapshot("after_place", self.env.time, target.name)
        else:
            self.graph.set_state(target.name,
                                 "held" if verification_ok(verify) else "free",
                                 self.env.time)
            self.fused_snapshot("after_grasp_hold", self.env.time, target.name)

        res.success = bool(verify["ok"])
        res.reason = "" if res.success else verify["reason"]
        res.failed_node = None if res.success else "verify_task"
        res.duration = self.env.time - t0
        res.scene_graph = self.graph.to_dict()
        res.perception = {"frames": self._perception_log}
        return res

    # ---------------- 抓取链 ----------------
    def _run_grasp_chain(self, pos, name: str, yaw: float,
                         res: ExecutionResult) -> NodeRecord:
        self._emit("progress", {"stage": "open_gripper"})
        r = self.skills.open_gripper()
        self._log(f"  [{'OK ' if r.ok else 'FAIL'}] open_gripper "
                  f"{(':: ' + r.reason) if r.reason else ''}")
        if not r.ok:
            rec = NodeRecord("open_gripper", "open_gripper", False, r.reason, 1,
                             r.metrics, self.env.time)
            res.records.append(rec)
            return rec

        r = self.skills.move_above(pos, yaw=yaw, allowed=name)
        self._log(f"  [{'OK ' if r.ok else 'FAIL'}] move_above "
                  f"{(':: ' + r.reason) if r.reason else ''}")
        res.records.append(NodeRecord("move_above_target", "move_above", r.ok,
                                      r.reason, 1, r.metrics, self.env.time))
        if not r.ok:
            return res.records[-1]

        grasp_pos = [pos[0], pos[1], pos[2] + 0.004]
        r = self.skills.descend_to(grasp_pos, yaw=yaw, allowed=name)
        self._log(f"  [{'OK ' if r.ok else 'FAIL'}] descend_to_grasp "
                  f"{(':: ' + r.reason) if r.reason else ''}")
        res.records.append(NodeRecord("descend_to_grasp", "descend_to", r.ok,
                                      r.reason, 1, r.metrics, self.env.time))
        if not r.ok:
            return res.records[-1]

        r = self.skills.close_gripper(name)
        self._log(f"  [{'OK ' if r.ok else 'FAIL'}] close_gripper "
                  f"peak={r.metrics.get('peak_force', 0):.2f} N")
        res.records.append(NodeRecord("close_gripper", "close_gripper", r.ok,
                                      r.reason, 1, r.metrics, self.env.time))
        if not r.ok:
            return res.records[-1]

        r = self.skills.lift_by(name)
        self._log(f"  [{'OK ' if r.ok else 'FAIL'}] lift_object "
                  f"dz={r.metrics.get('object_dz', 0)*1000:.0f} mm")
        res.records.append(NodeRecord("lift_object", "lift_by", r.ok, r.reason, 1,
                                      r.metrics, self.env.time))
        return res.records[-1]

    # ---------------- 校验 ----------------
    def _verify(self, target, destination) -> Dict[str, object]:
        obj_pos = self.env.body_pos(target.name)
        speed = self.contact.object_speed(target.name)
        out: Dict[str, object] = {
            "object": target.name,
            "final_position": np.round(obj_pos, 4).tolist(),
            "object_speed": float(speed),
            "source": "sim physics state (MuJoCo qpos)",
        }
        if destination is None:
            rep = self.contact.grasp_report(target.name)
            ok = (float(obj_pos[2]) > 0.5 and speed < 0.05
                  and rep.left_force_n > 0.3 and rep.right_force_n > 0.3)
            out.update({"ok": bool(ok), "held": bool(ok),
                        "left_force_n": round(rep.left_force_n, 4),
                        "right_force_n": round(rep.right_force_n, 4),
                        "reason": "" if ok else
                        "物体未被抬升 / 仍在运动 / 夹持力不足"})
            return out
        d_xy = float(np.linalg.norm(obj_pos[:2] - destination.position[:2]))
        # 参考面取目标区域"可见表面"（top_z）；区域是无碰撞厚 2.4 mm 的标记，
        # 所以方块实际落点会比该平面略低一点，这一点会在 z_error 里如实体现。
        surf_z = float(destination.top_z) if getattr(destination, "top_z", 0.0) \
            else float(destination.position[2])
        z_err = abs(float(obj_pos[2] - (surf_z + target.size[2] / 2)))
        ok = d_xy < 0.07 and z_err < 0.02 and speed < 0.02
        out.update({"ok": bool(ok), "xy_error_m": d_xy, "z_error_m": z_err,
                    "zone_surface_z": surf_z,
                    "destination": destination.name,
                    "reason": "" if ok else
                    f"放置偏差 xy={d_xy*1000:.1f} mm, z={z_err*1000:.1f} mm"})
        return out

    def _fail(self, res: ExecutionResult, node: str, reason: str,
              t0: float) -> ExecutionResult:
        res.success = False
        res.failed_node = node
        res.reason = reason
        res.duration = self.env.time - t0
        res.recovery_log.append(f"节点 {node} 失败：{reason}（未重试，避免碰撞风险）")
        res.scene_graph = self.graph.to_dict()
        res.perception = {"frames": self._perception_log}
        return res


def verification_ok(v: Dict[str, object]) -> bool:
    return bool(v.get("ok", False))
