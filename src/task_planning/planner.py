"""神经—符号混合任务规划（符号层：可运行的基础版）。

定位
----
本模块实现的是**符号规划层**：把中文指令解析出的任务规格，结合场景图，
展开成带前置条件的技能序列，并在执行前检查物理可行性。
"神经"部分目前只有一个明确定义的接入点（`NeuralPlannerBackend`），
没有训练模型、没有 MCTS、没有 RAG。

规划流程
--------
    指令 -> TaskSpec（规则解析）
         -> resolve_target   （在场景图中按颜色/类别检索）
         -> resolve_destination
         -> 展开技能序列（pick / place / verify）
         -> 前置条件检查（存在性、可抓取性、可达性、夹爪状态）

失败时返回 `failure_node` 与 `recovery` 建议，由执行器决定
局部重试 / 重新规划 / 终止。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scene_graph.graph import SceneGraph, SceneNode
from task_planning.language import TaskSpec

# 中文类别词 -> 场景图类别
CATEGORY_MAP = {
    "cube": "cube",
    "box": "box",
    "target_area": "region",
    "region": "region",
}


@dataclass
class PlanNode:
    name: str
    skill: str
    params: Dict[str, object] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    status: str = "pending"
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"node": self.name, "skill": self.skill,
                "params": {k: (v if not hasattr(v, "tolist") else v.tolist())
                           for k, v in self.params.items()},
                "preconditions": list(self.preconditions),
                "status": self.status, "detail": self.detail}


@dataclass
class Plan:
    spec: TaskSpec
    nodes: List[PlanNode] = field(default_factory=list)
    ok: bool = True
    failure_node: Optional[str] = None
    reason: str = ""
    recovery: List[str] = field(default_factory=list)
    target: Optional[SceneNode] = None
    destination: Optional[SceneNode] = None
    planner: str = "symbolic-planner-v1 (no neural/MCTS/RAG)"

    def as_dict(self) -> Dict[str, object]:
        return {
            "planner": self.planner,
            "ok": bool(self.ok),
            "failure_node": self.failure_node,
            "reason": self.reason,
            "recovery": list(self.recovery),
            "target": self.target.as_dict() if self.target else None,
            "destination": self.destination.as_dict() if self.destination else None,
            "nodes": [n.as_dict() for n in self.nodes],
        }


class NeuralPlannerBackend:
    """神经规划接入点（当前未实现）。

    未来可在这一层接入：
      * 学习式子目标生成（语言 -> 技能序列）
      * 场景图上的图神经网络 / MCTS 搜索
      * RAG 检索历史成功轨迹
    目前所有调用都会抛 NotImplementedError，避免"看起来实现"。
    """

    name = "neural-planner-placeholder"

    def plan(self, spec, graph):  # pragma: no cover - 明确未实现
        raise NotImplementedError(
            "神经规划后端尚未实现；当前仅提供符号规划器")


class TaskPlanner:
    def __init__(self, reach_tolerance: float = 0.004):
        self.reach_tolerance = float(reach_tolerance)

    # ---------------- 检索 ----------------
    def resolve_target(self, spec: TaskSpec, graph: SceneGraph):
        if spec.target is None:
            return None, "指令中未识别到目标物体"
        cat = CATEGORY_MAP.get(spec.target.category or "", None)
        node = graph.nearest(color=spec.target.color, category=cat)
        if node is None:
            avail = graph.describe() or "（场景图为空）"
            return None, (f"场景图中找不到 color={spec.target.color}, "
                          f"category={spec.target.category} 的物体；当前场景:\n{avail}")
        return node, ""

    def resolve_destination(self, spec: TaskSpec, graph: SceneGraph):
        if spec.destination is None:
            return None, ""
        cat = CATEGORY_MAP.get(spec.destination.category or "target_area", "region")
        node = graph.nearest(color=spec.destination.color, category=cat)
        if node is None:
            return None, (f"场景图中找不到 color={spec.destination.color}, "
                          f"category={spec.destination.category} 的目标区域")
        return node, ""

    # ---------------- 规划 ----------------
    def plan(self, spec: TaskSpec, graph: SceneGraph,
             gripper_opening: float = 0.0) -> Plan:
        plan = Plan(spec=spec)

        if spec.action == "unknown":
            plan.ok = False
            plan.failure_node = "parse_instruction"
            plan.reason = "无法解析指令动作"
            plan.recovery = ["请求用户重新表述指令"]
            return plan
        if spec.action in ("push", "open", "close"):
            plan.ok = False
            plan.failure_node = "unsupported_action"
            plan.reason = f"动作 '{spec.action}' 已识别但当前技能库未实现"
            plan.recovery = ["改用抓取-放置类指令"]
            return plan

        target, why = self.resolve_target(spec, graph)
        if target is None:
            plan.ok = False
            plan.failure_node = "resolve_target"
            plan.reason = why
            plan.recovery = ["重新感知（可能需要移动相机或清理遮挡）",
                             "检查物体是否被移出场景"]
            return plan
        plan.target = target

        if not target.graspable:
            plan.ok = False
            plan.failure_node = "check_graspable"
            plan.reason = f"{target.label} 被判定为不可抓取（类别 {target.category}）"
            plan.recovery = ["选择其它可抓取物体"]
            return plan

        if not graph.is_reachable(target):
            plan.ok = False
            plan.failure_node = "check_reachable"
            plan.reason = (f"{target.label} 距基座 "
                           f"{float(__import__('numpy').linalg.norm(target.position)):.3f} m "
                           f"超出可达范围 {graph.robot_reach:.2f} m 或当前被遮挡")
            plan.recovery = ["移动底座 / 调整物体位置后重新规划"]
            return plan

        destination = None
        if spec.needs_place:
            destination, why = self.resolve_destination(spec, graph)
            if destination is None:
                plan.ok = False
                plan.failure_node = "resolve_destination"
                plan.reason = why
                plan.recovery = ["重新感知目标区域", "改用其它放置区域"]
                return plan
            plan.destination = destination

        # ---- 展开技能序列 ----
        tp = target.position
        plan.nodes.append(PlanNode(
            "open_gripper", "open_gripper",
            {"width": 0.038},
            ["gripper_available"]))
        plan.nodes.append(PlanNode(
            "move_above_target", "move_above",
            {"pos": tp.tolist(), "dz": 0.14, "allowed": target.name},
            ["target_exists", "target_reachable"]))
        plan.nodes.append(PlanNode(
            "descend_to_grasp", "descend_to",
            {"pos": [float(tp[0]), float(tp[1]), float(tp[2] + 0.004)],
             "allowed": target.name},
            ["gripper_opened"]))
        plan.nodes.append(PlanNode(
            "close_gripper", "close_gripper",
            {"object": target.name},
            ["at_grasp_pose"]))
        plan.nodes.append(PlanNode(
            "lift_object", "lift_by",
            {"object": target.name, "dz": 0.12},
            ["grasp_contact_confirmed"]))
        if destination is not None:
            dp = destination.position
            plan.nodes.append(PlanNode(
                "transport_to_region", "transport_xy",
                {"xy": [float(dp[0]), float(dp[1])], "object": target.name},
                ["object_held", "safe_height_clear"]))
            plan.nodes.append(PlanNode(
                "descend_to_place", "place_at",
                {"xy": [float(dp[0]), float(dp[1])],
                 "height": float(dp[2] + target.size[2] / 2 + 0.004),
                 "object": target.name},
                ["above_destination"]))
            plan.nodes.append(PlanNode(
                "release", "release_and_retreat",
                {"object": target.name},
                ["at_place_height"]))
        else:
            plan.nodes.append(PlanNode(
                "retreat", "release_and_retreat",
                {"object": target.name},
                ["object_held"]))
        plan.nodes.append(PlanNode(
            "verify_task", "verify",
            {"object": target.name,
             "destination": destination.name if destination else None},
            ["task_executed"]))
        return plan

    # ---------------- 前置条件 ----------------
    def check_preconditions(self, plan: Plan, graph: SceneGraph,
                            gripper_opening: float) -> List[str]:
        unmet: List[str] = []
        target = plan.target
        for node in plan.nodes:
            for cond in node.preconditions:
                if cond == "target_exists" and (target is None
                                                or target.name not in graph.nodes):
                    unmet.append(f"{node.name}:{cond}")
                elif cond == "target_reachable" and target is not None \
                        and not graph.is_reachable(target):
                    unmet.append(f"{node.name}:{cond}")
                elif cond == "gripper_available" and gripper_opening < 0.0:
                    unmet.append(f"{node.name}:{cond}")
        return unmet
