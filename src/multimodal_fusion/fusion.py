"""多模态时空对齐与跨模态语义融合（可运行的基础版）。

本版本做什么
------------
* **时空对齐**：视觉帧、本体感知、接触读数都取自同一仿真步（同一 `data.time`），
  对齐误差直接计算并写入结果，不做插值假装对齐；
* **特征级融合（规则）**：把同一物体的视觉位姿、与 TCP 的距离、夹爪接触力、
  场景图状态合成一条融合观测，并给出跨模态一致性检查（冲突检测）；
* **接口预留**：`FusionBackend` 协议定义了未来接入学习式跨模态注意力网络的位置，
  当前实现 `RuleBasedFusion` 明确标注为规则版。

明确不做什么
------------
没有训练任何跨模态注意力网络，也没有加载预训练权重；所有融合规则都是可读的
几何/物理判定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np

from perception.types import Detection, GraspContactReport, ProprioState


@dataclass
class FusedObject:
    name: str
    label: str
    color: str
    position_vision: np.ndarray
    distance_to_tcp: float
    contact_force_n: float
    graph_state: str
    fused_state: str            # free / in_hand / placed / uncertain
    confidence: float
    sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "label": self.label, "color": self.color,
            "position_vision_m": np.round(self.position_vision, 4).tolist(),
            "distance_to_tcp_m": round(float(self.distance_to_tcp), 4),
            "contact_force_n": round(float(self.contact_force_n), 4),
            "graph_state": self.graph_state,
            "fused_state": self.fused_state,
            "confidence": round(float(self.confidence), 3),
            "sources": list(self.sources), "notes": list(self.notes),
        }


@dataclass
class FusionSnapshot:
    timestamp: float
    objects: List[FusedObject]
    alignment: Dict[str, float]
    conflicts: List[str]
    backend: str = "rule-based-fusion-v1 (no learned attention)"

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": round(float(self.timestamp), 4),
            "objects": [o.as_dict() for o in self.objects],
            "alignment": {k: round(float(v), 5) for k, v in self.alignment.items()},
            "conflicts": list(self.conflicts),
            "backend": self.backend,
        }


class FusionBackend(Protocol):
    """学习式跨模态融合的接入点（当前未实现）。

    未来实现者需要给出：
        fuse(frame: RGBDFrame, proprio: ProprioState, contact: GraspContactReport)
            -> FusionSnapshot
    例如基于 Transformer 的跨模态注意力：视觉 token + 本体 token + 接触 token。
    """
    name: str

    def fuse(self, detections, proprio, contact, graph) -> FusionSnapshot: ...


class RuleBasedFusion:
    """规则版融合：几何 + 物理阈值，无学习参数。"""

    name = "rule-based-fusion-v1"

    def __init__(self, tcp_proximity: float = 0.12,
                 contact_min: float = 0.3):
        self.tcp_proximity = float(tcp_proximity)
        self.contact_min = float(contact_min)

    def fuse(self, detections: List[Detection], proprio: ProprioState,
             contact: GraspContactReport, graph,
             timestamp: Optional[float] = None,
             vision_timestamp: Optional[float] = None) -> FusionSnapshot:
        ts = float(proprio.timestamp if timestamp is None else timestamp)
        alignment = {
            "vision_vs_proprio_s": float(abs((vision_timestamp if vision_timestamp is not None else ts) - ts)),
            "proprio_vs_contact_s": 0.0,   # 同一 mj_step 读取
        }
        objects: List[FusedObject] = []
        conflicts: List[str] = []

        tcp = np.asarray(proprio.tcp_position, float)
        gripper_open = float(proprio.gripper_opening)
        for det in detections:
            d_tcp = float(np.linalg.norm(det.center - tcp))
            body_name = getattr(det, "body_name", "") or det.geom_name
            force = float(contact.peak_force_n) if (
                contact.object_name and body_name == contact.object_name
            ) else 0.0
            node = graph.nodes.get(body_name) if graph is not None else None
            graph_state = node.state if node is not None else "unknown"

            notes: List[str] = []
            if det.occluded:
                notes.append("视觉上被遮挡，位置置信度下降")
            if d_tcp < self.tcp_proximity:
                notes.append(f"位于夹爪附近 ({d_tcp*1000:.0f} mm)")

            # 融合状态判定（规则）
            if graph_state == "held":
                fused = "in_hand"
            elif force > self.contact_min and d_tcp < self.tcp_proximity:
                fused = "in_hand"
            elif graph_state == "placed":
                fused = "placed"
            elif det.occluded and d_tcp > self.tcp_proximity:
                fused = "uncertain"
            else:
                fused = "free"

            # 冲突检测：图说"持有"但接触传感器无接触
            if graph_state == "held" and force < self.contact_min:
                conflicts.append(
                    f"{body_name}: 场景图标记为 held，但接触力仅 {force:.2f} N")
            # 冲突检测：接触存在但物体离 TCP 很远
            if force > self.contact_min and d_tcp > self.tcp_proximity:
                conflicts.append(
                    f"{body_name}: 有接触力但离 TCP {d_tcp*1000:.0f} mm，可能为环境接触")

            conf = det.confidence
            if det.occluded:
                conf *= 0.8
            if fused == "in_hand" and force > self.contact_min:
                conf = min(1.0, conf + 0.1)

            objects.append(FusedObject(
                name=body_name, label=det.label, color=det.color,
                position_vision=np.asarray(det.center, float),
                distance_to_tcp=d_tcp, contact_force_n=force,
                graph_state=graph_state, fused_state=fused,
                confidence=float(np.clip(conf, 0, 1)),
                sources=["vision:rgbd+seg", "proprio:tcp+gripper",
                         "sim_tactile:contact"] +
                        (["scene_graph"] if node is not None else []),
                notes=notes))

        return FusionSnapshot(timestamp=ts, objects=objects,
                              alignment=alignment, conflicts=conflicts,
                              backend=self.name)
