"""任务驱动动态场景图。

节点 = 场景中的物体/区域，带语义属性与位姿；
边   = 物体之间的空间关系（on / above / left_of / near …）与交互关系（held_by）。

动态更新
--------
* `update_from_detections()`：用视觉检测结果刷新节点位姿与可见性
  （经过抓取、搬运、放置后重新检测，位姿随之变化）；
* `set_state()`：由执行反馈写入 held / placed / lost 等交互状态；
* 每次更新记录 `(timestamp, node, old, new)` 变更历史，便于回溯。

坐标系约定：机器人基座坐标系，+x 朝前、+y 朝左、+z 朝上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REL_NEAR = 0.15          # m，near 关系阈值
REL_ON_TOL = 0.02        # m，on 关系的竖直容差
REL_LEFT_TOL = 0.03      # m，左右关系阈值
REL_ABOVE_TOL = 0.03     # m，上下关系阈值


@dataclass
class SceneNode:
    node_id: str
    name: str                       # 稳定标识（几何名/物体名）
    label: str                      # e.g. red_cube
    category: str                   # cube / box / region / surface
    color: str
    position: np.ndarray
    size: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    top_z: float = 0.0              # 可见顶面高度（区域=表面，方块=顶面）
    graspable: bool = False
    occluded: bool = False
    confidence: float = 0.0
    state: str = "free"             # free / held / placed / lost
    attributes: Dict[str, object] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    updates: int = 0
    pose_source: str = "vision"

    @property
    def z_min(self) -> float:
        return float(self.position[2] - self.size[2] / 2.0)

    @property
    def z_max(self) -> float:
        return float(self.position[2] + self.size[2] / 2.0)

    @property
    def xy(self) -> np.ndarray:
        return self.position[:2].copy()

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id, "name": self.name, "label": self.label,
            "category": self.category, "color": self.color,
            "position": np.round(self.position, 4).tolist(),
            "size": np.round(self.size, 4).tolist(),
            "top_z": round(float(self.top_z), 4),
            "quat": np.round(self.quat, 4).tolist(),
            "graspable": bool(self.graspable), "occluded": bool(self.occluded),
            "confidence": round(float(self.confidence), 3),
            "state": self.state,
            "attributes": dict(self.attributes),
            "first_seen": round(float(self.first_seen), 3),
            "last_seen": round(float(self.last_seen), 3),
            "updates": int(self.updates),
            "pose_source": self.pose_source,
        }


@dataclass
class SceneEdge:
    source: str
    target: str
    relation: str
    value: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        d = {"source": self.source, "target": self.target,
             "relation": self.relation}
        if self.value is not None:
            d["value"] = round(float(self.value), 4)
        return d


class SceneGraph:
    def __init__(self, robot_reach: float = 0.80):
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: List[SceneEdge] = []
        self.history: List[Dict[str, object]] = []
        self.robot_reach = float(robot_reach)

    # ---------------- 更新 ----------------
    def update_from_detections(self, detections: Iterable, timestamp: float) -> List[Dict]:
        changes = []
        for det in detections:
            # 场景图节点以**刚体名**为 key：后续抓取/放置/校验都需要用它查询物理状态
            key = getattr(det, "body_name", "") or det.geom_name or f"geom{det.geom_id}"
            new_pos = np.asarray(det.center, float)
            new_size = np.asarray(det.size, float)
            node = self.nodes.get(key)
            if node is None:
                node = SceneNode(
                    node_id=f"obj:{key}", name=key, label=det.label,
                    category=det.category, color=det.color,
                    position=new_pos, size=new_size,
                    top_z=float(det.top_z),
                    graspable=det.graspable, occluded=det.occluded,
                    confidence=det.confidence,
                    attributes={**dict(det.attributes), "geom_name": det.geom_name},
                    first_seen=timestamp, last_seen=timestamp, updates=1)
                self.nodes[key] = node
                changes.append({"t": timestamp, "node": key, "change": "appear",
                                "position": new_pos.tolist()})
            else:
                dist = float(np.linalg.norm(new_pos - node.position))
                old = node.position.copy()
                node.position = new_pos
                node.size = new_size
                node.top_z = float(det.top_z)
                node.label = det.label
                node.category = det.category
                node.color = det.color
                node.graspable = det.graspable
                node.occluded = det.occluded
                node.confidence = det.confidence
                node.attributes.update(det.attributes)
                node.attributes["geom_name"] = det.geom_name
                node.last_seen = timestamp
                node.updates += 1
                if dist > 0.002:
                    changes.append({"t": timestamp, "node": key, "change": "move",
                                    "from": old.tolist(), "to": new_pos.tolist(),
                                    "distance": round(dist, 5)})
        self.history.extend(changes)
        self.rebuild_relations()
        return changes

    def mark_missing(self, seen_keys: Iterable[str], timestamp: float):
        """这一帧没检测到的节点：标记为 lost（而不是删除，保留历史）。"""
        seen = set(seen_keys)
        for key, node in self.nodes.items():
            if key not in seen and node.state in ("free", "placed"):
                node.occluded = True
                node.state = "lost"
                self.history.append({"t": timestamp, "node": key,
                                     "change": "lost"})

    def set_state(self, name: str, state: str, timestamp: Optional[float] = None):
        node = self.nodes.get(name)
        if node is None:
            return
        node.state = state
        node.last_seen = float(timestamp if timestamp is not None else node.last_seen)
        self.history.append({"t": node.last_seen, "node": name, "change": "state",
                             "state": state})

    def update_pose(self, name: str, position, quat=None,
                    source: str = "vision", timestamp: Optional[float] = None):
        node = self.nodes.get(name)
        if node is None:
            return
        old = node.position.copy()
        node.position = np.asarray(position, float)
        if quat is not None:
            node.quat = np.asarray(quat, float)
        node.pose_source = source
        if timestamp is not None:
            node.last_seen = float(timestamp)
        self.history.append({"t": node.last_seen, "node": name, "change": "pose",
                             "from": old.tolist(),
                             "to": node.position.tolist(), "source": source})
        self.rebuild_relations()

    # ---------------- 关系 ----------------
    def rebuild_relations(self):
        edges: List[SceneEdge] = []
        items = list(self.nodes.values())
        for a in items:
            for b in items:
                if a is b:
                    continue
                d = float(np.linalg.norm(a.position - b.position))
                if d < REL_NEAR:
                    edges.append(SceneEdge(a.name, b.name, "near", d))
                dz = float(a.position[2] - b.position[2])
                if abs(dz) > REL_ABOVE_TOL:
                    edges.append(SceneEdge(a.name, b.name,
                                           "above" if dz > 0 else "below", abs(dz)))
                dy = float(a.position[1] - b.position[1])
                if abs(dy) > REL_LEFT_TOL:
                    edges.append(SceneEdge(a.name, b.name,
                                           "left_of" if dy > 0 else "right_of", abs(dy)))
                if b.category in ("surface", "region", "cube", "box") and \
                        abs(a.z_min - b.z_max) < REL_ON_TOL and self._xy_overlap(a, b):
                    edges.append(SceneEdge(a.name, b.name, "on", abs(a.z_min - b.z_max)))
        self.edges = edges

    @staticmethod
    def _xy_overlap(a: SceneNode, b: SceneNode) -> bool:
        return (abs(a.position[0] - b.position[0]) <= (a.size[0] + b.size[0]) / 2 + 0.02
                and abs(a.position[1] - b.position[1]) <= (a.size[1] + b.size[1]) / 2 + 0.02)

    def relations_of(self, name: str) -> List[SceneEdge]:
        return [e for e in self.edges if e.source == name or e.target == name]

    # ---------------- 查询 ----------------
    def find(self, color: Optional[str] = None, category: Optional[str] = None,
             label: Optional[str] = None, graspable: Optional[bool] = None,
             state: Optional[str] = None) -> List[SceneNode]:
        out = []
        for n in self.nodes.values():
            if color and n.color != color:
                continue
            if category and n.category != category:
                continue
            if label and n.label != label:
                continue
            if graspable is not None and n.graspable != graspable:
                continue
            if state and n.state != state:
                continue
            out.append(n)
        return out

    def nearest(self, color: Optional[str] = None,
                category: Optional[str] = None,
                graspable: Optional[bool] = None) -> Optional[SceneNode]:
        cands = self.find(color=color, category=category, graspable=graspable)
        if not cands:
            return None
        return min(cands, key=lambda n: float(np.linalg.norm(n.xy)))

    def is_reachable(self, node: SceneNode) -> bool:
        """工作空间粗判：基座到物体中心的水平距离 + 高度。"""
        r = float(np.linalg.norm(node.position))
        return r <= self.robot_reach and not node.occluded

    # ---------------- 输出 ----------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": [n.as_dict() for n in self.nodes.values()],
            "edges": [e.as_dict() for e in self.edges],
            "history": self.history[-200:],
            "n_nodes": len(self.nodes), "n_edges": len(self.edges),
        }

    def describe(self) -> str:
        lines = []
        for n in sorted(self.nodes.values(), key=lambda x: x.label):
            state = n.state
            occ = "遮挡" if n.occluded else "可见"
            lines.append(f"{n.label}({n.color}) pos=({n.position[0]:.3f},"
                         f"{n.position[1]:.3f},{n.position[2]:.3f}) "
                         f"size=({n.size[0]*1000:.0f}x{n.size[1]*1000:.0f}x"
                         f"{n.size[2]*1000:.0f}mm) {state}/{occ}")
        return "\n".join(lines)
