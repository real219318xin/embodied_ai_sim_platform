"""感知模块的公共数据结构。

设计原则
--------
* 所有数值都来自传感器通道（RGB / Depth / 分割 / 关节传感器 / 接触求解器），
  不直接读取被感知物体的真值坐标；真值只在评估脚本中用于计算误差。
* 每个结果都带 `source` 字段，明确标注数据来源，便于区分
  "视觉估计 / 本体感知 / 仿真接触数据"。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- 视觉
@dataclass
class CameraModel:
    """针孔相机模型（像素为方形，fovy 作用于图像高度）。"""
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray      # 相机 -> 世界 旋转
    t: np.ndarray      # 相机光心（世界坐标）

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    def deproject(self, u, v, depth):
        """像素 + 光轴深度 -> 世界坐标。u/v/depth 支持 numpy 广播。"""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)
        x = (u - self.cx) / self.fx * d
        y = -(v - self.cy) / self.fy * d
        z = -d
        pts_cam = np.stack([x, y, z], axis=-1)
        flat = pts_cam.reshape(-1, 3)
        pts_world = flat @ self.R.T + self.t
        return pts_world.reshape(pts_cam.shape)

    def project(self, pts_world):
        """世界坐标 -> 像素（返回 u, v, depth）。"""
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam = (pts - self.t) @ self.R
        d = -cam[:, 2]
        u = cam[:, 0] / np.maximum(d, 1e-9) * self.fx + self.cx
        v = -cam[:, 1] / np.maximum(d, 1e-9) * self.fy + self.cy
        return u, v, d

    def as_dict(self) -> Dict[str, Any]:
        return {"camera": self.name, "width": self.width, "height": self.height,
                "fx": round(self.fx, 3), "fy": round(self.fy, 3),
                "cx": round(self.cx, 3), "cy": round(self.cy, 3),
                "pos_world": np.round(self.t, 4).tolist()}


@dataclass
class RGBDFrame:
    """一帧 RGB-D + 语义分割数据。"""
    camera: CameraModel
    rgb: np.ndarray                 # (H, W, 3) uint8
    depth: np.ndarray               # (H, W) float32, 光轴 z 深度 (m)
    seg_ids: np.ndarray             # (H, W) int32
    seg_types: np.ndarray           # (H, W) int32 (mjtObj 类型)
    timestamp: float = 0.0

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def mask_of(self, geom_id: int) -> np.ndarray:
        return (self.seg_ids == geom_id) & (self.seg_types == 5)  # mjOBJ_GEOM = 5

    def point_cloud(self, mask: np.ndarray, max_depth: float = 6.0) -> np.ndarray:
        vs, us = np.nonzero(mask)
        if len(us) == 0:
            return np.zeros((0, 3))
        d = self.depth[vs, us].astype(np.float64)
        ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
        if not np.any(ok):
            return np.zeros((0, 3))
        return self.camera.deproject(us[ok], vs[ok], d[ok])

    def as_dict(self) -> Dict[str, Any]:
        return {"camera": self.camera.as_dict(),
                "timestamp": round(float(self.timestamp), 4),
                "rgb_shape": list(self.rgb.shape),
                "depth_valid_ratio": round(
                    float(np.mean(np.isfinite(self.depth) & (self.depth > 0))), 4),
                "depth_range_m": [round(float(np.nanmin(self.depth)), 4),
                                  round(float(np.nanmax(self.depth)), 4)]}


@dataclass
class Detection:
    """一个被检测到的物体（由 RGB-D + 分割得到，规则化几何/颜色分类）。"""
    label: str
    category: str
    color: str
    geom_id: int
    geom_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]      # u0, v0, u1, v1
    n_pixels: int
    center: np.ndarray                   # 世界坐标 3
    size: np.ndarray                     # 世界坐标 3（可见外接盒估计）
    top_z: float
    body_name: str = ""                  # MuJoCo 刚体名（物理查询用）
    occluded: bool = False
    occlusion_ratio: float = 0.0
    graspable: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "vision: RGB-D + semantic segmentation (rule-based)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "category": self.category, "color": self.color,
            "geom_id": int(self.geom_id), "geom_name": self.geom_name,
            "body_name": self.body_name,
            "confidence": round(float(self.confidence), 3),
            "bbox_uv": [int(x) for x in self.bbox],
            "n_pixels": int(self.n_pixels),
            "center_world": np.round(self.center, 4).tolist(),
            "size_world": np.round(self.size, 4).tolist(),
            "top_z": round(float(self.top_z), 4),
            "occluded": bool(self.occluded),
            "occlusion_ratio": round(float(self.occlusion_ratio), 3),
            "graspable": bool(self.graspable),
            "attributes": dict(self.attributes),
            "source": self.source,
        }


# ---------------------------------------------------------------- 本体感知
@dataclass
class ProprioState:
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_efforts: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray
    tcp_quaternion: np.ndarray
    gripper_opening: float              # 两指总开度 (m)
    gripper_finger_positions: np.ndarray
    motion_state: str                   # "moving" / "steady"
    max_joint_speed: float
    source: str = "proprioception: MuJoCo joint sensors (qpos/qvel/actuator_force)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(float(self.timestamp), 4),
            "joint_positions_rad": np.round(self.joint_positions, 5).tolist(),
            "joint_velocities_rad_s": np.round(self.joint_velocities, 5).tolist(),
            "joint_efforts_Nm": np.round(self.joint_efforts, 4).tolist(),
            "tcp_position_m": np.round(self.tcp_position, 5).tolist(),
            "tcp_rotation": np.round(self.tcp_rotation, 5).tolist(),
            "tcp_quaternion_wxyz": np.round(self.tcp_quaternion, 5).tolist(),
            "gripper_opening_m": round(float(self.gripper_opening), 5),
            "gripper_finger_positions_m": np.round(self.gripper_finger_positions, 5).tolist(),
            "motion_state": self.motion_state,
            "max_joint_speed_rad_s": round(float(self.max_joint_speed), 5),
            "source": self.source,
        }


# ---------------------------------------------------------------- 接触/触觉
@dataclass
class ContactSample:
    body_a: str
    body_b: str
    geom_a: str
    geom_b: str
    force_n: float
    force_world: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    penetration: float

    def as_dict(self) -> Dict[str, Any]:
        return {"body_a": self.body_a, "body_b": self.body_b,
                "geom_a": self.geom_a, "geom_b": self.geom_b,
                "force_n": round(float(self.force_n), 4),
                "force_world": np.round(self.force_world, 4).tolist(),
                "position_world": np.round(self.position, 5).tolist(),
                "normal": np.round(self.normal, 4).tolist(),
                "penetration_m": round(float(self.penetration), 6)}


@dataclass
class GraspContactReport:
    """夹爪-物体接触汇总。注意：这是 MuJoCo 约束求解器给出的**仿真接触力**，
    不是真实触觉硬件数据。"""
    in_contact: bool
    n_contacts: int
    peak_force_n: float
    total_force_n: float
    contact_positions: List[np.ndarray]
    left_finger_contact: bool
    right_finger_contact: bool
    object_speed: float
    grasp_stable: bool
    left_force_n: float = 0.0
    right_force_n: float = 0.0
    samples: List[ContactSample] = field(default_factory=list)
    object_name: Optional[str] = None
    source: str = "simulated tactile: MuJoCo contact solver (NOT real hardware)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "in_contact": bool(self.in_contact),
            "n_contacts": int(self.n_contacts),
            "peak_force_n": round(float(self.peak_force_n), 4),
            "total_force_n": round(float(self.total_force_n), 4),
            "left_force_n": round(float(self.left_force_n), 4),
            "right_force_n": round(float(self.right_force_n), 4),
            "contact_positions_world": [np.round(p, 5).tolist()
                                        for p in self.contact_positions],
            "left_finger_contact": bool(self.left_finger_contact),
            "right_finger_contact": bool(self.right_finger_contact),
            "object_speed_m_s": round(float(self.object_speed), 5),
            "grasp_stable": bool(self.grasp_stable),
            "object": self.object_name,
            "samples": [s.as_dict() for s in self.samples],
            "source": self.source,
        }
