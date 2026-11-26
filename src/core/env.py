"""通用仿真环境封装：加载场景、步进、渲染、接触查询。

所有物理量都直接来自 MuJoCo 的 MjModel / MjData，不做任何伪造。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")

ROBOT_BODY_ROOT = "link0"
ROBOT_BODIES_PREFIX = ("link", "hand", "left_finger", "right_finger")

# hand 坐标系原点到指腹中心（TCP）的 z 偏移，单位 m。
# 这是由官方 Panda 模型几何量得的常量，运动学与本体感知共用。
TCP_OFFSET = 0.103


def body_id(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"body not found: {name}")
    return i


def geom_id(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise KeyError(f"geom not found: {name}")
    return i


@dataclass
class ContactInfo:
    geom1: int
    geom2: int
    body1: str
    body2: str
    dist: float
    position: np.ndarray
    normal: np.ndarray
    # 6 维接触力（世界坐标系力矩/力），由 MuJoCo 约束求解得到
    force: np.ndarray

    @property
    def force_magnitude(self) -> float:
        """接触力大小 (N)。force = [力矩(3), 世界系力(3)]。"""
        return float(np.linalg.norm(self.force[3:]))


class SimEnv:
    """一个 MuJoCo 场景实例。"""

    def __init__(self, scene_path: str = DEFAULT_SCENE, keyframe: str = "scene_home"):
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.keyframe_name = keyframe
        self._key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if self._key_id < 0:
            raise ValueError(f"keyframe not found: {keyframe}")

    # ---------------- 基础状态 ----------------
    def reset(self, settle_steps: int = 200) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ---------------- 位姿查询 ----------------
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[body_id(self.model, name)].copy()

    def body_mat(self, name: str) -> np.ndarray:
        return self.data.xmat[body_id(self.model, name)].reshape(3, 3).copy()

    def body_quat(self, name: str) -> np.ndarray:
        return self.data.xquat[body_id(self.model, name)].copy()

    # ---------------- 渲染 ----------------
    def render(self, camera: str = "cam_rgb", width: int = 640, height: int = 480,
               depth: bool = False, segmentation: bool = False):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise KeyError(camera)
        renderer.update_scene(self.data, camera=cam_id)
        if depth:
            renderer.enable_depth_rendering()
        if segmentation:
            renderer.enable_segmentation_rendering()
        img = renderer.render()
        renderer.close()
        return img

    # ---------------- 接触 ----------------
    def contacts(self) -> List[ContactInfo]:
        out = []
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            raw = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, raw)
            # mj_contactForce 返回接触坐标系下的 [法向力, 切向1, 切向2, 力矩...]
            f_local = raw[:3]
            f_world = c.frame.reshape(3, 3) @ f_local
            out.append(ContactInfo(
                geom1=int(c.geom1), geom2=int(c.geom2),
                body1=self._body_name(int(m.geom_bodyid[c.geom1])),
                body2=self._body_name(int(m.geom_bodyid[c.geom2])),
                dist=float(c.dist),
                position=c.pos.copy(),
                normal=c.frame[:3].copy(),
                force=np.concatenate([raw[3:], f_world]),
            ))
        return out

    def _body_name(self, bid: int) -> str:
        n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        return n or f"body{bid}"

    def contacts_between(self, body_a: str, body_b: str) -> List[ContactInfo]:
        return [c for c in self.contacts()
                if {c.body1, c.body2} == {body_a, body_b}]

    def geom_name(self, gid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"

    # ---------------- 物体 ----------------
    def object_pos(self, name: str) -> np.ndarray:
        return self.body_pos(name)

    def is_robot_body(self, name: str) -> bool:
        return name.startswith(ROBOT_BODIES_PREFIX)

    def robot_object_contacts(self) -> List[ContactInfo]:
        return [c for c in self.contacts()
                if self.is_robot_body(c.body1) != self.is_robot_body(c.body2)
                and not (c.body1 in ("world", "bench") and c.body2 in ("world", "bench"))]
