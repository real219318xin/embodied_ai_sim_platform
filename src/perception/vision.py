"""视觉感知：RGB / Depth / 语义分割 → 物体检测（规则化颜色 + 几何分类）。

真实链路
--------
1. MuJoCo 离屏渲染给出同一时刻的 RGB、深度（沿光轴 z，单位 m）、几何分割；
2. 用相机内参把分割区域内的深度像素反投影成世界坐标点云；
3. 颜色：mask 内像素的 HSV 均值 → 最近颜色词（红/蓝/黄/绿/橙/紫/灰白黑）；
4. 类别：点云外接盒的尺寸比例 → cube / box / region / surface（规则判定）；
5. 位置/尺寸：点云外接盒；直立物体用"顶面 + 水平最短边"估计高度与中心，
   以补偿看不见的底面。

明确标注
--------
* 没有使用任何深度学习模型或预训练权重；
* 颜色与形状都来自当前帧像素与点云的规则判定（见 `confidence` 说明）；
* 物体真值坐标不参与检测，只在 `scripts/test_perception.py` 中用于计算误差。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np

from perception.types import CameraModel, Detection, RGBDFrame

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_PATH = os.path.join(ROOT, "config", "objects.json")

MJOBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

# 颜色词（HSV, OpenCV 约定 H∈[0,180], S/V∈[0,255]）
COLOR_RULES = [
    ("red",    [(0, 10), (170, 180)]),
    ("orange", [(11, 22)]),
    ("yellow", [(23, 35)]),
    ("green",  [(36, 85)]),
    ("blue",   [(86, 130)]),
    ("purple", [(131, 169)]),
]
ACHROMATIC = ("black", "gray", "white")


def load_registry() -> Dict[str, dict]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def hsv_to_color(h: float, s: float, v: float) -> str:
    """单像素/均值 HSV -> 颜色词（规则表）。"""
    if v < 45:
        return "black"
    if s < 45:
        return "white" if v > 170 else "gray"
    for name, ranges in COLOR_RULES:
        for lo, hi in ranges:
            if lo <= h <= hi:
                return name
    return "unknown"


class VisionSensor:
    def __init__(self, env, camera: str = "cam_depth", width: int = 640,
                 height: int = 480, registry: Optional[Dict[str, dict]] = None):
        self.env = env
        self.model = env.model
        self.camera_name = camera
        self.width = int(width)
        self.height = int(height)
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.cam_id < 0:
            raise KeyError(f"camera not found: {camera}")
        self.registry = registry if registry is not None else load_registry()
        self._renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width)
        self.robot_geom_ids = self._collect_robot_geoms()
        self.dynamic_body_ids = self._collect_free_bodies()
        self.region_geom_ids = self._collect_regions()

    # ---------------- 结构信息 ----------------
    def _collect_robot_geoms(self) -> set:
        ids = set()
        m = self.model
        for g in range(m.ngeom):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                      int(m.geom_bodyid[g])) or ""
            if bname.startswith(("link", "hand", "finger", "left_finger",
                                 "right_finger")):
                ids.add(g)
        return ids

    def _collect_free_bodies(self) -> set:
        """具有自由关节的刚体（物理上可被移动）。

        这是**动力学先验**，不属于视觉：它把"可以被搬动的物体"和
        "固定在场景里的家具"分开，避免把桌腿/台面误检成可抓方块。
        """
        m = self.model
        ids = set()
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                ids.add(int(m.jnt_bodyid[j]))
        return ids

    def _collect_regions(self) -> set:
        """语义表中登记为放置区域的静态几何（白名单）。

        只把 placement_region 纳入检测；支撑面（桌面/地面）属于场景环境，
        不作为"被感知的物体"，避免它们出现在物体列表里。
        """
        out = set()
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            role = self.registry.get(name or "", {}).get("role", "")
            if role == "placement_region":
                out.add(int(g))
        return out

    def camera_model(self) -> CameraModel:
        d = self.env.data
        fovy = np.radians(float(self.model.cam_fovy[self.cam_id]))
        fy = 0.5 * self.height / np.tan(fovy / 2)
        fx = fy                      # 方形像素
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        return CameraModel(name=self.camera_name, width=self.width,
                           height=self.height, fx=fx, fy=fy, cx=cx, cy=cy,
                           R=d.cam_xmat[self.cam_id].reshape(3, 3).copy(),
                           t=d.cam_xpos[self.cam_id].copy())

    # ---------------- 采集 ----------------
    def capture(self, timestamp: Optional[float] = None) -> RGBDFrame:
        self.env.forward()
        self._renderer.update_scene(self.env.data, camera=self.cam_id)
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().astype(np.float32).copy()
        self._renderer.disable_depth_rendering()

        self._renderer.enable_segmentation_rendering()
        seg = self._renderer.render()
        self._renderer.disable_segmentation_rendering()

        return RGBDFrame(
            camera=self.camera_model(), rgb=rgb, depth=depth,
            seg_ids=seg[..., 0].copy(), seg_types=seg[..., 1].copy(),
            timestamp=float(self.env.time if timestamp is None else timestamp))

    def close(self):
        try:
            self._renderer.close()
        except Exception:
            pass

    # ---------------- 检测 ----------------
    def detect(self, frame: RGBDFrame, min_pixels: int = 60,
               max_objects: int = 8) -> List[Detection]:
        ids = np.unique(frame.seg_ids[frame.seg_types == MJOBJ_GEOM])
        robot_mask = np.isin(frame.seg_ids, list(self.robot_geom_ids)) & \
            (frame.seg_types == MJOBJ_GEOM)

        dets: List[Detection] = []
        for gid in ids:
            gid = int(gid)
            if gid <= 0 or gid in self.robot_geom_ids:
                continue
            body = int(self.model.geom_bodyid[gid])
            if body not in self.dynamic_body_ids and gid not in self.region_geom_ids:
                continue        # 静态家具几何不参与物体检测
            mask = frame.mask_of(gid)
            n_pix = int(mask.sum())
            if n_pix < min_pixels:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
            pts = frame.point_cloud(mask)
            if len(pts) < 20:
                continue
            color, mean_rgb = self._estimate_color(frame.rgb, mask)
            category, size, center, top_z = self._estimate_geometry(pts)
            if category is None:
                continue
            vs, us = np.nonzero(mask)
            bbox = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
            occ_ratio = self._occlusion_ratio(robot_mask, bbox)
            conf = self._confidence(n_pix, mask, size, color)
            label = self._label(color, category)
            reg = self.registry.get(name, {})
            dets.append(Detection(
                label=label, category=category, color=color,
                geom_id=gid, geom_name=name, confidence=conf,
                bbox=bbox, n_pixels=n_pix, center=center, size=size,
                top_z=top_z, body_name=body_name, occluded=occ_ratio > 0.05,
                occlusion_ratio=occ_ratio,
                graspable=bool(reg.get("graspable", category in ("cube", "box"))),
                attributes={"mean_rgb": [int(x) for x in mean_rgb],
                            "role": reg.get("role", "object"),
                            "cn_name": reg.get("cn_name", "")},
            ))

        dets.sort(key=lambda d: -d.n_pixels)
        return dets[:max_objects]

    # ---------------- 内部估计 ----------------
    @staticmethod
    def _estimate_color(rgb: np.ndarray, mask: np.ndarray):
        pix = rgb[mask]
        if len(pix) == 0:
            return "unknown", (0, 0, 0)
        mean = pix.mean(axis=0)
        hsv = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_RGB2HSV)[0, 0]
        # 光照会让饱和度下降，先看像素级别投票的中位色相以提高鲁棒性
        sub = pix[::max(1, len(pix) // 400)].reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)
        h_med = float(np.median(hsv_all[:, 0]))
        s_med = float(np.median(hsv_all[:, 1]))
        v_med = float(np.median(hsv_all[:, 2]))
        if s_med < 45 or v_med < 45:
            color = hsv_to_color(float(hsv[0]), float(hsv[1]), float(hsv[2]))
        else:
            color = hsv_to_color(h_med, s_med, v_med)
        return color, mean

    @staticmethod
    def _estimate_geometry(pts: np.ndarray):
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        ext = hi - lo
        sx, sy, sz_vis = float(ext[0]), float(ext[1]), float(ext[2])
        max_xy = max(sx, sy)
        ratio = sz_vis / max(max_xy, 1e-6)
        if max_xy > 0.30:
            category = "surface"
            height = max(sz_vis, 0.02)
        elif sz_vis < 0.015 and max_xy >= 0.06:
            category = "region"
            height = max(sz_vis, 0.004)
        elif 0.02 <= max_xy <= 0.12 and min(sx, sy) / max(max_xy, 1e-6) >= 0.65 \
                and 0.6 <= ratio <= 1.3:
            # 三向尺寸接近且可见高度与水平尺寸相当 -> 立方体
            category = "cube"
            height = max(sz_vis, min(sx, sy)) if sz_vis >= 0.6 * max_xy \
                else min(sx, sy)
        elif 0.02 <= max_xy <= 0.25:
            category = "box"
            height = sz_vis if sz_vis >= 0.02 else min(sx, sy)
        else:
            category = "irregular"
            height = max(sz_vis, 0.01)

        center = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0,
                           hi[2] - height / 2.0])
        size = np.array([sx, sy, height])
        # 顶面高度用 95 分位而不是最大值：深度噪声会让个别像素跳高，
        # 对薄区域（目标区标记）尤其明显。
        top_z = float(np.percentile(pts[:, 2], 95))
        return category, size, center, top_z

    @staticmethod
    def _occlusion_ratio(robot_mask: np.ndarray, bbox) -> float:
        u0, v0, u1, v1 = bbox
        box = robot_mask[v0:v1 + 1, u0:u1 + 1]
        area = max(1, box.size)
        return float(box.sum()) / area

    @staticmethod
    def _confidence(n_pixels: int, mask: np.ndarray, size: np.ndarray,
                    color: str) -> float:
        """规则化置信度（非概率模型）：像素充足度 × 轮廓紧致度 × 颜色明确度。"""
        fill = float(mask.sum()) / max(1.0, float(
            (np.ptp(np.nonzero(mask)[1]) + 1) * (np.ptp(np.nonzero(mask)[0]) + 1)))
        pix_score = min(1.0, n_pixels / 300.0)
        color_score = 1.0 if color not in ("unknown", "gray") else 0.6
        return float(np.clip(0.55 * pix_score + 0.25 * fill + 0.20 * color_score,
                             0.0, 1.0))

    @staticmethod
    def _label(color: str, category: str) -> str:
        cn = {"cube": "cube", "box": "box", "region": "target_area",
              "surface": "surface", "irregular": "object"}
        return f"{color}_{cn.get(category, 'object')}"

    # ---------------- 可视化 ----------------
    @staticmethod
    def annotate(frame: RGBDFrame, dets: List[Detection]) -> np.ndarray:
        img = frame.rgb.copy()
        palette = {"red": (255, 60, 60), "blue": (60, 120, 255),
                   "yellow": (255, 220, 40), "green": (60, 220, 120),
                   "orange": (255, 150, 40), "gray": (200, 200, 200),
                   "white": (255, 255, 255), "black": (140, 140, 140)}
        for d in dets:
            u0, v0, u1, v1 = d.bbox
            c = palette.get(d.color, (0, 255, 255))
            cv2.rectangle(img, (u0, v0), (u1, v1), c, 2)
            text = f"{d.label} {d.confidence:.2f}"
            cv2.putText(img, text, (u0, max(12, v0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            cv2.circle(img, ((u0 + u1) // 2, (v0 + v1) // 2), 3, c, -1)
        return img

    @staticmethod
    def colorize_segmentation(frame: RGBDFrame) -> np.ndarray:
        ids = frame.seg_ids.astype(np.int64)
        rng = np.random.default_rng(2024)
        lut = rng.integers(40, 255, size=(int(ids.max()) + 2, 3), dtype=np.uint8)
        out = lut[np.clip(ids, 0, ids.max())]
        out[ids == 0] = 0
        return out
