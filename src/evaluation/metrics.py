"""实验评价指标（全部由仿真状态与接触数据计算，不含人工填写值）。

指标定义
--------
success            任务是否成功（由执行器校验得出）
duration_s         从开始感知到校验完成的总仿真时间（s）
tcp_path_length_m  TCP 实际走过的路程（对每步 tcp 位置累加，不是直线距离）
peak_grip_force_n  夹爪-目标接触力的逐步最大值（MuJoCo 接触求解器）
collision_events   机器人与非目标物体发生接触的次数（按"接触事件"计，
                   同一物体连续接触只算一次）
retries            抓取链重试次数
xy_error_mm        放置后物体与目标区域中心的水平偏差
z_error_mm         放置后物体底面与目标区域表面的高度偏差
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrialRecord:
    run_id: str
    trial: int
    seed: int
    instruction: str
    success: bool
    duration_s: float = 0.0
    tcp_path_length_m: float = 0.0
    peak_grip_force_n: float = 0.0
    collision_events: int = 0
    collision_bodies: List[str] = field(default_factory=list)
    table_contact_events: int = 0
    retries: int = 0
    xy_error_mm: Optional[float] = None
    z_error_mm: Optional[float] = None
    failed_node: Optional[str] = None
    reason: str = ""
    initial_object_xy: Optional[List[float]] = None
    initial_target_xy: Optional[List[float]] = None
    n_plan_nodes: int = 0

    def as_row(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id, "trial": self.trial, "seed": self.seed,
            "success": int(self.success),
            "duration_s": round(self.duration_s, 3),
            "tcp_path_length_m": round(self.tcp_path_length_m, 4),
            "peak_grip_force_n": round(self.peak_grip_force_n, 3),
            "collision_events": self.collision_events,
            "collision_bodies": ",".join(self.collision_bodies),
            "table_contact_events": self.table_contact_events,
            "retries": self.retries,
            "xy_error_mm": None if self.xy_error_mm is None
            else round(self.xy_error_mm, 2),
            "z_error_mm": None if self.z_error_mm is None
            else round(self.z_error_mm, 2),
            "failed_node": self.failed_node, "reason": self.reason,
            "initial_object_xy": None if self.initial_object_xy is None
            else [round(v, 4) for v in self.initial_object_xy],
            "initial_target_xy": None if self.initial_target_xy is None
            else [round(v, 4) for v in self.initial_target_xy],
            "n_plan_nodes": self.n_plan_nodes,
            "instruction": self.instruction,
        }


def path_length(points: List[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def summarize(records: List[TrialRecord]) -> Dict[str, object]:
    n = len(records)
    if n == 0:
        return {"n_trials": 0}
    succ = [r for r in records if r.success]
    xy = [r.xy_error_mm for r in records if r.xy_error_mm is not None]
    dur = [r.duration_s for r in records]
    path = [r.tcp_path_length_m for r in records]
    force = [r.peak_grip_force_n for r in records]
    coll = [r.collision_events for r in records]
    tbl = [r.table_contact_events for r in records]
    retry = [r.retries for r in records]
    return {
        "n_trials": n,
        "n_success": len(succ),
        "success_rate": round(len(succ) / n, 4),
        "mean_duration_s": round(float(np.mean(dur)), 3),
        "std_duration_s": round(float(np.std(dur)), 3),
        "mean_tcp_path_length_m": round(float(np.mean(path)), 4),
        "max_peak_grip_force_n": round(float(np.max(force)), 3) if force else 0.0,
        "mean_peak_grip_force_n": round(float(np.mean(force)), 3) if force else 0.0,
        "total_collision_events": int(np.sum(coll)),
        "mean_collisions_per_trial": round(float(np.mean(coll)), 3),
        "total_table_contacts": int(np.sum(tbl)),
        "mean_retries": round(float(np.mean(retry)), 3),
        "mean_xy_error_mm": round(float(np.mean(xy)), 2) if xy else None,
        "max_xy_error_mm": round(float(np.max(xy)), 2) if xy else None,
    }
