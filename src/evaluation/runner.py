"""多随机种子实验：批量执行任务、记录指标、导出 jsonl/csv/图表。

随机化范围（在**动作开始前**一次性设置的初始条件，不是动作过程中的干预）
--------------------------------------------------------------------------
* 红色方块初始 xy ±`jitter` m，绕 z 轴随机偏航 0~90°；
* 蓝色目标区域中心 xy ±`jitter` m；
* 其余（机器人、障碍物、相机）保持固定，便于横向比较。

每个 trial 都会写出：
    results/experiments/<run_id>/trial_XXX.json     完整过程（规划/场景图/融合/校验）
    results/experiment.jsonl                         一行一个 trial
    results/experiment.csv                           表格
    results/summary.csv                              按种子与总体汇总
    results/plots/*.png                              图表
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import SimEnv                      # noqa: E402
from core.utils import dump_json                 # noqa: E402
from evaluation.metrics import (TrialRecord, path_length,  # noqa: E402
                                summarize)
from feedback.executor import TaskExecutor       # noqa: E402
from motion_control.controller import PandaController  # noqa: E402
from perception.vision import VisionSensor       # noqa: E402
from scene_graph.graph import SceneGraph         # noqa: E402
from task_planning.language import parse_instruction  # noqa: E402

DEFAULT_INSTRUCTION = "抓起桌面上的红色方块，放到蓝色目标区域。"
OBJECT_BODY = "red_cube"
TARGET_BODY = "target_zone"


def make_run_id(tag: str = "") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{tag}" if tag else stamp


def _set_free_body_pose(env: SimEnv, name: str, xy, yaw_deg: float = 0.0):
    """设置物体初始位姿。

    * 有自由关节的物体（红方块）：写入自由关节 qpos（动作开始前，一次性）；
    * 静态区域（蓝目标区）：写入 model.body_pos（同样是初始条件，不是动作干预）。
    两者都只在 trial 开始、任何动作执行之前调用。
    """
    import mujoco
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
    if jid < 0:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body not found: {name}")
        env.model.body_pos[bid][:2] = xy
        mujoco.mj_forward(env.model, env.data)
        return
    qadr = int(env.model.jnt_qposadr[jid])
    dadr = int(env.model.jnt_dofadr[jid])
    half = np.radians(yaw_deg) / 2.0
    env.data.qpos[qadr + 0:qadr + 3] = [xy[0], xy[1], env.data.qpos[qadr + 2]]
    env.data.qpos[qadr + 3:qadr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    env.data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _nominal_xy(env: SimEnv, name: str) -> np.ndarray:
    return env.body_pos(name)[:2].copy()


def run_trial(run_id: str, trial: int, seed: int, instruction: str,
              jitter: float, shots_dir: Optional[str] = None,
              save_images: bool = False) -> tuple:
    """跑一个 trial，返回 (TrialRecord, 完整详情 dict)。"""
    rng = np.random.default_rng(seed * 1000 + trial)
    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()

    obj_xy0 = _nominal_xy(env, OBJECT_BODY)
    tgt_xy0 = _nominal_xy(env, TARGET_BODY)
    obj_xy = obj_xy0 + rng.uniform(-jitter, jitter, size=2)
    tgt_xy = tgt_xy0 + rng.uniform(-jitter, jitter, size=2)
    yaw = float(rng.uniform(0.0, 90.0))
    _set_free_body_pose(env, OBJECT_BODY, obj_xy, yaw)
    _set_free_body_pose(env, TARGET_BODY, tgt_xy, 0.0)
    env.step(120)            # 让物体落到台面并稳定

    vision = VisionSensor(env, camera="cam_depth", width=640, height=480)
    graph = SceneGraph()
    spec = parse_instruction(instruction)

    ctrl.enable_monitor(OBJECT_BODY, sample_every=5)
    ctrl.tcp_traj.clear()

    def on_event(name, payload):
        if name == "progress" and payload.get("stage") == "transport" \
                and save_images and shots_dir:
            from PIL import Image
            env.forward()
            Image.fromarray(env.render("cam_rgb", 960, 540)).save(
                os.path.join(shots_dir, f"exp_{run_id}_t{trial:02d}_transport.png"))

    executor = TaskExecutor(env, ctrl, graph, vision, verbose=False,
                            on_event=on_event)
    result = executor.run(spec, max_retries=2)
    mon = ctrl.disable_monitor()
    vision.close()

    rec = TrialRecord(
        run_id=run_id, trial=trial, seed=seed, instruction=instruction,
        success=bool(result.success), duration_s=float(result.duration),
        tcp_path_length_m=path_length(ctrl.tcp_traj),
        peak_grip_force_n=float(mon.get("peak_grip_force", 0.0)),
        collision_events=int(mon.get("collision_events", 0)),
        collision_bodies=sorted(mon.get("collision_bodies", [])),
        table_contact_events=int(mon.get("table_contacts", 0)),
        retries=int(result.retries),
        failed_node=result.failed_node, reason=result.reason,
        initial_object_xy=obj_xy.tolist(),
        initial_target_xy=tgt_xy.tolist(),
        n_plan_nodes=len((result.plan or {}).get("nodes", [])),
    )
    verify = result.verify or {}
    if verify.get("xy_error_m") is not None:
        rec.xy_error_mm = float(verify["xy_error_m"]) * 1000.0
    if verify.get("z_error_m") is not None:
        rec.z_error_mm = float(verify["z_error_m"]) * 1000.0

    detail = result.as_dict()
    detail["instruction"] = instruction
    detail["scene_graph_text"] = graph.describe()
    detail["randomization"] = {
        "object_xy": [round(float(v), 4) for v in obj_xy],
        "target_xy": [round(float(v), 4) for v in tgt_xy],
        "object_yaw_deg": round(yaw, 2),
        "jitter_m": jitter, "seed": seed,
        "note": "初始条件在动作开始前设置；动作过程全部由 actuator 驱动",
    }
    detail["metrics"] = rec.as_row()
    return rec, detail


def make_plots(records: List[TrialRecord], plots_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    os.makedirs(plots_dir, exist_ok=True)
    out = []
    trials = [r.trial for r in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(r.trial) for r in records], [r.duration_s for r in records],
           color=["#3a7" if r.success else "#c44" for r in records])
    ax.set_xlabel("trial"); ax.set_ylabel("duration (s)")
    ax.set_title("任务时长（绿=成功，红=失败）")
    p = os.path.join(plots_dir, "duration_by_trial.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(t) for t in trials],
           [r.xy_error_mm if r.xy_error_mm is not None else 0.0 for r in records],
           color="#37a")
    ax.axhline(70.0, color="#c44", ls="--", label="成功判据上限 70 mm")
    ax.set_xlabel("trial"); ax.set_ylabel("final xy error (mm)")
    ax.set_title("放置水平误差"); ax.legend()
    p = os.path.join(plots_dir, "placement_error_by_trial.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(t) for t in trials], [r.tcp_path_length_m for r in records],
            "o-", color="#a63")
    ax.set_xlabel("trial"); ax.set_ylabel("TCP path length (m)")
    ax.set_title("末端执行器路程")
    p = os.path.join(plots_dir, "tcp_path_length.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(t) for t in trials], [r.peak_grip_force_n for r in records],
           color="#777")
    ax.set_xlabel("trial"); ax.set_ylabel("peak grip force (N)")
    ax.set_title("夹爪接触力峰值（MuJoCo 接触求解器）")
    p = os.path.join(plots_dir, "peak_grip_force.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)

    seeds = sorted({r.seed for r in records})
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = [np.mean([r.success for r in records if r.seed == s]) for s in seeds]
    ax.bar([str(s) for s in seeds], rates, color="#2a7")
    ax.set_ylim(0, 1.05); ax.set_xlabel("seed"); ax.set_ylabel("success rate")
    ax.set_title("各随机种子成功率")
    p = os.path.join(plots_dir, "success_rate_by_seed.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    return out
