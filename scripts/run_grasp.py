"""阶段五运行脚本：执行一次完整的真实物理抓取-放置，并保存过程截图。

用法:
    python scripts/run_grasp.py

所有动作均由 PandaController 通过 actuator + mj_step 执行，
不修改物体坐标、不 teleport、不焊接、不播放预录动画。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv
from core.utils import dump_json
from motion_control.controller import PandaController
from task_planning.pick_place import PickPlaceTask

# 这些步骤完成后各截一张图（按步骤名过滤，避免刷屏）
SNAP_STEPS = {
    "open_gripper", "pregrasp", "descend_to_grasp", "close_gripper",
    "lift", "transport", "descend_to_place", "release", "check_placement",
}


def make_snapper(env, shots_dir):
    """返回 on_step 回调：在关键步骤后保存 cam_rgb / cam_side 两张图。"""
    from PIL import Image
    index = {"n": 0}

    def snap(step_name, ok):
        if step_name not in SNAP_STEPS:
            return
        index["n"] += 1
        tag = f"{index['n']:02d}_{step_name}"
        env.forward()
        for cam in ("cam_rgb", "cam_side"):
            img = env.render(cam, 1280, 720)
            Image.fromarray(img).save(
                os.path.join(shots_dir, f"grasp_{tag}_{cam}.png"))

    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=os.path.join(ROOT, "screenshots"))
    args = ap.parse_args()
    os.makedirs(args.shots, exist_ok=True)

    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()

    from PIL import Image
    env.forward()
    Image.fromarray(env.render("cam_rgb", 1280, 720)).save(
        os.path.join(args.shots, "grasp_00_start_cam_rgb.png"))

    task = PickPlaceTask(env, ctrl, on_step=make_snapper(env, args.shots))
    res = task.run()

    env.forward()
    Image.fromarray(env.render("cam_rgb", 1280, 720)).save(
        os.path.join(args.shots, "grasp_99_final_cam_rgb.png"))

    out_path = os.path.join(ROOT, "results", "grasp_last.json")
    dump_json(res, out_path)

    print("\n==== 任务摘要 ====")
    print("success        :", res["success"])
    print("reason         :", res["reason"] or "-")
    print("duration       : %.2f s" % res["duration"])
    print("peak contact F : %.3f N" % res["peak_contact_force"])
    print("collisions     :", res["collision_count"])
    print("final object   :", np.round(res["final_object_pos"], 4))
    print("trace saved    :", out_path)
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
