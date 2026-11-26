"""在 MuJoCo 可视化窗口里跑一次完整的"抓取-放置"（真实物理交互演示）。

与 `run_instruction.py` 的区别
------------------------------
* 打开 MuJoCo 官方被动查看器（mujoco.viewer），实时显示物理状态；
* 每个物理步都会推进仿真，界面按约 30~60 Hz 同步一次；
* 用的仍是 TaskExecutor 的完整链路（感知 -> 场景图 -> 规划 -> 技能 -> 校验），
  不是预置动画。

用法:
    python scripts/viewer_demo.py
    python scripts/viewer_demo.py --instruction "抓起红色方块。"
    python scripts/viewer_demo.py --no-gui      # 无窗口（只跑物理）
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco                                        # noqa: E402

from core.env import SimEnv                          # noqa: E402
from feedback.executor import TaskExecutor           # noqa: E402
from motion_control.controller import PandaController  # noqa: E402
from perception.vision import VisionSensor            # noqa: E402
from scene_graph.graph import SceneGraph             # noqa: E402
from task_planning.language import parse_instruction  # noqa: E402

DEFAULT_INSTRUCTION = "抓起桌面上的红色方块，放到蓝色目标区域。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--sync-every", type=int, default=10,
                    help="每 N 个物理步同步一次画面")
    args = ap.parse_args()

    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    vision = VisionSensor(env, camera="cam_depth", width=480, height=360)
    graph = SceneGraph()
    spec = parse_instruction(args.instruction)

    print("=" * 70)
    print("指令:", args.instruction)
    print("解析:", spec.parser, "| 动作:", spec.action,
          "| 目标:", spec.target.color if spec.target else None,
          "| 目的地:", spec.destination.color if spec.destination else None)

    viewer = None
    if not args.no_gui:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.55, 0.0, 0.55]
            print("可视化窗口已打开（关闭窗口即结束演示）")
        except Exception as exc:
            print("[warn] 无法打开可视化窗口:", exc)
            viewer = None

    counter = {"n": 0}

    def frame_hook(_env):
        if viewer is None:
            return
        counter["n"] += 1
        if counter["n"] % max(1, args.sync_every):
            return
        if not viewer.is_running():
            raise SystemExit("窗口已关闭")
        viewer.sync()

    ctrl.frame_hook = frame_hook

    def on_event(name, payload):
        if name == "progress":
            print(f"  t={env.time:6.2f}s  {payload}")

    executor = TaskExecutor(env, ctrl, graph, vision, verbose=True,
                            on_event=on_event)
    try:
        result = executor.run(spec, max_retries=2)
    finally:
        ctrl.frame_hook = None
        if viewer is not None:
            # 让窗口保留 3 秒，便于观察最终状态
            for _ in range(150):
                env.step(1)
                if not viewer.is_running():
                    break
                viewer.sync()
            viewer.close()
        vision.close()

    print("-" * 70)
    print("任务结果:", "成功" if result.success else "失败",
          "| 时长 %.2f s" % result.duration)
    if not result.success:
        print("失败节点:", result.failed_node, "|", result.reason)
    print("校验:", {k: v for k, v in result.verify.items() if k != "source"})
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
