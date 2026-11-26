"""中文指令驱动的完整流程：解析 -> 感知 -> 场景图 -> 规划 -> 执行 -> 校验。

用法:
    python scripts/run_instruction.py
    python scripts/run_instruction.py --instruction "抓起桌面上的红色方块，放到蓝色目标区域。"
    python scripts/run_instruction.py --instruction "抓起红色方块。" --no-shots

说明
----
* 中文解析为**规则系统**（task_planning/language.py），未接入大模型；
* 视觉来自 MuJoCo RGB-D + 语义分割，接触力来自 MuJoCo 接触求解器；
* 全部动作通过 actuator + mj_step 完成，不修改物体坐标。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv  # noqa: E402
from core.utils import dump_json  # noqa: E402
from feedback.executor import TaskExecutor  # noqa: E402
from motion_control.controller import PandaController  # noqa: E402
from perception.vision import VisionSensor  # noqa: E402
from scene_graph.graph import SceneGraph  # noqa: E402
from task_planning.language import parse_instruction  # noqa: E402

DEFAULT_INSTRUCTION = "抓起桌面上的红色方块，放到蓝色目标区域。"


def make_shotter(env, shots_dir, enabled=True):
    from PIL import Image
    counter = {"n": 0}

    def snap(tag):
        if not enabled:
            return
        counter["n"] += 1
        env.forward()
        for cam in ("cam_rgb", "cam_side"):
            img = env.render(cam, 1280, 720)
            Image.fromarray(img).save(
                os.path.join(shots_dir,
                             f"instr_{counter['n']:02d}_{tag}_{cam}.png"))
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--shots", default=os.path.join(ROOT, "screenshots"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "instruction_last.json"))
    ap.add_argument("--no-shots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.shots, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    snap = make_shotter(env=None, shots_dir=args.shots, enabled=False) \
        if args.no_shots else None

    print("=" * 70)
    print("指令:", args.instruction)
    spec = parse_instruction(args.instruction)
    print("解析器:", spec.parser, "| 大模型:", spec.llm)
    print("动作:", spec.action,
          "| 目标:", spec.target.as_dict() if spec.target else None)
    print("目的地:", spec.destination.as_dict() if spec.destination else None,
          "| 置信度:", spec.confidence)
    if spec.warnings:
        print("警告:", spec.warnings)

    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    vision = VisionSensor(env, camera="cam_depth", width=640, height=480)
    graph = SceneGraph()

    snap = make_shotter(env, args.shots, enabled=not args.no_shots)
    snap("start")

    events = []

    def on_event(name, payload):
        events.append({"event": name, "payload":
                       payload if name != "perception" else
                       {"t": payload.get("t"),
                        "detections": payload.get("detections")}})
        if name == "fusion":
            snap(f"fusion_{payload.get('tag', 'x')}")
        elif name == "progress" and payload.get("stage") in ("transport",):
            snap("transport")
        elif name == "progress" and payload.get("stage") == "verify":
            snap("after_place")

    executor = TaskExecutor(env, ctrl, graph, vision, verbose=True,
                            on_event=on_event)
    result = executor.run(spec, max_retries=2)

    snap("final")

    data = result.as_dict()
    data["instruction"] = args.instruction
    data["events"] = events
    data["scene_graph_final"] = graph.to_dict()
    data["scene_graph_text"] = graph.describe()
    dump_json(data, args.out)

    print("\n" + "=" * 70)
    print("任务结果:", "成功" if result.success else "失败")
    if not result.success:
        print("失败节点:", result.failed_node)
        print("原因:", result.reason)
    print("重试次数:", result.retries)
    print("总时长: %.2f s" % result.duration)
    print("校验:", {k: v for k, v in result.verify.items() if k != "source"})
    print("\n场景图:")
    print(graph.describe())
    print("\n结果文件:", args.out)
    vision.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
