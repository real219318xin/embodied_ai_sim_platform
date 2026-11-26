"""录制真实仿真的抓取-放置过程为 GIF（可选 MP4）。

录制内容
--------
* 每一帧都来自 MuJoCo 离屏渲染的**实际仿真画面**，不是后处理动画；
* 帧在 `mj_step` 循环里采样，因此画面与物理严格同步；
* 画面左下角标注仿真时间（s），便于核对节奏。

用法:
    python scripts/record_gif.py                        # 默认指令 + cam_rgb
    python scripts/record_gif.py --camera cam_side
    python scripts/record_gif.py --every 20 --width 560 --height 350
    python scripts/record_gif.py --mp4                  # 同时导出 mp4
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco                                        # noqa: E402
from PIL import Image, ImageDraw                     # noqa: E402

from core.env import SimEnv                          # noqa: E402
from feedback.executor import TaskExecutor           # noqa: E402
from motion_control.controller import PandaController  # noqa: E402
from perception.vision import VisionSensor            # noqa: E402
from scene_graph.graph import SceneGraph             # noqa: E402
from task_planning.language import parse_instruction  # noqa: E402

DEFAULT_INSTRUCTION = "抓起桌面上的红色方块，放到蓝色目标区域。"


class FrameGrabber:
    """按固定步数间隔从仿真里抓帧，并把仿真时间烧进画面。"""

    def __init__(self, env, camera="cam_rgb", width=640, height=400, every=25):
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.every = max(1, int(every))
        self.step_count = 0
        self.frames = []
        self.cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                        camera)
        if self.cam_id < 0:
            raise KeyError(f"camera not found: {camera}")
        self.renderer = mujoco.Renderer(env.model, height=self.height,
                                        width=self.width)

    def __call__(self, env):
        self.step_count += 1
        if self.step_count % self.every:
            return
        self.renderer.update_scene(env.data, camera=self.cam_id)
        img = self.renderer.render().copy()
        self.frames.append(self._caption(img, env.time))

    def _caption(self, img, t):
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        text = f"MuJoCo physics  t={t:5.2f}s"
        draw.text((8, self.height - 18), text, fill=(0, 0, 0))
        draw.text((7, self.height - 19), text, fill=(255, 255, 255))
        return im

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--camera", default="cam_rgb")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=25, help="每 N 个物理步抓一帧")
    ap.add_argument("--gif", default=os.path.join(ROOT, "videos",
                                                 "pick_place_demo.gif"))
    ap.add_argument("--gif-width", type=int, default=512)
    ap.add_argument("--colors", type=int, default=96, help="GIF 调色板颜色数")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--mp4-path", default=os.path.join(ROOT, "videos",
                                                      "pick_place_demo.mp4"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    print("=" * 70)
    print("指令:", args.instruction)
    print("相机:", args.camera, f"{args.width}x{args.height}",
          "| 每", args.every, "步抓帧")

    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    vision = VisionSensor(env, camera="cam_depth", width=640, height=480)
    graph = SceneGraph()
    spec = parse_instruction(args.instruction)

    grabber = FrameGrabber(env, camera=args.camera, width=args.width,
                           height=args.height, every=args.every)
    ctrl.frame_hook = grabber

    events = []

    def on_event(name, payload):
        if name == "progress":
            events.append((round(env.time, 2), payload))
            print(f"  t={env.time:6.2f}s  {payload}")

    executor = TaskExecutor(env, ctrl, graph, vision, verbose=True,
                            on_event=on_event)
    result = executor.run(spec, max_retries=2)
    ctrl.frame_hook = None

    # 结尾停留 1 s，便于看清最终状态
    for _ in range(20):
        grabber.frames.append(grabber.frames[-1])

    grabber.close()
    vision.close()

    frames = grabber.frames
    print(f"\n抓帧数: {len(frames)}  (采样间隔 {args.every} 步 = "
          f"{args.every * env.model.opt.timestep:.3f} s)")

    # GIF：缩放到目标宽度，限制调色板以控制体积
    w = args.gif_width
    h = int(round(frames[0].height * w / frames[0].width))
    small = [f.convert("RGB").resize((w, h), Image.LANCZOS) for f in frames]
    pal = [f.quantize(colors=args.colors, method=Image.MEDIANCUT,
                      dither=Image.FLOYDSTEINBERG)
           for f in small]
    ms = int(round(args.every * env.model.opt.timestep * 1000))
    pal[0].save(args.gif, save_all=True, append_images=pal[1:], duration=ms,
                loop=0, optimize=True, disposal=2)
    size_kb = os.path.getsize(args.gif) / 1024
    print(f"GIF: {args.gif}  ({w}x{h}, {len(pal)} 帧, {ms} ms/帧, {size_kb:.0f} KB)")

    if args.mp4:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(args.mp4_path, fourcc, 1000.0 / ms,
                                 (w, h))
            for f in small:
                vw.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
            vw.release()
            print("MP4:", args.mp4_path,
                  f"({os.path.getsize(args.mp4_path)/1024:.0f} KB)")
        except Exception as exc:      # 编码器不可用时不影响 GIF
            print("[warn] mp4 导出失败:", exc)

    print("-" * 70)
    print("任务结果:", "成功" if result.success else "失败",
          "| 时长 %.2f s" % result.duration,
          "| 节点 %d" % len(result.records))
    if not result.success:
        print("失败节点:", result.failed_node, "|", result.reason)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
