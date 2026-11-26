"""渲染场景截图（RGB + Depth + 分割），用于视觉检查与论文配图。

用法:
    python scripts/render_scene.py [--out screenshots/scene] [--steps 500]
"""
import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")


def save_depth_png(depth, path):
    from PIL import Image
    d = np.asarray(depth, dtype=np.float64).copy()
    d[~np.isfinite(d)] = 0.0
    lo, hi = float(d.min()), float(np.percentile(d, 99.5))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    img = np.clip((d - lo) / (hi - lo), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "screenshots", "scene"))
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    from PIL import Image
    for cam_name in ["cam_rgb", "cam_depth", "cam_side"]:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        r = mujoco.Renderer(model, height=720, width=1280)
        r.update_scene(data, camera=cid)
        rgb = r.render()
        Image.fromarray(rgb).save(f"{args.out}_{cam_name}.png")

        r.enable_depth_rendering()
        depth = r.render()
        lo, hi = save_depth_png(depth, f"{args.out}_{cam_name}_depth.png")

        r.enable_segmentation_rendering()
        seg = r.render()
        Image.fromarray((seg[:, :, 0] * 8 % 255).astype(np.uint8)).save(
            f"{args.out}_{cam_name}_seg.png")
        r.close()
        print(f"{cam_name}: depth[{lo:.3f},{hi:.3f}] m  saved")

    # 关键几何数值（用于核对场景尺度）
    for name in ["red_cube", "yellow_obstacle", "hand"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"{name:16s} pos={np.round(data.xpos[bid], 4)}")
    print("out prefix:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
