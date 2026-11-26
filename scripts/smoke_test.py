"""MuJoCo 最小可用性验证（阶段一）。

逐项验证：import / MjModel / MjData / mj_step / 离屏 RGB-D 渲染。
不涉及机械臂，仅证明本机 MuJoCo 物理与渲染链路可用。
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
os.makedirs(SHOTS, exist_ok=True)

MINIMAL_XML = """
<mujoco model="smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.55 0.55 0.6 1"/>
    <body name="box" pos="0 0 1.0">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="0.2"
            rgba="0.85 0.15 0.15 1" friction="1.0 0.05 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    report = {}

    report["mujoco_version"] = mujoco.__version__

    # 1) MjModel
    model = mujoco.MjModel.from_xml_string(MINIMAL_XML)
    report["nq"] = int(model.nq)
    report["nbody"] = int(model.nbody)
    report["ngeom"] = int(model.ngeom)

    # 2) MjData
    data = mujoco.MjData(model)
    report["initial_height"] = float(data.qpos[2])

    # 3) mj_step：自由落体应使高度下降，落地后静止
    for _ in range(1500):
        mujoco.mj_step(model, data)
    report["after_fall_height"] = float(data.qpos[2])
    report["fell_down"] = bool(data.qpos[2] < 0.9)
    report["z_velocity_abs"] = float(abs(data.qvel[2]))
    report["ncon"] = int(data.ncon)

    # 4) 离屏渲染 RGB + Depth
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=-1)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.close()

    report["rgb_shape"] = list(rgb.shape)
    report["depth_shape"] = list(depth.shape)
    report["depth_min"] = float(depth.min())
    report["depth_max"] = float(depth.max())
    report["rgb_mean"] = float(rgb.mean())

    try:
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(SHOTS, "smoke_rgb.png"))
        d = depth.copy()
        d[~np.isfinite(d)] = 0.0
        dn = (d - d.min()) / max(1e-9, (d.max() - d.min()))
        Image.fromarray((dn * 255).astype(np.uint8)).save(
            os.path.join(SHOTS, "smoke_depth.png"))
        report["images_saved"] = True
    except Exception as exc:  # pragma: no cover
        report["images_saved"] = False
        report["image_error"] = repr(exc)

    for k, v in report.items():
        print(f"{k}: {v}")

    ok = (
        report["fell_down"]
        and report["z_velocity_abs"] < 0.05
        and report["ncon"] > 0
        and report["rgb_shape"] == [480, 640, 3]
        and report["depth_max"] > 1.0
        and report["images_saved"]
    )
    print("SMOKE_TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
