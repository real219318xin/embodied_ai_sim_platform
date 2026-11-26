"""MuJoCo 交互式 Viewer 启动验证（阶段一第 5/6 项）。

使用 launch_passive 创建真实 OpenGL 窗口，显示一个弹跳立方体，
并在总计 duration 秒内持续 mj_step + sync（时间由墙钟控制）。
窗口标题包含 "MuJoCo Viewer Smoke Test"，便于外部截屏取证。
"""
import argparse
import time

import mujoco

XML = """
<mujoco model="viewer_smoke">
  <option timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.6 0.62 0.66 1"/>
    <body name="ball" pos="0 0 1.2">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.08" mass="0.3" rgba="0.85 0.2 0.15 1"/>
    </body>
    <body name="ramp" pos="1.2 0 0.4" euler="0 20 0">
      <geom type="box" size="0.5 0.4 0.03" rgba="0.2 0.5 0.85 1"
            friction="0.8 0.05 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    import mujoco.viewer as viewer

    with viewer.launch_passive(model, data) as v:
        print("VIEWER_WINDOW_CREATED:", v.is_running(), flush=True)
        t0 = time.time()
        while v.is_running() and (time.time() - t0) < args.duration:
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.002)
        print("VIEWER_SIM_TIME: %.3f" % data.time, flush=True)
    print("VIEWER_CLOSED_OK", flush=True)


if __name__ == "__main__":
    main()
