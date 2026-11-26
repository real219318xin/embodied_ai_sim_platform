"""夹爪执行器语义扫描：确定 ctrl 与手指开度的实际关系（实测，不猜测）。"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")


def run(ctrl8, steps=800):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(20):
        mujoco.mj_step(model, data)
    data.ctrl[7] = ctrl8
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos[7], data.qpos[8], data.actuator_force[7]


def main():
    print("ctrl8 | finger1 | finger2 | tendon_force")
    for c in [-500, -255, -100, 0, 32, 64, 128, 200, 255, 400, 1000]:
        q1, q2, f = run(c)
        print(f"{c:6.0f} | {q1:+.5f} | {q2:+.5f} | {f:+.3f}")

    # 尝试在没有 ctrl 的情况下，手指是否会被弹簧约束拉开（被动）
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[7] = data.qpos[8] = 0.04
    for _ in range(500):
        mujoco.mj_step(model, data)
    print("从 0.04 出发、ctrl 保持 keyframe(255):", np.round(data.qpos[7:9], 5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
