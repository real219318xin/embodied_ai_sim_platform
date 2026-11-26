"""夹爪位置伺服验证：阶跃响应 / 保持 / 限位。

用法: python scripts/probe_gripper3.py
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")

TENDON, GL, GR = 7, 8, 9  # 官方 actuator8, gripper_left, gripper_right


def new_sim():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d


def main():
    m, d = new_sim()
    print("nu =", m.nu, [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])

    def cmd(open_m, steps):
        d.ctrl[TENDON] = 0.0
        d.ctrl[GL] = d.ctrl[GR] = open_m
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return d.qpos[7], d.qpos[8]

    print("\n阶跃: 目标开度 -> 实际")
    for target in [0.0, 0.01, 0.02, 0.03, 0.04]:
        q1, q2 = cmd(target, 1000)
        print(f"  target={target*1000:5.1f} mm  actual=({q1*1000:6.2f}, {q2*1000:6.2f}) mm "
              f"err={abs(q1-target)*1000:.3f} mm")

    print("\n保持性: 目标 0.035 m，连续 5 s 每 1 s 采样")
    cmd(0.035, 500)
    for k in range(5):
        q1, q2 = cmd(0.035, 500)
        print(f"  t={k+1}s  q=({q1*1000:.2f}, {q2*1000:.2f}) mm")

    print("\n闭合到 0，检查是否接触限位（不应穿模到负值）")
    q1, q2 = cmd(0.0, 1500)
    print(f"  q=({q1*1000:.3f}, {q2*1000:.3f}) mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
