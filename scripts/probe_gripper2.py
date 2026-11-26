"""夹爪扩展执行器验证：主动张开 / 主动闭合 / 夹持方块。

用法: python scripts/probe_gripper2.py
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")

CLOSE_A, OPEN_A = 7, 8


def new_sim():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d


def drive(m, d, close, open_, steps):
    d.ctrl[CLOSE_A] = close
    d.ctrl[OPEN_A] = open_
    for _ in range(steps):
        mujoco.mj_step(m, d)


def main():
    m, d = new_sim()
    print("nu =", m.nu, "actuators:",
          [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
    print("初始 finger qpos:", np.round(d.qpos[7:9], 4))

    print("\n1) 主动张开 (open=255, close=0), 1.5 s")
    drive(m, d, 0.0, 255.0, 750)
    print("   finger qpos:", np.round(d.qpos[7:9], 4),
          "开度 %.1f mm" % (1000 * d.qpos[7]))

    print("\n2) 保持张开, 再 2 s（检查是否稳定不回弹）")
    drive(m, d, 0.0, 255.0, 1000)
    print("   finger qpos:", np.round(d.qpos[7:9], 4))

    print("\n3) 主动闭合 (close=255, open=0), 1.5 s")
    drive(m, d, 255.0, 0.0, 750)
    print("   finger qpos:", np.round(d.qpos[7:9], 4),
          "开度 %.1f mm" % (1000 * d.qpos[7]))

    print("\n4) 夹持测试：张开 -> 手动把机械臂降到方块上方 (仅用于探针验证)")
    # 探针里用 qpos 摆放只是为了快速验证夹持力学，不作为正式运动方式
    m2, d2 = new_sim()
    cube_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    hand_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_pos = d2.xpos[cube_id].copy()
    print("   cube:", np.round(cube_pos, 4))

    # 用位置控制把 qpos 拉到一个能让 TCP 落在方块上方的姿态（仅探针）
    # 先张开，再通过 IK 到位（复用正式模块在此处不可用，简化：直接设定目标 qpos 并由 actuator 执行）
    print("   (夹持力验证见 probe_grasp_force.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
