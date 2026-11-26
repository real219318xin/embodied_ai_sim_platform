"""探针脚本：确认 actuator 语义（位置伺服 / 夹爪开合方向 / 跟踪精度）。

不修改任何模型文件，只读取仿真结果。
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")


def hand_tcp(model, data, offset=0.103):
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    p = data.xpos[hid] + data.xmat[hid].reshape(3, 3) @ np.array([0, 0, offset])
    return p, data.xmat[hid].reshape(3, 3)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    qadr = [model.jnt_qposadr[j] for j in range(7)]
    print("=== 1. 初始（keyframe scene_home） ===")
    print("arm qpos:", np.round(data.qpos[qadr], 4))
    print("finger qpos:", np.round(data.qpos[7:9], 5))
    p, R = hand_tcp(model, data)
    print("TCP pos:", np.round(p, 4))
    print("hand R:\n", np.round(R, 3))

    print("\n=== 2. 夹爪 ctrl=0（观察是否闭合） ===")
    data.ctrl[:] = model.key_ctrl[0]
    data.ctrl[7] = 0.0
    for _ in range(1500):
        mujoco.mj_step(model, data)
    print("finger qpos:", np.round(data.qpos[7:9], 5),
          "-> 开度 %.1f mm" % (1000 * (data.qpos[7] + data.qpos[8])))

    print("\n=== 3. 夹爪 ctrl=255（观察是否张开） ===")
    data.ctrl[7] = 255.0
    for _ in range(1500):
        mujoco.mj_step(model, data)
    print("finger qpos:", np.round(data.qpos[7:9], 5),
          "-> 开度 %.1f mm" % (1000 * (data.qpos[7] + data.qpos[8])))

    print("\n=== 4. 关节位置伺服跟踪（ctrl 偏移 0.2 rad） ===")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    q0 = data.qpos[qadr].copy()
    data.ctrl[:7] = q0
    # 目标：joint1 +0.2, joint4 +0.2，其余不变
    tgt = q0.copy()
    tgt[0] += 0.2
    tgt[3] += 0.2
    data.ctrl[:7] = tgt
    for i in range(3000):
        mujoco.mj_step(model, data)
    q1 = data.qpos[qadr]
    print("ctrl target:", np.round(tgt, 4))
    print("qpos actual:", np.round(q1, 4))
    print("跟踪误差 (rad):", np.round(q1 - tgt, 5),
          " max=%.5f" % np.max(np.abs(q1 - tgt)))

    print("\n=== 5. 无控制（ctrl 保持 home）时是否稳定 ===")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    print("qpos:", np.round(data.qpos[qadr], 4))
    print("基座位置:", np.round(data.xpos[1], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
