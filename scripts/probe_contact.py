"""探针：抓取后逐条打印夹爪-方块接触，定位左右指接触不对称的原因。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv
from motion_control.controller import PandaController
from motion_control.kinematics import down_grasp_rotation
from perception.contact import ContactSensor


def main():
    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    obj = env.body_pos("red_cube")
    R = down_grasp_rotation(0.0)
    ctrl.open_gripper(0.038, steps=150)
    ctrl.move_to_pose(np.array([obj[0], obj[1], obj[2] + 0.14]), R,
                      max_duration=5.0, gripper=0.038)
    ctrl.move_linear(np.array([obj[0], obj[1], obj[2] + 0.004]), R,
                     speed=0.05, gripper=0.038, allow_body="red_cube",
                     tol_pos=0.0015)
    p, _ = ctrl.tcp()
    print("tcp after descend:", np.round(p, 5), "target:",
          np.round([obj[0], obj[1], obj[2] + 0.004], 5))
    g = ctrl.close_gripper(steps=500, object_body="red_cube", force_threshold=0.3)
    print("close info:", g)

    print("\n手指关节 qpos:", np.round(env.data.qpos[7:9], 5))
    print("cube pos:", np.round(env.body_pos("red_cube"), 4))
    print("fingers pos:", np.round(env.body_pos("left_finger"), 4),
          np.round(env.body_pos("right_finger"), 4))

    print("\n所有接触(cube 相关 / 手指相关):")
    for c in env.contacts():
        if "red_cube" in (c.body1, c.body2) or "finger" in (c.body1, c.body2) or c.body1 == "hand" or c.body2 == "hand":
            print(f"  {c.body1:<14s} <-> {c.body2:<14s} "
                  f"geom={env.geom_name(c.geom1)}/{env.geom_name(c.geom2)} "
                  f"F={c.force_magnitude:7.3f} N pen={-c.dist*1000:6.3f} mm "
                  f"pos={np.round(c.position,4)}")

    touch = ContactSensor(env)
    rep = touch.grasp_report("red_cube")
    print("\nreport:", {k: v for k, v in rep.as_dict().items() if k != "samples"})
    print("n samples:", len(rep.samples))
    for s in rep.samples[:8]:
        print("  ", s.body_a, s.body_b, round(s.force_n, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
