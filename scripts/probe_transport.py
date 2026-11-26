"""探针：抓取后测量搬运段的稳态误差（判断是伺服限幅还是关节力矩不足）。"""
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
                      max_duration=6.0, gripper=0.038, tol_pos=0.003)
    ctrl.move_linear(np.array([obj[0], obj[1], obj[2] + 0.004]), R,
                     speed=0.05, gripper=0.038, allow_body="red_cube",
                     tol_pos=0.0015)
    g = ctrl.close_gripper(steps=500, object_body="red_cube", force_threshold=0.3)
    print("close:", {k: g[k] for k in ("contact", "peak_force", "opening",
                                       "both_fingers")})
    mr = ctrl.move_linear(np.array([obj[0], obj[1], 0.62]), R, speed=0.05,
                          gripper=0.0, allow_body="red_cube")
    print("lift to safe_z:", mr.success, mr.reason,
          "err=%.1f mm rot=%.2f deg" % (mr.pos_err * 1000,
                                        np.degrees(mr.rot_err)))

    target = np.array([0.78, -0.10, 0.62])
    # 逐段走，每段打印误差，观察是否收敛
    p0, _ = ctrl.tcp()
    n = 8
    for k in range(1, n + 1):
        p = p0 + (target - p0) * (k / n)
        r = ctrl.move_to_pose(p, R, max_duration=3.0, tol_pos=0.005,
                              tol_rot=0.05, gripper=0.0,
                              allow_body="red_cube")
        p_now, _ = ctrl.tcp()
        print(f"  seg {k}: ok={r.success} err={np.linalg.norm(p_now-p)*1000:6.2f} mm"
              f" rot={np.degrees(float(np.arccos(np.clip((np.trace(R @ ctrl.tcp()[1].T)-1)/2,-1,1)))):5.2f} deg"
              f" tq={np.round(np.abs(ctrl.joint_effort()),1)}")

    touch = ContactSensor(env)
    rep = touch.grasp_report("red_cube")
    print("grasp after transport: L=%.2f R=%.2f stable=%s"
          % (rep.left_force_n, rep.right_force_n, rep.grasp_stable))
    print("cube pos:", np.round(env.body_pos("red_cube"), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
