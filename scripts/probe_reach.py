"""探针：测量搬运高度/目标点变化时的可达精度（判断是否为运动学极限）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv
from motion_control.controller import PandaController
from motion_control.kinematics import down_grasp_rotation


def main():
    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    R = down_grasp_rotation(0.0)

    for xy in [(0.78, -0.10), (0.795, -0.085), (0.80, -0.12)]:
        for z in (0.62, 0.60, 0.58, 0.55, 0.50):
            ctrl.reset()
            p = np.array([xy[0], xy[1], z])
            r = ctrl.move_to_pose(p, R, max_duration=6.0, tol_pos=0.002,
                                  tol_rot=0.05, gripper=0.0, check_collision=False)
            tcp, _ = ctrl.tcp()
            print(f"xy={xy} z={z:.2f} -> ok={r.success} err={np.linalg.norm(tcp-p)*1000:6.2f} mm"
                  f" rot={np.degrees(r.rot_err):4.2f} deg  q={np.round(ctrl.q(),2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
