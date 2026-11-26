"""阶段四验证：Panda 关节读取 / 末端位姿 / IK / 笛卡尔伺服 / 夹爪 / 限位 / 碰撞。

全部运动通过 actuator + mj_step 完成；脚本会打印实测数值作为证据。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv
from motion_control.controller import PandaController, OPEN_WIDE, OPEN_GRASP
from motion_control.kinematics import down_grasp_rotation

OUT = os.path.join(ROOT, "screenshots")


def main():
    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()
    ok_all = True

    print("=== 1. 关节状态读取 ===")
    print("q      :", np.round(ctrl.q(), 4))
    print("qdot   :", np.round(env.data.qvel[:7], 6))
    print("effort :", np.round(ctrl.joint_effort(), 3))
    p0, R0 = ctrl.tcp()
    print("TCP    :", np.round(p0, 4))
    print("R      :\n", np.round(R0, 3))

    print("\n=== 2. 关节限位检查 ===")
    lo, hi = ctrl.kin.jnt_lo, ctrl.kin.jnt_hi
    print("limits :", np.round(np.column_stack([lo, hi]), 3).tolist())
    print("当前在限位内:", ctrl.kin.within_limits(ctrl.q()))
    print("故意越界 (+5 rad) 是否被拒绝:",
          not ctrl.move_joints(ctrl.q() + 5.0, duration=0.1).success)

    print("\n=== 3. 逆运动学（DLS） ===")
    target_pos = np.array([0.66, -0.18, 0.60])
    R_down = down_grasp_rotation(0.0)
    res = ctrl.kin.solve_ik(target_pos, R_down)
    print(f"IK success={res.success} iters={res.iterations} "
          f"pos_err={res.pos_err*1000:.2f} mm rot_err={np.degrees(res.rot_err):.2f} deg")
    print("q_sol  :", np.round(res.q, 4))
    print("在限位内:", ctrl.kin.within_limits(res.q))
    ok_all &= res.success

    print("\n=== 4. 关节空间运动（min-jerk 插值 + 位置伺服） ===")
    mr = ctrl.move_joints(res.q, duration=2.0)
    print(f"move_joints success={mr.success} reason='{mr.reason}' steps={mr.steps}")
    p1, R1 = ctrl.tcp()
    print("TCP after move:", np.round(p1, 4),
          " 位置误差 %.2f mm" % (np.linalg.norm(p1 - target_pos) * 1000))
    print("关节跟踪误差 (rad):", np.round(ctrl.q() - res.q, 4))
    ok_all &= mr.success

    print("\n=== 5. 笛卡尔闭环伺服 ===")
    target2 = np.array([0.66, -0.18, 0.52])
    mr2 = ctrl.move_to_pose(target2, R_down, max_duration=4.0, gripper=OPEN_GRASP)
    p2, _ = ctrl.tcp()
    print(f"move_to_pose success={mr2.success} reason='{mr2.reason}' "
          f"steps={mr2.steps} pos_err={np.linalg.norm(p2-target2)*1000:.2f} mm")
    ok_all &= mr2.success

    print("\n=== 6. 夹爪开合（真实执行器） ===")
    o1 = ctrl.open_gripper(OPEN_WIDE, steps=250)
    o2 = ctrl.set_gripper(0.0, steps=350)
    print(f"open -> {o1*1000:.1f} mm ; close -> {o2*1000:.1f} mm")

    print("\n=== 7. 碰撞检测（接触扫描） ===")
    rep = ctrl.collision_report()
    print("当前碰撞数:", rep["count"], rep["collisions"][:2])

    print("\n=== 8. 抬升并截图 ===")
    ctrl.open_gripper(OPEN_WIDE, steps=200)
    ctrl.move_to_pose(np.array([0.60, -0.10, 0.75]), R_down, max_duration=5.0)
    ctrl.move_joints(ctrl.q_home, duration=2.5)
    print("回到 home，TCP:", np.round(ctrl.tcp()[0], 4))
    print("关节轨迹记录点数:", len(ctrl.traj))

    env.forward()
    img = env.render("cam_rgb", 1280, 720)
    from PIL import Image
    Image.fromarray(img).save(os.path.join(OUT, "motion_test.png"))
    print("截图:", os.path.join(OUT, "motion_test.png"))

    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
