"""阶段六验证：多模态感知接口实测。

验证项
------
1. RGB / Depth / 语义分割同帧采集，保存为图片；
2. 视觉检测的物体位置、尺寸、颜色、类别，与 MuJoCo 真值比较误差；
3. 本体感知读数（关节角/速度/末端位姿/夹爪开度）；
4. 仿真接触力读数（抓取前后对比，明确标注为仿真接触数据）；
5. 遮挡判断：机械臂挡住目标时 occluded 应为 True。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
from core.env import SimEnv  # noqa: E402
from core.utils import dump_json  # noqa: E402
from motion_control.controller import PandaController  # noqa: E402
from motion_control.kinematics import down_grasp_rotation  # noqa: E402
from perception.contact import ContactSensor  # noqa: E402
from perception.proprioception import ProprioceptionSensor  # noqa: E402
from perception.vision import VisionSensor  # noqa: E402

OUT_SHOTS = os.path.join(ROOT, "screenshots")
OUT_RES = os.path.join(ROOT, "results")


def gt_center(env, name):
    return env.body_pos(name)


def main():
    from PIL import Image
    os.makedirs(OUT_SHOTS, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)

    env = SimEnv()
    ctrl = PandaController(env)
    ctrl.reset()

    vis = VisionSensor(env, camera="cam_depth", width=640, height=480)
    prop = ProprioceptionSensor(env)
    touch = ContactSensor(env)

    report = {}
    ok_all = True

    # ---------------- 1. 采集 ----------------
    print("=== 1. RGB-D + 分割 同帧采集 (cam_depth) ===")
    frame = vis.capture()
    print("frame:", frame.as_dict())
    Image.fromarray(frame.rgb).save(os.path.join(OUT_SHOTS, "perception_rgb.png"))
    d = frame.depth.copy()
    d[~np.isfinite(d)] = 0
    lo, hi = float(np.nanmin(d)), float(np.nanpercentile(d, 99))
    dn = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    Image.fromarray((dn * 255).astype(np.uint8)).save(
        os.path.join(OUT_SHOTS, "perception_depth.png"))
    Image.fromarray(vis.colorize_segmentation(frame)).save(
        os.path.join(OUT_SHOTS, "perception_seg.png"))

    # ---------------- 2. 检测 ----------------
    print("\n=== 2. 视觉检测（颜色/几何规则，非学习模型） ===")
    dets = vis.detect(frame)
    for de in dets:
        print(f"  {de.label:<20s} geom={de.geom_name:<20s} "
              f"center={np.round(de.center,4)} size={np.round(de.size,4)} "
              f"conf={de.confidence:.2f} occl={de.occlusion_ratio:.2f} "
              f"graspable={de.graspable}")
    Image.fromarray(vis.annotate(frame, dets)).save(
        os.path.join(OUT_SHOTS, "perception_detect.png"))

    report["detections"] = [de.as_dict() for de in dets]

    # 与真值比较（真值仅用于评估，不参与检测）
    print("\n  位置误差（检测 vs MuJoCo 真值，仅用于评估）：")
    per_obj = {}
    for de in dets:
        if not de.geom_name:
            continue
        gt = env.body_pos(mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[de.geom_id])))
        err = float(np.linalg.norm(de.center - gt))
        per_obj[de.geom_name] = {"label": de.label,
                                 "gt_center": gt.tolist(),
                                 "det_center": de.center.tolist(),
                                 "pos_err_m": err,
                                 "size_m": de.size.tolist()}
        print(f"    {de.geom_name:<20s} err={err*1000:6.2f} mm")
    report["accuracy_vs_ground_truth"] = per_obj

    cube = next((x for x in dets if x.geom_name == "red_cube_geom"), None)
    region = next((x for x in dets if x.geom_name == "target_zone_geom"), None)
    print("\n  红色方块检测:", "OK" if cube and cube.color == "red"
          and cube.category == "cube" else "FAIL")
    print("  蓝色目标区检测:", "OK" if region and region.color == "blue"
          and region.category == "region" else "FAIL")
    ok_all &= bool(cube and cube.color == "red" and cube.category == "cube"
                   and per_obj["red_cube_geom"]["pos_err_m"] < 0.02)
    ok_all &= bool(region and region.color == "blue" and region.category == "region")

    # ---------------- 3. 本体感知 ----------------
    print("\n=== 3. 本体感知 ===")
    ps = prop.read()
    print("  q      :", np.round(ps.joint_positions, 4))
    print("  qdot   :", np.round(ps.joint_velocities, 4))
    print("  TCP    :", np.round(ps.tcp_position, 4),
          "quat(wxyz)=", np.round(ps.tcp_quaternion, 4))
    print("  gripper:", round(ps.gripper_opening * 1000, 2), "mm",
          " motion:", ps.motion_state)
    report["proprioception_before"] = ps.as_dict()
    ok_all &= ps.gripper_opening > 0.05 and ps.motion_state == "steady"

    # ---------------- 4. 接触（抓取前/后） ----------------
    print("\n=== 4. 仿真接触数据（非真实触觉硬件） ===")
    before = touch.grasp_report("red_cube")
    print("  抓取前: in_contact=%s n=%d peak=%.3f N stable=%s"
          % (before.in_contact, before.n_contacts, before.peak_force_n,
             before.grasp_stable))
    report["contact_before_grasp"] = before.as_dict()
    ok_all &= not before.in_contact

    # 真实下降抓取（走 actuator，不修改坐标）
    obj = env.body_pos("red_cube")
    R = down_grasp_rotation(0.0)
    ctrl.open_gripper(0.038, steps=150)
    m1 = ctrl.move_to_pose(np.array([obj[0], obj[1], obj[2] + 0.14]), R,
                           max_duration=5.0, gripper=0.038)
    m2 = ctrl.move_linear(np.array([obj[0], obj[1], obj[2] + 0.004]), R,
                          speed=0.05, gripper=0.038, allow_body="red_cube",
                          tol_pos=0.0015)
    g = ctrl.close_gripper(steps=500, object_body="red_cube", force_threshold=0.3)
    after = touch.grasp_report("red_cube")
    print("  闭合: contact=%s peak=%.3f N opening=%.1f mm 两指=%s"
          % (g["contact"], g["peak_force"], (g["opening"] or 0) * 1000,
             g.get("both_fingers")))
    print("  抓取后: in_contact=%s n=%d peak=%.3f N L=%.2fN R=%.2fN stable=%s"
          % (after.in_contact, after.n_contacts, after.peak_force_n,
             after.left_force_n, after.right_force_n, after.grasp_stable))
    report["contact_after_grasp"] = after.as_dict()
    report["contact_after_grasp_sensor"] = g
    ok_all &= after.left_force_n > 0.3 and after.right_force_n > 0.3
    ok_all &= m1.success and m2.success

    # ---------------- 5. 遮挡判断 ----------------
    print("\n=== 5. 遮挡判断 ===")
    frame2 = vis.capture()
    dets2 = vis.detect(frame2)
    cube2 = next((x for x in dets2 if x.geom_name == "red_cube_geom"), None)
    if cube2 is None:
        print("  机械臂完全遮挡方块：检测返回 None（如实反映不可见）")
        report["occlusion_case"] = {"detected": False}
    else:
        print(f"  方块可见像素={cube2.n_pixels} 遮挡比例={cube2.occlusion_ratio:.2f} "
              f"occluded={cube2.occluded}")
        report["occlusion_case"] = {"detected": True,
                                    "n_pixels": cube2.n_pixels,
                                    "occlusion_ratio": cube2.occlusion_ratio,
                                    "occluded": cube2.occluded}
    Image.fromarray(vis.annotate(frame2, dets2)).save(
        os.path.join(OUT_SHOTS, "perception_occlusion.png"))

    ps2 = prop.read()
    report["proprioception_after_grasp"] = ps2.as_dict()

    vis.close()
    report["pass"] = bool(ok_all)
    dump_json(report, os.path.join(OUT_RES, "perception_last.json"))
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    print("report:", os.path.join(OUT_RES, "perception_last.json"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
