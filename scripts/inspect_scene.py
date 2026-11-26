"""场景自检：加载 main_scene.xml 并输出机器人 / 场景结构报告。

用于阶段二、三的"不允许猜测 XML 结构"要求：把实际解析结果打印出来。
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scenes", "main_scene.xml")


def name_of(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or f"<unnamed:{idx}>"


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    print("=" * 70)
    print("MODEL:", SCENE)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
          f"ngeom={model.ngeom} njnt={model.njnt} ncam={model.ncam} nkey={model.nkey}")
    print(f"timestep={model.opt.timestep}  gravity={model.opt.gravity}  "
          f"integrator={model.opt.integrator}")

    print("-" * 70)
    print("BODIES")
    for i in range(model.nbody):
        p = model.body_parentid[i]
        print(f"  [{i:2d}] {name_of(model, mujoco.mjtObj.mjOBJ_BODY, i):<20s} "
              f"parent={name_of(model, mujoco.mjtObj.mjOBJ_BODY, p) if p else 'world':<12s} "
              f"mass={model.body_mass[i]:.4f} ngeom={int(model.body_geomnum[i])}")

    print("-" * 70)
    print("JOINTS (name / type / range / actuator)")
    act_by_joint = {}
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        act_by_joint.setdefault(jid, []).append(
            name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    for j in range(model.njnt):
        nm = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jt = model.jnt_type[j]
        lim = model.jnt_range[j]
        acts = act_by_joint.get(j, [])
        print(f"  [{j}] {nm:<16s} type={int(jt)} limited={int(model.jnt_limited[j])} "
              f"range=({lim[0]:+.4f},{lim[1]:+.4f}) qposadr={model.jnt_qposadr[j]} "
              f"dofadr={model.jnt_dofadr[j]} actuators={acts}")

    print("-" * 70)
    print("ACTUATORS")
    for a in range(model.nu):
        print(f"  [{a}] {name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<12s} "
              f"trnid={model.actuator_trnid[a]} gear={model.actuator_gear[a, 0]:.4f} "
              f"ctrlrange=({model.actuator_ctrlrange[a][0]:+.4f},"
              f"{model.actuator_ctrlrange[a][1]:+.4f}) "
              f"forcerange=({model.actuator_forcerange[a][0]:+.2f},"
              f"{model.actuator_forcerange[a][1]:+.2f})")

    print("-" * 70)
    print("EQUALITY CONSTRAINTS (夹爪同步关系)")
    for e in range(model.neq):
        print(f"  [{e}] type={int(model.eq_type[e])} obj1={model.eq_obj1id[e]} "
              f"obj2={model.eq_obj2id[e]} active={int(model.eq_active0[e])}")

    print("-" * 70)
    print("GEOMS (场景物体)")
    for g in range(model.ngeom):
        nm = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm.startswith(("link", "hand", "finger")):
            continue
        b = int(model.geom_bodyid[g])
        sz = model.geom_size[g]
        print(f"  [{g:2d}] {nm:<22s} body={name_of(model, mujoco.mjtObj.mjOBJ_BODY, b):<16s} "
              f"type={int(model.geom_type[g])} size=({sz[0]:.4f},{sz[1]:.4f},{sz[2]:.4f}) "
              f"contype={int(model.geom_contype[g])} conaffinity={int(model.geom_conaffinity[g])} "
              f"friction={model.geom_friction[g][0]:.2f} mass={model.body_mass[b]:.3f}")

    print("-" * 70)
    print("CAMERAS")
    for c in range(model.ncam):
        print(f"  [{c}] {name_of(model, mujoco.mjtObj.mjOBJ_CAMERA, c):<12s} "
              f"pos={np.round(model.cam_pos[c], 3)} fovy={model.cam_fovy[c]:.1f}")

    print("-" * 70)
    print("KEYFRAMES")
    for k in range(model.nkey):
        print(f"  [{k}] {name_of(model, mujoco.mjtObj.mjOBJ_KEY, k)} qpos={np.round(model.key_qpos[k], 3)}")

    # ---- 物理健全性检查 ----
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(1000):  # 2 s
        mujoco.mj_step(model, data)
    print("-" * 70)
    print("SETTLE TEST (2 s after keyframe)")
    for b in ["red_cube", "yellow_obstacle", "bench"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"  {b:<18s} pos={np.round(data.xpos[bid], 4)} "
              f"z_vel={data.cvel[bid][2]:+.5f}")
    print(f"  robot base link0 pos={np.round(data.xpos[1], 4)}")
    print(f"  ncon={data.ncon}  robot fallen? "
          f"{'YES' if data.xpos[1][2] < -0.01 else 'NO'}")

    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    print(f"  hand(pos) at settle = {np.round(data.xpos[tcp], 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
