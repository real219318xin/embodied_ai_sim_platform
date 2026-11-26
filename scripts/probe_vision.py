"""探针：确认 MuJoCo 分割渲染的返回格式与深度图语义。

需要确认：
1. enable_segmentation_rendering() 返回数组的形状与含义（object_id, object_type）。
2. 深度图数值是"沿光轴的 z 深度"还是"到相机的欧氏距离"。
3. 由深度 + 相机内参反投影得到的世界坐标，与 MuJoCo 真值 xpos 的偏差。
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.env import SimEnv

W, H = 640, 480


def main():
    env = SimEnv()
    env.reset()
    m, d = env.model, env.data

    cam_name = "cam_depth"
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    print("camera", cam_name, "pos", np.round(d.cam_xpos[cid], 4))
    print("camera xmat:\n", np.round(d.cam_xmat[cid].reshape(3, 3), 4))
    print("fovy(deg)", m.cam_fovy[cid])

    r = mujoco.Renderer(m, height=H, width=W)

    # ---- segmentation ----
    r.update_scene(d, camera=cid)
    r.enable_segmentation_rendering()
    seg = r.render()
    print("\nseg dtype/shape:", seg.dtype, seg.shape)
    ids = seg[..., 0]
    types = seg[..., 1]
    print("unique type values:", np.unique(types)[:10])
    uniq = np.unique(ids)
    print("n unique ids:", len(uniq))
    for u in uniq[:12]:
        if u == 0:
            continue
        masked = types[ids == u]
        t = int(masked[0])
        name = mujoco.mj_id2name(m, t, int(u))
        print(f"  id={u:4d} type={t} name={name}")

    # ---- depth ----
    r.disable_segmentation_rendering()
    r.enable_depth_rendering()
    depth = r.render()
    r.close()
    print("\ndepth shape", depth.shape, "min", depth.min(), "max", depth.max())

    # 相机内参（方形像素）
    fovy = np.radians(m.cam_fovy[cid])
    f = 0.5 * H / np.tan(fovy / 2)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    print("focal", f, "cx,cy", cx, cy)

    Rc = d.cam_xmat[cid].reshape(3, 3)   # camera -> world
    pc = d.cam_xpos[cid].copy()

    # 1) 用红色方块的分割 mask 反投影，检查 z-depth 假设
    seg_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom")
    print("\nred_cube_geom id:", seg_id)
    # 重新渲染分割（renderer 已关闭，重建一次）
    r2 = mujoco.Renderer(m, height=H, width=W)
    r2.update_scene(d, camera=cid)
    r2.enable_segmentation_rendering()
    seg2 = r2.render()
    r2.close()
    mask = (seg2[..., 0] == seg_id) & (seg2[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    print("mask pixels:", int(mask.sum()))
    if mask.sum() == 0:
        print("!! cube not visible from cam_depth")
        return 1

    vs, us = np.nonzero(mask)
    zs = depth[us, vs]          # depth[v, u]
    valid = np.isfinite(zs) & (zs > 0)
    us, vs, zs = us[valid], vs[valid], zs[valid]
    print("valid pixels:", len(zs), "z range", zs.min(), zs.max())

    # MuJoCo/OpenGL 相机坐标系：x 向右、y 向上、z 向后（视线方向为 -z），
    # 图像 v 轴向下，因此 y 分量取负。
    dirs = np.stack([(us - cx) / f, -(vs - cy) / f, -np.ones_like(zs)], axis=1)
    pts_z = (Rc @ (dirs * zs[:, None]).T).T + pc
    # 假设 B: depth 是欧氏距离
    norms = np.linalg.norm(dirs, axis=1)
    pts_d = (Rc @ (dirs / norms[:, None] * zs[:, None]).T).T + pc

    gt = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "red_cube")].copy()
    print("ground-truth cube xpos:", np.round(gt, 4))
    print("A(z-depth)   centroid:", np.round(pts_z.mean(0), 4),
          "err=%.2f mm" % (np.linalg.norm(pts_z.mean(0) - gt) * 1000))
    print("B(euclidean) centroid:", np.round(pts_d.mean(0), 4),
          "err=%.2f mm" % (np.linalg.norm(pts_d.mean(0) - gt) * 1000))

    # 用 mask 上最高点（离相机最近）估计物体表面
    print("A top surface z:", np.round(np.percentile(pts_z[:, 2], 95), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
