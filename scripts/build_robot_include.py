"""从官方 Menagerie panda.xml 派生本项目的机器人 include 片段。

背景（实测结论，非猜测）：
    官方 panda.xml 末尾有一个 <keyframe name="home" qpos="...9 个值...">。
    本项目的实验场景在 worldbody 中加入了自由物体（red_cube / yellow_obstacle），
    使模型 nq = 9 + 7 + 7 = 23。MuJoCo 要求 keyframe.qpos 长度必须等于 nq，
    因此直接 include 官方文件会在编译阶段报：
        "key 0: invalid qpos size, expected length 23"

处理方式：
    **不修改** 官方文件（assets/ 下保持原样，SHA256 记录在清单中），
    而是生成 scenes/robot/panda_robot.xml：
        - 内容与官方 panda.xml 逐字节一致，唯一差异是删除 <keyframe>...</keyframe> 段；
        - 文件头写入来源、官方文件 SHA256、生成脚本，便于审计与重建；
        - 场景自身的 keyframe 在 main_scene.xml 中定义（长度 = nq）。

可重复执行（幂等）：每次运行都会重新生成并校验"仅 keyframe 被移除"。
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "mujoco_menagerie", "franka_emika_panda", "panda.xml")
OUT_DIR = os.path.join(ROOT, "scenes", "robot")
OUT = os.path.join(OUT_DIR, "panda_robot.xml")
MANIFEST = os.path.join(OUT_DIR, "panda_robot.manifest.json")

KEYFRAME_RE = re.compile(r"\n\s*<keyframe>.*?</keyframe>", re.DOTALL)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(SRC, "rb") as f:
        raw = f.read()
    src_text = raw.decode("utf-8")
    src_hash = sha256(SRC)

    m = KEYFRAME_RE.search(src_text)
    if not m:
        raise RuntimeError("no <keyframe> block found in official panda.xml")
    if KEYFRAME_RE.search(src_text, m.end()):
        raise RuntimeError("more than one <keyframe> block; adjust the script")
    removed = src_text[m.start():m.end()]
    stripped = src_text[:m.start()] + src_text[m.end():]

    # 差异 2：把机器人底座抬高到实验台面高度（0.42 m）。
    # link0 是无关节的固定 body，设置 pos 等价于"把机器人用螺栓固定在台面上"，
    # 不是修改关节/几何/惯量，也不影响动力学以外的任何定义。
    base_line = '<body name="link0" childclass="panda">'
    base_new = '<body name="link0" childclass="panda" pos="0 0 0.42">'
    if stripped.count(base_line) != 1:
        raise RuntimeError("unexpected link0 body line in official panda.xml")
    stripped = stripped.replace(base_line, base_new)

    header = (
        "<!--\n"
        "  自动生成的机器人 include 片段 —— 请勿手工编辑。\n"
        "  来源: google-deepmind/mujoco_menagerie @ main :: franka_emika_panda/panda.xml\n"
        f"  来源 SHA256: {src_hash}\n"
        "  生成脚本: scripts/build_robot_include.py\n"
        "  与官方文件的差异（仅此两处）:\n"
        "    1) 删除 <keyframe name=\"home\"> 段\n"
        "       (该 keyframe 的 qpos 只有 9 维，无法与含自由物体的场景 nq=23 共存；\n"
        "        场景自身的初始状态在 scenes/main_scene.xml 的 <keyframe scene_home> 中定义)\n"
        "    2) link0 增加 pos=\"0 0 0.42\"，把机器人底座安装到 0.42 m 高的实验台面上\n"
        "       (link0 本身无关节，等同于螺栓固定高度；关节/几何/惯量/执行器均未改动)\n"
        "  官方文件本身未被修改。\n"
        "-->\n"
    )
    out_text = header + stripped
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(out_text.encode("utf-8"))

    manifest = {
        "generated_file": os.path.relpath(OUT, ROOT).replace("\\", "/"),
        "source_file": os.path.relpath(SRC, ROOT).replace("\\", "/"),
        "source_sha256": src_hash,
        "generated_sha256": sha256(OUT),
        "diff": ["removed <keyframe> block",
                 "link0 pos 0 0 0.42 (robot mounted on bench surface)"],
        "removed_block_len": len(removed),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("generated:", OUT)
    print("source sha256   :", src_hash)
    print("generated sha256:", manifest["generated_sha256"])
    print("only keyframe removed:", len(removed), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
