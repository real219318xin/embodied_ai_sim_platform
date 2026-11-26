"""平台级自动测试（快速，不渲染、不跑长动作）。

覆盖
----
* 场景与模型结构（自由度数量、执行器、相机、keyframe）
* 逆运动学精度与关节限位判断
* 中文指令规则解析（含无法解析的负例）
* 场景图关系与可达性判断
* 规划器在目标不存在时必须返回失败节点（不能伪造成功）
* 接触传感器在静置时的读数
"""
import numpy as np
import pytest

from core.env import SimEnv
from motion_control.kinematics import (PandaKinematics, down_grasp_rotation)
from perception.contact import ContactSensor
from scene_graph.graph import SceneGraph, SceneNode
from task_planning.language import parse_instruction
from task_planning.planner import TaskPlanner


@pytest.fixture(scope="module")
def env():
    e = SimEnv()
    e.reset(settle_steps=50)
    return e


# ------------------------------------------------ 场景
def test_scene_structure(env):
    m = env.model
    assert m.nq == 23, "7 臂关节 + 2 手指 + 2 自由物体(7+7) = 23"
    assert m.nu == 10, "7 臂 + tendon + 2 指 = 10 个执行器"
    cams = {mujoco_name for mujoco_name in
            ("cam_rgb", "cam_depth", "cam_side")}
    import mujoco
    for name in cams:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


def test_keyframe_and_objects(env):
    assert env._key_id >= 0
    assert env.body_pos("red_cube")[2] > 0.4
    assert env.body_pos("target_zone")[2] > 0.4


def test_all_actuators_finite(env):
    assert np.all(np.isfinite(env.data.ctrl))
    assert env.model.actuator_ctrlrange.shape[0] == env.model.nu


# ------------------------------------------------ 运动学
def test_ik_accuracy(env):
    kin = PandaKinematics(env)
    target = np.array([0.62, -0.16, 0.60])
    res = kin.solve_ik(target, down_grasp_rotation(0.0), q_seed=kin.q())
    assert res.pos_err < 5e-4, f"IK 位置误差 {res.pos_err*1000:.3f} mm"
    assert kin.within_limits(res.q)


def test_joint_limit_detection(env):
    kin = PandaKinematics(env)
    bad = kin.q().copy()
    bad[0] = 10.0
    assert not kin.within_limits(bad)
    assert kin.limit_violation(bad) > 0


def test_grasp_rotation_is_rotation():
    R = down_grasp_rotation(0.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert R[2, 2] < -0.9, "抓取姿态应朝下"


# ------------------------------------------------ 语言
def test_parse_full_instruction():
    spec = parse_instruction("抓起桌面上的红色方块，放到蓝色目标区域。")
    assert spec.parser == "rule-based-zh-v1"
    assert spec.llm is False
    assert spec.action in ("pick", "move")
    assert spec.target.color == "red" and spec.target.category == "cube"
    assert spec.destination.color == "blue"
    assert spec.destination.category == "target_area"
    assert spec.needs_place


def test_parse_place_only_instruction():
    spec = parse_instruction("把黄色盒子放到蓝色目标区域。")
    assert spec.action == "move"
    assert spec.target.color == "yellow"


def test_parse_pick_only_instruction():
    spec = parse_instruction("抓起红色方块。")
    assert spec.action == "pick"
    assert spec.destination is None
    assert not spec.needs_place


def test_parse_unparsable_instruction():
    spec = parse_instruction("今天天气不错")
    assert spec.action == "unknown"
    assert spec.confidence == 0.0
    assert spec.warnings


# ------------------------------------------------ 场景图
def _node(name, pos, size=(0.05, 0.05, 0.05), category="cube", color="red",
          graspable=True):
    return SceneNode(node_id=f"obj:{name}", name=name, label=f"{color}_{category}",
                     category=category, color=color,
                     position=np.array(pos, float), size=np.array(size, float),
                     graspable=graspable)


def test_scene_graph_relations():
    g = SceneGraph()
    g.nodes["cube"] = _node("cube", [0.6, -0.15, 0.445])
    g.nodes["zone"] = _node("zone", [0.78, -0.10, 0.4212],
                            size=(0.14, 0.14, 0.004), category="region",
                            color="blue", graspable=False)
    g.nodes["obstacle"] = _node("obstacle", [0.60, -0.08, 0.47],
                                size=(0.07, 0.07, 0.10), color="yellow")
    g.rebuild_relations()
    rel = {(e.source, e.target, e.relation) for e in g.edges}
    # 方块与障碍物相距 7 cm -> near；区域在方块右侧
    assert ("cube", "obstacle", "near") in rel
    assert ("cube", "zone", "right_of") in rel
    assert ("obstacle", "zone", "above") in rel
    assert len(g.find(color="red", category="cube")) == 1
    assert g.nearest(color="blue") is not None
    assert g.is_reachable(g.nodes["cube"])


def test_scene_graph_history():
    g = SceneGraph()
    g.nodes["cube"] = _node("cube", [0.6, -0.15, 0.445])
    g.update_pose("cube", [0.7, -0.10, 0.445], source="test", timestamp=1.0)
    assert any(h["change"] == "pose" for h in g.history)
    g.set_state("cube", "placed", 2.0)
    assert g.nodes["cube"].state == "placed"


# ------------------------------------------------ 规划
def test_planner_reports_missing_target():
    g = SceneGraph()
    g.nodes["zone"] = _node("zone", [0.78, -0.10, 0.4212],
                            size=(0.14, 0.14, 0.004), category="region",
                            color="blue", graspable=False)
    spec = parse_instruction("抓起红色方块，放到蓝色目标区域。")
    plan = TaskPlanner().plan(spec, g)
    assert plan.ok is False
    assert plan.failure_node == "resolve_target"


def test_planner_builds_skill_sequence():
    g = SceneGraph()
    g.nodes["cube"] = _node("cube", [0.62, -0.16, 0.446])
    g.nodes["zone"] = _node("zone", [0.78, -0.10, 0.4212],
                            size=(0.14, 0.14, 0.004), category="region",
                            color="blue", graspable=False)
    g.rebuild_relations()
    spec = parse_instruction("抓起红色方块，放到蓝色目标区域。")
    plan = TaskPlanner().plan(spec, g)
    assert plan.ok, plan.reason
    skills = [n.skill for n in plan.nodes]
    assert skills[0] == "open_gripper"
    assert "close_gripper" in skills and "verify" in skills
    assert "no neural" in plan.planner or "symbolic" in plan.planner


# ------------------------------------------------ 接触
def test_contact_sensor_idle(env):
    env.reset(settle_steps=50)
    touch = ContactSensor(env)
    rep = touch.grasp_report("red_cube")
    assert rep.in_contact is False
    assert "simulated" in rep.source.lower()
