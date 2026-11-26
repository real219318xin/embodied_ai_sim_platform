"""中文自然语言指令解析 —— **规则系统**（不是大语言模型）。

声明
----
* 本模块用正则 + 词典匹配解析有限的中文指令模式；
* 没有接入任何大模型、没有模型权重、没有 API 调用；
* 解析结果里的 `parser` 字段固定为 "rule-based-zh-v1"，
  `llm` 字段固定为 False，导出数据时一并保留。

支持的句式（可扩展词典）
------------------------
    抓起桌面上的红色方块，放到蓝色目标区域。
    把红色方块放进蓝色目标区域。
    拿黄色盒子放到蓝色目标区域。
    抓起红色方块。                （只抓取，不放置）
    移动红色方块到蓝色目标区域。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PARSER_NAME = "rule-based-zh-v1"

COLOR_WORDS: Dict[str, str] = {
    "红色": "red", "红": "red",
    "蓝色": "blue", "蓝": "blue",
    "黄色": "yellow", "黄": "yellow",
    "绿色": "green", "绿": "green",
    "橙色": "orange", "橙": "orange",
    "紫色": "purple", "紫": "purple",
    "灰色": "gray", "灰": "gray",
    "白色": "white", "白": "white",
}

CATEGORY_WORDS: Dict[str, str] = {
    "目标区域": "target_area", "目标区": "target_area", "区域": "target_area",
    "立方体": "cube", "正方体": "cube", "方块": "cube", "方塊": "cube",
    "盒子": "box", "箱子": "box", "障碍物": "box", "块": "cube",
}

ACTION_WORDS = [
    (("抓起", "抓", "拿起", "拿", "取", "拾起", "夹起", "夹"), "pick"),
    (("移动", "搬", "搬运"), "move"),
    (("推到", "推"), "push"),
    (("打开",), "open"),
    (("关闭",), "close"),
]

PLACE_MARKERS = ("放到", "放在", "放入", "放进", "置于", "移至", "移到", "至", "到")
# 仅用于"动作识别"的放置词（不含单字"到/至"，避免"找到"之类的误判）
PLACE_ACTION_MARKERS = ("放到", "放在", "放入", "放进", "置于", "移至", "移到")
STOPWORDS = ("桌面上的", "桌上的", "桌面上", "桌上", "地上的", "地面上的", "那个",
             "这个", "请", "帮我", "把", "将", "然后", "，", "。", "、", " ", "\n")


@dataclass
class EntityRef:
    text: str
    color: Optional[str] = None
    category: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {"text": self.text, "color": self.color, "category": self.category}


@dataclass
class TaskSpec:
    instruction: str
    action: str
    target: Optional[EntityRef] = None
    destination: Optional[EntityRef] = None
    constraints: List[str] = field(default_factory=list)
    parser: str = PARSER_NAME
    llm: bool = False
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)

    @property
    def needs_place(self) -> bool:
        return self.action in ("place", "move") or self.destination is not None

    def as_dict(self) -> Dict[str, object]:
        return {
            "instruction": self.instruction,
            "action": self.action,
            "target": self.target.as_dict() if self.target else None,
            "destination": self.destination.as_dict() if self.destination else None,
            "constraints": list(self.constraints),
            "needs_place": self.needs_place,
            "parser": self.parser,
            "llm": bool(self.llm),
            "confidence": round(float(self.confidence), 3),
            "warnings": list(self.warnings),
            "unmatched": list(self.unmatched),
        }


def _find_all(text: str, words: Dict[str, str]) -> List[tuple]:
    """返回 [(start, end, key, value)]，长词优先，避免"红色"被"红"抢先匹配。"""
    hits = []
    taken = [False] * len(text)
    for w in sorted(words, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            if not any(taken[i:i + len(w)]):
                hits.append((i, i + len(w), w, words[w]))
                for k in range(i, i + len(w)):
                    taken[k] = True
            start = i + 1
    return sorted(hits)


def _detect_action(text: str) -> Optional[str]:
    for words, action in ACTION_WORDS:
        for w in words:
            if w in text:
                return action
    # 没有显式抓取动词但出现"放到/放进…"时，按"抓取-放置"任务解析
    for w in PLACE_ACTION_MARKERS:
        if w in text:
            return "move"
    return None


def _entity_from_span(text: str, span: str) -> EntityRef:
    colors = [v for _, _, _, v in _find_all(span, COLOR_WORDS)]
    cats = [v for _, _, _, v in _find_all(span, CATEGORY_WORDS)]
    cleaned = span
    for s in STOPWORDS:
        cleaned = cleaned.replace(s, "")
    return EntityRef(text=cleaned.strip(), color=colors[0] if colors else None,
                     category=cats[0] if cats else None)


def parse_instruction(instruction: str) -> TaskSpec:
    text = (instruction or "").strip()
    spec = TaskSpec(instruction=text, action="unknown")
    if not text:
        spec.warnings.append("指令为空")
        return spec

    action = _detect_action(text)
    if action is None:
        spec.action = "unknown"
        spec.warnings.append("未识别到动作词（抓/放/移动…）")
        spec.unmatched = [text]
        return spec
    spec.action = action

    if any(k in text for k in ("避开", "绕过", "不要碰")):
        spec.constraints.append("avoid_obstacle")
    if any(k in text for k in ("小心", "轻放", "缓慢")):
        spec.constraints.append("gentle")

    # 用放置标记切分：前半段是被操作物体，后半段是目的地
    marker_pos = -1
    marker_len = 0
    for m in PLACE_MARKERS:
        i = text.find(m)
        if i > 0 and (marker_pos < 0 or i < marker_pos):
            marker_pos, marker_len = i, len(m)
    # "抓起" 里的"抓"不会与放置标记冲突；"放到"优先级由 find 顺序保证

    if marker_pos > 0:
        head, tail = text[:marker_pos], text[marker_pos + marker_len:]
        spec.target = _entity_from_span(text, head)
        spec.destination = _entity_from_span(text, tail)
        if spec.destination.category is None and spec.destination.color is not None:
            spec.warnings.append("目的地未给出类别，按"
                                 "『该颜色的放置区域』理解")
            spec.destination.category = "target_area"
    else:
        spec.target = _entity_from_span(text, text)
        if action == "pick":
            spec.destination = None
        else:
            spec.warnings.append("指令含放置类动作但未找到目的地")

    if spec.target is None or (spec.target.color is None and spec.target.category is None):
        spec.warnings.append("未识别到目标物体（颜色/类别缺失）")

    # 置信度：动作 + 目标 + 目的地齐全程度（规则打分，非概率模型）
    score = 0.4
    if spec.target and (spec.target.color or spec.target.category):
        score += 0.3
    if spec.destination and spec.destination.category:
        score += 0.25
    if not spec.warnings:
        score += 0.05
    spec.confidence = min(1.0, score)
    return spec
