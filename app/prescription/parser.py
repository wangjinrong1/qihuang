"""药方文本解析：提取药名、剂量、煎服备注，并做配伍禁忌检查与方剂比对。

支持常见写法：
  柴胡12g 黄芩9克 半夏9g(先煎)
  柴胡 12 / 黄芩（9g）/ 炙甘草6
  顿号、逗号、分号、换行、空格均可作分隔
"""
from __future__ import annotations

import re

from app.kg.graph import get_graph

_FULL = str.maketrans("０１２３４５６７８９．（）：，；", "0123456789.():,;")
_CHUNK = re.compile(
    r"^(?P<name>[\u4e00-\u9fa5]{1,10}?)[\s(:：]*(?P<dose>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>克|g|G|钱|两|枚|片|个)?[)）]?(?P<note>.*)$"
)
_NOTE_WORDS = ("先煎", "后下", "包煎", "另煎", "烊化", "冲服", "打碎", "捣碎", "研末")


def normalize(text: str) -> str:
    text = text.translate(_FULL)
    text = re.sub(r"(?<=[\u4e00-\u9fa5])[ \t]+(?=\d)", "", text)   # 柴胡 12 → 柴胡12
    return text


def _resolve(name: str, kg) -> tuple[str, str] | None:
    """返回 (显示名, 基础药名)；识别不了返回 None。"""
    known = kg.herb_names()
    if name in known or kg.base_name(name) in known:
        return name, kg.base_name(name)
    for i in range(1, len(name) - 1):           # 去掉前面混入的字（如“方药柴胡”）
        tail = name[i:]
        if len(tail) >= 2 and (tail in known or kg.base_name(tail) in known):
            return tail, kg.base_name(tail)
    return None


def parse(text: str) -> dict:
    kg = get_graph()
    herbs, unknown = [], []
    for raw in re.split(r"[,;、:：\n\r\t ]+", normalize(text)):
        raw = raw.strip("。.：: ")
        if not raw:
            continue
        m = _CHUNK.match(raw)
        if m:
            name, dose, unit, tail = m["name"], float(m["dose"]), m["unit"], m["note"]
        else:
            name, dose, unit, tail = raw, None, None, ""
            if not re.fullmatch(r"[\u4e00-\u9fa5]{2,10}", name):
                continue
        res = _resolve(name, kg)
        if not res:
            if dose is not None:
                unknown.append(raw)
            continue
        disp, base = res
        note = "".join(w for w in _NOTE_WORDS if w in tail) or ""
        dose_g, approx = None, False
        if dose is not None:
            if unit in (None, "g", "G", "克"):
                dose_g = dose
            elif unit == "钱":
                dose_g, approx = dose * 3, True
        prof = kg.herb_profile(base) or {}
        herbs.append({
            "name": disp, "base": base, "dose": dose, "unit": unit or ("g" if dose is not None else ""),
            "dose_g": dose_g, "dose_approx": approx, "unit_assumed": dose is not None and unit is None,
            "note": note, "nature": prof.get("nature", ""), "meridian": prof.get("meridian", ""),
            "flavor": prof.get("flavor", ""), "category": prof.get("category", ""),
        })
    conflicts = kg.check_incompatible([h["name"] for h in herbs])
    return {"herbs": herbs, "unknown": unknown, "conflicts": conflicts,
            "matches": match_formulas([h["base"] for h in herbs])}


def match_formulas(bases: list[str], top_k: int = 3, min_sim: float = 0.25) -> list[dict]:
    kg = get_graph()
    mine = set(bases)
    if not mine:
        return []
    out = []
    for n in kg.nodes.values():
        if n["标签"] != "方剂":
            continue
        comp = {kg.base_name(kg.nodes[h]["名称"]) for h in kg.formula_herbs.get(n["id"], [])}
        if not comp:
            continue
        inter, union = mine & comp, mine | comp
        sim = len(inter) / len(union)
        if sim >= min_sim:
            out.append({"name": n["名称"], "similarity": round(sim, 3),
                        "common": sorted(inter), "missing": sorted(comp - mine),
                        "extra": sorted(mine - comp)})
    out.sort(key=lambda x: -x["similarity"])
    return out[:top_k]
