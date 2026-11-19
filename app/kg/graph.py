"""中医知识图谱（内存版）：症状 → 证型 → 方剂 → 中药，另含配伍禁忌。

数据来自 data/kg/*.csv。当前用纯 Python 内存实现，零外部依赖；
数据量增大后可平滑替换为图数据库，对外接口（本类的方法）保持不变。
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from app.config import settings


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# 口语 → 图谱标准症状名（只收图谱里确实存在的标准名；可按需继续补充）
_ALIASES = [
    ("喉咙痛", "咽痛"), ("喉咙疼", "咽痛"), ("嗓子疼", "咽痛"), ("嗓子痛", "咽痛"), ("喉痛", "咽痛"),
    ("干咳", "咳嗽"), ("咳了", "咳嗽"), ("口渴", "口干"), ("嘴巴干", "口干"),
]


class KnowledgeGraph:
    def __init__(self, kg_dir: Path | None = None) -> None:
        d = kg_dir or settings.kg_dir
        self.nodes: dict[str, dict] = {r["id"]: r for r in _read(d / "nodes.csv")}
        self.edges = _read(d / "edges.csv")
        self.herb_table = {r["中文药草名称"]: r for r in _read(d / "herbs.csv")}
        self.incompatible = [
            (r["source"], r["target"], r["来源"]) for r in _read(d / "incompatible.csv")
        ]

        self.symptom_by_name: dict[str, str] = {}
        self.symptom_to_syndromes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.symptom_to_formulas: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.syndrome_symptoms: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.syndrome_formulas: dict[str, list[str]] = defaultdict(list)
        self.formula_herbs: dict[str, list[str]] = defaultdict(list)

        for nid, n in self.nodes.items():
            if n["标签"] == "症状":
                self.symptom_by_name[n["名称"]] = nid

        for e in self.edges:
            s, t, typ = e["source"], e["target"], e["type"]
            w = float(e.get("weight") or 1.0)
            sl, tl = self.nodes[s]["标签"], self.nodes[t]["标签"]
            if (sl, typ, tl) == ("症状", "关联", "证型"):
                self.symptom_to_syndromes[s].append((t, w))
                self.syndrome_symptoms[t].append((s, w))
            elif (sl, typ, tl) == ("证型", "治疗", "方剂"):
                self.syndrome_formulas[s].append(t)
            elif (sl, typ, tl) == ("症状", "治疗", "方剂"):
                self.symptom_to_formulas[s].append((t, w))
            elif (sl, typ, tl) == ("方剂", "包含", "中药"):
                self.formula_herbs[s].append(t)

        # 症状名按长度降序，保证“白痰”优先于“痰”这类较短词
        self._symptom_names = sorted(self.symptom_by_name, key=len, reverse=True)

    # ------------------------------------------------------------ 统计
    def stats(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            counts[n["标签"]] += 1
        return {
            "symptoms": counts["症状"],
            "syndromes": counts["证型"],
            "formulas": counts["方剂"],
            "herbs": len(self.herb_table),
            "edges": len(self.edges),
            "incompatible_pairs": len(self.incompatible),
        }

    # ------------------------------------------------------------ 症状识别
    def match_symptoms(self, text: str) -> list[str]:
        """从自由文本中找出图谱已收录的症状名（子串匹配，长词优先、不重叠）。"""
        text = "".join(text.split())
        for alias, std in _ALIASES:          # 口语说法 → 图谱里的标准症状名
            text = text.replace(alias, std)
        taken = [False] * len(text)
        found: list[str] = []
        for name in self._symptom_names:
            pos = text.find(name)
            if pos < 0:
                continue
            if any(taken[pos : pos + len(name)]):
                continue
            for i in range(pos, pos + len(name)):
                taken[i] = True
            found.append(name)
        return found

    # ------------------------------------------------------------ 证型推断
    def find_syndromes(self, symptom_names: list[str], top_k: int = 5) -> list[dict]:
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)
        for name in symptom_names:
            sid = self.symptom_by_name.get(name)
            for syn_id, w in self.symptom_to_syndromes.get(sid, []):
                scores[syn_id] += w
                matched[syn_id].append(name)

        results = []
        for syn_id, raw in scores.items():
            all_syms = self.syndrome_symptoms[syn_id]
            total = sum(w for _, w in all_syms) or 1.0
            norm = raw / math.sqrt(total)
            results.append((syn_id, raw, norm, total))
        results.sort(key=lambda x: (-x[2], -x[1]))

        top = results[:top_k]
        best = top[0][2] if top else 1.0
        out = []
        for syn_id, raw, norm, total in top:
            typical = [self.nodes[s]["名称"] for s, _ in
                       sorted(self.syndrome_symptoms[syn_id], key=lambda x: -x[1])]
            out.append({
                "id": syn_id,
                "name": self.nodes[syn_id]["名称"],
                "score": round(norm / best, 3) if best else 0,
                "coverage": round(raw / total, 3),
                "matched_symptoms": matched[syn_id],
                "typical_symptoms": typical,
                "formulas": [self.formula_detail(f) for f in self.syndrome_formulas.get(syn_id, [])],
            })
        return out

    def direct_formulas(self, symptom_names: list[str], top_k: int = 3) -> list[dict]:
        """症状直达方剂（图谱中“症状—治疗→方剂”的补充路径）。"""
        scores: dict[str, float] = defaultdict(float)
        for name in symptom_names:
            sid = self.symptom_by_name.get(name)
            for fid, w in self.symptom_to_formulas.get(sid, []):
                scores[fid] += w
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [dict(self.formula_detail(fid), via_score=round(sc, 2)) for fid, sc in ranked]

    # ------------------------------------------------------------ 方剂 / 中药
    def formula_detail(self, formula_id: str) -> dict:
        herbs = []
        for hid in self.formula_herbs.get(formula_id, []):
            h = self.nodes[hid]
            name = h["名称"]
            extra = self.herb_table.get(name, {})
            herbs.append({
                "name": name,
                "dose_g": _to_float(h.get("剂量")),
                "nature": norm_nature(h.get("草本属性") or extra.get("草本属性", "")),
                "meridian": norm_meridian(h.get("草本经络热带") or extra.get("草本经络热带", "")),
                "flavor": norm_flavor(h.get("香草味") or extra.get("香草味", "")),
            })
        conflicts = self.check_incompatible([h["name"] for h in herbs])
        return {
            "id": formula_id,
            "name": self.nodes[formula_id]["名称"],
            "herbs": herbs,
            "conflicts": conflicts,
        }

    def check_incompatible(self, herb_names: list[str]) -> list[dict]:
        """十八反 / 十九畏检查：药名含禁忌药名即视为命中（如“炙甘草”含“甘草”）。"""
        alias = {"白芍": "芍药", "赤芍": "芍药", "红参": "人参", "生晒参": "人参", "白参": "人参",
                 "川贝母": "贝母", "浙贝母": "贝母", "土贝母": "贝母", "全瓜蒌": "瓜蒌",
                 "瓜蒌皮": "瓜蒌", "瓜蒌仁": "瓜蒌", "栝楼": "瓜蒌", "制川乌": "川乌", "制草乌": "草乌"}
        norm = []
        for n in herb_names:
            base = self.base_name(n)
            norm.append((n, alias.get(base, base)))
        hits, seen = [], set()
        for a, b, src in self.incompatible:
            na = [orig for orig, nn in norm if a in nn or a in orig]
            nb = [orig for orig, nn in norm if b in nn or b in orig]
            for x in na:
                for y in nb:
                    if x == y:
                        continue
                    key = (frozenset((x, y)), src)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append({"a": x, "b": y, "rule": src})
        return hits

    # ------------------------------------------------------------ 药材查询
    PREP_CHARS = "生炙制炒酒姜蜜醋煅煨炮焦麸盐"

    def herb_names(self) -> set[str]:
        if not hasattr(self, "_herb_names"):
            names = set(self.herb_table)
            names |= {n["名称"] for n in self.nodes.values() if n["标签"] == "中药"}
            self._herb_names = {n for n in names if n}
        return self._herb_names

    def base_name(self, name: str) -> str:
        """归并炮制品：炙甘草→甘草、姜半夏→半夏；去前缀后不在库中则保持原名。"""
        known = self.herb_names()
        for i in range(1, min(3, len(name) - 1) + 1):
            if all(c in self.PREP_CHARS for c in name[:i]) and name[i:] in known:
                return name[i:]
        return name

    def herb_profile(self, name: str) -> dict | None:
        base = self.base_name(name)
        if base not in self.herb_names():
            return None
        row = self.herb_table.get(base, {})
        node = next((n for n in self.nodes.values()
                     if n["标签"] == "中药" and n["名称"] == base and n.get("草本属性")), {})
        return {
            "name": base,
            "category": row.get("草本类型", ""),
            "nature": norm_nature(row.get("草本属性") or node.get("草本属性", "")),
            "meridian": norm_meridian(row.get("草本经络热带") or node.get("草本经络热带", "")),
            "flavor": norm_flavor(row.get("香草味") or node.get("香草味", "")),
        }

    def formulas_with_herb(self, name: str) -> list[dict]:
        base = self.base_name(name)
        out = []
        for fid, hids in self.formula_herbs.items():
            for hid in hids:
                h = self.nodes[hid]
                if self.base_name(h["名称"]) == base:
                    out.append({"name": self.nodes[fid]["名称"], "dose_g": _to_float(h.get("剂量"))})
                    break
        return out

    def taboo_partners(self, name: str) -> list[dict]:
        base = self.base_name(name)
        out = []
        for a, b, src in self.incompatible:
            if a in base or base in a:
                out.append({"herb": b, "rule": src})
            elif b in base or base in b:
                out.append({"herb": a, "rule": src})
        return out

    def search_herbs(self, q: str, limit: int = 12) -> list[str]:
        q = q.strip()
        names = sorted(self.herb_names(), key=lambda n: (not n.startswith(q), len(n)))
        return [n for n in names if q and q in n][:limit]

    # ------------------------------------------------------------ 浏览
    def list_syndromes(self) -> list[dict]:
        return [{"id": n["id"], "name": n["名称"],
                 "symptom_count": len(self.syndrome_symptoms.get(n["id"], [])),
                 "formulas": [self.nodes[f]["名称"] for f in self.syndrome_formulas.get(n["id"], [])]}
                for n in self.nodes.values() if n["标签"] == "证型"]

    def syndrome_detail(self, syn_id: str) -> dict | None:
        n = self.nodes.get(syn_id)
        if not n or n["标签"] != "证型":
            return None
        syms = sorted(self.syndrome_symptoms.get(syn_id, []), key=lambda x: -x[1])
        return {"id": syn_id, "name": n["名称"],
                "symptoms": [self.nodes[s]["名称"] for s, _ in syms],
                "formulas": [self.formula_detail(f) for f in self.syndrome_formulas.get(syn_id, [])]}

    def list_formulas(self) -> list[dict]:
        return [self.formula_detail(n["id"]) for n in self.nodes.values() if n["标签"] == "方剂"]


# 药材属性数据源为机器翻译，部分词汇需还原为中医术语（原始 CSV 不改动，读取时规范化）
_NATURE_MAP = {"温暖": "温", "温和": "微温", "冷": "寒", "微冷": "微寒", "甚至": "平",
               "非常热": "大热", "不适用": ""}
_FLAVOR_MAP = {"甜": "甘", "微甜": "微甘", "辛辣": "辛", "刺鼻": "辛", "平淡": "淡",
               "咸味": "咸", "苦味": "苦", "涩味": "涩", "酸味": "酸", "苦涩": "苦、涩",
               "不适用": ""}


def norm_nature(v: str) -> str:
    return _NATURE_MAP.get((v or "").strip(), (v or "").strip())


def norm_flavor(v: str) -> str:
    out = []
    for part in (v or "").replace("，", "、").replace(",", "、").split("、"):
        part = _FLAVOR_MAP.get(part.strip(), part.strip())
        if part and part not in out:
            out.append(part)
    return "、".join(out)


def norm_meridian(v: str) -> str:
    v = (v or "").replace("其他经络", "").replace("胆囊", "胆").replace("经络", "经")
    parts = [p.strip() for p in v.replace(",", "，").split("，") if p.strip()]
    return "、".join(dict.fromkeys(parts))


def _to_float(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


@lru_cache(maxsize=1)
def get_graph() -> KnowledgeGraph:
    return KnowledgeGraph()
