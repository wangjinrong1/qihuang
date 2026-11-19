"""辨证诊疗流程（分阶段 + 流式事件）。

阶段：信息整理 → 舌象分析 → 图谱辨证 → 资料检索 → 分节生成 → 报告汇总
每个阶段都通过 yield 事件向前端汇报进度，后续要加“规划/审阅”等智能体环节时，
只需在 run() 里新增阶段即可。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from app.kg.graph import get_graph
from app.llm.engine import engine
from app.config import settings
from app.consult import guard
from app.llm.prompts import REPORT_SECTIONS, RULES, SECTION_TEMPLATE
from app.rag.retriever import get_retriever
from app.tongue import classifier

DISCLAIMER = "> ⚠️ 本报告由程序自动生成，仅供学习与辅助参考，不构成医疗建议。诊断与用药请咨询执业医师。"


# ---------------------------------------------------------------- 上下文文本
def patient_text(p: dict) -> str:
    rows = [("姓名", p.get("name")), ("性别", p.get("gender")), ("年龄", p.get("age")),
            ("主诉与症状", p.get("complaint")), ("病程", p.get("duration")),
            ("既往史/过敏史", p.get("history"))]
    return "\n".join(f"{k}：{v}" for k, v in rows if v) or "（未提供）"


def tongue_text(t: dict | None) -> str:
    if not t:
        return "（未上传舌象）"
    if t.get("demo"):
        return "（舌象为演示结果，不是真实识别，不得作为依据，也不要描述舌象）"
    return f"{t['label']}（置信度 {t['confidence']:.0%}）：{t['note']}"


def kg_text(syndromes: list[dict]) -> str:
    if not syndromes:
        return "（图谱未匹配到证型）"
    lines = []
    for s in syndromes[:3]:
        fs = "；".join(
            f"{f['name']}（{'、'.join(h['name'] for h in f['herbs'][:8])}…）" for f in s["formulas"][:2]
        ) or "无"
        lines.append(f"- 证型「{s['name']}」匹配度{s['score']:.0%}，命中症状：{'、'.join(s['matched_symptoms'])}；相关方剂：{fs}")
    return "\n".join(lines)


def rag_text(hits: list[dict]) -> str:
    return "\n".join(f"[{h['source']}] {h['text']}" for h in hits) or "（无）"


class Refs:
    """报告里的参考资料：全文统一编号，各小节引用同一份清单，重复段落只记一次。"""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self._seen: dict[tuple, int] = {}

    def add(self, hits: list[dict]) -> list[int]:
        ids = []
        for h in hits:
            key = (h["source"], h["text"][:30])
            if key not in self._seen:
                self.items.append(h)
                self._seen[key] = len(self.items)
            ids.append(self._seen[key])
        return ids

    def text(self, ids: list[int], max_chars: int = 220) -> str:
        return "\n".join(f"[{i}] 《{self.items[i - 1]['source']}》{self.items[i - 1]['text'][:max_chars]}"
                         for i in ids) or "（无）"


def section_queries(patient: dict, syndromes: list[dict], tongue: dict | None) -> list[str | None]:
    """每个小节各自的检索词（None=不检索）。主症 + 证型 + 该小节关注点，避免一条查询喂所有小节。"""
    complaint = (patient.get("complaint") or "")[:40]
    top = syndromes[0]["name"] if syndromes else ""
    base = (top or complaint).strip()
    out: list[str | None] = []
    for _, _, extra in REPORT_SECTIONS:
        if extra is None:
            out.append(None)
        elif extra == "":
            out.append(" ".join(x for x in (complaint, top, tongue["label"] if tongue else "") if x).strip() or None)
        else:
            out.append(f"{base} {extra}".strip() if base else None)
    return out


# ---------------------------------------------------------------- 名词速查（药材 / 穴位 → 搜索链接）
@lru_cache(maxsize=1)
def _acupoint_names() -> tuple[str, ...]:
    f = settings.web_dir / "atlas" / "points.js"
    if not f.exists():
        return ()
    return tuple(dict.fromkeys(re.findall(r"\['([^'\]]{2,6})','[A-Za-z]{1,3}[-\d]*[A-Za-z\d]*'", f.read_text(encoding="utf-8"))))


def _found(names, text: str, limit: int) -> list[str]:
    hit = sorted({n for n in names if len(n) >= 2 and n in text}, key=lambda n: (text.index(n), -len(n)))
    hit = [n for n in hit if not any(n != m and n in m for m in hit)]      # “甘草”被“炙甘草”包含就只留长的
    return hit[:limit]


def _search_url(name: str, suffix: str) -> str:
    return "https://www.baidu.com/s?wd=" + quote(f"{name} {suffix}")


def link_terms(text: str, limit: int = 10) -> dict[str, list[tuple[str, str]]]:
    """从报告文字里认出图谱里收录的药材和图册里的穴位，给搜索入口（只是搜索页，不是内容引用）。"""
    herbs = _found(get_graph().herb_names(), text, limit)
    points = _found(_acupoint_names(), text, limit)
    return {"药材": [(h, _search_url(h, "中药 功效")) for h in herbs],
            "穴位": [(p, _search_url(p, "穴位 定位")) for p in points]}


# ---------------------------------------------------------------- 主流程
def run(patient: dict, image_path: Path | None = None) -> Iterator[dict]:
    kg = get_graph()

    yield {"type": "stage", "stage": "collect", "label": "整理患者信息"}
    symptoms = kg.match_symptoms(patient.get("complaint", ""))
    yield {"type": "symptoms", "symptoms": symptoms}

    tongue = None
    if image_path:
        yield {"type": "stage", "stage": "tongue", "label": "舌象分析"}
        tongue = classifier.analyze(image_path)
        yield {"type": "tongue", "tongue": tongue}
    real_tongue = tongue if tongue and not tongue.get("demo") else None     # 演示结果不能当依据
    if real_tongue:
        symptoms = list(dict.fromkeys(symptoms + kg.match_symptoms(real_tongue["label"])))   # 舌象结论并入症状，参与后续检索

    flags = guard.red_flags(patient)
    if flags:
        yield {"type": "redflags", "items": flags}

    # ---- 依据闸门：信息不足就不出报告，只提示补充（参照中医门诊病历的要素）
    check = guard.assess(patient, real_tongue is not None, kg)
    if not check["ready"]:
        yield {"type": "insufficient", **check}
        yield {"type": "done"}
        return

    yield {"type": "stage", "stage": "kg", "label": "知识图谱辨证"}
    syndromes = kg.find_syndromes(symptoms, top_k=3)
    direct = kg.direct_formulas(symptoms, top_k=2)
    yield {"type": "syndromes", "syndromes": syndromes, "direct_formulas": direct}

    yield {"type": "stage", "stage": "rag", "label": "检索参考资料"}
    refs = Refs()
    retr = get_retriever()
    sec_ids: list[list[int]] = []
    for q in section_queries(patient, syndromes, real_tongue):
        sec_ids.append(refs.add(retr.search(q, k=3)) if q else [])
    yield {"type": "rag", "hits": refs.items}

    api = engine.info()["mode"] == "api"          # 云端大模型写得又快又长；本地小模型保持短小
    limit, max_tokens = (450, 900) if api else (200, 380)
    missing = guard.missing_info(patient)
    sup = guard.Support(patient, real_tongue, symptoms, kg)
    ctx = dict(patient=patient_text(patient), tongue=tongue_text(real_tongue or tongue), kg=kg_text(syndromes), limit=limit,
               rules=RULES, missing="；".join(missing) or "（主要方面均已提供）")
    sections: list[tuple[str, str]] = []
    demo = engine.info()["mode"] == "mock"
    for (title, hint, _), ids in zip(REPORT_SECTIONS, sec_ids):
        yield {"type": "stage", "stage": "section", "label": f"生成：{title}"}
        yield {"type": "section_start", "title": title}
        if demo:       # 演示模式没有真实模型：不输出占位的“辨证结论”，避免被当成真实分析
            note = "（演示模式：尚未接入模型，本节不生成内容。请在设置里接入模型后重新辨证。）"
            yield {"type": "token", "title": title, "text": note}
            sections.append((title, note))
            continue
        prompt = SECTION_TEMPLATE.format(title=title, hint=hint, rag=refs.text(ids), **ctx)
        buf: list[str] = []
        for tok in engine.stream(engine.build_messages(None, prompt), max_new_tokens=max_tokens):
            buf.append(tok)
            yield {"type": "token", "title": title, "text": tok}
        text = "".join(buf).strip()

        bad = guard.audit(title, text, sup)          # 防编造：不合格 → 重写一次 → 仍不合格就删句
        if bad:
            yield {"type": "section_reset", "title": title}
            buf = []
            for tok in engine.stream(engine.build_messages(None, prompt + guard.correction_note(bad)),
                                     max_new_tokens=max_tokens):
                buf.append(tok)
                yield {"type": "token", "title": title, "text": tok}
            text = "".join(buf).strip()
            bad = guard.audit(title, text, sup)
            if bad:
                text = guard.strip(text, bad) or "（该节生成内容含有患者未提供的信息，已自动删除，请补充症状后重新辨证。）"
                yield {"type": "section_replace", "title": title, "text": text}
        sections.append((title, text))

    yield {"type": "stage", "stage": "report", "label": "汇总报告"}
    kg_herbs = " ".join(h["name"] for sy in syndromes for f in sy["formulas"] for h in f["herbs"])
    links = link_terms("\n".join(t for _, t in sections) + "\n" + kg_herbs)
    report = build_report(patient, tongue, syndromes, refs.items, sections, links, flags, check)
    yield {"type": "report", "markdown": report}
    yield {"type": "done"}


# ---------------------------------------------------------------- 报告汇总
_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四"]


def _tongue_line(tongue: dict | None) -> str:
    if not tongue:
        return "未提供（未上传舌象，患者也未描述）"
    if tongue.get("demo"):
        return "演示结果，不是真实识别，**不作为辨证依据**"
    return f"{tongue['label']}（来源：图片识别，置信度 {tongue['confidence']:.0%}）——{tongue['note']}"


def build_report(patient, tongue, syndromes, hits, sections, links: dict | None = None,
                 flags: list[str] | None = None, check: dict | None = None) -> str:
    """报告结构参照中医门诊病历：一般资料→主诉→现病史→既往史→四诊→辅助检查→辨证分析→处置→医嘱→局限性。
    患者资料类内容全部原样取自输入，没有提供的写“未提供”，不由模型补写。"""
    p = patient
    asked, not_asked = guard.inquiry_status(p)
    md = ["# 中医辅助辨证报告", "", DISCLAIMER, ""]
    for f in flags or []:
        md += [f"> 🚨 **{f}**", ""]
    md += ["## 一、一般资料", "",
           f"姓名：{p.get('name') or '未提供'}　性别：{p.get('gender') or '未提供'}　年龄：{p.get('age') or '未提供'}", "",
           "## 二、主诉", "", p.get("complaint") or "未提供", "",
           "## 三、现病史", "",
           f"病程：{p.get('duration') or '未提供'}。其余症状见主诉；未提到的情况不作推测。", "",
           "## 四、既往史、过敏史", "", p.get("history") or "未提供", "",
           "## 五、四诊信息（仅记录已提供项）", "",
           "| 四诊 | 内容 |", "|---|---|",
           f"| 望诊 | 舌象：{_tongue_line(tongue)}；面色、形体、舌质舌下络脉等：未提供 |",
           "| 闻诊 | 未提供 |",
           f"| 问诊 | 已涉及：{'、'.join(a.split('（')[0] for a in asked if not a.startswith('脉')) or '仅主诉'}；"
           f"未涉及：{'、'.join(n.split('（')[0] for n in not_asked if not n.startswith('脉')) or '无'} |",
           f"| 切诊 | 脉象：{'见患者描述' if any(a.startswith('脉') for a in asked) else '未提供'} |", "",
           "## 六、体格检查与辅助检查", "", "未提供（本系统不包含检查结果，也不会编造）。", "",
           "## 七、知识图谱辨证参考", "", "图谱只按症状关键词匹配，匹配度不等于辨证结论。", ""]
    if not syndromes:
        md.append("图谱未能匹配到证型，依据不足，暂不能定证。")
    for i, sy in enumerate(syndromes, 1):
        md += [f"### {i}. {sy['name']}（匹配度 {sy['score']:.0%}）", "",
               f"- 命中症状：{'、'.join(sy['matched_symptoms'])}"]
        for f in sy["formulas"]:
            herbs = "、".join(f"{h['name']}{_dose(h)}" for h in f["herbs"])
            md.append(f"- 方剂 **{f['name']}**：{herbs}")
            for c in f["conflicts"]:
                md.append(f"  - ⚠️ 配伍禁忌（{c['rule']}）：{c['a']} 与 {c['b']}")
        md.append("")
    for i, (title, text) in enumerate(sections):
        md += [f"## {_NUM[7 + i]}、{title}", "", text, ""]
    n = 7 + len(sections)
    suggest = (check or {}).get("suggest", []) + [x for x in not_asked if not x.startswith("脉")]
    md += [f"## {_NUM[n]}、信息完整度与局限性", "",
           "- 依据来源：患者在系统中填写的文字、舌象图片识别结果、知识图谱、古籍资料；未经医师望闻问切四诊合参。",
           "- 报告中的诊断均为**倾向**，不是确诊；中医诊断应由执业医师结合面诊、脉诊和必要的检查作出。",
           f"- 建议补充：{'、'.join(dict.fromkeys(suggest)) or '无'}。补充后可重新生成报告。", ""]
    if hits:
        md += ["## 参考资料", "", "古籍原文经白话整理，正文中的[编号]对应下列条目：", ""]
        md += [f"{i}. 《{h['source']}》：{h['text'][:50].strip()}…" for i, h in enumerate(hits, 1)] + [""]
    if links and any(links.values()):
        md += ["## 名词速查", "", "以下链接只是搜索入口，网页内容请自行甄别，不构成用药或操作建议。", ""]
        for kind, items in links.items():
            if items:
                md.append(f"- {kind}：" + "、".join(f"[{n}]({u})" for n, u in items))
        md.append("")
    return "\n".join(md)


def _dose(h: dict) -> str:
    return f"{h['dose_g']:g}g" if h.get("dose_g") else ""
