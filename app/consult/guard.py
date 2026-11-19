"""报告防编造：只允许模型使用患者实际提供的信息。

做三件事：
1. missing_info()   —— 列出患者没提供的关键信息，作为“待追问项”交给模型，而不是让它自己补；
2. is_thin()        —— 信息过少时，辨证一节改写成“已掌握 / 倾向 / 待补充”，不写成完整诊断；
3. audit()/strip()  —— 生成后逐句检查：体温、脉象、查体、发病时间、病历叙述、未提供的症状、
                       与舌象不符的舌苔描述，命中就让模型重写一次，仍不行就把该句删掉。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- 待追问项
_DIMS = [
    ("寒热（是否怕冷、发热、体温多少）", r"恶寒|畏寒|怕冷|怕风|发热|发烧|体温|寒热|低热|高热"),
    ("局部特点（红肿、干痒、吞咽时是否更痛）", r"红肿|咽干|咽痒|扁桃体|干痛|灼热|吞咽|胀|刺痛|隐痛"),
    ("咳嗽咳痰（有无、痰色）", r"咳|痰"),
    ("口渴与饮水（喜冷饮还是热饮）", r"口渴|口干|喜饮|不渴|喜热饮|喜冷饮"),
    ("二便（大便干稀、小便颜色）", r"大便|小便|便秘|腹泻|溏|尿"),
    ("食欲与睡眠", r"食欲|纳差|胃口|睡眠|失眠|睡不"),
    ("脉象（如能自测或由医师提供）", r"脉"),
]


def patient_blob(p: dict) -> str:
    return " ".join(str(p.get(k) or "") for k in ("complaint", "duration", "history", "age", "gender"))


def missing_info(patient: dict, limit: int = 5) -> list[str]:
    blob = patient_blob(patient)
    return [name for name, pat in _DIMS if not re.search(pat, blob)][:limit]


def inquiry_status(patient: dict) -> tuple[list[str], list[str]]:
    """问诊要点里已提到 / 未提到的项目（按文字里是否出现相关词判断，用于报告的“四诊信息”表）。"""
    blob = patient_blob(patient)
    got = [name for name, pat in _DIMS if re.search(pat, blob)]
    return got, [name for name, _ in _DIMS if name not in got]


# ---------------------------------------------------------------- 信息充分性闸门
# 参照中医门诊病历的要素：主诉、现病史（病程）、四诊（寒热等问诊信息、舌象、脉象）。
# 必备项缺任何一项，就不生成辨证报告，只提示补充；脉象为建议项（线上难以获取），缺则在报告里写“未提供”。
_DURATION = re.compile(r"\d|[一二两三四五六七八九十半几数多]+\s*(?:天|日|周|个?月|年|小时|分钟)|反复|间断|持续|突然|晨起|昨|今|前天")
_COLD_HEAT = r"恶寒|畏寒|怕冷|怕风|发热|发烧|体温|寒热|低热|高热|不发热|没发烧|无发热|手脚凉|手足凉|身热"
_ASSOC = [
    ("咳嗽咳痰", r"咳|痰"),
    ("口渴或口干", r"口渴|口干|喜饮|不渴|喜热饮|喜冷饮"),
    ("大小便", r"大便|小便|便秘|腹泻|溏|尿|拉肚子"),
    ("食欲或睡眠", r"食欲|纳差|胃口|睡眠|失眠|睡不|吃不下"),
    ("出汗情况", r"汗"),
    ("局部特点（红肿、干痒、吞咽时疼痛、胀痛或刺痛等）", r"红肿|干痒|咽干|咽痒|扁桃体|干痛|灼热|吞咽|胀|刺痛|隐痛|冷痛|绞痛|跳痛"),
]
_RED_EXTRA = ["吞咽困难", "吞咽很困难", "呼吸困难", "喘不上气", "高热不退", "持续高烧", "颈部僵硬", "嘴唇发紫", "喉头水肿", "喘鸣"]


def assess(patient: dict, has_tongue: bool, kg=None) -> dict:
    """判断信息是否足够出具辨证报告。返回 {"ready", "have", "need", "suggest"}。"""
    blob = patient_blob(patient)
    complaint = (patient.get("complaint") or "").strip()
    duration = (patient.get("duration") or "").strip()
    have: list[dict] = []
    need: list[str] = []

    if complaint:
        have.append({"item": "主诉", "detail": complaint[:40]})
    else:
        need.append("主诉（最困扰的症状、部位）")
    if duration or _DURATION.search(complaint):
        have.append({"item": "病程", "detail": duration or "主诉中已提及"})
    else:
        need.append("病程（起病多久、是否在加重）")
    if re.search(_COLD_HEAT, blob):
        have.append({"item": "寒热情况", "detail": "已描述"})
    else:
        need.append("寒热情况（是否怕冷、发热，体温多少；没有也请写“无”）")
    if has_tongue or "舌" in blob:
        have.append({"item": "舌象", "detail": "已上传或已描述"})
    else:
        need.append("舌象（上传清晰舌象照片，或描述舌质、舌苔颜色厚薄）")
    got = [n for n, pat in _ASSOC if re.search(pat, blob)]
    if len(got) >= 2:
        have.append({"item": "伴随症状", "detail": "、".join(got)})
    else:
        left = [n for n, _ in _ASSOC if n not in got]
        need.append(f"伴随症状至少 2 项（目前 {len(got)} 项），可从这些里补充：" + "；".join(left[:4]))
    suggest = [] if re.search(r"脉", blob) else ["脉象（如有医师或自测结果）"]
    return {"ready": not need, "have": have, "need": need, "suggest": suggest}


def red_flags(patient: dict) -> list[str]:
    """红旗症状：与问诊模块共用同一套关键词，再补充咽喉/呼吸方面的几项。"""
    from app.consult.dialogue import GENERIC_CATEGORY, _redflags
    blob = patient_blob(patient)
    out = _redflags(blob, GENERIC_CATEGORY)
    if any(k in blob for k in _RED_EXTRA):
        out.append("提到了吞咽困难、呼吸困难或高热不退等情况，可能比较紧急，请尽快到医院就诊")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- 依据集合
class Support:
    """患者输入、舌象结论、图谱命中的症状：报告里只有出现在这里的内容才算“有依据”。"""

    def __init__(self, patient: dict, tongue: dict | None, symptoms: list[str], kg) -> None:
        self.kg = kg
        self.blob = patient_blob(patient)
        self.tongue_label = (tongue or {}).get("label", "") if tongue else ""
        self.symptoms = set(symptoms) | set(kg.match_symptoms(self.blob + self.tongue_label))
        self.tongue_chars = set(re.findall(r"[白黄薄厚腻红黑灰紫淡青]", self.tongue_label + self._patient_tongue()))

    def _patient_tongue(self) -> str:
        return "".join(re.findall(r"[舌苔][^，。；、\s]{0,8}", self.blob))


# ---------------------------------------------------------------- 检查规则
_TEMP = re.compile(r"\d{2}(?:\.\d)?\s*(?:℃|°C|度)")
_PULSE = re.compile(r"脉[浮沉迟数滑弦细紧缓濡涩洪]+")
_EXAM = re.compile(r"听诊|查体|体格检查|触诊|视诊|扁桃体|淋巴结|血常规|化验|检查结果|血压|心率")
_WHEN = re.compile(r"昨日|昨天|今晨|今早|前天|前日|上周|昨晚|夜间")
_DOCTOR = re.compile(r"医生(为|对|询问|还询问|进行|首先|随后)|遂就诊|就诊我院|来我院")
_TONGUE = re.compile(r"(?:舌(?:质|苔|象)?|苔)[^，。；、：\s]{0,8}")
_EXEMPT = re.compile(r"如[果有]?|若|倘|一旦|是否|有无|询问|追问|补充|进一步|常见|多见|典型|通常|一般|可见于|多表现|警惕|及时|立即|尽快|预警|就医|急诊")
_SPLIT = re.compile(r"(?<=[。！？\n])")


def _supported_in(blob: str, token: str) -> bool:
    return token in blob


def audit(title: str, text: str, sup: Support) -> list[dict]:
    """返回有问题的句子：[{"sentence": ..., "reasons": [...]}]。"""
    bad = []
    for sent in _SPLIT.split(text):
        s = sent.strip()
        if not s:
            continue
        why: list[str] = []
        if _DOCTOR.search(s):
            why.append("编造的问诊/就诊叙述")
        exempt = bool(_EXEMPT.search(s))
        if not exempt:
            for m in _TEMP.finditer(s):
                if not _supported_in(sup.blob, m.group(0)):
                    why.append(f"体温「{m.group(0)}」未提供")
            for pat, name in ((_PULSE, "脉象"), (_EXAM, "检查"), (_WHEN, "发病时间")):
                for m in pat.finditer(s):
                    if not _supported_in(sup.blob, m.group(0)):
                        why.append(f"{name}「{m.group(0)}」未提供")
            for m in _TONGUE.finditer(s):
                chars = set(re.findall(r"[白黄薄厚腻红黑灰紫淡青]", m.group(0)))
                extra = chars - sup.tongue_chars
                if extra:
                    why.append(f"舌象描述「{m.group(0)}」与舌象结论「{sup.tongue_label or '未上传'}」不符")
            if title.startswith("辨证"):
                extra = [x for x in sup.kg.match_symptoms(s) if x not in sup.symptoms]
                if extra:
                    why.append("未提供的症状：" + "、".join(extra))
        if why:
            bad.append({"sentence": s, "reasons": list(dict.fromkeys(why))})
    return bad


def strip(text: str, bad: list[dict]) -> str:
    """删掉有问题的句子，并清理只剩标题、没有内容的行。"""
    drop = {b["sentence"] for b in bad}
    kept = [seg for seg in _SPLIT.split(text) if seg.strip() and seg.strip() not in drop]
    lines = "".join(kept).split("\n")
    out: list[str] = []
    for i, ln in enumerate(lines):
        if re.fullmatch(r"\s*[\u4e00-\u9fa5（）()、]{2,12}[:：]\s*", ln):
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not nxt or re.fullmatch(r"\s*[\u4e00-\u9fa5（）()、]{2,12}[:：]\s*", nxt):
                continue
        out.append(ln)
    return "\n".join(out).strip()


def correction_note(bad: list[dict]) -> str:
    items = "\n".join(f"- 「{b['sentence'][:40]}…」：{'；'.join(b['reasons'])}" for b in bad[:6])
    return ("\n\n【上一稿不合格，必须重写】上一稿出现了患者没有提供、或与舌象结论矛盾的内容：\n" + items +
            "\n请整节重写：删除这些内容；缺少的信息写成“需进一步询问”，不要假设；"
            "不要用“患者自述/医生检查/听诊”之类的病历叙述。")
