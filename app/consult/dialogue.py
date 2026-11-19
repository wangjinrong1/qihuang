"""问诊对话状态机（参考灵枢智联的 Informer「先收集信息、够了再诊断」流程）。

为什么需要它：
    原来的智能问诊每一轮都是「检索 → 一次性要求模型给辨证+建议」，没有“信息够不够”的判断，
    所以用户只说一句“咳嗽”就会得到结论。灵枢的做法是把「收集信息」和「给出诊断」拆成两个阶段，
    由一个明确的结束条件切换。这里用规则（而不是让小模型自己判断）实现同样的逻辑：
    规则稳定、可测试，换任何模型都不会“忘记问诊”。

设计要点：
    - 无状态：每次请求都从完整对话历史重新推导状态，前端和接口基本不用改。
    - 阶段： urgent(红旗症状) / qa(纯知识问答) / collect(继续问) / conclude(给辨证小结) / followup(小结之后的追问)
    - 每轮只问一个主题；用户答“没有/还好/不清楚”也会记为已回答（通过上一轮问的是什么来判断）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings

CONCLUSION_MARK = "辨证小结"      # 模型给出结论时必须带的小标题；也用来判断“已经出过结论”

# --------------------------------------------------------------------------- 槽位定义
# detect：用户原话里出现这些说法 → 该信息已提供；ask_kw：上一轮助手的提问里含这些词 → 这一轮用户的回答属于该槽位
SLOTS: dict[str, dict] = {
    "duration": {
        "label": "病程与诱因",
        "detect": [r"[\d一二两三四五六七八九十半几多]+\s*(天|日|周|个?星期|个?月|年|小时|个?礼拜)", r"(昨天|前天|今天|昨晚|昨夜|今早|这几天|好几天|几天|一直|反复|刚开始|刚刚|两三天|半个月|很久|突然|慢慢|受凉后|淋雨|熬夜后|吃了.{0,6}后)"],
        "ask_kw": ["多久", "持续", "几天", "什么时候", "诱因", "起病", "开始"],
        "ask": "这个情况大概持续多久了？是突然出现的，还是慢慢加重的？发病前有没有受凉、淋雨、熬夜、吃了什么特别的东西？",
    },
    "cold_heat": {
        "label": "寒热与出汗",
        "detect": [r"(发烧|发热|低烧|高烧|不烧|没烧|退烧|怕冷|恶寒|畏寒|发冷|寒战|打寒颤|怕风|怕热|潮热|体温|手脚冰凉|手脚凉|身上冷|浑身热|汗)", r"3[5-9](\.\d)?\s*(度|℃|°)"],
        "ask_kw": ["发热", "发烧", "怕冷", "恶寒", "出汗", "寒热", "体温"],
        "ask": "现在有没有发热或怕冷？是怕冷明显（盖被子、穿厚衣服也觉得冷），还是发热明显、反而不太怕冷？有没有出汗，大概量过体温是多少度？",
    },
    "phlegm_nose": {
        "label": "痰与鼻涕",
        "detect": [r"(痰|涕|鼻塞|鼻子|鼻水|打喷嚏|喷嚏|干咳|无痰|没痰|咳不出|流鼻)"],
        "ask_kw": ["痰", "鼻涕", "鼻塞", "流涕"],
        "ask": "咳嗽有痰吗？痰是什么颜色、稀还是黏稠？有没有鼻塞、流鼻涕，鼻涕是清的、白的还是黄的？",
    },
    "throat": {
        "label": "咽喉情况",
        "detect": [r"(咽|喉咙|喉|嗓子|嗓)[^，。；,.;！？?!]{0,4}(痛|疼|痒|干|肿|红|哑|不舒服|异物|灼|烧|难受|刺)", r"(扁桃体|声音沙哑|失声|吞咽|咽下)"],
        "ask_kw": ["咽", "喉咙", "嗓子", "扁桃", "吞咽"],
        "ask": "喉咙具体是什么感觉？是干痒、刺痛还是火辣辣地痛？吞咽的时候更痛吗？自己对着镜子看，喉咙有没有发红、肿，或者扁桃体上有白点？",
    },
    "thirst": {
        "label": "口渴与饮水偏好",
        "detect": [r"(口干|口渴|不渴|渴|想喝水|不想喝水|喜热饮|喜冷饮|爱喝热|爱喝凉|口苦|口臭|口淡|口黏)"],
        "ask_kw": ["口渴", "口干", "喝水", "口苦", "饮水"],
        "ask": "平时口渴吗？更想喝热水还是凉水？嘴里有没有发苦、发黏或口臭？",
    },
    "appetite": {
        "label": "饮食胃口",
        "detect": [r"(食欲|胃口|吃不下|不想吃|没胃口|恶心|腹胀|胃胀|反酸|嗳气|打嗝|吃饭)"],
        "ask_kw": ["食欲", "胃口", "饮食", "吃饭"],
        "ask": "最近胃口怎么样？吃饭香不香，有没有恶心、腹胀、反酸？",
    },
    "stool_urine": {
        "label": "大小便",
        "detect": [r"(大便|小便|便秘|腹泻|拉肚子|尿|排便|便溏|解便|拉稀|干结|稀溏)"],
        "ask_kw": ["大便", "小便", "排便", "二便"],
        "ask": "大小便怎么样？大便是干、稀还是正常，一天几次？小便颜色偏黄吗？",
    },
    "sleep": {
        "label": "睡眠",
        "detect": [r"(睡眠|睡得|睡不|失眠|多梦|入睡|早醒|嗜睡|易醒|睡觉)"],
        "ask_kw": ["睡眠", "睡", "失眠"],
        "ask": "最近睡眠怎么样？入睡难不难，夜里会不会醒、多梦？",
    },
    "basic": {
        "label": "年龄性别与既往情况",
        "detect": [r"\d+\s*岁", r"(男|女|孕|怀孕|哺乳|宝宝|孩子|儿童|小孩|婴儿|老人|月经|过敏|高血压|糖尿病|慢性|在吃药|服用|吃了.{0,6}药|既往|病史|手术|心脏病|哮喘|没有.{0,4}(病|过敏))"],
        "ask_kw": ["年龄", "性别", "怀孕", "过敏", "慢性病", "病史", "在服", "用药", "吃药", "几岁", "多大"],
        "ask": "方便说一下您的年龄和性别吗？有没有怀孕或哺乳、药物或食物过敏、高血压糖尿病等慢性病，目前有没有在吃其他药？这些会影响调理建议是否安全。",
    },
}

# 通用问诊（没有专门类别时的“十问”简版）
GENERIC_CATEGORY = {
    "id": "generic",
    "name": "一般症状",
    "slot_order": ["duration", "cold_heat", "appetite", "stool_urine", "sleep", "basic"],
    "required": ["duration", "cold_heat"],
    "optional": ["appetite", "stool_urine", "sleep", "basic"],
    "optional_min": 2,
    "max_questions": 5,
    "redflags": [
        {"kw": ["胸痛", "胸口剧痛", "呼吸困难", "喘不上气", "意识不清", "昏迷", "抽搐", "咯血", "咳血", "吐血", "便血", "黑便", "大出血", "剧烈头痛", "突然剧烈头痛", "半身不遂", "口角歪斜", "说话不清", "突然看不见", "想自杀", "不想活"],
         "msg": "您提到的情况可能比较紧急，请尽快去医院急诊或拨打120"},
    ],
    "high_risk_note": "婴幼儿、孕妇/哺乳期、老年人、有慢性病者用药限制多，请务必先咨询医生",
    "patterns": [],
}

# 症状词：用于判断“这是不是在描述自己的不适”
_SYMPTOM_HINT = re.compile(r"(痛|疼|痒|晕|难受|不舒服|不适|失眠|恶心|呕|拉肚子|腹泻|便秘|胀|麻|乏力|没力气|累|发烧|发热|咳|喘|心慌|心悸|出汗|盗汗|口干|口苦|耳鸣|食欲|胃口|上火|过敏|起疹|长痘|月经|症状)")
# 知识类问题：问概念/药/穴位，而不是描述自己的症状
_KNOW_Q = re.compile(r"(是什么|什么是|什么意思|有什么(功效|作用|区别|禁忌)|功效|作用是|怎么(煎|熬|吃|用|泡|按|区分|理解)|能不能|可不可以|可以(和|与|一起)|禁忌|区别|为什么|原理|出处|歌诀|哪些药|哪个穴|穴位在哪|在哪里|组成|讲讲|介绍一下|科普)")
_FIRST_PERSON = re.compile(r"(我|宝宝|孩子|小孩|娃|老人|妈妈|爸爸|老公|老婆|媳妇|家人|儿子|女儿|最近|已经|有点|好几天|这几天|\d+\s*(天|周|个月|年)|昨天|前天|今天|昨晚|早上起来)")
_FORCE = re.compile(r"(直接(给|说|告诉|开|出)|给我(结论|结果|诊断|方子|开方|辨证)|不用(再)?问了|别(再)?问了|先(给|出)(个)?(结论|方子|诊断)|就(这些|这样)了?|只能(说|告诉)这些|你(直接)?(判断|辨证|诊断)一下|开个方|出个方|辨证一下|总结一下|给个建议|给点建议)")
_NEG = "没无不未非"


# --------------------------------------------------------------------------- 知识表加载
@lru_cache(maxsize=1)
def _categories() -> list[dict]:
    p = settings.kg_dir / "common_conditions.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["categories"]


def _category_by_id(cid: str) -> dict:
    for c in _categories():
        if c["id"] == cid:
            return c
    return GENERIC_CATEGORY


# --------------------------------------------------------------------------- 文本工具
def _clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[，,。；;！!？?\n]+", text) if c.strip()]


def kw_hit(text: str, kw: str) -> bool:
    """关键词命中，且不被『没/无/不/未』否定（例：『不发烧』不命中『发烧』）。"""
    start = 0
    while True:
        pos = text.find(kw, start)
        if pos < 0:
            return False
        if kw[0] in _NEG or not any(ch in _NEG for ch in text[max(0, pos - 2):pos]):
            return True
        start = pos + 1


def _slot_match(slot: str, clause: str) -> bool:
    return any(re.search(p, clause) for p in SLOTS[slot]["detect"])


def _questions(assistant_text: str) -> str:
    """取助手上一条回复里的所有问句（拼在一起），用来判断它在问什么。"""
    parts = re.split(r"(?<=[？?])", assistant_text)
    return "".join(p.strip() for p in parts if p.strip().endswith(("？", "?")))


def _asked_slot(assistant_text: str, category: dict) -> str | None:
    """助手上一轮在问哪个槽位：看所有问句里哪个槽位的关键词最先出现。

    不能只看最后一个问句：“喉咙有什么感觉？有没有红肿？”的最后一句里没有“喉咙”二字。
    """
    q = _questions(assistant_text)
    if not q:
        return None
    best, best_pos = None, len(q) + 1
    for s in category["slot_order"]:
        for k in SLOTS[s]["ask_kw"]:
            pos = q.find(k)
            if 0 <= pos < best_pos:
                best, best_pos = s, pos
    return best


# --------------------------------------------------------------------------- 状态
@dataclass
class ConsultState:
    mode: str                                   # urgent | qa | collect | conclude | followup
    category: dict = field(default_factory=lambda: GENERIC_CATEGORY)
    filled: dict[str, list[str]] = field(default_factory=dict)   # 槽位 -> 患者原话片段
    missing: list[str] = field(default_factory=list)
    ask_slot: str | None = None
    asked_count: int = 0                        # 已经追问了几轮
    first_turn: bool = False
    force: bool = False
    why: str = ""
    redflags: list[str] = field(default_factory=list)
    clues: list[dict] = field(default_factory=list)   # 证候线索 [{name, score, hit, miss, formulas...}]
    complaint: str = ""
    user_text: str = ""

    # ------------------------------------------------ 给前端 / 给模型看的文字
    def progress(self) -> dict:
        cat = self.category
        labels = {s: SLOTS[s]["label"] for s in cat["slot_order"]}
        return {
            "mode": self.mode,
            "category": cat["name"],
            "known": [labels[s] for s in cat["slot_order"] if s in self.filled],
            "unknown": [labels[s] for s in self.missing],
            "asked": self.asked_count,
            "max_questions": cat.get("max_questions", 4),
            "can_conclude": self.mode == "collect" and self.asked_count >= 1,
        }

    def card(self) -> str:
        """病情卡：把散落在多轮里的信息整理好交给模型，避免它自己从长对话里翻找。"""
        cat = self.category
        lines = ["【病情卡（据患者原话整理，模型须以此为准，不要重复询问已有项）】", f"- 主诉：{self.complaint or '（未明确）'}"]
        for s in cat["slot_order"]:
            if s in self.filled:
                lines.append(f"- {SLOTS[s]['label']}：{'；'.join(self.filled[s])}")
        if self.missing:
            lines.append("- 尚未了解：" + "、".join(SLOTS[s]["label"] for s in self.missing))
        if self.redflags:
            lines.append("- ⚠ 已出现的需就医信号：" + "；".join(self.redflags))
        return "\n".join(lines)

    def clues_text(self) -> str:
        if not self.clues:
            return ""
        out = ["【证候线索（程序根据患者原话匹配，只是参考，可信度有限，请结合病情卡自行判断）】"]
        for i, c in enumerate(self.clues, 1):
            out.append(f"{i}. {c['name']}｜已符合：{'、'.join(c['hit'])}｜尚未核实：{'、'.join(c['miss']) or '无'}"
                       f"｜代表方：{'、'.join(c['formulas'])}")
            out.append(f"   调护要点：{'；'.join(c['selfcare'][:3])}｜宜忌：忌{'、'.join(c['avoid'][:3])}")
        return "\n".join(out)


# --------------------------------------------------------------------------- 主入口
def analyze(question: str, history: list[dict] | None, force: bool = False) -> ConsultState:
    """根据完整对话重新推导问诊状态。history 不含当前这句。"""
    from app.llm.engine import engine   # 复用历史清洗（去掉报错/加载提示）

    hist = engine.clean_history(history, limit=60)
    user_msgs = [m["content"] for m in hist if m["role"] == "user"] + [question]
    assistant_msgs = [m["content"] for m in hist if m["role"] == "assistant"]
    all_user = "\n".join(user_msgs)
    force = force or bool(_FORCE.search(question))

    # 1) 已经给过辨证小结 → 之后是追问
    concluded = any(CONCLUSION_MARK in a for a in assistant_msgs)

    # 2) 类别
    category = _detect_category(all_user)

    # 3) 红旗：只在“这一句”新出现时打断（否则一直打断对话）
    flags_now = _redflags(question, category)
    flags_all = _redflags(all_user, category)

    # 4) 当前这句是不是纯知识问答
    know_only = bool(_KNOW_Q.search(question)) and not _FIRST_PERSON.search(question)
    looks_symptom = bool(_SYMPTOM_HINT.search(all_user)) or category is not GENERIC_CATEGORY

    st = ConsultState(mode="qa", category=category, user_text=all_user, force=force,
                      redflags=flags_all, complaint=_complaint(user_msgs))

    if flags_now:
        st.mode, st.why = "urgent", "；".join(flags_now)
        st.redflags = flags_all
        return st
    if know_only or not looks_symptom:
        st.mode = "followup" if concluded else "qa"
        return st

    # 5) 收集槽位
    filled = _fill_slots(user_msgs, hist, category)
    st.filled = filled
    st.missing = [s for s in category["slot_order"] if s not in filled]
    st.asked_count = sum(1 for a in assistant_msgs if CONCLUSION_MARK not in a)
    st.first_turn = len(user_msgs) == 1
    st.clues = match_patterns(all_user, category)

    if concluded:
        st.mode = "followup"
        return st

    ready, why = _ready(filled, st.asked_count, category)
    if force:
        ready, why = True, "患者要求直接给出建议"
    if ready:
        st.mode, st.why = "conclude", why
    else:
        st.mode = "collect"
        st.ask_slot = st.missing[0] if st.missing else None
        if st.ask_slot is None:          # 没有可问的了
            st.mode, st.why = "conclude", "能问的都问到了"
    return st


# --------------------------------------------------------------------------- 内部步骤
def _detect_category(text: str) -> dict:
    best, best_n = GENERIC_CATEGORY, 0
    for c in _categories():
        n = sum(1 for t in c["triggers"] if kw_hit(text, t))
        if n > best_n:
            best, best_n = c, n
    return best


def _redflags(text: str, category: dict) -> list[str]:
    msgs = []
    for rf in list(category.get("redflags", [])) + ([] if category is GENERIC_CATEGORY else GENERIC_CATEGORY["redflags"]):
        if any(kw_hit(text, k) for k in rf["kw"]) and rf["msg"] not in msgs:
            msgs.append(rf["msg"])
    return msgs


def _complaint(user_msgs: list[str]) -> str:
    first = user_msgs[0].strip()
    return first if len(first) <= 80 else first[:80] + "…"


def _fill_slots(user_msgs: list[str], hist: list[dict], category: dict) -> dict[str, list[str]]:
    filled: dict[str, list[str]] = {}
    # a) 关键词：用户说了什么就记什么
    for msg in user_msgs:
        for clause in _clauses(msg):
            for s in category["slot_order"]:
                if _slot_match(s, clause):
                    bucket = filled.setdefault(s, [])
                    if clause not in bucket and len(bucket) < 3:
                        bucket.append(clause)
    # b) 回答式：助手问了 X，用户的下一句无论说什么（包括“没有”“还好”“不清楚”）都算回答了 X
    msgs = [m for m in hist if m["role"] in ("user", "assistant")]
    pairs = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and CONCLUSION_MARK not in m["content"]:
            reply = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user" else None
            pairs.append((m["content"], reply))
    # 最后一条助手消息对应“当前这句”
    if msgs and msgs[-1]["role"] == "assistant" and CONCLUSION_MARK not in msgs[-1]["content"]:
        pairs[-1] = (msgs[-1]["content"], user_msgs[-1])
    for a_text, reply in pairs:
        if not reply:
            continue
        s = _asked_slot(a_text, category)
        if s and s not in filled:
            filled[s] = [reply.strip()[:40]]
    return filled


def _ready(filled: dict, asked: int, category: dict) -> tuple[bool, str]:
    req = category["required"]
    opt = category.get("optional", [])
    need_opt = category.get("optional_min", 2)
    if asked >= category.get("max_questions", 4):
        return True, f"已追问{asked}轮，达到上限"
    if all(s in filled for s in req) and sum(1 for s in opt if s in filled) >= need_opt:
        return True, "关键信息（" + "、".join(SLOTS[s]["label"] for s in req) + "）和足够的辅助信息都已了解"
    return False, ""


def match_patterns(text: str, category: dict, top_k: int = 2) -> list[dict]:
    """把患者原话与证型线索逐条比对（带否定处理）。"""
    out = []
    for p in category.get("patterns", []):
        hit, miss, score = [], [], 0
        for ev in p["evidence"]:
            if ev["w"] <= 0:
                continue
            if any(kw_hit(text, k) for k in ev["kw"]):
                hit.append(ev["label"]); score += ev["w"]
            else:
                miss.append(ev["label"])
        if score >= 2:
            out.append({"name": p["name"], "score": score, "hit": hit, "miss": miss[:3],
                        "formulas": p["formulas"], "selfcare": p["selfcare"], "avoid": p["avoid"],
                        "rag": p.get("rag", ""), "rag_must": p.get("rag_must", [])})
    out.sort(key=lambda x: -x["score"])
    return out[:top_k]
