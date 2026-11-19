"""端到端（不需要真模型）：用演示后端把 chat.stream_answer 的多轮流程跑一遍，模拟前端维护 history 的方式。
python tests/test_flow_e2e.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.consult import chat
from app.llm.engine import engine, _MockBackend

engine._backend = _MockBackend()
engine._state = "ready"


def turn(history, text, force=False):
    meta, ans = {}, ""
    for ev in chat.stream_answer(text, history, force):
        if ev["type"] == "meta": meta = ev
        elif ev["type"] == "token": ans += ev["text"]
    ans = ans.replace("【演示模式：尚未加载中医大模型，以下为占位回复】\n\n", "")
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": ans}]
    return meta, ans


def show(title, users, force_last=False):
    print("=" * 10, title)
    h, modes = [], []
    for i, u in enumerate(users):
        meta, ans = turn(h, u, force_last and i == len(users) - 1)
        p = meta["progress"]; modes.append(p["mode"])
        print(f"U{i+1}: {u}\n  [{p['mode']}] 已知={p['known']} 未知={p['unknown']} 证型={meta['syndromes']} 资料={meta['sources']}\n  A: {ans[:70].replace(chr(10),' ')}")
    return modes


m = show("感冒咳嗽喉咙痛（风热）", ["感冒了，咳嗽，喉咙痛", "有点发烧37.8，不怎么怕冷，出了点汗", "黄痰，黄鼻涕，嗓子红肿吞咽疼", "已经三天了", "那能吃点什么？"])
assert m[0] == "collect" and "conclude" in m and m[-1] == "followup", m
assert m.index("conclude") >= 2, "至少问两轮再下结论"
m = show("风寒，全部用“没有”回答", ["我咳嗽三天了", "怕冷，没怎么出汗", "白痰，清鼻涕", "没有"])
assert m[-1] == "conclude", m
m = show("一句话+强制结论", ["最近一直咳嗽", "你直接给我结论吧"], force_last=True)
assert m == ["collect", "conclude"], m
m = show("红旗", ["咳嗽两天，现在喘不上气，嘴唇发紫"]); assert m == ["urgent"]
m = show("知识问答", ["黄芪的功效是什么", "和党参有什么区别"]); assert set(m) == {"qa"}
m = show("失眠（通用问诊）", ["我最近失眠，睡不着", "两个月了", "没有", "还行"])
assert m[0] == "collect" and "conclude" in m and m.index("conclude") >= 2
assert all(x == "followup" for x in m[m.index("conclude") + 1:])   # 给出结论之后只做追问，不再重复盘问
print("\nALL OK")
