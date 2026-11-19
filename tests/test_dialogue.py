"""问诊状态机的离线测试（不需要模型）：python -m pytest tests -q  或  python tests/test_dialogue.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.consult.dialogue import analyze


def run(turns, force_last=False):
    """turns: [(用户话, 助手回复)]；返回每一轮 analyze 的状态。助手回复要模拟模型会问的话。"""
    hist, out = [], []
    for i, (u, a) in enumerate(turns):
        out.append(analyze(u, hist, force=(force_last and i == len(turns) - 1)))
        hist += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def test_single_word_does_not_conclude():
    sts = run([("咳嗽", "现在有没有发热或怕冷？")])
    assert sts[0].mode == "collect" and sts[0].ask_slot == "cold_heat"


def test_no_answer_counts_even_with_multi_question_reply():
    # 助手的提问里有两个问句，最后一个不含“喉咙”；用户答“没有”也要算回答了咽喉
    sts = run([
        ("我咳嗽三天了", "现在有没有发热或怕冷？出汗吗？"),
        ("怕冷，没怎么出汗", "咳嗽有痰吗？有鼻塞流涕吗？"),
        ("白痰，清鼻涕", "喉咙有什么感觉？有没有红肿？"),
        ("没有", ""),
    ])
    assert "咽喉情况" in sts[3].progress()["known"]
    assert sts[3].mode == "conclude"          # 关键信息已齐，不应该再追问同一个问题


def test_never_repeat_asked_slot():
    qs = {"cold_heat": "现在有没有发热或怕冷？", "phlegm_nose": "咳嗽有痰吗？有鼻涕吗？",
          "throat": "喉咙有什么感觉？有没有红肿？", "thirst": "平时口渴吗？更想喝热水还是凉水？"}
    hist, asked = [], []
    for u in ["感冒了", "没有", "没有", "没有", "没有", "没有"]:
        st = analyze(u, hist)
        if st.mode != "collect":
            break
        assert st.ask_slot not in asked, f"重复追问 {st.ask_slot}"
        asked.append(st.ask_slot)
        hist += [{"role": "user", "content": u}, {"role": "assistant", "content": qs.get(st.ask_slot, "最近睡眠怎么样？")}]
    assert len(asked) >= 2


def test_redflag():
    st = run([("咳嗽两天，现在喘不上气，嘴唇发紫", "")])[0]
    assert st.mode == "urgent"


def test_knowledge_qa():
    assert run([("黄芪的功效是什么", "")])[0].mode == "qa"


def test_force_conclude():
    sts = run([("最近一直咳嗽", "现在有没有发热或怕冷？"), ("你直接给我结论吧", "")], force_last=True)
    assert sts[1].mode == "conclude"


def test_followup_after_conclusion():
    sts = run([("感冒了，咳嗽，喉咙痛，发烧37.8，黄痰黄鼻涕", "## 辨证小结 …"), ("那能吃点什么？", "")])
    assert sts[1].mode == "followup"


def test_wind_heat_vs_cold_clues():
    st = run([("感冒了，咳嗽，喉咙痛，发烧37.8不怎么怕冷，出了点汗，黄痰黄鼻涕，嗓子红肿", "")])[0]
    assert st.clues and st.clues[0]["name"].startswith("风热")
    st = run([("我感冒了不发烧，就是怕冷没汗，白痰清鼻涕", "")])[0]
    assert st.clues and st.clues[0]["name"].startswith("风寒")


if __name__ == "__main__":
    import traceback
    ok = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try:
                f(); ok += 1; print("PASS", n)
            except Exception:
                print("FAIL", n); traceback.print_exc()
    print(ok, "passed")
