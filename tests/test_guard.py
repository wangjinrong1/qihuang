from app.consult import guard, pipeline
from app.kg.graph import get_graph

PATIENT = {"name": "中医ai助手", "gender": "男", "age": "40", "complaint": "喉咙痛", "duration": "两天", "history": "无"}
TONGUE = {"label": "白厚腻苔", "confidence": 0.91, "note": "苔白厚腻，多为寒湿或痰湿困脾。"}
FAKE = ("主诉：喉部疼痛\n病情描述：\n患者自述昨日突然感到咽喉不适，伴轻微咳嗽，吞咽较以往明显困难，且伴有发热，体温最高达38.5℃，"
        "口渴喜饮，鼻塞流涕，大便稀溏，舌苔薄黄，脉浮数。遂就诊我院中医科求治。\n其他检查：\n医生为其进行了初步体格检查，并对其进行问诊、查体。\n"
        "辨证要点：\n根据舌苔的白色厚腻，可以推断患者可能属于脾胃湿热证，主要表现为口苦、口黏、胸闷。\n")


def _sup():
    kg = get_graph()
    return kg, guard.Support(PATIENT, TONGUE, kg.match_symptoms("喉咙痛" + TONGUE["label"]), kg)


def test_fabricated_text_is_flagged_and_stripped():
    kg, sup = _sup()
    bad = guard.audit("辨证与诊断依据", FAKE, sup)
    joined = "".join(" ".join(b["reasons"]) for b in bad)
    for key in ("38.5", "脉浮数", "薄黄", "遂就诊", "查体"):
        assert key in joined or any(key in b["sentence"] for b in bad), key
    clean = guard.strip(FAKE, bad)
    for key in ("38.5", "薄黄", "脉浮数", "听诊", "遂就诊", "查体", "昨日"):
        assert key not in clean
    assert "喉部疼痛" in clean          # 与患者输入一致的内容保留


def test_supported_text_passes():
    _, sup = _sup()
    ok = "患者主诉喉咙痛，病程两天。舌象为白厚腻苔，提示寒湿或痰湿偏盛。需进一步询问：是否怕冷发热、咽部是否红肿。\n"
    assert guard.audit("辨证与诊断依据", ok, sup) == []


def test_conditional_warning_not_flagged():
    _, sup = _sup()
    t = "如果出现吞咽困难、体温超过38.5℃或呼吸困难，请立即就医。"
    assert guard.audit("预警与就医", t, sup) == []


FULL = dict(PATIENT, complaint="喉咙痛，吞咽时更痛，咽干，怕冷不发热，口渴想喝热水，大便偏稀")


def test_gate_blocks_thin_info():
    c = guard.assess(PATIENT, True, get_graph())
    assert not c["ready"]
    assert any("寒热" in x for x in c["need"]) and any("伴随症状" in x for x in c["need"])
    assert not any("舌象" in x for x in c["need"])            # 已上传舌象


def test_gate_needs_tongue_and_passes_when_complete():
    kg = get_graph()
    assert any("舌象" in x for x in guard.assess(FULL, False, kg)["need"])
    ok = guard.assess(FULL, True, kg)
    assert ok["ready"] and ok["need"] == []


def test_pipeline_blocks_insufficient_without_calling_model(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline.engine, "stream", lambda *a, **k: called.append(1) or iter(()))
    evs = list(pipeline.run(PATIENT, None))
    types = [e["type"] for e in evs]
    assert "insufficient" in types and "report" not in types and not called


def test_demo_tongue_is_not_evidence(monkeypatch):
    monkeypatch.setattr(pipeline.classifier, "analyze",
                        lambda p: {"label": "白厚腻苔", "confidence": 0.5, "note": "x", "demo": True})
    monkeypatch.setattr(pipeline.engine, "stream", lambda *a, **k: iter(()))
    p = dict(FULL, complaint=FULL["complaint"].replace("怕冷", "怕冷"))
    evs = list(pipeline.run(p, __import__("pathlib").Path("x.jpg")))
    ins = [e for e in evs if e["type"] == "insufficient"]
    assert ins and any("舌象" in x for x in ins[0]["need"])


def test_red_flags():
    assert guard.red_flags({"complaint": "喉咙痛 吞咽困难"})
    assert not guard.red_flags(FULL)


def test_pipeline_retries_then_strips(monkeypatch):
    calls = []

    def fake_stream(msgs, max_new_tokens=None):
        calls.append(msgs[-1]["content"])
        yield FAKE                          # 每次都编造 → 重写一次后仍不合格 → 删句

    monkeypatch.setattr(pipeline.engine, "stream", fake_stream)
    monkeypatch.setattr(pipeline.engine, "info", lambda: {"mode": "api"})
    monkeypatch.setattr(pipeline.classifier, "analyze", lambda p: dict(TONGUE))
    evs = list(pipeline.run(FULL, __import__("pathlib").Path("x.jpg")))
    assert any(e["type"] == "section_reset" for e in evs)
    assert any(e["type"] == "section_replace" for e in evs)
    assert any("上一稿不合格" in c for c in calls)
    rep = [e for e in evs if e["type"] == "report"][0]["markdown"]
    assert "38.5" not in rep and "脉浮数" not in rep
    for h in ("## 二、主诉", "## 五、四诊信息", "## 六、体格检查与辅助检查", "信息完整度与局限性"):
        assert h in rep
    assert "脉象：未提供" in rep and "闻诊 | 未提供" in rep


def test_demo_mode_writes_no_fake_analysis(monkeypatch):
    monkeypatch.setattr(pipeline.classifier, "analyze", lambda p: dict(TONGUE))
    monkeypatch.setattr(pipeline.engine, "info", lambda: {"mode": "mock"})
    called = []
    monkeypatch.setattr(pipeline.engine, "stream", lambda *a, **k: called.append(1) or iter(()))
    evs = list(pipeline.run(FULL, __import__("pathlib").Path("x.jpg")))
    rep = [e for e in evs if e["type"] == "report"][0]["markdown"]
    assert not called and "演示模式：尚未接入模型" in rep and "辨证思路" not in rep
