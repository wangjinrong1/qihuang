import pytest
from app.llm.engine import engine
from app.web import search as ws


def test_switch_bad_mode_keeps_current():
    before = engine.info()["mode"]
    with pytest.raises(RuntimeError):
        engine.switch("api", "", "https://api.deepseek.com", "deepseek-chat")   # 没填 Key
    assert engine.info()["mode"] == before


def test_switch_mock():
    assert engine.switch("mock")["mode"] == "mock"


def test_web_fallback_when_all_fail(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("x")
    monkeypatch.setattr(ws, "_bing", boom, raising=False)
    monkeypatch.setattr(ws, "_ddg", boom, raising=False)
    monkeypatch.setattr(ws.settings, "web_provider", "auto", raising=False)
    ws._CACHE.clear()
    r = ws.search("感冒 中医", n=3)
    assert r["results"] == [] and r["error"]
