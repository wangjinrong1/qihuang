"""模型自检：不启动网页，直接加载模型并生成一句话，确认环境没问题。
用法：python check_llm.py
"""
import time

from app.config import settings
from app.llm.engine import engine

print(f"基座模型 : {settings.llm_base_model}")
print(f"适配器   : {settings.llm_adapter_path or '(未配置)'}")
print(f"设备设置 : {settings.llm_device}")
print(f"API 模型 : {settings.llm_api_model}（{'已配置密钥' if settings.llm_api_key else '未配置密钥'}）")
print("加载中…（首次加载可能需要几分钟）")

t0 = time.time()
engine._load()
info = engine.info()
print(f"\n加载耗时 : {time.time() - t0:.1f}s    模式: {info['mode']}    设备: {info['device']}")
if info["error"]:
    print(f"\n[加载失败，已回退演示模式]\n{info['error']}")
    raise SystemExit(1)
if info["mode"] == "mock":
    raise SystemExit("当前是演示模式：请在 .env 中填写 LLM_ADAPTER_PATH（本地模型），或设置 LLM_MODE=api 并填写 LLM_API_KEY（云端）")

q = "请简要说明气虚和血虚的区别。"
print(f"\n问：{q}\n答：", end="", flush=True)
t1, n = time.time(), 0
for piece in engine.stream(engine.build_messages(None, q), max_new_tokens=200):
    print(piece, end="", flush=True)
    n += len(piece)
dt = time.time() - t1
print(f"\n\n生成 {n} 字，用时 {dt:.1f}s（约 {n / max(dt, 1e-6):.1f} 字/秒）")
