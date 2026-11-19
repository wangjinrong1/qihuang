"""全局配置：读取环境变量 / .env"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.split("#")[0].strip())


_load_dotenv()


class Settings:
    root = ROOT
    data_dir = ROOT / "data"
    kg_dir = ROOT / "data" / "kg"
    corpus_dir = ROOT / "data" / "corpus"
    web_dir = ROOT / "web"
    upload_dir = ROOT / "uploads"

    llm_mode = os.getenv("LLM_MODE", "auto").lower()
    llm_base_model = os.getenv("LLM_BASE_MODEL", "Qwen/Qwen1.5-1.8B-Chat")
    llm_adapter_path = os.getenv("LLM_ADAPTER_PATH", "").strip()
    llm_max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "768"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_device = os.getenv("LLM_DEVICE", "auto").lower()          # auto | cpu | cuda
    llm_repetition_penalty = float(os.getenv("LLM_REPETITION_PENALTY", "1.15"))  # 本地小模型防重复

    # 云端 API（OpenAI 兼容接口，如 DeepSeek）；LLM_MODE=api 时使用
    llm_api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com").strip()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    llm_api_model = os.getenv("LLM_API_MODEL", "deepseek-chat").strip()
    llm_api_timeout = int(os.getenv("LLM_API_TIMEOUT", "120"))
    llm_api_max_tokens = int(os.getenv("LLM_API_MAX_TOKENS", "2000"))     # 云端模型回答可以更长

    # 联网检索（补充模型知识不足）：auto = 有 SERPER_API_KEY 用 Serper(Google)，否则依次尝试 Bing 中国版、DuckDuckGo
    web_provider = os.getenv("WEB_SEARCH_PROVIDER", "auto").lower()      # auto | serper | bing | duckduckgo
    serper_api_key = os.getenv("SERPER_API_KEY", "").strip()
    web_timeout = float(os.getenv("WEB_SEARCH_TIMEOUT", "6"))
    llm_preload = os.getenv("LLM_PRELOAD", "1") not in ("0", "false", "False")

    tongue_weights = os.getenv("TONGUE_WEIGHTS", "").strip()

    # 中药图片识别：权重（TorchScript .pt，或 torchvision 的 state_dict）+ 类别名文件（每行一个药名）
    herb_weights = os.getenv("HERB_WEIGHTS", "").strip()
    herb_labels = os.getenv("HERB_LABELS", "").strip()
    herb_arch = os.getenv("HERB_ARCH", "resnet50").strip()

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))


settings = Settings()
