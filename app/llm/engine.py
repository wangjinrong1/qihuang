"""大模型引擎：本地中医微调模型（基座 + LoRA 适配器），无模型时自动进入演示模式。

对外统一接口：
    engine.info()                                  -> dict（含加载状态）
    engine.start_loading()                         -> 后台线程预加载模型
    engine.stream(messages, max_new_tokens=None)   -> Iterator[str]
    engine.generate(messages)                      -> str
messages 采用 OpenAI 风格：[{"role": "system|user|assistant", "content": "..."}]
"""
from __future__ import annotations

import threading
import time
from typing import Iterator

from app.config import settings
from app.llm.prompts import SYSTEM_PROMPT


class _MockBackend:
    """演示模式：不加载任何模型，返回占位文本，用于先看整体框架效果。"""

    name = "mock"
    device = "-"

    def stream(self, messages: list[dict], max_new_tokens: int | None = None) -> Iterator[str]:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = self._compose(user)
        for i in range(0, len(text), 4):
            time.sleep(0.015)
            yield text[i : i + 4]

    @staticmethod
    def _compose(user: str) -> str:
        head = "【演示模式：尚未加载中医大模型，以下为占位回复】\n\n"
        # --- 智能问诊的分阶段模板：按“本轮任务”给出对应形态的占位回复，便于不接模型也能走通多轮流程
        import re as _re
        if "【本轮任务：继续问诊" in user:
            m = _re.search(r"围绕「(.+?)」：(.+)", user)
            return head + "好的，我记下了。" + (m.group(2).strip() if m else "还想再了解一下您的情况，请再补充一些。")
        if "【本轮任务：给出辨证小结】" in user:
            return head + ("**症状归纳**：（接入模型后，这里会汇总整段对话里您说过的症状）\n\n"
                           "**辨证**：（根据病情卡与证候线索给出证型和依据）\n\n**调护建议**：（饮食、起居、食疗、穴位）\n\n"
                           "**方药参考**：只列代表方名称，需执业医师辨证后使用。\n\n**什么时候要去医院**：持续高热、呼吸困难、胸痛、咯血等。")
        if "【本轮任务：解答追问】" in user:
            return head + "（接入模型后，这里会结合整段对话回答您的追问。）"
        if "【本轮任务：风险提示" in user:
            return head + "您提到的情况可能比较紧急，建议尽快去医院急诊或拨打120。"
        if "【本轮任务：解答知识问题】" in user:
            return head + "（接入模型后，这里会结合参考资料回答这个知识问题。）"
        if "知识图谱检索结果" in user:
            return head + (
                "**辨证思路**：依据已匹配的症状在知识图谱中的关联权重，初步倾向于下列证型，"
                "请结合四诊合参进一步判断。\n\n"
                "**方药说明**：推荐方剂来自图谱“证型—方剂—药材”关系，剂量为图谱记录值，"
                "仅供参考；若出现配伍禁忌提示，须由医师审核。\n\n"
                "**生活调护**：饮食清淡，规律作息，保持情志舒畅。\n\n"
                "**需就医的情形**：症状持续加重、高热不退、胸痛气促、意识改变等应及时就医。"
            )
        if "【处方】" in user:
            return head + "（接入模型后，这里会给出组成与功效方向、配伍思路、禁忌提示和注意事项。）"
        q = user.split("用户问题：")[-1].split("\n")[0].strip()[:60]
        return head + f"你问的是：“{q}”。接入模型后，这里将由中医大模型结合知识图谱与古籍资料给出回答。"


class _LocalBackend:
    """基座模型 + LoRA 适配器（transformers + peft）。CPU/GPU 自动选择。"""

    name = "local"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        base, adapter = settings.llm_base_model, settings.llm_adapter_path

        want = settings.llm_device
        use_cuda = torch.cuda.is_available() if want == "auto" else want == "cuda"
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("LLM_DEVICE=cuda 但未检测到可用的 NVIDIA 显卡/CUDA 版 torch")
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32   # CPU 用 fp32：bf16 在多数 CPU 上更慢

        try:  # 适配器目录通常自带分词器；没有则用基座的
            self.tokenizer = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = self._load_base(AutoModelForCausalLM, base, dtype)
        if adapter:
            try:  # 合并 LoRA 到基座：推理更快，之后不再依赖 peft
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
            except Exception:
                model.load_adapter(adapter)
        self.model = model.to(self.device).eval()
        self._lock = threading.Lock()  # 单模型串行推理

    @staticmethod
    def _load_base(cls, base, dtype):
        try:                     # 新版 transformers 用 dtype
            return cls.from_pretrained(base, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        except TypeError:        # 旧版用 torch_dtype
            return cls.from_pretrained(base, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)

    def stream(self, messages: list[dict], max_new_tokens: int | None = None) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600)
        gen = dict(
            **inputs, streamer=streamer, pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens or settings.llm_max_new_tokens,
            do_sample=settings.llm_temperature > 0,
            temperature=max(settings.llm_temperature, 1e-3), top_p=0.8,
            repetition_penalty=settings.llm_repetition_penalty,
        )
        err: list[str] = []

        def _run():
            try:
                with self._lock, self._torch.no_grad():
                    self.model.generate(**gen)
            except Exception as e:  # 生成线程出错时必须收尾，否则前端会一直等待
                err.append(f"{type(e).__name__}: {e}")
                streamer.end()

        threading.Thread(target=_run, daemon=True).start()
        for piece in streamer:
            yield piece
        if err:
            yield f"\n\n（生成失败：{err[0]}）"


class _ApiBackend:
    """云端 API 后端：OpenAI 兼容的 /chat/completions 流式接口（DeepSeek 等）。

    注意：问诊内容会发送到第三方服务器，请勿填写真实姓名、证件号等隐私信息。
    """

    name = "api"
    device = "api"

    def __init__(self, base: str, key: str, model: str, timeout: int | None = None) -> None:
        if not (key or "").strip():
            raise RuntimeError("还没有填写 API Key")
        self.url = (base or "https://api.deepseek.com").rstrip("/") + "/chat/completions"
        self.key = key.strip()
        self.model = (model or "deepseek-chat").strip()
        self.timeout = timeout or settings.llm_api_timeout

    @staticmethod
    def _friendly(code: int, detail: str) -> str:
        hint = {401: "API Key 无效或已过期，请检查", 402: "账户余额不足，请充值",
                403: "没有权限访问该模型", 404: "接口地址或模型名不对", 429: "请求太频繁或额度用完，稍后再试"}.get(code, "")
        return f"（API 调用失败：HTTP {code}{'，' + hint if hint else ''}。{detail}）"

    def stream(self, messages: list[dict], max_new_tokens: int | None = None) -> Iterator[str]:
        import json
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": settings.llm_temperature,
            "max_tokens": max_new_tokens or settings.llm_api_max_tokens,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        # deepseek-reasoner 还会返回 reasoning_content（思考过程），这里只取最终回答
                        piece = json.loads(data)["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if piece:
                        yield piece
        except urllib.error.HTTPError as e:
            yield "\n\n" + self._friendly(e.code, e.read().decode("utf-8", "ignore")[:160])
        except Exception as e:
            yield f"\n\n（API 调用失败：{type(e).__name__}: {e}）"

    def ping(self) -> str | None:
        """连通性测试：成功返回 None，失败返回可读的错误说明。"""
        text = "".join(self.stream([{"role": "user", "content": "回复“好”即可"}], max_new_tokens=8))
        return text.strip("\n ") if "API 调用失败" in text else None


class LLMEngine:
    def __init__(self) -> None:
        self._backend = None
        self._error: str | None = None
        self._state = "idle"          # idle | loading | ready | error
        self._load_seconds: float | None = None
        self._lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self._switching = False
        # 运行时配置：页面上切换后生效（只保存在内存里，重启后回到 .env 的设置；API Key 不落盘）
        self._rt = {"mode": None, "api_key": settings.llm_api_key,
                    "api_base": settings.llm_api_base, "api_model": settings.llm_api_model}

    def _resolve_mode(self) -> str:
        mode = self._rt["mode"] or settings.llm_mode
        if mode == "auto":   # 有适配器 → 本地；否则有 API Key → 云端；都没有 → 演示
            if settings.llm_adapter_path:
                return "local"
            return "api" if self._rt["api_key"] else "mock"
        return mode

    def _make_api(self) -> _ApiBackend:
        return _ApiBackend(self._rt["api_base"], self._rt["api_key"], self._rt["api_model"])

    # -------------------------------------------------------------- 加载
    def _load(self) -> None:
        with self._lock:
            if self._backend is not None:
                return
            self._state = "loading"
            t0 = time.time()
            mode = self._resolve_mode()
            if mode in ("local", "api"):
                try:
                    self._backend = _LocalBackend() if mode == "local" else self._make_api()
                    self._state = "ready"
                except Exception as e:  # 依赖缺失 / 权重不存在 / 没填 Key → 降级演示模式
                    self._error = f"{type(e).__name__}: {e}"
                    self._backend = _MockBackend()
                    self._state = "error"
            else:
                self._backend = _MockBackend()
                self._state = "ready"
            self._load_seconds = round(time.time() - t0, 1)

    def start_loading(self) -> None:
        """服务启动时后台预加载，避免第一个请求等待几分钟。"""
        if self._backend is None and self._state != "loading":
            self._state = "loading"
            threading.Thread(target=self._load, daemon=True).start()

    # -------------------------------------------------------------- 运行时切换
    def switch(self, mode: str, api_key: str | None = None, api_base: str | None = None,
               api_model: str | None = None) -> dict:
        """切换模型。失败时保持原模型不变并抛出 RuntimeError（附可读原因）。

        mock / api 同步完成（api 会先做一次连通性测试）；local 加载慢，放到后台线程，
        期间继续用旧模型，加载成功后才替换，前端轮询 /api/status 看进度。
        """
        if mode not in ("local", "api", "mock"):
            raise RuntimeError("未知的模型类型")
        if not self._switch_lock.acquire(blocking=False):
            raise RuntimeError("正在切换模型，请稍候")
        try:
            if mode == "mock":
                self._commit(_MockBackend(), "mock", None)
            elif mode == "api":
                cand = dict(self._rt)
                if api_key and api_key.strip():
                    cand["api_key"] = api_key.strip()
                if api_base and api_base.strip():
                    cand["api_base"] = api_base.strip()
                if api_model and api_model.strip():
                    cand["api_model"] = api_model.strip()
                be = _ApiBackend(cand["api_base"], cand["api_key"], cand["api_model"])
                err = be.ping()
                if err:
                    raise RuntimeError(err)
                self._rt.update(cand)
                self._commit(be, "api", None)
            else:
                self._switching = True
                threading.Thread(target=self._switch_local, daemon=True).start()
                return self.info()
        except Exception:
            self._switch_lock.release()
            raise
        else:
            if mode != "local":
                self._switch_lock.release()
        return self.info()

    def _commit(self, backend, mode: str, error: str | None) -> None:
        self._rt["mode"] = mode
        self._backend, self._state, self._error = backend, "ready", error

    def _switch_local(self) -> None:
        t0 = time.time()
        try:
            be = _LocalBackend()
            self._commit(be, "local", None)
            self._load_seconds = round(time.time() - t0, 1)
        except Exception as e:
            self._error = f"切换到本地模型失败，已保持原模型：{type(e).__name__}: {e}"
        finally:
            self._switching = False
            self._switch_lock.release()

    def info(self) -> dict:
        key = self._rt["api_key"] or ""
        mode = self._backend.name if self._backend else self._resolve_mode()
        return {
            "mode": mode,
            "state": "loading" if self._switching else self._state,
            "switching": self._switching,
            "device": getattr(self._backend, "device", None),
            "load_seconds": self._load_seconds,
            "base_model": self._rt["api_model"] if mode == "api" else settings.llm_base_model,
            "adapter": settings.llm_adapter_path or None,
            "api_base": self._rt["api_base"],
            "api_model": self._rt["api_model"],
            "api_key_hint": (key[:3] + "…" + key[-4:]) if len(key) > 8 else ("已设置" if key else ""),
            "local_configured": bool(settings.llm_adapter_path),
            "error": self._error,
        }

    # -------------------------------------------------------------- 推理
    # 系统生成的提示/报错不是真实回答，混进历史会干扰模型
    _NOT_REAL = ("（模型正在加载", "（出错", "（请求失败", "（生成失败", "（API 调用失败", "【演示模式")

    @classmethod
    def clean_history(cls, history: list[dict] | None, limit: int = 30) -> list[dict]:
        """去掉系统提示/报错；若某条回答是系统提示，连同它对应的提问一起丢弃，避免连续两条 user。"""
        out: list[dict] = []
        for m in history or []:
            c = (m.get("content") or "").strip()
            role = m.get("role")
            if role not in ("user", "assistant") or not c:
                continue
            if c.startswith(cls._NOT_REAL):
                if role == "assistant" and out and out[-1]["role"] == "user":
                    out.pop()
                continue
            # 用户原话是病情依据，不截断；助手回复只截一下太长的部分（病情已被整理进“病情卡”）
            out.append({"role": role, "content": c if role == "user" else c[:1500]})
        out = out[-limit:]
        while out and out[0]["role"] != "user":   # 窗口切到一半时，保证以用户消息开头
            out.pop(0)
        return out

    @classmethod
    def build_messages(cls, history: list[dict] | None, user_text: str,
                       system: str | None = None, history_limit: int = 30) -> list[dict]:
        msgs = [{"role": "system", "content": system or SYSTEM_PROMPT}]
        msgs += cls.clean_history(history, limit=history_limit)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def stream(self, messages: list[dict], max_new_tokens: int | None = None) -> Iterator[str]:
        if self._backend is None:
            if self._state == "loading":
                yield "（模型正在加载中，请等左下角状态变为“已就绪”后再试）"
                return
            self._load()
        yield from self._backend.stream(messages, max_new_tokens)

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))


engine = LLMEngine()
