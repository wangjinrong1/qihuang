"""药方识别流程：（OCR）→ 解析 → 禁忌检查 / 方剂比对 → 大模型解读"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from app.llm.engine import engine
from app.llm.prompts import PRESCRIPTION_TEMPLATE
from app.prescription import ocr, parser
from app.rag.retriever import get_retriever


def run(text: str, image_path: Path | None = None) -> Iterator[dict]:
    if image_path:
        yield {"type": "stage", "label": "识别图片文字"}
        try:
            text = ocr.read_text(image_path)
        except ocr.OCRUnavailable as e:
            yield {"type": "error", "message": str(e)}
            return
        yield {"type": "text", "text": text}
        if not text.strip():
            yield {"type": "error", "message": "图片中没有识别到文字，请换更清晰的照片，或直接粘贴文字"}
            return

    yield {"type": "stage", "label": "解析药材与剂量"}
    result = parser.parse(text)
    yield {"type": "parsed", **result}
    if not result["herbs"]:
        yield {"type": "error", "message": "没有识别到药材，请检查文字（示例：柴胡12g 黄芩9g 半夏9g）"}
        return

    yield {"type": "stage", "label": "生成解读"}
    herbs = "；".join(
        f"{h['name']}{_dose(h)}{('（' + h['note'] + '）') if h['note'] else ''}"
        f"[{h['nature']}·{h['flavor']}·归{h['meridian']}]" for h in result["herbs"])
    conflicts = "；".join(f"{c['a']}与{c['b']}（{c['rule']}）" for c in result["conflicts"]) or "未发现十八反/十九畏"
    matches = "；".join(f"{m['name']}（相似度{m['similarity']:.0%}）" for m in result["matches"]) or "无明显相近方剂"
    hits = get_retriever().search("配伍禁忌 君臣佐使 " + " ".join(h["base"] for h in result["herbs"]), k=2)
    prompt = PRESCRIPTION_TEMPLATE.format(
        herbs=herbs, conflicts=conflicts, matches=matches,
        rag="\n".join(h["text"] for h in hits) or "（无）")
    for tok in engine.stream(engine.build_messages(None, prompt), max_new_tokens=600):
        yield {"type": "token", "text": tok}
    yield {"type": "done"}


def _dose(h: dict) -> str:
    return f"{h['dose']:g}{h['unit']}" if h.get("dose") is not None else ""
