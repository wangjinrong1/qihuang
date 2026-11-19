"""药方图片文字识别（OCR）。依赖 rapidocr-onnxruntime，未安装时给出明确提示。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class OCRUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise OCRUnavailable("未安装 OCR 组件：pip install rapidocr-onnxruntime；或直接粘贴药方文字") from e
    return RapidOCR()


def read_text(path: Path) -> str:
    """按从上到下、从左到右的顺序返回识别出的文字行。"""
    result, _ = _engine()(str(path))
    if not result:
        return ""
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    return "\n".join(r[1] for r in rows)


def available() -> bool:
    try:
        _engine()
        return True
    except OCRUnavailable:
        return False
