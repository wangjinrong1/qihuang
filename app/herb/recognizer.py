"""中药图片识别：图像分类（可选 torch）。

未配置权重时返回演示候选（仅用于联调界面，明确标注，不具备识别能力）。
配置方式见 .env：HERB_WEIGHTS（TorchScript .pt 或 torchvision state_dict）+ HERB_LABELS（类别名）。
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.kg.graph import get_graph


class _TorchHerb:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.labels = [l.strip() for l in Path(settings.herb_labels).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w = settings.herb_weights
        try:
            self.model = torch.jit.load(w, map_location=self.device)
        except Exception:  # 不是 TorchScript → 当作 torchvision state_dict
            import torchvision

            m = getattr(torchvision.models, settings.herb_arch)(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, len(self.labels))
            m.load_state_dict(torch.load(w, map_location=self.device, weights_only=True))
            self.model = m
        self.model = self.model.to(self.device).eval()

    def predict(self, path: Path, k: int) -> list[tuple[str, float]]:
        import torchvision.transforms as T
        from PIL import Image

        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(x), dim=1)[0]
            conf, idx = self.torch.topk(p, min(k, len(self.labels)))
        return [(self.labels[i], c) for c, i in zip(conf.tolist(), idx.tolist())]


@lru_cache(maxsize=1)
def _load():
    if settings.herb_weights and settings.herb_labels and Path(settings.herb_weights).exists() \
            and Path(settings.herb_labels).exists():
        try:
            return _TorchHerb(), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, None


def recognize(image_path: Path, k: int = 3) -> dict:
    model, err = _load()
    if model:
        cands = [{"name": n, "confidence": round(c, 3)} for n, c in model.predict(image_path, k)]
        demo = False
    else:
        names = sorted(get_graph().herb_names())
        h = int(hashlib.md5(image_path.read_bytes()).hexdigest(), 16)
        cands = [{"name": names[(h >> (8 * i)) % len(names)], "confidence": 0.0} for i in range(k)]
        demo = True
    return {"demo": demo, "candidates": cands, "error": err}


def status() -> dict:
    model, err = _load()
    return {"mode": "torch" if model else "demo", "error": err}
