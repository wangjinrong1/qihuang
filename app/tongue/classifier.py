"""舌象分析：ResNet18 六分类（可选）；未配置权重时返回演示结果。

类别与说明可在 TONGUE_NOTES 中继续完善；接入多模态大模型做描述时，
在 analyze() 里补一步即可。
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from app.config import settings

CLASS_NAMES = ["黑苔", "地图舌苔", "紫苔", "红苔黄厚腻苔", "红舌厚腻苔", "白厚腻苔"]

TONGUE_NOTES = {
    "黑苔": "苔黑，润燥不同意义不同：润者多阳虚寒盛，燥者多热极伤阴，需结合全身症状。",
    "地图舌苔": "舌苔剥落如地图，多提示胃阴不足或气阴两虚。",
    "紫苔": "舌苔偏紫，多与血行不畅、寒凝或热郁血瘀有关。",
    "红苔黄厚腻苔": "舌红苔黄厚腻，多为湿热蕴结。",
    "红舌厚腻苔": "舌红苔厚腻，多为热象兼痰湿。",
    "白厚腻苔": "苔白厚腻，多为寒湿或痰湿困脾。",
}


def _load_classes(weights: str) -> list[str]:
    """训练脚本会在权重旁写 <权重>.classes.json（类别顺序）；有就用它，没有就用默认 6 类。"""
    f = Path(weights + ".classes.json")
    if f.exists():
        names = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(names, list) and len(names) >= 2:
            return [str(n) for n in names]
    return list(CLASS_NAMES)


class _TorchClassifier:
    def __init__(self, weights: str) -> None:
        import torch
        import torchvision

        self.classes = _load_classes(weights)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, len(self.classes))
        m.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model = m.to(self.device).eval()

    def predict(self, path: Path) -> tuple[str, float]:
        import torchvision.transforms as T
        from PIL import Image

        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(x), dim=1)
            conf, idx = self.torch.max(p, 1)
        return self.classes[idx.item()], conf.item()


@lru_cache(maxsize=1)
def _load():
    w = settings.tongue_weights
    if w and Path(w).exists():
        try:
            return _TorchClassifier(w), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, None


def analyze(image_path: Path) -> dict:
    clf, err = _load()
    if clf:
        label, conf = clf.predict(image_path)
        demo = False
    else:  # 演示模式：按文件内容哈希稳定地给出一个类别，仅用于联调界面
        h = int(hashlib.md5(image_path.read_bytes()).hexdigest(), 16)
        label, conf, demo = CLASS_NAMES[h % len(CLASS_NAMES)], 0.0, True
    return {
        "label": label,
        "confidence": round(conf, 3),
        "note": TONGUE_NOTES.get(label, ""),
        "demo": demo,
        "error": err,
    }


def status() -> dict:
    clf, err = _load()
    return {"mode": "torch" if clf else "demo", "error": err}
