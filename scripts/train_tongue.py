"""训练舌苔分类模型（ResNet18），输出可直接给 TONGUE_WEIGHTS 用的权重。

数据集目录：每个类别一个文件夹，文件夹名就是类别名，例如
    dataset/
      白厚腻苔/ 001.jpg ...
      黑苔/ ...
也支持已经分好的   dataset/train/<类>/   dataset/val/<类>/   （val 也可叫 test）。
只有类别文件夹时，会按 --val-ratio 分层随机划分。

用法：
    python scripts/train_tongue.py --data D:/lingshu/tongue_coating_classify/dataset
    python scripts/train_tongue.py --data ... --epochs 40 --bs 32 --out models/tongue_resnet18.pth

产物：
    models/tongue_resnet18.pth           权重（torchvision resnet18 的 state_dict，fc 换成 N 类）
    models/tongue_resnet18.pth.classes.json   类别顺序（推理时自动读取，避免顺序对不上）
然后在 .env 里写  TONGUE_WEIGHTS=models/tongue_resnet18.pth  重启即可。

注意：
  * 如果数据集里有“同一张图做过增强后又存成多张”，请先去重/先划分再增强，否则验证集会和训练集泄漏，
    准确率虚高。
  * 颜色是舌象的主要信号，所以数据增强里不做色相/饱和度扰动，只做轻微亮度对比度变化。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["黑苔", "地图舌苔", "紫苔", "红苔黄厚腻苔", "红舌厚腻苔", "白厚腻苔"]   # 同 app/tongue/classifier.py


# Lingshu 数据集的文件夹是英文名，顺序与默认 6 类一一对应；识别到就自动换成中文类名
ALIASES = dict(zip(DEFAULT_CLASSES, [
    "black_tongue_coating", "map_tongue_coating", "purple_tongue_coating",
    "red_tongue_yellow_fur_thick_greasy", "red_tongue_thick_greasy", "white_tongue_thick_greasy"]))
_TO_ZH = {v: k for k, v in ALIASES.items()}


# ----------------------------------------------------------------- 纯 Python：找图、定类别顺序、划分
def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def class_dirs(root: Path) -> list[str]:
    """类别文件夹名（英文别名会换成中文类名）。"""
    return sorted(_TO_ZH.get(d.name, d.name) for d in root.iterdir() if d.is_dir() and list_images(d))


def decide_classes(found: list[str], preferred: list[str]) -> list[str]:
    """文件夹名与既定类别完全一致 → 沿用既定顺序；否则按文件夹名排序，并随权重一起记录。"""
    if set(found) == set(preferred):
        return list(preferred)
    return sorted(found)


def collect(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    out = []
    for i, c in enumerate(classes):
        folder = root / c if (root / c).is_dir() else root / ALIASES.get(c, c)
        if folder.is_dir():
            out += [(p, i) for p in list_images(folder)]
    return out


def stratified_split(items: list[tuple[Path, int]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by: dict[int, list] = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    tr, va = [], []
    for lst in by.values():
        rng.shuffle(lst)
        k = max(1, round(len(lst) * val_ratio)) if len(lst) > 1 else 0
        va += lst[:k]
        tr += lst[k:]
    return tr, va


def build_splits(data: Path, val_ratio: float, seed: int):
    sub = {d.name for d in data.iterdir() if d.is_dir()}
    val_name = next((n for n in ("val", "valid", "validation", "test") if n in sub), None)
    if "train" in sub and val_name:
        found = sorted(set(class_dirs(data / "train")) | set(class_dirs(data / val_name)))
        classes = decide_classes(found, DEFAULT_CLASSES)
        return classes, collect(data / "train", classes), collect(data / val_name, classes)
    classes = decide_classes(class_dirs(data), DEFAULT_CLASSES)
    tr, va = stratified_split(collect(data, classes), val_ratio, seed)
    return classes, tr, va


# ----------------------------------------------------------------- 训练
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据集根目录")
    ap.add_argument("--out", default=str(ROOT / "models" / "tongue_resnet18.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="Windows 下建议保持 0")
    ap.add_argument("--no-pretrained", action="store_true", help="不用 ImageNet 预训练（一般不建议）")
    a = ap.parse_args()

    data = Path(a.data)
    if not data.is_dir():
        sys.exit(f"数据集目录不存在：{data}")
    classes, train_items, val_items = build_splits(data, a.val_ratio, a.seed)
    if len(classes) < 2 or not train_items or not val_items:
        sys.exit("没有找到足够的数据：需要至少 2 个类别文件夹，且训练/验证集都不为空。")

    print(f"类别（{len(classes)}）：{classes}")
    cnt = Counter(c for _, c in train_items)
    print("训练集每类张数：", {classes[i]: cnt.get(i, 0) for i in range(len(classes))}, "| 验证集共", len(val_items))
    if set(classes) != set(DEFAULT_CLASSES):
        print("提示：数据集的类别名与 classifier.py 里的默认 6 类不同，类别顺序会写入 .classes.json，推理时自动使用；"
              "界面上的说明文字只对默认类别有预设，其他类别会显示为空。")

    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", dev)

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        # 与推理端一致
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomRotation(10), T.ColorJitter(brightness=0.15, contrast=0.15),
                          T.ToTensor(), norm])
    tf_val = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])           # 与推理端一致

    class DS(torch.utils.data.Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            return self.tf(Image.open(p).convert("RGB")), y

    # 类别不均衡：按类别频率的倒数做损失权重
    w = torch.tensor([1.0 / max(cnt.get(i, 0), 1) for i in range(len(classes))], dtype=torch.float)
    w = (w / w.sum() * len(classes)).to(dev)

    dl_tr = torch.utils.data.DataLoader(DS(train_items, tf_train), batch_size=a.bs, shuffle=True,
                                        num_workers=a.workers, drop_last=len(train_items) > a.bs)
    dl_va = torch.utils.data.DataLoader(DS(val_items, tf_val), batch_size=a.bs, num_workers=a.workers)

    try:
        m = torchvision.models.resnet18(weights=None if a.no_pretrained
                                        else torchvision.models.ResNet18_Weights.DEFAULT)
    except Exception as e:                                                  # 下载预训练权重失败（无网络等）
        print(f"预训练权重下载失败（{type(e).__name__}），改为随机初始化——效果会明显变差，建议联网重试。")
        m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
    m = m.to(dev)

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    def evaluate():
        m.eval()
        n = len(classes)
        conf = [[0] * n for _ in range(n)]                                   # conf[真实][预测]
        with torch.no_grad():
            for x, y in dl_va:
                pred = m(x.to(dev)).argmax(1).cpu()
                for t, p in zip(y.tolist(), pred.tolist()):
                    conf[t][p] += 1
        tot = sum(map(sum, conf))
        acc = sum(conf[i][i] for i in range(n)) / max(tot, 1)
        f1s = []
        for i in range(n):
            tp = conf[i][i]
            fp = sum(conf[r][i] for r in range(n)) - tp
            fn = sum(conf[i]) - tp
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return acc, sum(f1s) / n, conf

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        m.train()
        run = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(m(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
        sched.step()
        acc, f1, conf = evaluate()
        mark = ""
        if f1 > best:                                                        # 以宏平均 F1 选最佳，避免多数类拉高准确率
            best = f1
            torch.save(m.state_dict(), out)
            out.with_name(out.name + ".classes.json").write_text(
                json.dumps(classes, ensure_ascii=False), encoding="utf-8")
            best_conf = conf
            mark = "  ← 保存"
        print(f"epoch {ep:>3}/{a.epochs}  loss {run / len(train_items):.4f}  val_acc {acc:.3f}  macro_F1 {f1:.3f}{mark}")

    print(f"\n最佳 macro_F1 = {best:.3f}，权重已保存：{out}")
    print("混淆矩阵（行=真实，列=预测）：")
    print("      " + " ".join(f"{c[:4]:>6}" for c in classes))
    for i, row in enumerate(best_conf):
        rec = row[i] / max(sum(row), 1)
        print(f"{classes[i][:4]:>6}" + " ".join(f"{v:>6}" for v in row) + f"   召回 {rec:.2f}")
    print(f"\n下一步：在 .env 里加  TONGUE_WEIGHTS={out}  然后重启服务。")


if __name__ == "__main__":
    main()
