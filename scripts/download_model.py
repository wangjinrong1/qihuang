"""下载基座模型与 LoRA 适配器到 models/ 目录。

用法（国内网络建议加 --mirror）：
    python scripts/download_model.py --adapter <适配器的 HuggingFace 仓库名> --mirror
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen1.5-1.8B-Chat", help="基座模型仓库名")
ap.add_argument("--adapter", required=True, help="中医适配器仓库名")
ap.add_argument("--mirror", action="store_true", help="使用 hf-mirror.com 镜像（国内推荐）")
args = ap.parse_args()

if args.mirror:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("缺少 huggingface_hub，请先：python -m pip install -r requirements-llm.txt")

for label, repo, sub in (("基座模型", args.base, "base"), ("适配器", args.adapter, "adapter")):
    dest = ROOT / "models" / sub
    print(f"\n==> 下载{label}：{repo}\n    保存到 {dest}")
    snapshot_download(repo_id=repo, local_dir=str(dest))

print("\n下载完成。请在 .env 中填写：\n")
print("LLM_BASE_MODEL=models/base")
print("LLM_ADAPTER_PATH=models/adapter")
