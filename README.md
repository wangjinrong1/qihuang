# 岐黄智诊 · 中医智能辅助平台

把「中医大模型 + 知识图谱 + 古籍检索 + 舌象分析 + 药方/中药识别 + 3D 经络图谱」整合在一起的 Web 应用。后端 FastAPI，前端纯静态页面（无需 Node、无需构建）。

> 本项目输出仅供学习与辅助参考，**不构成医疗建议**，不能替代执业医师的诊断。

**三种运行深度，按需选择：**

| 档位 | 需要什么 | 效果 |
|---|---|---|
| ① 演示模式 | 只装 `requirements.txt` | 所有页面和流程都能走通，但回答和舌象结果都是占位内容 |
| ② 云端 API 模式 | ① + 一个 OpenAI 兼容接口的 Key（如 DeepSeek） | 不用下载模型、不用 torch，效果最好、门槛最低 |
| ③ 本地模型模式 | ① + torch + 下载基座与适配器权重 | 完全离线（联网检索除外），需要几 GB 磁盘 |

---

## 目录

- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [一、拉取代码与安装环境](#一拉取代码与安装环境)
- [二、权重与数据：仓库里有什么、没有什么](#二权重与数据仓库里有什么没有什么)
- [三、配置大模型](#三配置大模型)
- [四、训练舌苔分类模型](#四训练舌苔分类模型)
- [五、启动与使用](#五启动与使用)
- [六、配置项总表（.env）](#六配置项总表env)
- [七、测试](#七测试)
- [八、常见问题](#八常见问题)
- [功能模块与工作方式](#功能模块与工作方式)
- [接口](#接口)
- [声明与已知限制](#声明与已知限制)

---

## 目录结构

```
qihuang/
├── run.py                      # 启动入口
├── check_llm.py                # 大模型自检（不启动网页，直接加载并生成一句话）
├── requirements.txt            # 基础依赖（演示模式够用）
├── requirements-llm.txt        # 本地大模型依赖（torch / transformers / peft …）
├── .env.example                # 配置模板，复制为 .env 后修改
├── 启动说明.txt                 # Windows 下的命令速查
├── app/                        # 后端
│   ├── main.py                 #   接口入口
│   ├── config.py               #   读取 .env / 环境变量
│   ├── consult/                #   问诊与辨证：dialogue(多轮状态机) chat(问答) pipeline(辨证报告) guard(信息闸门与防编造)
│   ├── llm/                    #   engine(本地/云端/演示切换) prompts(提示词)
│   ├── kg/graph.py             #   知识图谱（症状→证型→方剂→中药，含十八反十九畏检查）
│   ├── rag/retriever.py        #   古籍检索（BM25，首次启动自动建索引）
│   ├── tongue/classifier.py    #   舌苔分类推理（ResNet18）
│   ├── herb/recognizer.py      #   中药图片识别 + 药名查询
│   ├── prescription/           #   药方 OCR、解析、比对
│   └── web/search.py           #   联网检索
├── web/                        # 前端静态页面；atlas/ 是独立的 3D 经络图谱
├── data/
│   ├── kg/                     #   知识图谱 csv + 常见病证表 common_conditions.json
│   └── corpus/                 #   古籍语料 txt
├── scripts/
│   ├── download_model.py       #   下载基座模型 + LoRA 适配器
│   └── train_tongue.py         #   训练舌苔分类（推荐使用的训练脚本）
├── tongue_coating_classify/    # 早期舌苔训练实验脚本与结果曲线（仅供参考，见第四节）
├── tests/                      # 自动测试
├── models/                     # 权重与缓存放这里（不入库，仅保留 .gitkeep）
└── uploads/                    # 运行时上传的图片（不入库，仅保留 .gitkeep）
```

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.10 及以上（推荐 3.10 ~ 3.12） |
| 系统 | Windows / Linux / macOS 均可（命令里分别给出了写法） |
| 磁盘 | 演示或 API 模式约 1 GB 以内；本地模型模式另需约 4 GB（基座约 3.5 GB + 适配器）和 torch 本身的体积 |
| 显卡 | 非必需。有 NVIDIA 显卡会自动用 GPU，没有则用 CPU（本地模型在 CPU 上会比较慢） |
| 网络 | 下载依赖和权重需要联网；国内下载 HuggingFace 权重请加 `--mirror` |

---

## 一、拉取代码与安装环境

### 1. 克隆仓库

```bash
git clone https://github.com/wangjinrong1/qihuang.git
cd qihuang
```

### 2. 创建并激活虚拟环境

**Windows（PowerShell）**

```powershell
python -m venv .venv
# 首次使用若提示"禁止运行脚本"，先执行下面这行（只对当前用户生效）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装基础依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

包含 FastAPI、Uvicorn、Pillow，以及药方图片识别用的 `rapidocr-onnxruntime`（不装 OCR 也能用，只是只能粘贴药方文字）。

### 4. 先跑起来看看（演示模式）

```bash
python run.py
```

浏览器打开 <http://127.0.0.1:8000>。此时左下角状态为演示模式，所有页面都能点，但模型回答、舌象结果是占位的，**不能当真**。确认能打开后，再按需继续第三、四节。

---

## 二、权重与数据：仓库里有什么、没有什么

仓库只放代码和小体积数据，**权重不在仓库里**（单文件超过 GitHub 的 100 MB 限制，也没必要入库）。

| 内容 | 在仓库里？ | 怎么获得 |
|---|---|---|
| 知识图谱 `data/kg/*` | ✅ 已包含 | — |
| 古籍语料 `data/corpus/*.txt` | ✅ 已包含 | — |
| 3D 经络图谱及人体模型 | ✅ 已包含 | — |
| 古籍检索索引 `models/rag_index.pkl` | ❌ | 首次启动自动生成并缓存（约十几秒，缓存约 93 MB），语料变动时自动重建 |
| 大模型基座（Qwen1.5-1.8B-Chat） | ❌ | `python scripts/download_model.py`，见第三节 |
| 中医 LoRA 适配器 | ❌ | 同上，示例仓库：`CMLM/ZhongJing-2-1_8b` |
| 舌苔分类权重 `models/tongue_resnet18.pth` | ❌ | 用自己的数据集训练，见第四节 |
| 舌苔数据集 | ❌ | 需自行准备（含 6 类舌苔图片，目录格式见第四节） |
| 中药图片识别权重 | ❌ | 自备（可选），见第三节末尾 |
| `.env`（含 API Key） | ❌ | 复制 `.env.example` 自行填写，**切勿上传** |

---

## 三、配置大模型

先复制配置模板：

```bash
# Windows PowerShell
Copy-Item .env.example .env
# Linux / macOS
cp .env.example .env
```

`LLM_MODE` 取值：

- `auto`（默认）：配了 `LLM_ADAPTER_PATH` → 本地；否则配了 `LLM_API_KEY` → 云端；都没配 → 演示模式
- `local`：本地基座 + LoRA 适配器
- `api`：云端 API
- 演示模式没有单独取值，什么都没配就是演示模式

### 方案 A：云端 API（最省事）

在 `.env` 里写：

```ini
LLM_MODE=api
LLM_API_BASE=https://api.deepseek.com
LLM_API_KEY=你的密钥
LLM_API_MODEL=deepseek-chat
```

不需要下载模型，也不需要 torch。任何 OpenAI 兼容接口都可以，改 `LLM_API_BASE` 和 `LLM_API_MODEL` 即可。

> 使用云端 API 时，问诊内容会发送到第三方服务器，**请勿输入真实姓名、证件号等隐私信息**。

也可以不写进 `.env`，启动后在页面的“模型”按钮里临时填 Key（只保存在进程内存，不写磁盘）。

### 方案 B：本地中医大模型

**1）安装 PyTorch**

- 只用 CPU：直接装即可
  ```bash
  python -m pip install torch torchvision
  ```
- NVIDIA 显卡：先到 <https://pytorch.org> 选你的系统和 CUDA 版本，复制它生成的安装命令。形如：
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
  （`cu128` 要换成你实际的 CUDA 版本。作者本机用的是 `torch==2.14.1`、`torchvision==0.29.1` 的 `cu132` 版本。）

装完可以确认一下：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**2）安装其余推理依赖**

```bash
python -m pip install -r requirements-llm.txt
```

> 注意：`requirements-llm.txt` 里的 `torch` 一行，如果你已经按上一步装好了 CUDA 版，pip 会认为已满足，不会覆盖。

**3）下载基座与适配器到 `models/`**

```bash
python scripts/download_model.py --adapter CMLM/ZhongJing-2-1_8b --mirror
```

- `--adapter`：必填，LoRA 适配器的 HuggingFace 仓库名
- `--base`：基座模型，默认 `Qwen/Qwen1.5-1.8B-Chat`
- `--mirror`：使用 `hf-mirror.com` 镜像，国内网络推荐；海外网络去掉即可
- 基座下载到 `models/base`，适配器下载到 `models/adapter`；基座约 3.5 GB，下载较慢，断了重跑会续传

**4）修改 `.env`**

```ini
LLM_MODE=local
LLM_BASE_MODEL=models/base
LLM_ADAPTER_PATH=models/adapter
LLM_DEVICE=auto          # auto | cpu | cuda
LLM_MAX_NEW_TOKENS=768   # CPU 上建议 300~400，否则很慢
```

**5）自检**

```bash
python check_llm.py
```

脚本会加载模型、生成一句话，并打印设备、加载耗时和生成速度。加载失败会给出原因；如果显示“当前是演示模式”，说明 `.env` 没配对。

### 可选：中药图片识别权重

中药图片识别需要**自备**训练好的权重和类别文件，不配则为演示结果（按药名查询性味归经等功能不受影响）：

```ini
HERB_WEIGHTS=models/herb.pt        # TorchScript .pt，或 torchvision 的 state_dict
HERB_LABELS=models/herb_labels.txt # 每行一个药名，顺序与训练时一致
HERB_ARCH=resnet50                 # 权重是 state_dict 时需要指定网络结构
```

---

## 四、训练舌苔分类模型

不训练也能运行，只是舌象分析是演示结果（按图片哈希随机给一个类别，仅用于联调界面）。训练后得到真实的 ResNet18 六分类。

### 1. 分类类别（默认 6 类）

黑苔、地图舌苔、紫苔、红苔黄厚腻苔、红舌厚腻苔、白厚腻苔。

### 2. 准备数据集

每个类别一个文件夹，文件夹名就是类别名：

```
dataset/
├── 白厚腻苔/    001.jpg …
├── 黑苔/        …
└── …（共 6 个文件夹）
```

也支持已经分好的结构：`dataset/train/<类>/` 与 `dataset/val/<类>/`（`val` 也可以叫 `valid` / `validation` / `test`）。

如果文件夹是英文名（`black_tongue_coating`、`map_tongue_coating`、`purple_tongue_coating`、`red_tongue_yellow_fur_thick_greasy`、`red_tongue_thick_greasy`、`white_tongue_thick_greasy`），脚本会自动换成中文类名，顺序与默认 6 类一致。

**⚠ 防止验证集泄漏**：如果数据集里有“同一张图做过增强后又存成多份”的副本，必须先去重，或者先划分训练/验证再增强，否则验证集和训练集会重叠，准确率会虚高。

### 3. 开始训练

```bash
python scripts/train_tongue.py --data D:/你的数据集目录
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | 必填 | 数据集根目录 |
| `--out` | `models/tongue_resnet18.pth` | 权重输出路径 |
| `--epochs` | 30 | 训练轮数 |
| `--bs` | 32 | 批大小，显存不够就调小 |
| `--lr` | 3e-4 | 学习率（AdamW + 余弦退火） |
| `--val-ratio` | 0.2 | 只有类别文件夹时，按此比例分层随机划分验证集 |
| `--seed` | 42 | 随机种子 |
| `--workers` | 0 | DataLoader 进程数，Windows 下建议保持 0 |
| `--no-pretrained` | 关 | 不使用 ImageNet 预训练（一般不建议） |

训练细节：以 ImageNet 预训练的 ResNet18 微调；损失按类别频率倒数加权并带 label smoothing，缓解类别不均衡；数据增强不做色相/饱和度扰动（颜色是舌象的主要信号）；**以验证集宏平均 F1 选最佳轮次保存**，训练结束会打印混淆矩阵和每类召回。首次训练需要联网下载预训练权重；下载失败会退化为随机初始化，效果明显变差，请联网重试。

### 4. 产物与接入

训练完成后 `models/` 下会有：

- `tongue_resnet18.pth`：权重（`state_dict`，`fc` 层为 N 类）
- `tongue_resnet18.pth.classes.json`：类别顺序（推理时自动读取，避免顺序对不上）

在 `.env` 里写：

```ini
TONGUE_WEIGHTS=models/tongue_resnet18.pth
```

重启服务即可。如果训练用了自定义类别名，类别顺序会随权重记录，推理自动使用，但界面上的说明文字只对默认 6 类有预设。

> 这只是辅助参考：对非舌象图片也会强行给出某个类别，置信度很低时请忽略结果。

### 关于 `tongue_coating_classify/`

这是早期的训练/测试实验（`train/train.py`、`test.py`，以及训练曲线和日志），保留作参考。它的类别文件夹用英文名，数据划分和增强写法与 `scripts/train_tongue.py` 不同，**要得到可直接接入应用的权重，请使用 `scripts/train_tongue.py`**。

---

## 五、启动与使用

```bash
python run.py
```

默认监听 <http://127.0.0.1:8000>，在 `.env` 里用 `HOST` / `PORT` 修改。`HOST=0.0.0.0` 会让局域网内其他设备也能访问，请确认网络环境可信，且不要在公网暴露。

首次启动：

1. 构建古籍检索索引（约十几秒），之后读缓存约 2 秒；
2. 若配置了本地模型，会后台预加载（`LLM_PRELOAD=1`），日志里会出现 `Loading weights` 进度条；
3. 页面左下角状态栏会依次显示“模型加载中…”→“中医大模型已就绪”。加载失败会自动回退到演示模式，原因显示在状态栏的悬浮提示里。

访问页面后：

- **智能问诊**：多轮对话，信息够了再给辨证小结
- **辨证报告**：填写患者信息（可上传舌象图片）生成报告，可下载/打印
- **药方识别**：粘贴药方文字或上传处方照片
- **中药识别**：上传药材图片，或按药名查询
- **知识库**：证型/方剂浏览、古籍资料检索
- **经络图谱**：3D 人体穴位图

**运行时切换模型**：左下角状态栏或问诊页顶部的“模型：…⚙”按钮，可在 云端 API / 本地中医模型 / 演示模式 之间切换。点“测试并切换”会先发一次连通性测试，失败则保持原模型并提示原因（Key 无效、余额不足等）。

**联网检索**：问诊页的开关，在出结论 / 追问 / 知识问答阶段用症状或证型关键词检索网页摘要，作为“参考资料”交给模型，回答里会标明“据网络资料”并显示来源链接。信息收集阶段与紧急情况不联网。建议配置 `SERPER_API_KEY`（serper.dev 有免费额度）以获得稳定结果；免 Key 的 Bing / DuckDuckGo 靠解析网页，页面样式变化时可能失效，失败会自动退回本地资料。

---

## 六、配置项总表（.env）

所有配置都从项目根目录的 `.env` 读取（也可用同名环境变量）。以 `.env.example` 为准，这里是说明：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODE` | `auto` | `auto` / `local` / `api` |
| `LLM_BASE_MODEL` | `Qwen/Qwen1.5-1.8B-Chat` | 基座，HuggingFace 名或本地目录（用下载脚本后填 `models/base`） |
| `LLM_ADAPTER_PATH` | 空 | LoRA 适配器目录或仓库名；空则不走本地模型 |
| `LLM_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `LLM_MAX_NEW_TOKENS` | `768` | 单次回答最长 token 数，CPU 建议 300~400 |
| `LLM_TEMPERATURE` | `0.7` | 采样温度 |
| `LLM_REPETITION_PENALTY` | `1.15` | 本地小模型防重复，1.1~1.3，过大会胡言乱语 |
| `LLM_PRELOAD` | `1` | 启动时是否后台预加载模型 |
| `LLM_API_BASE` | `https://api.deepseek.com` | 云端接口地址（OpenAI 兼容） |
| `LLM_API_KEY` | 空 | 云端密钥 |
| `LLM_API_MODEL` | `deepseek-chat` | 云端模型名，以服务商文档为准 |
| `LLM_API_TIMEOUT` | `120` | 云端请求超时（秒） |
| `TONGUE_WEIGHTS` | 空 | 舌苔分类权重路径；空则演示 |
| `HERB_WEIGHTS` / `HERB_LABELS` / `HERB_ARCH` | 空 / 空 / `resnet50` | 中药图片识别 |
| `WEB_SEARCH_PROVIDER` | `auto` | `auto` / `serper` / `bing` / `duckduckgo` |
| `SERPER_API_KEY` | 空 | Serper（Google）搜索密钥 |
| `WEB_SEARCH_TIMEOUT` | `6` | 联网检索超时（秒） |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 监听地址与端口 |

---

## 七、测试

测试使用演示后端，**不需要模型、不需要 torch**。

```bash
python -m pip install pytest
python -m pytest tests -q
```

| 文件 | 内容 |
|---|---|
| `tests/test_dialogue.py` | 多轮问诊状态机（也可 `python tests/test_dialogue.py` 直接运行） |
| `tests/test_flow_e2e.py` | 问诊端到端流程（也可直接 `python` 运行） |
| `tests/test_guard.py` | 辨证报告的信息闸门与防编造检查 |
| `tests/test_switch_web.py` | 联网检索失败时的降级 |

---

## 八、常见问题

**PowerShell 提示“禁止运行脚本”**
执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，再激活虚拟环境。

**下载模型很慢或超时**
国内网络加 `--mirror`；也可以手动下载模型文件放进 `models/base`、`models/adapter`，再按第三节填 `.env`。

**页面左下角一直是“演示模式”**
`.env` 没生效或路径写错。先跑 `python check_llm.py`，它会打印当前读到的配置和失败原因。注意 `.env` 要放在项目根目录，和 `run.py` 同级。

**本地模型回答很慢**
CPU 推理本来就慢。把 `LLM_MAX_NEW_TOKENS` 调到 300~400，或改用云端 API 模式。

**有显卡却在用 CPU**
`python -c "import torch; print(torch.cuda.is_available())"` 若为 `False`，说明装的是 CPU 版 torch，需按 pytorch.org 重装 CUDA 版；或者 `.env` 里 `LLM_DEVICE` 被设成了 `cpu`。

**药方图片识别提示“未安装 OCR 组件”**
`python -m pip install rapidocr-onnxruntime`，或直接粘贴药方文字。

**更新了前端文件，页面还是旧的**
浏览器缓存，按 `Ctrl+F5` 强制刷新。日志里的 `favicon.ico 404` 无影响。

**端口被占用**
修改 `.env` 里的 `PORT`，或关闭占用 8000 端口的程序。

**古籍检索结果不更新**
索引缓存在 `models/rag_index.pkl`，语料变动会自动重建；若仍异常，删掉该文件重启即可。

**推送代码到 GitHub 前**
`.env`、`.venv/`、`models/`、`uploads/`、舌苔数据集都已在 `.gitignore` 中；提交前用 `git status` 再核对一遍，确认没有密钥、权重和上传的舌象图片。

---

## 功能模块与工作方式

| 模块 | 位置 | 说明 |
|---|---|---|
| 智能问诊 | `app/consult/chat.py`、`dialogue.py` | 症状识别 + 图谱检索 + 古籍检索 → 作为上下文交给大模型，流式输出 |
| 辨证报告 | `app/consult/pipeline.py`、`guard.py` | 患者信息 → 舌象 → 图谱辨证 → 分节检索古籍 → 分节生成 → 汇总 Markdown 报告 |
| 知识图谱 | `app/kg/graph.py` | 症状→证型→方剂→中药，含十八反/十九畏检查，数据在 `data/kg/` |
| 古籍检索 | `app/rag/retriever.py` | BM25 零依赖检索，语料在 `data/corpus/*.txt` |
| 舌象分析 | `app/tongue/classifier.py` | ResNet18 六分类，未配权重时为演示结果 |
| 药方识别 | `app/prescription/` | 文字/照片(OCR) → 提取药名剂量煎法 → 十八反十九畏检查 → 相近方剂比对 → 大模型解读 |
| 中药识别 | `app/herb/recognizer.py` | 药材图片分类（需权重）+ 按名称查性味归经、所在方剂、禁忌 |
| 经络图谱 | `web/atlas/` | 3D 人体穴位图，作为独立页面嵌入 |
| 大模型 | `app/llm/engine.py` | 本地(基座+LoRA) / 云端 API / 演示 三种后端 |

### 智能问诊（多轮）

参考灵枢智联“先收集信息、够了再诊断”的思路，`app/consult/dialogue.py` 的状态机每轮根据完整对话历史判断阶段：

| 阶段 | 触发 | 行为 |
|---|---|---|
| `urgent` | 本轮出现红旗症状（呼吸困难、胸痛、咯血…） | 先提示尽快就医 |
| `qa` | 纯知识问题（药、穴位、概念） | 直接回答，附古籍参考 |
| `collect` | 在描述症状但关键信息不够 | 复述已知要点 + 每次只问一个问题，不下结论 |
| `conclude` | 关键信息齐了 / 追问达到上限 / 用户说“直接给结论”或点按钮 | 输出“辨证小结” |
| `followup` | 已给过小结后的追问 | 结合全部对话回答，必要时调整判断 |

- 常见病证知识表：`data/kg/common_conditions.json`（目前覆盖感冒/咳嗽/咽痛等外感肺系，含证型线索、红旗症状、调护与禁忌）。新增病类只需在 `categories` 里加一项（`triggers` 触发词、`slot_order/required/optional` 要问的信息、`patterns` 证型线索）。
- 古籍语料：`data/corpus/` 下放 `.txt`（格式：`<篇名>` 分节），首次启动自动建 BM25 索引并缓存；检索带相关度门槛，宁缺毋滥。

### 辨证报告的信息闸门与防编造

辨证报告由 `app/consult/guard.py` 把关：

- **信息不足不出报告**：主诉、病程等关键信息过少时（例如只有“喉咙痛、两天”），会被拦下并提示需要补充哪些信息，而不是硬生成一份报告。
- **防编造**：模型生成的每一句话会与患者实际输入、舌象结果和知识图谱匹配；凭空出现的体温、脉象、舌苔颜色、“遂就诊”“查体”等没有依据的内容会被标出并删除。
- **演示模式不出报告**：未接入模型时，报告页只提示“演示模式：尚未接入模型”，不会输出伪造的辨证思路。
- 报告正文用 `[编号]` 引用古籍，文末列“参考资料”和“名词速查”（药材、穴位的搜索入口，不构成用药或操作建议）。

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 运行状态（模型、联网、舌象、OCR、图谱等） |
| POST | `/api/llm/switch` | 运行时切换模型 |
| POST | `/api/chat` | 问答（SSE） |
| POST | `/api/consult` | 辨证流程（SSE，表单：`patient` JSON + 可选 `image`） |
| POST | `/api/tongue` | 单独舌象分析 |
| POST | `/api/prescription` | 药方识别（SSE，表单：`text` 和/或 `image`） |
| POST | `/api/herb/recognize` | 药材图片识别 |
| GET | `/api/herb/{名称}`、`/api/herb/search?q=` | 药材详情、药名联想 |
| GET | `/api/kg/syndromes`、`/api/kg/syndrome/{id}`、`/api/kg/formulas` | 知识库浏览 |
| GET | `/api/corpus/books`、`/api/corpus/book`、`/api/corpus/search` | 古籍书目、书内浏览、全库检索 |

启动后访问 <http://127.0.0.1:8000/docs> 可查看 FastAPI 自动生成的接口文档。

---

## 声明与已知限制

本项目输出仅供学习与辅助参考，不构成医疗建议。3D 人体网格来自 CC0 资产，见 `web/atlas/HUMAN-MODEL-LICENSE.txt`；Three.js 许可见 `web/atlas/THREE-LICENSE.txt`。

- 药材性味归经来自机器翻译的词表，已在读取时做术语规范化，但仍需对照药典校对。
- 图谱中的“方剂”数据来自病例汇总，部分方剂药味过多、并非经典组成，相近方剂比对仅供参考。
- 舌象分类是 6 类辅助判断，对非舌象图片也会强行给出类别；中药图片识别需自备权重。
- 本地 1.8B 小模型的医学能力有限，输出需要人工审核；云端模型效果更好，但会把问诊内容发送给第三方。
- 目前常见病证知识表只覆盖外感肺系（感冒/咳嗽/咽痛等），其他病类的多轮问诊信息收集较弱。
