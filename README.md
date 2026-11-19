# 岐黄智诊 · 中医智能辅助平台（框架版）

一个把「中医大模型 + 知识图谱 + 古籍检索 + 舌象分析 + 3D 经络图谱」整合在一起的 Web 应用。
当前为**可运行的框架**：未接入模型时自动进入演示模式，所有页面和流程都能完整走通。

## 快速开始

```bash
pip install -r requirements.txt
python run.py            # 打开 http://127.0.0.1:8000
```

## 功能模块

| 模块 | 位置 | 说明 |
|---|---|---|
| 智能问诊 | `app/consult/chat.py` | 问题 → 症状识别 + 图谱检索 + 资料检索 → 作为上下文交给大模型，流式输出 |
| 辨证报告 | `app/consult/pipeline.py` | 患者信息 → 舌象 → 图谱辨证 → 资料检索 → 分节生成 → 汇总 Markdown 报告（可下载/打印） |
| 知识图谱 | `app/kg/graph.py` | 症状→证型→方剂→中药，含十八反/十九畏检查；数据在 `data/kg/` |
| 资料检索 | `app/rag/retriever.py` | 目前是零依赖 TF-IDF，语料放 `data/corpus/*.txt`（现有 3 篇仅为示例） |
| 舌象分析 | `app/tongue/classifier.py` | ResNet18 六分类；未配置权重时为演示结果 |
| 药方识别 | `app/prescription/` | 粘贴文字或上传处方照片（OCR）→ 提取药名/剂量/煎法 → 十八反十九畏检查 → 相近方剂比对 → 大模型解读 |
| 中药识别 | `app/herb/recognizer.py` | 药材图片分类（需权重，未配置时为演示）+ 按名称查询性味归经、所在方剂、禁忌 |
| 经络图谱 | `web/atlas/` | 3D 人体穴位图，作为独立页面嵌入 |
| 大模型 | `app/llm/engine.py` | 基座 + LoRA 适配器；无模型时演示模式 |

## 接入中医大模型

1. 安装推理依赖：`python -m pip install -r requirements-llm.txt`（NVIDIA 显卡需先按 pytorch.org 装 CUDA 版 torch）
2. 下载模型：`python scripts/download_model.py --adapter <适配器仓库名> --mirror`
3. 复制 `.env.example` 为 `.env`，填写：
   ```
   LLM_BASE_MODEL=models/base
   LLM_ADAPTER_PATH=models/adapter
   ```
4. 自检：`python check_llm.py`（会加载模型并生成一句话，显示设备和速度）
5. 启动：`python run.py`，左下角状态栏显示“模型加载中…”→“中医大模型已就绪”

### 云端 API 模式（可选，效果更好）

在 `.env` 里写：

```
LLM_MODE=api
LLM_API_KEY=<你的密钥>
LLM_API_MODEL=deepseek-chat
```

不需要下载模型、不需要 torch。想对比本地模型，把 `LLM_MODE` 改成 `local` 后重启即可。
问诊内容会发给第三方服务器，请勿输入真实隐私信息。

加载失败会自动回退到演示模式，原因显示在状态栏的悬浮提示中。

## 接口

- `GET  /api/status` 运行状态
- `POST /api/chat` 问答（SSE）
- `POST /api/consult` 辨证流程（SSE，表单：`patient` JSON + 可选 `image`）
- `POST /api/tongue` 单独舌象分析
- `POST /api/prescription` 药方识别（SSE，表单：`text` 和/或 `image`）
- `POST /api/herb/recognize` 药材图片识别；`GET /api/herb/{名称}` 药材详情；`GET /api/herb/search?q=` 药名联想
- `GET  /api/kg/syndromes`、`/api/kg/formulas` 知识库浏览

## 声明

本项目输出仅供学习与辅助参考，不构成医疗建议。3D 人体网格来自 CC0 资产，见 `web/atlas/HUMAN-MODEL-LICENSE.txt`；Three.js 许可见 `web/atlas/THREE-LICENSE.txt`。

## 已知限制

- 药材性味归经来自机器翻译的词表，已在读取时做术语规范化，但仍需对照药典校对。
- 图谱中的“方剂”数据来自病例汇总，部分方剂药味过多、并非经典组成，相近方剂比对仅供参考。
- 中药图片识别需要自备训练好的权重与类别文件（`HERB_WEIGHTS` / `HERB_LABELS`）。


## 智能问诊的工作方式（多轮）

参考灵枢智联“先收集信息、够了再诊断”的思路，把问诊拆成几个阶段，由 `app/consult/dialogue.py` 的状态机每轮根据**完整对话历史**判断：

| 阶段 | 触发 | 行为 |
| --- | --- | --- |
| `urgent` | 本轮出现红旗症状（呼吸困难、胸痛、咯血…） | 先提示尽快就医 |
| `qa` | 纯知识问题（药、穴位、概念） | 直接回答，附古籍参考 |
| `collect` | 在描述症状但关键信息不够 | 复述已知要点 + 每次只问一个问题，不下结论 |
| `conclude` | 关键信息齐了 / 追问达到上限 / 用户说“直接给结论”或点按钮 | 输出「辨证小结」 |
| `followup` | 已给过小结后的追问 | 结合全部对话回答，必要时调整判断 |

- 常见病证知识表：`data/kg/common_conditions.json`（目前覆盖感冒/咳嗽/咽痛等外感肺系，含风寒、风热、风燥、痰湿、痰热等证型线索、红旗症状、调护与禁忌）。新增病类只需在 `categories` 里加一项（`triggers` 触发词、`slot_order/required/optional` 要问的信息、`patterns` 证型线索）。
- 古籍语料：`data/corpus/` 下放 `.txt`（格式：`<篇名>` 分节），首次启动自动建 BM25 索引并缓存到 `models/rag_index.pkl`；语料变动自动重建。检索带相关度门槛，宁缺毋滥。
- 测试：`python tests/test_dialogue.py`（状态机）、`python tests/test_flow_e2e.py`（端到端，用演示后端，无需模型）。


## 运行时切换模型 / 联网检索

- 左下角状态栏或问诊页顶部的“模型：…⚙”按钮，可在 **云端 API（DeepSeek 等）/ 本地中医模型 / 演示模式** 之间切换。
  - API 模式填 Key、模型名即可（接口地址默认 `https://api.deepseek.com`，任何 OpenAI 兼容接口都行）。点“测试并切换”会先发一次连通性测试，**失败则保持原模型**并提示原因（Key 无效 / 余额不足等）。
  - Key 只保存在服务进程内存里，不写磁盘；也可以预先写进 `.env` 的 `LLM_API_KEY`。
  - 提示：用云端 API 时，问诊内容会发送到第三方服务器，请不要填写真实姓名、证件号等隐私。
- 问诊页的“联网检索”开关：在**出结论 / 追问 / 知识问答**阶段，用症状或证型关键词（不带整段对话）检索网页摘要，作为“参考资料”交给模型，回答里会标明“据网络资料”，并在下方显示来源链接。信息收集阶段与紧急情况不联网。检索失败会提示并自动退回本地资料。
- 搜索源见 `.env.example`；建议配置 `SERPER_API_KEY` 以获得稳定结果，免 Key 的 Bing/DuckDuckGo 靠页面解析，样式变化时可能失效。


## 训练舌苔分类权重

没有权重时舌象分析是“演示结果”（按图片哈希随机给一个类别，仅用于联调界面，**不能当真**）。用自己的数据集训练：

```bash
pip install torch torchvision          # NVIDIA 显卡请先按 pytorch.org 装对应 CUDA 版
python scripts/train_tongue.py --data D:/你的数据集目录          # 每类一个文件夹，或已分好 train/val
```

- 训练会用 ImageNet 预训练的 ResNet18 微调，以验证集宏平均 F1 选最佳，并打印混淆矩阵和每类召回。
- 产物 `models/tongue_resnet18.pth` 和 `models/tongue_resnet18.pth.classes.json`（类别顺序，推理时自动读取）。
- `.env` 里写 `TONGUE_WEIGHTS=models/tongue_resnet18.pth`，重启即可。
- 数据集若含“同一张图的多份增强副本”，要先去重或先划分再增强，否则验证集泄漏、指标虚高。
- 这只是辅助参考，对非舌象图片也会强行给出某个类别；置信度很低时请忽略结果。

- Lingshu 数据集的英文文件夹名（如 `black_tongue_coating`）会被自动识别并换成中文类名，顺序与默认 6 类一致。

## 古籍资料库与辨证报告

- `data/corpus/` 现有 84 本古籍（约 36MB，含 Lingshu 的 81 本全部导入）；「知识库」页新增“古籍资料”，可按书目浏览、书内搜索、全库检索。
- 首次启动会构建检索索引（约十几秒，缓存约 93MB 在 `models/rag_index.pkl`），之后读缓存约 2 秒。
- 辨证报告按“分节检索”生成：每一节单独检索古籍，正文用 [编号] 引用，文末列“参考资料”和“名词速查”（药材、穴位的搜索入口，只是搜索页，不构成用药或操作建议）。
