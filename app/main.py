"""岐黄智诊 · 后端入口（FastAPI）"""
from __future__ import annotations

import json
import uuid
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.consult import chat as chat_mod
from app.consult import pipeline
from app.kg.graph import get_graph
from app.llm.engine import engine
from app.rag.retriever import get_retriever
from app.herb import recognizer as herb_rec
from app.prescription import ocr as rx_ocr
from app.prescription import pipeline as rx_pipeline
from app.tongue import classifier
from app.web import search as web_search_mod

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    if settings.llm_preload:
        engine.start_loading()      # 后台加载模型，页面可先打开
    yield


app = FastAPI(title="岐黄智诊", version="0.1.0", lifespan=lifespan)

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        try:
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # 把错误也推给前端展示
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _save_image(file: UploadFile | None):
    if not file or not file.filename:
        return None
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in ALLOWED_IMG:
        raise HTTPException(400, "仅支持 jpg / png / webp / bmp 图片")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(400, "图片不能超过 8MB")
    settings.upload_dir.mkdir(exist_ok=True)
    path = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------ 状态
@app.get("/api/status")
def status():
    return {
        "llm": engine.info(),
        "web": {"providers": web_search_mod.available_providers()},
        "tongue": classifier.status(),
        "herb": herb_rec.status(),
        "ocr": rx_ocr.available(),
        "kg": get_graph().stats(),
        "rag": get_retriever().stats(),
    }


# ------------------------------------------------------------------ 模型切换
class SwitchReq(BaseModel):
    mode: str                        # local | api | mock
    api_key: str | None = None       # 只在内存里保存，不写入磁盘；留空表示沿用已设置的
    api_base: str | None = None
    api_model: str | None = None


@app.post("/api/llm/switch")
def llm_switch(req: SwitchReq):
    """运行时切换模型。api 模式会先做一次连通性测试，失败则保持原模型并返回原因。"""
    try:
        return engine.switch(req.mode, req.api_key, req.api_base, req.api_model)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ 问答
class ChatReq(BaseModel):
    question: str
    history: list[dict] = []
    force_conclude: bool = False      # 前端“信息够了，直接给建议”按钮
    web_search: bool = False          # 前端“联网检索”开关


@app.post("/api/chat")
def chat(req: ChatReq):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return _sse(chat_mod.stream_answer(req.question.strip(), req.history, req.force_conclude, req.web_search))


# ------------------------------------------------------------------ 辨证
@app.post("/api/consult")
async def consult(patient: str = Form(...), image: UploadFile | None = File(None)):
    try:
        p = json.loads(patient)
    except json.JSONDecodeError:
        raise HTTPException(400, "patient 必须是 JSON")
    if not (p.get("complaint") or "").strip():
        raise HTTPException(400, "请填写主诉与症状")
    path = await _save_image(image)
    return _sse(pipeline.run(p, path))


@app.post("/api/tongue")
async def tongue(image: UploadFile = File(...)):
    path = await _save_image(image)
    return classifier.analyze(path)


# ------------------------------------------------------------------ 药方识别
@app.post("/api/prescription")
async def prescription(text: str = Form(""), image: UploadFile | None = File(None)):
    path = await _save_image(image)
    if not path and not text.strip():
        raise HTTPException(400, "请粘贴药方文字或上传药方图片")
    return _sse(rx_pipeline.run(text, path))


# ------------------------------------------------------------------ 中药识别 / 查询
@app.post("/api/herb/recognize")
async def herb_recognize(image: UploadFile = File(...)):
    path = await _save_image(image)
    return herb_rec.recognize(path)


@app.get("/api/herb/search")
def herb_search(q: str = ""):
    return get_graph().search_herbs(q)


@app.get("/api/herb/{name}")
def herb_detail(name: str):
    kg = get_graph()
    prof = kg.herb_profile(name)
    if not prof:
        raise HTTPException(404, "图谱中未收录该药材")
    return {**prof, "formulas": kg.formulas_with_herb(name), "taboo": kg.taboo_partners(name)}


# ------------------------------------------------------------------ 知识库浏览
@app.get("/api/kg/syndromes")
def kg_syndromes():
    return get_graph().list_syndromes()


@app.get("/api/kg/syndrome/{syn_id}")
def kg_syndrome(syn_id: str):
    d = get_graph().syndrome_detail(syn_id)
    if not d:
        raise HTTPException(404, "没有这个证型")
    return d

@app.get("/api/kg/formulas")
def kg_formulas():
    return get_graph().list_formulas()


# ------------------------------------------------------------------ 古籍资料库
@app.get("/api/corpus/books")
def corpus_books():
    return get_retriever().books()


@app.get("/api/corpus/book")
def corpus_book(name: str, offset: int = 0, limit: int = 20):
    """读一本书的原文片段（分页）。"""
    r = get_retriever().book_chunks(name, offset, max(1, min(limit, 50)))
    if not r["total"]:
        raise HTTPException(404, "没有这本书")
    return r

@app.get("/api/corpus/search")
def corpus_search(q: str, k: int = 10, book: str = ""):
    """在古籍资料库里检索；book 非空时只看这一本。返回 [{source, text, score}]。"""
    q = q.strip()
    if not q:
        raise HTTPException(400, "请输入检索词")
    k = max(1, min(k, 30))
    hits = get_retriever().search(q, k=200 if book else k, min_coverage=0.3)
    if book:
        hits = [h for h in hits if h["source"].split("·")[0] == book]
    return hits[:k]


# ------------------------------------------------------------------ 页面
@app.get("/")
def index():
    return FileResponse(settings.web_dir / "index.html")


app.mount("/atlas", StaticFiles(directory=settings.web_dir / "atlas", html=True), name="atlas")
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
