"""轻量检索：对 data/corpus/*.txt 按「书 → 篇名 → 段落」切块，字二元组 BM25 + 倒排索引（零依赖）。

相比旧版：
  - 支持整本古籍（几十 MB 也能毫秒级检索）：旧版每次查询都遍历全部块；
  - 返回的每条结果带 书名·篇名，且有「覆盖度」门槛：查询里的关键词没有足够多出现在文段里就不返回，
    宁可不给资料，也不要把不相关的段落塞给模型（这是旧版误导模型的原因之一）；
  - 清洗语料里残留的 “\\n”、“属性：” 等标记；
  - 首次建索引后缓存到 models/rag_index.pkl，语料文件变化时自动重建。

接口保持不变：search(query, k) -> [{source, text, score}]；后续可替换为向量库。
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache

from app.config import settings

CHUNK_MAX = 260          # 每块最多字数（太长会稀释关键词，也占模型上下文）
CHUNK_MIN = 30
_HEADER_RE = re.compile(r"^(书名|作者|朝代|年份)[:：]")
_RULE_RE = re.compile(r"^[\-—－=＝_~*·.\s]{6,}$")                    # 分隔线：-----、=====
_NUM_TITLE_RE = re.compile(r"^\d{1,3}\s*[\.、．]\s*([^。，；！？]{1,30})$")  # “1.发热”这类编号小标题
_STOP = set("的了是在和与及或也而之其以为有无不我你他她它这那一")
_INDEX_VERSION = 5


def _clean(text: str) -> str:
    text = text.replace("\\n", "\n").replace("<目录>", "\n").replace("\\", "")
    text = re.sub(r"^\s*属性[:：]\s*", "", text, flags=re.M)
    return text


def _grams(text: str) -> list[str]:
    """字二元组；跳过标点和停用字，避免“的了”之类把检索带偏。"""
    out: list[str] = []
    for seg in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", text):
        seg = "".join(ch for ch in seg if ch not in _STOP)
        if len(seg) == 1:
            out.append(seg)
        out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return out


def _split_long(par: str) -> list[str]:
    if len(par) <= CHUNK_MAX:
        return [par]
    out, cur = [], ""
    for sent in re.split(r"(?<=[。！？；])", par):
        if len(cur) + len(sent) > CHUNK_MAX and cur:
            out.append(cur)
            cur = ""
        cur += sent
    if cur:
        out.append(cur)
    return out


_BOOK_RE = re.compile(r"^书名[：:]\s*(.+?)\s*$", re.M)


def _book_name(path, raw: str) -> str:
    """书名优先取正文里的“书名：xxx”。文件名在部分 Windows 解压工具下会变成乱码，不能当显示名用。"""
    m = _BOOK_RE.search(raw[:600])
    if m:
        return m.group(1)
    stem = path.stem
    try:                                   # UTF-8 文件名被当成 GBK 解出来的情况，尽量还原
        fixed = stem.encode("gbk").decode("utf-8")
        if fixed != stem:
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return stem


def _chunk_file(path) -> list[tuple[str, str, str]]:
    """返回 [(书名, 篇名, 文本)]。"""
    raw = _clean(path.read_text(encoding="utf-8"))
    book, title, buf, out = _book_name(path, raw), "", [], []
    lines = [l.strip() for l in raw.splitlines()]
    for i, l in enumerate(lines[:15]):       # 开头若有“来源/发布人 + 分隔线”的页眉，整段丢掉
        if _RULE_RE.match(l):
            lines = lines[i + 1:]
            break

    def flush():
        nonlocal buf
        if buf:
            out.append((book, title, "".join(buf)))
            buf = []

    for line in lines:
        if not line or line == "<目录>" or _HEADER_RE.match(line) or _RULE_RE.match(line):
            continue
        m = _NUM_TITLE_RE.match(line)
        if m:
            flush()
            title = m.group(1).strip()
            continue
        if line.startswith("<篇名>"):
            flush()
            title = line[4:].strip()
            continue
        for piece in _split_long(line):
            if sum(len(b) for b in buf) + len(piece) > CHUNK_MAX:
                flush()
            buf.append(piece)
    flush()
    return [c for c in out if len(c[2]) >= CHUNK_MIN]


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.avg_len = 1.0
        if not self._load_cache():
            self._build()
            self._save_cache()

    # ------------------------------------------------------------ 建索引 / 缓存
    def _files(self):
        return sorted(settings.corpus_dir.glob("*.txt"))

    def _signature(self) -> list:
        return [_INDEX_VERSION] + [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in self._files()]

    @property
    def _cache_path(self):
        return settings.root / "models" / "rag_index.pkl"

    def _load_cache(self) -> bool:
        try:
            data = pickle.loads(self._cache_path.read_bytes())
            if data["sig"] != self._signature():
                return False
            self.chunks, self.postings, self.idf, self.avg_len = (
                data["chunks"], data["postings"], data["idf"], data["avg_len"])
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_bytes(pickle.dumps({
                "sig": self._signature(), "chunks": self.chunks, "postings": self.postings,
                "idf": self.idf, "avg_len": self.avg_len}, protocol=4))
        except Exception:
            pass   # 缓存失败不影响使用

    def _build(self) -> None:
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_len = 0
        for p in self._files():
            for book, title, text in _chunk_file(p):
                cid = len(self.chunks)
                grams = _grams(text)
                self.chunks.append({"source": book, "title": title, "text": text, "len": len(grams)})
                total_len += len(grams)
                for g, n in Counter(grams).items():
                    post[g].append((cid, n))
        n = max(len(self.chunks), 1)
        self.avg_len = total_len / n
        self.postings = dict(post)
        self.idf = {g: math.log(1 + (n - len(v) + 0.5) / (len(v) + 0.5)) for g, v in post.items()}

    # ------------------------------------------------------------ 检索
    def search(self, query: str, k: int = 3, min_coverage: float = 0.4, must: list[str] | None = None,
               k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """BM25 排序。min_coverage：命中的查询词（按 idf 加权）占全部查询词的比例下限；
        must：文段里必须至少出现其中一个词（锚点词），用来挡掉“字面相似但主题无关”的段落。"""
        q = [g for g in dict.fromkeys(_grams(query)) if g in self.idf]
        if not q or not self.chunks:
            return []
        q_weight = sum(self.idf[g] for g in q)
        scores: dict[int, float] = defaultdict(float)
        covered: dict[int, float] = defaultdict(float)
        for g in q:
            w = self.idf[g]
            for cid, tf in self.postings[g]:
                dl = self.chunks[cid]["len"]
                scores[cid] += w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / self.avg_len))
                covered[cid] += w
        ranked = sorted(((s, cid) for cid, s in scores.items() if covered[cid] / q_weight >= min_coverage),
                        reverse=True)
        out, seen = [], set()
        for s, cid in ranked:
            c = self.chunks[cid]
            if must and not any(m in c["text"] for m in must):
                continue
            key = (c["source"], c["text"][:30])
            if key in seen:
                continue
            seen.add(key)
            out.append({"source": c["source"] if not c["title"] or c["title"] == c["source"]
                        else f"{c['source']}·{c['title']}",
                        "text": c["text"], "score": round(s, 2),
                        "coverage": round(covered[cid] / q_weight, 2)})
            if len(out) >= k:
                break
        return out

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "documents": len({c["source"] for c in self.chunks})}

    def book_chunks(self, name: str, offset: int = 0, limit: int = 20) -> dict:
        """按书取原文片段（按书中顺序分页），供前端点开一本书直接阅读。"""
        if getattr(self, "_by_book", None) is None:
            by: dict[str, list[dict]] = {}
            for c in self.chunks:
                by.setdefault(c["source"], []).append(c)
            self._by_book = by
        rows = self._by_book.get(name, [])
        offset = max(0, offset)
        part = rows[offset:offset + limit]
        return {"name": name, "total": len(rows), "offset": offset,
                "items": [{"title": c.get("title", ""), "text": c["text"]} for c in part]}

    def books(self) -> list[dict]:
        """资料库书目：书名、作者、朝代、篇幅、片段数（供前端“古籍资料”浏览）。"""
        if getattr(self, "_books", None) is None:
            n_chunks = Counter(c["source"] for c in self.chunks)
            rows = []
            for p in self._files():
                with open(p, encoding="utf-8", errors="ignore") as f:
                    head = f.read(600)
                name = _book_name(p, head)
                meta = {k: (m.group(1).strip() if (m := re.search(rf"^{k}[：:][ \t]*([^\n]*)$", head, re.M)) else "")
                        for k in ("作者", "朝代")}
                rows.append({"name": name, "author": meta["作者"], "dynasty": meta["朝代"],
                             "size_kb": p.stat().st_size // 1024, "chunks": n_chunks.get(name, 0)})
            self._books = sorted(rows, key=lambda r: r["name"])
        return self._books


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
