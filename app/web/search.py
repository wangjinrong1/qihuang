"""联网检索：补充古籍/图谱没覆盖到的知识（零依赖，只用标准库）。

数据源（WEB_SEARCH_PROVIDER=auto 时按顺序尝试，前一个失败/无结果就换下一个）：
  1. Serper（Google，需 SERPER_API_KEY，稳定，推荐）
  2. Bing 中国版（无需 Key，国内可直接访问，页面解析，样式变化时可能失效）
  3. DuckDuckGo（无需 Key，国内通常需要代理）

隐私：只把“症状/证型关键词”发给搜索引擎，不发整段对话；调用方负责构造短查询。
安全：网页内容是不可信输入，只作为“参考资料”交给模型，提示词里已声明其中的指令一律不执行。
"""
from __future__ import annotations

import base64
import html
import json
import re
import time
import urllib.parse
import urllib.request
from urllib.parse import parse_qs, urlparse

from app.config import settings

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36")
_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 600
_TAG = re.compile(r"<[^>]+>")


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", s))).strip()


def _get(url: str, data: bytes | None = None, headers: dict | None = None) -> str:
    h = {"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=settings.web_timeout) as r:
        return r.read().decode("utf-8", "ignore")


# --------------------------------------------------------------------------- 各数据源
def _serper(q: str, n: int) -> list[dict]:
    body = json.dumps({"q": q, "gl": "cn", "hl": "zh-cn", "num": n}).encode()
    raw = _get("https://google.serper.dev/search", body,
               {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"})
    return [{"title": e.get("title", ""), "url": e.get("link", ""), "snippet": e.get("snippet", "")}
            for e in json.loads(raw).get("organic", [])[:n]]


def _bing_real_url(href: str) -> str:
    """Bing 的结果链接是跳转地址（/ck/a?...&u=a1<base64>），还原成真实网址。"""
    if "/ck/a" in href:
        u = parse_qs(urlparse(html.unescape(href)).query).get("u", [""])[0]
        if u.startswith("a1"):
            u = u[2:]
            try:
                return base64.urlsafe_b64decode(u + "=" * (-len(u) % 4)).decode("utf-8")
            except Exception:
                pass
    return html.unescape(href)


def parse_bing(page: str, n: int) -> list[dict]:
    out = []
    for blk in re.findall(r'<li class="b_algo".*?</li>', page, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        sn = re.search(r'<p[^>]*>(.*?)</p>', blk, re.S)
        out.append({"title": _text(m.group(2)), "url": _bing_real_url(m.group(1)),
                    "snippet": _text(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    return out


def _bing(q: str, n: int) -> list[dict]:
    return parse_bing(_get("https://cn.bing.com/search?" + urllib.parse.urlencode({"q": q, "setlang": "zh-cn"})), n)


def parse_ddg(page: str, n: int) -> list[dict]:
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                         page, re.S):
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = urllib.parse.unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group(3), re.S)
        out.append({"title": _text(m.group(2)), "url": href, "snippet": _text(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    return out


def _ddg(q: str, n: int) -> list[dict]:
    page = _get("https://html.duckduckgo.com/html/", urllib.parse.urlencode({"q": q, "kl": "cn-zh"}).encode(),
                {"Content-Type": "application/x-www-form-urlencoded"})
    return parse_ddg(page, n)


_PROVIDERS = {"serper": _serper, "bing": _bing, "duckduckgo": _ddg}


def _order() -> list[str]:
    p = settings.web_provider
    if p in _PROVIDERS:
        return [p]
    return (["serper"] if settings.serper_api_key else []) + ["bing", "duckduckgo"]


# --------------------------------------------------------------------------- 对外接口
def available_providers() -> list[str]:
    return _order()


def search(query: str, n: int = 4) -> dict:
    """返回 {"results": [{title,url,snippet}], "provider": str|None, "error": str|None}。永远不抛异常。"""
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return {"results": [], "provider": None, "error": "空查询"}
    hit = _CACHE.get(query)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return {"results": hit[1], "provider": "cache", "error": None}
    errors = []
    for name in _order():
        try:
            res = [r for r in _PROVIDERS[name](query, n + 2) if r["url"].startswith("http") and (r["title"] or r["snippet"])]
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
            continue
        if res:
            res = _dedupe(res)[:n]
            _CACHE[query] = (time.time(), res)
            return {"results": res, "provider": name, "error": None}
        errors.append(f"{name}: 无结果")
    return {"results": [], "provider": None, "error": "；".join(errors) or "无可用数据源"}


def _dedupe(res: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in res:
        host = urlparse(r["url"]).netloc
        if host in seen:           # 同一网站只取一条，增加信息来源多样性
            continue
        seen.add(host)
        out.append(r)
    return out
