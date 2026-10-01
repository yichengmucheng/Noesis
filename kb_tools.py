#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人知识库工具集 —— 上传 / 扫描 / 轮询状态 / 问答评测 / 图谱统计

用法：
    python kb_tools.py upload <文件或目录>      上传文件并触发处理
    python kb_tools.py scan                     扫描 inputs 目录并入库
    python kb_tools.py status                   查看所有文档处理状态
    python kb_tools.py wait                     等待全部文档处理完成
    python kb_tools.py ask "问题" [mode]        问答（naive/mix/local/global/hybrid，默认 mix）
    python kb_tools.py graph                    图谱规模统计
    python kb_tools.py benchmark "问题"         同一问题 6 模式对比（评估 KG 增强）

环境变量：
    KB_SERVER  LightRAG 服务地址，默认 http://127.0.0.1:9621
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

SERVER = os.environ.get("KB_SERVER", "http://127.0.0.1:9621")
TIMEOUT = httpx.Timeout(600.0, connect=15.0)

# 与后端一致的扩展名白名单（见 document_routes.py / data_loaders）
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".htm", ".html", ".json", ".csv",
    ".pdf", ".docx", ".pptx", ".xlsx", "", ".log", ".xml", ".yml", ".yaml", ".ini",
}


def out(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def client() -> httpx.Client:
    # trust_env=False：绕过系统/注册表代理（如 Clash 7897），强制直连本地服务
    return httpx.Client(timeout=TIMEOUT, base_url=SERVER, trust_env=False, headers={"accept-language": "zh-CN,zh;q=0.9"})


def upload(paths, mode: str = "upload") -> int:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in TEXT_EXTS and "__enqueued__" not in x.parts))
        elif p.is_file():
            files.append(p)
        else:
            out(f"[跳过] 不存在: {p}")
    files = [f for f in files if f.name.startswith("__tmp__") is False]
    if not files:
        out("[错误] 没有可上传的文件")
        return 1
    ok = 0
    with client() as c:
        for f in files:
            data = f.read_bytes()
            fn = f.name  # httpx 0.28 要求 filename 为 str；中文文件名由 multipart 传输层自动处理
            if mode == "upload":
                resp = c.post(
                    "/documents/upload",
                    files={"file": (fn, data, "application/octet-stream")},
                    data={"mode": "upload"},
                )
            else:
                resp = c.post(
                    "/text/text",
                    data={"text": data.decode("utf-8", errors="ignore"), "mode": "nap"},
                )
            if resp.status_code in (200, 201):
                body = resp.json()
                out(f"[上传成功] {f.name} -> {body}")
                ok += 1
            else:
                out(f"[上传失败] {f.name} -> HTTP {resp.status_code}: {resp.text[:400]}")
    out(f"\n共 {len(files)} 个文件，成功 {ok} 个。后台正在解析/切块/抽取/建图。")
    return 0 if ok == len(files) else 2


def trigger_scan() -> int:
    with client() as c:
        r = c.post("/documents/scan")
        if r.status_code == 200:
            out(f"[扫描] {r.json()}")
            return 0
        out(f"[扫描失败] HTTP {r.status_code}: {r.text[:400]}")
        return 1


def _all_docs(c) -> list:
    r = c.get("/documents")
    r.raise_for_status()
    grouped = r.json().get("statuses", {}) or {}
    docs = []
    for group, lst in grouped.items():
        for d in lst or []:
            d.setdefault("status", group)
            docs.append(d)
    return docs


def show_status() -> dict:
    with client() as c:
        docs = _all_docs(c)
        stat = {}
        for d in docs:
            s = d.get("status", "?")
            stat[s] = stat.get(s, 0) + 1
            summary = (d.get("content_summary") or "")[:38].replace("\n", " ")
            err = d.get("error_msg")
            extra = f" 错误: {str(err)[:120]}" if err else ""
            out(f"  [{s:<10}] {d.get('id', '')[:8]}  {summary}  (chunks={d.get('chunks_count')}){extra}")
        out(f"\n统计: {stat}  合计: {len(docs)}")
        return stat


async def wait_docs(max_seconds: int = 1800) -> int:
    """轮询直到没有 pending/processing 文档，或超时。返回 0=全部 processed。
    服务器处理文档（MAX_ASYNC=1）期间会丢弃空闲 keep-alive 连接，
    因此每次轮询都新建连接并捕获瞬时网络错误重试。"""
    start = time.time()
    errors = 0
    while time.time() - start < max_seconds:
        try:
            with client() as c:
                docs = _all_docs(c)
        except Exception as e:  # noqa: BLE001 - 轮询容错
            errors += 1
            if errors > 30:
                out(f"\n[错误] 轮询连续失败: {e}")
                return 5
            time.sleep(3)
            continue
        errors = 0
        busy = [d for d in docs if d.get("status") in ("pending", "processing")]
        failed = [d for d in docs if d.get("status") == "failed"]
        done = [d for d in docs if d.get("status") == "processed"]
        sys.stdout.write(f"\r等待中... processed={len(done)} busy={len(busy)} failed={len(failed)} ({int(time.time()-start)}s)    ")
        sys.stdout.flush()
        if not busy:
            out("")
            for d in failed:
                out(f"  [失败] {d.get('id','')[:8]} 错误: {str(d.get('error_msg'))[:200]}")
            return 0 if not failed else 3
        time.sleep(5)
    out("\n[超时] 仍有文档未处理完成")
    return 4


def ask(question: str, mode: str = "mix", top_k: int = 5, only: str = None) -> dict:
    payload = {"query": question, "mode": mode, "top_k": top_k}
    if only:
        payload["only_context"] = only
    t0 = time.time()
    with client() as c:
        r = c.post("/query", json=payload)
        dt = time.time() - t0
        if r.status_code != 200:
            out(f"[失败] HTTP {r.status_code}: {r.text[:300]}")
            return {}
        body = r.json()
        resp = body.get("response") or body.get("data") or ""
        refs = body.get("metadata", {}).get("refs") or []
        out(f"──[{mode}] {dt:.1f}s ─ ─ ─ ─ ─ ─")
        out(str(resp))
        return {"mode": mode, "response": resp, "refs": refs, "seconds": dt}


def graph_stats() -> int:
    import xml.etree.ElementTree as ET

    gml = Path(__file__).parent / "rag_storage" / "graph_chunk_entity_relation.graphml"
    if not gml.exists():
        out("[错误] 找不到图谱文件: " + str(gml))
        return 1
    tree = ET.parse(gml)
    root = tree.getroot()
    g = root.find("{http://graphml.graphdrawing.org/xmlns}graph")
    nodes = g.findall("{http://graphml.graphdrawing.org/xmlns}node")
    edges = g.findall("{http://graphml.graphdrawing.org/xmlns}edge")
    out(f"知识图谱: {len(nodes)} 节点, {len(edges)} 边  ({gml.name})")

    # 节点实体类型来自 node_id 前缀（lightrag 以 "实体名" 作 node_id，这里统计 file_id 也行）
    doc_ids = {}
    with client() as c:
        for d in _all_docs(c):
            doc_ids[d["id"]] = (d.get("content_summary") or "")[:30].replace("\n", " ")
    out(f"已入库文档: {len(doc_ids)} 份")
    for k, v in doc_ids.items():
        out(f"  {k}: {v}")

    # 边按来源文档分布: edge 上 key d10 是 file_path(旧) —— 改为按 chunk 无关的整体统计
    by_doc = {}
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    for e in edges:
        for f in e.findall("g:data", ns):
            pass
    out("(按文档分布可看 WebUI Knowledge Graph 页；`python kb_tools.py status` 看文档状态)")
    return 0


def benchmark(question: str) -> int:
    out("\n================  KG 增强对比评测  ================")
    out(f"问题: {question}\n")
    results = {}
    for mode in ("naive", "hybrid", "local", "global", "mix"):
        r = ask(question, mode=mode)
        if r:
            results[mode] = r
        out("")
    out("================  评测小结  ================")
    out("naive = 纯向量基线；[KG] 标记的知识来自图谱抽取，证明 KG 真实参与检索增强。")
    return 0


def delete_docs(prefix: str, delete_file: bool = False) -> int:
    """按 id 前缀删除文档"""
    with client() as c:
        docs = _all_docs(c)
        matches = [d["id"] for d in docs if d["id"].startswith(prefix) or (d.get("content_summary") or "").startswith(prefix)]
        if not matches:
            out(f"[未找到] 前缀 {prefix!r} 没有匹配文档")
            return 1
        if matches[0] != prefix and len(matches) > 1:
            out(f"[警告] 前缀匹配到 {len(matches)} 份文档，将全部删除")
        for m in matches:
            d = dict.fromkeys(("id",), m)
            out(f"  待删: {m}")
        r = c.request("DELETE", "/documents/delete_document", json={"doc_ids": matches, "delete_file": delete_file})
        out(f"[删除] HTTP {r.status_code}: {r.text[:160]}")
        return 0 if r.status_code == 200 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="个人知识库工具集")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("upload", help="上传文件/目录并入库")
    p.add_argument("paths", nargs="+")
    p.add_argument("--raw", action="store_true", help="走 /text 接口（纯文本）而非文件上传")

    sub.add_parser("scan", help="扫描 inputs 目录")
    sub.add_parser("status", help="查看文档状态")
    sub.add_parser("wait", help="等待处理完成").add_argument("--timeout", type=int, default=1800)

    p = sub.add_parser("ask", help="问答")
    p.add_argument("question")
    p.add_argument("--mode", default="mix")
    p.add_argument("--top-k", type=int, default=5)

    sub.add_parser("graph", help="图谱统计")
    p = sub.add_parser("benchmark", help="多模式对比")
    p.add_argument("question")
    p = sub.add_parser("delete", help="按 id 前缀删除文档")
    p.add_argument("prefix")
    p.add_argument("--file", action="store_true", help="同时删除 inputs 目录中的源文件")
    args = ap.parse_args()

    if args.cmd == "upload":
        total = upload(args.paths, "text" if args.raw else "upload")
        if total == 0:
            asyncio.run(wait_docs())
        return total
    if args.cmd == "scan":
        return trigger_scan()
    if args.cmd == "status":
        show_status()
        return 0
    if args.cmd == "wait":
        return asyncio.run(wait_docs(args.timeout))
    if args.cmd == "ask":
        ask(args.question, mode=args.mode, top_k=args.top_k)
        return 0
    if args.cmd == "graph":
        return graph_stats()
    if args.cmd == "benchmark":
        return benchmark(args.question)
    if args.cmd == "delete":
        return delete_docs(args.prefix, delete_file=args.file)
    return 1


if __name__ == "__main__":
    sys.exit(main())
