"""测试里配置的入库实现。只在 INGEST_RUNTIME 指向这里时使用。

向量由文本内容哈希得到，实体名只从抽取提示里的正文截取。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from lightrag.product_index import IndexRuntime


def embed_vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


async def embed(texts: list[str]):
    return np.vstack([embed_vector(text) for text in texts])


def _source_text(system_prompt: str | None, prompt: str | None) -> str:
    user = (prompt or "").strip()
    parts = [part.strip() for part in re.split(r"\n\s*\n", user) if part.strip()]
    body = [
        part
        for part in parts
        if not part.startswith(("请从下面", "先实体", "---"))
        and "<|COMPLETE|>" not in part
    ]
    if body:
        return "\n\n".join(body)
    blocks = re.findall(r"---文本---\s*(.*)\Z", system_prompt or "", flags=re.S)
    if blocks:
        return blocks[-1].strip()
    return parts[-1] if parts else user


def _grounded(text: str, left: str, right: str, relation: str) -> str | None:
    if left in text and right in text and relation in text:
        return "\n".join([
            f"entity<|#|>{left}<|#|>概念<|#|>{left}",
            f"entity<|#|>{right}<|#|>概念<|#|>{right}",
            f"relation<|#|>{left}<|#|>{right}<|#|>{relation}<|#|>{text[:80]}",
            "<|COMPLETE|>",
        ])
    return None


async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    text = _source_text(system_prompt, prompt)
    for left, right, relation in (
        ("光合作用", "葡萄糖", "产生"),
        ("葡萄糖", "能量", "提供"),
        ("天津财经大学", "信息管理", "就读"),
        ("红烧肉", "酱油", "加入"),
        ("西湖", "杭州", "位于"),
    ):
        grounded = _grounded(text, left, right, relation)
        if grounded:
            return grounded
    if "导致" in text:
        left, _, right = text.partition("导致")
        left, right = left.strip()[:40], right.strip()[:40]
        if left and right and left in text and right in text:
            return "\n".join([
                f"entity<|#|>{left}<|#|>部件<|#|>{left}",
                f"entity<|#|>{right}<|#|>部件<|#|>{right}",
                f"relation<|#|>{left}<|#|>{right}<|#|>导致<|#|>{text[:80]}",
                "<|COMPLETE|>",
            ])
    pieces = [part.strip() for part in re.split(r"[\n。；]+", text) if len(part.strip()) >= 2]
    if not pieces:
        return "<|COMPLETE|>"
    lines = [f"entity<|#|>{piece[:40]}<|#|>部件<|#|>{piece[:40]}" for piece in pieces[:2] if piece[:40] in text]
    lines.append("<|COMPLETE|>")
    return "\n".join(lines)


def runtime() -> IndexRuntime:
    return IndexRuntime(embed=embed, llm=llm, gleaning=0)
