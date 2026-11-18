"""Markdown 知识库切分器。

两层实现，对应设计文档 FR-2「结构感知切分」：
1. split_fixed       —— 按字符数硬切，作为基线，用来观察切分会坏在哪；
2. split_by_heading  —— 先按 Markdown 标题切小节，再在节内按段落拼接到目标长度。

字符数是 token 数的粗略代理：中文通常 1 字 ≈ 1 个 token，
所以 300 字符大致等于 300 token，接近子块上限（200~350 token）。

运行：python -m supportpilot.ingest.splitter --corpus data/corpus
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from supportpilot.config import (
    DEFAULT_FIXED_OVERLAP,
    DEFAULT_FIXED_SIZE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    get_settings,
)
from supportpilot.logging import configure_logging

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = "。！？：；…）】」』"


@dataclass
class Chunk:
    """一个待向量化的文本块。heading_path 会写进 payload 用于引用展示。"""

    ordinal: int
    heading_path: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def split_fixed(
    text: str,
    size: int = DEFAULT_FIXED_SIZE,
    overlap: int = DEFAULT_FIXED_OVERLAP,
) -> list[str]:
    """按字符数硬切，相邻块重叠 overlap 个字符。

    步长必须是 size - overlap；若 overlap >= size，步长小于等于 0，
    range 会原地踏步或倒退，表现为死循环。
    """
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    step = size - overlap
    chunks = [text[start : start + size] for start in range(0, len(text), step)]
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _iter_sections(markdown: str) -> list[tuple[str, str]]:
    """把 Markdown 拆成 (标题路径, 正文) 列表。

    标题路径形如「Mac 连接公司 VPN 配置指南 > 4. 导入证书并建立连接」，
    标题本身不进正文，而是记录在路径里，方便引用时说明答案出自哪一节。
    """
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    body: list[str] = []

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue

        if "".join(body).strip():
            sections.append((" > ".join(heading_stack), "\n".join(body).strip()))

        level = len(match.group(1))
        while len(heading_stack) >= level:
            heading_stack.pop()
        heading_stack.append(match.group(2))
        body = []

    if "".join(body).strip():
        sections.append((" > ".join(heading_stack), "\n".join(body).strip()))
    return sections


def _paragraphs(text: str) -> list[str]:
    """按空行切段落。表格、列表天然是连续的行块，不会被空行打断。"""
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def split_by_heading(
    markdown: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """结构感知切分：小节内按段落贪心拼接，超长段落才回退到定长切。"""
    raw: list[Chunk] = []

    for heading_path, body in _iter_sections(markdown):
        for paragraph in _paragraphs(body):
            # 只有单个段落本身就超长时才硬切，且切口带重叠
            pieces = (
                split_fixed(paragraph, max_chars, max_chars // 6)
                if len(paragraph) > max_chars
                else [paragraph]
            )
            for piece in pieces:
                previous = raw[-1] if raw else None
                if (
                    previous is not None
                    and previous.heading_path == heading_path
                    and previous.char_count + len(piece) + 2 <= max_chars
                ):
                    previous.text += "\n\n" + piece
                else:
                    raw.append(Chunk(len(raw), heading_path, piece))

    # 章节末尾常常剩下很短的块，检索出来信息量不足，并回同章节上一块
    merged: list[Chunk] = []
    for chunk in raw:
        is_short = chunk.char_count < min_chars
        same_section = bool(merged) and merged[-1].heading_path == chunk.heading_path
        if is_short and same_section:
            merged[-1].text += "\n\n" + chunk.text
        else:
            merged.append(chunk)

    for ordinal, chunk in enumerate(merged):
        chunk.ordinal = ordinal
    return merged


def detect_fixed_issues(chunks: list[str]) -> list[str]:
    """把定长切分的典型损伤识别出来，用于回归对比。判据是启发式，不是严格解析。"""
    issues: list[str] = []
    for index, chunk in enumerate(chunks):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        tail = lines[-1].strip()
        if tail.startswith("#"):
            issues.append(f"块 {index}: 以标题结尾，正文被切到下一块")
        elif "|" in tail and tail.count("|") % 2 == 1:
            # 完整表格行的竖线是偶数个，奇数说明这一行被切掉了一截
            issues.append(f"块 {index}: 表格被从中间劈开")
        elif tail[-1] not in _SENTENCE_END:
            issues.append(f"块 {index}: 句子被截断 → …{tail[-14:]}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="按章节切分知识库语料")
    parser.add_argument("--corpus", type=Path, default=None, help="语料目录，默认取配置")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 INFO")
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    settings = get_settings()
    corpus_dir = args.corpus or settings.corpus_dir

    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"语料目录为空：{corpus_dir}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        fixed = split_fixed(text, settings.fixed_size, settings.fixed_overlap)
        structured = split_by_heading(text, settings.max_chars, settings.min_chars)
        issues = detect_fixed_issues(fixed)

        logger.info(
            "文档切分完成",
            extra={
                "fields": {
                    "document": path.name,
                    "fixed_chunks": len(fixed),
                    "structured_chunks": len(structured),
                    "fixed_issues": len(issues),
                }
            },
        )

        print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
        avg_fixed = sum(map(len, fixed)) // len(fixed)
        avg_structured = sum(c.char_count for c in structured) // len(structured)
        print(f"定长切分：{len(fixed)} 块，平均 {avg_fixed} 字符")
        print(f"结构感知：{len(structured)} 块，平均 {avg_structured} 字符")

        print("\n[定长切分] 前 3 块：")
        for index, text_chunk in enumerate(fixed[:3]):
            preview = text_chunk.replace("\n", " ")[:40]
            print(f"  {index:>2}  {len(text_chunk):>3} 字符 | {preview}")

        if issues:
            print(f"\n[定长切分] 检出 {len(issues)} 处损伤：")
            for issue in issues:
                print(f"  - {issue}")

        print("\n[结构感知] 全部块：")
        for structured_chunk in structured:
            preview = structured_chunk.text.replace("\n", " ")[:36]
            path = structured_chunk.heading_path
            print(
                f"  {structured_chunk.ordinal:>2}  {structured_chunk.char_count:>3} 字符 | {path}"
            )
            print(f"       {preview}")


if __name__ == "__main__":
    main()
