"""关键词匹配基线：不做向量化，先看切分有没有把答案留住。

三步：
1. 把问题拆成关键词（英文/数字整词 + 中文按两字组合）；
2. 看每张卡片命中几个关键词，按覆盖率排序；
3. 看标准答案所在的那一节排第几名，并落盘结果供下次对比。

它不理解同义词，也不懂语序，分数只代表「字面像不像」。
正因为它简单，跑出来的分数才是干净的对照系。

运行：python -m supportpilot.eval.keyword_baseline
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from supportpilot.config import Settings, get_settings
from supportpilot.ingest.splitter import Chunk, split_by_heading
from supportpilot.logging import configure_logging

logger = logging.getLogger(__name__)

# 英文、数字、错误码这类词区分度高，权重给 3；中文两字组合区分度低，权重给 1
EN_TERM_WEIGHT = 3.0
ZH_TERM_WEIGHT = 1.0

# 问句里高频但没有检索价值的组合，命中它们只会抬高分噪声
STOP_BIGRAMS = {
    "什么",
    "怎么",
    "如何",
    "为什",
    "么会",
    "是什",
    "不是",
    "可以",
    "需要",
    "是否",
    "哪些",
    "一下",
    "这个",
    "那个",
    "一个",
    "应该",
    "就是",
    "还是",
    "以后",
    "之后",
    "时候",
    "处理",
    "原因",
    "问题",
    "一般",
    "通常",
}

TOP_K = 5


@dataclass
class RankedChunk:
    score: float
    doc: str
    chunk: Chunk
    hits: list[str]


def query_terms(question: str) -> dict[str, float]:
    """把问句拆成 {关键词: 权重}。中文没有空格，所以用相邻两字组成词。"""
    terms: dict[str, float] = {}

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]*", question):
        terms[token.lower()] = EN_TERM_WEIGHT

    for run in re.findall(r"[\u4e00-\u9fff]+", question):
        for index in range(len(run) - 1):
            bigram = run[index : index + 2]
            if bigram not in STOP_BIGRAMS:
                terms[bigram] = ZH_TERM_WEIGHT

    return terms


def coverage(terms: dict[str, float], text: str) -> tuple[float, list[str]]:
    """命中的关键词权重之和 / 全部关键词权重之和，取值 0~1。"""
    lowered = text.lower()
    hits = [term for term in terms if term in lowered]
    if not hits:
        return 0.0, []
    total = sum(terms.values())
    return sum(terms[term] for term in hits) / total, hits


def load_chunks(corpus_dir: Path, max_chars: int, min_chars: int) -> list[tuple[str, Chunk]]:
    """把每篇文档都切一遍，得到 (文档名, 卡片) 的完整清单。"""
    chunks: list[tuple[str, Chunk]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in split_by_heading(text, max_chars, min_chars):
            chunks.append((path.name, chunk))
    return chunks


def rank_chunks(terms: dict[str, float], chunks: list[tuple[str, Chunk]]) -> list[RankedChunk]:
    """按覆盖率从高到低排序。标题路径也参与匹配，因为正文本身不含章节名。"""
    ranked = []
    for doc_name, chunk in chunks:
        searchable = f"{chunk.heading_path}\n{chunk.text}"
        score, hits = coverage(terms, searchable)
        ranked.append(RankedChunk(score, doc_name, chunk, hits))

    ranked.sort(key=lambda item: (-item.score, item.doc, item.chunk.ordinal))
    return ranked


def evaluate(settings: Settings) -> dict[str, object]:
    questions = json.loads(settings.questions_file.read_text(encoding="utf-8"))["questions"]
    chunks = load_chunks(settings.corpus_dir, settings.max_chars, settings.min_chars)
    if not chunks:
        raise SystemExit(f"语料目录为空：{settings.corpus_dir}")

    recall = {1: 0, 3: 0, 5: 0}
    answerable = 0
    abstain_ok = 0
    unanswerable = 0
    details: list[dict[str, object]] = []

    print(f"语料共切出 {len(chunks)} 张卡片，评测集共 {len(questions)} 题\n")

    for question in questions:
        terms = query_terms(question["question"])
        ranked = rank_chunks(terms, chunks)
        top = ranked[0]

        if question["type"] == "unanswerable":
            unanswerable += 1
            # 基线没有「我不知道」的能力，只能规定：一个词都没命中才算拒答
            refused = top.score == 0.0
            abstain_ok += int(refused)
            print(f"{question['id']}  {'拒答成功' if refused else '强行作答（基线局限）'}")
            print(f"      最高分 {top.score:.2f}，返回 {top.doc} / {top.chunk.heading_path}")
            details.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "refused": refused,
                    "top_score": round(top.score, 4),
                    "top": f"{top.doc} / {top.chunk.heading_path}",
                }
            )
            continue

        answerable += 1
        gold_headings = [question["gold_heading"], *question.get("extra_headings", [])]
        positions = [
            position
            for position, item in enumerate(ranked, start=1)
            if item.doc == question["gold_doc"] and item.chunk.heading_path in gold_headings
        ]
        best = min(positions) if positions else None

        for k in recall:
            if best is not None and best <= k:
                recall[k] += 1

        rank_text = f"第 {best} 名" if best is not None else "未命中"
        found = f"标准块 {len(gold_headings)} 个，找到 {len(positions)} 个"
        print(f"{question['id']}  gold {rank_text}  ({found})")
        print(f"      榜首 {top.doc} / {top.score:.2f} / {top.chunk.heading_path}")
        print(f"      命中词 {', '.join(top.hits[:4])}")
        details.append(
            {
                "id": question["id"],
                "type": question["type"],
                "best_rank": best,
                "gold_found": len(positions),
                "gold_total": len(gold_headings),
                "top_score": round(top.score, 4),
                "top": f"{top.doc} / {top.chunk.heading_path}",
            }
        )

    print("\n" + "=" * 56)
    print(f"可回答题 {answerable} 道，不可回答题 {unanswerable} 道")
    metrics = {}
    for k in (1, 3, 5):
        value = recall[k] / answerable if answerable else 0.0
        metrics[f"recall@{k}"] = round(value, 4)
        print(f"Recall@{k}: {recall[k]}/{answerable} = {value:.0%}")
    print(f"拒答正确: {abstain_ok}/{unanswerable}")

    logger.info(
        "关键词基线评测完成",
        extra={
            "fields": {
                "chunks": len(chunks),
                "questions": len(questions),
                "metrics": metrics,
                "abstain_ok": abstain_ok,
                "unanswerable": unanswerable,
            }
        },
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "keyword_coverage",
        "corpus_dir": str(settings.corpus_dir),
        "chunks": len(chunks),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "metrics": metrics,
        "abstain_ok": abstain_ok,
        "details": details,
    }


def main() -> None:
    configure_logging()
    settings = get_settings()
    result = evaluate(settings)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output = settings.results_dir / "keyword_baseline.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：{output}")


if __name__ == "__main__":
    main()
