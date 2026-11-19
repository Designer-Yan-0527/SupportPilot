"""向量检索基线：BGE-M3 把卡片和问题都变成坐标，Qdrant 按距离找最近邻。

和 keyword_baseline 用同一份考卷、同一套记分，保证对比公平：
唯一的变量是「检索员」——一个数「字面重叠」，一个算「语义距离」。

运行：python -m supportpilot.eval.vector_baseline
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from supportpilot.config import Settings, get_settings
from supportpilot.eval.keyword_baseline import load_chunks
from supportpilot.logging import configure_logging

logger = logging.getLogger(__name__)

COLLECTION = "chunks"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5
# 拒答线：榜首相似度低于它就承认「不知道」。0.45 是拍脑袋的初始值，看完成绩再调
ABSTAIN_THRESHOLD = 0.45


def evaluate(settings: Settings) -> dict[str, object]:
    questions = json.loads(settings.questions_file.read_text(encoding="utf-8"))["questions"]
    chunks = load_chunks(settings.corpus_dir, settings.max_chars, settings.min_chars)
    if not chunks:
        raise SystemExit(f"语料目录为空：{settings.corpus_dir}")

    # ---- 第 2 段：加载模型，建档案室 ----
    print(f"加载嵌入模型 {MODEL_NAME}（首次约 10~20 秒）...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension() or 1024

    client = QdrantClient(path=str(settings.qdrant_dir))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # ---- 第 3 段：四件套入库（id + 坐标 + payload 三件套）----
    texts = [f"{chunk.heading_path}\n{chunk.text}" for _, chunk in chunks]
    print(f"为 {len(texts)} 张卡片计算坐标...")
    vectors = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=index,
            vector=vectors[index].tolist(),
            payload={
                "doc": doc,
                "heading_path": chunk.heading_path,
                "text": chunk.text,
            },
        )
        for index, (doc, chunk) in enumerate(chunks)
    ]
    client.upsert(COLLECTION, points=points)

    # ---- 第 4 段：考试与记分 ----
    recall = {1: 0, 3: 0, 5: 0}
    answerable = 0
    abstain_ok = 0
    unanswerable = 0
    details: list[dict[str, object]] = []

    print(f"档案室就绪：{len(points)} 张卡片，评测 {len(questions)} 题\n")

    for question in questions:
        query_vector = model.encode(question["question"]).tolist()
        hits = client.query_points(COLLECTION, query=query_vector, limit=TOP_K).points
        top = hits[0]
        # payload 类型标注为「可能为空」；正常入库的卡片必有 payload，此处兜底防崩溃
        payload = top.payload or {}

        if question["type"] == "unanswerable":
            unanswerable += 1
            refused = top.score < ABSTAIN_THRESHOLD
            abstain_ok += int(refused)
            print(f"{question['id']}  {'拒答成功' if refused else '强行作答（可调拒答线）'}")
            top_desc = f"{payload['doc']} / {payload['heading_path']}"
            print(f"      最高分 {top.score:.2f}，返回 {top_desc}")
            details.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "refused": refused,
                    "top_score": round(top.score, 4),
                    "top": f"{payload['doc']} / {payload['heading_path']}",
                }
            )
            continue

        answerable += 1
        gold_headings = [question["gold_heading"], *question.get("extra_headings", [])]
        # 只在返回的前 5 名里找 gold——不在就是「未命中」，与 Recall@5 的口径一致
        positions = [
            position
            for position, hit in enumerate(hits, start=1)
            if hit.payload is not None
            and hit.payload["doc"] == question["gold_doc"]
            and hit.payload["heading_path"] in gold_headings
        ]
        best = min(positions) if positions else None

        for k in recall:
            if best is not None and best <= k:
                recall[k] += 1

        rank_text = f"第 {best} 名" if best is not None else "未命中"
        top_line = f"{payload['doc']} / {top.score:.2f} / {payload['heading_path']}"
        print(f"{question['id']}  gold {rank_text}")
        print(f"      榜首 {top_line}")
        details.append(
            {
                "id": question["id"],
                "type": question["type"],
                "best_rank": best,
                "top_score": round(top.score, 4),
                "top": f"{payload['doc']} / {payload['heading_path']}",
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
        "向量基线评测完成",
        extra={
            "fields": {
                "chunks": len(points),
                "questions": len(questions),
                "metrics": metrics,
                "abstain_ok": abstain_ok,
                "unanswerable": unanswerable,
            }
        },
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": f"vector_{MODEL_NAME.lower().replace('/', '_')}",
        "corpus_dir": str(settings.corpus_dir),
        "chunks": len(points),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "metrics": metrics,
        "abstain_ok": abstain_ok,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "details": details,
    }


def main() -> None:
    configure_logging()
    settings = get_settings()
    result = evaluate(settings)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output = settings.results_dir / "vector_baseline.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：{output}")
    print("模型与索引仅本次运行使用，进程结束即释放；下次运行自动重建。")


if __name__ == "__main__":
    main()
