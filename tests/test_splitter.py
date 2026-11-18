"""切分器行为测试：重点覆盖边界，而不是重复验证正常路径。"""

from __future__ import annotations

import pytest

from supportpilot.ingest.splitter import (
    detect_fixed_issues,
    split_by_heading,
    split_fixed,
)


def _text(length: int) -> str:
    return "".join(str(i % 10) for i in range(length))


def test_fixed_rejects_overlap_not_less_than_size() -> None:
    with pytest.raises(ValueError):
        split_fixed("abc", size=10, overlap=10)


def test_fixed_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        split_fixed("abc", size=0, overlap=0)


def test_fixed_handles_empty_and_blank_text() -> None:
    assert split_fixed("") == []
    assert split_fixed("   \n\t ") == []


def test_fixed_respects_size_and_overlap() -> None:
    chunks = split_fixed(_text(95), size=30, overlap=10)

    assert all(len(chunk) <= 30 for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous[-10:] == following[:10]


def test_heading_path_is_nested() -> None:
    markdown = "# A\n\na1\n\n## B\n\nb1\n\n### C\n\nc1\n"
    chunks = split_by_heading(markdown)
    assert [chunk.heading_path for chunk in chunks] == ["A", "A > B", "A > B > C"]


def test_document_without_heading() -> None:
    chunks = split_by_heading("没有标题的一段正文。")
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""


def test_table_stays_in_one_chunk() -> None:
    markdown = "# 表\n\n| 错误码 | 含义 |\n| --- | --- |\n| E-409 | 证书过期 |\n"
    chunks = split_by_heading(markdown, max_chars=1000, min_chars=1)

    assert len(chunks) == 1
    assert "| E-409 | 证书过期 |" in chunks[0].text


def test_short_tail_merges_into_same_section() -> None:
    body = "正文内容" * 12
    markdown = f"# T\n\n{body}\n\n短\n"
    chunks = split_by_heading(markdown, max_chars=60, min_chars=20)

    assert len(chunks) == 1
    assert "短" in chunks[0].text


def test_short_chunk_does_not_cross_sections() -> None:
    markdown = "# T\n\n第一节正文。\n\n## S\n\n短\n"
    chunks = split_by_heading(markdown, max_chars=200, min_chars=20)

    assert [chunk.heading_path for chunk in chunks] == ["T", "T > S"]


def test_ordinals_are_contiguous_after_merge() -> None:
    markdown = "# T\n\n" + "\n\n".join(["段落内容" * 8] * 4)
    chunks = split_by_heading(markdown, max_chars=40, min_chars=20)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_detect_fixed_issues_flags_heading_tail() -> None:
    chunks = ["正文正文\n## 4. 导入证书", "证书内容。"]
    issues = detect_fixed_issues(chunks)

    assert len(issues) == 1
    assert "标题" in issues[0]


def test_detect_fixed_issues_flags_broken_table_row() -> None:
    chunks = ["| 错误码 | 含义 |\n|", " --- | --- |"]
    issues = detect_fixed_issues(chunks)

    # 前半块被判定为表格劈开，后半块是残缺片段，也会被标记
    assert any("表格" in issue for issue in issues)
