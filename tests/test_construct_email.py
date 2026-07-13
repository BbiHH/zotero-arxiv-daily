"""Tests for Markdown email rendering."""

from zotero_arxiv_daily.construct_email import render_email, render_paper_markdown
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(tldr="中文总结。", affiliations=["MIT"])]

    content = render_email(papers)

    assert content.startswith("# Daily arXiv Papers")
    assert "今日精选论文：1 篇。" in content
    assert "## 1. Sample Paper Title" in content
    assert "**作者：** Author A, Author B, Author C" in content
    assert "**机构：** MIT" in content
    assert "**发布日期：** 2026-07-13" in content
    assert "### TLDR\n\n中文总结。" in content
    assert "### Abstract\n\nThis paper explores a novel approach" in content
    assert "[arXiv](https://arxiv.org/abs/2026.00001)" in content
    assert "[PDF](https://arxiv.org/pdf/2026.00001)" in content
    assert "Relevance" not in content
    assert "<table" not in content


def test_render_email_empty_list():
    content = render_email([])
    assert "No Papers Today" in content


def test_render_email_limits_authors_to_three():
    paper = make_sample_paper(authors=[f"Author {i}" for i in range(10)], tldr="ok")
    content = render_email([paper])

    assert "Author 0, Author 1, Author 2, ..." in content
    assert "Author 3" not in content
    assert "Author 9" not in content


def test_render_email_limits_affiliations_to_three():
    paper = make_sample_paper(
        affiliations=[f"Uni {i}" for i in range(8)],
        tldr="ok",
    )
    content = render_email([paper])

    assert "Uni 0, Uni 1, Uni 2, ..." in content
    assert "Uni 3" not in content
    assert "Uni 7" not in content


def test_render_email_unknown_affiliation():
    content = render_email([make_sample_paper(affiliations=None, tldr="ok")])
    assert "Unknown Affiliation" in content


def test_render_paper_markdown_has_stable_section_order():
    paper = make_sample_paper(tldr="总结", affiliations=["MIT"])
    content = render_paper_markdown(paper, 3)

    markers = [
        "## 3. Sample Paper Title",
        "**作者：**",
        "**机构：**",
        "**发布日期：**",
        "### TLDR",
        "### Abstract",
        "**链接：**",
    ]
    positions = [content.index(marker) for marker in markers]
    assert positions == sorted(positions)
