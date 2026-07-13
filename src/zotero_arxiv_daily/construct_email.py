from .protocol import Paper


def _limited_names(values: list[str] | None, limit: int, unknown: str) -> str:
    names = values or []
    rendered = ", ".join(names[:limit])
    if len(names) > limit:
        rendered += ", ..."
    return rendered or unknown


def render_paper_markdown(paper: Paper, index: int) -> str:
    """Render one paper as stable Markdown suitable for humans and downstream LLMs."""
    authors = _limited_names(paper.authors, 3, "Unknown Authors")
    affiliations = _limited_names(paper.affiliations, 3, "Unknown Affiliation")
    published_date = paper.published_date or "Unknown Date"
    tldr = paper.tldr or "TLDR 生成失败：本次无法生成可靠的中文总结，请查看英文摘要。"
    abstract = paper.abstract or "No abstract available."

    links = []
    if paper.url:
        links.append(f"[arXiv]({paper.url})")
    if paper.pdf_url:
        links.append(f"[PDF]({paper.pdf_url})")
    rendered_links = " | ".join(links) or "No link available."

    return f"""## {index}. {paper.title or 'Untitled'}

**作者：** {authors}

**机构：** {affiliations}

**发布日期：** {published_date}

### TLDR

{tldr}

### Abstract

{abstract}

**链接：** {rendered_links}
"""


def render_email(papers: list[Paper]) -> str:
    """Build the email's canonical Markdown/plain-text representation."""
    if not papers:
        return "# Daily arXiv Papers\n\nNo Papers Today. Take a Rest!\n"

    blocks = [render_paper_markdown(paper, index) for index, paper in enumerate(papers, 1)]
    content = "\n---\n\n".join(blocks)
    return (
        f"# Daily arXiv Papers\n\n"
        f"今日精选论文：{len(papers)} 篇。\n\n"
        f"{content}\n\n"
        "---\n\n"
        "To unsubscribe, remove your email in your GitHub Action settings.\n"
    )
