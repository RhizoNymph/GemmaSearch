import html
from typing import List, Dict
from duckduckgo_search import DDGS
import arxiv
from tools.fetch import _render_and_extract

_LAST_RESULTS: List[Dict] = []

def _search_ddg(query: str, k: int = 10) -> str:
    """Return formatted DuckDuckGo results and cache for click()."""
    _LAST_RESULTS.clear()

    with DDGS() as ddgs:
        for r in ddgs.text(query, backend="lite", max_results=k):
            _LAST_RESULTS.append(r)

    return "\n\n".join(
        f"【{i}†{r['title']}†{r['href']}\n{html.unescape(r.get('body', ''))}】"
        for i, r in enumerate(_LAST_RESULTS)
    )

def _search_arxiv(query: str, k: int = 10) -> str:
    """Return formatted arXiv results and cache for click()."""
    _LAST_RESULTS.clear()
    
    search = arxiv.Search(
        query=query,
        max_results=k,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    for result in search.results():
        _LAST_RESULTS.append({
            'title': result.title,
            'href': result.entry_id,
            'body': result.summary,
            'authors': [author.name for author in result.authors],
            'published': result.published.strftime("%Y-%m-%d"),
            'pdf_url': result.pdf_url
        })
    
    return "\n\n".join(
        f"【{i}†{r['title']}†{r['href']}\n"
        f"Authors: {', '.join(r['authors'])}\n"
        f"Published: {r['published']}\n"
        f"{html.unescape(r['body'])}】"
        for i, r in enumerate(_LAST_RESULTS)
    )

def _click_on_result(rank: int) -> str:
    if not _LAST_RESULTS:
        raise RuntimeError("click() called before any search()")
    if not (0 <= rank < len(_LAST_RESULTS)):
        raise ValueError(f"rank must be 0‑{len(_LAST_RESULTS)-1}")
    return _render_and_extract(_LAST_RESULTS[rank]["href"])
