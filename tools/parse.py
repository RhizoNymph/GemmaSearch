from typing import List, Dict, Optional

from tools.search import _search_ddg, _search_arxiv, _click_on_result
from tools.fetch import _render_and_extract

def get_observation(function_name: str, args: Dict) -> str:
    """Bridge from the model's function call → real tool output."""
    if function_name == "search_ddg":
        return _search_ddg(args["query"])
    if function_name == "search_arxiv":
        return _search_arxiv(args["query"])
    if function_name == "click":
        return _click_on_result(args.get("rank", 0))  # rank defaults to 0 if not provided
    if function_name == "open":
        return _render_and_extract(args["url"])    
    raise ValueError(f"Unsupported function or 'finish' called as observation: {function_name!r}")

def parse_tool_call_gemma(llm_text: str) -> Optional[str]:
    """Parse the tool call from the LLM text."""

    if "```tool_code" not in llm_text:
        return None
        
    try:
        parts = llm_text.split("```tool_code")
        if len(parts) < 2:
            return None
            
        code_block = parts[1].split("```")[0].strip()
        return code_block
    except Exception:
        return None