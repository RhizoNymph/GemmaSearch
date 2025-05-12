from lxml import html as lxml_html
from playwright.sync_api import TimeoutError as PWTimeout
from readability import Document
from utils.playwright_manager import browser_context
import tempfile
import os
import requests
from PyPDF2 import PdfReader
import io

def _render_and_extract(url: str, timeout_ms: int = 20_000) -> str:
    """Render page in Playwright, then distil main readable text."""
    if url.lower().endswith('.pdf'):
        return _download_and_parse_pdf(url)
        
    p = browser_context.new_page()
    try:
        p.goto(url, timeout=timeout_ms, wait_until="networkidle")
        html_src = p.content()
    except PWTimeout:
        raise RuntimeError(f"Timed‑out fetching {url}")
    finally:
        p.close()

    readable = Document(html_src)
    main_html = readable.summary()
    return lxml_html.fromstring(main_html).text_content().strip()

def _download_and_parse_pdf(url: str) -> str:
    """Download and parse a PDF file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        pdf_content = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_content)
        
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
            
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"Error downloading/parsing PDF: {str(e)}")