import atexit
from playwright.sync_api import sync_playwright

_play = sync_playwright().start()
_browser = _play.firefox.launch(headless=True, timeout=60_000)
browser_context = _browser.new_context()

def _cleanup_playwright():
    print("Closing Playwright context and browser...")
    if browser_context:
        browser_context.close()
    if _browser:
        _browser.close()
    if _play:
        _play.stop()
    print("Playwright closed.")

atexit.register(_cleanup_playwright)