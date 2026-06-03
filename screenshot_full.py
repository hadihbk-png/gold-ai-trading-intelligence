from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("h1", timeout=15000)
    page.wait_for_timeout(4000)
    # Full-page screenshot to capture all Phase 3 expanders
    page.screenshot(path="dashboard_fullpage.png", full_page=True)
    browser.close()
    print("Full-page screenshot saved.")
