from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("h1", timeout=15000)
    page.wait_for_timeout(3000)
    # Scroll to bottom to reveal the Phase 3 expanders
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(2000)
    page.screenshot(path="dashboard_phase3_bottom.png", full_page=False)
    # Click Feature Importance expander
    expanders = page.locator("summary")
    if expanders.count() > 0:
        expanders.first.click()
        page.wait_for_timeout(2000)
        page.screenshot(path="dashboard_fi_expander.png", full_page=False)
    browser.close()
    print("Screenshots saved.")
