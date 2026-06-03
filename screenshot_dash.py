from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    # Poll until the APEX Metals AI title appears (retrain complete + page rerun)
    for i in range(60):
        page.wait_for_timeout(5000)
        content = page.content()
        if "APEX Metals AI" in content and "retraining" not in content.lower():
            break
        print(f"Still retraining... ({i+1}/60)")
    page.wait_for_timeout(3000)
    page.screenshot(path="dashboard_phase3.png", full_page=False)
    browser.close()
    print("Screenshot saved.")
