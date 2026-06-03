from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    # Poll until retrain completes
    for i in range(80):
        page.wait_for_timeout(4000)
        content = page.content()
        if "APEX Metals AI" in content and "retraining" not in content.lower():
            print(f"Done after {i*4}s")
            break
        if i % 3 == 0:
            print(f"Waiting... {i*4}s")
    page.wait_for_timeout(2000)
    page.screenshot(path="dashboard_optimised.png", full_page=False)
    browser.close()
    print("Screenshot saved.")
