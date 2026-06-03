from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("h1", timeout=15000)
    page.wait_for_timeout(3000)
    # Scroll the Streamlit main area (it uses a specific scrollable div)
    page.evaluate("""
        const main = document.querySelector('[data-testid="stMain"]') ||
                     document.querySelector('.main') ||
                     document.querySelector('[class*="main"]');
        if (main) main.scrollTop = 99999;
        else window.scrollTo(0, 99999);
    """)
    page.wait_for_timeout(2000)
    page.screenshot(path="dashboard_bottom.png", full_page=False)
    browser.close()
    print("Done.")
