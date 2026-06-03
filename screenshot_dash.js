const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto("http://localhost:8501", { waitUntil: "networkidle", timeout: 30000 });
  await page.waitForSelector("h1", { timeout: 20000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: "dashboard_phase3.png", fullPage: false });
  await browser.close();
  console.log("Screenshot saved.");
})();
