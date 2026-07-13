import asyncio
import urllib.parse
import re
from playwright.async_api import async_playwright

async def _find_company_domain(company_name: str) -> str:
    query = f'"{company_name}" official website'
    encoded = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            print("Title:", await page.title())
            
            for link in await page.query_selector_all("a.result__url"):
                href = await link.get_attribute("href")
                print("href:", href)
                if href:
                    actual_url = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
                    print("actual_url:", actual_url)
                    d = re.search(r"https?://(?:www\.)?([^/]+)", actual_url)
                    if d:
                        domain = d.group(1).lower()
                        print("extracted domain:", domain)
                        if domain not in ["linkedin.com", "wikipedia.org", "facebook.com", "twitter.com", "glassdoor.com", "indeed.com", "youtube.com"] and "google" not in domain:
                            await browser.close()
                            return domain
            await browser.close()
    except Exception as e:
        print("Error:", e)
    return ""

async def main():
    domain = await _find_company_domain("Feedzai")
    print(f"Domain: '{domain}'")

if __name__ == "__main__":
    asyncio.run(main())
