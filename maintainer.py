import asyncio
import os
import random
import zipfile
from playwright.async_api import async_playwright

BASE = "https://sandbox.pikker.ee/analysis"
DOWNLOAD_DIR = "sandbox/reports/raw"
max_downloads = 10

async def get_high_score_ids(page, threshold=5.0):
    await page.goto(BASE, wait_until="networkidle")
    await page.wait_for_selector("#recent tbody tr", timeout=15000)

    rows = await page.query_selector_all("#recent tbody tr")

    ids = []

    for row in rows:
        cols = await row.query_selector_all("td")
        if len(cols) < 6:
            continue

        analysis_id = (await cols[0].inner_text()).strip()
        score_text = (await cols[5].inner_text()).strip()

        try:
            score = float(score_text.split(":")[1].strip())
        except:
            continue

        if score >= threshold:
            ids.append(analysis_id)

    return ids

async def get_md5(page, analysis_id):
    summary_url = f"{BASE}/{analysis_id}/summary/"
    await page.goto(summary_url, wait_until="networkidle")
    
    try:
        element = page.locator("th:text-is('MD5') + td")
        md5_hash = await element.inner_text(timeout=5000)
        return md5_hash.strip()
    except Exception as e:
        print(f"[-] Could not extract MD5 for {analysis_id}: {e}")
        return None

async def download_report(page, analysis_id, md5_hash):
    export_url = f"{BASE}/{analysis_id}/export/"
    await page.goto(export_url, wait_until="networkidle")

    await page.wait_for_selector("form")

    await page.evaluate("""
    (() => {
        document.querySelectorAll("input[name='dirs']").forEach(el => {
            el.checked = false;
        });

        const reports = document.querySelector("input[name='dirs'][value='reports']");
        if (reports) {
            reports.checked = true;
        }
    })();
    """)

    async with page.expect_download() as download_info:
        await page.click("button[type=submit]")

    download = await download_info.value
    
    zip_path = os.path.join(DOWNLOAD_DIR, f"{analysis_id}.zip")
    await download.save_as(zip_path)
    print(f"[+] Downloaded {analysis_id}.zip")

    json_path = os.path.join(DOWNLOAD_DIR, f"{md5_hash}.json")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            if 'reports/report.json' in z.namelist():
                with open(json_path, 'wb') as f:
                    f.write(z.read('reports/report.json'))
                print(f"[+] Extracted and saved as {md5_hash}.json")
            else:
                print(f"[-] reports/report.json not found inside {analysis_id}.zip")
    except zipfile.BadZipFile:
        print(f"[!] Invalid zip file downloaded for {analysis_id}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"[-] Purged original archive {analysis_id}.zip")

    return True

async def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    existing_md5s = {
        f.split(".")[0]
        for f in os.listdir(DOWNLOAD_DIR)
        if f.endswith(".json")
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
            accept_downloads=True
        )

        page = await context.new_page()

        ids = await get_high_score_ids(page, threshold=7.0)
        print(f"[+] Found {len(ids)} high-scoring reports.")

        downloaded_count = 0

        for aid in ids:
            # Enforce max downloads limit
            if downloaded_count >= max_downloads:
                print("[+] Reached maximum download limit.")
                break

            md5_hash = await get_md5(page, aid)
            if not md5_hash:
                continue
            
            if md5_hash in existing_md5s:
                print(f"[i] Already have report for MD5 {md5_hash} (ID: {aid}), skipping.")
                continue

            try:
                success = await download_report(page, aid, md5_hash)
                if not success:
                    continue
                
                downloaded_count += 1
                existing_md5s.add(md5_hash) 
            except Exception as e:
                print(f"[!] Failed {aid}: {e}")
                continue

            if downloaded_count < max_downloads:
                delay = random.uniform(8, 12)
                print(f"[i] Sleeping {delay:.2f}s")
                await asyncio.sleep(delay)

        await browser.close()


if __name__ == "__main__":
    import sys
    max_downloads = int(sys.argv[1])
    asyncio.run(main())