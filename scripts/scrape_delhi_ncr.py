import hashlib
import logging
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
import pandas as pd
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}
BASE_URL = "https://www.99acres.com/search/property/buy/{region}?city={city_id}&preference=S&area_unit=1&res_com=R"
CITY_IDS = {"Gurgaon": 12, "Noida": 21, "Delhi": 6, "Faridabad": 35, "Ghaziabad": 36}
REQUEST_DELAY = 2.0


def parse_listings_html(html: str, region: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("div._3bgq4"):
        try:
            title_tag = card.select_one("div._2ipzj")
            price_tag = card.select_one("div._1hziv")
            area_tag = card.select_one("div._2r7aN")
            locality_tag = card.select_one("div._3Cp4n")
            floor_tag = card.select_one("div.d-flex")

            if not all([title_tag, price_tag, area_tag]):
                continue

            title = title_tag.get_text(strip=True)
            bhk = int(title.split()[0]) if title[0].isdigit() else 2
            price_text = price_tag.get_text(strip=True).replace("₹", "").replace(",", "").strip()
            price_inr = _parse_inr(price_text)
            area_sqft = float(area_tag.get_text(strip=True).replace("sqft", "").strip())
            locality = locality_tag.get_text(strip=True).split(",")[0] if locality_tag else "Unknown"
            floor_text = floor_tag.get_text(strip=True) if floor_tag else "Floor 1 of 5"
            floor, total_floors = _parse_floor(floor_text)

            rows.append({
                "region": region,
                "bhk": bhk,
                "area_sqft": area_sqft,
                "floor": floor,
                "total_floors": total_floors,
                "age_years": 5.0,
                "furnishing": "Semi-Furnished",
                "locality": locality,
                "parking": 1,
                "lift": 1,
                "metro_dist_km": 1.5,
                "price_inr": price_inr,
                "rent_inr": 0,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _parse_inr(text: str) -> float:
    text = text.lower().replace(",", "")
    if "cr" in text:
        return float(text.replace("cr", "").strip()) * 10_000_000
    if "l" in text or "lac" in text:
        return float(text.replace("lac", "").replace("l", "").strip()) * 100_000
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_floor(text: str) -> tuple[int, int]:
    try:
        parts = text.lower().replace("floor", "").replace("of", "/").split("/")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return 1, 5


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    def _hash(row):
        key = f"{row.get('locality','')}|{row.get('area_sqft',0)}|{row.get('floor',0)}|{row.get('price_inr',0)}"
        return hashlib.sha256(key.encode()).hexdigest()
    df = df.copy()
    df["_hash"] = df.apply(_hash, axis=1)
    df = df.drop_duplicates(subset="_hash").drop(columns=["_hash"])
    return df.reset_index(drop=True)


def fetch_region_listings(region: str, max_pages: int = 3) -> pd.DataFrame:
    city_id = CITY_IDS.get(region, 6)
    all_rows = []
    with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
        for page in range(1, max_pages + 1):
            url = BASE_URL.format(region=region.lower(), city_id=city_id) + f"&page={page}"
            logger.info("Fetching %s page %d", region, page)
            try:
                resp = client.get(url)
                resp.raise_for_status()
                df_page = parse_listings_html(resp.text, region)
                all_rows.append(df_page)
            except Exception as e:
                logger.warning("Failed to fetch %s page %d: %s", region, page, e)
            time.sleep(REQUEST_DELAY)
    if not all_rows:
        return pd.DataFrame()
    return deduplicate(pd.concat(all_rows, ignore_index=True))


def main() -> None:
    with open("configs/delhi_ncr_regions.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs("src/data/delhi_ncr", exist_ok=True)

    for region_cfg in config["regions"]:
        if not region_cfg["model_ready"]:
            continue
        region = region_cfg["name"]
        df_new = fetch_region_listings(region, max_pages=5)

        if df_new.empty:
            logger.warning("No listings fetched for %s", region)
            continue

        out_path = f"src/data/delhi_ncr/{region.lower()}_live.csv"
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            df_combined = deduplicate(pd.concat([df_existing, df_new], ignore_index=True))
        else:
            df_combined = df_new

        df_combined.to_csv(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df_combined), out_path)


if __name__ == "__main__":
    main()
