### Market data
### Literature lays out data suggestions and pipeline implementation

import pandas as pd
import yfinance as yf

sp500 = yf.download("^GSPC", start="2015-01-01", end="2025-01-01")

# DataFrame with Date index and columns Open, High, Low, Close (used for Returns), Volume, Adj Close
sp500['Return'] = sp500['Close'].pct_change()
sp500 = sp500.reset_index() # Date column for merging
sp500['Date'] = pd.to_datetime(sp500['Date']).dt.date 

sp500['Date'] = pd.to_datetime(sp500['Date']).dt.strftime('%Y-%m-%d') # Converting Date to string format for merging

# Fetching macro data with VIX:
vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01")[['Close']]
vix.rename(columns={'Close':'VIX'}, inplace=True)
vix = vix.reset_index()
vix['Date'] = pd.to_datetime(vix['Date']).dt.date
vix['Date'] = pd.to_datetime(vix['Date']).dt.strftime('%Y-%m-%d')

# Merging the two DataFrames on 'Date'
merged_data = pd.merge(sp500, vix, on='Date', how='inner')
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=False)
merged_data.dropna(inplace=True)

merged_data.to_csv('sp500_vix_data.csv', index=False)



### NYT Article Summary Data
import os, time, json, sys
from datetime import date
from typing import List, Dict

import requests
import pandas as pd

# Configuration:
NYT_API_KEY = os.getenv("NYT_API_KEY", "your_real_key")

START_DATE  = date(2015, 1, 1)
END_DATE    = date(2025, 1, 1)
CSV_PATH    = "nyt_business_archive.csv"
REQS_PER_MIN = 5
SLEEP_SEC   = 60 / REQS_PER_MIN        # The 12s pause forces code to stay within the API rate limit of 5-calls per minute
CHECKPOINT  = "archive_checkpoint.json"
ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

def archive_month(year: int, month: int, max_retries: int = 5) -> Dict: # Wrapper handles 429 and retries
    """Fetch one month from NYT Archive API with rate-limit back-off."""
    url    = ARCHIVE_URL.format(year=year, month=month)
    params = {"api-key": NYT_API_KEY}

    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 429:                # Too Many Requests
            wait = int(resp.headers.get("Retry-After", "15"))
            print(f"429 received — sleeping {wait}s and retrying …")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Failed to fetch {year}-{month:02} after {max_retries} retries")

def iter_months(start: date, end: date):
    y, m = start.year, start.month
    while True:
        current = date(y, m, 1)
        if current > end.replace(day=1):
            break
        yield current
        m += 1
        if m == 13:
            m, y = 1, y + 1

def load_checkpoint():
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT) as fp:
            return date.fromisoformat(json.load(fp)["last_month"])
    return None

def save_checkpoint(d: date):
    with open(CHECKPOINT, "w") as fp:
        json.dump({"last_month": d.isoformat()}, fp)

def append_rows(rows: List[Dict]):
    pd.DataFrame(rows).to_csv(
        CSV_PATH,
        mode="a",
        index=False,
        header=not os.path.isfile(CSV_PATH),
    )

def is_business(doc: Dict) -> bool:
    sec  = (doc.get("section_name") or "").lower()
    desk = (doc.get("news_desk")    or "").lower()
    return "business" in sec or "business" in desk

def main():
    resume_from = load_checkpoint()
    months      = list(iter_months(START_DATE, END_DATE))
    if resume_from:
        months = [m for m in months if m > resume_from]
        print(f"Resuming after {resume_from} — {len(months)} months left.")

    for first in months:
        print(f"Fetching {first:%Y-%m} …", end=" ", flush=True)
        data = archive_month(first.year, first.month)      
        docs = data["response"]["docs"]

        rows = [
            {
                "Date": doc["pub_date"][:10],
                "Headline": doc["headline"]["main"],
                "Summary": doc.get("abstract") or doc.get("snippet", ""),
                "Section": doc.get("section_name") or doc.get("news_desk"),
                "URL": doc["web_url"],
            }
            for doc in docs if is_business(doc)
        ]

        append_rows(rows)
        save_checkpoint(first)
        print(f"kept {len(rows):3d} / {len(docs):3d} docs")
        time.sleep(SLEEP_SEC)                              

    print("All months processed. CSV downloaded")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved, just rerun to continue.")



### Aggregate Sector Mapping for NYT Article Summaries
import os, time, json, sys, re
from datetime import date
from typing import List, Dict

import requests
import pandas as pd

path = "nyt_business_archive.csv"

df = pd.read_csv(path, header=None)  # no header to see all columns
max_cols = df.shape[1]

if max_cols == 5 and "Summary" not in pd.read_csv(path, nrows=0).columns:
    df.columns = ["Date", "Headline", "Summary", "Section", "URL"]
    df.to_csv(path, index=False)
    print("✔ Added missing 'Summary' header and rewrote file.")
else:
    print("No header problem detected — nothing changed.")

# Configuration:
NYT_API_KEY  = os.getenv("NYT_API_KEY", "your_key")
START_DATE   = date(2015, 1, 1)
END_DATE     = date(2025, 1, 1)
CSV_PATH     = "nyt_business_archive.csv"
AGG_PATH     = "nyt_aggregated_data.csv"
REQS_PER_MIN = 5
SLEEP_SEC    = 60 / REQS_PER_MIN       # 12 s → 5 req/min
CHECKPOINT   = "archive_checkpoint.json"
ARCHIVE_URL  = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

def archive_month(year: int, month: int, retries=5) -> Dict: # NYT API call with 429 back-off to handle rate limits
    url, params = ARCHIVE_URL.format(year=year, month=month), {"api-key": NYT_API_KEY}
    for _ in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:                       # Too Many Requests
            wait = int(r.headers.get("Retry-After", "15"))
            print(f"⚠429 – sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {retries} retries: {year}-{month:02}")

def iter_months(start: date, end: date): # Generator = iterates over months in a date range
    y, m = start.year, start.month
    while True:
        first = date(y, m, 1)
        if first > end.replace(day=1):
            break
        yield first
        m += 1
        if m == 13:
            m, y = 1, y + 1

def load_checkpoint(): # Setting up checkpoint management for resuming crawls
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT) as fp:
            return date.fromisoformat(json.load(fp)["last_month"])
    return None

def save_checkpoint(d: date): # Saving the last processed month to a checkpoint file
    with open(CHECKPOINT, "w") as fp:
        json.dump({"last_month": d.isoformat()}, fp)

def append_rows(rows: List[Dict]): # Rows to the CSV file, creating it if it doesn't exist
    pd.DataFrame(rows).to_csv(
        CSV_PATH,
        mode="a",
        index=False,
        header=not os.path.isfile(CSV_PATH),
    )

def is_business(doc: Dict) -> bool: # Business filter
    sec  = (doc.get("section_name") or "").lower()
    desk = (doc.get("news_desk")    or "").lower()
    return "business" in sec or "business" in desk

def crawl_nyt_archive(): # Main crawling function to fetch NYT archive data
    resume_from = load_checkpoint()
    months      = list(iter_months(START_DATE, END_DATE))
    if resume_from:
        months = [m for m in months if m > resume_from]
        print(f"Resuming after {resume_from} – {len(months)} months left.")

    for first in months:
        print(f"Fetching {first:%Y-%m} …", end=" ", flush=True)
        data = archive_month(first.year, first.month)
        docs = data["response"]["docs"]

        rows = [
            {
                "Date": doc["pub_date"][:10],
                "Headline": doc["headline"]["main"],
                "Summary": doc.get("abstract") or doc.get("snippet", ""),
                "Section": doc.get("section_name") or doc.get("news_desk"),
                "URL": doc["web_url"],
            }
            for doc in docs if is_business(doc)
        ]

        append_rows(rows)
        save_checkpoint(first)
        print(f"kept {len(rows):3d} / {len(docs):3d}")
        time.sleep(SLEEP_SEC)

    print("NYT crawl complete.")

# Sector classification & aggregation
SECTOR_MAP: dict[str, list[str]] = {
    # Information Tech Secotr:
    "Software & IT Services": [
        "software", "saas", "cloud", "it services", "consulting",
        "microsoft", "adobe", "oracle", "sap", "salesforce", "servicenow",
        "workday", "vmware", "accenture", "infosys", "tcs", "capgemini",
    ],
    "Hardware & Devices": [
        "hardware", "pc", "laptop", "smartphone", "iphone", "ipad",
        "dell", "hp", "lenovo", "asus", "acer", "logitech",
    ],
    "Semiconductors": [
        "chip", "chips", "semiconductor", "fab", "foundry",
        "intel", "amd", "nvidia", "qualcomm", "tsmc", "broadcom",
        "micron", "arm holdings", "sk hynix",
    ],
    "Internet & Social Media": [
        "google", "alphabet", "youtube", "search engine",
        "meta", "facebook", "instagram", "whatsapp", "threads",
        "twitter", "x corp", "snapchat", "tiktok", "reddit",
        "linkedin", "pinterest", "social media",
    ],
    # Communication & Media Sector:
    "Telecommunications": [
        "telecom", "5g", "wireless", "broadband",
        "verizon", "at&t", "t-mobile", "comcast", "charter",
        "vodafone", "telefonica", "bt group", "rogers", "singtel",
    ],
    "Media & Entertainment": [
        "media", "streaming", "disney", "espn", "hulu",
        "netflix", "warner bros", "hbo", "paramount", "peacock",
        "sony pictures", "universal", "box office", "cinema",
    ],
    # Consumer Sector:
    "Retail & E-Commerce": [
        "retail", "e-commerce", "amazon", "alibaba", "shopify",
        "ebay", "etsy", "walmart", "target", "costco", "kroger",
        "best buy", "flipkart", "mercado libre",
    ],
    "Consumer Goods & Apparel": [
        "nike", "adidas", "lululemon", "puma", "under armour",
        "apparel", "footwear", "luxury", "lvmh", "gucci", "burberry",
        "rolex", "hermes", "tapestry",
    ],
    "Food & Beverage": [
        "food", "beverage", "coca-cola", "pepsico", "nestlé",
        "restaurant", "fast food", "mcdonald", "starbucks",
        "yum brands", "kfc", "pizza hut", "chipotle",
        "kraft", "general mills", "heinz", "tyson foods",
    ],
    "Hospitality & Leisure": [
        "hotel", "marriott", "hilton", "hyatt", "airbnb",
        "booking.com", "expedia", "travel", "cruise", "carnival",
        "royal caribbean", "las vegas sands", "mgm resorts",
    ],
    "Automotive": [
        "automotive", "auto", "car", "vehicle", "ev",
        "tesla", "general motors", "ford", "stellantis",
        "volkswagen", "toyota", "nissan", "bmw", "mercedes",
        "hyundai", "kia", "rivian", "lucid",
    ],
    # Healthcare Sector:
    "Pharmaceuticals": [
        "pharma", "drug", "medicine", "vaccine", "fda",
        "pfizer", "moderna", "johnson & johnson", "merck",
        "novartis", "roche", "astrazeneca", "bayer", "gsk",
    ],
    "Biotechnology": [
        "biotech", "gene therapy", "crispr", "genomics",
        "illumina", "gilead", "amgen", "biogen", "regeneron",
        "vertex", "bluebird bio",
    ],
    "Medical Devices & Services": [
        "medical device", "medtech", "diagnostics", "surgical",
        "medtronic", "boston scientific", "abbott", "stryker",
        "philips healthcare", "siemens healthineers", "cardinal health",
        "hospital", "clinic", "healthcare services",
    ],
    # Energy & Utilities Sector:
    "Oil & Gas": [
        "oil", "gas", "petroleum", "upstream", "downstream",
        "exxon", "chevron", "bp", "shell", "totalenergies",
        "conocophillips", "aramco", "occidental", "slb",
    ],
    "Renewables & Clean Energy": [
        "renewable", "solar", "wind", "geothermal", "hydro",
        "clean energy", "green energy", "next era", "sunpower",
        "first solar", "enphase", "vestas", "siemens gamesa",
        "hydrogen", "electrolyzer", "fuel cell",
    ],
    "Utilities": [
        "utility", "power grid", "electricity", "water utility",
        "natural gas utility", "duke energy", "southern company",
        "dominion", "pg&e", "national grid", "aes",
    ],
    # Financials Sector:
    "Banks": [
        "bank", "commercial bank", "jpmorgan", "bank of america",
        "citigroup", "wells fargo", "goldman sachs", "morgan stanley",
        "u.s. bancorp", "hsbc", "barclays", "santander", "dbs",
    ],
    "Investment & Asset Management": [
        "asset manager", "blackrock", "vanguard", "fidelity",
        "state street", "schwab", "hedge fund", "private equity",
        "kkr", "carried interest", "mutual fund", "etf",
        "sovereign wealth fund",
    ],
    "Insurance": [
        "insurance", "insurer", "aig", "allstate", "progressive",
        "metlife", "prudential", "chubb", "berkshire hathaway insurance",
        "reinsurance", "lloyd's", "actuarial",
    ],
    "Fintech & Payments": [
        "fintech", "payment", "visa", "mastercard",
        "american express", "paypal", "block inc", "square",
        "stripe", "sofi", "robinhood", "buy now pay later",
        "klarna", "affirm", "ant financial",
    ],
    "Cryptocurrency & Blockchain": [
        "bitcoin", "ethereum", "crypto", "blockchain",
        "coinbase", "binance", "defi", "nft",
        "stablecoin", "mining rig", "hashrate",
    ],
    # Industrial Sector:
    "Aerospace & Defense": [
        "aerospace", "defense", "boeing", "airbus", "northrop",
        "lockheed martin", "raytheon", "bae systems", "general dynamics",
        "drones", "satellite", "nasa contract",
    ],
    "Transportation & Logistics": [
        "shipping", "freight", "logistics", "supply chain",
        "fedex", "ups", "dhl", "maersk", "csx", "union pacific",
        "delta airlines", "american airlines", "united airlines",
        "railroad", "port congestion",
    ],
    "Manufacturing & Machinery": [
        "manufacturing", "factory", "industrial", "caterpillar",
        "3m", "general electric", "siemens", "honeywell", "emerson",
        "robotics", "automation", "abb", "fanuc",
    ],
    "Construction & Engineering": [
        "construction", "engineering", "infrastructure",
        "bechtel", "fluor", "jacobs", "skanska", "kiewit",
        "turner construction", "architect", "building materials",
    ],
    "Chemicals & Specialty Materials": [
        "chemical", "chemicals", "specialty chemical",
        "dupont", "dow", "basf", "lyondellbasell", "air products",
        "eastman", "evonik", "synthetic rubber", "petrochemical",
    ],
    "Metals & Mining": [
        "mining", "metal", "steel", "aluminum", "copper",
        "iron ore", "rio tinto", "bhp", "vale", "newmont",
        "glencore", "lithium", "nickel", "rare earth",
    ],
    "Agriculture": [
        "agriculture", "farming", "crop", "soybean", "corn",
        "wheat", "cargill", "archer daniels midland", "bunge",
        "deere", "monsanto", "fertilizer", "nutrien", "potash",
    ],
    # Real Estate Sector:
    "Real Estate": [
        "real estate", "realtor", "reit", "property", "mortgage",
        "office vacancy", "housing market", "zillow", "redfin",
        "wework", "commercial property", "residential property",
        "industrial park", "logistics park",
    ],
    # ESG / Government / Education Sectors:
    "Environmental & ESG": [
        "esg", "sustainability", "carbon", "emissions",
        "carbon credit", "offset", "green bond", "climate risk",
        "cop28", "environmental regulation",
    ],
    "Government & Policy": [
        "government", "regulation", "legislation", "policy",
        "federal reserve", "congress", "white house",
        "eu commission", "trade tariff", "sanction", "geopolitics",
    ],
    "Education": [
        "education", "edtech", "university", "college", "school",
        "coursera", "edx", "udemy", "chegg", "student loan",
    ],
}

def assign_sector(text: str) -> str: # Function to assign sectors based on keywords in the summary text
    text_low = text.lower()
    for sector, keywords in SECTOR_MAP.items():
        if any(kw in text_low for kw in keywords):
            return sector
    return "General"
  
def aggregate_nyt(df: pd.DataFrame) -> pd.DataFrame: # Aggregation function to combine headlines and summaries by Date and Sector
    """Combine all Business headlines/summaries into one row per Date × Sector."""
    return (
        df.groupby(["Date", "Sector"])
          .agg({
              "Headline": lambda x: " | ".join(x.dropna().astype(str)),
              "Summary":  lambda x: " | ".join(x.dropna().astype(str)),
          })
          .reset_index()
    )

# Main function to run the entire process
def main():
    crawl_nyt_archive() # Crawl or resume NYT archive

    nyt_df = pd.read_csv(CSV_PATH) # Loading full Business CSV
    nyt_df["Headline"] = nyt_df["Headline"].astype(str)
    nyt_df["Summary"] = nyt_df["Summary"].fillna("").astype(str)

    nyt_df["Sector"] = nyt_df["Summary"].fillna("").apply(assign_sector) # Assigning sectors

    nyt_aggregated = aggregate_nyt(nyt_df) # Aggregate is saved
    nyt_aggregated.to_csv(AGG_PATH, index=False)
    print(f"Aggregated file written: {AGG_PATH}")
  
    if "news_df" in globals(): # Reuters clean-up if news_df already exists
        news_df["Article"] = (
            news_df["Article"]
            .str.replace(r"By .*? \|", "", regex=True)
            .str.replace(r"\n+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        print("Reuters news_df cleaned.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – progress saved, just rerun to continue.")