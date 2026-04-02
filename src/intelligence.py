"""Mentat intelligence layer — news sentiment via Groq API."""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta

import httpx
from bs4 import BeautifulSoup

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ── Scrapers ──────────────────────────────────────────────────────────────────

def scrape_moneycontrol_headlines() -> list[dict]:
    """Scrape MC market news. Returns list of {headline, url, source}."""
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://www.moneycontrol.com/news/business/markets/",
        "https://www.moneycontrol.com/news/business/stocks/",
    ]
    articles = []
    for url in urls:
        try:
            resp = httpx.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.select("li.clearfix h2 a, .news_listing a"):
                title = tag.get_text(strip=True)
                href = tag.get("href", "")
                if title and len(title) > 20:
                    articles.append(
                        {
                            "headline": title,
                            "url": href,
                            "source": "MoneyControl",
                        }
                    )
        except Exception:
            continue
    return articles[:30]  # cap at 30 per source


def scrape_economic_times_headlines() -> list[dict]:
    """Scrape ET Markets headlines."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://economictimes.indiatimes.com/markets/stocks/news"
    articles = []
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.select("div.eachStory h3 a, .story-box h3 a"):
            title = tag.get_text(strip=True)
            if title and len(title) > 20:
                articles.append(
                    {
                        "headline": title,
                        "url": tag.get("href", ""),
                        "source": "Economic Times",
                    }
                )
    except Exception:
        pass
    return articles[:30]


def scrape_bse_announcements(tickers: list[str]) -> list[dict]:
    """
    BSE corporate announcements — the highest signal source.
    Results, order wins, board meetings, insider buying.
    Uses BSE's public API endpoint.
    """
    announcements = []
    today = date.today().strftime("%Y%m%d")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    url = (
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        f"?strCat=-1&strPrevDate={yesterday}&strScrip=&strSearch=P"
        f"&strToDate={today}&strType=C&subcategory=-1"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.bseindia.com",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
        data = resp.json()
        for item in data.get("Table", [])[:50]:
            announcements.append(
                {
                    "headline": item.get("HEADLINE", ""),
                    "company": item.get("SLONGNAME", ""),
                    "scrip": item.get("SCRIP_CD", ""),
                    "source": "BSE Announcement",
                    "category": item.get("CATEGORYNAME", ""),
                }
            )
    except Exception:
        pass
    return announcements


# ── Groq sentiment analysis ───────────────────────────────────────────────────

def analyse_sentiment_batch(
    articles: list[dict],
    known_tickers: list[str],
) -> list[dict]:
    """
    Send headlines to Groq API in batches.
    Returns structured sentiment signals per stock.
    """
    if not articles:
        return []

    if not GROQ_API_KEY:
        print("  [WARN] GROQ_API_KEY not set. Skipping sentiment analysis.")
        return []

    headlines_text = "\n".join(
        f"- [{a['source']}] {a['headline']}" for a in articles[:40]
    )

    tickers_str = ", ".join(known_tickers[:50])

    prompt = f"""You are a financial analyst reviewing Indian stock market news for NSE/BSE.

Today's headlines:
{headlines_text}

Known stocks in the portfolio universe: {tickers_str}

For each headline that creates a SPECIFIC directional catalyst for a stock in the next 5 trading days:
Return a JSON array. Each object must have:
- "ticker": NSE ticker with .NS suffix (or "MARKET" for index-level signal)
- "direction": "BULLISH" or "BEARISH"
- "confidence": integer 1-10 (10 = near-certain catalyst, 1 = very speculative)
- "reason": one sentence, max 15 words
- "headline_ref": first 5 words of the triggering headline

Only include stocks where the news is a DIRECT catalyst — earnings surprise, order win, regulatory decision, management change, or promoter activity. Ignore vague macro news unless it directly impacts a sector.

Return ONLY valid JSON array. No preamble. No markdown. Empty array [] if no clear catalysts."""

    try:
        resp = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  [WARN] Groq API error: {resp.status_code}")
            return []

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        signals = json.loads(raw)
        return signals if isinstance(signals, list) else []
    except json.JSONDecodeError:
        print("  [WARN] Groq returned invalid JSON")
        return []
    except Exception as e:
        print(f"  [WARN] Sentiment API error: {e}")
        return []


# ── Full intelligence run ─────────────────────────────────────────────────────

def run_intelligence_layer(known_tickers: list[str]) -> dict:
    """
    Collect all news sources and return structured sentiment signals.
    Called by pipeline.py as part of the morning run.
    """
    print("  [INTEL] Scraping MoneyControl...")
    mc_articles = scrape_moneycontrol_headlines()
    time.sleep(1)

    print("  [INTEL] Scraping Economic Times...")
    et_articles = scrape_economic_times_headlines()
    time.sleep(1)

    print("  [INTEL] Fetching BSE announcements...")
    bse_articles = scrape_bse_announcements(known_tickers)

    all_articles = mc_articles + et_articles + bse_articles
    print(f"  [INTEL] Analysing {len(all_articles)} headlines with Groq...")

    signals = analyse_sentiment_batch(all_articles, known_tickers)
    print(f"  [INTEL] Found {len(signals)} sentiment signals")

    # Index by ticker for fast lookup in pipeline
    by_ticker: dict[str, list] = {}
    for sig in signals:
        ticker = sig.get("ticker", "MARKET")
        by_ticker.setdefault(ticker, []).append(sig)

    return {
        "signals": signals,
        "by_ticker": by_ticker,
        "n_articles": len(all_articles),
        "date": date.today().isoformat(),
    }
