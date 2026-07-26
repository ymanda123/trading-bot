"""Live financial news for the AI strategy prompt (ai_strategy.py). Pulls
recent headlines from public RSS feeds -- CNBC, Yahoo Finance, and
Bloomberg -- so the model has real, current news to reason about instead of
relying on training data. No API key required.

CNBC's general Finance feed is filtered down to crypto-relevant headlines
since it isn't crypto-specific; the Yahoo BTC-USD feed and Bloomberg Crypto
feed are already on-topic and used as-is.
"""

import logging
from datetime import datetime, timezone

import feedparser
import requests

logger = logging.getLogger("trading-bot")

_FEEDS = [
    {"source": "CNBC", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "filter_keywords": True},
    {"source": "Yahoo Finance", "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US", "filter_keywords": False},
    {"source": "Bloomberg", "url": "https://www.bloomberg.com/feeds/crypto/news.rss", "filter_keywords": False},
]

_CRYPTO_KEYWORDS = ("bitcoin", "crypto", "btc", "ethereum", "blockchain", "stablecoin")

_TIMEOUT_SECONDS = 8


def _mentions_crypto(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _CRYPTO_KEYWORDS)


def _published_at(entry) -> datetime:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _fetch_one_feed(feed: dict) -> list[dict]:
    resp = requests.get(feed["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    items = []
    for entry in parsed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue
        if feed["filter_keywords"] and not _mentions_crypto(title + " " + entry.get("summary", "")):
            continue
        items.append({"source": feed["source"], "title": title, "published_at": _published_at(entry)})
    return items


def fetch_news(max_items: int = 8) -> list[dict]:
    """Best-effort fetch across all feeds. A single feed failing (timeout,
    HTTP error, malformed XML) doesn't block the others -- returns whatever
    succeeded, most recent first, or an empty list if every feed failed."""
    all_items = []
    for feed in _FEEDS:
        try:
            all_items.extend(_fetch_one_feed(feed))
        except Exception:
            logger.warning("News feed fetch failed for %s", feed["source"], exc_info=True)

    all_items.sort(key=lambda item: item["published_at"], reverse=True)
    return all_items[:max_items]


def format_news_for_prompt(items: list[dict]) -> str:
    if not items:
        return "(no live news available)"
    return "\n".join(f"- [{item['source']}] {item['title']}" for item in items)
