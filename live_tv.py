"""Resolves each network's currently-live YouTube video ID for the
dashboard's "Live Market TV" panel (server.py's /live-tv).

The dashboard is a static page, and a browser can't do this resolution
itself: YouTube's channel/live pages don't send CORS headers, and there's
no free official API for "what's live on this channel right now" --
which is why the static page pins a specific video ID per network.
Most of those pins are effectively permanent (Bloomberg, Yahoo Finance,
and CNN each run one persistent 24/7 stream), but NBC News NOW starts a
brand-new video ID for every day's live broadcast, so a hardcoded pin
there goes stale on a ~24h clock. This module re-resolves the current
one here on the backend instead, on a short cache, so the frontend can
always ask for whatever's live right now rather than relying on a pin
that someone has to remember to update.
"""

import logging
import re
import time

import requests

logger = logging.getLogger("trading-bot")

_TIMEOUT_SECONDS = 8
_CACHE_TTL_SECONDS = 600  # 10 minutes -- fresh enough, avoids hammering YouTube

# channel_id: YouTube's stable per-channel ID (this never changes).
# fallback_video_id: last known-good pinned video. Used if a live scrape
# ever fails (network blip, rate limit, YouTube page layout change) or
# finds nothing currently live, so the dashboard still shows *something*
# instead of a blank iframe. CNBC has none: it has no persistent free
# live stream to fall back to (its cable feed is CNBC+ only), so when
# nothing's live there the dashboard is meant to show nothing, same as
# it always has.
NETWORKS = {
    "bloomberg": {"label": "Bloomberg Television", "channel_id": "UCIALMKvObZNtJ6AmdCLP7Lg", "fallback_video_id": "QB5BNdBFujE"},
    "yahoo":     {"label": "Yahoo Finance",         "channel_id": "UCEAZeUIeJs0IjQiqTCdVSIg", "fallback_video_id": "KQp-e_XQnDE"},
    "cnbc":      {"label": "CNBC Television",       "channel_id": "UCrp_UI8XtuYfpiqluWLD7Lw", "fallback_video_id": None},
    "nbc":       {"label": "NBC News",              "channel_id": "UCeY0bbntWzzVIaj2z3QigXg", "fallback_video_id": "TQkfLoqeI2M"},
    "cnn":       {"label": "CNN",                   "channel_id": "UCupvZG-5ko_eiXAupbDfxWw", "fallback_video_id": "GotlA1KKWoo"},
}

_VIDEO_ID_RE = re.compile(r'"videoId":"([^"]+)"')
_IS_LIVE_NOW_RE = re.compile(r'"isLiveNow":(true|false)')

_HEADERS = {"User-Agent": "Mozilla/5.0"}

_cache: dict[str, tuple[float, dict]] = {}


def _scrape_candidate_video_id(channel_id: str) -> str | None:
    """The channel's /live page redirects (client-side) to whatever's
    currently live; the first "videoId" in its rendered page state is
    that video -- but this page doesn't reliably say whether that video
    is *actually* live right now (it can point at the most recent
    broadcast even after it's ended), hence the separate liveness check
    in _is_video_live below."""
    url = f"https://www.youtube.com/channel/{channel_id}/live"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    match = _VIDEO_ID_RE.search(resp.text)
    return match.group(1) if match else None


def _is_video_live(video_id: str) -> bool:
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    match = _IS_LIVE_NOW_RE.search(resp.text)
    return bool(match and match.group(1) == "true")


def _resolve_one(key: str, info: dict) -> dict:
    entry = {"label": info["label"], "channel_id": info["channel_id"], "video_id": None, "source": "none"}
    try:
        candidate = _scrape_candidate_video_id(info["channel_id"])
        if candidate and _is_video_live(candidate):
            entry["video_id"] = candidate
            entry["source"] = "live"
    except Exception:
        logger.warning("Live TV resolve failed for %s", key, exc_info=True)

    if entry["video_id"] is None and info["fallback_video_id"]:
        entry["video_id"] = info["fallback_video_id"]
        entry["source"] = "fallback"

    return entry


def get_live_tv() -> dict:
    """Returns {network_key: {"label", "channel_id", "video_id", "source"}}
    for every configured network, resolving from YouTube (and caching for
    _CACHE_TTL_SECONDS) only when the cached entry has expired. "source" is
    "live" when freshly confirmed live, "fallback" when the scrape failed
    or found nothing live and a pinned ID was used instead, or "none" when
    neither is available."""
    now = time.time()
    result = {}
    for key, info in NETWORKS.items():
        cached = _cache.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            result[key] = cached[1]
            continue

        entry = _resolve_one(key, info)
        _cache[key] = (now, entry)
        result[key] = entry
    return result
