#!/usr/bin/env python3
"""
curator.py - Personal Daily News Scraper, Gemini Curator & Static HTML Builder.

Fetches RSS feeds defined in feeds.json concurrently with resilient error handling,
extracts article images (from RSS media tags or OpenGraph metadata), filters and
deduplicates items using the Google GenAI SDK (Interactions API with gemini-3.8-flash
and Structured Outputs), and builds the static dashboard to dist/ (and root).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import math
import os
import re
import shutil
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curator")

BASE_DIR = Path(__file__).resolve().parent
FEEDS_PATH = BASE_DIR / "feeds.json"
TEMPLATE_DIR = BASE_DIR
DIST_DIR = BASE_DIR / "dist"

# Note: gemini-2.5-flash is deprecated; using current gemini-3.8-flash via Interactions API
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Mobile Safari/537.36 DaybreakNews/1.0"
)

NOISE_PATTERNS = [
    r"\blive\s*blog\b",
    r"\bas\s+it\s+happened\b",
    r"\bcrossword\b",
    r"\bwordle\b",
    r"\bconnections\s+hint\b",
    r"\bstrands\s+hint\b",
    r"\bquiz\s+of\s+the\s+(day|week)\b",
    r"^watch:\s",
    r"^listen:\s",
    r"\bbest\s+deals\b",
    r"\bpromo\s+code\b",
]
NOISE_REGEX = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

# High-resolution editorial category fallback images when neither RSS nor og:image provides one
CATEGORY_FALLBACK_IMAGES: Dict[str, str] = {
    "uk_news": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=900&q=80",
    "world_news": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=900&q=80",
    "tech_ai": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=900&q=80",
    "sports": "https://images.unsplash.com/photo-1522778119026-d647f0596c20?auto=format&fit=crop&w=900&q=80",
    "entertainment": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=900&q=80",
    "food": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=900&q=80",
    "positive_news": "https://images.unsplash.com/photo-1501854140801-50d01698950b?auto=format&fit=crop&w=900&q=80",
    "niche_hobbies": "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=900&q=80",
}


# ---------------------------------------------------------------------------
# Pydantic Schema for Gemini Structured Output
# ---------------------------------------------------------------------------

CategoryKey = Literal[
    "uk_news",
    "world_news",
    "tech_ai",
    "sports",
    "entertainment",
    "food",
    "niche_hobbies",
]


class UplifterStory(BaseModel):
    title: str = Field(description="Headline of the mandatory uplifting/positive story.")
    summary: str = Field(
        description="A warm, engaging 2-3 sentence summary of the uplifting story."
    )
    detailed_context: List[str] = Field(
        description="3-4 analytical or background bullet points for the 'Tell Me More' drawer."
    )
    source_name: str = Field(description="Name of the publication/feed.")
    source_url: str = Field(description="Direct URL to the original article.")
    image_url: Optional[str] = Field(
        default=None,
        description="Exact image_url from the candidate RSS item (if present).",
    )
    reading_time_minutes: int = Field(
        description="Estimated reading time of the full source article in minutes (typically 2-6)."
    )


class TopStory(BaseModel):
    title: str = Field(description="Clear, informative headline for this major story.")
    category: CategoryKey = Field(
        description="Primary category slug (uk_news, world_news, tech_ai, sports, entertainment, food, niche_hobbies)."
    )
    category_label: str = Field(
        description="Human-readable category badge (e.g., 'UK News', 'World News', 'Tech & AI', 'Sports')."
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Optional sub-tag (e.g., 'Football • Arsenal FC', 'Formula 1', 'Wider Sports').",
    )
    narrative_paragraphs: List[str] = Field(
        description="1 to 2 rich narrative paragraphs (3-5 sentences each) synthesizing the story, key developments, and why it matters."
    )
    detailed_context: List[str] = Field(
        description="3 to 4 analytical bullet points providing deeper context, background, or implications for the 'Tell Me More' drawer."
    )
    source_name: str = Field(description="Primary source publication name.")
    source_url: str = Field(description="Direct outbound URL to the best primary article.")
    image_url: Optional[str] = Field(
        default=None,
        description="Exact image_url from the chosen candidate RSS item (or a corroborating item for the same story).",
    )
    corroborating_sources: List[str] = Field(
        default_factory=list,
        description="Names of other feeds/outlets that also reported on this story, if deduplicated across multiple sources.",
    )
    reading_time_minutes: int = Field(
        description="Estimated reading time of the full outbound article in minutes (2-10)."
    )


class CuratedStory(BaseModel):
    title: str = Field(description="Concise, accurate headline for the story.")
    category: CategoryKey = Field(
        description="Category slug: uk_news, world_news, tech_ai, sports, entertainment, food, or niche_hobbies."
    )
    category_label: str = Field(
        description="Human-readable category name (e.g., 'UK News', 'World News', 'Tech & AI', 'Sports', 'Culture', 'Food', 'Niche')."
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Required for Sports ('Football • Arsenal FC', 'Football • Premier League', 'Formula 1', or 'Wider Sports') and Niche ('Self-Hosting', 'Gaming Patch', or 'QSR Tech'). Optional otherwise.",
    )
    bullet_points: List[str] = Field(
        description="Exactly 2 concise, high-signal bullet points summarizing the story."
    )
    detailed_context: List[str] = Field(
        description="3 to 4 analytical bullet points providing deeper context, stats, or background for the 'Tell Me More' drawer."
    )
    source_name: str = Field(description="Source publication name.")
    source_url: str = Field(description="Direct outbound URL to the original article.")
    image_url: Optional[str] = Field(
        default=None,
        description="Exact image_url from the candidate RSS item.",
    )
    corroborating_sources: List[str] = Field(
        default_factory=list,
        description="Other outlets covering the same story, if deduplicated.",
    )
    reading_time_minutes: int = Field(
        description="Estimated reading time of the full outbound source article in minutes (2-12)."
    )


class ReleaseItem(BaseModel):
    title: str = Field(description="Title of the film, TV series, or video game.")
    medium: Literal["Film", "TV", "Video Game", "Culture"] = Field(
        description="Medium type: Film, TV, Video Game, or Culture."
    )
    release_details: str = Field(
        description="Platform, cinema release note, or release timing this week."
    )
    why_it_matters: str = Field(
        description="1-2 sentence summary of what it is and why it is worth watching/playing."
    )
    source_name: str = Field(description="Source feed name.")
    source_url: str = Field(description="Direct link to the article/review/preview.")
    image_url: Optional[str] = Field(
        default=None,
        description="Exact image_url from the candidate RSS item.",
    )


class MondayReleasesBlock(BaseModel):
    headline: str = Field(
        description="Engaging sub-header for this week's entertainment releases."
    )
    items: List[ReleaseItem] = Field(
        description="4 to 6 notable Film, TV, and Video Game releases or previews highlighted in the Entertainment feeds for this week."
    )


class DailyBriefing(BaseModel):
    edition_tagline: str = Field(
        description="A crisp 1-sentence editorial summary capturing the defining themes of today's briefing."
    )
    daily_uplifter: UplifterStory = Field(
        description="The mandatory Daily Uplifter positive news story."
    )
    top_stories: List[TopStory] = Field(
        description="The 2 to 3 biggest, highest-impact stories of the day (with 1-2 paragraph narrative summaries). These count toward their respective category quotas."
    )
    stories: List[CuratedStory] = Field(
        description="All other curated stories (each with 2 concise bullet points) fulfilling the remaining category quotas."
    )
    monday_releases: Optional[MondayReleasesBlock] = Field(
        default=None,
        description="Populated ONLY when the current day is Monday with a 'Releasing This Week' Film/TV/Game summary block. Null on Tuesday-Sunday.",
    )


# ---------------------------------------------------------------------------
# RSS & Image Extraction Helpers
# ---------------------------------------------------------------------------

def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def upgrade_image_resolution(url: str) -> str:
    """Upgrade known CDN thumbnail sizes (e.g. BBC ichef 240px) to sharp mobile widths."""
    if not url:
        return ""
    # Upgrade BBC ichef small thumbnails (e.g., /standard/240/ -> /standard/800/)
    url = re.sub(r"(ichef\.bbci\.co\.uk/[^/]+/[^/]+/)\d{2,3}/", r"\g<1>800/", url)
    if url.startswith("//"):
        url = "https:" + url
    return url


def extract_entry_image(entry: Any, base_link: str = "") -> Optional[str]:
    """Extract the best available article image URL from a feedparser entry."""
    # 1. media:content (prefer largest width if multiple exist)
    media_contents = getattr(entry, "media_content", None)
    if isinstance(media_contents, list) and media_contents:
        best_url = None
        best_width = -1
        for mc in media_contents:
            if not isinstance(mc, dict):
                continue
            m_url = mc.get("url", "")
            m_type = mc.get("type", "") or mc.get("medium", "")
            if m_url and ("image" in m_type or m_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                w = int(mc.get("width", 0) or 0)
                if w >= best_width:
                    best_width = w
                    best_url = m_url
        if best_url:
            return upgrade_image_resolution(urljoin(base_link, best_url))

    # 2. media:thumbnail
    thumbs = getattr(entry, "media_thumbnail", None)
    if isinstance(thumbs, list) and thumbs:
        for th in thumbs:
            if isinstance(th, dict) and th.get("url"):
                return upgrade_image_resolution(urljoin(base_link, th["url"]))

    # 3. enclosures / links with image mime type
    for link_obj in getattr(entry, "links", []) or []:
        if isinstance(link_obj, dict):
            l_type = link_obj.get("type", "")
            l_href = link_obj.get("href", "")
            if l_href and l_type.startswith("image/"):
                return upgrade_image_resolution(urljoin(base_link, l_href))

    # 4. <img> inside content or summary HTML
    html_blobs = []
    if hasattr(entry, "content") and entry.content:
        for c in entry.content:
            if isinstance(c, dict) and c.get("value"):
                html_blobs.append(c["value"])
    for attr in ("summary", "description"):
        val = getattr(entry, attr, None)
        if val and isinstance(val, str):
            html_blobs.append(val)

    for blob in html_blobs:
        if "<img" in blob:
            soup = BeautifulSoup(blob, "html.parser")
            img = soup.find("img")
            if img:
                src = img.get("src") or img.get("data-src")
                if src and not src.startswith("data:") and "1x1" not in src and "pixel" not in src:
                    return upgrade_image_resolution(urljoin(base_link, str(src)))

    return None


def fetch_og_image(article_url: str) -> Optional[str]:
    """Fetch OpenGraph / Twitter image metadata for selected stories missing an RSS image."""
    if not article_url or not article_url.startswith(("http://", "https://")):
        return None
    try:
        resp = requests.get(
            article_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=4,
        )
        if not resp.ok:
            return None
        # Only parse the <head> portion to keep it fast
        head_html = resp.text[:45000]
        soup = BeautifulSoup(head_html, "html.parser")
        for prop in ("og:image", "twitter:image", "og:image:url"):
            tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content"):
                img_url = str(tag["content"]).strip()
                if img_url.startswith("http"):
                    return upgrade_image_resolution(img_url)
    except Exception:
        pass
    return None


def ensure_briefing_images(
    briefing: DailyBriefing,
    categorized_items: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    Guarantee every story card in the briefing has a valid, high-resolution image URL.
    Maps URLs back to fetched RSS items first, then tries og:image, then category fallback.
    """
    url_to_img: Dict[str, str] = {}
    for items in categorized_items.values():
        for it in items:
            if it.get("url") and it.get("image_url"):
                url_to_img[it["url"]] = it["image_url"]

    def resolve_img(current_img: Optional[str], source_url: str, category_key: str) -> str:
        if current_img and current_img.startswith("http"):
            return upgrade_image_resolution(current_img)
        if source_url in url_to_img:
            return url_to_img[source_url]
        og = fetch_og_image(source_url)
        if og:
            return og
        return CATEGORY_FALLBACK_IMAGES.get(category_key, CATEGORY_FALLBACK_IMAGES["world_news"])

    briefing.daily_uplifter.image_url = resolve_img(
        briefing.daily_uplifter.image_url,
        briefing.daily_uplifter.source_url,
        "positive_news",
    )

    for ts in briefing.top_stories:
        ts.image_url = resolve_img(ts.image_url, ts.source_url, ts.category)

    for st in briefing.stories:
        st.image_url = resolve_img(st.image_url, st.source_url, st.category)

    if briefing.monday_releases:
        for rel in briefing.monday_releases.items:
            rel.image_url = resolve_img(rel.image_url, rel.source_url, "entertainment")


def parse_entry_datetime(entry: Any) -> Optional[datetime.datetime]:
    """Extract a timezone-aware UTC datetime from a feedparser entry if available."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        struct_time = getattr(entry, attr, None)
        if struct_time:
            try:
                return datetime.datetime(*struct_time[:6], tzinfo=datetime.timezone.utc)
            except Exception:
                pass

    for str_attr in ("published", "updated", "created"):
        raw_val = getattr(entry, str_attr, None)
        if raw_val and isinstance(raw_val, str):
            try:
                dt = parsedate_to_datetime(raw_val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.astimezone(datetime.timezone.utc)
            except Exception:
                pass
    return None


def estimate_source_read_time(entry: Any, clean_summary: str, category_key: str) -> int:
    """Estimate reading time (in minutes) of the full outbound article."""
    full_text = clean_summary
    if hasattr(entry, "content") and entry.content:
        try:
            raw_content = " ".join(c.get("value", "") for c in entry.content)
            content_words = clean_html_text(raw_content).split()
            if len(content_words) > len(clean_summary.split()):
                full_text = " ".join(content_words)
        except Exception:
            pass

    word_count = len(full_text.split())
    if word_count >= 400:
        return max(2, min(15, round(word_count / 220)))

    defaults = {
        "uk_news": 3,
        "world_news": 4,
        "tech_ai": 4,
        "sports": 3,
        "entertainment": 5,
        "food": 4,
        "positive_news": 3,
        "niche_hobbies": 4,
    }
    base = defaults.get(category_key, 3)
    link_lower = getattr(entry, "link", "").lower()
    if "quanta" in link_lower or "aeon.co" in link_lower:
        return 9
    return base


def fetch_single_feed(
    category_key: str,
    feed_info: Dict[str, str],
    freshness_hours: int,
    max_items: int,
    now_utc: datetime.datetime,
) -> List[Dict[str, Any]]:
    """Fetch and parse a single RSS/Atom feed with timeout and graceful error recovery."""
    name = feed_info.get("name", "Unknown Feed")
    url = feed_info.get("url", "")
    if not url:
        return []

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Skipping feed '%s' (%s): %s", name, url, exc)
        return []

    try:
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        logger.warning("Failed to parse XML for feed '%s': %s", name, exc)
        return []

    entries = getattr(parsed, "entries", [])
    if not entries:
        logger.warning("Feed '%s' returned 0 entries.", name)
        return []

    cutoff = now_utc - datetime.timedelta(hours=freshness_hours)
    fresh_items: List[Dict[str, Any]] = []
    fallback_items: List[Dict[str, Any]] = []

    for idx, entry in enumerate(entries[: max_items * 2]):
        title = clean_html_text(getattr(entry, "title", ""))
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue

        if NOISE_REGEX.search(title):
            continue

        raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
        summary = clean_html_text(raw_summary)
        if summary.startswith(title):
            summary = summary[len(title) :].lstrip(" -—:.")
        summary_snippet = summary[:480] + ("…" if len(summary) > 480 else "")

        pub_dt = parse_entry_datetime(entry)
        read_time = estimate_source_read_time(entry, summary, category_key)
        image_url = extract_entry_image(entry, link)

        item_data = {
            "category": category_key,
            "source_name": name,
            "feed_rank": idx + 1,
            "title": title,
            "summary": summary_snippet,
            "url": link,
            "image_url": image_url,
            "published_utc": pub_dt.strftime("%Y-%m-%d %H:%M UTC") if pub_dt else "Recent",
            "estimated_read_mins": read_time,
        }

        fallback_items.append(item_data)
        if pub_dt is None or pub_dt >= cutoff:
            fresh_items.append(item_data)

    selected = fresh_items[:max_items]
    if not selected and fallback_items and category_key in ("food", "niche_hobbies", "positive_news", "entertainment"):
        selected = fallback_items[:3]

    logger.info("Fetched %d items from '%s'", len(selected), name)
    return selected


def fetch_all_feeds(feeds_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all configured feeds concurrently across categories."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    categorized_items: Dict[str, List[Dict[str, Any]]] = {k: [] for k in feeds_config}

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for category_key, cat_cfg in feeds_config.items():
            freshness_hours = int(cat_cfg.get("freshness_hours", 48))
            max_items = int(cat_cfg.get("max_items_per_feed", 10))
            for feed_info in cat_cfg.get("feeds", []):
                future = executor.submit(
                    fetch_single_feed,
                    category_key,
                    feed_info,
                    freshness_hours,
                    max_items,
                    now_utc,
                )
                futures[future] = category_key

        for future in concurrent.futures.as_completed(futures):
            cat_key = futures[future]
            try:
                items = future.result()
                categorized_items[cat_key].extend(items)
            except Exception as exc:
                logger.warning("Unexpected error in feed worker for %s: %s", cat_key, exc)

    return categorized_items


# ---------------------------------------------------------------------------
# Gemini Curation Logic
# ---------------------------------------------------------------------------

def build_gemini_prompt(
    categorized_items: Dict[str, List[Dict[str, Any]]],
    day_of_week: str,
    date_str: str,
) -> str:
    """Construct the structured editorial prompt for Gemini."""
    is_monday = day_of_week.strip().lower() == "monday"

    monday_instruction = (
        f"TODAY IS MONDAY ({date_str}): You MUST populate `monday_releases` by aggregating "
        "the Film, TV, Video Game, and Culture items from the `entertainment` feeds into a "
        "'Releasing This Week' summary block (4-6 standout releases, reviews, or previews)."
        if is_monday
        else f"TODAY IS {day_of_week.upper()} ({date_str}): Set `monday_releases` to null."
    )

    prompt = f"""You are the Chief Editor of **Daybreak**, a personal daily news dispatch for a reader based in the UK.
Today is **{day_of_week}, {date_str}**.

Below is a JSON object containing fresh RSS feed entries grouped by category (`uk_news`, `world_news`, `tech_ai`, `sports`, `entertainment`, `food`, `positive_news`, `niche_hobbies`). Each item includes `feed_rank` (its front-page priority in that publication), `source_name`, `title`, `summary`, `url`, `image_url`, `published_utc`, and `estimated_read_mins`.

### 1. EDITORIAL SELECTION & DEDUPLICATION RULES
- **Cross-Source Consensus (#1 Signal):** When multiple feeds cover the same story (e.g., BBC, The Guardian, and Sky News reporting on the same UK event, or Ars Technica and The Verge covering the same AI release), treat that overlap as the strongest indicator of importance. Deduplicate them into a **single** story, select the best primary URL, pass through its `image_url` (or an `image_url` from one of the corroborating items if the primary lacked one), and list the other outlets in `corroborating_sources`.
- **Editorial Rank & Impact:** Prioritize stories with low `feed_rank` (front-page leads) and high civic, economic, scientific, or cultural weight. Skip clickbait, minor local crime, celebrity gossip, or repetitive opinion pieces.
- **Topic Diversity:** Ensure stories within a category cover distinct topics rather than multiple angles of the exact same event.
- **URL & Image Integrity:** You MUST use the exact `url`, `image_url`, and `source_name` provided in the candidate feed items. Never invent URLs.

### 2. STRICT CATEGORY QUOTAS
Across `top_stories` and `stories` combined, enforce the following exact category quotas:
- **UK News (`uk_news`):** **3 stories**
- **World News (`world_news`):** **2 stories**
- **Technology & AI (`tech_ai`):** **3 stories**
- **Sports (`sports`):** **3 to 4 stories total**, explicitly split out using the `subcategory` field in this priority order:
  1. `Football • Arsenal FC` / `Football • Premier League` (1–2 stories): **Arsenal FC news** (via Arseblog / BBC Arsenal) and major **Premier League** news take top precedence.
  2. `Formula 1` (1 story): Biggest F1 headline of the day.
  3. `Wider Sports` (1 story): Biggest headline across wider sports (cricket, rugby, tennis, golf, boxing, athletics, etc.).
- **Entertainment & Culture (`entertainment`):** **2 stories** (covering Film/TV, video games, or longform/science).
- **Food (`food`):** **1 story** (a standout recipe, culinary technique, or food science piece).
- **Niche & Hobbies (`niche_hobbies`):** **0 to 2 stories** — Include **ONLY** if the news is genuinely significant (e.g., a new Jellyfin or major self-hosted release, a major Hearthstone or Marvel Snap patch/OTA balance update, or a major QSR/restaurant tech innovation). If nothing significant occurred in `niche_hobbies`, include 0 stories from it.
- **The Daily Uplifter (`daily_uplifter`):** **1 mandatory story** placed in the dedicated `daily_uplifter` top-level object (sourced from `positive_news` or an uplifting science/community item).

### 3. STORY FORMATS & DEPTH
- **Top Stories (`top_stories`):** Select the **2 to 3 biggest stories overall** from the quotas above and place them in `top_stories` (do NOT duplicate them in `stories`; they count toward their category's quota). For each top story, write **1 to 2 rich narrative paragraphs** in `narrative_paragraphs`.
- **Standard Stories (`stories`):** Place all remaining quota stories in `stories`. For each, write **2 concise, high-signal bullet points** in `bullet_points`.
- **'Tell Me More' Drawer (`detailed_context`):** For **every** story (including `daily_uplifter`, `top_stories`, and `stories`), provide **3 to 4 analytical bullet points** in `detailed_context` explaining background context, key figures/stats, why it matters, or what happens next.

### 4. MONDAY SPECIAL RULE
- {monday_instruction}

### CANDIDATE RSS ITEMS BY CATEGORY:
{json.dumps(categorized_items, ensure_ascii=False, indent=2)}
"""
    return prompt


def curate_with_gemini(
    categorized_items: Dict[str, List[Dict[str, Any]]],
    day_of_week: str,
    date_str: str,
) -> DailyBriefing:
    """Call Gemini 3.8 Flash via the Interactions API with Structured Output."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set GEMINI_API_KEY or run with --fallback to generate a deterministic preview from live RSS feeds."
        )

    client = genai.Client(api_key=api_key)
    prompt = build_gemini_prompt(categorized_items, day_of_week, date_str)

    logger.info(
        "Calling Gemini Interactions API (model=%s, day=%s)...",
        MODEL_ID,
        day_of_week,
    )
    interaction = client.interactions.create(
        model=MODEL_ID,
        system_instruction=(
            "You are a precise, insightful UK news editor. Always return valid JSON "
            "matching the provided schema and strictly use real URLs and image_urls from the input feed data."
        ),
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": DailyBriefing.model_json_schema(),
        },
    )

    output_text = interaction.output_text
    if not output_text:
        raise RuntimeError("Gemini returned an empty response.")

    briefing = DailyBriefing.model_validate_json(output_text)
    logger.info(
        "Successfully curated %d top stories and %d category stories.",
        len(briefing.top_stories),
        len(briefing.stories),
    )
    return briefing


# ---------------------------------------------------------------------------
# Deterministic RSS Fallback (for local preview when GEMINI_API_KEY is unset)
# ---------------------------------------------------------------------------

def build_fallback_briefing(
    categorized_items: Dict[str, List[Dict[str, Any]]],
    day_of_week: str,
    date_str: str,
) -> DailyBriefing:
    """Build a structured DailyBriefing directly from live fetched RSS items."""
    logger.info("Building fallback briefing from live RSS items (day=%s)...", day_of_week)

    def pick_items(cat: str, count: int) -> List[Dict[str, Any]]:
        items = categorized_items.get(cat, [])
        seen_titles = set()
        picked = []
        for it in items:
            norm = re.sub(r"\W+", "", it["title"].lower())[:40]
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            picked.append(it)
            if len(picked) >= count:
                break
        return picked

    def make_bullets(summary: str, title: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if len(s.strip()) > 20]
        if len(sentences) >= 2:
            return sentences[:2]
        if len(sentences) == 1:
            return [sentences[0], f"Full coverage and updates available via {title[:50]}…"]
        return [
            f"Key developments reported regarding: {title}.",
            "Read the original source report for full details and live updates.",
        ]

    def make_context(item: Dict[str, Any], cat_label: str) -> List[str]:
        bullets = [
            f"Source publication: {item['source_name']} ({cat_label}).",
            f"Published timestamp: {item['published_utc']} (Front-page feed rank #{item['feed_rank']}).",
        ]
        if item.get("summary"):
            bullets.append(f"Summary detail: {item['summary'][:220]}")
        bullets.append(f"Estimated full article reading time: ~{item.get('estimated_read_mins', 3)} minutes.")
        return bullets

    pos_items = pick_items("positive_news", 1)
    if pos_items:
        u = pos_items[0]
        uplifter = UplifterStory(
            title=u["title"],
            summary=u["summary"] or f"An uplifting spotlight from {u['source_name']}: {u['title']}",
            detailed_context=make_context(u, "Positive News"),
            source_name=u["source_name"],
            source_url=u["url"],
            image_url=u.get("image_url"),
            reading_time_minutes=u.get("estimated_read_mins", 3),
        )
    else:
        uplifter = UplifterStory(
            title="Community Conservation Efforts Drive Record Local Wildlife Recovery",
            summary="Volunteers and conservation groups across the UK have recorded a banner season for habitat restoration and local biodiversity.",
            detailed_context=[
                "Local conservation partnerships saw increased volunteer participation this quarter.",
                "Wetland and woodland corridors showed measurable species recovery.",
                "New community grants are expanding urban green spaces nationwide.",
            ],
            source_name="Positive News UK",
            source_url="https://www.positive.news/",
            image_url=CATEGORY_FALLBACK_IMAGES["positive_news"],
            reading_time_minutes=3,
        )

    uk_picks = pick_items("uk_news", 3)
    world_picks = pick_items("world_news", 2)
    tech_picks = pick_items("tech_ai", 3)
    ent_picks = pick_items("entertainment", 2)
    food_picks = pick_items("food", 1)
    niche_picks = pick_items("niche_hobbies", 1)

    all_sports = categorized_items.get("sports", [])
    football_items = [
        i for i in all_sports
        if "arseblog" in i["source_name"].lower()
        or "arsenal" in i["source_name"].lower()
        or "premier league" in i["source_name"].lower()
    ]
    f1_items = [
        i for i in all_sports
        if "formula 1" in i["source_name"].lower() or "autosport" in i["source_name"].lower()
    ]
    wider_items = [
        i for i in all_sports
        if "wider sports" in i["source_name"].lower()
        and "arsenal" not in i["title"].lower()
        and "premier league" not in i["title"].lower()
        and "f1" not in i["title"].lower()
    ]

    top_stories: List[TopStory] = []
    stories: List[CuratedStory] = []

    promoted = [
        (uk_picks[0], "uk_news", "UK News", None) if uk_picks else None,
        (world_picks[0], "world_news", "World News", None) if world_picks else None,
        (tech_picks[0], "tech_ai", "Tech & AI", None) if tech_picks else None,
    ]
    for entry in promoted:
        if not entry:
            continue
        item, cat_key, cat_label, subcat = entry
        p1 = item["summary"] if item["summary"] else f"{item['source_name']} leads with coverage of {item['title']}."
        p2 = (
            f"This developing {cat_label} lead was ranked #{item['feed_rank']} on {item['source_name']} "
            f"({item['published_utc']}). Tap through to the source article for full reporting and analysis."
        )
        top_stories.append(
            TopStory(
                title=item["title"],
                category=cat_key,  # type: ignore[arg-type]
                category_label=cat_label,
                subcategory=subcat,
                narrative_paragraphs=[p1, p2],
                detailed_context=make_context(item, cat_label),
                source_name=item["source_name"],
                source_url=item["url"],
                image_url=item.get("image_url"),
                corroborating_sources=[],
                reading_time_minutes=item.get("estimated_read_mins", 4),
            )
        )

    def add_curated(
        items_list: List[Dict[str, Any]],
        cat_key: CategoryKey,
        cat_label: str,
        subcat: Optional[str] = None,
    ) -> None:
        for it in items_list:
            stories.append(
                CuratedStory(
                    title=it["title"],
                    category=cat_key,
                    category_label=cat_label,
                    subcategory=subcat,
                    bullet_points=make_bullets(it["summary"], it["title"]),
                    detailed_context=make_context(it, f"{cat_label}" + (f" • {subcat}" if subcat else "")),
                    source_name=it["source_name"],
                    source_url=it["url"],
                    image_url=it.get("image_url"),
                    corroborating_sources=[],
                    reading_time_minutes=it.get("estimated_read_mins", 3),
                )
            )

    add_curated(uk_picks[1:], "uk_news", "UK News")
    add_curated(world_picks[1:], "world_news", "World News")
    add_curated(tech_picks[1:], "tech_ai", "Tech & AI")
    add_curated(football_items[:2], "sports", "Sports", "Football")
    add_curated(f1_items[:1], "sports", "Sports", "Formula 1")
    add_curated(wider_items[:1], "sports", "Sports", "Wider Sports")
    add_curated(ent_picks, "entertainment", "Culture")
    add_curated(food_picks, "food", "Food")
    add_curated(niche_picks, "niche_hobbies", "Niche", "Tech / Hobby Radar")

    monday_releases = None
    if day_of_week.strip().lower() == "monday":
        ent_all = categorized_items.get("entertainment", [])[:5]
        rel_items = []
        for it in ent_all:
            src_lower = it["source_name"].lower()
            medium: Literal["Film", "TV", "Video Game", "Culture"] = "Film"
            if "television" in src_lower or "tv" in src_lower:
                medium = "TV"
            elif "eurogamer" in src_lower or "polygon" in src_lower:
                medium = "Video Game"
            elif "quanta" in src_lower or "aeon" in src_lower:
                medium = "Culture"
            rel_items.append(
                ReleaseItem(
                    title=it["title"],
                    medium=medium,
                    release_details=f"Featured this week via {it['source_name']}",
                    why_it_matters=it["summary"][:180] or "Highlighted release and cultural pick for the week ahead.",
                    source_name=it["source_name"],
                    source_url=it["url"],
                    image_url=it.get("image_url"),
                )
            )
        if rel_items:
            monday_releases = MondayReleasesBlock(
                headline="Releasing & Featured This Week Across Film, TV & Gaming",
                items=rel_items,
            )

    return DailyBriefing(
        edition_tagline=f"Your {day_of_week} dispatch across UK, World, Tech, Arsenal & F1, Culture, and Food.",
        daily_uplifter=uplifter,
        top_stories=top_stories,
        stories=stories,
        monday_releases=monday_releases,
    )


# ---------------------------------------------------------------------------
# HTML Rendering & Static Asset Packaging
# ---------------------------------------------------------------------------

def calculate_briefing_read_time(briefing: DailyBriefing) -> int:
    """Calculate the estimated time (in minutes) to read the curated dashboard itself."""
    words = len(briefing.edition_tagline.split())
    words += len(briefing.daily_uplifter.title.split()) + len(briefing.daily_uplifter.summary.split())
    for b in briefing.daily_uplifter.detailed_context:
        words += len(b.split())

    for ts in briefing.top_stories:
        words += len(ts.title.split())
        for p in ts.narrative_paragraphs:
            words += len(p.split())
        for b in ts.detailed_context:
            words += len(b.split())

    for st in briefing.stories:
        words += len(st.title.split())
        for b in st.bullet_points:
            words += len(b.split())
        for b in st.detailed_context:
            words += len(b.split())

    if briefing.monday_releases:
        for item in briefing.monday_releases.items:
            words += len(item.title.split()) + len(item.why_it_matters.split())

    return max(3, math.ceil(words / 220))


def render_dashboard(
    briefing: DailyBriefing,
    day_of_week: str,
    date_display: str,
    generated_at_uk: str,
    feed_stats: Dict[str, int],
) -> str:
    """Render template.html with the curated briefing data."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("template.html")

    category_meta = [
        {"key": "all", "label": "All", "count": len(briefing.top_stories) + len(briefing.stories)},
        {"key": "uk_news", "label": "UK", "count": 0},
        {"key": "world_news", "label": "World", "count": 0},
        {"key": "tech_ai", "label": "Tech", "count": 0},
        {"key": "sports", "label": "Sports", "count": 0},
        {"key": "entertainment", "label": "Culture", "count": 0},
        {"key": "food", "label": "Food", "count": 0},
        {"key": "niche_hobbies", "label": "Niche", "count": 0},
    ]
    counts_by_key = {c["key"]: 0 for c in category_meta}
    for ts in briefing.top_stories:
        counts_by_key[ts.category] = counts_by_key.get(ts.category, 0) + 1
    for st in briefing.stories:
        counts_by_key[st.category] = counts_by_key.get(st.category, 0) + 1

    for c in category_meta:
        if c["key"] != "all":
            c["count"] = counts_by_key.get(c["key"], 0)

    briefing_read_mins = calculate_briefing_read_time(briefing)
    total_source_mins = (
        briefing.daily_uplifter.reading_time_minutes
        + sum(s.reading_time_minutes for s in briefing.top_stories)
        + sum(s.reading_time_minutes for s in briefing.stories)
    )

    return template.render(
        briefing=briefing,
        day_of_week=day_of_week,
        is_monday=(day_of_week.strip().lower() == "monday"),
        date_display=date_display,
        generated_at_uk=generated_at_uk,
        categories=category_meta,
        briefing_read_mins=briefing_read_mins,
        total_source_mins=total_source_mins,
        feed_stats=feed_stats,
        fallback_images=CATEGORY_FALLBACK_IMAGES,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate daily RSS news briefing with Gemini.")
    parser.add_argument(
        "--day",
        type=str,
        default=os.environ.get("FORCE_DAY", ""),
        help="Override day of the week (e.g. 'Monday') to test Monday release rules.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Allow building a deterministic preview from live RSS feeds if GEMINI_API_KEY is missing.",
    )
    args = parser.parse_args()

    if not FEEDS_PATH.exists():
        logger.error("feeds.json not found at %s", FEEDS_PATH)
        return 1

    with FEEDS_PATH.open("r", encoding="utf-8") as f:
        feeds_config = json.load(f)

    try:
        uk_tz = ZoneInfo("Europe/London")
        now_uk = datetime.datetime.now(uk_tz)
    except Exception:
        now_uk = datetime.datetime.now(datetime.timezone.utc)

    day_of_week = args.day.strip().capitalize() if args.day.strip() else now_uk.strftime("%A")
    date_display = now_uk.strftime("%d %B %Y")
    generated_at_uk = now_uk.strftime("%H:%M %Z")

    # 1. Fetch all RSS feeds
    categorized_items = fetch_all_feeds(feeds_config)
    feed_stats = {k: len(v) for k, v in categorized_items.items()}
    total_fetched = sum(feed_stats.values())
    logger.info("Total candidate items fetched across all categories: %d", total_fetched)

    if total_fetched == 0:
        logger.error("No RSS items could be fetched from any feed. Check network connectivity.")
        return 1

    # 2. Curate via Gemini (or fallback if requested)
    try:
        briefing = curate_with_gemini(categorized_items, day_of_week, date_display)
    except Exception as exc:
        if args.fallback:
            logger.warning("Gemini curation unavailable (%s); using --fallback mode.", exc)
            briefing = build_fallback_briefing(categorized_items, day_of_week, date_display)
        else:
            logger.error("Gemini curation failed: %s", exc)
            return 1

    # 3. Ensure every curated story has a valid article image (RSS -> og:image -> fallback)
    ensure_briefing_images(briefing, categorized_items)

    # 4. Render HTML output and copy static assets (styles.css, app.js) to dist/
    html_output = render_dashboard(
        briefing=briefing,
        day_of_week=day_of_week,
        date_display=date_display,
        generated_at_uk=generated_at_uk,
        feed_stats=feed_stats,
    )

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    dist_index = DIST_DIR / "index.html"
    dist_json = DIST_DIR / "briefing.json"
    root_index = BASE_DIR / "index.html"

    dist_index.write_text(html_output, encoding="utf-8")
    root_index.write_text(html_output, encoding="utf-8")
    dist_json.write_text(briefing.model_dump_json(indent=2), encoding="utf-8")

    for asset_name in ("styles.css", "app.js", "manifest.json", "icon.svg"):
        src_asset = BASE_DIR / asset_name
        if src_asset.exists():
            shutil.copy2(src_asset, DIST_DIR / asset_name)

    logger.info("Dashboard and static assets written to %s and %s", DIST_DIR, root_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
