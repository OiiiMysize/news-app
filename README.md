# Daybreak — Personal Daily Dispatch

An automated, serverless mobile-first news dashboard that scrapes curated RSS feeds, filters and synthesizes them using **Google Gemini (`gemini-3.8-flash`)** via the **Interactions API**, and publishes a static HTML dashboard to **GitHub Pages** every morning at **06:30 UTC**.

---

## Architecture & Files

- **[`feeds.json`](feeds.json)** — Categorized RSS/Atom feed configuration with per-category freshness windows and item caps:
  - **UK News:** BBC UK, The Guardian UK, Sky News UK
  - **World News:** BBC World, The Guardian World, Reuters World, Al Jazeera
  - **Technology & AI:** Ars Technica, Hacker News (`100+` points frontpage), The Verge
  - **Sports:** Split across **Football** (Arseblog, BBC Arsenal, BBC Premier League), **Formula 1** (BBC F1, Autosport), and **Wider Sports** (BBC Sport Top Stories, The Guardian Sport)
  - **Entertainment & Culture:** Empire, Guardian Film & TV, Eurogamer, Polygon, Quanta Magazine, Aeon
  - **Food & Culinary:** Serious Eats, The Guardian Food, BBC Food
  - **Positive News:** Good News Network, Positive News UK, BBC Uplifting Stories
  - **Niche & Hobbies:** Jellyfin GitHub Releases & Blog, LinuxServer.io, Hearthstone & Marvel Snap patch notes, QSR Magazine, Restaurant Dive
- **[`curator.py`](curator.py)** — Concurrent RSS fetcher, article image extractor (`media:content`, `media:thumbnail`, enclosures, OpenGraph `og:image`, and category fallbacks), Gemini `gemini-3.8-flash` structured-output curator, and Jinja2 static HTML builder.
- **[`template.html`](template.html)** — Mobile-first Jinja2 HTML template with 16:9 article cover images, bookmark buttons, *The Daily Uplifter* hero card, *Tell Me More* collapsible context drawers, Monday's *Releasing This Week* Entertainment panel, and a floating **Morphing Bottom Navigation Bar** with icons and labels.
- **[`styles.css`](styles.css)** — Standalone stylesheet implementing `light-dark()` theme tokens, spring-physics morphing bottom nav (`--grad-iris-cyan`, `--ease-spring`), card media framing, and intrinsic `<details>` drawer animations.
- **[`app.js`](app.js)** — Standalone client script handling the morphing bottom nav blob physics, category filtering, `localStorage`-backed **Saved for Later** bookmarking across daily rebuilds, and system-aware dark/light mode toggling.
- **[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)** — GitHub Actions workflow scheduled for `06:30 UTC` daily (plus manual `workflow_dispatch` support).

---

## Daily Quotas & Editorial Rules

1. **Cross-Source Consensus & Deduplication:** Stories reported across multiple feeds are merged into a single entry with corroborating sources noted.
2. **Category Quotas:**
   - **UK News:** 3 stories
   - **World News:** 2 stories
   - **Technology & AI:** 3 stories
   - **Sports:** 3–4 stories split into **Football** (Arsenal FC & Premier League priority), **Formula 1**, and **Wider Sports**
   - **Entertainment & Culture:** 2 stories (+ **Monday Special** *"Releasing This Week"* Film/TV/Gaming panel)
   - **Food:** 1 recipe or culinary feature
   - **The Daily Uplifter:** 1 mandatory uplifting story at the top of the dashboard
   - **Niche & Hobbies:** 0–2 stories (included *only* when a major release, patch, or industry announcement drops)
3. **Story Depth:**
   - **Lead Stories (Top 2–3):** 1–2 paragraph narrative synthesis + 3–4 *Tell Me More* analytical bullets.
   - **Category Stories:** 2 concise summary bullets + 3–4 *Tell Me More* analytical bullets.

---

## Local Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with Gemini API key:**
   ```bash
   export GEMINI_API_KEY="your-api-key"
   python curator.py
   ```

3. **Test Monday's *"Releasing This Week"* mode on any day:**
   ```bash
   python curator.py --day Monday
   ```

4. **Preview locally without an API key (live RSS deterministic fallback):**
   ```bash
   python curator.py --fallback --day Monday
   ```
   This generates `index.html` and `dist/index.html`.

---

## GitHub Pages Deployment Setup

1. Push this repository to GitHub.
2. Go to **Settings → Secrets and variables → Actions** and add a repository secret named:
   - `GEMINI_API_KEY`: Your Google AI Studio API key.
3. Go to **Settings → Pages** and under **Build and deployment → Source**, select **GitHub Actions**.
4. Trigger a manual test run anytime under **Actions → Curate & Deploy Daily News Briefing → Run workflow**.

