/**
 * Daybreak — Mobile-First Personal News Dispatch
 * Client-Side Controller (app.js)
 *
 * Handles:
 * 1. Morphing Bottom Navigation Bar (c09-bnav spring blob & category filtering)
 * 2. Save-for-Later Article Persistence (localStorage across daily builds)
 * 3. Expand/Collapse All 'Tell Me More' Drawers
 * 4. System-Aware Two-State Theme Toggle
 */

(function () {
  const STORAGE_KEY_SAVED = 'daybreak_saved_articles_v1';

  const nav = document.getElementById('c09-bnav');
  const blob = document.getElementById('c09-blob');
  const navBtns = nav ? Array.from(nav.querySelectorAll('.c09-bnav__btn')) : [];

  const uplifterSection = document.getElementById('section-uplifter');
  const mondaySection = document.getElementById('section-monday-releases');
  const topSection = document.getElementById('section-top-stories');
  const curatedSection = document.getElementById('section-curated-stories');
  const savedSection = document.getElementById('section-saved');
  const savedGrid = document.getElementById('saved-cards-grid');
  const savedEmptyState = document.getElementById('saved-empty-state');
  const savedCountBadge = document.getElementById('saved-count-badge');
  const clearSavedBtn = document.getElementById('clear-saved-btn');
  const visibleCountLabel = document.getElementById('visible-count-label');

  let currentFilter = 'all';

  // -------------------------------------------------------------------------
  // 1. Saved Articles State (localStorage)
  // -------------------------------------------------------------------------

  function getSavedArticles() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_SAVED);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  function setSavedArticles(list) {
    try {
      localStorage.setItem(STORAGE_KEY_SAVED, JSON.stringify(list));
    } catch (e) {}
    updateSavedUIState();
  }

  function isArticleSaved(url) {
    if (!url) return false;
    return getSavedArticles().some((item) => item.source_url === url);
  }

  function toggleSaveArticle(storyData) {
    if (!storyData || !storyData.source_url) return false;
    const list = getSavedArticles();
    const existingIdx = list.findIndex((item) => item.source_url === storyData.source_url);

    let nowSaved = false;
    if (existingIdx >= 0) {
      list.splice(existingIdx, 1);
      nowSaved = false;
    } else {
      list.unshift({
        ...storyData,
        saved_at: new Date().toLocaleDateString('en-GB', {
          day: 'numeric',
          month: 'short',
        }),
      });
      nowSaved = true;
    }

    setSavedArticles(list);
    if (currentFilter === 'saved') {
      renderSavedSection();
    }
    return nowSaved;
  }

  function extractStoryDataFromCard(cardEl) {
    const btn = cardEl.querySelector('.save-btn');
    if (btn && btn.dataset.story) {
      try {
        return JSON.parse(btn.dataset.story);
      } catch (e) {}
    }
    return null;
  }

  function updateSavedUIState() {
    const savedList = getSavedArticles();
    const savedUrls = new Set(savedList.map((s) => s.source_url));

    // Update badge in bottom morphing nav
    if (savedCountBadge) {
      savedCountBadge.textContent = String(savedList.length);
      savedCountBadge.hidden = savedList.length === 0;
    }

    if (clearSavedBtn) {
      clearSavedBtn.hidden = savedList.length === 0;
    }

    // Sync all bookmark buttons in the DOM
    document.querySelectorAll('.save-btn').forEach((btn) => {
      const url = btn.getAttribute('data-url') || '';
      const saved = savedUrls.has(url);
      btn.setAttribute('aria-pressed', saved ? 'true' : 'false');
      btn.setAttribute(
        'aria-label',
        saved ? 'Remove from saved articles' : 'Save article for later'
      );
      btn.setAttribute('title', saved ? 'Saved for later (tap to remove)' : 'Save for later');
    });

    // Re-align blob in case the Saved badge changed button width
    const activeBtn = nav ? nav.querySelector('.c09-bnav__btn--active') : null;
    if (activeBtn) {
      moveBlob(activeBtn, false);
    }
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  function renderSavedSection() {
    if (!savedGrid || !savedEmptyState) return;
    const savedList = getSavedArticles();

    savedGrid.innerHTML = '';

    if (savedList.length === 0) {
      savedEmptyState.hidden = false;
      savedGrid.hidden = true;
      return;
    }

    savedEmptyState.hidden = true;
    savedGrid.hidden = false;

    savedList.forEach((story) => {
      const article = document.createElement('article');
      article.className =
        'story-card surface-card rounded-2xl overflow-hidden flex flex-col justify-between shadow-sm transition';

      const subcatHtml = story.subcategory
        ? `<span class="text-[11px] font-mono font-semibold px-2.5 py-0.5 rounded-full elevated-surface" style="color: var(--accent-iris);">${escapeHtml(story.subcategory)}</span>`
        : '';

      const savedDateHtml = story.saved_at
        ? `<span class="text-[11px] font-mono opacity-75">Saved ${escapeHtml(story.saved_at)}</span>`
        : '';

      const bodyItems = Array.isArray(story.summary_list) ? story.summary_list : [];
      const bodyHtml = story.is_narrative
        ? `<div class="space-y-2.5 text-sm leading-relaxed mb-4" style="color: var(--text-secondary);">
            ${bodyItems.map((p) => `<p>${escapeHtml(p)}</p>`).join('')}
           </div>`
        : `<ul class="space-y-2 text-sm leading-relaxed mb-4 pl-4 list-disc" style="color: var(--text-secondary);">
            ${bodyItems.map((b) => `<li>${escapeHtml(b)}</li>`).join('')}
           </ul>`;

      const contextItems = Array.isArray(story.detailed_context) ? story.detailed_context : [];
      const drawerHtml =
        contextItems.length > 0
          ? `<details class="tell-me-more">
              <summary class="inline-flex items-center gap-1.5 text-xs font-mono font-semibold uppercase tracking-wider py-1 hover:opacity-80" style="color: var(--text-primary);">
                <span>Tell Me More (${contextItems.length} points)</span>
                <svg class="drawer-chevron w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </summary>
              <div class="drawer-body mt-2.5 p-3.5 rounded-xl elevated-surface">
                <ul class="space-y-2 text-xs leading-relaxed pl-4 list-disc" style="color: var(--text-secondary);">
                  ${contextItems.map((c) => `<li>${escapeHtml(c)}</li>`).join('')}
                </ul>
              </div>
            </details>`
          : '';

      article.innerHTML = `
        <div>
          <div class="card-media-wrap">
            <img
              src="${escapeHtml(story.image_url || 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=900&q=80')}"
              alt="${escapeHtml(story.title)}"
              loading="lazy"
              width="800"
              height="450"
            />
            <div class="absolute inset-x-0 top-0 p-3 flex items-center justify-between gap-2 bg-gradient-to-b from-black/60 via-black/20 to-transparent">
              <span class="text-[11px] font-mono font-semibold uppercase tracking-wider px-2.5 py-1 rounded-full bg-black/65 text-white backdrop-blur-md">
                ${escapeHtml(story.category_label || 'Saved')}
              </span>
              <button
                type="button"
                class="save-btn"
                data-url="${escapeHtml(story.source_url)}"
                aria-pressed="true"
                aria-label="Remove from saved articles"
                title="Remove from saved"
              >
                <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
              </button>
            </div>
          </div>

          <div class="p-4 sm:p-5">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div class="flex items-center gap-1.5 flex-wrap">
                ${subcatHtml}
                ${savedDateHtml}
              </div>
              <span class="text-xs font-mono" style="color: var(--text-secondary);">
                ~${Number(story.reading_time_minutes || 4)} min read
              </span>
            </div>

            <h3 class="font-serif text-lg sm:text-xl font-bold leading-snug mb-2.5">
              <a href="${escapeHtml(story.source_url)}" target="_blank" rel="noopener noreferrer" class="hover:underline">
                ${escapeHtml(story.title)}
              </a>
            </h3>

            ${bodyHtml}
          </div>
        </div>

        <div class="px-4 sm:px-5 pb-4 sm:pb-5 space-y-3 pt-3 border-t" style="border-color: var(--border-soft);">
          ${drawerHtml}
          <div class="flex items-center justify-between gap-2 pt-1 text-xs font-mono">
            <a
              href="${escapeHtml(story.source_url)}"
              target="_blank"
              rel="noopener noreferrer"
              class="inline-flex items-center gap-1 font-semibold hover:underline"
              style="color: var(--text-primary);"
            >
              <span>Read on ${escapeHtml(story.source_name || 'Source')}</span>
              <span aria-hidden="true">↗</span>
            </a>
            <button
              type="button"
              class="remove-saved-link text-xs font-mono underline opacity-75 hover:opacity-100"
              style="color: var(--accent-iris);"
              data-url="${escapeHtml(story.source_url)}"
            >
              Remove
            </button>
          </div>
        </div>
      `;

      const removeBtn = article.querySelector('.save-btn');
      const removeTextBtn = article.querySelector('.remove-saved-link');
      const handleRemove = () => {
        toggleSaveArticle(story);
      };
      if (removeBtn) removeBtn.addEventListener('click', handleRemove);
      if (removeTextBtn) removeTextBtn.addEventListener('click', handleRemove);

      savedGrid.appendChild(article);
    });
  }

  // Delegate clicks on .save-btn in the main daily feed
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.save-btn');
    if (!btn) return;
    // Ignore if inside #saved-cards-grid (already handled directly)
    if (savedGrid && savedGrid.contains(btn)) return;

    const rawStory = btn.getAttribute('data-story');
    if (!rawStory) return;
    try {
      const storyData = JSON.parse(rawStory);
      toggleSaveArticle(storyData);
    } catch (err) {}
  });

  if (clearSavedBtn) {
    clearSavedBtn.addEventListener('click', () => {
      setSavedArticles([]);
      renderSavedSection();
    });
  }

  // -------------------------------------------------------------------------
  // 2. Morphing Bottom Navigation Bar & Category Filtering
  // -------------------------------------------------------------------------

  function moveBlob(targetBtn, smoothScroll = true) {
    if (!nav || !blob || !targetBtn) return;
    navBtns.forEach((b) => {
      const active = b === targetBtn;
      b.classList.toggle('c09-bnav__btn--active', active);
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
    });

    const offsetLeft = targetBtn.offsetLeft - 6;
    const width = targetBtn.offsetWidth;
    blob.style.width = `${width}px`;
    blob.style.transform = `translateX(${offsetLeft}px)`;

    if (smoothScroll && typeof targetBtn.scrollIntoView === 'function') {
      targetBtn.scrollIntoView({
        behavior: 'smooth',
        inline: 'center',
        block: 'nearest',
      });
    }
  }

  function applyCategoryFilter(selectedCategory) {
    currentFilter = selectedCategory;
    const storyCards = document.querySelectorAll(
      '#section-top-stories .story-card, #section-curated-stories .story-card'
    );

    if (selectedCategory === 'saved') {
      if (uplifterSection) uplifterSection.hidden = true;
      if (mondaySection) mondaySection.hidden = true;
      if (topSection) topSection.hidden = true;
      if (curatedSection) curatedSection.hidden = true;
      if (savedSection) {
        savedSection.hidden = false;
        renderSavedSection();
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }

    if (savedSection) savedSection.hidden = true;
    if (curatedSection) curatedSection.hidden = false;

    // Show Daily Uplifter on "All", hide when drilling into a specific topic so mobile screen focuses on topic
    if (uplifterSection) {
      uplifterSection.hidden = selectedCategory !== 'all';
    }

    let visibleTopCount = 0;
    let visibleTotalCount = 0;

    storyCards.forEach((card) => {
      const cardCat = card.getAttribute('data-category');
      const shouldShow = selectedCategory === 'all' || cardCat === selectedCategory;
      card.hidden = !shouldShow;
      if (shouldShow) {
        visibleTotalCount += 1;
        if (topSection && topSection.contains(card)) {
          visibleTopCount += 1;
        }
      }
    });

    if (topSection) {
      topSection.hidden = visibleTopCount === 0;
    }

    if (mondaySection) {
      mondaySection.hidden = !(
        selectedCategory === 'all' || selectedCategory === 'entertainment'
      );
    }

    if (visibleCountLabel) {
      const activeBtn = nav
        ? nav.querySelector(`.c09-bnav__btn[data-filter="${selectedCategory}"] .c09-bnav__label`)
        : null;
      const labelText = activeBtn ? activeBtn.textContent : 'All';
      visibleCountLabel.textContent =
        selectedCategory === 'all'
          ? `All Dispatches (${visibleTotalCount})`
          : `${labelText} (${visibleTotalCount})`;
    }
  }

  navBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      moveBlob(btn, true);
      const filterKey = btn.getAttribute('data-filter') || 'all';
      applyCategoryFilter(filterKey);
    });
  });

  window.addEventListener('resize', () => {
    const activeBtn = nav ? nav.querySelector('.c09-bnav__btn--active') : null;
    if (activeBtn) moveBlob(activeBtn, false);
  });

  // Initialize blob position & saved state on first frame
  requestAnimationFrame(() => {
    updateSavedUIState();
    if (navBtns.length > 0) {
      moveBlob(navBtns[0], false);
    }
  });

  // -------------------------------------------------------------------------
  // 3. Expand / Collapse All 'Tell Me More' Drawers
  // -------------------------------------------------------------------------

  const toggleAllBtn = document.getElementById('toggle-all-drawers');
  let allExpanded = false;
  if (toggleAllBtn) {
    toggleAllBtn.addEventListener('click', () => {
      allExpanded = !allExpanded;
      document.querySelectorAll('details.tell-me-more').forEach((el) => {
        el.open = allExpanded;
      });
      toggleAllBtn.textContent = allExpanded ? 'Collapse Context' : 'Expand Context';
    });
  }

  // -------------------------------------------------------------------------
  // 4. System-Aware Two-State Dark Mode Toggle
  // -------------------------------------------------------------------------

  const themeBtn = document.getElementById('theme-toggle');
  const themeLabel = document.getElementById('theme-toggle-label');
  const metaScheme = document.querySelector('meta[name="color-scheme"]');
  const darkMql = window.matchMedia('(prefers-color-scheme: dark)');

  function getRenderedScheme() {
    const currentMeta = metaScheme ? metaScheme.content : 'light dark';
    if (currentMeta === 'dark' || currentMeta === 'light') {
      return currentMeta;
    }
    return darkMql.matches ? 'dark' : 'light';
  }

  function updateThemeButtonLabel() {
    if (!themeLabel) return;
    const rendered = getRenderedScheme();
    themeLabel.textContent = rendered === 'dark' ? 'Light' : 'Dark';
  }

  if (themeBtn && metaScheme) {
    themeBtn.addEventListener('click', () => {
      const systemScheme = darkMql.matches ? 'dark' : 'light';
      const renderedScheme = getRenderedScheme();
      const targetScheme = renderedScheme === 'dark' ? 'light' : 'dark';

      try {
        if (targetScheme !== systemScheme) {
          localStorage.setItem('color-scheme', targetScheme);
          metaScheme.content = targetScheme;
        } else {
          localStorage.removeItem('color-scheme');
          metaScheme.content = 'light dark';
        }
      } catch (e) {
        metaScheme.content = targetScheme;
      }
      updateThemeButtonLabel();
    });

    darkMql.addEventListener('change', updateThemeButtonLabel);
    window.addEventListener('storage', (e) => {
      if (e.key === 'color-scheme') {
        metaScheme.content = e.newValue ?? 'light dark';
        updateThemeButtonLabel();
      } else if (e.key === STORAGE_KEY_SAVED) {
        updateSavedUIState();
        if (currentFilter === 'saved') renderSavedSection();
      }
    });
    updateThemeButtonLabel();
  }

  // -------------------------------------------------------------------------
  // 5. Graceful Fallback for Broken External Article Images
  // -------------------------------------------------------------------------
  document.addEventListener(
    'error',
    (e) => {
      const target = e.target;
      if (target && target.tagName === 'IMG' && !target.dataset.fallbackApplied) {
        target.dataset.fallbackApplied = 'true';
        target.src =
          'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=900&q=80';
      }
    },
    true
  );
})();

