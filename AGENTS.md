# StockPicker_AS - project notes for AI assistants

Read this first. It captures decisions made over several chats so a new chat does not start from zero.

## What this is
A Streamlit stock dashboard (`app.py`) for **long-term buy-point decisions**, plus a daily scanner
(`scan_universe.py`) that ranks stocks and sends alerts. Owner is a retail investor, not a developer:
explain things in plain language and say what to upload where.

## Deployment (important)
- Hosted on **Streamlit Community Cloud** from a GitHub repo. The assistant cannot push to GitHub.
  The user **uploads changed files through the GitHub web UI**; Streamlit then redeploys.
- `requirements.txt` is unpinned: streamlit, yfinance, pandas, ta, matplotlib, python-docx, requests.
- Files the user must upload after changes: `app.py`, `Favorites.csv`, `scan_universe.py`,
  `.github/workflows/daily_scan.yml` (create via GitHub "Add file > Create new file", typing the path with slashes), `AGENTS.md` (optional).
- The repo is assumed public. **Never put holdings (shares / cost basis) in the repo.** The My portfolio tab keeps them
  in the browser session only; the user downloads/uploads a CSV on their own computer.
- Local: Python 3.12 at `C:\Users\u215035\AppData\Local\Programs\Python\Python312\python.exe`, Windows, PowerShell.
  Run the app with `streamlit run app.py`. Headless checks use `streamlit.testing.v1.AppTest`.
- `app.py` uses **CRLF** line endings. When patching with scripts use `newline="\r\n"`; verify no LF-only lines.
  Compile with `python -m py_compile app.py` after every edit.
- Not a git repo locally (no `.git`).

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app and the shared scoring code (importable: `main()` is guarded, `set_page_config` is inside `main`). |
| `ai_assistant.py` | AI plumbing with no Streamlit/app imports: Gemini REST transport (`gemini_transport`), model fallback, `chat_turn` tool loop, guards (daily cap, access-code lockout), system prompt + `APP_KNOWLEDGE`, `TOOL_DECLS`. |
| `scan_universe.py` | Daily scanner. Imports `app`. Writes `rankings.csv`, `scan_state.json`, and `alert.md` + `alert_title.txt` when a favorite newly enters a list. |
| `.github/workflows/daily_scan.yml` | Runs the scan weekdays 21:30 UTC, commits results, opens a GitHub issue assigned to the owner (that is the alert). Manual run has a TEST-ONLY `limit` input. |
| `Favorites.csv` | Column `Symbol`. The user's main list (56 symbols incl. 6 ETFs). Seeds the app's favorites each session and the scan's alert filter. |
| `stockList.csv` | Column `Symbol`, ~527 symbols. Wider universe for the scan. |
| `StockSeller_v2.py` / `v3` | Older Word-report sell-signal scripts. Their rules are reused in `compute_sell_signals`. |
| `StockPickerComprehensive_v6.py` | Older report script. Still has the capex-sign FCF bug and the old `'Capital Expenditures'` row name (FCF multiple shows N/A). Not fixed; not used by the app. |

## App tabs (in order)
Price charts, Buy checklist, Business snapshot, Fair value, Entry and exit plan, Analyst view, News, Test the signals (single stock) | Market today, AI assistant, Compare stocks, Top 10 lists, My portfolio, My favorites (across stocks), How it works (15 tabs in total) | How it works.
Tabs were renamed for plain language (old names: Charts, Long-term checklist, Financial metrics, Valuation, Paper investing check, Compare, Rankings, Portfolio, Favorites, Explanation).
Business snapshot (`render_business_snapshot`, `snapshot_sections`, `snapshot_notes`) replaced the flat Financial metrics list: headline cards, strengths/watch-outs,
four charts, signal-coloured detail tables, CSV download; thresholds mirror the Buy checklist. `fetch_quality_metrics` now also returns a `series` DataFrame.
Entry and exit plan (`render_plan_tab`, `build_entry_ladder`, `build_exit_ladder`): tranche entry ladder (Value or Pullback style) and an editable exit ladder
from model fair value or the user's average cost (shares/cost come from `st.session_state["pf_base"]`, browser session only), plus a CSV download.
News and Market today (`render_news_tab`, `render_market_tab`, `get_news`, `get_market_snapshot`, `market_reading`): **Google News RSS is the primary headline feed**
(tested: Yahoo `Ticker.news` failed for every ticker while Google returned 28-100 items each), Yahoo news is the backup. Search uses the cleaned company name
(`_clean_company_name`) so short tickers like T work. Titles and links only (no article text); `_md_safe` escapes dollar signs and markdown characters; fund-filing filler
and non-Latin headlines are filtered. Market today is rule-based text from index, VIX, 10Y yield, oil, gold, sector-ETF and favorites moves; it says what moved, not why.
Big-move banner: S&P <= -2%, Nasdaq <= -3%, VIX >= 30 or +15% in a day. No API keys are used.
The top bar is ONE selectbox (key `ticker_box`, `accept_new_options=True`): pick or type a symbol and it loads immediately; My favorites "Load" sets `pending_ticker`.
**AI assistant tab** (`render_ai_tab`, `_build_ai_tools`, `render_ai_chart` in app.py; plumbing in `ai_assistant.py`). Gemini free tier via plain HTTPS (no SDK), models tried in order
`gemini-3.5-flash-lite`, `3.1-flash-lite`, `2.5-flash-lite` (free quotas are per model; Flash models allow only ~20/day, Flash-Lite ~500). The model never runs code: it can only call the
11 whitelisted tools (stock summary via `score_symbol`, compare, price stats, news, market today, entry/exit ladders, favorites scan, portfolio (opt-in only), show_chart, open_stock).
Secrets: `GEMINI_API_KEY` and `AI_ACCESS_CODE` in Streamlit Cloud Settings > Secrets (or env vars locally); the tab FAILS CLOSED without the access code because the app is public.
Limits: 30 messages per session, 400 model calls per day app-wide, 10 wrong codes per 15 minutes locks everyone. Gemini 3 needs `thoughtSignature` parts echoed back unchanged and
the `functionCall.id` echoed in `functionResponse` (tool exchanges are dropped from stored history after each turn). Always escape `$` in model text before `st.markdown`.
Not yet verified against the live Gemini service (no key was available while building); the tab has a "Check the AI connection" button for that. Portfolio is only sent when the user ticks the box.
The How it works tab documents every metric and must be kept in sync with the code whenever a threshold or rule changes.

## Scoring logic (keep consistent everywhere)
- **Technicals (4h bars from 1h data):** Bollinger 200/2, RSI 10, MACD 8/21/5. Reversal confidence = 4 checks x 25 points.
- **Long-term checklist** = Quality (9 checks), Valuation (6), Timing (5); each Pass/Fail/N-A, N/A ignored in scoring.
  Verdict: Quality >= 70 and Valuation >= 50 -> candidate (timing only sizes the entry; it does not veto).
  Banks/insurers (Sector = Financial Services): ROIC, leverage, interest coverage, FCF, DCF checks are N/A; DCF shows "not meaningful".
  ETFs: no statements, so only Timing applies.
- **Valuation defaults per stock** (`suggest_valuation_defaults(fin, quality, valctx, profile)`): a stock type is auto-detected by `classify_stock_type`
  (order: Financial, Cyclical, Growth, Core) and can be overridden in the Fair value tab (`val_profile_choice`, reset to Auto when the ticker changes).
  Types change only g, K and MoS: **Core** g = median rev/EPS/FCF CAGR clamped 3-10%, K = own median P/E clamped 12-22;
  **Growth** (revenue CAGR >= 12%) two-stage fade: g1 = median clamped 10-25% for 5 years, fading linearly to 5% by year 10, converted to one flat rate with
  `fade_growth_equivalent`; K clamp 15-30; **Cyclical** (sector Energy or industry keywords in `_CYCLICAL_INDUSTRY_WORDS`: airline, steel, mining, shipping, etc.)
  g 2-5%, K 6-14, MoS 35%; **Financial** g 3-8%, K 8-14. Leveraged or negative-FCF names: K capped at 15, MoS 35%. Otherwise MoS 25% (15% if Quality >= 80%).
  Do not add "heavy construction" to the cyclical words (it wrongly caught CAT). Cyclicals get stricter on purpose (UAL fair ~46 vs price much higher).
  **r = S&P 500 trailing 3Y annualized + 2%, capped at 15%** (user's choice, deliberately unchanged), same for all stocks. Net debt = Yahoo's "Net Debt" row.
- **Rankings** (`score_symbol`, `rank_results`): rankable = EQUITY with >= 4 scored Quality and >= 3 scored Valuation checks.
  Buy list: Quality >= 50, ordered by 0.45 Quality + 0.40 Valuation + 0.15 Timing. Avoid list: Valuation <= 33 or Quality < 50,
  ordered by 0.5 (100 - Valuation) + 0.3 (100 - Quality) + 0.2 overbought. Top 10 each; buy names are removed from avoid.
- **Profit-taking** (`compute_sell_signals`, `trim_signal`): 3 overbought signals on daily bars (close > BB(15,2) upper, RSI(10) > 70,
  MACD(8/21/5) below signal). Trim = 3/3 and price above fair value in all models; Review = Quality < 50; Watch = 2+ signals,
  or 1 signal and above fair value; else Hold. My portfolio tab adds gain %, weight (> 15% note) and sector concentration (> 35% warning).
- **Buy zone** (`score_symbol` fields `buy_gap_pct`, `buy_zone`, `stock_type`): Quality >= 50 and price vs the EPS-model 10Y buy price;
  gap% = (price / buy - 1) x 100; "In zone" <= 0, "Near" <= 5. Shown as "Buy-price watch" on the Top 10 lists tab.
- **Alerts:** only for **favorites** that **newly** enter the top-10 buy list, the top-10 avoid list, the Trim level, or the buy zone compared with the
  previous `scan_state.json` (keys include `buy_zone`, `new_buy_zone`). The first run (or a missing key) has no previous state, so everything counts as new (one baseline alert).

## Lessons learned (do not repeat)
- **Paper investing rule must not require all three signals on one bar.** BB-low + RSI<30 + an exact MACD cross on the same bar
  gave zero trades for 26 of 55 favorites over 2 years (daily bars: zero for all 56). The simulator now has a "signal window"
  (default 5 bars: oversold within the window, MACD cross now; 1 = old strict rule) and a Daily bar option (up to 10 years,
  BB 20/RSI 10/MACD 8/21/5) because Yahoo hourly data covers only ~2 years. Hourly downloads use explicit start/end dates, not
  `period=`, which fails for recently listed stocks such as RDDT.
- **`yf.download` is not thread-safe** (shared global state): parallel threads return other tickers' prices. The scanner uses
  `ProcessPoolExecutor`. Never score symbols with threads. The app's live scans are sequential.
- Yahoo rate-limits shared cloud IPs. Failed lookups must raise (so `st.cache_data` does not cache them). Pattern in `app.py`: `_fetch_*_cached` functions
  (financial metrics, quality metrics, valuation context) RAISE on Yahoo failure; public wrappers go through `_guarded` with a 2-minute `_NEG_CACHE`; the
  "Try again" button calls `_clear_data_caches()`; `render_data_gap_banner` shows the last saved scan row. `score_symbol` raises when a non-ETF has no statements.
  An empty result IS cached by `st.cache_data`, so never return `{}` from a cached function on failure. The scan aborts without
  writing results if fewer than 50% of symbols score, so a bad day never overwrites good data or creates false alerts.
- Capex is negative in Yahoo cash-flow data; FCF = Yahoo "Free Cash Flow" else CFO - abs(capex).
- Yahoo gives only ~4 fiscal years of statements, so growth rates cover about 3 years.
- With r capped at 15% and g/K capped (10% / 22), most large caps show "above fair value", so Valuation scores compress and
  fast growers always looked expensive. The Growth profile (fade model) now eases this (NVDA fair ~70 -> ~209). Rankings are relative, not absolute.
- Streamlit session state is lost on refresh, so anything that must persist lives in repo files (`Favorites.csv`, `rankings.csv`).
- Avoid `$` in `st.markdown` text (it triggers math rendering). Use words like "dollars".
- `use_container_width` is deprecated in newer Streamlit but still works; switch to `width="stretch"` if a future release removes it.

## Open items / ideas (not built yet)
1. Done: AI chat assistant (see above). Done: stock-type valuation profiles, buy-price alerts, Yahoo resilience, Entry and exit plan tab. Could still add: a small risk adjustment to r by type (user declined so far).
2. Favorites persistence beyond `Favorites.csv` (URL parameter, browser local storage, or a Google Sheet) was discussed and parked.
3. Done: news section and Market today panel (free sources). Could add: Finnhub (free key) as a steadier news source, an AI summary of the headlines
   ("why is the market moving"), headlines inside the daily alert email. Still open: earnings-date / ex-dividend warnings, score history over time, peer comparison,
   AI ideas not built: summaries in the daily alert email, the AI choosing among user-saved screens.
4. Longer fundamental history (SEC filings or a paid feed) would improve Quality accuracy the most.
5. Fix the FCF capex bug in `StockPickerComprehensive_v6.py` if that script is still used.

## Working agreements with the user
- Say clearly which files changed and which ones to upload. Test before reporting (compile + AppTest, and a real scan on a few symbols).
- Not investment advice: keep that caveat in the app text.
