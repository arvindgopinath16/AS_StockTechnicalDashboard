import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


DOWNLOAD_DELAY = 2.0
RATE_LIMIT_WAIT = 60
MAX_RETRIES = 2


def normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo OHLCV data to columns: Open, High, Low, Close, Volume."""
    expected = ["Open", "High", "Low", "Close", "Volume"]
    out = data.copy()

    if isinstance(out.columns, pd.MultiIndex):
        lv0 = [str(c) for c in out.columns.get_level_values(0)]
        lv1 = [str(c) for c in out.columns.get_level_values(1)]
        if set(expected).issubset(set(lv0)):
            out.columns = lv0
        elif set(expected).issubset(set(lv1)):
            out.columns = lv1
        else:
            # Fallback: try either side of the tuple that matches OHLCV names
            flat_cols = []
            for c0, c1 in out.columns.to_list():
                if str(c0) in expected:
                    flat_cols.append(str(c0))
                elif str(c1) in expected:
                    flat_cols.append(str(c1))
                else:
                    flat_cols.append(str(c0))
            out.columns = flat_cols

    if "Adj Close" in out.columns:
        out = out.drop(columns=["Adj Close"])

    missing = [c for c in expected if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    return out[expected]


@st.cache_data(ttl=900, show_spinner=False)
def load_default_symbols() -> list[str]:
    csv_path = Path("stockList.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Symbol" in df.columns:
            return sorted(df["Symbol"].dropna().astype(str).str.strip().unique().tolist())
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]


@st.cache_data(ttl=900, show_spinner=False)
def download_data(stock: str) -> pd.DataFrame:
    data = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = yf.download(
                stock,
                period="6mo",
                interval="1h",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "rate limit" in err_msg or "too many requests" in err_msg:
                if attempt < MAX_RETRIES:
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                return pd.DataFrame()
            raise

        if data is not None and not data.empty:
            return data

        if attempt < MAX_RETRIES:
            time.sleep(RATE_LIMIT_WAIT)

    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def download_data_for_paper(stock: str, years: int) -> pd.DataFrame:
    """Download 1h data for paper trading; Yahoo 1h allows up to ~730 days."""
    period_days = min(730, max(30, int(years * 365)))
    period = f"{period_days}d"
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = yf.download(
                stock,
                period=period,
                interval="1h",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "rate limit" in err_msg or "too many requests" in err_msg:
                if attempt < MAX_RETRIES:
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                return pd.DataFrame()
            raise
        if data is not None and not data.empty:
            return data
        if attempt < MAX_RETRIES:
            time.sleep(RATE_LIMIT_WAIT)
    return pd.DataFrame()


def calculate_indicators(raw_data: pd.DataFrame, stock: str) -> pd.DataFrame:
    data = normalize_ohlcv_columns(raw_data)

    data.index = pd.to_datetime(data.index).tz_localize(None)
    data_4h = data.resample("4h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    bb_indicator = BollingerBands(close=data_4h["Close"], window=200, window_dev=2)
    data_4h["BB_High"] = bb_indicator.bollinger_hband()
    data_4h["BB_Low"] = bb_indicator.bollinger_lband()
    data_4h["RSI"] = RSIIndicator(data_4h["Close"], window=10).rsi()

    macd_indicator = MACD(data_4h["Close"], window_slow=21, window_fast=8, window_sign=5)
    data_4h["MACD"] = macd_indicator.macd()
    data_4h["MACD_Signal"] = macd_indicator.macd_signal()

    return data_4h.dropna()


def calculate_reversal_confidence(raw_data: pd.DataFrame, stock: str) -> dict:
    """
    Confidence score for potential reversal (0-100), based on:
    OBV divergence, 30-day POC, volume bar strength, and VWAP.
    """
    data = normalize_ohlcv_columns(raw_data)

    data.index = pd.to_datetime(data.index).tz_localize(None)
    daily = data.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    scores = {"OBV Divergence": 0, "Volume Profile (POC)": 0, "Volume Bars": 0, "VWAP": 0}

    if len(daily) < 30:
        total = 0
        label = "Likely Dead Cat Bounce" if total < 40 else "Neutral"
        return {"total": total, "label": label, "scores": scores}

    # 1) OBV Divergence (bullish): OBV trending up while price flat/down
    obv_window = daily.tail(20).copy()
    close_diff = obv_window["Close"].diff().fillna(0.0)
    direction = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * obv_window["Volume"]).cumsum()
    price_trend = float(obv_window["Close"].iloc[-1] - obv_window["Close"].iloc[0])
    obv_trend = float(obv.iloc[-1] - obv.iloc[0])
    if obv_trend > 0 and price_trend <= 0:
        scores["OBV Divergence"] = 25

    # 2) Volume Profile POC (30-day): score if current close > POC
    vp = daily.tail(30).copy()
    n_bins = 20
    bin_codes, bin_edges = pd.cut(vp["Close"], bins=n_bins, labels=False, retbins=True, include_lowest=True)
    vp["bin"] = bin_codes
    volume_by_bin = vp.groupby("bin")["Volume"].sum()
    if not volume_by_bin.empty:
        poc_bin = int(volume_by_bin.idxmax())
        poc_price = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)
        if float(vp["Close"].iloc[-1]) > poc_price:
            scores["Volume Profile (POC)"] = 25

    # 3) Volume bars: today's green candle volume > 1.2x 20-day avg
    avg20_vol = float(daily["Volume"].tail(20).mean())
    today = daily.iloc[-1]
    is_green = float(today["Close"]) > float(today["Open"])
    if avg20_vol > 0 and is_green and float(today["Volume"]) > 1.2 * avg20_vol:
        scores["Volume Bars"] = 25

    # 4) VWAP: score if current price > 30-day VWAP
    typical_price = (vp["High"] + vp["Low"] + vp["Close"]) / 3.0
    vol_sum = float(vp["Volume"].sum())
    if vol_sum > 0:
        vwap_30 = float((typical_price * vp["Volume"]).sum() / vol_sum)
        if float(vp["Close"].iloc[-1]) > vwap_30:
            scores["VWAP"] = 25

    total = int(sum(scores.values()))
    if total > 75:
        label = "Strong Reversal Signal"
    elif total < 40:
        label = "Likely Dead Cat Bounce"
    else:
        label = "Neutral / Mixed"

    return {"total": total, "label": label, "scores": scores}


def run_paper_simulation(
    df_4h: pd.DataFrame,
    money_invested: float,
    pct_taken_out: float,
) -> dict:
    """
    Simulate paper trading on 4h bars.
    Buy: Close < BB_Low, RSI < 30, MACD crosses above Signal.
    Sell: Close > BB_High, RSI > 70, MACD crosses below Signal; sell pct_taken_out of position.
    """
    total_invested = 0.0
    shares = 0.0
    cost_basis = 0.0
    cash_from_sales = 0.0
    realized_gain = 0.0
    events: list = []

    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i - 1]
        close = float(row["Close"])

        # Buy: below BB low, RSI < 30, MACD just crossed above Signal
        buy_signal = (
            close < row["BB_Low"]
            and row["RSI"] < 30
            and row["MACD"] > row["MACD_Signal"]
            and prev["MACD"] <= prev["MACD_Signal"]
        )
        if buy_signal:
            sh = money_invested / close
            shares += sh
            cost_basis += money_invested
            total_invested += money_invested
            events.append((df_4h.index[i], "Buy", sh, close, money_invested))

        # Sell: above BB high, RSI > 70, MACD just crossed below Signal; sell pct_taken_out of shares
        sell_signal = (
            shares > 0
            and close > row["BB_High"]
            and row["RSI"] > 70
            and row["MACD"] < row["MACD_Signal"]
            and prev["MACD"] >= prev["MACD_Signal"]
        )
        if sell_signal:
            to_sell = shares * pct_taken_out
            if to_sell > 1e-8:
                proceeds = to_sell * close
                cost_of_sold = (to_sell / shares) * cost_basis
                realized_gain += proceeds - cost_of_sold
                cash_from_sales += proceeds
                shares -= to_sell
                cost_basis -= cost_of_sold
                events.append((df_4h.index[i], "Sell", to_sell, close, proceeds))

    latest_close = float(df_4h.iloc[-1]["Close"])
    current_value = shares * latest_close
    unrealized_gain = current_value - cost_basis
    final_wealth = cash_from_sales + current_value
    rate_of_return_pct = (
        (final_wealth - total_invested) / total_invested * 100.0 if total_invested else 0.0
    )

    return {
        "total_invested": total_invested,
        "realized_gain": realized_gain,
        "unrealized_gain": unrealized_gain,
        "current_value": current_value,
        "cash_from_sales": cash_from_sales,
        "shares": shares,
        "cost_basis": cost_basis,
        "rate_of_return_pct": rate_of_return_pct,
        "final_wealth": final_wealth,
        "n_buys": sum(1 for e in events if e[1] == "Buy"),
        "n_sells": sum(1 for e in events if e[1] == "Sell"),
        "events": events,
    }


def _retry(fn, tries: int = 2, wait: float = 1.5):
    """Call fn(), retrying after a short pause when Yahoo throws (rate limits are common on shared cloud IPs)."""
    last_exc = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i < tries - 1:
                time.sleep(wait * (i + 1))
    raise last_exc


def _has_data(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return not v.empty
    if isinstance(v, dict):
        return any(x is not None for x in v.values())
    return True


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_analyst_raw(stock: str) -> dict:
    """
    Pull every analyst-related Yahoo endpoint independently, so one failure does not blank the tab.
    Raises (and therefore is NOT cached) when nothing useful came back, so a rate-limited call never
    leaves the tab empty for the next 15 minutes.
    """
    t = yf.Ticker(stock)
    sources = {
        "info": lambda: t.info,
        "price_targets": lambda: t.analyst_price_targets,
        "ratings": lambda: t.recommendations_summary,
        "actions": lambda: t.upgrades_downgrades,
        "eps_estimate": lambda: t.earnings_estimate,
        "rev_estimate": lambda: t.revenue_estimate,
        "growth": lambda: t.growth_estimates,
        "eps_trend": lambda: t.eps_trend,
        "eps_revisions": lambda: t.eps_revisions,
    }
    raw: dict = {"status": {}}

    def _grab(name):
        try:
            val = _retry(sources[name])
            return name, val, ("ok" if _has_data(val) else "no data")
        except Exception as exc:
            return name, None, f"error: {type(exc).__name__}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, val, status in pool.map(_grab, list(sources)):
            raw[name] = val
            raw["status"][name] = status

    if not any(_has_data(raw.get(k)) for k in ("info", "price_targets", "ratings")):
        detail = "; ".join(f"{k}: {v}" for k, v in raw["status"].items())
        raise RuntimeError(f"Yahoo returned no analyst data (possibly rate-limited). {detail}")
    return raw


def _clear_analyst_cache():
    _fetch_analyst_raw.clear()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def build_analyst_view(raw: dict) -> dict:
    """Turn the raw Yahoo payloads into the numbers the tab and the verdict use."""
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    view: dict = {"available": True, "status": raw.get("status", {})}

    # --- rating counts (strong buy / buy / hold / sell / strong sell) over the last 3 months
    cols = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    ratings = raw.get("ratings")
    table = None
    if isinstance(ratings, pd.DataFrame) and not ratings.empty and "period" in ratings.columns:
        try:
            table = ratings.set_index("period")[[c for c in cols if c in ratings.columns]]
            table = table.apply(pd.to_numeric, errors="coerce").fillna(0)
            for c in cols:
                if c not in table.columns:
                    table[c] = 0.0
            table = table[cols]
        except Exception:
            table = None
    view["ratings_table"] = table

    def _row_stats(row):
        total = float(row.sum())
        if total <= 0:
            return None, None, None, None
        mean = (row["strongBuy"] * 1 + row["buy"] * 2 + row["hold"] * 3 + row["sell"] * 4 + row["strongSell"] * 5) / total
        return (
            float(mean),
            float((row["strongBuy"] + row["buy"]) / total * 100.0),
            float(row["hold"] / total * 100.0),
            float((row["sell"] + row["strongSell"]) / total * 100.0),
        )

    rating_mean = buy_pct = hold_pct = sell_pct = buy_pct_change = None
    n_from_counts = None
    if table is not None and not table.empty:
        now_row = table.loc["0m"] if "0m" in table.index else table.iloc[0]
        old_row = table.loc["-3m"] if "-3m" in table.index else table.iloc[-1]
        rating_mean, buy_pct, hold_pct, sell_pct = _row_stats(now_row)
        n_from_counts = int(now_row.sum())
        if len(table) >= 2:
            _, old_buy_pct, _, _ = _row_stats(old_row)
            if buy_pct is not None and old_buy_pct is not None:
                buy_pct_change = buy_pct - old_buy_pct
    if rating_mean is None:
        rating_mean = _num(info.get("recommendationMean"))
    view.update(
        {
            "rating_mean": rating_mean,
            "buy_pct": buy_pct,
            "hold_pct": hold_pct,
            "sell_pct": sell_pct,
            "buy_pct_change": buy_pct_change,
        }
    )

    key = (info.get("recommendationKey") or "").strip().lower()
    view["yahoo_label"] = key.replace("_", " ").title() if key and key != "none" else None

    # --- price targets (fall back to the info dict)
    pt = raw.get("price_targets") if isinstance(raw.get("price_targets"), dict) else {}
    view["target_mean"] = _num(pt.get("mean")) or _num(info.get("targetMeanPrice"))
    view["target_median"] = _num(pt.get("median")) or _num(info.get("targetMedianPrice"))
    view["target_high"] = _num(pt.get("high")) or _num(info.get("targetHighPrice"))
    view["target_low"] = _num(pt.get("low")) or _num(info.get("targetLowPrice"))
    n_info = info.get("numberOfAnalystOpinions")
    view["num_analysts"] = int(n_info) if _num(n_info) is not None else n_from_counts

    # --- EPS estimate revisions (current fiscal year estimate now vs 90 / 30 days ago)
    eps_rev_pct = None
    eps_rev_basis = None
    trend = raw.get("eps_trend")
    if isinstance(trend, pd.DataFrame) and not trend.empty and "0y" in trend.index:
        row = trend.loc["0y"]
        cur = _num(row.get("current"))
        for col, label in (("90daysAgo", "90 days"), ("30daysAgo", "30 days")):
            old = _num(row.get(col))
            if cur is not None and old not in (None, 0):
                # cap at +/-100% so a swing from a tiny base (or into losses) does not show absurd percentages
                eps_rev_pct = max(-100.0, min(100.0, (cur - old) / abs(old) * 100.0))
                eps_rev_basis = label
                break
    view["eps_rev_pct"] = eps_rev_pct
    view["eps_rev_basis"] = eps_rev_basis

    # --- tables for display
    view["eps_estimate"] = raw.get("eps_estimate") if isinstance(raw.get("eps_estimate"), pd.DataFrame) else None
    view["rev_estimate"] = raw.get("rev_estimate") if isinstance(raw.get("rev_estimate"), pd.DataFrame) else None
    view["growth"] = raw.get("growth") if isinstance(raw.get("growth"), pd.DataFrame) else None
    view["eps_revisions"] = raw.get("eps_revisions") if isinstance(raw.get("eps_revisions"), pd.DataFrame) else None
    actions = raw.get("actions")
    view["actions"] = actions.head(10).copy() if isinstance(actions, pd.DataFrame) and not actions.empty else None
    return view


def fetch_analyst_view(stock: str) -> dict:
    """Never raises: returns {'available': False, 'error': ...} when Yahoo gives nothing."""
    try:
        return build_analyst_view(_fetch_analyst_raw(stock))
    except Exception as exc:
        return {"available": False, "error": str(exc)[:400]}


def analyst_verdict(view: dict, price: float) -> dict:
    """
    Transparent verdict from four signals, each +1 (bullish), 0 (neutral) or -1 (bearish):
      1. Consensus rating (average of analyst ratings, 1 = Strong Buy ... 5 = Strong Sell)
      2. Upside from the current price to the mean price target (+10% or more bullish, -5% or less bearish)
      3. Rating trend: change in % of Buy/Strong Buy ratings vs 3 months ago
      4. EPS revisions: change in the current-year EPS estimate vs 90 days ago
    Total >= +2 -> Potential Buy; <= -2 -> Potential Sell; otherwise Neutral. Needs at least 2 available signals.
    """
    rows = []

    rm = view.get("rating_mean")
    if rm is None:
        rows.append(("Consensus rating", "N/A", None))
    else:
        sc = 1 if rm <= 2.3 else (-1 if rm >= 3.3 else 0)
        rows.append(("Consensus rating", f"{rm:.2f} (1 = Strong Buy, 5 = Strong Sell)", sc))

    tm = view.get("target_mean")
    if tm is None or not price:
        rows.append(("Upside to mean target", "N/A", None))
    else:
        up = (tm / price - 1.0) * 100.0
        sc = 1 if up >= 10 else (-1 if up <= -5 else 0)
        rows.append(("Upside to mean target", f"{up:+.1f}% (target ${tm:.2f} vs ${price:.2f})", sc))

    bc = view.get("buy_pct_change")
    if bc is None:
        rows.append(("Rating trend (3M)", "N/A", None))
    else:
        sc = 1 if bc >= 5 else (-1 if bc <= -5 else 0)
        rows.append(("Rating trend (3M)", f"{bc:+.1f} pts in % Buy ratings (now {view['buy_pct']:.0f}%)", sc))

    er = view.get("eps_rev_pct")
    if er is None:
        rows.append(("EPS revisions", "N/A", None))
    else:
        sc = 1 if er >= 1 else (-1 if er <= -1 else 0)
        rows.append(("EPS revisions", f"{er:+.1f}% vs {view.get('eps_rev_basis', '90 days')} ago (current-year EPS)", sc))

    scored = [r[2] for r in rows if r[2] is not None]
    total, n = sum(scored), len(scored)
    if n < 2:
        verdict = "Insufficient data"
    elif total >= 2:
        verdict = "Potential Buy"
    elif total <= -2:
        verdict = "Potential Sell"
    else:
        verdict = "Neutral"
    return {
        "rows": [{"Signal": a, "Reading": b, "Score": ("N/A" if c is None else f"{c:+d}")} for a, b, c in rows],
        "total": total,
        "n": n,
        "verdict": verdict,
    }


def plot_rating_distribution(view: dict):
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    table = view.get("ratings_table")
    if table is None or table.empty:
        ax.text(0.5, 0.5, "Rating counts not available.", ha="center")
        ax.axis("off")
        return fig
    names = {"0m": "Now", "-1m": "1M ago", "-2m": "2M ago", "-3m": "3M ago"}
    colors = {"strongBuy": "#1a9850", "buy": "#91cf60", "hold": "#c7c7c7", "sell": "#fc8d59", "strongSell": "#d73027"}
    labels = {"strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "strongSell": "Strong Sell"}
    rows = list(table.index)[::-1]  # oldest first so "Now" ends up at the bottom
    for y, period in enumerate(rows):
        left = 0.0
        for c in table.columns:
            v = float(table.loc[period, c])
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=colors[c], label=labels[c] if y == 0 else None, edgecolor="white")
            ax.text(left + v / 2, y, f"{int(v)}", ha="center", va="center", fontsize=8)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names.get(p, p) for p in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of analysts", fontsize=8)
    ax.set_title("Analyst ratings over the last 3 months", fontsize=10)
    handles, labs = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_price_targets(stock: str, price: float, view: dict):
    fig, ax = plt.subplots(figsize=(8, 1.9))
    low, high = view.get("target_low"), view.get("target_high")
    mean, median = view.get("target_mean"), view.get("target_median")
    if low is None or high is None:
        ax.text(0.5, 0.5, "Price target range not available.", ha="center")
        ax.axis("off")
        return fig
    ax.hlines(0, low, high, color="#9ecae1", linewidth=16, label=f"Analyst range ${low:.0f} - ${high:.0f}")
    if median is not None:
        ax.plot([median], [0], "s", color="#3182bd", markersize=8, label=f"Median ${median:.0f}")
    if mean is not None:
        ax.plot([mean], [0], "D", color="#08519c", markersize=8, label=f"Mean ${mean:.0f}")
    ax.axvline(price, color="red", linestyle="--", linewidth=1.5, label=f"Current ${price:.0f}")
    ax.set_xlim(min(low, price) * 0.97, max(high, price) * 1.03)
    ax.set_yticks([])
    ax.set_title(f"{stock} - price targets vs current price", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _format_estimates(df: pd.DataFrame, money: bool = False) -> pd.DataFrame:
    """Readable estimates table: friendly period names, growth as %, big numbers in billions."""
    names = {"0q": "Current quarter", "+1q": "Next quarter", "0y": "Current year", "+1y": "Next year"}
    out = df.copy()
    out.index = [names.get(str(i), str(i)) for i in out.index]
    for c in out.columns:
        if c == "growth":
            out[c] = out[c].map(lambda x: "N/A" if pd.isna(x) else f"{x * 100:+.1f}%")
        elif c == "numberOfAnalysts":
            out[c] = out[c].map(lambda x: "N/A" if pd.isna(x) else f"{int(x)}")
        else:
            out[c] = out[c].map(
                lambda x: "N/A" if pd.isna(x) else (f"{x / 1e9:.2f}B" if money and abs(x) >= 1e6 else f"{x:.2f}")
            )
    return out.rename(columns={"avg": "Average", "low": "Low", "high": "High", "yearAgoEps": "Year ago EPS",
                               "yearAgoRevenue": "Year ago revenue", "numberOfAnalysts": "# analysts", "growth": "Growth"})


@st.cache_data(ttl=900, show_spinner=False)
def fetch_financial_metrics(stock: str) -> dict:
    """
    Mirror the financial-metric logic from StockPickerComprehensive_v6.check_stocks
    for a single ticker so the web app matches the report.
    """
    ticker = yf.Ticker(stock)

    # Try to get the three core statements
    try:
        financials = ticker.financials.T if isinstance(ticker.financials, pd.DataFrame) else pd.DataFrame()
        balance_sheet = ticker.balance_sheet.T
        cashflow = ticker.cashflow.T
    except Exception:
        return {}

    # Sort from oldest -> newest for consistent multi‑year calculations
    if not financials.empty:
        financials = financials.sort_index()
    if not balance_sheet.empty:
        balance_sheet = balance_sheet.sort_index()
    if not cashflow.empty:
        cashflow = cashflow.sort_index()

    # Load static info dict once (used for market cap, dividend rate, etc.)
    info: dict = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # Market cap: prefer fast_info, fall back to info
    market_cap = None
    try:
        fast_info = getattr(ticker, "fast_info", None)
        market_cap = getattr(fast_info, "market_cap", None) if fast_info is not None else None
    except Exception:
        market_cap = None
    if market_cap is None:
        market_cap = info.get("marketCap", None)

    metrics: dict[str, object] = {
        "P/E (3Y Avg)": None,
        "P/E (1Y)": None,
        "Revenue Growth (2Y %)": None,
        "Profit Growth (2Y vs Today)": None,
        "Shares Outstanding (2Y)": None,
        "Div Rate (FWD)": None,
        "FCF Multiple (MC/FCF)": None,
        "Market Cap": market_cap,
        "Sector PE": None,
        "EPS (1Y)": None,
        "Net Debt": None,
        "Sector": info.get("sector"),
    }

    # Net income based metrics (P/E and profit growth)
    if not financials.empty and "Net Income" in financials.columns:
        ni = financials["Net Income"].dropna()

        # 1-year P/E = Market Cap / last year's profit
        if len(ni) >= 1 and market_cap is not None:
            last_profit = ni.iloc[-1]
            if pd.notnull(last_profit) and last_profit != 0:
                metrics["P/E (1Y)"] = market_cap / last_profit

        # 3-year average P/E and profit growth vs 2 years ago
        if len(ni) >= 3:
            if market_cap is not None:
                avg_profit = ni.iloc[-3:].mean()
                if pd.notnull(avg_profit) and avg_profit != 0:
                    metrics["P/E (3Y Avg)"] = market_cap / avg_profit

            latest_profit = ni.iloc[-1]
            profit_two_years_ago = ni.iloc[-3]
            if pd.notnull(latest_profit) and pd.notnull(profit_two_years_ago):
                if latest_profit > profit_two_years_ago:
                    metrics["Profit Growth (2Y vs Today)"] = "Up"
                elif latest_profit < profit_two_years_ago:
                    metrics["Profit Growth (2Y vs Today)"] = "Down"
                else:
                    metrics["Profit Growth (2Y vs Today)"] = "Flat"

    # Revenue growth: today vs 2 years ago
    if not financials.empty and "Total Revenue" in financials.columns:
        rev = financials["Total Revenue"].dropna()
        if len(rev) >= 3:
            latest_rev = rev.iloc[-1]
            rev_two_years_ago = rev.iloc[-3]
            if pd.notnull(latest_rev) and pd.notnull(rev_two_years_ago) and rev_two_years_ago != 0:
                growth = (latest_rev - rev_two_years_ago) / abs(rev_two_years_ago) * 100
                metrics["Revenue Growth (2Y %)"] = growth

    # Shares outstanding: buyback / issuing / neutral over last 2 years
    latest_shares = None
    if not balance_sheet.empty and "Share Issued" in balance_sheet.columns:
        shares = balance_sheet["Share Issued"].dropna()
        if len(shares) >= 3:
            latest_shares = shares.iloc[-1]
            shares_two_years_ago = shares.iloc[-3]
            if pd.notnull(latest_shares) and pd.notnull(shares_two_years_ago) and shares_two_years_ago != 0:
                change_pct = (latest_shares - shares_two_years_ago) / shares_two_years_ago * 100
                flag = "Neutral"
                if change_pct > 5:
                    flag = "Issuing Shares"
                elif change_pct < -5:
                    flag = "Buying Back"
                metrics["Shares Outstanding (2Y)"] = f"{flag} ({change_pct:.1f}%)"
                metrics["Shares Latest"] = latest_shares

    # EPS (1Y): Net Income / Shares Issued (latest year)
    if (
        not financials.empty
        and "Net Income" in financials.columns
        and latest_shares not in (None, 0)
    ):
        ni = financials["Net Income"].dropna()
        if len(ni) >= 1:
            latest_profit = ni.iloc[-1]
            if pd.notnull(latest_profit):
                metrics["EPS (1Y)"] = latest_profit / latest_shares

    # Free cash flow: CFO - capex for last year (match various yfinance column names)
    if not cashflow.empty:
        cf_row = cashflow.iloc[-1]

        def _find_val(*names, fallback_substrings=None):
            for n in names:
                for idx in cf_row.index:
                    if str(idx).strip() == str(n).strip():
                        v = cf_row.loc[idx]
                        if v is not None and pd.notnull(v):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
            if fallback_substrings:
                for idx in cf_row.index:
                    s = str(idx).lower()
                    if all(sub.lower() in s for sub in fallback_substrings):
                        v = cf_row.loc[idx]
                        if v is not None and pd.notnull(v):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
            return None

        cfo = _find_val(
            "Total Cash From Operating Activities",
            "Net Cash Provided by Operating Activities",
            "Net Cash Provided by (Used in) Operating Activities",
            "Cash From Operating Activities",
            fallback_substrings=("operating", "cash"),
        )
        capex = _find_val(
            "Capital Expenditures",
            "Purchase Of Property Plant And Equipment",
            "Capital Expenditure",
            "Purchase of Property, Plant and Equipment",
            fallback_substrings=("capital", "expend"),
        )
        if capex is None:
            capex = _find_val(
                "Purchase Of Property Plant And Equipment",
                fallback_substrings=("property", "equipment"),
            )

        # Yahoo reports capex as a NEGATIVE number, so use abs() (prefer Yahoo's own FCF row when present)
        fcf = _find_val("Free Cash Flow")
        if fcf is None and cfo is not None and capex is not None:
            fcf = cfo - abs(capex)
        if fcf is not None:
            if fcf not in (0, None):
                metrics["FCF (1Y)"] = fcf
                if market_cap is not None:
                    metrics["FCF Multiple (MC/FCF)"] = market_cap / fcf

    # Net debt in dollars (for DCF: enterprise -> equity). yfinance balance sheet returns values in dollars.
    if not balance_sheet.empty:
        bs_row = balance_sheet.iloc[-1]
        def _bs(key, *alt):
            for k in [key] + list(alt):
                for idx in bs_row.index:
                    if str(idx).strip() == str(k).strip():
                        v = bs_row.loc[idx]
                        if v is not None and pd.notnull(v):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
            return None
        lt_debt = _bs("Long Term Debt", "Long-Term Debt")
        st_debt = _bs("Short Long Term Debt", "Short Term Debt", "Short Long Term Debt")
        cash = _bs("Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments")
        if any(x is not None for x in (lt_debt, st_debt, cash)):
            net_debt_dollars = (lt_debt or 0) + (st_debt or 0) - (cash or 0)
            metrics["Net Debt"] = net_debt_dollars

        # Prefer Yahoo's own Net Debt line: it includes current debt and leases and nets cash + short-term investments
        yahoo_nd = None
        if "Net Debt" in balance_sheet.columns:
            s_nd = pd.to_numeric(balance_sheet["Net Debt"], errors="coerce").dropna()
            if not s_nd.empty:
                yahoo_nd = float(s_nd.iloc[-1])
        if yahoo_nd is None and "Total Debt" in balance_sheet.columns:
            cash_col = next(
                (c for c in ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents") if c in balance_sheet.columns),
                None,
            )
            s_td = pd.to_numeric(balance_sheet["Total Debt"], errors="coerce").dropna()
            s_cs = pd.to_numeric(balance_sheet[cash_col], errors="coerce").dropna() if cash_col else pd.Series(dtype=float)
            if not s_td.empty and not s_cs.empty:
                yahoo_nd = float(s_td.iloc[-1]) - float(s_cs.iloc[-1])
        if yahoo_nd is not None:
            metrics["Net Debt"] = yahoo_nd

    # Forward dividend rate (per share, annualised, when available)
    div_rate_fwd = info.get("dividendRate")
    if pd.notnull(div_rate_fwd):
        metrics["Div Rate (FWD)"] = div_rate_fwd

    # Sector PE via simple ETF mapping similar to the report
    try:
        sector = info.get("sector")
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financial Services": "XLF",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Basic Materials": "XLB",
            "Energy": "XLE",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
        }
        if sector in sector_etfs:
            sector_t = yf.Ticker(sector_etfs[sector])
            sector_pe = sector_t.info.get("trailingPE")
            if pd.notnull(sector_pe):
                metrics["Sector PE"] = sector_pe
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Long-term entry framework: quality, valuation context, trend, checklist
# ---------------------------------------------------------------------------

def _safe_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _get_row(df, names, contains=None) -> pd.Series:
    """Numeric Series (oldest -> newest) for the first matching line item of a yfinance statement."""
    empty = pd.Series(dtype=float)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty
    labels = {str(i).strip().lower(): i for i in df.index}

    def _series(label):
        row = df.loc[label]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        s = pd.to_numeric(row, errors="coerce").dropna()
        if s.empty:
            return None
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    for n in names:
        lab = labels.get(str(n).strip().lower())
        if lab is not None:
            s = _series(lab)
            if s is not None:
                return s
    if contains:
        for low, lab in labels.items():
            if all(c.lower() in low for c in contains):
                s = _series(lab)
                if s is not None:
                    return s
    return empty


def _by_year(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    out = s.copy()
    out.index = out.index.year
    return out[~out.index.duplicated(keep="last")]


def _cagr(s: pd.Series):
    """Compound annual growth (%) between first and last point, plus years spanned."""
    s = s.dropna()
    if len(s) < 2:
        return None, 0.0
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0 or first <= 0 or last <= 0:
        return None, years
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0, years


def _up_years(s: pd.Series):
    """(# of year-over-year increases, # of year-over-year comparisons)."""
    s = s.dropna()
    if len(s) < 2:
        return 0, 0
    diffs = s.diff().dropna()
    return int((diffs > 0).sum()), int(len(diffs))


@st.cache_data(ttl=900, show_spinner=False)
def download_daily(stock: str, years: int = 5) -> pd.DataFrame:
    """Daily bars for long-term trend / valuation-history work."""
    for attempt in range(2):
        try:
            data = yf.download(stock, period=f"{years}y", interval="1d", progress=False, auto_adjust=False)
        except Exception:
            data = None
        if data is not None and not data.empty:
            return data
        if attempt == 0:
            time.sleep(5)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quality_metrics(stock: str) -> dict:
    """Business quality and balance-sheet strength from Yahoo annual statements (about 4 fiscal years)."""
    out: dict = {"available": False}
    try:
        t = yf.Ticker(stock)
        inc, bs, cf = t.income_stmt, t.balance_sheet, t.cashflow
    except Exception:
        return out
    try:
        info = t.info or {}
    except Exception:
        info = {}

    revenue = _get_row(inc, ["Total Revenue", "Operating Revenue"])
    gross = _get_row(inc, ["Gross Profit"])
    op_inc = _get_row(inc, ["Operating Income", "EBIT"])
    net_inc = _get_row(inc, ["Net Income", "Net Income Common Stockholders"])
    ebitda = _get_row(inc, ["EBITDA", "Normalized EBITDA"])
    ebit = _get_row(inc, ["EBIT", "Operating Income"])
    interest = _get_row(inc, ["Interest Expense", "Interest Expense Non Operating"])
    tax_rate_s = _get_row(inc, ["Tax Rate For Calcs"])
    eps_s = _get_row(inc, ["Diluted EPS", "Basic EPS"])

    cfo = _get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], contains=("operating", "cash"))
    capex = _get_row(cf, ["Capital Expenditure", "Capital Expenditures"], contains=("capital", "expend"))
    fcf = _get_row(cf, ["Free Cash Flow"])
    if fcf.empty and not cfo.empty and not capex.empty:
        fcf = (cfo - capex.abs()).dropna()  # Yahoo reports capex as a negative number
    sbc = _get_row(cf, ["Stock Based Compensation"])

    total_debt = _get_row(bs, ["Total Debt"])
    net_debt_s = _get_row(bs, ["Net Debt"])
    cash = _get_row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    equity = _get_row(bs, ["Stockholders Equity", "Common Stock Equity"])
    invested = _get_row(bs, ["Invested Capital"])
    cur_assets = _get_row(bs, ["Current Assets"])
    cur_liab = _get_row(bs, ["Current Liabilities"])

    if revenue.empty and net_inc.empty:
        return out

    rev_y, gross_y, op_y, ni_y, fcf_y = (_by_year(s) for s in (revenue, gross, op_inc, net_inc, fcf))
    table = pd.DataFrame(
        {
            "Revenue ($B)": rev_y / 1e9,
            "Gross margin %": gross_y / rev_y * 100,
            "Operating margin %": op_y / rev_y * 100,
            "Net margin %": ni_y / rev_y * 100,
            "FCF margin %": fcf_y / rev_y * 100,
        }
    ).sort_index().dropna(how="all")

    # ROIC = NOPAT / invested capital (latest fiscal year)
    roic = None
    if not op_inc.empty:
        tr = _safe_float(tax_rate_s.iloc[-1]) if not tax_rate_s.empty else 0.21
        tr = min(max(tr if tr is not None else 0.21, 0.0), 0.35)
        ic = _safe_float(invested.iloc[-1]) if not invested.empty else None
        if ic is None and not equity.empty:
            ic = float(equity.iloc[-1]) + (float(total_debt.iloc[-1]) if not total_debt.empty else 0.0) - (
                float(cash.iloc[-1]) if not cash.empty else 0.0
            )
        if ic is not None and ic > 0:
            roic = float(op_inc.iloc[-1]) * (1.0 - tr) / ic * 100.0

    # Leverage
    net_debt = _safe_float(net_debt_s.iloc[-1]) if not net_debt_s.empty else None
    if net_debt is None and not total_debt.empty and not cash.empty:
        net_debt = float(total_debt.iloc[-1]) - float(cash.iloc[-1])
    ebitda_latest = _safe_float(ebitda.iloc[-1]) if not ebitda.empty else None
    net_debt_ebitda = None
    if net_debt is not None and ebitda_latest is not None and ebitda_latest > 0:
        net_debt_ebitda = net_debt / ebitda_latest

    interest_cov = None
    if not interest.empty and not ebit.empty:
        ie = abs(float(interest.iloc[-1]))
        if ie > 0:
            interest_cov = float(ebit.iloc[-1]) / ie

    current_ratio = None
    if not cur_assets.empty and not cur_liab.empty and float(cur_liab.iloc[-1]) != 0:
        current_ratio = float(cur_assets.iloc[-1]) / float(cur_liab.iloc[-1])

    sbc_pct_fcf = None
    if not sbc.empty and not fcf.empty and float(fcf.iloc[-1]) > 0:
        sbc_pct_fcf = abs(float(sbc.iloc[-1])) / float(fcf.iloc[-1]) * 100.0

    op_margin_change = None
    om = table["Operating margin %"].dropna() if "Operating margin %" in table.columns else pd.Series(dtype=float)
    if len(om) >= 2:
        op_margin_change = float(om.iloc[-1] - om.iloc[0])

    rev_cagr, rev_years = _cagr(revenue)
    ni_cagr, _ = _cagr(net_inc)
    fcf_cagr, _ = _cagr(fcf)
    eps_cagr, _ = _cagr(eps_s)
    ni_up, ni_n = _up_years(net_inc)
    eps_up, eps_n = _up_years(eps_s)
    rev_up, rev_n = _up_years(revenue)

    roe = _safe_float(info.get("returnOnEquity"))
    roa = _safe_float(info.get("returnOnAssets"))
    out.update(
        {
            "available": True,
            "table": table,
            "roic": roic,
            "roe": roe * 100.0 if roe is not None else None,
            "roa": roa * 100.0 if roa is not None else None,
            "op_margin_change": op_margin_change,
            "rev_cagr": rev_cagr,
            "rev_years": rev_years,
            "ni_cagr": ni_cagr,
            "fcf_cagr": fcf_cagr,
            "eps_cagr": eps_cagr,
            "ni_up": ni_up,
            "ni_n": ni_n,
            "eps_up": eps_up,
            "eps_n": eps_n,
            "rev_up": rev_up,
            "rev_n": rev_n,
            "fcf_pos": int((fcf > 0).sum()) if not fcf.empty else None,
            "fcf_n": int(len(fcf)) if not fcf.empty else None,
            "net_debt": net_debt,
            "ebitda": ebitda_latest,
            "net_debt_ebitda": net_debt_ebitda,
            "interest_coverage": interest_cov,
            "current_ratio": current_ratio,
            "sbc_pct_fcf": sbc_pct_fcf,
        }
    )
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_valuation_context(stock: str) -> dict:
    """Forward P/E, PEG, EV/EBITDA, 10Y yield, and the stock's own P/E history (annual EPS basis)."""
    out: dict = {
        "forward_pe": None, "peg": None, "ev_ebitda": None, "treasury_10y": None,
        "pe_hist": None, "pe_current": None, "pe_percentile": None,
        "pe_median": None, "pe_p25": None, "pe_p75": None, "pe_years": None,
    }
    try:
        t = yf.Ticker(stock)
        info = t.info or {}
    except Exception:
        return out

    out["forward_pe"] = _safe_float(info.get("forwardPE"))
    out["peg"] = _safe_float(info.get("trailingPegRatio")) or _safe_float(info.get("pegRatio"))
    out["ev_ebitda"] = _safe_float(info.get("enterpriseToEbitda"))

    try:
        y = _safe_float(yf.Ticker("^TNX").fast_info.get("lastPrice"))
        if y is not None:
            out["treasury_10y"] = y / 10.0 if y > 25 else y  # some feeds quote yield x10
    except Exception:
        pass

    # P/E history: price / latest annual diluted EPS (available ~60 days after fiscal year-end)
    try:
        daily = download_daily(stock, 5)
        eps = _get_row(t.income_stmt, ["Diluted EPS", "Basic EPS"])
        if not daily.empty and not eps.empty:
            px = normalize_ohlcv_columns(daily)["Close"].astype(float)
            px.index = pd.to_datetime(px.index).tz_localize(None)
            p = pd.DataFrame({"Date": px.index.astype("datetime64[ns]"), "Close": px.values}).sort_values("Date")
            e = pd.DataFrame(
                {"avail": (eps.index + pd.Timedelta(days=60)).astype("datetime64[ns]"), "EPS": eps.values.astype(float)}
            ).sort_values("avail")
            m = pd.merge_asof(p, e, left_on="Date", right_on="avail", direction="backward").dropna(subset=["EPS"])
            m = m[m["EPS"] > 0].copy()
            if len(m) >= 60:
                m["PE"] = m["Close"] / m["EPS"]
                cur = float(m["PE"].iloc[-1])
                out.update(
                    {
                        "pe_hist": m[["Date", "Close", "EPS", "PE"]].reset_index(drop=True),
                        "pe_current": cur,
                        "pe_percentile": float((m["PE"] <= cur).mean() * 100.0),
                        "pe_median": float(m["PE"].median()),
                        "pe_p25": float(m["PE"].quantile(0.25)),
                        "pe_p75": float(m["PE"].quantile(0.75)),
                        "pe_years": float((m["Date"].iloc[-1] - m["Date"].iloc[0]).days / 365.25),
                    }
                )
    except Exception:
        pass
    return out


def compute_trend_context(daily: pd.DataFrame, bench: pd.DataFrame) -> dict:
    """Long-term trend context from ~5Y of daily bars: 50/200D MA, drawdowns, weekly RSI, relative strength."""
    out: dict = {}
    if daily is None or daily.empty:
        return out
    try:
        d = normalize_ohlcv_columns(daily)
    except Exception:
        return out
    d.index = pd.to_datetime(d.index).tz_localize(None)
    close = d["Close"].astype(float).dropna()
    if len(close) < 210:
        return out

    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    hi52, lo52 = float(close.tail(252).max()), float(close.tail(252).min())
    dd_series = (close / close.cummax() - 1.0) * 100.0
    weekly = close.resample("W").last().dropna()
    wrsi = RSIIndicator(weekly, window=14).rsi() if len(weekly) > 20 else pd.Series(dtype=float)

    def _ret(s: pd.Series, days: int):
        if len(s) <= days:
            return None
        return (float(s.iloc[-1]) / float(s.iloc[-days - 1]) - 1.0) * 100.0

    bench_close = None
    if bench is not None and not bench.empty:
        try:
            b = normalize_ohlcv_columns(bench)
            b.index = pd.to_datetime(b.index).tz_localize(None)
            bench_close = b["Close"].astype(float).dropna()
        except Exception:
            bench_close = None

    r6, r12 = _ret(close, 126), _ret(close, 252)
    b6 = _ret(bench_close, 126) if bench_close is not None else None
    b12 = _ret(bench_close, 252) if bench_close is not None else None

    out.update(
        {
            "price": last,
            "ma50": float(ma50.iloc[-1]),
            "ma200": float(ma200.iloc[-1]),
            "pct_vs_ma200": (last / float(ma200.iloc[-1]) - 1.0) * 100.0,
            "ma50_above_ma200": bool(ma50.iloc[-1] > ma200.iloc[-1]),
            "dd_52w": (last / hi52 - 1.0) * 100.0,
            "pos_52w": (last - lo52) / (hi52 - lo52) * 100.0 if hi52 > lo52 else None,
            "max_dd": float(dd_series.min()),
            "weekly_rsi": float(wrsi.dropna().iloc[-1]) if not wrsi.dropna().empty else None,
            "ret_6m": r6,
            "ret_1y": r12,
            "rel_6m": (r6 - b6) if (r6 is not None and b6 is not None) else None,
            "rel_1y": (r12 - b12) if (r12 is not None and b12 is not None) else None,
            "series": pd.DataFrame({"Close": close, "MA50": ma50, "MA200": ma200, "Drawdown %": dd_series}),
            "weekly_rsi_series": wrsi,
        }
    )
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def sp500_trailing_return(years: int = 3):
    """Annualized S&P 500 index (^GSPC) price return over the last `years` years, in percent."""
    try:
        data = yf.download("^GSPC", period=f"{years + 2}y", interval="1d", progress=False, auto_adjust=False)
        close = data["Close"]
        close = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        start = close.index[-1] - pd.DateOffset(years=years)
        base = close[close.index <= start]
        if base.empty:
            return None
        return float(((close.iloc[-1] / base.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0)
    except Exception:
        return None


def eps_model_values(eps: float, g: float, k: float, r: float, years: int):
    """EPS model: (price in year X, fair value today)."""
    price_future = eps * (1 + g) ** years * k
    fair = price_future / (1 + r) ** years if r > -1 else None
    return price_future, fair


def is_financial(fin: dict) -> bool:
    return bool(fin) and fin.get("Sector") == "Financial Services"


def compute_dcf_intrinsic(fin: dict, g: float, k: float, r: float, horizon: int = 10):
    """10Y FCF DCF with terminal multiple; equity value = EV - net debt; returns intrinsic value per share."""
    if not fin or r <= -1 or is_financial(fin):
        return None  # a cash-flow DCF is not meaningful for banks / insurers
    fcf_1y = fin.get("FCF (1Y)")
    shares_latest = fin.get("Shares Latest")
    net_debt = fin.get("Net Debt")
    if fcf_1y is None or shares_latest in (None, 0):
        return None
    fcf_scale = 1.0
    if abs(fcf_1y) > 0 and abs(fcf_1y) < 1e7 and shares_latest > 1e6:
        fcf_scale = 1000.0  # FCF looks like it is reported in thousands
    fcf_dollars = fcf_1y * fcf_scale

    enterprise = 0.0
    for t in range(1, horizon + 1):
        enterprise += fcf_dollars * (1 + g) ** t / (1 + r) ** t
    enterprise += fcf_dollars * (1 + g) ** horizon * k / (1 + r) ** horizon

    equity = max(0.0, enterprise - (net_debt or 0.0))
    return equity / shares_latest


def _zone_row(model: str, price: float, fair: float, buy: float) -> dict:
    if price <= buy:
        status = "Buy zone"
    elif price <= fair:
        status = "Between buy price and fair value"
    else:
        status = "Above fair value"
    return {
        "Model": model,
        "Fair value": fair,
        "Buy price": buy,
        "Current price": price,
        "Upside to fair value %": (fair / price - 1.0) * 100.0 if price else None,
        "Status": status,
    }


def summarize_valuation(fin: dict, price: float, g: float, k: float, r: float, mos: float) -> list:
    """Fair value / buy price vs current price for the EPS model (5Y, 10Y) and the DCF."""
    rows: list = []
    if not fin:
        return rows
    eps = fin.get("EPS (1Y)")
    if eps is not None and eps > 0:
        for years in (5, 10):
            _, fair = eps_model_values(eps, g, k, r, years)
            if fair is not None:
                rows.append(_zone_row(f"EPS model ({years}Y)", price, fair, fair * (1 - mos)))
    intrinsic = compute_dcf_intrinsic(fin, g, k, r)
    if intrinsic is not None and intrinsic > 0:
        rows.append(_zone_row("DCF (10Y)", price, intrinsic, intrinsic * (1 - mos)))
    return rows


def build_sensitivity_grid(fin: dict, model: str, g_pct: float, k: float, r_pct: float, mos_pct: float) -> pd.DataFrame:
    """Buy price grid across growth (rows) and terminal multiple (columns)."""
    g_vals = [g_pct + d for d in (-5.0, -2.5, 0.0, 2.5, 5.0)]
    k_vals = sorted({max(1.0, k + d) for d in (-6.0, -3.0, 0.0, 3.0, 6.0)})
    r, mos = r_pct / 100.0, mos_pct / 100.0
    eps = fin.get("EPS (1Y)") if fin else None
    grid = pd.DataFrame(
        index=[f"g = {gv:g}%" for gv in g_vals], columns=[f"K = {kv:g}" for kv in k_vals], dtype=float
    )
    for gv, gl in zip(g_vals, grid.index):
        for kv, kl in zip(k_vals, grid.columns):
            val = None
            if model.startswith("EPS"):
                if eps is not None and eps > 0:
                    _, fair = eps_model_values(eps, gv / 100.0, kv, r, 10)
                    val = fair * (1 - mos) if fair is not None else None
            else:
                intrinsic = compute_dcf_intrinsic(fin, gv / 100.0, kv, r)
                val = intrinsic * (1 - mos) if intrinsic is not None else None
            grid.loc[gl, kl] = val if val is not None else float("nan")
    return grid


def _chk(name: str, passed, value: str) -> dict:
    return {"Check": name, "Value": value, "Result": "na" if passed is None else ("pass" if passed else "fail")}


def _fmt(v, suffix: str = "", nd: int = 1) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}{suffix}"


def build_long_term_checklist(price: float, fin: dict, quality: dict, valctx: dict, trend: dict, val_rows: list, st_sig: dict) -> dict:
    """Pass / fail / n.a. checks grouped into Quality, Valuation and Timing."""
    q = quality or {}
    v = valctx or {}
    tr = trend or {}
    fin = fin or {}

    # ---- Quality ----
    nd, eb = q.get("net_debt"), q.get("ebitda")
    if nd is None:
        lev_pass, lev_val = None, "N/A"
    elif nd <= 0:
        lev_pass, lev_val = True, "Net cash"
    elif eb is None:
        lev_pass, lev_val = None, "N/A"
    elif eb <= 0:
        lev_pass, lev_val = False, "EBITDA <= 0"
    else:
        lev_pass, lev_val = nd / eb <= 3.0, f"{nd / eb:.1f}x"

    eps_n = q.get("eps_n") or 0
    eps_pass = (q["eps_up"] / eps_n >= 0.6) if (eps_n and q.get("eps_up") is not None) else None
    fcf_n = q.get("fcf_n") or 0
    fcf_pos = (q["fcf_pos"] == fcf_n) if (fcf_n and q.get("fcf_pos") is not None) else None
    shares_txt = fin.get("Shares Outstanding (2Y)")
    quality_checks = [
        _chk("ROIC >= 10%", None if q.get("roic") is None else q["roic"] >= 10, _fmt(q.get("roic"), "%")),
        _chk(
            "Operating margin not shrinking (>= -2 pts)",
            None if q.get("op_margin_change") is None else q["op_margin_change"] >= -2.0,
            _fmt(q.get("op_margin_change"), " pts"),
        ),
        _chk("Revenue CAGR >= 5%", None if q.get("rev_cagr") is None else q["rev_cagr"] >= 5, _fmt(q.get("rev_cagr"), "%")),
        _chk(
            "EPS up in >= 60% of years",
            eps_pass,
            f"{q['eps_up']}/{eps_n} yrs" if eps_n else "N/A",
        ),
        _chk("FCF positive every year", fcf_pos, f"{q['fcf_pos']}/{fcf_n} yrs" if fcf_n else "N/A"),
        _chk("Net debt / EBITDA <= 3x", lev_pass, lev_val),
        _chk(
            "Interest coverage >= 5x",
            None if q.get("interest_coverage") is None else q["interest_coverage"] >= 5,
            _fmt(q.get("interest_coverage"), "x"),
        ),
        _chk(
            "SBC <= 20% of FCF",
            None if q.get("sbc_pct_fcf") is None else q["sbc_pct_fcf"] <= 20,
            _fmt(q.get("sbc_pct_fcf"), "%"),
        ),
        _chk(
            "No heavy dilution (2Y)",
            None if not shares_txt else not str(shares_txt).startswith("Issuing"),
            str(shares_txt) if shares_txt else "N/A",
        ),
    ]

    # ---- Valuation ----
    by_model = {r["Model"]: r for r in val_rows}
    eps10, dcf = by_model.get("EPS model (10Y)"), by_model.get("DCF (10Y)")

    def _vs_buy(row):
        if not row:
            return None, "N/A"
        return price <= row["Buy price"], f"${price:.1f} vs ${row['Buy price']:.1f}"

    eps_pass, eps_val = _vs_buy(eps10)
    dcf_pass, dcf_val = _vs_buy(dcf)

    pct = v.get("pe_percentile")
    pe_val = "N/A" if pct is None else f"{pct:.0f}th pct (P/E {v['pe_current']:.1f}, median {v['pe_median']:.1f})"

    mcap, fcf1 = fin.get("Market Cap"), fin.get("FCF (1Y)")
    fcf_yield = (fcf1 / mcap * 100.0) if (mcap and fcf1 is not None and mcap > 0) else None
    pe1 = fin.get("P/E (1Y)")
    fwd = v.get("forward_pe")

    valuation_checks = [
        _chk("Price <= EPS-model buy price (10Y)", eps_pass, eps_val),
        _chk("Price <= DCF buy price (10Y)", dcf_pass, dcf_val),
        _chk("P/E at or below 40th pct of own history", None if pct is None else pct <= 40, pe_val),
        _chk("FCF yield >= 4%", None if fcf_yield is None else fcf_yield >= 4, _fmt(fcf_yield, "%")),
        _chk(
            "Forward P/E below trailing P/E",
            None if (fwd is None or pe1 is None or pe1 <= 0) else fwd < pe1,
            "N/A" if (fwd is None or pe1 is None) else f"{fwd:.1f} vs {pe1:.1f}",
        ),
        _chk("PEG <= 2", None if v.get("peg") is None else v["peg"] <= 2, _fmt(v.get("peg"))),
    ]

    # ---- Timing ----
    triggers = [st_sig.get("below_bb_low"), st_sig.get("rsi_extreme"), st_sig.get("macd_crossover")]
    n_trig = sum(1 for x in triggers if x)
    timing_checks = [
        _chk(
            "Price within 5% of / above 200-day MA",
            None if tr.get("pct_vs_ma200") is None else tr["pct_vs_ma200"] >= -5,
            _fmt(tr.get("pct_vs_ma200"), "% vs 200D"),
        ),
        _chk(
            "Pullback: >= 10% below 52-week high",
            None if tr.get("dd_52w") is None else tr["dd_52w"] <= -10,
            _fmt(tr.get("dd_52w"), "%"),
        ),
        _chk(
            "Weekly RSI below 50 (momentum cooled)",
            None if tr.get("weekly_rsi") is None else tr["weekly_rsi"] < 50,
            _fmt(tr.get("weekly_rsi")),
        ),
        _chk(
            "Not collapsing vs SPY (6M gap > -15 pts)",
            None if tr.get("rel_6m") is None else tr["rel_6m"] > -15,
            _fmt(tr.get("rel_6m"), " pts"),
        ),
        _chk("4h triggers: >= 2 of BB low / RSI<30 / MACD>Signal", n_trig >= 2, f"{n_trig}/3"),
    ]
    # Debt, cash-flow and ROIC based checks are not meaningful for banks / insurers
    if is_financial(fin):
        skip = ("ROIC", "FCF positive", "Net debt / EBITDA", "Interest coverage", "SBC", "FCF yield", "Price <= DCF")
        for c in quality_checks + valuation_checks:
            if c["Check"].startswith(skip):
                c["Result"], c["Value"] = "na", "N/A (financial)"

    return {"quality": quality_checks, "valuation": valuation_checks, "timing": timing_checks}


def pillar_score(checks: list):
    """(percent passed, passed, scored) ignoring N/A checks."""
    scored = [c for c in checks if c["Result"] != "na"]
    if not scored:
        return None, 0, 0
    passed = sum(1 for c in scored if c["Result"] == "pass")
    return passed / len(scored) * 100.0, passed, len(scored)


def suggest_valuation_defaults(fin: dict, quality: dict, valctx: dict) -> dict:
    """Stock-specific starting assumptions: growth g (%), terminal multiple K, margin of safety (%)."""
    q = quality or {}
    v = valctx or {}
    notes: list = []

    growth = [x for x in (q.get("rev_cagr"), q.get("eps_cagr"), q.get("fcf_cagr")) if x is not None]
    if growth:
        g = min(10.0, max(3.0, float(pd.Series(growth).median())))
        g = round(g * 2) / 2.0
        notes.append(f"g {g:g}% = median of historical revenue / EPS / FCF growth (kept within 3-10%)")
    else:
        g = 7.0
        notes.append("g 7% (generic; no growth history)")

    nd_ebitda = q.get("net_debt_ebitda")
    risky = (
        (nd_ebitda is not None and nd_ebitda > 3)
        or ((q.get("net_debt") or 0) > 0 and q.get("ebitda") is not None and q["ebitda"] <= 0)
        or (q.get("fcf_pos") is not None and q.get("fcf_n") and q["fcf_pos"] < q["fcf_n"])
    ) and not is_financial(fin)

    pe_med = v.get("pe_median")
    if pe_med is not None:
        k = float(round(min(22.0, max(12.0, pe_med))))
        notes.append(f"K {k:g} = own median P/E (kept within 12-22)")
    else:
        k = 18.0
        notes.append("K 18 (generic; no P/E history)")
    if risky and k > 15.0:
        k = 15.0
        notes.append("K capped at 15 (leverage or negative FCF)")

    q_pct = pillar_score(build_long_term_checklist(0.0, fin, q, v, {}, [], {})["quality"])[0]
    if risky:
        mos = 35.0
        notes.append("MoS 35% (leverage or negative FCF)")
    elif q_pct is not None and q_pct >= 80:
        mos = 15.0
        notes.append("MoS 15% (high-quality business)")
    else:
        mos = 25.0
        notes.append("MoS 25% (default)")
    return {"g": g, "k": k, "mos": mos, "notes": "; ".join(notes)}


def _reset_val_inputs(sugg: dict, r_default: float):
    st.session_state["val_g_pct"] = float(sugg["g"])
    st.session_state["val_k_multiple"] = float(sugg["k"])
    st.session_state["val_mos_pct"] = float(sugg["mos"])
    st.session_state["val_r_pct"] = float(r_default)


def long_term_verdict(scores: dict):
    """Returns (streamlit level, headline, detail)."""
    q, v, t = scores["quality"][0], scores["valuation"][0], scores["timing"][0]
    if q is None or v is None:
        return "info", "Not enough data", "Yahoo did not return enough statement data to score quality and valuation."
    # Timing sizes the entry; it does not veto a good business at a good price.
    t_ok = t is None or t >= 50
    if q >= 70 and v >= 50 and t_ok:
        return "success", "Strong long-term entry candidate - build the position in tranches", "Quality and valuation pass and timing supports adding now."
    if q >= 70 and v >= 50:
        return "info", "Good business at a fair price - start about 1/3 of a position", "Timing is not ideal yet; add the rest on pullbacks or when the timing score improves."
    if q >= 70:
        return "warning", "Quality business, but not cheap enough yet", "Add to a watchlist and wait for a better price (or revisit your growth / multiple assumptions)."
    if v >= 50 and q < 50:
        return "error", "Cheap, but quality is weak - value-trap / falling-knife risk", "Cheap valuation alone is not a long-term buy point."
    return "warning", "Not a long-term buy point yet", "Quality or valuation criteria are not met."


def plot_long_term_trend(stock: str, trend: dict):
    s = trend["series"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1, 1]})
    axes[0].plot(s.index, s["Close"], label="Close", color="blue", linewidth=1.0)
    axes[0].plot(s.index, s["MA50"], label="50D MA", color="orange", linewidth=0.9)
    axes[0].plot(s.index, s["MA200"], label="200D MA", color="red", linewidth=0.9)
    axes[0].set_title(f"{stock} - long-term trend (daily, ~5Y)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(s.index, s["Drawdown %"], 0, color="#d62728", alpha=0.35)
    axes[1].set_ylabel("Drawdown %", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    w = trend.get("weekly_rsi_series")
    if w is not None and not w.empty:
        axes[2].plot(w.index, w.values, color="purple", linewidth=0.9)
        axes[2].axhline(30, color="green", linestyle="--", linewidth=0.8)
        axes[2].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Weekly RSI", fontsize=8)
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pe_history(stock: str, valctx: dict):
    h = valctx.get("pe_hist")
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    if h is None or h.empty:
        axes[0].text(0.5, 0.5, "P/E history not available for this ticker.", ha="center")
        axes[1].axis("off")
        return fig
    axes[0].plot(h["Date"], h["PE"], color="blue", linewidth=1.0, label="P/E (price / latest annual EPS)")
    axes[0].axhline(valctx["pe_median"], color="gray", linestyle="--", linewidth=0.9, label="Median")
    axes[0].fill_between(h["Date"], valctx["pe_p25"], valctx["pe_p75"], color="gray", alpha=0.15, label="25th-75th pct")
    axes[0].scatter([h["Date"].iloc[-1]], [h["PE"].iloc[-1]], color="red", zorder=5, label="Now")
    axes[0].set_ylim(0, float(h["PE"].quantile(0.98)) * 1.2)
    axes[0].set_title(f"{stock} - P/E vs own history (~{valctx['pe_years']:.1f}Y)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(h["Date"], h["Close"], color="gray", linewidth=1.0)
    axes[1].set_ylabel("Price", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_technicals(stock: str, data: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(data.index, data["Close"], label="Close Price", color="blue")
    axes[0].plot(data.index, data["BB_High"], label="BB High", color="red", linestyle="--")
    axes[0].plot(data.index, data["BB_Low"], label="BB Low", color="green", linestyle="--")
    axes[0].set_title(f"{stock} - Close Price and Bollinger Bands")
    axes[0].legend(loc="upper left")

    axes[1].plot(data.index, data["RSI"], label="RSI", color="blue")
    axes[1].axhline(30, color="green", linestyle="--")
    axes[1].axhline(70, color="red", linestyle="--")
    axes[1].set_title(f"{stock} - RSI")
    axes[1].legend(loc="upper left")

    axes[2].plot(data.index, data["MACD"], label="MACD", color="blue")
    axes[2].plot(data.index, data["MACD_Signal"], label="MACD Signal", color="red")
    axes[2].set_title(f"{stock} - MACD")
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_reversal_scorecard(scores: dict, total: int, label: str):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    crit = list(scores.keys())
    vals = [scores[k] for k in crit]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]

    ax.barh(crit, vals, color=colors, alpha=0.9)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Score per criterion (0 or 25)")
    ax.set_title(f"Reversal Confidence Scorecard: {total}/100 ({label})")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for y, v in enumerate(vals):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=9)
    fig.tight_layout()
    return fig


def compute_reversal_diagnostics(raw_data: pd.DataFrame, stock: str) -> dict | None:
    """
    Compute the underlying numeric pieces used by the reversal confidence score.
    Returns:
      - daily dataframe (1D resample)
      - poc_price (scalar)
      - vwap_30 (scalar)
      - avg20_vol (scalar)
      - obv proxy series (20-day window, based on daily close direction)
      - today's candle metrics for the volume rule
    """
    data = normalize_ohlcv_columns(raw_data)
    data.index = pd.to_datetime(data.index).tz_localize(None)

    daily = data.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    if len(daily) < 30:
        return None

    # OBV divergence proxy (20-day): OBV-like cumulative sum driven by daily close direction.
    obv_window = daily.tail(20).copy()
    close_diff = obv_window["Close"].diff().fillna(0.0)
    direction = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv_proxy = (direction * obv_window["Volume"]).cumsum()

    # Volume Profile (POC) proxy (30-day): bin the daily closes and assign daily volume to bins.
    vp = daily.tail(30).copy()
    n_bins = 20
    bin_codes, bin_edges = pd.cut(
        vp["Close"], bins=n_bins, labels=False, retbins=True, include_lowest=True
    )
    vp["bin"] = bin_codes
    volume_by_bin = vp.groupby("bin")["Volume"].sum()
    if volume_by_bin.empty:
        return None
    poc_bin = int(volume_by_bin.idxmax())
    poc_price = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)

    # VWAP (30-day) proxy: typical-price weighted by daily volume
    typical_price = (vp["High"] + vp["Low"] + vp["Close"]) / 3.0
    vol_sum = float(vp["Volume"].sum())
    vwap_30 = float((typical_price * vp["Volume"]).sum() / vol_sum) if vol_sum > 0 else None

    avg20_vol = float(daily["Volume"].tail(20).mean())
    today = daily.iloc[-1]
    is_green = float(today["Close"]) > float(today["Open"])
    vol_ratio = float(today["Volume"]) / avg20_vol if avg20_vol > 0 else None

    return {
        "daily": daily,
        "obv_window": obv_window,
        "obv_proxy": obv_proxy,
        "poc_price": poc_price,
        "vwap_30": vwap_30,
        "avg20_vol": avg20_vol,
        "today": today,
        "is_green": is_green,
        "vol_ratio": vol_ratio,
    }


def plot_reversal_diagnostics(raw_1h: pd.DataFrame, indicator_4h: pd.DataFrame, stock: str, reversal_conf: dict):
    diag = compute_reversal_diagnostics(raw_1h, stock)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.2, 1.3, 1.6]})

    if not diag or diag.get("vwap_30") is None:
        axes[0].text(0.5, 0.5, "Not enough data to plot reversal diagnostics.", ha="center")
        for ax in axes[1:]:
            ax.axis("off")
        return fig

    # Price panel (4h): Close + BB + VWAP and POC levels (horizontal lines).
    ax_price = axes[0]
    ax_price.plot(indicator_4h.index, indicator_4h["Close"], label="Close (4h)", color="blue", linewidth=1.0)
    ax_price.plot(indicator_4h.index, indicator_4h.get("BB_High"), label="BB High (2σ)", color="red", linestyle="--", linewidth=0.9)
    ax_price.plot(indicator_4h.index, indicator_4h.get("BB_Low"), label="BB Low (2σ)", color="green", linestyle="--", linewidth=0.9)
    ax_price.axhline(diag["vwap_30"], label="VWAP (30D)", color="purple", linestyle="-", linewidth=1.0)
    ax_price.axhline(diag["poc_price"], label="POC (30D)", color="orange", linestyle="--", linewidth=1.1)

    ax_price.set_title(f"Reversal diagnostics – {stock} | Score {reversal_conf['total']}/100")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.25)

    # OBV panel (daily, 20-day proxy): OBV-like proxy + scaled Close (so you can see divergence).
    ax_obv = axes[1]
    obv_proxy = diag["obv_proxy"]
    ax_obv.plot(obv_proxy.index, obv_proxy.values, label="OBV proxy (20D)", color="#1f77b4")
    close = diag["obv_window"]["Close"]
    if close.max() != close.min():
        scaled_close = (close - close.min()) / (close.max() - close.min())
        scaled_obv = (obv_proxy - obv_proxy.min())
        if scaled_obv.max() != 0:
            scaled_close = scaled_close * scaled_obv.max() + obv_proxy.min()
        ax_obv.plot(close.index, scaled_close.values, label="Close (scaled)", color="#7f7f7f", linewidth=1.0)
    ax_obv.set_ylabel("OBV proxy")
    ax_obv.legend(loc="upper left", fontsize=8)
    ax_obv.grid(True, alpha=0.25)

    # Volume panel (daily, 30-day window): daily volume bars + 20D avg line.
    ax_vol = axes[2]
    daily = diag["daily"]
    avg20 = diag["avg20_vol"]
    vp_window = daily.tail(30).copy()
    vol_colors = []
    for _, r in vp_window.iterrows():
        green = float(r["Close"]) > float(r["Open"])
        vol_pass = avg20 > 0 and float(r["Volume"]) >= 1.2 * avg20
        vol_colors.append("#2ca02c" if (green and vol_pass) else "#c7c7c7")
    ax_vol.bar(vp_window.index, vp_window["Volume"].values, color=vol_colors, width=0.8)
    ax_vol.axhline(avg20, color="blue", linestyle="--", linewidth=1.0, label="20D avg vol")
    ax_vol.set_ylabel("Volume")
    ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Stock Technical Dashboard", layout="wide")
    # Reduce vertical spacing; compact tables
    st.markdown("""
        <style>
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
        [data-testid="stDataFrame"] { font-size: 0.9rem; }
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { padding: 0.25rem 0.5rem !important; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Stock Technical Dashboard")
    st.caption("Bollinger Bands (200/2), RSI (10), and MACD (8/21/5) on 4-hour candles.")

    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    favorites = list(st.session_state["favorites"]) if st.session_state["favorites"] else []
    favorites = [str(s).strip().upper() for s in favorites if s]
    symbols = load_default_symbols()
    if not isinstance(symbols, list):
        symbols = list(symbols) if symbols else ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
    symbols = [str(s).strip().upper() for s in symbols if s]
    fav_set = set(favorites)
    options = favorites + [s for s in symbols if s not in fav_set]
    if not options:
        options = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Select stock ticker", options=options, index=0)
    with col2:
        custom_stock = st.text_input("Or type custom ticker", value="").strip().upper()
    with col3:
        st.write("")  # align button
        st.write("")
        load_clicked = st.button("Load chart", type="primary")

    stock = custom_stock if custom_stock else selected_stock

    if "active_stock" not in st.session_state:
        st.session_state["active_stock"] = None

    if load_clicked:
        st.session_state["active_stock"] = stock
    # Auto-load the selected ticker on first visit so the app shows content immediately
    elif st.session_state["active_stock"] is None and stock:
        st.session_state["active_stock"] = stock

    active_stock = st.session_state["active_stock"]

    if not active_stock:
        st.info("Select a stock and click 'Load chart' to view charts, financial metrics, and valuation.")
        return

    with st.spinner(f"Loading data for {active_stock}..."):
        raw_data = download_data(active_stock)

    if raw_data.empty:
        st.error("No data was returned from Yahoo Finance. Try another ticker or retry in a minute.")
        return

    try:
        indicator_data = calculate_indicators(raw_data, active_stock)
    except Exception as exc:
        st.error(f"Could not calculate indicators for {active_stock}: {exc}")
        return

    if indicator_data.empty:
        st.warning("Not enough data points after indicator calculations.")
        return

    latest = indicator_data.iloc[-1]
    below_bb_low = latest["Close"] < latest["BB_Low"]
    rsi_extreme = latest["RSI"] < 30
    macd_crossover = latest["MACD"] > latest["MACD_Signal"]
    reversal_conf = calculate_reversal_confidence(raw_data, active_stock)

    # Fetch financial metrics once for both Financial metrics and Valuation tabs
    fin = fetch_financial_metrics(active_stock)
    analyst = fetch_analyst_view(active_stock)

    # Long-term context (quality, valuation history, trend). Each degrades gracefully if Yahoo data is missing.
    quality = fetch_quality_metrics(active_stock)
    valctx = fetch_valuation_context(active_stock)
    trend = compute_trend_context(download_daily(active_stock), download_daily("SPY"))

    # Default required return r = trailing 3-year S&P 500 annualized return + 2%, capped at 15% (fallback 12%)
    sp3 = sp500_trailing_return(3)
    r_default = round(min(sp3 + 2.0, 15.0), 1) if sp3 is not None else 12.0

    # Stock-specific starting assumptions; reset whenever the active ticker changes (r keeps any override)
    sugg = suggest_valuation_defaults(fin, quality, valctx)
    defaults_changed = st.session_state.get("val_defaults_for") != active_stock
    for key, val in (("val_g_pct", sugg["g"]), ("val_k_multiple", sugg["k"]), ("val_mos_pct", sugg["mos"])):
        if defaults_changed or key not in st.session_state:
            st.session_state[key] = float(val)
    if "val_r_pct" not in st.session_state:
        st.session_state["val_r_pct"] = float(r_default)
    st.session_state["val_defaults_for"] = active_stock

    tab_charts, tab_lt, tab_fin, tab_val, tab_paper, tab_analyst, tab_fav, tab_explain = st.tabs(
        [
            "Charts",
            "Long-term checklist",
            "Financial metrics",
            "Valuation",
            "Paper investing check",
            "Analyst view",
            "Favorites",
            "Explanation",
        ]
    )

    current_price = float(latest["Close"])

    with tab_charts:
        c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
        c0.metric("Current price", f"${current_price:.2f}")
        c1.metric("Below BB Low", "Yes" if below_bb_low else "No")
        c2.metric("RSI < 30", "Yes" if rsi_extreme else "No")
        c3.metric("MACD > Signal", "Yes" if macd_crossover else "No")
        r1, r2 = st.columns([1, 2])
        r1.metric("Reversal confidence", f"{reversal_conf['total']}/100")
        r2.metric("Confidence label", reversal_conf["label"])
        comp_df = pd.DataFrame(
            [{"Criterion": k, "Score": v} for k, v in reversal_conf["scores"].items()]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True, height=182)
        with st.expander("Rule values used (for debugging / transparency)", expanded=False):
            diag = compute_reversal_diagnostics(raw_data, active_stock)
            if not diag:
                st.info("Not enough data to compute rule values yet.")
            else:
                latest_close = float(diag["today"]["Close"])
                st.write(
                    {
                        "Latest close": f"${latest_close:.2f}",
                        "VWAP (30D)": f"${diag['vwap_30']:.2f}" if diag.get("vwap_30") is not None else "—",
                        "POC (30D proxy)": f"${diag['poc_price']:.2f}" if diag.get("poc_price") is not None else "—",
                        "Volume (today)": f"{diag['today']['Volume']:.0f}",
                        "Avg volume (20D)": f"{diag['avg20_vol']:.0f}",
                        "Volume ratio": f"{diag['vol_ratio']:.2f}x" if diag.get("vol_ratio") is not None else "—",
                        "Latest candle green?": bool(diag["is_green"]),
                    }
                )
                # OBV divergence diagnostics
                obv_proxy = diag["obv_proxy"]
                obv_trend = float(obv_proxy.iloc[-1] - obv_proxy.iloc[0])
                price_trend = float(diag["obv_window"]["Close"].iloc[-1] - diag["obv_window"]["Close"].iloc[0])
                st.write(
                    {
                        "OBV proxy trend (20D)": f"{obv_trend:.2f}",
                        "Price change (20D)": f"{price_trend:.2f}",
                    }
                )
        st.caption("Reversal confidence visualization (VWAP, POC, OBV proxy, volume strength).")
        diag_fig = plot_reversal_diagnostics(raw_data, indicator_data, active_stock, reversal_conf)
        st.pyplot(diag_fig, use_container_width=True)

        with st.expander("Score breakdown chart (0 or 25 per rule)", expanded=False):
            score_fig = plot_reversal_scorecard(
                reversal_conf["scores"], reversal_conf["total"], reversal_conf["label"]
            )
            st.pyplot(score_fig, use_container_width=True)
        fig = plot_technicals(active_stock, indicator_data)
        st.pyplot(fig)
        with st.expander("Last 30 periods (4h)", expanded=False):
            st.dataframe(
                indicator_data[["Close", "BB_High", "BB_Low", "RSI", "MACD", "MACD_Signal"]].tail(30),
                height=280,
            )

    with tab_lt:
        st.subheader(f"Long-term buy-point checklist - {active_stock}")
        st.caption(
            "Three pillars: business quality, valuation, and timing. Thresholds are rules of thumb "
            "(see Explanation tab). Valuation checks use the assumptions from the Valuation tab."
        )
        g_in = st.session_state.get("val_g_pct", 10.0)
        k_in = st.session_state.get("val_k_multiple", 15.0)
        r_in = st.session_state.get("val_r_pct", r_default)
        mos_in = st.session_state.get("val_mos_pct", 25.0)
        lt_val_rows = summarize_valuation(fin, current_price, g_in / 100.0, k_in, r_in / 100.0, mos_in / 100.0)
        checklist = build_long_term_checklist(
            current_price,
            fin,
            quality,
            valctx,
            trend,
            lt_val_rows,
            {
                "below_bb_low": bool(below_bb_low),
                "rsi_extreme": bool(rsi_extreme),
                "macd_crossover": bool(macd_crossover),
            },
        )
        pillar_scores = {key: pillar_score(checks) for key, checks in checklist.items()}
        level, headline, detail = long_term_verdict(pillar_scores)
        getattr(st, level)(f"**{headline}** - {detail}")

        pillars = (("quality", "Quality"), ("valuation", "Valuation"), ("timing", "Timing"))
        for col, (key, label) in zip(st.columns(3), pillars):
            pct, passed, scored = pillar_scores[key]
            col.metric(f"{label} score", "N/A" if pct is None else f"{passed}/{scored} ({pct:.0f}%)")

        for col, (key, label) in zip(st.columns(3), pillars):
            with col:
                df_chk = pd.DataFrame(checklist[key])
                df_chk["Result"] = df_chk["Result"].map({"pass": "Pass", "fail": "Fail", "na": "N/A"})
                st.dataframe(
                    df_chk[["Check", "Value", "Result"]],
                    hide_index=True,
                    use_container_width=True,
                    height=36 * (len(df_chk) + 1) + 3,
                )

        ch1, ch2 = st.columns(2)
        with ch1:
            if trend.get("series") is not None:
                st.pyplot(plot_long_term_trend(active_stock, trend), use_container_width=True)
            else:
                st.info("Not enough daily history for the long-term trend chart (needs about 210 trading days).")
        with ch2:
            st.pyplot(plot_pe_history(active_stock, valctx), use_container_width=True)

        with st.expander("Quality detail (annual statements from Yahoo Finance)", expanded=False):
            if not quality.get("available"):
                st.info("Annual statement data is not available for this ticker.")
            else:
                tbl = quality["table"].copy()
                tbl.index = [str(i) for i in tbl.index]
                st.dataframe(tbl.round(1), use_container_width=True)
                st.write(
                    {
                        "ROIC": _fmt(quality.get("roic"), "%"),
                        "ROE (Yahoo)": _fmt(quality.get("roe"), "%"),
                        "Revenue CAGR": _fmt(quality.get("rev_cagr"), f"% over {quality.get('rev_years', 0):.1f}y"),
                        "Net income CAGR": _fmt(quality.get("ni_cagr"), "%"),
                        "EPS CAGR": _fmt(quality.get("eps_cagr"), "%"),
                        "FCF CAGR": _fmt(quality.get("fcf_cagr"), "%"),
                        "Current ratio": _fmt(quality.get("current_ratio"), "x"),
                        "EV / EBITDA": _fmt(valctx.get("ev_ebitda"), "x"),
                    }
                )
                st.caption("Yahoo provides about 4 fiscal years, so growth rates cover roughly 3 years.")

    with tab_fin:
        st.subheader(f"Financial metrics - {active_stock}")
        if not fin:
            st.info("Financial metrics are not available for this ticker right now.")
        else:
            fin_pretty: dict[str, object] = {}
            for k, v in fin.items():
                if isinstance(v, (int, float)):
                    if k == "Market Cap" or k == "Net Debt":
                        av = abs(v)
                        if av >= 1e9:
                            fin_pretty[k] = f"{v / 1e9:.1f}B"
                        elif av >= 1e6:
                            fin_pretty[k] = f"{v / 1e6:.1f}M"
                        else:
                            fin_pretty[k] = f"{v:.1f}"
                    elif k == "FCF (1Y)":
                        if abs(v) >= 1e9:
                            fin_pretty[k] = f"{v / 1e9:.1f}B"
                        elif abs(v) >= 1e6:
                            fin_pretty[k] = f"{v / 1e6:.1f}M"
                        else:
                            fin_pretty[k] = f"{v:.1f}"
                    elif k.endswith("(2Y %)") or k.endswith("Growth (2Y %)"):
                        fin_pretty[k] = f"{v:.1f}%"
                    else:
                        fin_pretty[k] = f"{v:.1f}"
                else:
                    fin_pretty[k] = v if v is not None else "N/A"
            # Compact table: current price row + one row per metric
            rows = [{"Metric": "Current price (latest close)", "Value": f"${current_price:.2f}"}]
            rows += [{"Metric": k, "Value": str(v)} for k, v in fin_pretty.items()]
            df_fin = pd.DataFrame(rows)
            st.dataframe(df_fin, use_container_width=True, height=min(44 * (len(rows) + 1), 400), hide_index=True)

    with tab_val:
        st.subheader(f"Valuation - {active_stock}")
        st.metric("Current price (latest close)", f"${current_price:.2f}")
        if not fin or fin.get("EPS (1Y)") is None:
            st.info("Valuation inputs are not available (missing Net Income or Shares Issued data).")
        else:
            eps = fin.get("EPS (1Y)")
            st.markdown(f"**M₀ (EPS)**: **{eps:.1f}**")
            if quality.get("available"):
                st.caption(
                    "Reference for choosing g (historical): "
                    f"revenue CAGR {_fmt(quality.get('rev_cagr'), '%')}, "
                    f"EPS CAGR {_fmt(quality.get('eps_cagr'), '%')}, "
                    f"FCF CAGR {_fmt(quality.get('fcf_cagr'), '%')}. "
                    "Long-run growth rarely stays above these."
                )

            v1, v2, v3, v4 = st.columns(4)
            with v1:
                g_pct = st.number_input("Growth g (%)", step=0.5, min_value=-50.0, max_value=100.0, key="val_g_pct")
            with v2:
                k_multiple = st.number_input("Terminal K", step=1.0, min_value=0.0, key="val_k_multiple")
            with v3:
                r_pct = st.number_input("Return r (%)", step=0.5, min_value=0.0, max_value=50.0, key="val_r_pct")
            with v4:
                mos_pct = st.number_input("MoS (%)", step=1.0, min_value=0.0, max_value=90.0, key="val_mos_pct")

            cap_note = " (capped at 15%)" if (sp3 is not None and sp3 + 2.0 > 15.0) else ""
            r_note = (
                f"Default r = S&P 500 trailing 3Y annualized return ({sp3:.1f}%) + 2%{cap_note} = {r_default:.1f}%. "
                if sp3 is not None
                else "Default r = 12% (S&P 500 return unavailable). "
            )
            st.caption(r_note + "Suggested for this stock: " + sugg["notes"] + ". Edit any input to override.")
            st.button("Reset inputs to suggested", key="reset_val_inputs", on_click=_reset_val_inputs, args=(sugg, r_default))
            g = g_pct / 100.0
            r = r_pct / 100.0
            mos = mos_pct / 100.0

            rows = []
            for years in (1, 5, 10):
                metric_future = eps * (1 + g) ** years
                price_future = metric_future * k_multiple
                fair_value = price_future / (1 + r) ** years if r > -1 else None
                buy_price = fair_value * (1 - mos) if fair_value is not None else None
                rows.append(
                    {
                        "Horizon (years)": years,
                        "Price Year X": None if price_future is None else round(price_future, 1),
                        "Fair Value Today": None if fair_value is None else round(fair_value, 1),
                        "Buy Price (with MOS)": None if buy_price is None else round(buy_price, 1),
                    }
                )
            val_df = pd.DataFrame(rows)
            st.dataframe(val_df, hide_index=True)

            # DCF model: equity value = EV - Net Debt, per share (shared helper)
            intrinsic_dcf = compute_dcf_intrinsic(fin, g, k_multiple, r)
            if intrinsic_dcf is None:
                if is_financial(fin):
                    st.info(
                        "DCF is not meaningful for banks and insurers (debt and cash are part of the operating business). "
                        "Use the EPS-based model above, plus P/B and ROE."
                    )
                else:
                    st.info("DCF model not available (missing FCF or share count).")
            else:
                buy_price_dcf = intrinsic_dcf * (1 - mos)
                dcf_df = pd.DataFrame(
                    [
                        {"Metric": "Current price", "Value": f"${current_price:.2f}"},
                        {"Metric": "DCF Intrinsic (10y)", "Value": f"${round(intrinsic_dcf, 1):.1f}"},
                        {"Metric": "DCF Buy (with MOS)", "Value": f"${round(buy_price_dcf, 1):.1f}"},
                    ]
                )
                st.markdown("**DCF (10-year; equity = EV − Net Debt)**")
                st.dataframe(dcf_df, hide_index=True)

            # Current price vs fair value / buy price for every model
            st.markdown("**Is the current price in the buy zone?**")
            zone_rows = summarize_valuation(fin, current_price, g, k_multiple, r, mos)
            if not zone_rows:
                st.info("No model produced a positive fair value (check EPS / FCF).")
            else:
                zone_df = pd.DataFrame(zone_rows)
                for col_name in ("Fair value", "Buy price", "Current price"):
                    zone_df[col_name] = zone_df[col_name].map(lambda x: f"${x:.1f}")
                zone_df["Upside to fair value %"] = zone_df["Upside to fair value %"].map(lambda x: f"{x:+.1f}%")
                st.dataframe(zone_df, hide_index=True, use_container_width=True)
                n_buy = sum(1 for z in zone_rows if z["Status"] == "Buy zone")
                if n_buy == len(zone_rows):
                    st.success(f"Current price is at or below the buy price in all {n_buy} models.")
                elif n_buy:
                    st.info(f"Current price is in the buy zone for {n_buy} of {len(zone_rows)} models.")
                else:
                    st.warning("Current price is above the buy price in every model with these assumptions.")

            # Sensitivity grid: buy price across g and K
            st.markdown("**Sensitivity: 10Y buy price across growth (g) and terminal multiple (K)**")
            sens_model = st.radio(
                "Model", ["EPS model (10Y)", "DCF (10Y)"], horizontal=True, key="sens_model", label_visibility="collapsed"
            )
            grid = build_sensitivity_grid(fin, sens_model, g_pct, k_multiple, r_pct, mos_pct)
            if grid.isna().all().all():
                st.info("Not enough data for this model's sensitivity grid.")
            else:

                def _cell_color(v, price=current_price):
                    if pd.isna(v):
                        return ""
                    if price <= v:
                        return "background-color: #d4edda; color: #155724"
                    return "background-color: #f8d7da; color: #721c24"

                styler = grid.style.format("${:.1f}", na_rep="—")
                styler = styler.map(_cell_color) if hasattr(styler, "map") else styler.applymap(_cell_color)
                st.dataframe(styler, use_container_width=True)
                st.caption(
                    f"Each cell is the buy price (fair value x (1 − MoS)) at your r = {r_pct:g}% and MoS = {mos_pct:g}%. "
                    f"Green = current price (${current_price:.1f}) is at or below it; red = above."
                )

    with tab_paper:
        st.subheader(f"Paper investing check – {active_stock}")
        st.caption(
            "Simulates: buy when price below BB lower band, RSI < 30, and MACD crosses above Signal; "
            "sell the chosen % when price above BB upper band, RSI > 70, and MACD crosses below Signal. "
            "Uses 4-hour bars over the selected period."
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            paper_years = st.number_input("Number of years", value=2, min_value=1, max_value=2, step=1, key="paper_years")
        with p2:
            paper_money = st.number_input("Money invested per buy ($)", value=1000.0, min_value=1.0, step=100.0, key="paper_money")
        with p3:
            paper_pct = st.number_input("Money taken out (%)", value=10.0, min_value=1.0, max_value=100.0, step=1.0, key="paper_pct")
        pct_decimal = paper_pct / 100.0

        with st.spinner(f"Loading {paper_years} year(s) of 4h data for {active_stock}..."):
            raw_paper = download_data_for_paper(active_stock, paper_years)
        if raw_paper.empty:
            st.warning("No historical data for this period. Yahoo 1h data is limited to about 2 years.")
        else:
            try:
                paper_4h = calculate_indicators(raw_paper, active_stock)
            except Exception as exc:
                st.error(f"Could not compute indicators: {exc}")
                paper_4h = pd.DataFrame()
            if paper_4h.empty or len(paper_4h) < 2:
                st.warning("Not enough 4h bars after computing indicators (need 200+ for BB).")
            else:
                res = run_paper_simulation(paper_4h, paper_money, pct_decimal)
                st.metric("Current price (latest close)", f"${current_price:.2f}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Realized gain", f"${res['realized_gain']:,.2f}")
                m2.metric("Unrealized gain", f"${res['unrealized_gain']:,.2f}")
                m3.metric("Rate of return", f"{res['rate_of_return_pct']:.1f}%")
                m4.metric("Total invested", f"${res['total_invested']:,.2f}")
                st.caption(
                    f"Final wealth: ${res['final_wealth']:,.2f} "
                    f"(cash from sales: ${res['cash_from_sales']:,.2f}, "
                    f"current position: ${res['current_value']:,.2f} on {res['shares']:.2f} shares). "
                    f"Buys: {res['n_buys']}, Sells: {res['n_sells']}."
                )
                with st.expander("Buy / sell events", expanded=False):
                    if not res["events"]:
                        st.write("No events in this period.")
                    else:
                        ev_df = pd.DataFrame(
                            [
                                {
                                    "Time": e[0],
                                    "Action": e[1],
                                    "Shares": round(e[2], 4),
                                    "Price": round(e[3], 2),
                                    "Amount ($)": round(e[4], 2),
                                }
                                for e in res["events"]
                            ]
                        )
                        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    with tab_analyst:
        st.subheader(f"Analyst view - {active_stock}")
        st.caption(
            "Consensus, price targets and estimate revisions from Yahoo Finance. These are third-party analyst opinions, "
            "not a recommendation from this app. The Explanation tab shows exactly how the verdict is calculated."
        )
        if not analyst.get("available"):
            st.warning(
                "Analyst data could not be loaded from Yahoo Finance right now (often a temporary rate limit on shared "
                "servers). Failed lookups are not cached, so retrying usually works."
            )
            st.button("Retry analyst data", key="retry_analyst", on_click=_clear_analyst_cache)
            if analyst.get("error"):
                st.caption(f"Details: {analyst['error']}")
        else:
            av = analyst_verdict(analyst, current_price)
            banner = {"Potential Buy": "success", "Potential Sell": "error", "Neutral": "info"}.get(av["verdict"], "warning")
            getattr(st, banner)(
                f"**Analyst verdict: {av['verdict']}** (score {av['total']:+d}, based on {av['n']} of 4 available signals; "
                "+2 or more = Potential Buy, -2 or less = Potential Sell)"
            )

            tm = analyst.get("target_mean")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Yahoo consensus", analyst.get("yahoo_label") or "N/A")
            m2.metric("Avg rating (1-5)", _fmt(analyst.get("rating_mean"), nd=2))
            m3.metric("Analysts", "N/A" if analyst.get("num_analysts") is None else str(analyst["num_analysts"]))
            m4.metric(
                "Mean target",
                "N/A" if tm is None else f"${tm:.2f}",
                None if tm is None else f"{(tm / current_price - 1) * 100:+.1f}% vs ${current_price:.2f}",
            )
            lo, hi = analyst.get("target_low"), analyst.get("target_high")
            m5.metric("Target range", "N/A" if lo is None or hi is None else f"${lo:.0f} - ${hi:.0f}")

            left, right = st.columns([1, 1])
            with left:
                st.pyplot(plot_rating_distribution(analyst), use_container_width=True)
            with right:
                st.markdown("**How the verdict is built**")
                st.dataframe(pd.DataFrame(av["rows"]), hide_index=True, use_container_width=True, height=36 * 5 + 3)
                st.caption(
                    f"Score = sum of the four signals ({av['total']:+d}). Needs at least 2 available signals; "
                    "N/A signals are ignored."
                )

            st.pyplot(plot_price_targets(active_stock, current_price, analyst), use_container_width=True)

            with st.expander("Earnings and revenue estimates, revisions", expanded=False):
                if analyst.get("eps_estimate") is not None:
                    st.markdown("**EPS estimates**")
                    st.dataframe(_format_estimates(analyst["eps_estimate"]), use_container_width=True)
                if analyst.get("rev_estimate") is not None:
                    st.markdown("**Revenue estimates ($)**")
                    st.dataframe(_format_estimates(analyst["rev_estimate"], money=True), use_container_width=True)
                if analyst.get("eps_revisions") is not None:
                    st.markdown("**EPS estimate revisions (number of analysts)**")
                    rev_df = analyst["eps_revisions"].copy()
                    rev_df.index = [{"0q": "Current quarter", "+1q": "Next quarter", "0y": "Current year", "+1y": "Next year"}.get(str(i), str(i)) for i in rev_df.index]
                    st.dataframe(rev_df, use_container_width=True)
                if all(analyst.get(k) is None for k in ("eps_estimate", "rev_estimate", "eps_revisions")):
                    st.info("No estimate data available for this ticker.")

            with st.expander("Recent analyst actions (latest 10)", expanded=False):
                acts = analyst.get("actions")
                if acts is None:
                    st.info("No recent upgrades / downgrades available.")
                else:
                    a = acts.reset_index()
                    date_col = a.columns[0]
                    def _pt(row):
                        cur, prior = row.get("currentPriceTarget"), row.get("priorPriceTarget")
                        if pd.isna(cur) or cur in (0, 0.0):
                            return "-"
                        return f"${cur:.0f}" if pd.isna(prior) or prior in (0, 0.0) else f"${prior:.0f} -> ${cur:.0f}"
                    disp = pd.DataFrame(
                        {
                            "Date": pd.to_datetime(a[date_col]).dt.strftime("%Y-%m-%d"),
                            "Firm": a.get("Firm"),
                            "Rating": a.get("FromGrade").fillna("").astype(str) + " -> " + a.get("ToGrade").fillna("").astype(str),
                            "Action": a.get("Action"),
                            "Price target": a.apply(_pt, axis=1),
                        }
                    )
                    st.dataframe(disp, hide_index=True, use_container_width=True)

            with st.expander("Data source status", expanded=False):
                labels = {
                    "info": "Company summary (info)", "price_targets": "Price targets", "ratings": "Rating counts",
                    "actions": "Upgrades / downgrades", "eps_estimate": "EPS estimates", "rev_estimate": "Revenue estimates",
                    "growth": "Growth estimates", "eps_trend": "EPS trend", "eps_revisions": "EPS revisions",
                }
                st.dataframe(
                    pd.DataFrame([{"Source": labels.get(k, k), "Status": v} for k, v in analyst["status"].items()]),
                    hide_index=True,
                    use_container_width=True,
                )
                st.button("Reload analyst data", key="reload_analyst", on_click=_clear_analyst_cache)

    with tab_fav:
        st.subheader("Favorites")
        st.caption("Favorites appear at the top of the ticker dropdown. Add the current stock or pick one to load.")
        fav_list = st.session_state["favorites"]
        if active_stock and active_stock not in fav_list:
            if st.button("Add current stock to Favorites", key="add_fav"):
                st.session_state["favorites"] = fav_list + [active_stock]
                st.rerun()
        if active_stock and active_stock in fav_list:
            st.caption(f"**{active_stock}** is in your favorites.")
        if not fav_list:
            st.info("No favorites yet. Load a stock, then click 'Add current stock to Favorites' above.")
        else:
            for ticker in fav_list:
                r1, r2, r3 = st.columns([2, 1, 1])
                with r1:
                    st.text(ticker)
                with r2:
                    if st.button("Load", key=f"load_fav_{ticker}"):
                        st.session_state["active_stock"] = ticker
                        st.rerun()
                with r3:
                    if st.button("Remove", key=f"rem_fav_{ticker}"):
                        st.session_state["favorites"] = [t for t in fav_list if t != ticker]
                        st.rerun()

    with tab_explain:
        st.subheader("Explanation: what each tab helps you do and how to read every number")
        st.markdown(
            "Use this page as a guide to the whole app. Each section below covers one tab: "
            "**what it is for**, **how to use it**, and **what each metric means and how to read it**."
        )

        with st.expander("Start here: what each tab is for and a suggested workflow", expanded=True):
            st.markdown("""
| Tab | The question it answers | Use it when |
|---|---|---|
| **Charts** | Is the price stretched or turning right now? | Timing an entry over the next days to weeks |
| **Long-term checklist** | Is this a good business, at a good price, at a sensible moment? | First screen for any stock; the main decision tab |
| **Financial metrics** | How healthy are the company's profit, growth, cash and debt? | Checking the business behind the ticker |
| **Valuation** | What could it be worth, and what is the most I should pay? | Setting a buy price from your own assumptions |
| **Paper investing check** | Would the app's technical buy and sell rules have made money on this stock? | Testing the timing rules before trusting them |
| **Analyst view** | What do professional analysts think, and is that changing? | A cross-check against your own view |
| **Favorites** | Shortcuts to the stocks I follow | Saving a search through the dropdown |

**Suggested workflow**
1. Pick a ticker at the top (dropdown, or type your own) and click **Load chart**.
2. Open **Long-term checklist**. Read the verdict and the three scores (Quality, Valuation, Timing). A failed check is a question to investigate, not an automatic no.
3. Open **Financial metrics** and **Valuation** to understand the business and set your own growth, multiple and margin-of-safety assumptions. Compare the current price with the **Buy price**.
4. Use **Charts** to judge timing, and **Paper investing check** to see whether those timing signals have actually worked for this stock.
5. Cross-check with **Analyst view**. Add stocks you follow to **Favorites**.

**Ideal long-term buy:** a high-quality business (Quality), priced below your buy price (Valuation), at a moment when the trend is intact and the price has pulled back (Timing).
""")

        with st.expander("Top bar and general notes (ticker, current price, data freshness)"):
            st.markdown("""
- **Select stock ticker:** your Favorites appear first, then the stocks from your stock list.
- **Or type custom ticker:** overrides the dropdown. Use the Yahoo Finance symbol (for example BRK-B for Berkshire).
- **Load chart:** loads the selected ticker into every tab. Changing the dropdown or typing a ticker does nothing until you click this button. On your first visit the app loads the first ticker automatically.
- **Current price:** the latest close of the most recent 4-hour bar, so it can lag the live price slightly.
- **Data source:** Yahoo Finance for everything. Prices refresh about every 15 minutes, company statements and valuation history about every hour, so numbers may not change on every click.
- **Missing data:** when Yahoo has no value, the app shows N/A and ignores that item when scoring.
- **Not investment advice.** The app organizes public data and simple rules; it cannot see the future.
""")

        with st.expander("Charts tab: short-term timing"):
            st.markdown("""
**What it helps you do:** judge whether the price is unusually low or turning up right now, and whether volume supports a bounce. It says nothing about the quality of the business, so use it to time an entry, not to choose what to own.

**The four tiles at the top**
- **Current price:** latest 4-hour close.
- **Below BB Low:** Yes when the price is under the lower Bollinger Band. The band is the 200-bar average of 4-hour closes minus 2 standard deviations (roughly 100 trading days). Yes means the price is unusually low compared with its recent range.
- **RSI < 30:** RSI (10 bars) measures recent buying versus selling pressure on a 0-100 scale. Under 30 is called oversold (selling may be exhausted); over 70 is overbought.
- **MACD > Signal:** MACD is the gap between a fast (8) and slow (21) moving average; the Signal line is a 5-bar average of MACD. MACD above Signal means upward momentum is building.

**How to read them:** all three Yes is the classic oversold-and-turning-up setup. It is a timing clue, not a reason to buy a weak company, because a falling stock can stay oversold for a long time.

**Reversal confidence (0-100):** a score for whether a bounce has real support behind it. Four checks are worth 25 points each:
1. **OBV divergence:** over the last 20 days, volume flow (up days add volume, down days subtract it) is rising while price is flat or down. This suggests buyers are quietly accumulating.
2. **Volume Profile (POC):** the Point of Control is the price level where the most volume traded over the last 30 days (daily closes grouped into 20 price bins). Scores when the latest close is above it, meaning price has moved above where most trading happened.
3. **Volume bars:** the latest daily candle is green (close above open) and its volume is more than 1.2 times the 20-day average, meaning a bounce on real participation.
4. **VWAP:** the volume-weighted average price over the last 30 days, using the typical price (high + low + close) divided by 3. Scores when the latest close is above it, meaning buyers are in control.

Because each check is worth 25, the score can only be 0, 25, 50, 75 or 100. **Above 75** (so 100, all four) is **Strong Reversal Signal**. **Below 40** (0 or 25) is **Likely Dead Cat Bounce**. 50 and 75 show as **Neutral / Mixed**. OBV and POC use daily bars built from the same hourly data (a proxy, not exchange volume-at-price).

**Charts and tables on the tab**
- **Reversal diagnostics chart:** price with Bollinger Bands, the VWAP line (purple) and the POC line (orange); below it the OBV line against scaled price (to spot divergence) and daily volume bars (green = green candle on volume at least 1.2 times average).
- **Score breakdown chart** and **Rule values used:** the same four checks as bars and as the exact numbers behind them, so you can see why a check passed or failed.
- **Bollinger / RSI / MACD chart:** price with the bands, then RSI with 30 and 70 lines, then MACD against its Signal.
- **Last 30 periods (4h):** the raw indicator values.
""")

        with st.expander("Long-term checklist tab: the main decision tab"):
            st.markdown("""
**What it helps you do:** answer "is this a good long-term buy right now?" in one place. It scores three pillars, each made of Pass / Fail / N/A checks (N/A checks are ignored):
- **Quality:** is it a business worth owning for years?
- **Valuation:** is the price attractive?
- **Timing:** is this a sensible moment to start?

**How to read the page**
- **Banner:** the verdict (below).
- **Three scores:** for example 7/9 (78%) means 7 of the 9 scored checks passed.
- **Three tables:** each check, its current value, and Pass / Fail / N/A. Thresholds are rules of thumb, not guarantees. A Fail tells you where to look, not that the stock is bad.
- **Long-term trend chart (daily, about 5 years):** price with 50-day (orange) and 200-day (red) averages, the drawdown from the previous peak, and weekly RSI with 30 and 70 lines.
- **P/E vs own history chart:** today's P/E (red dot) against its own median (dashed) and middle 50% range (gray band), based on price divided by the latest annual EPS over roughly 3-4 years.
- **Quality detail:** yearly revenue, margins and growth rates behind the Quality checks. Yahoo provides about 4 fiscal years, so growth rates cover roughly 3 years.

**Verdict.** Timing sizes the entry; it does not veto a good business at a fair price.
- Quality at least 70%, Valuation at least 50% and Timing at least 50%: **strong candidate, build the position in tranches**.
- Quality at least 70%, Valuation at least 50%, Timing under 50%: **start about one third of a position**, add on pullbacks.
- Quality at least 70% but Valuation under 50%: **good business, not cheap enough**; put it on a watchlist.
- Valuation at least 50% but Quality under 50%: **value-trap / falling-knife risk**.
- Anything else: **not a long-term buy point yet**.

**Quality checks (is it worth owning?)**
- **ROIC at least 10%:** operating profit after tax divided by invested capital (debt plus equity). How much profit each dollar of capital earns; sustained high ROIC usually signals a competitive advantage.
- **Operating margin not shrinking (change at least -2 points):** pricing power. Falling margins suggest rising costs or competition.
- **Revenue CAGR at least 5%:** the business is growing faster than inflation.
- **EPS up in at least 60% of years:** per-share earnings grow consistently, including the effect of buybacks.
- **FCF positive every year:** the business produces real cash, not just accounting profit.
- **Net debt / EBITDA at most 3x:** years of earnings needed to repay net debt. Net cash passes; negative EBITDA fails.
- **Interest coverage at least 5x:** operating profit divided by interest expense; a cushion if earnings fall.
- **Stock-based compensation at most 20% of FCF:** high stock pay quietly dilutes you and flatters cash flow.
- **No heavy dilution:** the share count has not grown more than 5% in 2 years.

**Valuation checks (is the price attractive?)**
- **Price at or below the EPS-model buy price (10Y)** and **at or below the DCF buy price (10Y):** use the growth, multiple, return and margin-of-safety inputs from the Valuation tab, so they are only as good as your assumptions.
- **P/E at or below the 40th percentile of its own history:** the stock is cheaper than it usually is.
- **FCF yield at least 4%:** free cash flow divided by market cap. About the cash return if you bought the whole company; 4% equals paying 25 times FCF.
- **Forward P/E below trailing P/E:** analysts expect earnings to grow.
- **PEG at most 2:** P/E relative to expected growth (from Yahoo); how much you pay for each unit of growth.

**Timing checks (is this a sensible moment?)**
- **Price within 5% of or above the 200-day average:** the long-term trend is not broken (avoids falling knives).
- **At least 10% below the 52-week high:** a real pullback rather than chasing a rally.
- **Weekly RSI below 50:** momentum has cooled.
- **Not collapsing versus SPY:** the 6-month return is not more than 15 points behind the S&P 500 ETF.
- **4-hour triggers, at least 2 of 3:** the Charts tiles (Below BB Low, RSI under 30, MACD above Signal).

**Banks and insurers (Financial Services):** ROIC, leverage, interest coverage, FCF and DCF checks show N/A, because debt and cash are part of how they operate. Use the EPS model for them.
""")

        with st.expander("Financial metrics tab: the business behind the ticker"):
            st.markdown("""
**What it helps you do:** see the company's profitability, growth, share count, cash flow and debt in one list, and compare its price multiples with its sector. All values come from Yahoo's annual financial statements. P/E figures use reported (GAAP) profit, which can include one-off items and may differ from the adjusted earnings you see elsewhere.

- **Current price (latest close):** the last 4-hour close, shown for reference.
- **P/E (3Y Avg):** market cap divided by the average net income of the last 3 fiscal years. Smooths out one unusual year. Lower means cheaper, but check that profit is not collapsing.
- **P/E (1Y):** market cap divided by the latest year's net income. If P/E (1Y) is well below P/E (3Y Avg), recent profit is above its average (or the price fell).
- **Revenue Growth (2Y %):** the percent change in sales from two years ago to the latest year. Growing sales are the foundation of long-term returns.
- **Profit Growth (2Y vs Today):** Up, Down or Flat comparing net income today with two years ago.
- **Shares Outstanding (2Y):** Issuing Shares, Buying Back or Neutral, with the percent change. More than 5% up is dilution (your slice shrinks); more than 5% down means buybacks (your slice grows).
- **Div Rate (FWD):** expected annual dividend per share, when the company pays one.
- **FCF Multiple (MC/FCF):** market cap divided by free cash flow. Lower is cheaper. 20 means you pay 20 dollars per 1 dollar of yearly free cash flow (a 5% FCF yield).
- **Market Cap:** share price times shares outstanding, shown in billions (B) or millions (M).
- **Sector PE:** the trailing P/E of the sector fund (for example XLK for technology) matching the company's sector. It is an approximate benchmark, not a true peer average.
- **EPS (1Y):** net income divided by shares for the latest year. This is the starting point (M0) of the EPS valuation model.
- **Net Debt:** Yahoo's figure for total debt (including leases) minus cash and short-term investments. A negative number means more cash than debt (net cash).
- **Sector:** Yahoo's sector label; it also decides whether debt and cash-flow checks apply (they do not for financial companies).
- **Shares Latest:** the share count used for EPS and for the DCF per-share value.
- **FCF (1Y):** free cash flow for the latest year, from Yahoo's cash flow statement, or operating cash flow minus capital spending when that line is missing.
""")

        with st.expander("Valuation tab: what it could be worth and the most to pay"):
            st.markdown("""
**What it helps you do:** turn your own beliefs about growth and future valuation into a **fair value** and a **buy price**, and see whether today's price is in the buy zone. The answers depend entirely on your inputs, so treat them as scenarios, not predictions.

**The four inputs**
- **Growth g (%):** how fast you expect earnings (and free cash flow) to grow each year. Staying above 10% for ten years is rare.
- **Terminal K:** the multiple the market pays for earnings (or cash flow) at the end of the period. 15 to 20 is typical for a mature company.
- **Return r (%):** the yearly return you demand; the rate used to bring future values back to today. Default: S&P 500 trailing 3-year annualized return plus 2%, capped at 15%, so the hurdle does not balloon after a strong market.
- **MoS (%):** margin of safety, a cushion for being wrong. A 25% margin means you only buy at 75% of fair value.

Defaults are suggested per stock: g is the median of its past revenue, EPS and FCF growth (kept between 3% and 10%); K is its own median P/E (between 12 and 22, capped at 15 for leveraged or cash-burning companies); MoS is 25% (15% for high-quality businesses, 35% for leveraged or negative-FCF ones). Edit any input; **Reset inputs to suggested** restores them. The caption under the inputs also shows the stock's historical growth to keep g realistic.

**EPS-based model (1, 5 and 10 years)**
- **M0 (EPS):** last fiscal year earnings per share, the starting point.
- **Price Year X** = EPS x (1 + g)^X x K. What the stock could cost in X years under your assumptions.
- **Fair Value Today** = Price Year X / (1 + r)^X. The most you could pay today and still earn your required return r.
- **Buy Price (with MOS)** = Fair Value Today x (1 - MoS).
- If EPS is zero or negative this model is not meaningful.

**DCF model (10 years)**
- Takes last year's free cash flow, grows it at g for 10 years, adds a terminal value (year-10 cash flow x K), and brings everything back to today at r. This gives the value of the whole business (enterprise value).
- **Equity value** = enterprise value - net debt. **DCF Intrinsic** = equity value / shares. **DCF Buy** = intrinsic x (1 - MoS).
- Here K is a multiple of free cash flow, while in the EPS model it is a multiple of earnings; the same K input drives both.
- Not shown for banks and insurers, or when free cash flow is negative.

**Is the current price in the buy zone?** One row per model:
- **Fair value / Buy price / Current price:** the values being compared.
- **Upside to fair value %:** fair value divided by the current price, minus 1. Negative means the price is above fair value under your assumptions.
- **Status:** **Buy zone** (price at or below Buy price), **Between buy price and fair value**, or **Above fair value**.

**Sensitivity grid:** the 10-year Buy price for several growth rates (rows) and terminal multiples (columns), for the model you choose. **Green** cells mean today's price is at or below that buy price; **red** means above. If most of the grid is red, the stock needs optimistic assumptions to look cheap.
""")

        with st.expander("Paper investing check tab: test the timing rules on history"):
            st.markdown("""
**What it helps you do:** see how the app's own technical buy and sell rules would have performed on this stock over the last 1-2 years, before you rely on them.

**Inputs**
- **Number of years:** 1 or 2 (Yahoo only provides about 2 years of hourly data).
- **Money invested per buy:** the amount added on every buy signal (default 1,000 dollars).
- **Money taken out (%):** the share of your current position sold on every sell signal (default 10%).

**Rules (4-hour bars)**
- **Buy** when the close is below the lower Bollinger Band, RSI is under 30, **and** MACD has just crossed above its Signal line.
- **Sell** when the close is above the upper Bollinger Band, RSI is over 70, **and** MACD has just crossed below its Signal line (only if you hold shares).

**Results**
- **Realized gain:** profit locked in by sales (sale proceeds minus the cost of the shares sold).
- **Unrealized gain:** profit or loss on shares you still hold, at the latest price.
- **Rate of return:** (final wealth - total invested) / total invested. Final wealth is cash from sales plus the value of shares still held.
- **Total invested:** the sum of all buys. **Buy / sell events** lists every trade.

**Read it with care:** the strict rules fire only occasionally, so a few trades prove little. The test ignores fees, and it does not compare with simply buying and holding. Past results do not predict future ones. The first indicator values need about 200 bars, so the earliest weeks of data produce no signals.
""")

        with st.expander("Analyst view tab: what professional analysts think"):
            st.markdown("""
**What it helps you do:** see the Wall Street consensus (ratings, price targets, estimate changes) and whether it is improving or deteriorating, as a cross-check on your own view. It is third-party opinion from **Yahoo Finance**, not a recommendation from this app, and it is **not** part of the Long-term checklist score.

**What the tab shows**
- **Yahoo consensus:** Yahoo's own label (for example Buy).
- **Avg rating (1-5):** the average analyst rating, where 1 is Strong Buy, 3 is Hold and 5 is Strong Sell, calculated from the rating counts in the chart.
- **Analysts:** how many analysts cover the stock.
- **Mean target and Target range:** the average 12-month price target (with the percent difference from today's price) and the lowest-to-highest targets.
- **How the verdict is built table:** each of the four signals, its value, and its score (+1, 0 or -1), so you can see exactly what drove the verdict.
- **Ratings chart:** how many analysts say Strong Buy / Buy / Hold / Sell / Strong Sell now and over the last 3 months. Look for ratings drifting up or down.
- **Price target chart:** the analyst range with the mean, median and the current price marked.
- **Estimates expander:** EPS and revenue forecasts for the current and next quarter and year, plus how many analysts raised or cut their EPS estimates.
- **Recent analyst actions:** the latest upgrades, downgrades and target changes by firm.
- **Data source status:** which Yahoo lookups worked. Each source is fetched separately, so one failure does not blank the tab. If everything fails (usually a temporary Yahoo limit) a message and a **Retry** button appear, and failures are never cached.

**How the verdict is calculated.** Four signals each score +1 (bullish), 0 (neutral) or -1 (bearish):
1. **Consensus rating:** average of 2.3 or lower = +1; 3.3 or higher = -1; otherwise 0.
2. **Upside to mean target:** (mean target / current price) - 1. +10% or more = +1; -5% or lower = -1; otherwise 0.
3. **Rating trend (3M):** change in the share of Buy and Strong Buy ratings versus 3 months ago. +5 points or more = +1; -5 points or less = -1; otherwise 0.
4. **EPS revisions:** change in the current-year EPS estimate versus 90 days ago (30 days if 90 is missing; capped at plus or minus 100%). +1% or more = +1; -1% or less = -1; otherwise 0.

**Total of +2 or higher = Potential Buy. -2 or lower = Potential Sell. Anything between = Neutral.** With fewer than 2 usable signals the result is **Insufficient data**; N/A signals are ignored.

**Reading it correctly**
- Yahoo's label can disagree with the verdict. A stock most analysts rate Buy can still score Neutral because its price is already near the average target or ratings are slipping. That disagreement is useful information.
- Analyst ratings skew positive (few Sells), so Hold is closer to a soft negative. The trend and the estimate revisions matter more than the label.
- Targets and ratings lag the market, and small companies may have few or no analysts.
""")

        with st.expander("Favorites tab: your shortlist"):
            st.markdown("""
**What it helps you do:** keep the stocks you follow one click away instead of searching the dropdown every time.
- **Add current stock to Favorites** saves the stock you are viewing. Favorites then appear at the top of the ticker dropdown.
- **Load** opens that stock in every tab. **Remove** deletes it from the list.
- **Important:** Favorites are kept only for your current browser session. Refreshing the page or opening the app in a new window starts with an empty list, so note down your tickers if you want to keep them.
""")

        with st.expander("Data sources, limits and caveats"):
            st.markdown("""
- **One source:** everything comes from Yahoo Finance. Values can be delayed, revised or missing, and different websites may show different numbers for the same metric (for example adjusted versus reported earnings).
- **Limited history:** Yahoo provides about 4 fiscal years of statements, so growth rates cover roughly 3 years, and hourly price data covers 2 years at most.
- **Reported (GAAP) profit:** P/E, EPS and growth use reported net income, which one-off items can distort.
- **Rules of thumb:** every threshold (for example ROIC at least 10%) is a common guideline, not a law. Different industries have different norms.
- **Assumption-driven models:** fair value and buy price depend on the growth, multiple and return you choose. Change them to see the range; do not rely on a single number.
- **Banks and insurers** need different measures (book value, return on equity); several checks show N/A for them.
- **Rate limits:** Yahoo can temporarily block heavy usage on shared servers. If something shows N/A or an error, wait a minute and reload.
- **Not investment advice.** This app helps you organize research; the decision and the risk are yours.
""")

        st.caption("Data source: Yahoo Finance. Not investment advice.")


if __name__ == "__main__":
    main()
