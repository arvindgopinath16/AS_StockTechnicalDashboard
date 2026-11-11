import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

import ai_assistant
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
    # Explicit dates (not period=) because yfinance mishandles period= for recently listed stocks such as RDDT.
    period_days = min(700, max(30, int(years * 365)))
    start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = yf.download(
                stock,
                start=start_date,
                end=end_date,
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


def calculate_indicators_daily(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Daily bars with Bollinger (20, 2), RSI (10) and MACD (8/21/5). Used by the paper test for periods longer than
    Yahoo's 2-year hourly-data limit."""
    data = normalize_ohlcv_columns(raw_data)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.dropna()
    bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()
    data["RSI"] = RSIIndicator(data["Close"], window=10).rsi()
    macd = MACD(data["Close"], window_slow=21, window_fast=8, window_sign=5)
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    return data.dropna()


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
    lookback: int = 1,
) -> dict:
    """
    Simulate paper trading on bars that already carry BB / RSI / MACD columns (4h or daily).
    Buy: MACD crosses above Signal on this bar, and (Close < BB_Low and RSI < 30) on this bar or any of the previous
    lookback - 1 bars. lookback = 1 is the strict rule: all three on the same bar.
    Sell: the mirror image (Close > BB_High and RSI > 70, MACD crosses below Signal); sell pct_taken_out of position.
    """
    lb = max(1, int(lookback))
    oversold = ((df_4h["Close"] < df_4h["BB_Low"]) & (df_4h["RSI"] < 30)).astype(int).rolling(lb, min_periods=1).max().astype(bool).to_numpy()
    overbought = ((df_4h["Close"] > df_4h["BB_High"]) & (df_4h["RSI"] > 70)).astype(int).rolling(lb, min_periods=1).max().astype(bool).to_numpy()
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

        # Buy: oversold (below BB low and RSI < 30) now or within the window, MACD just crossed above Signal
        buy_signal = (
            bool(oversold[i])
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
            and bool(overbought[i])
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


_NEG_CACHE: dict = {}  # (function, symbol) -> time of last failure; keeps a failing lookup from being retried on every click
_NEG_TTL = 120.0


def _guarded(fn, stock: str, empty):
    """Call a cached fetcher that RAISES when Yahoo fails (so st.cache_data never stores the failure).
    On failure return empty() and skip retrying that symbol for a couple of minutes."""
    key = (getattr(fn, "__name__", str(fn)), stock)
    t = _NEG_CACHE.get(key)
    if t is not None and time.time() - t < _NEG_TTL:
        return empty()
    try:
        return fn(stock)
    except Exception:
        _NEG_CACHE[key] = time.time()
        return empty()


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
                lambda x: "N/A" if pd.isna(x) else (str(x) if not pd.api.types.is_number(x) else (f"{x / 1e9:.2f}B" if money and abs(x) >= 1e6 else f"{x:.2f}"))
            )
    return out.rename(columns={"avg": "Average", "low": "Low", "high": "High", "yearAgoEps": "Year ago EPS",
                               "yearAgoRevenue": "Year ago revenue", "numberOfAnalysts": "# analysts", "growth": "Growth"})


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_financial_metrics_cached(stock: str) -> dict:
    """
    Mirror the financial-metric logic from StockPickerComprehensive_v6.check_stocks
    for a single ticker so the web app matches the report.
    """
    ticker = yf.Ticker(stock)

    # Try to get the three core statements
    def _load_statements():
        fin_raw = ticker.financials
        return (
            fin_raw.T if isinstance(fin_raw, pd.DataFrame) else pd.DataFrame(),
            ticker.balance_sheet.T,
            ticker.cashflow.T,
        )

    try:
        financials, balance_sheet, cashflow = _retry(_load_statements, tries=2)
    except Exception as exc:
        raise RuntimeError(f"Yahoo statements unavailable: {exc}")

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
        info = _retry(lambda: ticker.info, tries=2) or {}
    except Exception as exc:
        raise RuntimeError(f"Yahoo company info unavailable: {exc}")
    if (
        financials.empty and balance_sheet.empty and cashflow.empty
        and str(info.get("quoteType") or "").upper() == "EQUITY"
    ):
        raise RuntimeError("Yahoo returned no statements for this company")

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
        "Industry": info.get("industry"),
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


def fetch_financial_metrics(stock: str) -> dict:
    """Company ratios from Yahoo statements. Returns {} when Yahoo fails (failures are never cached)."""
    return _guarded(_fetch_financial_metrics_cached, stock, dict)


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
def _fetch_quality_metrics_cached(stock: str) -> dict:
    """Business quality and balance-sheet strength from Yahoo annual statements (about 4 fiscal years)."""
    out: dict = {"available": False}
    t = yf.Ticker(stock)
    try:
        inc, bs, cf = _retry(lambda: (t.income_stmt, t.balance_sheet, t.cashflow), tries=2)
    except Exception as exc:
        raise RuntimeError(f"Yahoo statements unavailable: {exc}")
    try:
        info = _retry(lambda: t.info, tries=2) or {}
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
        if str(info.get("quoteType") or "").upper() == "EQUITY":
            raise RuntimeError("Yahoo returned no statements for this company")
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

    shares_s = _get_row(bs, ["Share Issued", "Ordinary Shares Number"])
    series = pd.DataFrame(
        {
            "Revenue": rev_y / 1e9,
            "Net income": ni_y / 1e9,
            "Free cash flow": fcf_y / 1e9,
            "Shares (billions)": _by_year(shares_s) / 1e9,
        }
).sort_index().dropna(how="all")
    roe = _safe_float(info.get("returnOnEquity"))
    roa = _safe_float(info.get("returnOnAssets"))
    out.update(
        {
            "available": True,
            "table": table,
            "series": series,
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


def fetch_quality_metrics(stock: str) -> dict:
    """Business quality from Yahoo statements. Returns {"available": False} when Yahoo fails (never cached)."""
    return _guarded(_fetch_quality_metrics_cached, stock, lambda: {"available": False})


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_valuation_context_cached(stock: str) -> dict:
    """Forward P/E, PEG, EV/EBITDA, 10Y yield, and the stock's own P/E history (annual EPS basis)."""
    out: dict = {
        "forward_pe": None, "peg": None, "ev_ebitda": None, "treasury_10y": None,
        "pe_hist": None, "pe_current": None, "pe_percentile": None,
        "pe_median": None, "pe_p25": None, "pe_p75": None, "pe_years": None,
    }
    try:
        t = yf.Ticker(stock)
        info = _retry(lambda: t.info, tries=2) or {}
    except Exception as exc:
        raise RuntimeError(f"Yahoo company info unavailable: {exc}")

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


def fetch_valuation_context(stock: str) -> dict:
    """Forward P/E, PEG and own P/E history. Returns an all-empty dict when Yahoo fails (never cached)."""
    return _guarded(
        _fetch_valuation_context_cached, stock,
        lambda: {
            "forward_pe": None, "peg": None, "ev_ebitda": None, "treasury_10y": None, "pe_hist": None, "pe_current": None,
            "pe_percentile": None, "pe_median": None, "pe_p25": None, "pe_p75": None, "pe_years": None,
        },
    )


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
        _chk(
            "4h triggers: >= 2 of BB low / RSI<30 / MACD>Signal",
            None if all(x is None for x in triggers) else n_trig >= 2,
            "N/A" if all(x is None for x in triggers) else f"{n_trig}/3",
        ),
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


STOCK_TYPES = ("Core", "Growth", "Cyclical", "Financial")
_CYCLICAL_INDUSTRY_WORDS = (
    "airline", "steel", "aluminum", "copper", "gold", "silver", "coal", "industrial metals", "marine shipping",
    "trucking", "residential construction", "oil & gas", "uranium",
)


def classify_stock_type(fin: dict, quality: dict) -> tuple:
    """(type, reason). Financial > Cyclical (sector / industry) > Growth (sales growing 12%+ a year) > Core."""
    fin, q = fin or {}, quality or {}
    sector = str(fin.get("Sector") or "").lower()
    industry = str(fin.get("Industry") or "").lower()
    if is_financial(fin):
        return "Financial", "bank, insurer or other financial company"
    if sector == "energy" or any(w in industry for w in _CYCLICAL_INDUSTRY_WORDS):
        label = fin.get("Industry") or fin.get("Sector") or "this industry"
        return "Cyclical", f"{label}: profits swing with the economic cycle"
    rev = q.get("rev_cagr")
    if rev is not None and rev >= 12.0:
        return "Growth", f"sales growing {rev:.1f}% a year (12% or more)"
    return "Core", "steady business (sales growth under 12% a year)"


def fade_growth_equivalent(g1_pct: float, terminal_pct: float = 5.0, hold_years: int = 5, total_years: int = 10) -> float:
    """Flat yearly rate (%) that gives the same 10-year growth as: g1 for `hold_years`, then a straight-line fade to the terminal rate."""
    factor = 1.0
    for t in range(1, total_years + 1):
        if t <= hold_years:
            g = g1_pct
        else:
            g = g1_pct - (g1_pct - terminal_pct) * (t - hold_years) / (total_years - hold_years)
        factor *= 1.0 + g / 100.0
    return (factor ** (1.0 / total_years) - 1.0) * 100.0


def suggest_valuation_defaults(fin: dict, quality: dict, valctx: dict, profile: str = None) -> dict:
    """Stock-specific starting assumptions: growth g (%), terminal multiple K, margin of safety (%).
    The stock type (Core / Growth / Cyclical / Financial) decides how g and K are limited."""
    q = quality or {}
    v = valctx or {}
    auto_type, why = classify_stock_type(fin, q)
    stype = profile if profile in STOCK_TYPES else auto_type
    if stype == auto_type:
        notes: list = [f"Stock type {stype} ({why})"]
    else:
        notes = [f"Stock type {stype} (your choice; detected {auto_type})"]

    growth = [x for x in (q.get("rev_cagr"), q.get("eps_cagr"), q.get("fcf_cagr")) if x is not None]
    raw_g = float(pd.Series(growth).median()) if growth else None
    rev = q.get("rev_cagr")

    if stype == "Growth":
        base = raw_g if raw_g is not None else (rev if rev is not None else 12.0)
        g1 = min(25.0, max(10.0, base))
        g = round(fade_growth_equivalent(g1) * 2) / 2.0
        cap_txt = ", capped at 25%" if base > 25.0 else ""
        notes.append(
            f"g {g:g}% = 10-year average of {g1:.0f}% a year for 5 years (historical median growth {base:.1f}%{cap_txt}) "
            "then fading to 5% by year 10"
        )
    elif stype == "Cyclical":
        base = rev if rev is not None else (raw_g if raw_g is not None else 3.0)
        g = round(min(5.0, max(2.0, base)) * 2) / 2.0
        notes.append(f"g {g:g}% (recent growth {base:.1f}% is not extrapolated for a cyclical business; limited to 2-5%)")
    elif stype == "Financial":
        base = raw_g if raw_g is not None else 5.0
        g = round(min(8.0, max(3.0, base)) * 2) / 2.0
        notes.append(f"g {g:g}% (historical median growth {base:.1f}%, limited to 3-8% for a financial company)")
    elif raw_g is not None:
        g = round(min(10.0, max(3.0, raw_g)) * 2) / 2.0
        if raw_g > 10.0:
            notes.append(f"g {g:g}% (historical median growth {raw_g:.1f}%, capped at 10% because a 10-year projection should not assume a hot streak lasts)")
        elif raw_g < 3.0:
            notes.append(f"g {g:g}% (historical median growth {raw_g:.1f}%, raised to the 3% floor)")
        else:
            notes.append(f"g {g:g}% = median of historical revenue / EPS / FCF growth ({raw_g:.1f}%)")
    else:
        g = 7.0
        notes.append("g 7% (generic; no growth history)")

    nd_ebitda = q.get("net_debt_ebitda")
    risky = (
        (nd_ebitda is not None and nd_ebitda > 3)
        or ((q.get("net_debt") or 0) > 0 and q.get("ebitda") is not None and q["ebitda"] <= 0)
        or (q.get("fcf_pos") is not None and q.get("fcf_n") and q["fcf_pos"] < q["fcf_n"])
    ) and not is_financial(fin)

    lo, hi, fallback = {"Core": (12.0, 22.0, 18.0), "Growth": (15.0, 30.0, 25.0), "Cyclical": (6.0, 14.0, 10.0), "Financial": (8.0, 14.0, 11.0)}[stype]
    pe_med = v.get("pe_median")
    if pe_med is not None:
        k = float(round(min(hi, max(lo, pe_med))))
        if pe_med > hi:
            notes.append(f"K {k:g} (own median P/E {pe_med:.1f}, capped at {hi:g} for this stock type)")
        elif pe_med < lo:
            notes.append(f"K {k:g} (own median P/E {pe_med:.1f}, raised to the {lo:g} floor for this stock type)")
        else:
            notes.append(f"K {k:g} = own median P/E ({pe_med:.1f})")
    else:
        k = fallback
        notes.append(f"K {k:g} (generic for this stock type; no P/E history)")
    if risky and k > 15.0:
        k = 15.0
        notes.append("K capped at 15 (leverage or negative FCF)")

    q_pct = pillar_score(build_long_term_checklist(0.0, fin, q, v, {}, [], {})["quality"])[0]
    if risky:
        mos = 35.0
        notes.append("MoS 35% (leverage or negative FCF)")
    elif stype == "Cyclical":
        mos = 35.0
        notes.append("MoS 35% (cyclical earnings can be at a peak)")
    elif q_pct is not None and q_pct >= 80:
        mos = 15.0
        notes.append("MoS 15% (high-quality business)")
    else:
        mos = 25.0
        notes.append("MoS 25% (default)")
    return {"g": g, "k": k, "mos": mos, "notes": "; ".join(notes), "profile": stype, "auto_profile": auto_type, "profile_reason": why}


def _reset_val_inputs(sugg: dict, r_default: float):
    st.session_state["val_g_pct"] = float(sugg["g"])
    st.session_state["val_k_multiple"] = float(sugg["k"])
    st.session_state["val_mos_pct"] = float(sugg["mos"])
    st.session_state["val_r_pct"] = float(r_default)


def _clear_data_caches():
    """Forget cached company data and any recent failures, so the next load asks Yahoo again."""
    for fn in (_fetch_financial_metrics_cached, _fetch_quality_metrics_cached, _fetch_valuation_context_cached, _fetch_analyst_raw,
               _fetch_news_cached, _fetch_market_snapshot_cached, _fetch_moves_cached, _score_or_raise, _ai_chart_data):
        try:
            fn.clear()
        except Exception:
            pass
    _NEG_CACHE.clear()


def render_data_gap_banner(stock: str):
    """Shown when Yahoo returned no company statements for an operating company: explains, offers a retry, and shows the last saved scan."""
    st.warning(
        f"Yahoo Finance did not return company statements for {stock} right now. This is usually a temporary limit on shared "
        "servers, not a problem with the stock. Nothing from the failed lookup is saved, so pressing Try again (or simply "
        "waiting a couple of minutes) fetches fresh data."
    )
    st.button("Try again", key="retry_company_data", on_click=_clear_data_caches)
    df, state = load_rankings()
    if df is None or df.empty:
        return
    m = df[(df["symbol"] == stock) & (df["error"] == "")]
    if m.empty:
        return
    r = m.iloc[0]
    with st.expander(f"Last saved scan for {stock} ({state.get('asof') or 'date unknown'})", expanded=True):
        c = st.columns(4)
        c[0].metric("Verdict then", r["verdict_short"] or "N/A")
        c[1].metric("Quality / Valuation / Timing", f"{_f(r['quality_pct'])} / {_f(r['valuation_pct'])} / {_f(r['timing_pct'])}")
        c[2].metric("Price then", _f(r["price"], "${:.2f}"))
        c[3].metric("Buy price then", _f(r["buy_eps10"], "${:.2f}"))
        st.caption("Numbers from the last daily scan, shown only because live data is unavailable. They may be a day or more old.")


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


# ---------------------------------------------------------------------------
# Favorites file, instrument type, profit-taking signals, scoring and rankings
# (shared by the Streamlit app and by scan_universe.py, the daily scanner)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RANKINGS_FILE = "rankings.csv"
STATE_FILE = "scan_state.json"
ETF_TYPES = {"ETF", "MUTUALFUND", "INDEX"}


def _data_path(name: str):
    """Find a data file next to app.py (or in the working directory), ignoring upper/lower case."""
    for folder in (BASE_DIR, Path.cwd()):
        try:
            for p in folder.iterdir():
                if p.is_file() and p.name.lower() == name.lower():
                    return p
        except OSError:
            continue
    return None


def _read_symbols(name: str) -> list:
    p = _data_path(name)
    if p is None:
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    if col is None:
        return []
    out: list = []
    for s in df[col].dropna().astype(str):
        s = s.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def load_favorites_file() -> list:
    """Your permanent favorites list (Favorites.csv, column 'Symbol')."""
    return _read_symbols("Favorites.csv")


def load_universe() -> list:
    """Everything the daily scan scores: Favorites first, then stockList.csv."""
    out: list = []
    for s in _read_symbols("Favorites.csv") + _read_symbols("stockList.csv"):
        if s not in out:
            out.append(s)
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_instrument_info(stock: str) -> dict:
    """Instrument type (EQUITY / ETF ...), name and sector. Raises when Yahoo returns nothing, so failures are not cached."""
    info = _retry(lambda: yf.Ticker(stock).info, tries=2) or {}
    qt = str(info.get("quoteType") or "").upper()
    if not qt:
        raise RuntimeError("no instrument info")
    return {
        "type": qt,
        "name": str(info.get("shortName") or info.get("longName") or ""),
        "sector": info.get("sector"),
    }


def get_instrument_info(stock: str) -> dict:
    try:
        return fetch_instrument_info(stock)
    except Exception:
        return {"type": "UNKNOWN", "name": "", "sector": None}


def compute_sell_signals(daily: pd.DataFrame) -> dict:
    """Profit-taking signals on daily bars (same rules as StockSeller_v2.py):
    close above the upper Bollinger Band (15, 2), RSI(10) above 70, MACD (8/21/5) below its signal line."""
    out = {"count": None, "above_bb": None, "rsi_hot": None, "macd_bearish": None, "rsi": None}
    if daily is None or daily.empty:
        return out
    try:
        close = normalize_ohlcv_columns(daily)["Close"].astype(float).dropna()
    except Exception:
        return out
    if len(close) < 40:
        return out
    bb = BollingerBands(close=close, window=15, window_dev=2)
    rsi = RSIIndicator(close, window=10).rsi()
    macd = MACD(close, window_slow=21, window_fast=8, window_sign=5)
    above = bool(close.iloc[-1] > bb.bollinger_hband().iloc[-1])
    hot = bool(rsi.iloc[-1] > 70)
    bear = bool(macd.macd().iloc[-1] < macd.macd_signal().iloc[-1])
    out.update(
        {"count": int(above) + int(hot) + int(bear), "above_bb": above, "rsi_hot": hot, "macd_bearish": bear, "rsi": float(rsi.iloc[-1])}
    )
    return out


def _above_fair(val_rows: list):
    """True when the price is above fair value in every available valuation model; None when there is no model."""
    if not val_rows:
        return None
    return all(r["Status"] == "Above fair value" for r in val_rows)


def trim_signal(sell_count, above_fair, q_pct, q_n) -> tuple:
    """Profit-taking level: Trim / Review / Watch / Hold, with a short reason."""
    weak = q_pct is not None and (q_n or 0) >= 4 and q_pct < 50
    n = sell_count or 0
    if n >= 3 and above_fair:
        return "Trim", "3/3 overbought signals and price above fair value"
    if weak:
        return "Review", f"Quality score only {q_pct:.0f}% - re-check the thesis"
    if n >= 2 or (above_fair and n >= 1):
        return "Watch", f"{n}/3 overbought signals" + (" and price above fair value" if above_fair else "")
    return "Hold", "No profit-taking signals"


def _short_verdict(headline: str) -> str:
    for start, short in (
        ("Strong long-term entry", "Strong candidate"),
        ("Good business at a fair price", "Good, fair price"),
        ("Quality business, but not cheap", "Quality, not cheap"),
        ("Cheap, but quality is weak", "Value-trap risk"),
        ("Not a long-term buy point", "Not yet"),
        ("Not enough data", "No data"),
        ("ETF", "ETF (timing only)"),
    ):
        if headline.startswith(start):
            return short
    return headline


def score_symbol(stock: str, r_default: float, bench=None) -> dict:
    """Run the full long-term checklist for one symbol with its suggested valuation defaults.
    Returns one flat row (used by the Top 10 lists tab, the My portfolio tab and the daily scan)."""
    row = {
        "symbol": stock, "name": "", "type": "UNKNOWN", "sector": "", "price": None, "price_date": "",
        "verdict": "", "verdict_short": "", "quality_pct": None, "quality_n": 0, "valuation_pct": None,
        "valuation_n": 0, "timing_pct": None, "timing_n": 0, "eligible": False, "buy_score": None,
        "avoid_score": None, "sell_signals": None, "rsi_daily": None, "above_fair": None, "upside_pct": None,
        "fair_eps10": None, "buy_eps10": None, "fair_dcf": None, "buy_dcf": None, "trim_level": "",
        "trim_reason": "", "g_pct": None, "k": None, "mos_pct": None, "buy_gap_pct": None, "buy_zone": "",
        "stock_type": "", "error": "",
    }
    try:
        inst = get_instrument_info(stock)
        row.update(type=inst["type"], name=inst["name"], sector=inst.get("sector") or "")
        daily = download_daily(stock)
        if daily is None or daily.empty:
            raise ValueError("no price history")
        if bench is None:
            bench = download_daily("SPY")
        trend = compute_trend_context(daily, bench)
        close = normalize_ohlcv_columns(daily)["Close"].astype(float).dropna()
        price = float(trend.get("price") or close.iloc[-1])
        row["price"] = price
        row["price_date"] = str(pd.to_datetime(close.index[-1]).date())

        sig: dict = {}
        try:
            ind = calculate_indicators(download_data(stock), stock)
            last = ind.iloc[-1]
            sig = {
                "below_bb_low": bool(last["Close"] < last["BB_Low"]),
                "rsi_extreme": bool(last["RSI"] < 30),
                "macd_crossover": bool(last["MACD"] > last["MACD_Signal"]),
            }
        except Exception:
            sig = {}

        is_etf = row["type"] in ETF_TYPES
        if is_etf:
            fin, quality, valctx, val_rows = {}, {}, {}, []
        else:
            fin = fetch_financial_metrics(stock) or {}
            quality = fetch_quality_metrics(stock) or {}
            valctx = fetch_valuation_context(stock) or {}
            if not fin and not quality.get("available"):
                raise ValueError("no company statements from Yahoo")
            sugg = suggest_valuation_defaults(fin, quality, valctx)
            row.update(g_pct=sugg["g"], k=sugg["k"], mos_pct=sugg["mos"], stock_type=sugg["profile"])
            val_rows = summarize_valuation(fin, price, sugg["g"] / 100.0, sugg["k"], r_default / 100.0, sugg["mos"] / 100.0)

        checklist = build_long_term_checklist(price, fin, quality, valctx, trend, val_rows, sig)
        scores = {k: pillar_score(v) for k, v in checklist.items()}
        q_pct, _, q_n = scores["quality"]
        v_pct, _, v_n = scores["valuation"]
        t_pct, _, t_n = scores["timing"]
        headline = "ETF - timing only" if is_etf else long_term_verdict(scores)[1]
        row.update(
            verdict=headline, verdict_short=_short_verdict(headline), quality_pct=q_pct, quality_n=q_n,
            valuation_pct=v_pct, valuation_n=v_n, timing_pct=t_pct, timing_n=t_n,
        )

        sell = compute_sell_signals(daily)
        above_fair = _above_fair(val_rows)
        level, why = trim_signal(sell["count"], above_fair, q_pct, q_n)
        row.update(sell_signals=sell["count"], rsi_daily=sell["rsi"], above_fair=above_fair, trim_level=level, trim_reason=why)

        by_model = {r["Model"]: r for r in val_rows}
        e10, d10 = by_model.get("EPS model (10Y)"), by_model.get("DCF (10Y)")
        if e10:
            row.update(fair_eps10=e10["Fair value"], buy_eps10=e10["Buy price"])
        if d10:
            row.update(fair_dcf=d10["Fair value"], buy_dcf=d10["Buy price"])
        ups = [r["Upside to fair value %"] for r in val_rows if r.get("Upside to fair value %") is not None]
        row["upside_pct"] = float(sum(ups) / len(ups)) if ups else None

        # Only operating companies with enough scored checks can be ranked
        row["eligible"] = bool(not is_etf and q_pct is not None and v_pct is not None and q_n >= 4 and v_n >= 3)
        buy_px = row.get("buy_eps10")
        if buy_px and buy_px > 0 and not is_etf and q_pct is not None and q_pct >= 50:
            gap = (price / buy_px - 1.0) * 100.0
            row["buy_gap_pct"] = gap
            row["buy_zone"] = "In zone" if gap <= 0 else ("Near" if gap <= 5 else "")
        if row["eligible"]:
            t_eff = 50.0 if t_pct is None else t_pct
            row["buy_score"] = 0.45 * q_pct + 0.40 * v_pct + 0.15 * t_eff
            overbought = 0.0 if sell["count"] is None else sell["count"] / 3.0 * 100.0
            row["avoid_score"] = 0.5 * (100.0 - v_pct) + 0.3 * (100.0 - q_pct) + 0.2 * overbought
    except Exception as exc:
        row["error"] = str(exc)[:120] or exc.__class__.__name__
    return row


@st.cache_data(ttl=1800, show_spinner=False)
def score_symbol_cached(stock: str, r_default: float) -> dict:
    return score_symbol(stock, r_default)


_NUM_COLS = (
    "price", "quality_pct", "quality_n", "valuation_pct", "valuation_n", "timing_pct", "timing_n", "buy_score",
    "avoid_score", "sell_signals", "rsi_daily", "upside_pct", "fair_eps10", "buy_eps10", "fair_dcf", "buy_dcf",
    "g_pct", "k", "mos_pct", "buy_gap_pct",
)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("buy_zone", "stock_type"):
        if c not in out.columns:
            out[c] = ""
    if "buy_gap_pct" not in out.columns:
        out["buy_gap_pct"] = float("nan")
    for c in _NUM_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "eligible" in out.columns:
        out["eligible"] = out["eligible"].map(lambda x: str(x).strip().lower() == "true" if not isinstance(x, bool) else x)
    for c in ("name", "sector", "verdict", "verdict_short", "trim_level", "trim_reason", "error", "type", "price_date", "buy_zone", "stock_type"):
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out


def rank_results(df: pd.DataFrame, n: int = 10) -> tuple:
    """Top-n buy candidates and top-n avoid / overvalued names among rankable operating companies.
    Buy: Quality >= 50%, ordered by 0.45 Quality + 0.40 Valuation + 0.15 Timing.
    Avoid: Valuation <= 33% or Quality < 50%, ordered by 0.5 (100 - Valuation) + 0.3 (100 - Quality) + 0.2 Overbought."""
    empty = pd.DataFrame()
    if df is None or df.empty:
        return empty, empty
    d = clean_results(df)
    ok = d[d["eligible"]]
    if ok.empty:
        return empty, empty
    buy = ok[ok["quality_pct"] >= 50].sort_values(["buy_score", "valuation_pct"], ascending=False).head(n)
    rest = ok[~ok["symbol"].isin(buy["symbol"])]
    avoid = rest[(rest["valuation_pct"] <= 33.4) | (rest["quality_pct"] < 50)].sort_values("avoid_score", ascending=False).head(n)
    return buy, avoid


def _f(x, fmt: str = "{:.0f}", suffix: str = "") -> str:
    try:
        if x is None or pd.isna(x):
            return "-"
        return fmt.format(x) + suffix
    except Exception:
        return "-"


@st.cache_data(ttl=300, show_spinner=False)
def load_rankings():
    """(results DataFrame or None, scan state dict) from the files written by scan_universe.py."""
    df, state = None, {}
    p = _data_path(RANKINGS_FILE)
    if p is not None:
        try:
            df = clean_results(pd.read_csv(p))
        except Exception:
            df = None
    s = _data_path(STATE_FILE)
    if s is not None:
        try:
            state = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    return df, state


def _rank_view(df: pd.DataFrame, favs: set, new_syms: set) -> pd.DataFrame:
    rows = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        sym = r["symbol"]
        rows.append(
            {
                "#": i,
                "Symbol": f"{sym} \u2605" if sym in favs else sym,
                "Name": (r["name"] or "")[:26],
                "Verdict": r["verdict_short"],
                "Q / V / T %": f"{_f(r['quality_pct'])} / {_f(r['valuation_pct'])} / {_f(r['timing_pct'])}",
                "Price": _f(r["price"], "{:.2f}"),
                "Upside to fair": _f(r["upside_pct"], "{:+.0f}", "%"),
                "Overbought": "-" if pd.isna(r["sell_signals"]) else f"{int(r['sell_signals'])}/3",
                "New": "NEW" if sym in new_syms else "",
            }
        )
    return pd.DataFrame(rows)


def render_rankings_tab(favorites: list, r_default: float):
    st.subheader("Top 10 lists: best to buy and most to avoid")
    file_df, state = load_rankings()
    live_df = st.session_state.get("live_rankings")
    favs = set(favorites)

    df, source = (live_df, "live scan of your favorites (this session)") if live_df is not None else (file_df, "daily scan")
    if df is None or df.empty:
        st.info(
            "No scan results yet. The daily GitHub scan writes rankings.csv, or you can run a live scan of your "
            "Favorites below (takes several minutes)."
        )
    else:
        asof = state.get("asof", "") if source == "daily scan" else "now"
        n_ok = int((df["error"] == "").sum())
        st.caption(
            f"Source: {source}. As of {asof or 'unknown'}. {n_ok} of {len(df)} symbols scored, "
            f"{int(df['eligible'].sum())} rankable (operating companies with enough data). \u2605 = in your Favorites."
        )
        buy, avoid = rank_results(df)
        new_buy = set(state.get("new_buy", [])) if source == "daily scan" else set()
        new_avoid = set(state.get("new_avoid", [])) if source == "daily scan" else set()

        hits_b = [s for s in buy["symbol"] if s in favs] if not buy.empty else []
        hits_a = [s for s in avoid["symbol"] if s in favs] if not avoid.empty else []
        if hits_b:
            st.success("Your favorites in the Top 10 Buy list: " + ", ".join(hits_b))
        if hits_a:
            st.warning("Your favorites in the Top 10 Avoid list: " + ", ".join(hits_a))
        if not hits_b and not hits_a:
            st.info("None of your favorites are in either top 10 list right now.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 buy candidates**")
            if buy.empty:
                st.info("No rankable candidates.")
            else:
                st.dataframe(_rank_view(buy, favs, new_buy), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Top 10 avoid / overvalued**")
            if avoid.empty:
                st.info("No names meet the avoid criteria.")
            else:
                st.dataframe(_rank_view(avoid, favs, new_avoid), hide_index=True, use_container_width=True)

        fav_df = df[df["symbol"].isin(favs)].copy()
        new_zone = set(state.get("new_buy_zone", [])) if source == "daily scan" else set()
        st.markdown("**Buy-price watch (your favorites)**")
        zone = fav_df[fav_df["buy_zone"].isin(["In zone", "Near"])].copy()
        if zone.empty:
            st.info("No favorites are at, or within 5% above, their Buy price right now.")
        else:
            zone = zone.sort_values("buy_gap_pct")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Symbol": zone["symbol"],
                        "Status": zone["buy_zone"].map({"In zone": "At or below Buy price", "Near": "Within 5% above"}),
                        "Price": zone["price"].map(lambda x: _f(x, "{:.2f}")),
                        "Buy price": zone["buy_eps10"].map(lambda x: _f(x, "{:.2f}")),
                        "Gap to Buy price": zone["buy_gap_pct"].map(lambda x: _f(x, "{:+.1f}", "%")),
                        "Quality %": zone["quality_pct"].map(_f),
                        "Stock type": zone["stock_type"],
                        "Verdict": zone["verdict_short"],
                        "New": zone["symbol"].map(lambda s: "NEW" if s in new_zone else ""),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.caption(
            "Buy price = the 10-year EPS-model buy price (fair value less the margin of safety) for that stock type. "
            "Only stocks with a Quality score of 50% or more are listed. Crossing below the Buy price triggers an alert."
        )
        st.markdown("**Profit-taking watch (your favorites)**")
        order = {"Trim": 0, "Review": 1, "Watch": 2, "Hold": 3}
        watch = fav_df[fav_df["trim_level"].isin(["Trim", "Review", "Watch"])].copy()
        if watch.empty:
            st.info("No favorites are showing profit-taking signals.")
        else:
            watch["_o"] = watch["trim_level"].map(order)
            watch = watch.sort_values(["_o", "sell_signals"], ascending=[True, False])
            st.dataframe(
                pd.DataFrame(
                    {
                        "Symbol": watch["symbol"],
                        "Signal": watch["trim_level"],
                        "Why": watch["trim_reason"],
                        "Price": watch["price"].map(lambda x: _f(x, "{:.2f}")),
                        "Upside to fair": watch["upside_pct"].map(lambda x: _f(x, "{:+.0f}", "%")),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("All favorites: scores and signals", expanded=False):
            if fav_df.empty:
                st.info("None of your favorites are in these results.")
            else:
                fav_df = fav_df.sort_values("buy_score", ascending=False, na_position="last")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Symbol": fav_df["symbol"],
                            "Type": fav_df["type"],
                            "Verdict": fav_df["verdict_short"],
                            "Quality %": fav_df["quality_pct"].map(_f),
                            "Valuation %": fav_df["valuation_pct"].map(_f),
                            "Timing %": fav_df["timing_pct"].map(_f),
                            "Price": fav_df["price"].map(lambda x: _f(x, "{:.2f}")),
                            "Upside to fair": fav_df["upside_pct"].map(lambda x: _f(x, "{:+.0f}", "%")),
                            "Profit-taking": fav_df["trim_level"],
                            "Note": fav_df["error"],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    with st.expander("Scan my Favorites now (live, slow)", expanded=False):
        st.caption("Scores every symbol in your Favorites list with the same rules. Expect roughly 5-10 seconds per stock.")
        if st.button("Run live scan of my Favorites", key="run_live_scan"):
            results = []
            bar = st.progress(0.0, text="Starting...")
            bench = download_daily("SPY")
            for i, sym in enumerate(favorites, start=1):
                bar.progress(i / max(len(favorites), 1), text=f"Scoring {sym} ({i}/{len(favorites)})")
                results.append(score_symbol(sym, r_default, bench))
            st.session_state["live_rankings"] = clean_results(pd.DataFrame(results))
            bar.empty()
            st.rerun()
        if live_df is not None and st.button("Clear live results", key="clear_live_scan"):
            st.session_state.pop("live_rankings", None)
            st.rerun()
    st.caption(
        "How the lists are built: see the How it works tab. These lists are a screening aid, not advice, and depend on the "
        "default valuation assumptions for each stock."
    )


def portfolio_action(level: str, gain_pct, weight_pct) -> str:
    heavy = weight_pct is not None and weight_pct > 15.0
    if level == "Trim":
        if gain_pct is not None and gain_pct > 0:
            text = "Consider trimming 10-25% to lock in part of the gain"
        else:
            text = "Overbought and above fair value, but you are at a loss: review before selling"
    elif level == "Review":
        text = "Quality has weakened: re-check why you own it"
    elif level == "Watch":
        text = "Keep watching; decide your trim level in advance"
    else:
        text = "Hold"
    if heavy:
        text += "; position is over 15% of the portfolio, consider rebalancing"
    return text


def render_portfolio_tab(r_default: float):
    st.subheader("Portfolio: your positions and profit-taking signals")
    st.caption(
        "Enter your positions below. They are kept only in this browser session and are never saved to GitHub. "
        "Download the CSV to keep a copy on your computer, and upload it next session."
    )
    if "pf_base" not in st.session_state:
        st.session_state["pf_base"] = pd.DataFrame({"Symbol": pd.Series(dtype=str), "Shares": pd.Series(dtype=float), "Avg cost": pd.Series(dtype=float)})
        st.session_state["pf_version"] = 0

    up = st.file_uploader("Load saved positions (CSV with columns Symbol, Shares, Avg cost)", type="csv", key="pf_upload")
    if up is not None and st.session_state.get("pf_loaded") != (up.name, up.size):
        try:
            loaded = pd.read_csv(up)
            loaded.columns = [str(c).strip() for c in loaded.columns]
            colmap = {c.lower().replace("_", " "): c for c in loaded.columns}
            loaded = loaded.rename(columns={colmap.get("symbol", "Symbol"): "Symbol", colmap.get("shares", "Shares"): "Shares", colmap.get("avg cost", "Avg cost"): "Avg cost"})
            st.session_state["pf_base"] = loaded[["Symbol", "Shares", "Avg cost"]]
            st.session_state["pf_version"] += 1
            st.session_state["pf_loaded"] = (up.name, up.size)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")

    edited = st.data_editor(
        st.session_state["pf_base"],
        num_rows="dynamic",
        use_container_width=True,
        key=f"pf_editor_{st.session_state['pf_version']}",
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0, format="%.4f"),
            "Avg cost": st.column_config.NumberColumn("Avg cost per share", min_value=0.0, step=0.01, format="%.2f"),
        },
    )

    pos = edited.copy()
    pos["Symbol"] = pos["Symbol"].fillna("").astype(str).str.strip().str.upper()
    pos["Shares"] = pd.to_numeric(pos["Shares"], errors="coerce")
    pos["Avg cost"] = pd.to_numeric(pos["Avg cost"], errors="coerce")
    pos = pos[(pos["Symbol"] != "") & (pos["Shares"] > 0) & (pos["Avg cost"] > 0)]
    if pos.empty:
        st.info("Add at least one position (symbol, shares and average cost) to see gains and profit-taking signals.")
        return
    pos["Cost total"] = pos["Shares"] * pos["Avg cost"]
    pos = pos.groupby("Symbol", as_index=False).agg({"Shares": "sum", "Cost total": "sum"})
    pos["Avg cost"] = pos["Cost total"] / pos["Shares"]

    st.download_button(
        "Download positions as CSV",
        data=pos[["Symbol", "Shares", "Avg cost"]].to_csv(index=False),
        file_name="my_positions.csv",
        mime="text/csv",
    )

    rows = []
    with st.spinner("Scoring your positions..."):
        for _, p in pos.iterrows():
            s = score_symbol_cached(p["Symbol"], r_default)
            price = s.get("price")
            if price is None:
                rows.append({"Symbol": p["Symbol"], "Shares": p["Shares"], "Avg cost": p["Avg cost"], "Price": None, "Value": None,
                             "Cost total": p["Cost total"], "Sector": s.get("sector") or s.get("type"), "Level": "", "Reason": s.get("error") or "No price"})
                continue
            value = p["Shares"] * price
            rows.append(
                {
                    "Symbol": p["Symbol"], "Shares": p["Shares"], "Avg cost": p["Avg cost"], "Price": price, "Value": value,
                    "Cost total": p["Cost total"], "Sector": s.get("sector") or s.get("type") or "",
                    "Verdict": s.get("verdict_short", ""), "Level": s.get("trim_level", ""), "Reason": s.get("trim_reason", ""),
                    "Upside": s.get("upside_pct"),
                }
            )
    tbl = pd.DataFrame(rows)
    valued = tbl.dropna(subset=["Value"])
    total_value = float(valued["Value"].sum())
    total_cost = float(valued["Cost total"].sum())
    gain = total_value - total_cost
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio value", f"${total_value:,.0f}")
    m2.metric("Total gain", f"${gain:,.0f}", f"{(gain / total_cost * 100.0) if total_cost else 0.0:+.1f}%")
    m3.metric("Positions", f"{len(tbl)}")

    tbl["Gain $"] = tbl["Value"] - tbl["Cost total"]
    tbl["Gain %"] = (tbl["Value"] / tbl["Cost total"] - 1.0) * 100.0
    tbl["Weight %"] = tbl["Value"] / total_value * 100.0 if total_value else None
    tbl["Suggested action"] = [
        portfolio_action(r["Level"], None if pd.isna(r["Gain %"]) else r["Gain %"], None if pd.isna(r["Weight %"]) else r["Weight %"])
        if r["Level"] else r["Reason"]
        for _, r in tbl.iterrows()
    ]
    order = {"Trim": 0, "Review": 1, "Watch": 2, "Hold": 3, "": 4}
    tbl["_o"] = tbl["Level"].map(order)
    tbl = tbl.sort_values(["_o", "Gain %"], ascending=[True, False])
    show = pd.DataFrame(
        {
            "Symbol": tbl["Symbol"],
            "Shares": tbl["Shares"].map(lambda x: _f(x, "{:,.2f}")),
            "Avg cost": tbl["Avg cost"].map(lambda x: _f(x, "{:,.2f}")),
            "Price": tbl["Price"].map(lambda x: _f(x, "{:,.2f}")),
            "Value": tbl["Value"].map(lambda x: _f(x, "${:,.0f}")),
            "Gain %": tbl["Gain %"].map(lambda x: _f(x, "{:+.1f}", "%")),
            "Weight %": tbl["Weight %"].map(lambda x: _f(x, "{:.1f}", "%")),
            "Signal": tbl["Level"],
            "Suggested action": tbl["Suggested action"],
        }
    )

    def _lvl_color(v):
        return {
            "Trim": "background-color: #f8d7da; color: #721c24",
            "Review": "background-color: #fff3cd; color: #856404",
            "Watch": "background-color: #e2e3e5; color: #383d41",
        }.get(v, "")

    styler = show.style.map(_lvl_color, subset=["Signal"]) if hasattr(show.style, "map") else show.style.applymap(_lvl_color, subset=["Signal"])
    st.dataframe(styler, hide_index=True, use_container_width=True)

    sec = valued.groupby(valued["Sector"].replace("", "Unknown"))["Value"].sum().sort_values(ascending=False)
    if total_value and not sec.empty:
        st.markdown("**Sector weights**")
        sec_df = pd.DataFrame({"Sector": sec.index, "Weight %": (sec.values / total_value * 100.0).round(1)})
        st.dataframe(sec_df, hide_index=True, use_container_width=True)
        big = sec_df[sec_df["Weight %"] > 35]
        if not big.empty:
            st.warning("Concentration: " + ", ".join(f"{r['Sector']} is {r['Weight %']:.0f}% of the portfolio" for _, r in big.iterrows()) + " (over 35%).")
    st.caption(
        "Signals use the same rules as the Top 10 lists tab. Trim = 3 of 3 overbought signals and price above fair value; "
        "Review = Quality score under 50%; Watch = 2 overbought signals, or price above fair value with 1. "
        "Taxes and your own plan matter: this is a prompt to think, not an instruction to sell."
    )


_CMP_GREEN = "background-color: #d4edda; color: #155724"

# (section, label, key, format, direction: +1 higher is better / -1 lower is better / 0 no highlight, positive_only)
_CMP_ROWS = [
    ("Overview", "Name", "name", None, 0, False),
    ("Overview", "Sector / type", "sector", None, 0, False),
    ("Overview", "Price", "price", "{:,.2f}", 0, False),
    ("Overview", "Market cap", "mcap", "mcap", 0, False),
    ("Overview", "Verdict", "verdict_short", None, 0, False),
    ("Scores", "Quality score %", "quality_pct", "{:.0f}", 1, False),
    ("Scores", "Valuation score %", "valuation_pct", "{:.0f}", 1, False),
    ("Scores", "Timing score %", "timing_pct", "{:.0f}", 1, False),
    ("Scores", "Buy score (0-100)", "buy_score", "{:.0f}", 1, False),
    ("Valuation", "P/E (latest year)", "pe1", "{:.1f}", -1, True),
    ("Valuation", "P/E (3Y average earnings)", "pe3", "{:.1f}", -1, True),
    ("Valuation", "Forward P/E", "fwd_pe", "{:.1f}", -1, True),
    ("Valuation", "PEG", "peg", "{:.2f}", -1, True),
    ("Valuation", "EV / EBITDA", "ev_ebitda", "{:.1f}", -1, True),
    ("Valuation", "FCF yield %", "fcf_yield", "{:.1f}", 1, False),
    ("Valuation", "P/E percentile vs own history", "pe_pct", "{:.0f}", -1, False),
    ("Valuation", "Upside to fair value %", "upside_pct", "{:+.0f}", 1, False),
    ("Valuation", "Buy price: EPS model (10Y)", "buy_eps10", "{:,.2f}", 0, False),
    ("Valuation", "Buy price: DCF (10Y)", "buy_dcf", "{:,.2f}", 0, False),
    ("Valuation", "Assumed growth g %", "g_pct", "{:.1f}", 0, False),
    ("Valuation", "Assumed terminal multiple K", "k", "{:.0f}", 0, False),
    ("Quality", "ROIC %", "roic", "{:.1f}", 1, False),
    ("Quality", "Operating margin change (pts)", "opm_chg", "{:+.1f}", 1, False),
    ("Quality", "Revenue CAGR %", "rev_cagr", "{:.1f}", 1, False),
    ("Quality", "EPS CAGR %", "eps_cagr", "{:.1f}", 1, False),
    ("Quality", "FCF CAGR %", "fcf_cagr", "{:.1f}", 1, False),
    ("Quality", "Net debt / EBITDA (x)", "nd_ebitda", "{:.1f}", -1, False),
    ("Quality", "Interest coverage (x)", "int_cov", "{:.1f}", 1, False),
    ("Quality", "Stock compensation % of FCF", "sbc", "{:.1f}", -1, False),
    ("Quality", "Dividend (forward, per share)", "div", "{:.2f}", 0, False),
    ("Timing", "Price vs 200-day average %", "vs200", "{:+.1f}", 0, False),
    ("Timing", "Below 52-week high %", "dd52", "{:.1f}", 0, False),
    ("Timing", "Weekly RSI", "wrsi", "{:.0f}", 0, False),
    ("Timing", "6-month return vs SPY (pts)", "rel6", "{:+.1f}", 1, False),
    ("Timing", "1-year return %", "ret1y", "{:+.1f}", 1, False),
    ("Timing", "Overbought signals (of 3)", "sell_signals", "{:.0f}", -1, False),
    ("Timing", "Profit-taking signal", "trim_level", None, 0, False),
]


def _fmt_big(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "-"
        ax = abs(float(x))
        if ax >= 1e12:
            return f"{x / 1e12:.2f}T"
        if ax >= 1e9:
            return f"{x / 1e9:.1f}B"
        return f"{x / 1e6:.0f}M"
    except Exception:
        return "-"


def _cmp_values(row: dict, fin: dict, q: dict, v: dict, tr: dict) -> dict:
    """Flat metric dict for the Compare stocks table."""
    nd, eb = q.get("net_debt"), q.get("ebitda")
    nd_ebitda = (nd / eb) if (nd is not None and eb is not None and eb > 0) else None
    mcap, fcf1 = fin.get("Market Cap"), fin.get("FCF (1Y)")
    fcf_yield = (fcf1 / mcap * 100.0) if (mcap and fcf1 is not None and mcap > 0 and not is_financial(fin)) else None
    out = dict(row)
    out.update(
        {
            "sector": row.get("sector") or row.get("type") or "",
            "name": (row.get("name") or "")[:26],
            "mcap": mcap, "pe1": fin.get("P/E (1Y)"), "pe3": fin.get("P/E (3Y Avg)"), "fwd_pe": v.get("forward_pe"),
            "peg": v.get("peg"), "ev_ebitda": v.get("ev_ebitda"), "fcf_yield": fcf_yield, "pe_pct": v.get("pe_percentile"),
            "roic": q.get("roic"), "opm_chg": q.get("op_margin_change"), "rev_cagr": q.get("rev_cagr"),
            "eps_cagr": q.get("eps_cagr"), "fcf_cagr": q.get("fcf_cagr"), "nd_ebitda": nd_ebitda,
            "int_cov": q.get("interest_coverage"), "sbc": q.get("sbc_pct_fcf"), "div": fin.get("Div Rate (FWD)"),
            "vs200": tr.get("pct_vs_ma200"), "dd52": None if tr.get("dd_52w") is None else -tr["dd_52w"],
            "wrsi": tr.get("weekly_rsi"), "rel6": tr.get("rel_6m"), "ret1y": tr.get("ret_1y"),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Business snapshot tab (formerly "Business snapshot")
# ---------------------------------------------------------------------------

_SIGNAL_COLORS = {
    "Strong": "#2ea043", "Cheap": "#2ea043",
    "OK": "#d29922", "Fair": "#d29922",
    "Watch": "#f85149", "Pricey": "#f85149",
}


def _sig_style(val) -> str:
    colour = _SIGNAL_COLORS.get(str(val))
    return f"color: {colour}; font-weight: 600;" if colour else "color: #8b949e;"


def _money_short(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    a, sign = abs(float(x)), ("-" if x < 0 else "")
    if a >= 1e12:
        return f"{sign}${a / 1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.0f}M"
    return f"{sign}${a:,.0f}"


def _tier(v, strong, ok, higher: bool = True) -> str:
    """Strong / OK / Watch for a number against two cut-offs ('' when there is no number)."""
    if v is None or pd.isna(v):
        return ""
    if higher:
        return "Strong" if v >= strong else ("OK" if v >= ok else "Watch")
    return "Strong" if v <= strong else ("OK" if v <= ok else "Watch")


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct_txt(v, nd: int = 1) -> str:
    return "N/A" if v is None or pd.isna(v) else f"{v:.{nd}f}%"


def _latest_and_change(s: pd.Series, unit: str = " pts"):
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    if s.empty:
        return None, ""
    latest = float(s.iloc[-1])
    if len(s) < 2:
        return latest, ""
    return latest, f"{latest - float(s.iloc[0]):+.1f}{unit} since {s.index[0]}"


def snapshot_sections(price: float, fin: dict, q: dict, v: dict) -> list:
    """Detail rows for the Business snapshot tab: [(section title, [row dicts])]."""
    fin, q, v = fin or {}, q or {}, v or {}
    bank = is_financial(fin)
    table = q.get("table") if isinstance(q.get("table"), pd.DataFrame) else pd.DataFrame()
    series = q.get("series") if isinstance(q.get("series"), pd.DataFrame) else pd.DataFrame()
    na_bank = "N/A (financial company)"

    def col(name):
        return table[name] if name in table.columns else pd.Series(dtype=float)

    def row(metric, latest, detail="", signal="", good=""):
        return {"Metric": metric, "Latest": latest, "Trend / detail": detail, "Signal": signal, "What good looks like": good}

    # ---- Profitability
    gm, gm_chg = _latest_and_change(col("Gross margin %"))
    om, om_chg = _latest_and_change(col("Operating margin %"))
    nm, nm_chg = _latest_and_change(col("Net margin %"))
    fm, fm_chg = _latest_and_change(col("FCF margin %"))
    omc = q.get("op_margin_change")
    fm_s = col("FCF margin %").dropna()
    fm_drop = float(fm_s.iloc[-1] - fm_s.iloc[0]) if len(fm_s) >= 2 else None
    roic, roe = q.get("roic"), q.get("roe")
    profit = [
        row("Gross margin", _pct_txt(gm), gm_chg, "", "Higher is better; steady or rising is the goal"),
        row("Operating margin", _pct_txt(om), om_chg,
            "" if omc is None else ("Strong" if omc >= 2 else ("OK" if omc >= -2 else "Watch")),
            "Not shrinking by more than 2 points"),
        row("Net margin", _pct_txt(nm), nm_chg, "", "Profit kept from each dollar of sales"),
    ]
    if bank:
        profit.append(row("Free-cash-flow margin", na_bank, "", "", ""))
    else:
        profit.append(row("Free-cash-flow margin", _pct_txt(fm), fm_chg,
                          "" if fm_drop is None else ("Watch" if fm_drop <= -5 else "OK"),
                          "Cash left after investing, as a share of sales; a drop of 5 points or more is a flag"))
    profit.append(row("Return on invested capital (ROIC)", na_bank if bank else _pct_txt(roic), "",
                      "" if bank else _tier(roic, 15, 10), "Strong 15% or more; OK 10% or more"))
    profit.append(row("Return on equity (ROE)", _pct_txt(roe), "", _tier(roe, 15, 10), "Strong 15% or more; OK 10% or more"))

    # ---- Growth
    ry = q.get("rev_years") or 0
    span = f"over {ry:.1f} years" if ry else ""

    def g_row(name, val):
        return row(name, _pct_txt(val), span if val is not None else "", _tier(val, 10, 5), "Strong 10% a year or more; OK 5% or more")

    eps_n, eps_up = q.get("eps_n") or 0, q.get("eps_up")
    eps_sig = "" if (not eps_n or eps_up is None) else ("Strong" if eps_up == eps_n else ("OK" if eps_up / eps_n >= 0.6 else "Watch"))
    rev2, pg = fin.get("Revenue Growth (2Y %)"), fin.get("Profit Growth (2Y vs Today)")
    growth = [
        g_row("Revenue growth per year", q.get("rev_cagr")),
        g_row("Net income growth per year", q.get("ni_cagr")),
        g_row("Earnings per share growth per year", q.get("eps_cagr")),
        g_row("Free cash flow growth per year", None if bank else q.get("fcf_cagr")),
        row("Years earnings per share rose", f"{eps_up} of {eps_n}" if eps_n else "N/A", "", eps_sig, "At least 60% of years"),
        row("Revenue, total change over 2 years", "N/A" if rev2 is None else f"{rev2:+.1f}%", "Total over two years, not per year", "", ""),
        row("Net income vs 2 years ago", pg or "N/A", "", {"Up": "Strong", "Flat": "OK", "Down": "Watch"}.get(pg, ""), "Up"),
    ]

    # ---- Cash and balance sheet
    fcf1 = fin.get("FCF (1Y)")
    ni_s = series["Net income"].dropna() if "Net income" in series.columns else pd.Series(dtype=float)
    ni_latest = float(ni_s.iloc[-1]) * 1e9 if not ni_s.empty else None
    fcf_conv = (fcf1 / ni_latest * 100.0) if (fcf1 is not None and ni_latest and ni_latest > 0) else None
    nd, eb, nde = q.get("net_debt"), q.get("ebitda"), q.get("net_debt_ebitda")
    fcf_pos, fcf_n = q.get("fcf_pos"), q.get("fcf_n")
    if bank:
        cash = [row("Debt, cash flow and coverage", na_bank, "Not meaningful for banks and insurers", "", "")]
    else:
        if nd is None:
            nd_text = "N/A"
        elif nd <= 0:
            nd_text = "Net cash " + _money_short(abs(nd))
        else:
            nd_text = _money_short(nd)
        if nd is not None and nd <= 0:
            lev_text, lev_sig = "Net cash", "Strong"
        elif nde is not None:
            lev_text, lev_sig = f"{nde:.1f}x", _tier(nde, 1.5, 3.0, higher=False)
        elif nd is not None and eb is not None and eb <= 0:
            lev_text, lev_sig = "EBITDA is zero or negative", "Watch"
        else:
            lev_text, lev_sig = "N/A", ""
        cov, cur = q.get("interest_coverage"), q.get("current_ratio")
        cash = [
            row("Free cash flow (latest year)", _money_short(fcf1),
                f"positive in {fcf_pos} of {fcf_n} years" if fcf_n else "",
                "" if (not fcf_n or fcf_pos is None) else ("Strong" if fcf_pos == fcf_n else "Watch"), "Positive every year"),
            row("Free cash flow as % of net income", _pct_txt(fcf_conv, 0), "", _tier(fcf_conv, 90, 70), "Strong 90% or more; OK 70% or more"),
            row("Net debt (debt minus cash)", nd_text, "", "Strong" if (nd is not None and nd <= 0) else "", "Negative means more cash than debt"),
            row("Net debt / EBITDA", lev_text, "", lev_sig, "Strong 1.5x or less; OK 3x or less"),
            row("Interest coverage", "N/A" if cov is None else f"{cov:.1f}x", "", _tier(cov, 10, 5), "Strong 10x or more; OK 5x or more"),
            row("Current ratio", "N/A" if cur is None else f"{cur:.2f}", "", _tier(cur, 1.5, 1.0), "Strong 1.5 or more; OK 1.0 or more"),
        ]

    # ---- Shareholders
    sh = fin.get("Shares Outstanding (2Y)")
    sh_sig = "" if not sh else ("Strong" if str(sh).startswith("Buying") else ("Watch" if str(sh).startswith("Issuing") else "OK"))
    rate = fin.get("Div Rate (FWD)")
    if rate:
        div_text = f"${rate:.2f} a year" + (f" ({rate / price * 100:.1f}% yield)" if price else "")
    else:
        div_text = "No dividend"
    sbc = q.get("sbc_pct_fcf")
    holders = [
        row("Share count change (2Y)", sh or "N/A", "", sh_sig, "Buying back helps you; issuing new shares dilutes you"),
        row("Stock compensation as % of free cash flow", na_bank if bank else _pct_txt(sbc, 0), "",
            "" if bank else _tier(sbc, 10, 20, higher=False), "Strong 10% or less; OK 20% or less"),
        row("Dividend", div_text, "", "", "Cash paid out to shareholders"),
    ]

    # ---- Valuation multiples
    pe1, pe3 = fin.get("P/E (1Y)"), fin.get("P/E (3Y Avg)")
    fwd, peg, evx = v.get("forward_pe"), v.get("peg"), v.get("ev_ebitda")
    pct, mcap, sector_pe = v.get("pe_percentile"), fin.get("Market Cap"), fin.get("Sector PE")
    fcf_yield = (fcf1 / mcap * 100.0) if (not bank and mcap and fcf1 is not None and mcap > 0) else None
    if pe1 is not None and pe1 > 0 and sector_pe:
        sec_sig = "Cheap" if pe1 <= sector_pe * 0.9 else ("Pricey" if pe1 >= sector_pe * 1.2 else "Fair")
    else:
        sec_sig = ""
    fwd_sig = "" if (fwd is None or pe1 is None or pe1 <= 0) else ("Strong" if fwd < pe1 else "Watch")
    multiples = [
        row("P/E, latest year", "N/A" if pe1 is None else f"{pe1:.1f}", "" if pe3 is None else f"3-year average {pe3:.1f}", "",
            "Price relative to last year's profit; lower is cheaper"),
        row("P/E versus its own history",
            "N/A" if pct is None else f"{v['pe_current']:.1f} (median {v['pe_median']:.1f})",
            "" if pct is None else f"{_ordinal(pct)} percentile over {v['pe_years']:.1f} years",
            "" if pct is None else ("Cheap" if pct <= 40 else ("Fair" if pct <= 70 else "Pricey")),
            "40th percentile or lower is cheap"),
        row("Forward P/E", "N/A" if fwd is None else f"{fwd:.1f}", "" if pe1 is None else f"latest-year P/E {pe1:.1f}", fwd_sig,
            "Below the latest-year P/E means analysts expect profits to grow"),
        row("PEG ratio", "N/A" if peg is None else f"{peg:.2f}", "",
            "" if peg is None else ("Cheap" if peg <= 1 else ("Fair" if peg <= 2 else "Pricey")), "1 or less is cheap; 2 or less is fair"),
        row("EV / EBITDA", "N/A" if evx is None else f"{evx:.1f}", "", "", "Includes debt; compare with similar companies"),
        row("Free-cash-flow yield", na_bank if bank else _pct_txt(fcf_yield),
            f"{mcap / fcf1:.0f}x price to free cash flow" if (fcf_yield and fcf1) else "",
            "" if fcf_yield is None else ("Cheap" if fcf_yield >= 4 else ("Fair" if fcf_yield >= 2 else "Pricey")),
            "4% or more is cheap"),
        row("P/E versus its sector fund", "N/A" if (pe1 is None or not sector_pe) else f"{pe1:.1f} vs {sector_pe:.1f}", "", sec_sig,
            "Rough benchmark only: the P/E of the matching sector ETF"),
    ]
    return [
        ("Profitability", profit),
        ("Growth", growth),
        ("Cash and balance sheet", cash),
        ("Shareholders", holders),
        ("Valuation multiples", multiples),
    ]


def snapshot_notes(price: float, fin: dict, q: dict, v: dict):
    """Plain-language strengths and watch-outs, using the same cut-offs as the Buy checklist."""
    fin, q, v = fin or {}, q or {}, v or {}
    bank = is_financial(fin)
    table = q.get("table") if isinstance(q.get("table"), pd.DataFrame) else pd.DataFrame()
    since = f" since {table.index[0]}" if not table.empty else ""
    good, watch = [], []

    if not bank:
        roic = q.get("roic")
        if roic is not None:
            if roic >= 15:
                good.append(f"Earns {roic:.0f}% a year on the money invested in the business (ROIC), a sign of a strong competitive position.")
            elif roic < 10:
                watch.append(f"Earns only {roic:.1f}% a year on invested capital (ROIC), below the 10% we look for.")
    omc = q.get("op_margin_change")
    if omc is not None:
        if omc >= 2:
            good.append(f"Operating margin has widened by {omc:.1f} points{since}: the business is becoming more profitable.")
        elif omc <= -2:
            watch.append(f"Operating margin has narrowed by {abs(omc):.1f} points{since}: costs are growing faster than sales.")
    rc = q.get("rev_cagr")
    if rc is not None:
        if rc >= 10:
            good.append(f"Sales are growing about {rc:.0f}% a year.")
        elif rc < 5:
            watch.append(f"Sales growth is slow (about {rc:.1f}% a year).")
    if not bank:
        fm = table["FCF margin %"].dropna() if "FCF margin %" in table.columns else pd.Series(dtype=float)
        if len(fm) >= 2 and fm.iloc[-1] - fm.iloc[0] <= -5:
            watch.append(
                f"Free cash flow fell from {fm.iloc[0]:.0f}% to {fm.iloc[-1]:.0f}% of sales{since}: check whether heavier investment or weaker cash conversion is the cause."
            )
        fcf1 = fin.get("FCF (1Y)")
        series = q.get("series") if isinstance(q.get("series"), pd.DataFrame) else pd.DataFrame()
        ni_s = series["Net income"].dropna() if "Net income" in series.columns else pd.Series(dtype=float)
        if fcf1 is not None and not ni_s.empty and ni_s.iloc[-1] > 0:
            conv = fcf1 / (float(ni_s.iloc[-1]) * 1e9) * 100.0
            if conv >= 100:
                good.append(f"Free cash flow is {conv:.0f}% of reported profit: profits are backed by real cash.")
            elif conv < 70:
                watch.append(f"Only {conv:.0f}% of reported profit turned into free cash: look at capital spending and working capital.")
        nd, nde = q.get("net_debt"), q.get("net_debt_ebitda")
        if nd is not None and nd <= 0:
            good.append("Holds more cash than debt (net cash), so it has room to ride out a bad year.")
        elif nde is not None and nde > 3:
            watch.append(f"Debt is high: net debt is {nde:.1f} times EBITDA (we prefer 3 or less).")
        cov = q.get("interest_coverage")
        if cov is not None and cov < 5:
            watch.append(f"Interest is covered only {cov:.1f} times by operating profit (we prefer 5 or more).")
        sbc = q.get("sbc_pct_fcf")
        if sbc is not None and sbc > 20:
            watch.append(f"Stock compensation equals {sbc:.0f}% of free cash flow, which quietly dilutes owners.")
    sh = fin.get("Shares Outstanding (2Y)")
    if sh and str(sh).startswith("Buying"):
        good.append(f"Is shrinking its share count ({sh}): each share is a bigger slice of the company.")
    elif sh and str(sh).startswith("Issuing"):
        watch.append(f"Is issuing new shares ({sh}), which shrinks your slice.")

    pct = v.get("pe_percentile")
    if pct is not None:
        if pct <= 30:
            good.append(
                f"Today's P/E ({v['pe_current']:.0f}) is lower than on about {100 - pct:.0f}% of days over the last {v['pe_years']:.0f} years: cheap against its own history."
            )
        elif pct >= 80:
            watch.append(
                f"Today's P/E ({v['pe_current']:.0f}) is higher than on about {pct:.0f}% of days over the last {v['pe_years']:.0f} years: expensive against its own history."
            )
    mcap, fcf1 = fin.get("Market Cap"), fin.get("FCF (1Y)")
    if not bank and mcap and fcf1 is not None and mcap > 0:
        fy = fcf1 / mcap * 100.0
        if fy >= 4:
            good.append(f"Free-cash-flow yield is {fy:.1f}%: a reasonable cash return for the price paid.")
        elif fy < 2:
            watch.append(f"Free-cash-flow yield is only {fy:.1f}%: you pay a lot for each dollar of cash the business makes.")
    return good[:6], watch[:6]


def _bars(df: pd.DataFrame, height: int = 240):
    try:
        st.bar_chart(df, stack=False, height=height)
    except TypeError:
        st.bar_chart(df, height=height)


def render_business_snapshot(active_stock: str, price: float, fin: dict, quality: dict, valctx: dict):
    st.subheader(f"Business snapshot - {active_stock}")
    try:
        inst = get_instrument_info(active_stock)
    except Exception:
        inst = {}
    if inst.get("type") in ETF_TYPES:
        st.info(
            "This is a fund (ETF), so it has no company financial statements. Use Price charts and the Buy checklist "
            "(only the Timing part applies to funds)."
        )
        return
    fin, q, v = fin or {}, quality or {}, valctx or {}
    if not fin and not q.get("available"):
        st.info("Business data is not available for this ticker right now (new listing, fund, or a gap at Yahoo).")
        return

    bank = is_financial(fin)
    sector = fin.get("Sector")
    st.caption(
        f"{sector + '. ' if sector else ''}Annual statements from Yahoo Finance (about 4 fiscal years, so growth rates cover about 3 years). "
        "Signals are prompts to investigate, not verdicts. Not investment advice."
    )

    # ---- headline cards
    mcap, pe1, pe3 = fin.get("Market Cap"), fin.get("P/E (1Y)"), fin.get("P/E (3Y Avg)")
    fwd, peg = v.get("forward_pe"), v.get("peg")
    fcf1 = fin.get("FCF (1Y)")
    fcf_yield = (fcf1 / mcap * 100.0) if (not bank and mcap and fcf1 is not None and mcap > 0) else None
    rate = fin.get("Div Rate (FWD)")
    nd, nde = q.get("net_debt"), q.get("net_debt_ebitda")
    if bank:
        lev_value, lev_delta = "N/A", "financial company"
    elif nd is not None and nd <= 0:
        lev_value, lev_delta = "Net cash", _money_short(abs(nd))
    elif nde is not None:
        lev_value, lev_delta = f"{nde:.1f}x", _money_short(nd)
    else:
        lev_value, lev_delta = "N/A", None
    cards = st.columns(6)
    cards[0].metric("Market cap", _money_short(mcap))
    cards[1].metric("P/E (latest year)", "N/A" if pe1 is None else f"{pe1:.1f}", None if pe3 is None else f"3-year avg {pe3:.1f}", delta_color="off")
    cards[2].metric("Forward P/E", "N/A" if fwd is None else f"{fwd:.1f}", None if peg is None else f"PEG {peg:.2f}", delta_color="off")
    cards[3].metric("Free-cash-flow yield", _pct_txt(fcf_yield), None if not fcf1 else f"FCF {_money_short(fcf1)}", delta_color="off")
    cards[4].metric("Dividend yield", f"{rate / price * 100:.1f}%" if (rate and price) else "None", f"${rate:.2f} a year" if rate else None, delta_color="off")
    cards[5].metric("Net debt / EBITDA", lev_value, lev_delta, delta_color="off")

    # ---- strengths and watch-outs
    good, watch = snapshot_notes(price, fin, q, v)
    n1, n2 = st.columns(2)
    with n1:
        with st.container(border=True):
            st.markdown(":green[**Strengths**]")
            st.markdown("\n".join(f"- {t}" for t in good) if good else "Nothing stands out as unusually strong.")
    with n2:
        with st.container(border=True):
            st.markdown(":orange[**Watch-outs**]")
            st.markdown("\n".join(f"- {t}" for t in watch) if watch else "No red flags against the checklist cut-offs.")

    # ---- charts
    series = q.get("series") if isinstance(q.get("series"), pd.DataFrame) else pd.DataFrame()
    table = q.get("table") if isinstance(q.get("table"), pd.DataFrame) else pd.DataFrame()
    if not series.empty or not table.empty:
        st.markdown("**Trends by fiscal year**")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        s = series.copy()
        s.index = [str(i) for i in s.index]
        t = table.copy()
        t.index = [str(i) for i in t.index]
        with c1:
            st.caption("Revenue and net income (billions of dollars)")
            cols = [c for c in ("Revenue", "Net income") if c in s.columns]
            if cols:
                _bars(s[cols])
        with c2:
            st.caption("Margins (% of sales)")
            cols = [c for c in ("Gross margin %", "Operating margin %", "Net margin %", "FCF margin %") if c in t.columns]
            if cols:
                st.line_chart(t[cols], height=240)
        with c3:
            st.caption("Net income versus free cash flow (billions of dollars)")
            cols = [c for c in ("Net income", "Free cash flow") if c in s.columns]
            if cols and not bank:
                _bars(s[cols])
            elif bank:
                st.write("Not meaningful for banks and insurers.")
        with c4:
            st.caption("Shares outstanding (billions): falling is good")
            if "Shares (billions)" in s.columns and not s["Shares (billions)"].dropna().empty:
                _bars(s[["Shares (billions)"]].dropna())

    # ---- detail tables
    st.markdown("**Details by area**")
    st.caption(
        "Signal colours: green = Strong or Cheap, amber = OK or Fair, red = Watch or Pricey, grey = information only. "
        "Cut-offs match the Buy checklist where it has one."
    )
    export_rows = []
    for title, rows in snapshot_sections(price, fin, q, v):
        st.markdown(f"**{title}**")
        df = pd.DataFrame(rows)
        sty = df.style
        sty = (getattr(sty, "map", None) or sty.applymap)(_sig_style, subset=["Signal"])
        st.dataframe(sty, hide_index=True, width="stretch", height=(len(df) + 1) * 35 + 3)
        export_rows += [{"Section": title, **r} for r in rows]

    st.download_button(
        "Download these numbers (CSV)",
        data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
        file_name=f"{active_stock}_business_snapshot.csv",
        mime="text/csv",
    )
    if not table.empty:
        with st.expander("Year-by-year table"):
            yr = table.copy()
            yr.index = [str(i) for i in yr.index]
            st.dataframe(yr.round(1), width="stretch")


# ---------------------------------------------------------------------------
# Entry and exit plan tab: staged buying and staged profit-taking
# ---------------------------------------------------------------------------

def build_entry_ladder(anchor: float, n: int, step_pct: float, total: float, price: float) -> list:
    """n buy tranches. Tranche 1 triggers at `anchor`; each next one is `step_pct` lower than the previous trigger step."""
    rows = []
    per = total / n if n else 0.0
    for i in range(n):
        trigger = anchor * (1.0 - step_pct / 100.0 * i)
        reached = price <= trigger
        buy_at = min(price, trigger)
        rows.append(
            {
                "Tranche": i + 1,
                "Buy at or below": trigger,
                "Distance from today %": (trigger / price - 1.0) * 100.0 if price else None,
                "Amount": per,
                "Shares": per / buy_at if buy_at > 0 else None,
                "Status": "Reached: buy now" if reached else "Waiting",
            }
        )
    return rows


def build_exit_ladder(base: float, levels_pct: list, sell_pcts: list, price: float, shares: float, avg_cost: float) -> list:
    """Profit-taking steps. Trigger = base x (1 + level%). Each step sells a % of the ORIGINAL position."""
    rows = []
    sold_total = 0.0
    for i, (lv, sp) in enumerate(zip(levels_pct, sell_pcts), start=1):
        sp = max(0.0, min(float(sp), 100.0 - sold_total))
        sold_total += sp
        trigger = base * (1.0 + float(lv) / 100.0)
        sell_sh = shares * sp / 100.0 if shares else 0.0
        rows.append(
            {
                "Step": i,
                "Sell when price reaches": trigger,
                "Distance from today %": (trigger / price - 1.0) * 100.0 if price else None,
                "Sell % of position": sp,
                "Shares to sell": sell_sh if shares else None,
                "Cash received": sell_sh * trigger if shares else None,
                "Gain locked in": (trigger - avg_cost) * sell_sh if (shares and avg_cost) else None,
                "Status": "Reached: consider selling" if price >= trigger else "Waiting",
            }
        )
    return rows


def _usd(x) -> str:
    return "-" if x is None or pd.isna(x) else f"${x:,.2f}"


def _plan_df(rows: list, money_cols: tuple, pct_cols: tuple, shares_cols: tuple = ()) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].map(_usd)
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "-" if x is None or pd.isna(x) else f"{x:+.1f}%" if "Distance" in c else f"{x:.0f}%")
    for c in shares_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "-" if x is None or pd.isna(x) else f"{x:,.2f}")
    return out


def render_plan_tab(active_stock: str, price: float, fin: dict, quality: dict, trend: dict, indicator_data, pillar_scores: dict, r_default: float):
    st.subheader(f"Entry and exit plan - {active_stock}")
    st.caption(
        "Turns the checklist into a staged plan: buy in tranches at prices you decide in advance, and take profit in steps. "
        "Deciding the levels before the price moves is what keeps emotion out. Planning aid only, not investment advice."
    )
    try:
        is_fund = get_instrument_info(active_stock).get("type") in ETF_TYPES
    except Exception:
        is_fund = False

    g_in = st.session_state.get("val_g_pct", 10.0)
    k_in = st.session_state.get("val_k_multiple", 15.0)
    r_in = st.session_state.get("val_r_pct", r_default)
    mos_in = st.session_state.get("val_mos_pct", 25.0)
    val_rows = [] if is_fund else summarize_valuation(fin, price, g_in / 100.0, k_in, r_in / 100.0, mos_in / 100.0)
    by_model = {r["Model"]: r for r in val_rows}
    pick = by_model.get("EPS model (10Y)") or by_model.get("DCF (10Y)")
    buy_px = pick["Buy price"] if pick else None
    fair_px = pick["Fair value"] if pick else None

    q_pct, _, q_n = pillar_scores.get("quality", (None, 0, 0))
    level, headline, detail = long_term_verdict(pillar_scores)
    getattr(st, level)(f"**{headline}** - {detail}")

    # ---- reference levels
    bb_low = None
    if indicator_data is not None and len(indicator_data):
        bb_low = float(indicator_data["BB_Low"].iloc[-1])
    ma200 = (trend or {}).get("ma200")
    ser = (trend or {}).get("series")
    lo52 = float(ser["Close"].tail(252).min()) if ser is not None and len(ser) else None
    ref = st.columns(6)
    ref[0].metric("Current price", _usd(price))
    ref[1].metric("Buy price (model)", _usd(buy_px))
    ref[2].metric("Fair value (model)", _usd(fair_px))
    ref[3].metric("Lower Bollinger band (4h)", _usd(bb_low))
    ref[4].metric("200-day average", _usd(ma200))
    ref[5].metric("52-week low", _usd(lo52))
    st.caption("Buy price and fair value come from the Fair value tab (10-year EPS model, or the DCF when there is no EPS). Edit the inputs there to move them.")

    if q_pct is not None and q_pct < 50 and not is_fund:
        st.warning(f"Quality score is only {q_pct:.0f}%. Cheap prices on a weak business are a trap, so treat the entry ladder below as reference only and do not add until the checklist improves.")

    # ---------------- entry ----------------
    st.markdown("### 1. Entry: buy in tranches")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        total = st.number_input("Total to invest (dollars)", min_value=100.0, value=10000.0, step=500.0, key="plan_total")
    with e2:
        n_tr = int(st.number_input("Number of tranches", min_value=2, max_value=6, value=3, step=1, key="plan_n"))
    with e3:
        styles = ["Value: start at the Buy price", "Pullback: start at today's price"]
        style = st.radio("Ladder style", styles, key="plan_style", help="Value waits for the model's Buy price. Pullback buys the first tranche now, then adds on dips.")
    with e4:
        step = st.number_input("Step between tranches (%)", min_value=1.0, max_value=30.0, value=7.5, step=0.5, key="plan_step")
    use_value = style == styles[0]
    if use_value and buy_px is None:
        st.info("No model Buy price is available for this ticker, so the ladder starts at today's price instead.")
        use_value = False
    anchor = buy_px if use_value else price
    entry = build_entry_ladder(anchor, n_tr, step, total, price)
    st.dataframe(
        _plan_df(entry, ("Buy at or below", "Amount"), ("Distance from today %",), ("Shares",)),
        hide_index=True, width="stretch", height=(len(entry) + 1) * 35 + 3,
    )
    reached = sum(1 for r in entry if r["Status"].startswith("Reached"))
    if reached:
        st.success(f"{reached} of {n_tr} tranches are reached at today's price: about {_usd(total * reached / n_tr)} could be invested now.")
    elif buy_px is not None and use_value:
        st.info(f"Price is {(price / buy_px - 1) * 100:.1f}% above the Buy price, so no tranche is reached yet. Be patient or use the Pullback style.")
    sup = [f"{n} {_usd(v)}" for n, v in (("lower Bollinger band", bb_low), ("200-day average", ma200), ("52-week low", lo52)) if v]
    if sup:
        st.caption("Support levels to compare with your triggers: " + "; ".join(sup) + ". A trigger near one of these has extra reason to hold.")

    # ---------------- exit ----------------
    st.markdown("### 2. Exit: take profit in steps")
    holdings = st.session_state.get("pf_base")
    d_sh, d_cost = 0.0, 0.0
    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        m = holdings[holdings["Symbol"].astype(str).str.upper() == active_stock]
        if not m.empty:
            d_sh, d_cost = float(m["Shares"].iloc[0] or 0.0), float(m["Avg cost"].iloc[0] or 0.0)
    x1, x2, x3 = st.columns(3)
    with x1:
        shares = st.number_input("Shares I own", min_value=0.0, value=d_sh, step=1.0, key=f"plan_shares_{active_stock}")
    with x2:
        avg_cost = st.number_input("My average cost per share", min_value=0.0, value=d_cost, step=1.0, key=f"plan_cost_{active_stock}")
    with x3:
        bases = ["Model fair value", "My average cost"]
        basis = st.radio("Measure steps from", bases, key="plan_basis", help="Fair value suits a valuation-driven exit. Average cost suits a pure profit target.")
    base_px = fair_px if basis == bases[0] else (avg_cost or None)
    if base_px is None:
        st.info("Enter your average cost (or pick Model fair value on a stock that has one) to build the exit ladder. Holdings stay in this browser session only and are never saved.")
    else:
        if basis == bases[0]:
            defaults = pd.DataFrame({"Trigger (% above fair value)": [0.0, 20.0, 40.0], "Sell (% of my shares)": [10.0, 15.0, 25.0]})
        else:
            defaults = pd.DataFrame({"Trigger (% above my cost)": [30.0, 60.0, 100.0], "Sell (% of my shares)": [10.0, 15.0, 25.0]})
        edited = st.data_editor(
            defaults, hide_index=True, num_rows="fixed", width="stretch",
            key=f"plan_exit_{'fair' if basis == bases[0] else 'cost'}_{active_stock}",
        )
        levels, sells = edited.iloc[:, 0].fillna(0.0).tolist(), edited.iloc[:, 1].fillna(0.0).tolist()
        if sum(sells) > 100.0:
            st.warning("The sell percentages add up to more than 100%. The last steps are cut back to what is left.")
        exit_rows = build_exit_ladder(base_px, levels, sells, price, shares, avg_cost)
        st.dataframe(
            _plan_df(exit_rows, ("Sell when price reaches", "Cash received", "Gain locked in"), ("Distance from today %", "Sell % of position"), ("Shares to sell",)),
            hide_index=True, width="stretch", height=(len(exit_rows) + 1) * 35 + 3,
        )
        if not shares:
            st.caption("Enter the shares you own to see share counts, cash received and gains.")

    # ---- technical profit-taking read
    sell = compute_sell_signals(download_daily(active_stock))
    above_fair = _above_fair(val_rows)
    t_level, t_why = trim_signal(sell["count"], above_fair, q_pct, q_n)
    s1, s2 = st.columns(2)
    s1.metric("Overbought signals today", "N/A" if sell["count"] is None else f"{sell['count']} of 3")
    s2.metric("Profit-taking level", t_level, t_why, delta_color="off")
    if t_level == "Trim":
        st.warning("The overbought signals and the valuation both say Trim today. Many investors take the first step now rather than waiting for the next price trigger.")
    st.caption(
        "Overbought signals: close above the upper Bollinger band (15, 2), RSI (10) above 70, and MACD below its signal line, on daily bars."
    )

    # ---- rules to re-check
    with st.expander("When to stop and re-check the plan", expanded=False):
        st.markdown(
            "- **Stop adding** and re-run the Buy checklist if the price falls below your last tranche by more than one step, "
            "or if the Quality score drops under 50%.\n"
            "- **Update your average cost** in My portfolio after each tranche, because every buy changes it.\n"
            "- **Re-run the exit levels** after each earnings report: fair value moves when profit estimates change.\n"
            "- A ladder is a plan, not a promise. If the business story changes, change the plan."
        )

    export = [{"Section": "Entry", **r} for r in entry]
    if base_px is not None:
        export += [{"Section": "Exit", **r} for r in exit_rows]
    st.download_button(
        "Download this plan (CSV)",
        data=pd.DataFrame(export).to_csv(index=False).encode("utf-8"),
        file_name=f"{active_stock}_entry_exit_plan.csv",
        mime="text/csv",
    )


def render_compare_tab(active_stock: str, favorites: list, r_default: float):
    st.subheader("Compare stocks side by side")
    st.caption(
        "Pick 2 to 4 stocks. Green marks the best value in a row (only where higher or lower is clearly better). "
        "Each stock uses its own suggested valuation defaults, so Valuation scores compare each price against that stock's own assumptions."
    )
    options = list(dict.fromkeys([active_stock] + list(favorites)))
    picked = st.multiselect("Choose stocks to compare (your current stock and Favorites)", options, default=[active_stock], max_selections=4)
    extra_txt = st.text_input("Add other tickers (comma separated)", value="", key="cmp_extra")
    extras = [s.strip().upper() for s in extra_txt.split(",") if s.strip()]
    symbols = list(dict.fromkeys(picked + extras))
    if len(symbols) > 4:
        st.caption("Comparing the first 4 only.")
        symbols = symbols[:4]
    if len(symbols) < 2:
        st.info("Choose at least two stocks (or type another ticker above) to compare.")
        return

    snaps: dict = {}
    failed: list = []
    bar = st.progress(0.0, text="Loading...")
    bench = download_daily("SPY")
    for i, sym in enumerate(symbols, start=1):
        bar.progress(i / len(symbols), text=f"Loading {sym} ({i}/{len(symbols)}) - the first load takes several seconds per stock")
        try:
            row = score_symbol_cached(sym, r_default)
            if row.get("price") is None:
                failed.append(f"{sym} ({row.get('error') or 'no data'})")
                continue
            etf = row["type"] in ETF_TYPES
            fin = {} if etf else (fetch_financial_metrics(sym) or {})
            q = {} if etf else (fetch_quality_metrics(sym) or {})
            v = {} if etf else (fetch_valuation_context(sym) or {})
            daily = download_daily(sym)
            tr = compute_trend_context(daily, bench)
            snaps[sym] = (row, _cmp_values(row, fin, q, v, tr), daily)
        except Exception as exc:
            failed.append(f"{sym} ({str(exc)[:60]})")
    bar.empty()
    if failed:
        st.warning("Could not load: " + ", ".join(failed))
    if len(snaps) < 2:
        st.info("Need at least two stocks with data to compare.")
        return
    cols = list(snaps)

    # ---- quick takeaways ----
    def _best(key: str, high: bool = True):
        cand = {s: snaps[s][0].get(key) for s in cols}
        cand = {s: x for s, x in cand.items() if x is not None and not pd.isna(x)}
        if len(cand) < 2:
            return None
        target = max(cand.values()) if high else min(cand.values())
        return [s for s, x in cand.items() if x == target], target

    notes = []
    for label, key, high, fmt in (
        ("Highest buy score", "buy_score", True, "{:.0f}"),
        ("Highest quality score", "quality_pct", True, "{:.0f}%"),
        ("Cheapest versus fair value", "upside_pct", True, "{:+.0f}% upside"),
        ("Most overbought", "sell_signals", True, "{:.0f} of 3 signals"),
    ):
        b = _best(key, high)
        if b and not (key == "sell_signals" and b[1] < 2):
            if key == "upside_pct":
                detail = f"{b[1]:.0f}% upside" if b[1] >= 0 else f"still {abs(b[1]):.0f}% above fair value"
                label = "Most upside to fair value" if b[1] >= 0 else "Closest to fair value"
            else:
                detail = fmt.format(b[1])
            notes.append(f"- **{label}:** {', '.join(b[0])} ({detail})")
    if notes:
        st.markdown("\n".join(notes))

    # ---- table ----
    recs, colors = [], []
    for section, label, key, fmt, direction, pos_only in _CMP_ROWS:
        rec, col, nums = {"Section": section, "Metric": label}, {"Section": "", "Metric": ""}, {}
        for s in cols:
            x = snaps[s][1].get(key)
            if fmt is None:
                rec[s] = "-" if (x is None or x == "" or (isinstance(x, float) and pd.isna(x))) else str(x)
            elif fmt == "mcap":
                rec[s] = _fmt_big(x)
            else:
                rec[s] = _f(x, fmt)
            col[s] = ""
            if direction and isinstance(x, (int, float)) and not pd.isna(x) and (not pos_only or x > 0):
                nums[s] = float(x)
        if direction and len(nums) >= 2:
            target = max(nums.values()) if direction > 0 else min(nums.values())
            winners = [s for s, x in nums.items() if x == target]
            if len(winners) < len(nums):
                for w in winners:
                    col[w] = _CMP_GREEN
        recs.append(rec)
        colors.append(col)
    tbl = pd.DataFrame(recs)
    color_df = pd.DataFrame(colors, columns=tbl.columns)
    st.dataframe(tbl.style.apply(lambda _: color_df, axis=None), hide_index=True, use_container_width=True, height=36 * (len(tbl) + 1) + 3)

    # ---- charts ----
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Pillar scores (%)**")
        bars = pd.DataFrame(
            {s: [snaps[s][0].get("quality_pct"), snaps[s][0].get("valuation_pct"), snaps[s][0].get("timing_pct")] for s in cols},
            index=["Quality", "Valuation", "Timing"],
        ).astype(float).fillna(0.0)
        st.bar_chart(bars)
    with ch2:
        st.markdown("**Price, last 12 months (start = 100)**")
        series = {}
        for s in cols:
            try:
                c = normalize_ohlcv_columns(snaps[s][2])["Close"].astype(float).dropna().tail(252)
                c.index = pd.to_datetime(c.index).tz_localize(None)
                series[s] = c / c.iloc[0] * 100.0
            except Exception:
                continue
        if series:
            st.line_chart(pd.DataFrame(series))
    st.caption("Screening aid only, not investment advice. Green does not mean buy: read the Buy checklist for each stock.")


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


# ---------------------------------------------------------------------------
# News and market context (free sources, no keys). Google News RSS is the primary feed
# because it kept working when Yahoo was rate-limiting; Yahoo's news is the backup.
# ---------------------------------------------------------------------------
_NEWS_NOISE = ("quote & history", "stock price, news", "charts, data", "price, news, quote", "live stock price", "stock quote")
# Auto-generated fund-filing posts ("fund X acquires new shares of Y") add nothing for a long-term investor
_NEWS_FILLER = ("acquires new", "acquires shares", "sells shares of", "buys shares of", "boosts stake", "trims stake", "raises stake",
                "lowers stake", "increases stake", "reduces stake", "takes position in", "new position in", "shares bought by", "shares sold by")
_NAME_SUFFIX = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|holdings?|plc|ltd|limited|group|n\.v|nv|class [a-c]|cl [a-c]|common stock|ordinary shares?|adr)\b\.?",
    re.I,
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _clean_company_name(name: str) -> str:
    """'United Airlines Holdings, Inc.' -> 'United Airlines' (a better news search phrase)."""
    n = re.sub(r"\([^)]*\)", " ", name or "")
    n = _NAME_SUFFIX.sub(" ", n)
    n = re.sub(r"[,]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip(" .-")
    return n or (name or "").strip()


def _parse_news_time(text):
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except Exception:
        try:
            dt = pd.to_datetime(text, utc=True).to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _google_news(query: str, days: int) -> list:
    """Headlines from Google News RSS. Raises on network or HTTP errors."""
    url = f"https://news.google.com/rss/search?q={quote(f'{query} when:{days}d')}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).findall("./channel/item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3].rstrip()
        out.append({"title": title, "source": source, "published": _parse_news_time(it.findtext("pubDate")),
                    "url": it.findtext("link") or "", "via": "Google News"})
    return out


def _yahoo_time(c: dict):
    stamp = c.get("pubDate") or c.get("displayTime")
    if stamp:
        return _parse_news_time(stamp)
    if c.get("providerPublishTime"):
        return datetime.fromtimestamp(c["providerPublishTime"], tz=timezone.utc)
    return None


def _yahoo_news(symbol: str) -> list:
    """Headlines tagged to a symbol by Yahoo Finance (backup source). Raises when Yahoo is unavailable."""
    out = []
    for n in (yf.Ticker(symbol).news or []):
        c = n.get("content") or n
        title = (c.get("title") or "").strip()
        link = ((c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url")) or c.get("link") or ""
        if title and link:
            out.append({"title": title, "source": ((c.get("provider") or {}).get("displayName")) or c.get("publisher") or "",
                        "published": _yahoo_time(c),
                        "url": link, "via": "Yahoo Finance"})
    return out


def _dedupe_news(items: list) -> list:
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x.get("published") or _EPOCH, reverse=True):
        title = it.get("title") or ""
        low = title.lower()
        if not title or any(p in low for p in _NEWS_NOISE) or any(p in low for p in _NEWS_FILLER):
            continue
        if sum(1 for ch in title if ord(ch) > 0x24F) > 0.3 * len(title):  # mostly non-Latin script (not readable for this app)
            continue
        key = re.sub(r"[^a-z0-9]+", "", low)[:70]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_news_cached(symbol: str, name: str, kind: str) -> list:
    """kind = 'stock' | 'etf' | 'market'. RAISES when no source worked, so a failure is never cached."""
    items, ok, errors = [], False, []
    if kind == "market":
        queries, first_days, wide_days, need = ["stock market today", "Wall Street stocks Dow S&P 500 Nasdaq"], 1, 3, 8
    else:
        word = "ETF" if kind == "etf" else "stock"
        queries = [f'"{name}" {word}' if name else f"{symbol} {word}"]
        first_days, wide_days, need = 7, 30, 5
    try:
        for q in queries:
            items += _google_news(q, first_days)
        if len(items) < need:
            for q in queries:
                items += _google_news(q, wide_days)
        ok = True
    except Exception as exc:
        errors.append(exc)
    if (not ok) or len(items) < 3:
        try:
            items += _yahoo_news("^GSPC" if kind == "market" else symbol)
            ok = True
        except Exception as exc:
            errors.append(exc)
    if not ok:
        raise RuntimeError(f"no news source answered: {errors[:1]}")
    if kind != "market" and not name:
        pat = re.compile(rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])")
        items = [i for i in items if pat.search(i.get("title") or "")]
    return _dedupe_news(items)


def get_news(symbol: str, name: str = "", kind: str = "stock"):
    """Returns (headlines, worked). worked=False means no source answered (shown to the user as a retry notice)."""
    key = ("news", symbol, kind)
    t = _NEG_CACHE.get(key)
    if t is not None and time.time() - t < _NEG_TTL:
        return [], False
    try:
        return _fetch_news_cached(symbol, _clean_company_name(name), kind), True
    except Exception:
        _NEG_CACHE[key] = time.time()
        return [], False


def _news_ago(dt) -> str:
    if not dt:
        return ""
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    d = int(secs // 86400)
    return f"{d} day{'s' if d > 1 else ''} ago"


def _md_safe(text: str) -> str:
    """Escape characters that Streamlit markdown would treat as formatting or math (headlines often contain dollar signs)."""
    return re.sub(r"([\\`*_~$\[\]])", r"\\\1", text or "")


def render_headlines(items: list, limit: int = 15, max_age_days=None):
    now = datetime.now(timezone.utc)
    shown = []
    for it in items:
        if max_age_days is not None and it.get("published") and (now - it["published"]).days >= max_age_days:
            continue
        shown.append(it)
        if len(shown) >= limit:
            break
    if not shown:
        st.info("No headlines found for this period.")
        return
    lines = []
    for it in shown:
        url = (it.get("url") or "").replace(" ", "%20").replace(")", "%29")
        meta = ", ".join(x for x in [_md_safe(it.get("source") or ""), _news_ago(it.get("published"))] if x)
        lines.append(f"- [{_md_safe(it['title'])}]({url}) - {meta}" if url else f"- {_md_safe(it['title'])} - {meta}")
    st.markdown("\n".join(lines))


_MKT_INDEX = [
    ("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones"), ("^RUT", "Russell 2000 (small companies)"),
    ("^VIX", "VIX (fear gauge)"), ("^TNX", "10-year Treasury yield"), ("CL=F", "Oil (WTI)"), ("GC=F", "Gold"),
]
_MKT_SECTORS = [
    ("XLK", "Technology"), ("XLC", "Communication"), ("XLY", "Consumer discretionary"), ("XLP", "Consumer staples"),
    ("XLE", "Energy"), ("XLF", "Financials"), ("XLV", "Health care"), ("XLI", "Industrials"),
    ("XLB", "Materials"), ("XLRE", "Real estate"), ("XLU", "Utilities"),
]


def _download_closes(symbols: list, days: int) -> pd.DataFrame:
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=days)
    data = yf.download(symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d",
                       auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise RuntimeError("no price data")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    return close


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_market_snapshot_cached() -> dict:
    """Index, sector and trend numbers from Yahoo daily prices. RAISES when the S&P 500 is missing (so failures are not cached)."""
    close = _download_closes([s for s, _ in _MKT_INDEX + _MKT_SECTORS], 420)
    if "^GSPC" not in close.columns or close["^GSPC"].dropna().shape[0] < 2:
        raise RuntimeError("S&P 500 data missing")

    def rows(pairs):
        out = []
        for sym, label in pairs:
            if sym not in close.columns:
                continue
            s = close[sym].dropna()
            if len(s) < 2:
                continue
            last, prev = float(s.iloc[-1]), float(s.iloc[-2])
            if sym == "^TNX" and last > 20:  # some feeds quote the yield times 10
                last, prev = last / 10, prev / 10
            out.append({"Symbol": sym, "Name": label, "Last": last, "Prev": prev,
                        "Change %": (last / prev - 1) * 100 if prev else None, "Change pts": last - prev})
        return pd.DataFrame(out)

    sp = close["^GSPC"].dropna()
    last = float(sp.iloc[-1])
    ma200 = float(sp.tail(200).mean()) if len(sp) >= 150 else None
    hi52 = float(sp.tail(252).max())
    return {
        "indices": rows(_MKT_INDEX), "sectors": rows(_MKT_SECTORS),
        "sp": {"last": last, "ma200": ma200, "pct_vs_200": (last / ma200 - 1) * 100 if ma200 else None,
               "high52": hi52, "drawdown": (last / hi52 - 1) * 100},
        "asof": sp.index[-1].strftime("%Y-%m-%d"),
    }


def get_market_snapshot():
    key = ("market_snapshot", "")
    t = _NEG_CACHE.get(key)
    if t is not None and time.time() - t < _NEG_TTL:
        return None
    try:
        return _fetch_market_snapshot_cached()
    except Exception:
        _NEG_CACHE[key] = time.time()
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_moves_cached(symbols: tuple) -> pd.DataFrame:
    close = _download_closes(list(symbols), 12)
    out = []
    for sym in close.columns:
        s = close[sym].dropna()
        if len(s) >= 2 and float(s.iloc[-2]) > 0:
            out.append({"Symbol": sym, "Price": float(s.iloc[-1]), "Change %": (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100})
    if not out:
        raise RuntimeError("no moves")
    return pd.DataFrame(out)


def get_moves(symbols: list):
    key = ("moves", "")
    t = _NEG_CACHE.get(key)
    if t is not None and time.time() - t < _NEG_TTL:
        return None
    try:
        return _fetch_moves_cached(tuple(symbols))
    except Exception:
        _NEG_CACHE[key] = time.time()
        return None


def _snap_change(snap, symbol: str):
    if not snap:
        return None
    df = snap["indices"]
    row = df[df["Symbol"] == symbol]
    if row.empty or pd.isna(row.iloc[0]["Change %"]):
        return None
    return float(row.iloc[0]["Change %"])


def _move_words(x: float) -> str:
    if abs(x) < 0.15:
        return "about flat"
    return f"{'up' if x > 0 else 'down'} {abs(x):.1f}%"


def market_reading(snap: dict, moves):
    """Plain-language reading of the numbers. Returns (bullets, is_big_move). It describes WHAT moved; the headlines give the reasons."""
    ind = snap["indices"].set_index("Symbol")
    sec = snap["sectors"].sort_values("Change %", ascending=False)
    bullets = []

    def chg(sym):
        return float(ind.loc[sym, "Change %"]) if sym in ind.index and pd.notna(ind.loc[sym, "Change %"]) else None

    sp_c, nas_c, dow_c = chg("^GSPC"), chg("^IXIC"), chg("^DJI")
    parts = [f"{n} {_move_words(v)}" for n, v in (("S&P 500", sp_c), ("Nasdaq", nas_c), ("Dow", dow_c)) if v is not None]
    line = "; ".join(parts) + "."
    if nas_c is not None and dow_c is not None:
        if nas_c - dow_c <= -1:
            line += " The technology-heavy Nasdaq is doing much worse than the Dow, so the selling is concentrated in growth and tech names."
        elif nas_c - dow_c >= 1:
            line += " The Nasdaq is doing much better than the Dow, so growth and tech names are leading."
    bullets.append(line)

    vix_last = vix_c = None
    if "^VIX" in ind.index:
        vix_last, vix_c = float(ind.loc["^VIX", "Last"]), chg("^VIX")
        band = "calm" if vix_last < 15 else "normal" if vix_last < 20 else "elevated (investors are nervous)" if vix_last < 30 else "high (stress)"
        bullets.append(f"The VIX fear gauge is {vix_last:.1f}, which is {band}.")
    if "^TNX" in ind.index:
        bps = float(ind.loc["^TNX", "Change pts"]) * 100
        bullets.append(f"The 10-year Treasury yield is {float(ind.loc['^TNX', 'Last']):.2f}% ({bps:+.0f} basis points today). "
                       "Rising yields tend to pressure growth stocks; falling yields tend to help them.")
    for sym, label in (("CL=F", "Oil"), ("GC=F", "Gold")):
        v = chg(sym)
        if v is not None and abs(v) >= 2:
            bullets.append(f"{label} is {_move_words(v)}, a large move.")
    if not sec.empty:
        up = int((sec["Change %"] > 0).sum())
        best, worst = sec.iloc[:2], sec.iloc[-2:]
        fmt_s = lambda df_: ", ".join(f"{row['Name']} ({row['Change %']:+.1f}%)" for _, row in df_.iterrows())
        bullets.append(f"{up} of {len(sec)} sectors are up. Strongest: {fmt_s(best)}. Weakest: {fmt_s(worst[::-1])}.")
    spd = snap["sp"]
    trend = ""
    if spd.get("pct_vs_200") is not None:
        trend = f"The S&P 500 is {abs(spd['pct_vs_200']):.1f}% {'above' if spd['pct_vs_200'] >= 0 else 'below'} its 200-day average. "
    dd = spd["drawdown"]
    kind = "near its 52-week high" if dd > -5 else "in a pullback (5-10% off the high)" if dd > -10 else "in a correction (10-20% off the high)" if dd > -20 else "in bear-market territory (20%+ off the high)"
    bullets.append(f"{trend}It is {abs(dd):.1f}% below its 52-week high, {kind}.")
    if moves is not None and not moves.empty:
        down = int((moves["Change %"] < 0).sum())
        worst3 = moves.sort_values("Change %").head(3)
        bullets.append(f"{down} of your {len(moves)} favorites are down today. Biggest drops: "
                       + ", ".join(f"{row['Symbol']} ({row['Change %']:+.1f}%)" for _, row in worst3.iterrows()) + ".")
    big = bool((sp_c is not None and sp_c <= -2) or (nas_c is not None and nas_c <= -3)
               or (vix_last is not None and vix_last >= 30) or (vix_c is not None and vix_c >= 15))
    return bullets, big


def _colour_col(raw: pd.Series, invert=()) -> list:
    """Green for positive, red for negative; rows in invert (VIX, yields) use the opposite colours."""
    out = []
    for i, v in raw.items():
        if pd.isna(v) or abs(v) < 1e-9:
            out.append("")
            continue
        good = v > 0
        if i in invert:
            good = not good
        out.append("color: #2e9e5b; font-weight: 600" if good else "color: #d64545; font-weight: 600")
    return out


def render_market_tab(favorites: list):
    st.subheader("Market today")
    st.caption("What moved today, in plain numbers, and the day's headlines. Prices are Yahoo's latest daily data and may lag by some minutes.")
    snap = get_market_snapshot()
    if snap is None:
        st.warning("Yahoo did not return market prices right now (usually a short rate limit). Headlines below may still work.")
        if st.button("Try again", key="retry_market"):
            _clear_data_caches()
            st.rerun()
    moves = get_moves(favorites) if favorites else None
    if snap is not None:
        bullets, big = market_reading(snap, moves)
        if big:
            st.warning("**Big market move today.** One day rarely changes the value of a good business. Before acting, open the Entry and exit plan tab: "
                       "staged buying exists for days like this. Check that your Quality scores still hold, and read the headlines below for the cause.")
        st.markdown("**Automatic reading of the numbers** (what moved, not why):")
        st.markdown("\n".join(f"- {b}" for b in bullets))

        ind = snap["indices"].copy()
        show = pd.DataFrame({
            "Market": ind["Name"],
            "Level": [f"{v:.2f}%" if s == "^TNX" else f"{v:,.2f}" for v, s in zip(ind["Last"], ind["Symbol"])],
            "Change today": [f"{p * 100:+.0f} basis points" if s == "^TNX" else f"{c:+.2f}%" for p, c, s in zip(ind["Change pts"], ind["Change %"], ind["Symbol"])],
        })
        raw = pd.Series([p if s == "^TNX" else c for p, c, s in zip(ind["Change pts"], ind["Change %"], ind["Symbol"])], index=show.index)
        inv = [i for i, s in zip(show.index, ind["Symbol"]) if s in ("^VIX", "^TNX")]
        styler = show.style.apply(lambda col_: _colour_col(raw, inv), subset=["Change today"])
        st.dataframe(styler, hide_index=True, use_container_width=True)

        sec = snap["sectors"].sort_values("Change %")
        if not sec.empty:
            st.markdown("**Sectors today** (sector funds)")
            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.barh(sec["Name"], sec["Change %"], color=["#2e9e5b" if v >= 0 else "#d64545" for v in sec["Change %"]])
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Change today (%)")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption(f"Latest daily bar: {snap['asof']}. 'Today' means the latest session in the data.")

    if moves is not None and not moves.empty:
        st.markdown("**Your favorites: biggest moves**")
        c1, c2 = st.columns(2)
        fmt = {"Price": "{:,.2f}", "Change %": "{:+.2f}%"}
        with c1:
            st.caption("Biggest drops")
            st.dataframe(moves.sort_values("Change %").head(5).style.format(fmt), hide_index=True, use_container_width=True)
        with c2:
            st.caption("Biggest gains")
            st.dataframe(moves.sort_values("Change %", ascending=False).head(5).style.format(fmt), hide_index=True, use_container_width=True)

    st.markdown("**Market headlines**")
    items, ok = get_news("^GSPC", "", "market")
    if not ok:
        st.warning("No news source answered right now. Try again in a minute.")
        if st.button("Try again", key="retry_market_news"):
            _clear_data_caches()
            st.rerun()
    else:
        render_headlines(items, limit=12, max_age_days=3)
    st.caption("Headlines come from Google News (links go to the publisher), with Yahoo as a backup. Only titles are shown. "
               "A headline's explanation can be wrong or incomplete, so use it as a starting point. Not investment advice.")


def _last_day_change(df: pd.DataFrame):
    """% change of the latest price versus the last bar of the previous trading day."""
    try:
        s = df["Close"].dropna()
        days = pd.Series(s.values, index=pd.DatetimeIndex(s.index).normalize())
        prev = days[days.index < days.index[-1]]
        if prev.empty:
            return None
        return (float(s.iloc[-1]) / float(prev.iloc[-1]) - 1) * 100
    except Exception:
        return None


def render_news_tab(stock: str, indicator_data: pd.DataFrame):
    inst = get_instrument_info(stock)
    name = inst.get("name") or ""
    kind = "etf" if inst.get("type") in ETF_TYPES else "stock"
    st.subheader(f"News - {stock}" + (f" ({name})" if name else ""))
    chg = _last_day_change(indicator_data)
    sp_c = _snap_change(get_market_snapshot(), "^GSPC")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{stock} today", f"{chg:+.2f}%" if chg is not None else "N/A")
    m2.metric("S&P 500 today", f"{sp_c:+.2f}%" if sp_c is not None else "N/A")
    m3.metric("Difference", f"{chg - sp_c:+.2f} points" if chg is not None and sp_c is not None else "N/A")
    if chg is not None and sp_c is not None:
        if abs(chg) >= 3 and abs(sp_c) < 1:
            st.info(f"**Stock-specific move.** {stock} moved much more than the market, so the reason is probably company news. Read the headlines below.")
        elif abs(chg) >= 2 and abs(sp_c) >= 1 and chg * sp_c > 0:
            extra = chg - sp_c
            st.info("**Mostly a market-wide move.** " + (f"{stock} moved {abs(extra):.1f} points more than the market, so check the headlines for anything specific." if abs(extra) >= 2
                                                           else "The stock is moving with the market. See the Market today tab."))
        elif abs(chg) < 2:
            st.caption("An ordinary day for this stock (under 2% either way).")
    if not name:
        st.caption("Yahoo did not give the company name right now, so the search used the ticker only and may include other companies. Try again later for a cleaner list.")
    period = st.radio("Show headlines from", ["Last 3 days", "Last 7 days", "Everything found"], index=1, horizontal=True, key="news_period")
    max_age = {"Last 3 days": 3, "Last 7 days": 7}.get(period)
    items, ok = get_news(stock, name, kind)
    if not ok:
        st.warning("No news source answered right now (Google News and Yahoo both failed). This is usually temporary.")
        if st.button("Try again", key="retry_news"):
            _clear_data_caches()
            st.rerun()
    else:
        render_headlines(items, limit=15, max_age_days=max_age)
    st.caption("Headlines come from Google News (links go to the publisher), with Yahoo as a backup. Only titles are shown; open a link to read the story. "
               "Searches use the company name, but a headline can still be about a different company with a similar name, so check before acting. "
               "News is context for your decision, not a buy or sell signal. Not investment advice.")


# ---------------------------------------------------------------------------
# Ask Charlie (the AI assistant): the fixed menu of tools the model may use (it never runs code), the chart renderer and the chat tab.
# Connection, limits and the tool loop live in ai_assistant.py.
# ---------------------------------------------------------------------------
_SYMBOL_ALIASES = {
    "S&P 500": "^GSPC", "S&P500": "^GSPC", "S&P": "^GSPC", "SP500": "^GSPC", "SPX": "^GSPC", "NASDAQ": "^IXIC", "NASDAQ COMPOSITE": "^IXIC",
    "DOW": "^DJI", "DOW JONES": "^DJI", "RUSSELL 2000": "^RUT", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F", "BITCOIN": "BTC-USD",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")
_CHART_DAYS = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653}
_TOOL_LABELS = {
    "get_stock_summary": "Buy checklist", "compare_stocks": "comparison", "get_price_stats": "price statistics", "get_news_headlines": "news",
    "get_market_today": "market numbers", "get_entry_plan": "entry plan", "get_exit_plan": "exit plan",
    "get_favorites_overview": "favorites scan", "get_my_portfolio": "your portfolio", "show_chart": "chart", "open_stock": "load stock",
}


def _secret(name: str) -> str:
    """Read a secret from Streamlit's secrets (Cloud: app Settings > Secrets), falling back to an environment variable for local use."""
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.environ.get(name, "") or "")


def _ai_symbol(raw) -> str:
    t = str(raw or "").strip().upper()
    t = _SYMBOL_ALIASES.get(t, t)
    if not _SYMBOL_RE.match(t):
        raise ValueError(f"'{raw}' is not a valid ticker symbol")
    return t


def _r_default_value() -> float:
    sp3 = sp500_trailing_return(3)
    return round(min(sp3 + 2.0, 15.0), 1) if sp3 is not None else 12.0


@st.cache_data(ttl=1800, show_spinner=False)
def _score_or_raise(stock: str, r_default: float) -> dict:
    """Like score_symbol_cached, but a failed lookup raises so it is never cached."""
    row = score_symbol(stock, r_default)
    if row.get("error"):
        raise RuntimeError(row["error"])
    return row


def _rn(x, d=1):
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), d)
    except Exception:
        return None


def _ai_row(symbol) -> tuple:
    sym = _ai_symbol(symbol)
    r = _r_default_value()
    return sym, _score_or_raise(sym, r), r


def _compact_row(row: dict, r: float) -> dict:
    return {
        "symbol": row["symbol"], "name": row["name"], "type": row["type"], "sector": row["sector"], "stock_type": row["stock_type"] or None,
        "price": _rn(row["price"], 2), "price_date": row["price_date"], "verdict": row["verdict"],
        "scores_pct": {"quality": _rn(row["quality_pct"], 0), "valuation": _rn(row["valuation_pct"], 0), "timing": _rn(row["timing_pct"], 0)},
        "checks_scored": {"quality": row["quality_n"], "valuation": row["valuation_n"], "timing": row["timing_n"]},
        "fair_value_eps10": _rn(row["fair_eps10"], 2), "buy_price_eps10": _rn(row["buy_eps10"], 2),
        "fair_value_dcf": _rn(row["fair_dcf"], 2), "buy_price_dcf": _rn(row["buy_dcf"], 2),
        "avg_upside_to_fair_pct": _rn(row["upside_pct"]), "buy_zone": row["buy_zone"] or "not in zone",
        "price_vs_buy_price_pct": _rn(row["buy_gap_pct"]), "overbought_signals_of_3": row["sell_signals"], "rsi_daily": _rn(row["rsi_daily"]),
        "trim_level": row["trim_level"], "trim_reason": row["trim_reason"],
        "assumptions": {"growth_pct": _rn(row["g_pct"]), "exit_multiple": _rn(row["k"]), "margin_of_safety_pct": _rn(row["mos_pct"]), "required_return_pct": r},
        "ranked_in_top10_lists": bool(row["eligible"]),
    }


def _ai_price_stats(symbol) -> dict:
    sym = _ai_symbol(symbol)
    daily = download_daily(sym, years=2)
    if daily is None or daily.empty:
        raise ValueError("no price history")
    close = normalize_ohlcv_columns(daily)["Close"].astype(float).dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) is not None else pd.to_datetime(close.index)
    last = float(close.iloc[-1])

    def back(n):
        return _rn((last / float(close.iloc[-1 - n]) - 1) * 100) if len(close) > n else None

    prev_year = close[close.index.year < close.index[-1].year]
    ytd = _rn((last / float(prev_year.iloc[-1]) - 1) * 100) if not prev_year.empty else None
    hi, lo = float(close.tail(252).max()), float(close.tail(252).min())
    out = {"symbol": sym, "price": _rn(last, 2), "date": str(close.index[-1].date()),
           "change_pct": {"1d": back(1), "1w": back(5), "1m": back(21), "3m": back(63), "6m": back(126), "1y": back(252), "ytd": ytd},
           "high_52w": _rn(hi, 2), "low_52w": _rn(lo, 2), "pct_from_52w_high": _rn((last / hi - 1) * 100)}
    if len(close) >= 50:
        out["pct_vs_50d_avg"] = _rn((last / float(close.tail(50).mean()) - 1) * 100)
    if len(close) >= 200:
        out["pct_vs_200d_avg"] = _rn((last / float(close.tail(200).mean()) - 1) * 100)
    return out


def _ai_news(symbol) -> dict:
    s = str(symbol or "").strip().upper()
    if s in ("MARKET", "STOCK MARKET", "THE MARKET"):
        sym, items, ok = "MARKET", *get_news("^GSPC", "", "market")
    else:
        sym = _ai_symbol(s)
        inst = get_instrument_info(sym)
        items, ok = get_news(sym, inst.get("name") or "", "etf" if inst.get("type") in ETF_TYPES else "stock")
    if not ok:
        raise RuntimeError("no news source answered right now")
    now = datetime.now(timezone.utc)
    recent = [i for i in items if not i.get("published") or (now - i["published"]).days < 7][:8]
    return {"symbol": sym, "headlines": [{"title": i["title"], "source": i["source"], "age": _news_ago(i.get("published"))} for i in recent]}


def _ai_market(favorites: list) -> dict:
    snap = get_market_snapshot()
    if snap is None:
        raise RuntimeError("market prices are not available from Yahoo right now")
    moves = get_moves(favorites) if favorites else None
    bullets, big = market_reading(snap, moves)
    ind = snap["indices"]
    return {
        "as_of": snap["asof"], "big_move_today": big, "reading": bullets,
        "indices": [{"name": r["Name"], "level": _rn(r["Last"], 2), "change_pct": _rn(r["Change %"], 2),
                     **({"change_basis_points": _rn(r["Change pts"] * 100, 0)} if r["Symbol"] == "^TNX" else {})} for _, r in ind.iterrows()],
        "sectors": [{"name": r["Name"], "change_pct": _rn(r["Change %"], 2)} for _, r in snap["sectors"].sort_values("Change %").iterrows()],
    }


def _ai_entry_plan(symbol, total_amount, tranches=3, style="value", step_pct=7.5) -> dict:
    sym, row, _ = _ai_row(symbol)
    price = float(row["price"])
    buy = row["buy_eps10"] or row["buy_dcf"]
    total = float(total_amount)
    if total <= 0:
        raise ValueError("total_amount must be above zero")
    n = int(max(2, min(6, int(tranches or 3))))
    step = float(max(1.0, min(25.0, float(step_pct or 7.5))))
    note = ""
    if str(style).lower() == "value" and buy and buy > 0:
        anchor, used = float(buy), "value (starts at the model buy price)"
    else:
        anchor, used = price, "pullback (starts at today's price)"
        if str(style).lower() == "value":
            note = "No model buy price for this stock, so the ladder starts at today's price."
    rows = build_entry_ladder(anchor, n, step, total, price)
    out = {"symbol": sym, "current_price": _rn(price, 2), "style": used, "step_pct": step, "note": note, "quality_pct": _rn(row["quality_pct"], 0),
           "tranches": [{"tranche": r["Tranche"], "buy_at_or_below": _rn(r["Buy at or below"], 2), "distance_from_today_pct": _rn(r["Distance from today %"]),
                         "amount": _rn(r["Amount"], 2), "approx_shares": _rn(r["Shares"], 2), "status": r["Status"]} for r in rows]}
    if row["quality_pct"] is not None and row["quality_pct"] < 50:
        out["warning"] = "Quality is below 50%, so staged buying only averages down into a weaker business. Re-check the business first."
    return out


def _ai_exit_plan(symbol, shares=None, avg_cost=None) -> dict:
    sym, row, _ = _ai_row(symbol)
    price = float(row["price"])
    shares = float(shares or 0)
    avg_cost = float(avg_cost or 0)
    fair = row["fair_eps10"] or row["fair_dcf"]
    if avg_cost > 0:
        base, levels, basis = avg_cost, [30, 60, 100], "your average cost (steps are percent above cost)"
    elif fair and fair > 0:
        base, levels, basis = float(fair), [0, 20, 40], "the model fair value (steps are percent above fair value)"
    else:
        raise ValueError("no model fair value for this stock and no average cost was given")
    rows = build_exit_ladder(base, levels, [10, 15, 25], price, shares, avg_cost)
    return {"symbol": sym, "current_price": _rn(price, 2), "based_on": basis, "overbought_signals_of_3": row["sell_signals"], "trim_level": row["trim_level"],
            "steps": [{"step": r["Step"], "sell_when_price_reaches": _rn(r["Sell when price reaches"], 2), "distance_from_today_pct": _rn(r["Distance from today %"]),
                       "sell_pct_of_position": r["Sell % of position"], "shares_to_sell": _rn(r["Shares to sell"], 2), "cash_received": _rn(r["Cash received"], 2),
                       "gain_locked_in": _rn(r["Gain locked in"], 2), "status": r["Status"]} for r in rows]}


def _ai_favorites(favorites: list) -> dict:
    df, state = load_rankings()
    if df is None or df.empty:
        raise RuntimeError("no daily scan results are saved yet")
    fav = set(favorites)
    f = df[df["symbol"].isin(fav)]

    def pack(d, cols):
        return [{c: (_rn(r[c], 1) if isinstance(r[c], float) else r[c]) for c in cols} for _, r in d.iterrows()]

    buy, avoid = rank_results(df)
    return {
        "scan_date": state.get("asof"), "favorites_scanned": int(len(f)),
        "in_buy_zone": pack(f[f["buy_zone"] == "In zone"], ["symbol", "price", "buy_eps10", "buy_gap_pct", "quality_pct", "verdict_short"]),
        "near_buy_zone": pack(f[f["buy_zone"] == "Near"], ["symbol", "price", "buy_eps10", "buy_gap_pct", "quality_pct", "verdict_short"]),
        "trim_or_review": pack(f[f["trim_level"].isin(["Trim", "Review"])], ["symbol", "trim_level", "trim_reason", "quality_pct"]),
        "top10_buy": [{"symbol": r["symbol"], "quality": _rn(r["quality_pct"], 0), "valuation": _rn(r["valuation_pct"], 0), "timing": _rn(r["timing_pct"], 0),
                       "favorite": r["symbol"] in fav} for _, r in buy.iterrows()],
        "top10_avoid": [{"symbol": r["symbol"], "quality": _rn(r["quality_pct"], 0), "valuation": _rn(r["valuation_pct"], 0), "favorite": r["symbol"] in fav}
                        for _, r in avoid.iterrows()],
    }


def _ai_portfolio() -> dict:
    if not st.session_state.get("ai_share_portfolio"):
        raise PermissionError("the user has not chosen to share their portfolio with Charlie")
    pf = st.session_state.get("pf_base")
    if pf is None or pf.empty:
        raise ValueError("the My portfolio tab has no positions entered yet")
    out = []
    for _, r in pf.dropna(subset=["Symbol"]).head(15).iterrows():
        sym = str(r["Symbol"]).strip().upper()
        shares, cost = float(r.get("Shares") or 0), float(r.get("Avg cost") or 0)
        try:
            px = _ai_price_stats(sym)["price"]
        except Exception:
            px = None
        out.append({"symbol": sym, "shares": shares, "avg_cost": cost, "price": px,
                    "gain_pct": _rn((px / cost - 1) * 100) if px and cost else None, "value": _rn(px * shares, 2) if px else None})
    total = sum(x["value"] or 0 for x in out)
    for x in out:
        x["weight_pct"] = _rn((x["value"] or 0) / total * 100) if total else None
    return {"positions": out, "total_value": _rn(total, 2)}


@st.cache_data(ttl=900, show_spinner=False)
def _ai_chart_data(symbols: tuple, days: int) -> pd.DataFrame:
    return _download_closes(list(symbols), days + 320)


def _chart_window(spec: dict, close: pd.DataFrame) -> pd.DataFrame:
    start = close.index[-1] - pd.Timedelta(days=spec["days"])
    return close[close.index >= start]


def _ai_show_chart(ctx: dict, symbols, period="1y", mode=None) -> dict:
    syms = []
    for s in list(symbols or [])[:4]:
        t = _ai_symbol(s)
        if t not in syms:
            syms.append(t)
    if not syms:
        raise ValueError("no symbols given")
    period = period if period in _CHART_DAYS or period == "ytd" else "1y"
    days = (datetime.now() - datetime(datetime.now().year, 1, 1)).days + 2 if period == "ytd" else _CHART_DAYS[period]
    mode = mode if mode in ("price", "compare", "drawdown") else ("price" if len(syms) == 1 else "compare")
    if mode == "price" and len(syms) > 1:
        mode = "compare"
    spec = {"symbols": syms, "period": period, "days": days, "mode": mode}
    close = _ai_chart_data(tuple(syms), days)
    win = _chart_window(spec, close)
    rets, missing = {}, []
    for s in syms:
        if s in win.columns and win[s].dropna().shape[0] >= 2:
            col = win[s].dropna()
            rets[s] = _rn((float(col.iloc[-1]) / float(col.iloc[0]) - 1) * 100)
        else:
            missing.append(s)
    if not rets:
        raise ValueError("Yahoo returned no price history for those symbols")
    spec["symbols"] = [s for s in syms if s in rets]
    ctx["charts"].append(spec)
    return {"ok": True, "mode": mode, "period": period, "total_return_pct_over_period": rets, "no_data_for": missing,
            "note": "The chart is displayed to the user below your answer."}


def render_ai_chart(spec: dict):
    try:
        close = _ai_chart_data(tuple(spec["symbols"]), spec["days"])
    except Exception:
        st.caption("Chart data is not available right now.")
        return
    win = _chart_window(spec, close)
    if any(sym not in win.columns or win[sym].dropna().shape[0] < 2 for sym in spec["symbols"]):
        st.caption("Chart data is not available for one of those symbols right now.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    mode = spec["mode"]
    if mode == "price":
        s = spec["symbols"][0]
        full = close[s].dropna()
        w = win[s].dropna()
        ax.plot(w.index, w.values, color="#1f77b4", linewidth=1.6, label=s)
        for n_, col_ in ((50, "#ff7f0e"), (200, "#7f7f7f")):
            ma = full.rolling(n_).mean()
            ma = ma[ma.index >= w.index[0]]
            if not ma.dropna().empty:
                ax.plot(ma.index, ma.values, color=col_, linewidth=1.0, linestyle="--", label=f"{n_}-day average")
        ax.set_ylabel("Price")
        ax.set_title(f"{s} - {spec['period']}")
    elif mode == "drawdown":
        for s in spec["symbols"]:
            w = win[s].dropna()
            if len(w) > 1:
                ax.plot(w.index, (w / w.cummax() - 1) * 100, linewidth=1.4, label=s)
        ax.set_ylabel("% below the running high")
        ax.set_title(f"Fall from the high - {spec['period']}")
    else:
        for s in spec["symbols"]:
            w = win[s].dropna()
            if len(w) > 1:
                ax.plot(w.index, w / float(w.iloc[0]) * 100, linewidth=1.5, label=s)
        ax.axhline(100, color="gray", linewidth=0.8)
        ax.set_ylabel("Start = 100")
        ax.set_title(f"Performance compared - {spec['period']}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _build_ai_tools(ctx: dict, favorites: list) -> dict:
    def summary(symbol):
        sym, row, r = _ai_row(symbol)
        return _compact_row(row, r)

    def compare(symbols):
        out = []
        for s in list(symbols or [])[:4]:
            try:
                out.append(summary(s))
            except Exception as exc:
                out.append({"symbol": str(s), "error": str(exc)[:120]})
        return {"stocks": out}

    def open_stock(symbol):
        sym = _ai_symbol(symbol)
        ctx["actions"].append({"type": "open_stock", "symbol": sym})
        return {"ok": True, "note": f"{sym} will be loaded in the dashboard's other tabs."}

    return {
        "get_stock_summary": summary, "compare_stocks": compare, "get_price_stats": _ai_price_stats, "get_news_headlines": _ai_news,
        "get_market_today": lambda: _ai_market(favorites), "get_entry_plan": _ai_entry_plan, "get_exit_plan": _ai_exit_plan,
        "get_favorites_overview": lambda: _ai_favorites(favorites), "get_my_portfolio": _ai_portfolio,
        "show_chart": lambda symbols, period="1y", mode=None: _ai_show_chart(ctx, symbols, period, mode), "open_stock": open_stock,
    }


def _tools_used_text(tools: list) -> str:
    seen = []
    for t in tools:
        label = _TOOL_LABELS.get(t["tool"], t["tool"])
        a = t.get("args") or {}
        who = a.get("symbol") or ", ".join(map(str, a.get("symbols") or []))
        item = f"{label} ({who})" if who else label
        if item not in seen:
            seen.append(item)
    return ", ".join(seen)


def _render_ai_setup(has_key: bool, has_code: bool):
    st.info("Ask Charlie is switched off until two private settings are added. This keeps strangers from using up your free AI quota, because this app's web address is public.")
    st.markdown(
        "**One-time setup (about 5 minutes)**\n"
        "1. Go to **aistudio.google.com**, sign in with a Google account, click **Get API key**, and create a key (free, no card).\n"
        "2. In Streamlit Cloud open your app, then **Settings > Secrets**, and paste these two lines (put your own key and a code of your choice inside the quotes):\n"
    )
    st.code('GEMINI_API_KEY = "paste-your-key-here"\nAI_ACCESS_CODE = "choose-a-code-only-you-know"', language="toml")
    st.markdown(
        "3. Save. The app restarts, and this tab will ask for your access code.\n\n"
        f"Status: API key {'found' if has_key else 'missing'}; access code {'found' if has_code else 'missing'}.\n\n"
        "Never type the key into any file that you upload to GitHub. On your own computer you can instead set the same two names as environment variables."
    )


def render_ai_tab(active_stock: str, favorites: list):
    st.subheader("Ask Charlie")
    st.caption("Named in honor of Charlie Munger. This is an AI that looks up your dashboard's own numbers; it is not him and does not speak for him.")
    key, code = _secret("GEMINI_API_KEY"), _secret("AI_ACCESS_CODE")
    if not key or not code:
        _render_ai_setup(bool(key), bool(code))
        return
    if not st.session_state.get("ai_unlocked"):
        with st.form("ai_unlock"):
            pw = st.text_input("Access code", type="password")
            go = st.form_submit_button("Unlock")
        if go:
            ok, msg = ai_assistant.check_access_code(pw, code)
            if ok:
                st.session_state["ai_unlocked"] = True
                st.rerun()
            else:
                st.error(msg)
        st.caption("Ask about a stock, the market, an entry or exit plan, or request a chart. The code keeps strangers away from your AI quota.")
        return

    st.session_state.setdefault("ai_msgs", [])
    st.session_state.setdefault("ai_hist", [])
    st.session_state.setdefault("ai_count", 0)
    left = ai_assistant.MAX_MESSAGES_PER_SESSION - st.session_state["ai_count"]

    st.caption("Ask in your own words. Charlie answers using this dashboard's own numbers (it looks them up, it does not guess). "
               "It can compare stocks, summarize news, build entry and exit plans, draw charts, and load a stock into the other tabs. Not investment advice.")
    top = st.columns([3, 1])
    with top[0]:
        st.checkbox("Let Charlie see my portfolio (shares and cost from the My portfolio tab)", key="ai_share_portfolio",
                    help="Off by default. Free-tier Google AI may keep and review prompts, so share holdings only if you are comfortable with that.")
    with top[1]:
        if st.button("Clear chat", key="ai_clear"):
            st.session_state["ai_msgs"], st.session_state["ai_hist"] = [], []
            st.rerun()

    stock = active_stock or "AAPL"
    quick = {
        f"Explain {stock}": f"Explain {stock} to me in plain English: is it a good business, is the price attractive, and what does the Buy checklist say?",
        "Why is the market moving?": "What is happening in the market today, and what do the headlines say about why?",
        f"{stock} news": f"Summarize the latest news for {stock} and tell me if anything looks important.",
        f"Entry plan for {stock}": f"Build a 3-tranche entry plan for {stock} for 10000 dollars.",
        f"Chart {stock} vs S&P 500": f"Show {stock} against the S&P 500 for the last year.",
        "My favorites": "Which of my favorites look most interesting right now, and why?",
    }
    qcols = st.columns(len(quick))
    clicked = None
    for col, (label, text) in zip(qcols, quick.items()):
        if col.button(label, key=f"ai_q_{label}", use_container_width=True):
            clicked = text

    def show_message(m):
        with st.chat_message(m["role"]):
            st.markdown(m["text"].replace("$", "\\$"))
            for spec in m.get("charts", []):
                render_ai_chart(spec)
            if m.get("tools"):
                st.caption("Looked up: " + m["tools"])

    for m in st.session_state["ai_msgs"]:
        show_message(m)

    typed = st.chat_input("Ask about a stock, the market, a plan, or a chart...", key="ai_input")
    prompt = clicked or typed
    st.caption(f"Messages left in this session: {max(left, 0)}")
    if not prompt:
        with st.expander("Check the AI connection"):
            if st.button("Run a quick test", key="ai_test"):
                try:
                    ai_assistant._generate(ai_assistant.gemini_transport(key), {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}],
                                                                              "generationConfig": {"maxOutputTokens": 20}})
                    st.success(f"Connected. Model in use: {ai_assistant._good_model['name']}.")
                except ai_assistant.AIError as exc:
                    st.error(exc.message)
            st.caption(f"Calls used today across the whole app: {ai_assistant.DAILY_CALL_CAP - ai_assistant.budget_left()} of {ai_assistant.DAILY_CALL_CAP} allowed.")
        return
    if left <= 0:
        st.warning("You have reached the message limit for this session. Click Clear chat or reload the page to start again.")
        return
    prompt = prompt.strip()[: ai_assistant.MAX_INPUT_CHARS]
    st.session_state["ai_count"] += 1
    user_m = {"role": "user", "text": prompt}
    show_message(user_m)
    ctx = {"charts": [], "actions": []}
    try:
        with st.chat_message("assistant"):
            with st.spinner("Looking things up..."):
                result = ai_assistant.chat_turn(
                    ai_assistant.gemini_transport(key),
                    ai_assistant.build_system_prompt(active_stock, datetime.now().strftime("%A %Y-%m-%d"), bool(st.session_state.get("ai_share_portfolio"))),
                    st.session_state["ai_hist"], prompt, ai_assistant.TOOL_DECLS, _build_ai_tools(ctx, favorites),
                )
            st.markdown(result["text"].replace("$", "\\$"))
            for spec in ctx["charts"]:
                render_ai_chart(spec)
            tools_txt = _tools_used_text(result["tools"])
            if tools_txt:
                st.caption("Looked up: " + tools_txt)
    except ai_assistant.AIError as exc:
        st.error(exc.message)
        return
    st.session_state["ai_hist"] = result["history"]
    st.session_state["ai_msgs"] += [user_m, {"role": "assistant", "text": result["text"], "charts": ctx["charts"], "tools": tools_txt}]
    opens = [a for a in ctx["actions"] if a.get("type") == "open_stock"]
    if opens:
        st.session_state["pending_ticker"] = opens[-1]["symbol"]
        st.rerun()


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
        st.session_state["favorites"] = load_favorites_file()
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

    # One ticker box: pick from the list (favorites first) or type any Yahoo symbol and press Enter.
    # Everything loads as soon as the box changes, so there is no separate "Load chart" step.
    pending = st.session_state.pop("pending_ticker", None)
    if pending:
        st.session_state["ticker_box"] = str(pending).strip().upper()
    if not st.session_state.get("ticker_box"):
        st.session_state["ticker_box"] = options[0]
    current_pick = str(st.session_state["ticker_box"]).strip()
    box_options = options if current_pick in options else options + [current_pick]

    col1, col2 = st.columns([2, 3])
    with col1:
        picked = st.selectbox(
            "Stock ticker",
            options=box_options,
            key="ticker_box",
            accept_new_options=True,
            help="Your favorites are listed first, then your stock list. Click the box and type to search. "
                 "To look up any other stock, type its Yahoo Finance symbol (for example BRK-B) and press Enter.",
        )
    with col2:
        st.write("")
        st.caption("Pick from the list or type any symbol and press Enter. Everything below updates right away.")

    stock = str(picked or "").strip().upper()
    st.session_state["active_stock"] = stock or None

    active_stock = st.session_state["active_stock"]

    if not active_stock:
        st.info("Pick or type a stock ticker above to see charts, business metrics and valuation.")
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

    # Fetch financial metrics once for both Business snapshot and Fair value tabs
    fin = fetch_financial_metrics(active_stock)
    analyst = fetch_analyst_view(active_stock)

    # Long-term context (quality, valuation history, trend). Each degrades gracefully if Yahoo data is missing.
    quality = fetch_quality_metrics(active_stock)
    valctx = fetch_valuation_context(active_stock)
    trend = compute_trend_context(download_daily(active_stock), download_daily("SPY"))
    if not fin and not quality.get("available") and get_instrument_info(active_stock)["type"] not in ETF_TYPES:
        render_data_gap_banner(active_stock)
    _scan_asof = load_rankings()[1].get("asof")
    st.caption(
        f"Data as of: prices to {pd.to_datetime(latest.name):%Y-%m-%d %H:%M} (Yahoo, refreshed about every 15 minutes); "
        f"company statements refreshed about hourly; last daily scan: {_scan_asof or 'not run yet'}."
    )

    # Default required return r = trailing 3-year S&P 500 annualized return + 2%, capped at 15% (fallback 12%)
    sp3 = sp500_trailing_return(3)
    r_default = round(min(sp3 + 2.0, 15.0), 1) if sp3 is not None else 12.0

    # Stock-specific starting assumptions; reset whenever the active ticker changes (r keeps any override)
    if st.session_state.get("val_defaults_for_stock") != active_stock:
        st.session_state["val_profile_choice"] = "Auto"
    profile_choice = st.session_state.get("val_profile_choice", "Auto")
    sugg = suggest_valuation_defaults(fin, quality, valctx, None if profile_choice == "Auto" else profile_choice)
    defaults_changed = st.session_state.get("val_defaults_for") != (active_stock, profile_choice)
    for key, val in (("val_g_pct", sugg["g"]), ("val_k_multiple", sugg["k"]), ("val_mos_pct", sugg["mos"])):
        if defaults_changed or key not in st.session_state:
            st.session_state[key] = float(val)
    if "val_r_pct" not in st.session_state:
        st.session_state["val_r_pct"] = float(r_default)
    st.session_state["val_defaults_for"] = (active_stock, profile_choice)
    st.session_state["val_defaults_for_stock"] = active_stock

    tab_charts, tab_lt, tab_fin, tab_val, tab_plan, tab_analyst, tab_news, tab_paper, tab_market, tab_ai, tab_cmp, tab_rank, tab_port, tab_fav, tab_explain = st.tabs(
        [
            "Price charts",
            "Buy checklist",
            "Business snapshot",
            "Fair value",
            "Entry and exit plan",
            "Analyst view",
            "News",
            "Test the signals",
            "Market today",
            "Ask Charlie",
            "Compare stocks",
            "Top 10 lists",
            "My portfolio",
            "My favorites",
            "How it works",
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
        st.subheader(f"Buy checklist - {active_stock}")
        st.caption(
            "Three pillars: business quality, valuation, and timing. Thresholds are rules of thumb "
            "(see How it works tab). Valuation checks use the assumptions from the Fair value tab."
        )
        if get_instrument_info(active_stock)["type"] in ETF_TYPES:
            st.info(
                "This is an ETF / fund. Quality and Valuation checks need company statements, so only the Timing "
                "pillar applies. Use the trend and timing checks, and judge the fund by what it holds."
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

    with tab_market:
        render_market_tab(favorites)

    with tab_ai:
        render_ai_tab(active_stock, favorites)

    with tab_cmp:
        render_compare_tab(active_stock, st.session_state["favorites"], r_default)

    with tab_rank:
        render_rankings_tab(st.session_state["favorites"], r_default)

    with tab_port:
        render_portfolio_tab(r_default)

    with tab_fin:
        render_business_snapshot(active_stock, current_price, fin, quality, valctx)

    with tab_val:
        st.subheader(f"Fair value - {active_stock}")
        st.metric("Current price (latest close)", f"${current_price:.2f}")
        if not fin or fin.get("EPS (1Y)") is None:
            st.info("Valuation inputs are not available (missing Net Income or Shares Issued data).")
        else:
            eps = fin.get("EPS (1Y)")
            ps1, ps2 = st.columns([1, 3])
            with ps1:
                st.selectbox(
                    "Stock type", ["Auto", *STOCK_TYPES], key="val_profile_choice",
                    help="Growth, cyclical (airlines, energy, metals), financial and core stocks get different growth and multiple limits. Auto picks from the data; choose one to override.",
                )
            with ps2:
                st.caption(
                    f"Detected: {sugg['auto_profile']} ({sugg['profile_reason']}). The starting growth, multiple and margin of safety "
                    "below follow the stock type; edit any input to override."
                )
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

    with tab_news:
        render_news_tab(active_stock, indicator_data)

    with tab_paper:
        st.subheader(f"Test the signals – {active_stock}")
        st.caption(
            "Simulates: buy when price is below the lower Bollinger Band with RSI under 30 and MACD crosses above its "
            "Signal line; sell the chosen % when price is above the upper band with RSI over 70 and MACD crosses below its "
            "Signal. The signal window lets the band and RSI condition have happened in the last few bars (1 = all on the same bar)."
        )
        p0, p1, p2, p3, p4 = st.columns(5)
        with p0:
            paper_bar = st.selectbox("Bar size", ["4-hour (up to 2 years)", "Daily (up to 10 years)"], key="paper_bar")
        daily_mode = paper_bar.startswith("Daily")
        max_years = 10 if daily_mode else 2
        if st.session_state.get("paper_years", 2) > max_years:
            st.session_state["paper_years"] = max_years
        with p1:
            paper_years = st.number_input("Number of years", value=2, min_value=1, max_value=max_years, step=1, key="paper_years")
        with p2:
            paper_money = st.number_input("Money invested per buy ($)", value=1000.0, min_value=1.0, step=100.0, key="paper_money")
        with p3:
            paper_pct = st.number_input("Money taken out (%)", value=10.0, min_value=1.0, max_value=100.0, step=1.0, key="paper_pct")
        with p4:
            paper_window = st.number_input(
                "Signal window (bars)", value=5, min_value=1, max_value=20, step=1, key="paper_window",
                help="1 = band, RSI and MACD all on the same bar (very strict). 5 = the oversold or overbought condition may have happened in the last 5 bars.",
            )
        pct_decimal = paper_pct / 100.0

        bar_label = "daily" if daily_mode else "4h"
        with st.spinner(f"Loading {paper_years} year(s) of {bar_label} data for {active_stock}..."):
            if daily_mode:
                raw_paper = download_daily(active_stock, int(paper_years))
            else:
                raw_paper = download_data_for_paper(active_stock, paper_years)
        if raw_paper is None or raw_paper.empty:
            st.warning("No historical data for this period. Yahoo hourly data is limited to about 2 years; choose Daily bars for longer tests.")
        else:
            try:
                paper_4h = calculate_indicators_daily(raw_paper) if daily_mode else calculate_indicators(raw_paper, active_stock)
            except Exception as exc:
                st.error(f"Could not compute indicators: {exc}")
                paper_4h = pd.DataFrame()
            if paper_4h.empty or len(paper_4h) < 2:
                st.warning(f"Not enough {bar_label} bars after computing indicators.")
            else:
                res = run_paper_simulation(paper_4h, paper_money, pct_decimal, int(paper_window))
                st.metric("Current price (latest close)", f"${current_price:.2f}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Realized gain", f"${res['realized_gain']:,.2f}")
                m2.metric("Unrealized gain", f"${res['unrealized_gain']:,.2f}")
                m3.metric("Rate of return", f"{res['rate_of_return_pct']:.1f}%")
                m4.metric("Total invested", f"${res['total_invested']:,.2f}")
                buy_hold = (float(paper_4h["Close"].iloc[-1]) / float(paper_4h["Close"].iloc[0]) - 1.0) * 100.0
                m5.metric("Buy and hold, same period", f"{buy_hold:+.1f}%")
                st.caption(
                    f"Tested {len(paper_4h):,} {bar_label} bars from {paper_4h.index[0].date()} to {paper_4h.index[-1].date()} "
                    f"with a signal window of {int(paper_window)} bar(s)."
                )
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

    with tab_plan:
        render_plan_tab(active_stock, current_price, fin, quality, trend, indicator_data, pillar_scores, r_default)

    with tab_analyst:
        st.subheader(f"Analyst view - {active_stock}")
        st.caption(
            "Consensus, price targets and estimate revisions from Yahoo Finance. These are third-party analyst opinions, "
            "not a recommendation from this app. The How it works tab shows exactly how the verdict is calculated."
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
            if av["n"] == 0:
                try:
                    is_fund = get_instrument_info(active_stock).get("type") in ETF_TYPES
                except Exception:
                    is_fund = False
                if is_fund:
                    st.info(
                        "This is a fund (ETF). Wall Street analysts publish ratings, price targets and earnings estimates for "
                        "individual companies, not for funds, so there is nothing to show here. Use Price charts and the Buy "
                        "checklist (Timing) for funds, or open the Analyst view of one of the fund's biggest holdings."
                    )
                else:
                    st.warning(
                        "Yahoo returned no analyst ratings, price targets or estimates for this ticker. Small or newly listed "
                        "companies can have none, but it can also be a temporary gap. Open the status table below and try Reload."
                    )
                with st.expander("Data source status", expanded=not is_fund):
                    st.dataframe(
                        pd.DataFrame([{"Source": k, "Status": v} for k, v in analyst["status"].items()]),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.button("Reload analyst data", key="reload_analyst_empty", on_click=_clear_analyst_cache)
            else:
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
        st.caption(
            "Your list loads from Favorites.csv and appears at the top of the ticker dropdown. Changes made here last only "
            "for this session; edit Favorites.csv on GitHub to change the permanent list."
        )
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
                        st.session_state["pending_ticker"] = ticker
                        st.rerun()
                with r3:
                    if st.button("Remove", key=f"rem_fav_{ticker}"):
                        st.session_state["favorites"] = [t for t in fav_list if t != ticker]
                        st.rerun()

    with tab_explain:
        st.subheader("How it works: what each tab helps you do and how to read every number")
        st.markdown(
            "Use this page as a guide to the whole app. Each section below covers one tab: "
            "**what it is for**, **how to use it**, and **what each metric means and how to read it**."
        )

        with st.expander("Start here: what each tab is for and a suggested workflow", expanded=True):
            st.markdown("""
| Tab | The question it answers | Use it when |
|---|---|---|
| **Price charts** | Is the price stretched or turning right now? | Timing an entry over the next days to weeks |
| **Buy checklist** | Is this a good business, at a good price, at a sensible moment? | First screen for any stock; the main decision tab |
| **Business snapshot** | How healthy and fast-growing is the business, and what stands out? | Understanding the company behind the ticker |
| **Fair value** | What could it be worth, and what is the most I should pay? | Setting a buy price from your own assumptions |
| **Entry and exit plan** | How do I buy in stages and take profit in stages? | Turning the buy price and fair value into a price-by-price plan |
| **Analyst view** | What do professional analysts think, and is that changing? | A cross-check against your own view |
| **News** | Why did this stock move, and what is being said about it? | After a big move or an alert, before you act |
| **Test the signals** | Would the app's technical buy and sell rules have made money on this stock? | Testing the timing rules before trusting them |
| **Market today** | What is the whole market doing today, and what are the headlines? | On a big up or down day, or as a morning check |
| **Ask Charlie** | Can I just ask in plain words? | Questions, comparisons, plans and charts without clicking through tabs |
| **Compare stocks** | Which of these 2-4 stocks is the better choice? | Choosing between similar stocks, or checking a candidate against something you own |
| **Top 10 lists** | Which stocks look best to buy, or most overvalued, right now? | Finding ideas across your favorites and the wider stock list |
| **My portfolio** | Which of my positions are worth taking profit on? | After you enter your shares and cost basis |
| **My favorites** | Shortcuts to the stocks I follow | Saving a search through the dropdown |
| **How it works** | What does each number mean? | Whenever a metric is unclear (this page) |

The first eight tabs are about the **one stock you loaded**. The next six look **across stocks** or at your own holdings.

**Suggested workflow**
1. Pick a ticker in the **Stock ticker** box at the top (or type any symbol and press Enter). Everything loads right away.
2. Open **Buy checklist**. Read the verdict and the three scores (Quality, Valuation, Timing). A failed check is a question to investigate, not an automatic no.
3. Open **Business snapshot** to see the business's strengths and watch-outs, then **Fair value** to set your own growth, multiple and margin-of-safety assumptions. Compare the current price with the **Buy price**.
4. Open **Entry and exit plan** to split your money into tranches and set the prices at which you would sell part of the position.
5. Use **Price charts** to judge timing, and **Test the signals** to see whether those timing signals have actually worked for this stock.
6. Cross-check with **Analyst view**, and read **News** after any big move.
7. Use **Market today** to see whether a move is the market or just this stock. Use **Compare stocks** to choose between candidates, **Top 10 lists** for new ideas (and alerts on your favorites), and **My portfolio** for profit-taking signals on what you own. Keep the stocks you follow in **My favorites**.

**Ideal long-term buy:** a high-quality business (Quality), priced below your buy price (Valuation), at a moment when the trend is intact and the price has pulled back (Timing).
""")

        with st.expander("Top bar and general notes (ticker, current price, data freshness)"):
            st.markdown("""
- **Stock ticker:** one box. Your Favorites appear first, then the stocks from your stock list. Click it and type to search the list.
- **Any other stock:** type its Yahoo Finance symbol in the same box (for example BRK-B for Berkshire) and press Enter.
- **No Load button:** choosing or typing a ticker loads it into every tab straight away. The **Load** button in My favorites fills this same box.
- **Current price:** the latest close of the most recent 4-hour bar, so it can lag the live price slightly.
- **Data source:** Yahoo Finance for prices and company data; Google News (with Yahoo as backup) for headlines. Prices refresh about every 15 minutes, company statements and valuation history about every hour, so numbers may not change on every click.
- **Missing data:** when Yahoo has no value, the app shows N/A and ignores that item when scoring.
- **Data as of line:** under the top bar, it shows the date of the latest price bar, how often data refreshes, and the date of the last daily scan.
- **When Yahoo is having a bad moment:** Yahoo sometimes returns nothing (more often on shared cloud servers). The app then shows a yellow notice instead of empty tabs, with a **Try again** button and, when available, the last saved scan result for that stock. A failed lookup is never remembered: the next click asks Yahoo again (with a short pause of about two minutes between automatic retries so Yahoo is not hammered).
- **Not investment advice.** The app organizes public data and simple rules; it cannot see the future.
""")

        with st.expander("Price charts tab: short-term timing"):
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

        with st.expander("Buy checklist tab: the main decision tab"):
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
- **Price at or below the EPS-model buy price (10Y)** and **at or below the DCF buy price (10Y):** use the growth, multiple, return and margin-of-safety inputs from the Fair value tab, so they are only as good as your assumptions.
- **P/E at or below the 40th percentile of its own history:** the stock is cheaper than it usually is.
- **FCF yield at least 4%:** free cash flow divided by market cap. About the cash return if you bought the whole company; 4% equals paying 25 times FCF.
- **Forward P/E below trailing P/E:** analysts expect earnings to grow.
- **PEG at most 2:** P/E relative to expected growth (from Yahoo); how much you pay for each unit of growth.

**Timing checks (is this a sensible moment?)**
- **Price within 5% of or above the 200-day average:** the long-term trend is not broken (avoids falling knives).
- **At least 10% below the 52-week high:** a real pullback rather than chasing a rally.
- **Weekly RSI below 50:** momentum has cooled.
- **Not collapsing versus SPY:** the 6-month return is not more than 15 points behind the S&P 500 ETF.
- **4-hour triggers, at least 2 of 3:** the Price charts tiles (Below BB Low, RSI under 30, MACD above Signal).

**Banks and insurers (Financial Services):** ROIC, leverage, interest coverage, FCF and DCF checks show N/A, because debt and cash are part of how they operate. Use the EPS model for them.

**ETFs and funds:** they have no company statements, so Quality and Valuation show N/A and only the Timing pillar applies.
""")

        with st.expander("Business snapshot tab: the business behind the ticker"):
            st.markdown("""
**What it helps you do:** see how healthy and how fast-growing the business is, and whether the price looks reasonable, on one page. All values come from Yahoo's annual financial statements (about 4 fiscal years, so growth rates cover about 3 years). Profit figures are reported (GAAP) numbers and can include one-off items, so they can differ from the adjusted earnings you see elsewhere.

**Top row (six headline numbers)**
- **Market cap:** share price times shares outstanding.
- **P/E (latest year):** market cap divided by last year's net income, with the 3-year average underneath. A P/E well below its 3-year average means recent profit is above its average (or the price fell).
- **Forward P/E:** price divided by the profit analysts expect next year, with the **PEG ratio** underneath (P/E divided by expected growth; 1 or less is cheap, 2 or less is fair).
- **Free-cash-flow yield:** free cash flow divided by market cap. Higher is cheaper; 4% or more is the checklist's pass mark. The underlying multiple is market cap divided by free cash flow (20x is a 5% yield).
- **Dividend yield:** the expected yearly dividend per share divided by the price.
- **Net debt / EBITDA:** how many years of operating earnings it would take to repay debt after using cash. **Net cash** means cash exceeds debt.

**Strengths and watch-outs:** plain-language notes written from the numbers using the same cut-offs as the Buy checklist (ROIC, margin direction, growth, cash conversion, debt, interest cover, stock compensation, share count, P/E against its own history, cash yield). A watch-out is a question to investigate, not a reason to reject the stock.

**Trends by fiscal year:** four small charts: revenue and net income; the four margins; net income versus free cash flow (a widening gap, with cash falling behind profit, deserves a closer look); and the share count (falling is good).

**Details by area.** Each row shows the latest number, a trend or detail, a **Signal** and what good looks like. Signal colours: green is Strong or Cheap, amber is OK or Fair, red is Watch or Pricey, grey means information only.
- **Profitability:** gross, operating, net and free-cash-flow margins (profit as a share of sales, with the change since the first year shown), **ROIC** (operating profit after tax divided by the capital invested in the business; 10% or more passes, 15% or more is strong) and **ROE** (net income divided by shareholders' equity, from Yahoo).
- **Growth:** yearly growth (compound rate) of revenue, net income, earnings per share and free cash flow, and how many years earnings per share rose. **Revenue, total change over 2 years** is the full two-year change, not a yearly rate. Net income vs 2 years ago is Up, Down or Flat.
- **Cash and balance sheet:** free cash flow (Yahoo's figure, or operating cash flow minus capital spending), free cash flow as a share of net income (how much of reported profit becomes cash), **net debt** (Yahoo's total debt including leases minus cash; negative means net cash), net debt / EBITDA, **interest coverage** (operating profit divided by interest expense) and the **current ratio** (short-term assets divided by short-term bills; 1 or more means bills can be paid from current assets).
- **Shareholders:** share count change over 2 years (more than 5% up is Issuing, more than 5% down is Buying Back), stock compensation as a share of free cash flow, and the dividend.
- **Valuation multiples:** P/E for the latest year, P/E against its own history (percentile: 40th or lower is cheap, above 70th is pricey), forward P/E, PEG, EV / EBITDA (enterprise value, which includes debt, divided by operating earnings before depreciation), cash yield, and P/E against the matching **sector fund** (a rough benchmark, not a true peer average).

**Banks, insurers and funds:** for banks and insurers, margins and returns are shown but debt, cash-flow and ROIC rows are marked not applicable. Funds (ETFs) have no company statements, so the tab explains that and points you to the charts and the Timing checks.

**Download:** the button saves every row on the tab as a CSV file. The **Year-by-year table** expander shows the margin history behind the charts.
""")

        with st.expander("Fair value tab: what it could be worth and the most to pay"):
            st.markdown("""
**What it helps you do:** turn your own beliefs about growth and future valuation into a **fair value** and a **buy price**, and see whether today's price is in the buy zone. The answers depend entirely on your inputs, so treat them as scenarios, not predictions.

**The four inputs**
- **Growth g (%):** how fast you expect earnings (and free cash flow) to grow each year. Staying above 10% for ten years is rare.
- **Terminal K:** the multiple the market pays for earnings (or cash flow) at the end of the period. 15 to 20 is typical for a mature company.
- **Return r (%):** the yearly return you demand; the rate used to bring future values back to today. Default: S&P 500 trailing 3-year annualized return plus 2%, capped at 15%, so the hurdle does not balloon after a strong market.
- **MoS (%):** margin of safety, a cushion for being wrong. A 25% margin means you only buy at 75% of fair value.

**Stock type (Auto or your choice).** Companies grow and trade differently, so the app first detects a type and uses it to set the suggested g, K and MoS. The **Stock type** box on the tab shows what was detected and why; pick another type to override it (it resets to Auto when you load a new ticker). The return r is the same for every type.
- **Core** (default): g is the median of past revenue, EPS and FCF growth kept between 3% and 10%; K is the stock's own median P/E kept between 12 and 22.
- **Growth** (revenue growing 12% a year or more): a two-stage model. Growth starts at the historical median (kept between 10% and 25%) for 5 years, then fades in a straight line to 5% by year 10. The tab turns that into one equivalent flat g so the same formulas work. K is kept between 15 and 30. This stops fast growers from always looking expensive while still not assuming growth lasts forever.
- **Cyclical** (energy, airlines, steel, mining, shipping, homebuilders and similar): g between 2% and 5%, K between 6 and 14, MoS 35%. Earnings of these companies swing with the economy, so peak earnings should not be capitalised at a high multiple.
- **Financial** (banks and insurers): g between 3% and 8%, K between 8 and 14.
- **Leveraged or cash-burning companies** of any type: K capped at 15 and MoS 35%. Otherwise MoS is 15% for high-quality businesses (Quality 80% or more) and 25% for the rest.

The line under the inputs shows the raw historical growth and says when it was capped. The caps exist because the model compounds g for 10 years and a fast recent streak rarely lasts that long. If you believe a company can keep growing faster, type a higher g yourself. Edit any input; **Reset inputs to suggested** restores the suggestion.

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

        with st.expander("Entry and exit plan tab: buy in stages, sell in stages"):
            st.markdown("""
**What it helps you do:** turn the buy price and fair value from the Fair value tab into a price-by-price plan, so you decide in advance how to buy and how to take profit instead of deciding in the heat of the moment.

**Top of the tab:** a verdict banner and six reference prices: current price, the model **Buy price**, **Fair value**, the lower Bollinger band (4-hour), the 200-day average and the 52-week low. A warning appears when Quality is below 50%, because staged buying of a weak business only averages down into a problem.

**Entry ladder (buying in stages)**
- **Total to invest** is split evenly into **tranches** (2 to 6).
- **Ladder style:** **Value** starts at the model Buy price and steps down from there; **Pullback** starts at today's price. **Step %** is the gap between tranches.
- The table shows each tranche's price, amount, shares and whether the price has been reached ("Reached: buy now" or "Waiting"). If the stock has no model buy price (ETFs, or negative earnings), only the Pullback style is available.

**Exit ladder (selling in stages)**
- The ladder starts from the model **Fair value**, or from your **average cost** when you enter your position on the My portfolio tab (the browser session only; nothing is stored).
- Defaults sell 10%, 15% and 25% of the position at 0%, 20% and 40% above fair value (30%, 60% and 100% above cost when based on cost). Edit the table to suit you; if the percentages add up to more than 100% the last steps are cut back.
- When you have shares and a cost, the table shows shares to sell, cash received and the gain locked in at each step.

**Overbought signals and trim level:** the same three overbought signals and Trim / Review / Watch / Hold level used on the My portfolio tab.

**When to stop and re-check** lists the situations in which the plan should be reviewed (for example a drop in Quality or a broken business story) rather than followed mechanically. A CSV download saves the plan.

This is a planning aid built on your own assumptions. It is not investment advice.
""")

        with st.expander("News tab: why did it move, and what is being said"):
            st.markdown("""
**What it helps you do:** find the likely reason behind a move without leaving the app, and see whether the move was about the company or about the whole market.

**The three numbers at the top:** the stock's move versus the previous trading day, the S&P 500's move, and the difference. A note under them says whether the move looks **stock-specific** (the stock moved 3% or more while the market moved less than 1%) or **mostly market-wide**.

**Headlines:** the latest headlines for the company, newest first, with publisher and age. You can show the last 3 days, last 7 days or everything found. Only titles are shown; the link opens the publisher's page.

**Where the headlines come from:** Google News (free, no account needed) is the main source because it kept working when Yahoo was rate-limiting. Yahoo Finance's own news is used as a backup when Google returns very little or fails. Duplicate headlines and generic quote pages are removed. The search uses the company name, which avoids mix-ups for short tickers such as T or F.

**Limits:** a headline can be late, wrong, or about another company with a similar name. News explains moves after the fact; it is context for your decision, not a buy or sell signal. If both sources fail, the tab says so and offers **Try again**.
""")

        with st.expander("Market today tab: the whole market in one place"):
            st.markdown("""
**What it helps you do:** see in a minute what the market is doing and why people say it is moving, before you decide anything about one stock.

**Automatic reading of the numbers:** a few plain sentences built from the data, not written by a person or an AI. They say what moved (indexes, Nasdaq versus Dow, the VIX fear gauge, the 10-year yield, oil and gold, which sectors lead and lag, how far the S&P 500 is from its 200-day average and 52-week high, and how many of your favorites are down). They do not claim to know the cause; the headlines give the reasons.

**Big move warning:** shown when the S&P 500 is down 2% or more, the Nasdaq is down 3% or more, or the VIX is above 30 (or jumped 15% or more in a day). It reminds you that one day rarely changes a good business and points you to the staged-buying plan.

**How to read the VIX and yield:** the VIX under 15 is calm, 15-20 normal, 20-30 nervous, above 30 stress. For the VIX and the yield the colours are reversed, so a rise shows red.

**Tables and chart:** an index table, a sector bar chart (sector funds), and your favorites' biggest drops and gains today. Then up to 12 market headlines from the last 3 days.

**Limits:** prices are Yahoo's daily data and can lag by minutes; "today" means the latest session in the data. If Yahoo does not answer, the tab says so and the headlines may still work.
""")

        with st.expander("Ask Charlie tab: ask in plain words"):
            st.markdown("""
**What it helps you do:** ask a question in your own words instead of clicking through tabs: "Explain MSFT", "Why is the market down?", "Compare NVDA and AMD", "Build a 4-step entry plan for 5000 dollars", "Show UAL vs DAL for 2 years", "Which of my favorites are near their buy price?". The one-click buttons above the chat ask common questions for you.

**How it works:** Charlie is named in honor of Charlie Munger, but it is an AI (Google's Gemini model), not him, and it does not speak for him or quote him. It does not guess numbers. It can only use a fixed menu of this dashboard's own functions: the Buy checklist scores, fair value and buy price, price statistics, news headlines, today's market numbers, the entry and exit ladders, your favorites scan, and chart drawing. Under each answer, **Looked up** shows which functions were used, so you can see where the numbers came from. It also knows this app's definitions (Quality, Valuation, Timing, buy zone, Trim and so on).

**What it cannot do:** it cannot trade, cannot see your accounts, cannot read your GitHub files, and does not run code. It does not see your portfolio unless you tick **Let Charlie see my portfolio**.

**Privacy:** your questions and the data it looks up are sent to Google. On the free tier, Google may keep and review prompts, so share your holdings only if you are comfortable with that. Your API key and access code live in Streamlit's secrets, not in the repository.

**Limits and safety:** it asks for an access code (the app's web address is public, and the free AI quota is limited). Each session allows 30 messages, and the app as a whole is capped at 400 AI calls a day. If the free quota runs out you will see a clear message. An AI can misread numbers or phrase things too confidently, so treat answers as a starting point and check the tab it points to. Not investment advice.
""")

        with st.expander("Test the signals tab: how the timing rules would have done"):
            st.markdown("""
**What it helps you do:** see how the app's own technical buy and sell rules would have performed on this stock over the last 1-2 years, before you rely on them.

**Inputs**
- **Bar size:** **4-hour** bars (built from hourly data, which Yahoo limits to about 2 years) or **Daily** bars (up to 10 years). Daily mode uses Bollinger (20 days, 2 standard deviations), RSI (10) and MACD (8/21/5).
- **Number of years:** 1 or 2 for 4-hour bars, 1 to 10 for daily bars.
- **Signal window (bars):** how many recent bars the oversold or overbought condition may have occurred in. 1 is the strict original rule (everything on the same bar); the default 5 allows the condition within the last 5 bars.
- **Money invested per buy:** the amount added on every buy signal (default 1,000 dollars).
- **Money taken out (%):** the share of your current position sold on every sell signal (default 10%).

**Rules**
- **Buy** when MACD has just crossed above its Signal line, **and** within the signal window the close was below the lower Bollinger Band with RSI under 30. In plain words: it was oversold recently and momentum has just turned up.
- **Sell** when MACD has just crossed below its Signal line, **and** within the window the close was above the upper Bollinger Band with RSI over 70 (only if you hold shares).
- With a window of 1, all three conditions must happen on the same bar. That is so rare that many stocks show no trades at all, because by the time MACD turns up, the price has usually already moved back above the lower band.

**Results**
- **Realized gain:** profit locked in by sales (sale proceeds minus the cost of the shares sold).
- **Unrealized gain:** profit or loss on shares you still hold, at the latest price.
- **Rate of return:** (final wealth - total invested) / total invested. Final wealth is cash from sales plus the value of shares still held.
- **Total invested:** the sum of all buys. **Buy / sell events** lists every trade.
- **Buy and hold, same period:** the return from simply owning the stock from the first to the last bar tested. Compare it with the Rate of return: if the signals do not beat it, they added little.

**Read it with care:** even with the window, the rules fire only a few times, so a handful of trades proves little. The test ignores fees and taxes, and the rate of return is measured on the money you put in over time, so it is not directly comparable with buy and hold. Past results do not predict future ones. The 4-hour indicators need about 200 bars before the first signal.
""")

        with st.expander("Analyst view tab: what professional analysts think"):
            st.markdown("""
**What it helps you do:** see the Wall Street consensus (ratings, price targets, estimate changes) and whether it is improving or deteriorating, as a cross-check on your own view. It is third-party opinion from **Yahoo Finance**, not a recommendation from this app, and it is **not** part of the Buy checklist score.

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
- **Funds (ETFs):** analysts do not cover funds, so the tab shows a short explanation instead of empty numbers.
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

        with st.expander("Compare stocks tab: put 2-4 stocks side by side"):
            st.markdown("""
**What it helps you do:** choose between similar stocks, or check a candidate against something you already own, by lining up the same numbers for each.

**How to use it:** pick stocks from the list (your current stock and your Favorites) and/or type other tickers separated by commas. You need at least 2 and at most 4. The first load takes several seconds per stock; repeat loads are fast.

**What you see**
- **Takeaways:** which stock has the highest buy score, the highest quality, the most upside to fair value, and (if two or more signals are on) which is the most overbought.
- **Table:** Overview, Scores, Valuation, Quality and Timing rows, using the same definitions as the other tabs. **Green** marks the best value in a row, but only for rows where higher or lower is clearly better (for example higher ROIC, lower P/E, lower debt). Rows such as price, assumed growth, price versus the 200-day average and weekly RSI are shown without highlighting. Negative or zero P/E values are ignored when picking the best.
- **Pillar scores chart:** Quality, Valuation and Timing percentages for each stock.
- **Price chart:** the last 12 months with every stock starting at 100, so you can compare performance fairly.

**Read it with care:** each stock's valuation uses its own suggested growth, multiple and margin of safety, so Valuation scores show how each price compares with that stock's own assumptions, not with a common yardstick. Compare ETFs only on Timing. Green means better on that row, not buy.
""")

        with st.expander("Top 10 lists tab: top 10 buy and top 10 avoid"):
            st.markdown("""
**What it helps you do:** find the best buy candidates and the most overvalued names across a large list of stocks, and see how your Favorites compare, without opening stocks one at a time.

**Where the data comes from:** a daily scan (a GitHub Action) scores every symbol in Favorites.csv and stockList.csv with the same Buy checklist and each stock's suggested valuation defaults, then saves rankings.csv. This tab shows that file. **Scan my Favorites now** scores only your Favorites on demand (slow, and it replaces the file view until you clear it).

**Who can be ranked:** operating companies with enough scored checks (at least 4 Quality checks and 3 Valuation checks). ETFs are scored on Timing only and are never ranked. Banks and insurers have several checks that show N/A, so they usually fall below the minimum: they still get a verdict but are not ranked.

**Top 10 buy candidates:** stocks with a Quality score of at least 50%, ordered by a buy score = 45% Quality + 40% Valuation + 15% Timing. Quality and price matter most; timing is a smaller tie-breaker, matching the verdict logic.

**Top 10 avoid / overvalued:** stocks whose Valuation score is 33% or lower, or whose Quality score is below 50%, ordered by an avoid score = 50% (100 - Valuation) + 30% (100 - Quality) + 20% overbought (how many of the 3 sell signals are on). This is a list of stocks not to start buying now, or to review if you own them. It is not an automatic sell list.

**Columns:** Q / V / T is the Quality, Valuation and Timing score in percent. **Upside to fair** is the average gap between fair value and the price across the available models (negative means the price is above fair value). **Overbought** is how many of the 3 profit-taking signals are on. A star marks your Favorites and **NEW** marks stocks that entered the list in the latest scan.

**Buy-price watch:** your Favorites with Quality of 50% or more, showing how far the price is from the model Buy price (10-year EPS model). **In zone** means the price is at or below the Buy price; **Near** means within 5% above it. The gap is (price / Buy price - 1) in percent, so a negative gap means below the Buy price. The list also shows the stock type used for the valuation.

**Profit-taking watch:** your Favorites with a Trim, Review or Watch signal (rules under the My portfolio tab).

**Alerts:** the same daily scan opens a GitHub issue, which GitHub emails to you, when one of your Favorites newly enters the top 10 buy list or the top 10 avoid list, newly reaches the Trim level, or newly drops to **your buy price** (the Buy-price watch "In zone" status). Only changes against the previous scan count, so you are not alerted every day for the same stock. The very first scan after this feature is added counts everything as new and sends one baseline alert.

**Limits:** results depend on the default growth, multiple and margin of safety for each stock. The stock type (Core, Growth, Cyclical, Financial) changes the defaults, and cyclicals are deliberately held to a stricter standard. Treat the lists as a place to start research.
""")

        with st.expander("My portfolio tab: your positions and profit-taking signals"):
            st.markdown("""
**What it helps you do:** see your gain or loss on each position and which holdings are worth trimming, reviewing or just watching.

**How to use it:**
1. Type your positions in the table: **Symbol**, **Shares** and **Avg cost per share**. Add or delete rows as needed.
2. Positions stay only in this browser session and are never saved to GitHub. Click **Download positions as CSV** to keep a copy on your computer, and upload it next session.

**What it shows:** portfolio value and total gain, then one row per position: value, gain percent, weight in the portfolio, a **Signal** and a **Suggested action**. Duplicate symbols are combined. A sector table warns when one sector is more than 35% of the portfolio.

**Profit-taking signals** (the same rules as your old StockSeller reports, plus valuation and quality):
- **Overbought signals (0 to 3):** close above the upper Bollinger Band (15 days, 2 standard deviations), daily RSI above 70, and MACD below its signal line.
- **Trim:** all 3 overbought signals **and** the price is above fair value in every available valuation model. Consider trimming about 10-25%.
- **Review:** the Quality score is below 50%. Re-check why you own it.
- **Watch:** 2 or more overbought signals, or 1 signal with the price above fair value.
- **Hold:** none of the above.

A position over 15% of the portfolio adds a rebalancing note. If you are at a loss, a Trim signal asks you to review rather than sell.

**Read it with care:** signals are prompts, not instructions. Selling a core long-term holding has tax and opportunity costs, and fast growers can stay overbought for a long time.
""")

        with st.expander("My favorites tab: your shortlist"):
            st.markdown("""
**What it helps you do:** keep the stocks you follow one click away instead of searching the dropdown every time.
- Your list comes from **Favorites.csv** in the project and appears at the top of the ticker dropdown. The Top 10 lists tab and the daily scan use the same file.
- **Add current stock to Favorites** adds the stock you are viewing for this session only.
- **Load** opens that stock in every tab. **Remove** deletes it from the list.
- **Important:** additions and removals made in the app last only for the current browser session. To change the permanent list, edit Favorites.csv on GitHub.
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
