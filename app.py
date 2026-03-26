import time
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
    data = raw_data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col == stock for col in data.columns):
        if len(data.columns) == 6:
            data = data.iloc[:, :-1]
        if len(data.columns) == 5:
            data.columns = expected_cols
        else:
            raise ValueError(f"Unexpected columns for {stock}: {list(data.columns)}")

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
    data = raw_data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col == stock for col in data.columns):
        if len(data.columns) == 6:
            data = data.iloc[:, :-1]
        if len(data.columns) == 5:
            data.columns = expected_cols

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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_analyst_view(stock: str) -> dict:
    """
    Get analyst consensus and price targets from Yahoo Finance.
    Returns sentiment: Potential Buy / Neutral / Potential Sell.
    """
    out: dict = {
        "sentiment": None,
        "recommendation_key": None,
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "num_analysts": None,
        "current_vs_target": None,
    }
    try:
        ticker = yf.Ticker(stock)
        info = getattr(ticker, "info", None) or {}
        key = (info.get("recommendationKey") or "").strip().lower()
        mean_t = info.get("targetMeanPrice")
        high_t = info.get("targetHighPrice")
        low_t = info.get("targetLowPrice")
        n_analysts = info.get("numberOfAnalystOpinions")
        rec_mean = info.get("recommendationMean")

        if key in ("strong_buy", "buy"):
            out["sentiment"] = "Potential Buy"
        elif key in ("sell", "strong_sell"):
            out["sentiment"] = "Potential Sell"
        elif key == "hold" or key:
            out["sentiment"] = "Neutral"
        elif rec_mean is not None:
            try:
                r = float(rec_mean)
                if r <= 2.0:
                    out["sentiment"] = "Potential Buy"
                elif r >= 4.0:
                    out["sentiment"] = "Potential Sell"
                else:
                    out["sentiment"] = "Neutral"
            except (TypeError, ValueError):
                out["sentiment"] = "Neutral"
        else:
            out["sentiment"] = "—"

        if key:
            out["recommendation_key"] = key.replace("_", " ").title()
        if mean_t is not None:
            try:
                out["target_mean"] = float(mean_t)
            except (TypeError, ValueError):
                pass
        if high_t is not None:
            try:
                out["target_high"] = float(high_t)
            except (TypeError, ValueError):
                pass
        if low_t is not None:
            try:
                out["target_low"] = float(low_t)
            except (TypeError, ValueError):
                pass
        if n_analysts is not None:
            try:
                out["num_analysts"] = int(n_analysts)
            except (TypeError, ValueError):
                pass

        # Optional: get_analyst_price_targets() as fallback for targets
        if out["target_mean"] is None:
            try:
                targets = ticker.get_analyst_price_targets()
                if isinstance(targets, dict):
                    out["target_mean"] = targets.get("mean") or targets.get("targetMeanPrice")
                    out["target_high"] = out["target_high"] or targets.get("high")
                    out["target_low"] = out["target_low"] or targets.get("low")
                elif isinstance(targets, pd.DataFrame) and not targets.empty:
                    if "Mean Target" in targets.columns:
                        out["target_mean"] = float(targets["Mean Target"].iloc[0])
                    if "High Target" in targets.columns:
                        out["target_high"] = float(targets["High Target"].iloc[0])
                    if "Low Target" in targets.columns:
                        out["target_low"] = float(targets["Low Target"].iloc[0])
            except Exception:
                pass
    except Exception:
        pass
    return out


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

        if cfo is not None and capex is not None:
            fcf = cfo - capex
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

    tab_charts, tab_fin, tab_val, tab_paper, tab_analyst, tab_fav, tab_explain = st.tabs(
        [
            "Charts",
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
        fig = plot_technicals(active_stock, indicator_data)
        st.pyplot(fig)
        with st.expander("Last 30 periods (4h)", expanded=False):
            st.dataframe(
                indicator_data[["Close", "BB_High", "BB_Low", "RSI", "MACD", "MACD_Signal"]].tail(30),
                height=280,
            )

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

            v1, v2, v3, v4 = st.columns(4)
            with v1:
                g_pct = st.number_input("Growth g (%)", value=10.0, step=0.5, min_value=-50.0, max_value=100.0, key="val_g_pct")
            with v2:
                k_multiple = st.number_input("Terminal K", value=15.0, step=1.0, min_value=0.0, key="val_k_multiple")
            with v3:
                r_pct = st.number_input("Return r (%)", value=12.0, step=0.5, min_value=0.0, max_value=50.0, key="val_r_pct")
            with v4:
                mos_pct = st.number_input("MoS (%)", value=25.0, step=1.0, min_value=0.0, max_value=90.0, key="val_mos_pct")

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

            # DCF model: enterprise value from FCF, then equity value = EV - Net Debt, then per share
            fcf_1y = fin.get("FCF (1Y)")
            shares_latest = fin.get("Shares Latest")
            net_debt = fin.get("Net Debt")

            if fcf_1y is None or shares_latest in (None, 0):
                st.info("DCF model not available (missing FCF or share count).")
            else:
                # yfinance often reports cash flow in thousands; normalize to dollars for consistent per-share result
                fcf_scale = 1.0
                if abs(fcf_1y) > 0 and abs(fcf_1y) < 1e7 and shares_latest > 1e6:
                    # FCF looks like thousands (e.g. 3e6 for $3B); scale to dollars
                    fcf_scale = 1000.0
                fcf_dollars = fcf_1y * fcf_scale

                dcf_horizon = 10
                dcf_enterprise = 0.0
                for t in range(1, dcf_horizon + 1):
                    cf_t = fcf_dollars * (1 + g) ** t
                    dcf_enterprise += cf_t / (1 + r) ** t
                cf_n = fcf_dollars * (1 + g) ** dcf_horizon
                tv = cf_n * k_multiple
                dcf_enterprise += tv / (1 + r) ** dcf_horizon

                # Convert to equity value (what shareholders get) by subtracting net debt
                net_debt_dollars = (net_debt or 0) if net_debt is not None else 0
                dcf_equity = max(0.0, dcf_enterprise - net_debt_dollars)
                intrinsic_dcf = dcf_equity / shares_latest
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
        st.subheader(f"Analyst view – {active_stock}")
        st.caption(
            "Consensus and price targets from Yahoo Finance (third‑party analyst data, not a recommendation from this app)."
        )
        st.metric("Current price (latest close)", f"${current_price:.2f}")

        sentiment = analyst.get("sentiment") or "—"
        color = "green" if sentiment == "Potential Buy" else "red" if sentiment == "Potential Sell" else "gray"
        st.markdown(f"**Analyst sentiment:** :{color}[**{sentiment}**]")

        a1, a2, a3 = st.columns(3)
        with a1:
            if analyst.get("recommendation_key"):
                st.metric("Recommendation (raw)", analyst["recommendation_key"])
        with a2:
            if analyst.get("num_analysts") is not None:
                st.metric("Number of analysts", str(analyst["num_analysts"]))
        with a3:
            if analyst.get("target_mean") is not None:
                mean_t = analyst["target_mean"]
                vs = "Above" if current_price > mean_t else "Below" if current_price < mean_t else "At"
                st.metric("Mean price target", f"${mean_t:.2f}", f"{vs} target")

        if analyst.get("target_high") is not None or analyst.get("target_low") is not None:
            row_targets = [
                {"Target": "Low", "Price": f"${analyst['target_low']:.2f}" if analyst.get("target_low") is not None else "—"},
                {"Target": "Mean", "Price": f"${analyst['target_mean']:.2f}" if analyst.get("target_mean") is not None else "—"},
                {"Target": "High", "Price": f"${analyst['target_high']:.2f}" if analyst.get("target_high") is not None else "—"},
            ]
            st.dataframe(pd.DataFrame(row_targets), use_container_width=True, hide_index=True)
        if (sentiment == "—" and analyst.get("target_mean") is None and not analyst.get("recommendation_key")):
            st.info("Analyst consensus and price targets are not available for this ticker from the current source.")

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
        st.subheader("Explanation of metrics and valuation calculators")
        st.markdown("This tab explains every metric and calculator used in the app.")

        with st.expander("Charts – Technical indicators", expanded=True):
            st.markdown("""
**Data:** 1-hour Yahoo Finance data resampled to **4-hour candles** (last 6 months).

- **Bollinger Bands (BB)**  
  Upper and lower bands around price (200-period SMA ± 2 standard deviations).  
  - **Below BB Low:** Price below the lower band is often considered oversold (potential support).  
  - The chart shows Close price with BB High (red dashed) and BB Low (green dashed).

- **RSI (Relative Strength Index)**  
  10-period RSI (0–100).  
  - **RSI &lt; 30:** Oversold; **RSI &gt; 70:** Overbought.  
  - The chart includes reference lines at 30 (green) and 70 (red).

- **MACD (Moving Average Convergence Divergence)**  
  Fast 8, slow 21, signal 5 (on 4-hour data).  
  - **MACD &gt; Signal:** Bullish momentum.  
  - The chart plots MACD line (blue) and Signal line (red).
""")

        with st.expander("Paper investing check – Simulation rules"):
            st.markdown("""
Uses **4-hour bars** over the selected number of years (Yahoo 1h data, up to 2 years).

- **Buy signal (all must be true):** Close below BB lower band (outside 2 std dev), RSI &lt; 30, and MACD crosses **above** MACD Signal.  
  On each buy, the simulator invests the **Money invested** amount at that bar’s close.

- **Sell signal (all must be true):** Close above BB upper band, RSI &gt; 70, and MACD crosses **below** MACD Signal.  
  On each sell, the simulator sells **Money taken out %** of the current share position at that bar’s close.

- **Realized gain:** Sum of (sale proceeds − cost basis of shares sold) on all sell events.  
- **Unrealized gain:** Current value of remaining shares minus their cost basis.  
- **Rate of return:** (Final wealth − Total invested) ÷ Total invested × 100%, where final wealth = cash from sales + current position value.
""")

        with st.expander("Reversal confidence score (0-100)"):
            st.markdown("""
This score adds four checks (25 points each):

- **OBV Divergence (+25):** OBV trends up while price is flat/down (bullish divergence).
- **Volume Profile POC (+25):** Current price is above the 30-day Point of Control.
- **Volume Bars (+25):** Latest daily candle is green and volume is greater than 1.2x the 20-day average.
- **VWAP (+25):** Current price is above the 30-day VWAP.

**Interpretation**
- **> 75:** Strong Reversal Signal
- **< 40:** Likely Dead Cat Bounce
- Otherwise: Neutral / Mixed
""")

        with st.expander("Analyst view – Third-party consensus"):
            st.markdown("""
Data in the **Analyst view** tab comes from **Yahoo Finance**, which aggregates analyst ratings and price targets.

- **Analyst sentiment** is mapped from the consensus recommendation: **Potential Buy** (strong buy / buy), **Neutral** (hold), **Potential Sell** (sell / strong sell). This is not a recommendation from this app.
- **Mean / high / low price targets** are analyst estimates; "Above/Below target" compares the current price to the mean target.
- Availability varies by ticker; some symbols have little or no analyst data from this source.
""")

        with st.expander("Financial metrics – Definitions"):
            st.markdown("""
- **P/E (3Y Avg):** Market cap ÷ average Net Income over the last three reported fiscal years.  
- **P/E (1Y):** Market cap ÷ most recent year’s Net Income.  
- **Revenue Growth (2Y %):** % change in Total Revenue from two years ago to the most recent year.  
- **Profit Growth (2Y vs Today):** Whether Net Income is **Up**, **Down**, or **Flat** vs. two years ago.  
- **Shares Outstanding (2Y):** Whether the company is **Issuing Shares**, **Buying Back**, or **Neutral** (based on % change in shares issued over two years; ±5% threshold).  
- **Div Rate (FWD):** Forward annual dividend per share (when available).  
- **FCF Multiple (MC/FCF):** Market cap ÷ Free Cash Flow (Cash from Operations − Capital Expenditures, latest year).  
- **Market Cap:** Current equity market capitalization (share price × shares outstanding).  
- **Sector PE:** Trailing P/E of a sector ETF (e.g. XLK for Technology, XLV for Healthcare) matching the company’s sector.  
- **EPS (1Y):** Net Income ÷ shares issued for the most recent fiscal year (used as M₀ in the valuation model).  
- **Net Debt:** Long-term debt + short-term debt − cash and equivalents (used to convert DCF enterprise value to equity value).  
- **FCF (1Y):** Free cash flow for the latest year (Cash from Operations − Capital Expenditures); used in the DCF model.
""")

        with st.expander("Valuation – EPS-based model (5- and 10-year)"):
            st.markdown("""
**Starting metric (M₀):** Last fiscal year Net Income per share (EPS).

**Inputs:**  
- **g:** Estimated annual growth rate of earnings (e.g. 5%, 10%, 15%).  
- **K:** Terminal multiple (P/E or price-to-FCF) you expect in 5 or 10 years.  
- **r:** Desired annual return (discount rate), e.g. 10%–15%.  
- **Margin of safety (MoS):** Cushion (e.g. 25%) applied to fair value to get a buy price.

**Formulas:**  
- **Price in year X:** $\\text{Price}_X = M_0 \\times (1+g)^X \\times K$  
- **Fair value today:** $\\text{Fair Value} = \\text{Price}_X \\div (1+r)^X$  
- **Buy price (with MoS):** $\\text{Buy Price} = \\text{Fair Value} \\times (1 - \\text{MoS})$

The table shows 5-year and 10-year horizons: Price Year X, Fair Value Today, and Buy Price (with MOS).
""")

        with st.expander("Valuation – DCF model (10-year)"):
            st.markdown("""
**Basis:** Free Cash Flow (FCF) for the latest year, grown at rate **g**, discounted at rate **r**, plus a terminal value.

**Formula:**  
$\\text{DCF} = \\sum_{t=1}^{10} \\frac{\\text{CF}_t}{(1+r)^t} + \\frac{\\text{TV}}{(1+r)^{10}}$  
where $\\text{CF}_t = \\text{FCF}_{1Y} \\times (1+g)^t$ and $\\text{TV} = \\text{CF}_{10} \\times K$.

**From enterprise value to equity value:**  
- DCF sum gives **enterprise value** (value of the whole firm).  
- **Equity value** = Enterprise value − **Net debt** (so debt is accounted for).  
- **DCF Intrinsic Price** = Equity value ÷ shares outstanding.  
- **DCF Buy Price (with MOS)** = DCF Intrinsic Price × (1 − Margin of safety).

Uses the same **g**, **r**, **K**, and **Margin of safety** as the EPS-based model. FCF and net debt come from the financial statements.
""")

        st.caption("Data source: Yahoo Finance. Not investment advice.")


if __name__ == "__main__":
    main()
