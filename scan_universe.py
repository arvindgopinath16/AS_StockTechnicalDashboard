"""Daily scan for the Stock Technical Dashboard.

Scores every symbol in Favorites.csv + stockList.csv with the same long-term checklist the app uses,
ranks the top 10 buy and top 10 avoid names, and writes:

  rankings.csv     one row per symbol (read by the Rankings tab)
  scan_state.json  the current lists plus which names are NEW since the previous scan
  alert.md         only when one of your FAVORITES newly entered the top 10 buy list, the top 10
  alert_title.txt  avoid list, or newly reached the Trim level (the GitHub workflow turns it into an issue)

Run locally:   python scan_universe.py --limit 10
Run for real:  python scan_universe.py            (the GitHub Action does this every weekday)
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
logging.getLogger("streamlit").setLevel(logging.ERROR)

import pandas as pd  # noqa: E402

import app  # noqa: E402  (shared scoring code; importing it does not start the Streamlit UI)

logging.getLogger("streamlit").setLevel(logging.ERROR)


def _line(r: pd.Series) -> str:
    def f(x, fmt="{:.0f}"):
        return "-" if pd.isna(x) else fmt.format(x)

    up = "-" if pd.isna(r["upside_pct"]) else f"{r['upside_pct']:+.0f}%"
    return (
        f"- **{r['symbol']}** ({(r['name'] or '')[:30]}) - {r['verdict_short']}; "
        f"Quality {f(r['quality_pct'])}%, Valuation {f(r['valuation_pct'])}%, Timing {f(r['timing_pct'])}%; "
        f"price {f(r['price'], '{:.2f}')}, upside to fair value {up}"
    )


def score_one(job: tuple) -> dict:
    """Runs in a worker process."""
    sym, r_default = job
    time.sleep(0.3)  # gentle pacing for Yahoo
    return app.score_symbol(sym, r_default)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only scan the first N symbols (testing)")
    ap.add_argument("--symbols", type=str, default="", help="comma separated list, overrides the universe (testing)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-ok", type=float, default=0.5, help="abort without writing if fewer than this share scored")
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--owner", type=str, default=os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    favs = app.load_favorites_file()
    fav_set = set(favs)
    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or app.load_universe()
    if args.limit:
        universe = universe[: args.limit]
    if not universe:
        print("No symbols to scan (Favorites.csv / stockList.csv not found).")
        return 1
    print(f"Scanning {len(universe)} symbols ({len(fav_set & set(universe))} favorites) ...", flush=True)

    sp3 = app.sp500_trailing_return(3)
    r_default = round(min(sp3 + 2.0, 15.0), 1) if sp3 is not None else 12.0
    bench = app.download_daily("SPY")
    print(f"Required return r = {r_default}% (S&P 500 3Y = {sp3})", flush=True)

    t0 = time.time()
    rows = []
    # Separate PROCESSES, not threads: yfinance's download() keeps shared global state, so parallel threads
    # mix up each other's prices. Each process has its own state and Yahoo session.
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, row in enumerate(ex.map(score_one, [(s, r_default) for s in universe], chunksize=1), start=1):
            rows.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(universe)} done ({time.time() - t0:.0f}s)", flush=True)

    # One slower retry pass for anything Yahoo refused
    failed = [i for i, r in enumerate(rows) if r["error"]]
    if failed:
        print(f"Retrying {len(failed)} failed symbols ...", flush=True)
        for i in failed:
            time.sleep(3)
            retry = app.score_symbol(rows[i]["symbol"], r_default, bench)
            if not retry["error"]:
                rows[i] = retry

    df = app.clean_results(pd.DataFrame(rows))
    n_ok = int((df["error"] == "").sum())
    print(f"Scored {n_ok}/{len(df)} symbols in {time.time() - t0:.0f}s", flush=True)
    if n_ok / len(df) < args.min_ok:
        print("Too many failures (Yahoo rate limit?). Keeping the previous results and failing the run.")
        return 1

    buy, avoid = app.rank_results(df)
    top_buy = buy["symbol"].tolist() if not buy.empty else []
    top_avoid = avoid["symbol"].tolist() if not avoid.empty else []
    trim = df.loc[df["trim_level"] == "Trim", "symbol"].tolist()

    state_path = out_dir / app.STATE_FILE
    prev = None
    if state_path.exists():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    p_buy, p_avoid, p_trim = (set(prev.get(k, [])) for k in ("top_buy", "top_avoid", "trim")) if prev else (set(), set(), set())
    new_buy = [s for s in top_buy if s not in p_buy]
    new_avoid = [s for s in top_avoid if s not in p_avoid]
    new_trim = [s for s in trim if s not in p_trim]

    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    state = {
        "asof": asof, "universe": len(df), "scored": n_ok, "r_default": r_default,
        "top_buy": top_buy, "top_avoid": top_avoid, "trim": trim,
        "new_buy": new_buy, "new_avoid": new_avoid, "new_trim": new_trim,
    }

    # ---- write results ----
    df = df.sort_values("buy_score", ascending=False, na_position="last")
    round_cols = [c for c in app._NUM_COLS if c in df.columns]
    df[round_cols] = df[round_cols].round(2)
    df.to_csv(out_dir / app.RANKINGS_FILE, index=False)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ---- alert: only favorites that NEWLY entered a list ----
    by_sym = df.set_index("symbol")

    def _row(sym: str) -> pd.Series:
        r = by_sym.loc[sym].copy()
        r["symbol"] = sym
        return r

    a_buy = [s for s in new_buy if s in fav_set]
    a_avoid = [s for s in new_avoid if s in fav_set]
    a_trim = [s for s in new_trim if s in fav_set]
    alert_md, alert_title = out_dir / "alert.md", out_dir / "alert_title.txt"
    for p in (alert_md, alert_title):
        if p.exists():
            p.unlink()
    if a_buy or a_avoid or a_trim:
        parts = []
        if args.owner:
            parts.append(f"@{args.owner}")
        parts.append(f"Daily scan {asof}: your favorites changed position in the rankings.\n")
        if a_buy:
            parts.append("### Newly in the Top 10 BUY list")
            parts += [_line(_row(s)) for s in a_buy]
            parts.append("")
        if a_avoid:
            parts.append("### Newly in the Top 10 AVOID list")
            parts += [_line(_row(s)) for s in a_avoid]
            parts.append("")
        if a_trim:
            parts.append("### Newly at the TRIM level (profit-taking)")
            for s in a_trim:
                r = by_sym.loc[s]
                parts.append(f"- **{s}**: {r['trim_reason']}; price {r['price']:.2f}")
            parts.append("")
        parts.append("Open the Rankings and Portfolio tabs for detail. Screening aid only, not investment advice.")
        alert_md.write_text("\n".join(parts), encoding="utf-8")
        bits = []
        if a_buy:
            bits.append("BUY: " + ", ".join(a_buy))
        if a_avoid:
            bits.append("AVOID: " + ", ".join(a_avoid))
        if a_trim:
            bits.append("TRIM: " + ", ".join(a_trim))
        alert_title.write_text("Stock alert - " + " | ".join(bits), encoding="utf-8")
        print("ALERT:", " | ".join(bits))
    else:
        print("No favorites newly entered a list; no alert.")

    print("Top 10 buy:  ", ", ".join(top_buy))
    print("Top 10 avoid:", ", ".join(top_avoid))
    print("Trim level:  ", ", ".join(trim) or "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
