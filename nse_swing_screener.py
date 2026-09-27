#!/usr/bin/env python3
"""
NSE Swing Trade Screener
=========================
1. Finds the top-performing "Broad Market Index" from NSE's live heatmap
   (https://www.nseindia.com/market-data/live-market-indices/heatmap)
2. Pulls all constituent stocks of that index
3. Applies a swing-trading filter, matching the TradingView screener setup:
      - Open > LTP        -> stock currently trading BELOW today's open (pulled back intraday)
      - Gap % > 0.01%     -> (Open - PrevClose) / PrevClose * 100  > 0.01
      - RSI(14) > 60      -> daily RSI, computed from last ~60 trading days of closes
      - Vol chg % > 100%  -> today's volume vs YESTERDAY's volume (previous trading day)

Notes / caveats
----------------
- Uses NSE's public (unofficial) JSON endpoints. NSE does rate-limit and
  occasionally blocks bot-like traffic; the script uses a browser-like
  session + small delays + retries to stay polite.
- This is a research/screening tool, NOT investment advice. Always do your
  own due diligence before trading.

Usage
-----
    python3 nse_swing_screener.py
    python3 nse_swing_screener.py --index "NIFTY NEXT 50"   # skip auto-detect
    python3 nse_swing_screener.py --rsi 60 --volchg 100 --gap 0.01
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID")

BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-market-indices/heatmap",
}

# The set of indices NSE tags under the "Broad Market Indices" heatmap panel.
BROAD_MARKET_INDICES = {
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100",
    "NIFTY MIDCAP 150", "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100",
    "NIFTY SMALLCAP 250", "NIFTY LARGEMIDCAP 250", "NIFTY MIDSMALLCAP 400",
    "NIFTY 100", "NIFTY 200", "NIFTY500 MULTICAP 50:25:25", "NIFTY 500",
    "NIFTY FPI 150", "NIFTY500 LMS EQUAL WEIGHT", "NIFTY MIDSMALL 400",
    "NIFTY MICROCAP 250", "NIFTY TOTAL MARKET",
}


class NSESession:
    """Handles the cookie handshake NSE requires before its API responds."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._warm_up()

    def _warm_up(self):
        # NSE requires a visit to a normal page first to set cookies,
        # otherwise the API endpoints return 401/403.
        try:
            self.s.get(BASE, timeout=10)
            time.sleep(1)
            self.s.get(f"{BASE}/market-data/live-market-indices/heatmap", timeout=10)
            time.sleep(1)
        except requests.RequestException as e:
            print(f"[warn] warm-up request failed: {e}", file=sys.stderr)

    def get_json(self, path, params=None, retries=3):
        url = f"{BASE}{path}"
        for attempt in range(retries):
            try:
                r = self.s.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    return r.json()
                # session likely expired / blocked -> refresh and retry
                self._warm_up()
            except requests.RequestException as e:
                print(f"[warn] request failed ({e}), retrying...", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def find_top_broad_market_index(sess: NSESession) -> str:
    """Returns the index name with the highest %change among broad market indices."""
    data = sess.get_json("/api/allIndices")
    rows = data.get("data", [])

    candidates = []
    for row in rows:
        name = row.get("index", row.get("indexSymbol", "")).strip().upper()
        if name in BROAD_MARKET_INDICES:
            pchg = row.get("percentChange", row.get("perChange", None))
            if pchg is not None:
                candidates.append((name, float(pchg)))

    if not candidates:
        raise RuntimeError("Could not find broad market indices in /api/allIndices response")

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_name, top_pchg = candidates[0]
    print(f"Top broad market index right now: {top_name} ({top_pchg:+.2f}%)")
    return top_name


def get_index_constituents(sess: NSESession, index_name: str) -> pd.DataFrame:
    """Live quote snapshot for every stock in the given index."""
    data = sess.get_json("/api/equity-stockIndices", params={"index": index_name})
    rows = data.get("data", [])
    df = pd.DataFrame(rows)
    # Drop the index-level summary row NSE includes (symbol == index name)
    df = df[df["symbol"] != index_name].reset_index(drop=True)
    return df


def get_historical_prices(sess: NSESession, symbol: str, days: int = 90) -> pd.DataFrame:
    """Daily OHLCV history for RSI + average-volume calc."""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    params = {
        "symbol": symbol,
        "series": '["EQ"]',
        "from": from_date.strftime("%d-%m-%Y"),
        "to": to_date.strftime("%d-%m-%Y"),
    }
    data = sess.get_json("/api/historical/cm/equity", params=params)
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["CH_TIMESTAMP"] = pd.to_datetime(df["CH_TIMESTAMP"])
    df = df.sort_values("CH_TIMESTAMP").reset_index(drop=True)
    df = df.rename(columns={
        "CH_CLOSING_PRICE": "close",
        "CH_TOT_TRADED_QTY": "volume",
    })
    return df[["CH_TIMESTAMP", "close", "volume"]]


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return np.nan
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_vol_chg_pct(volumes: pd.Series) -> float:
    """Today's volume vs the PREVIOUS trading day's volume."""
    if len(volumes) < 2:
        return np.nan
    today_vol = volumes.iloc[-1]
    prev_vol = volumes.iloc[-2]
    if prev_vol == 0 or np.isnan(prev_vol):
        return np.nan
    return float((today_vol - prev_vol) / prev_vol * 100)


def screen(index_name, sess, rsi_thresh, volchg_thresh, gap_thresh, delay=0.6):
    live = get_index_constituents(sess, index_name)
    if live.empty:
        print("No constituents found for this index.")
        return pd.DataFrame()

    results = []
    total = len(live)
    for i, row in live.iterrows():
        symbol = row["symbol"]
        open_p = row.get("open")
        ltp = row.get("lastPrice")
        prev_close = row.get("previousClose")
        chg_pct = row.get("pChange")

        if None in (open_p, ltp, prev_close) or prev_close == 0:
            continue

        gap_pct = (open_p - prev_close) / prev_close * 100

        # Cheap filters first, before hitting the historical API
        if not (open_p > ltp and gap_pct > gap_thresh):
            continue

        print(f"[{i+1}/{total}] {symbol}: passed gap/price filter, checking RSI & volume...")
        hist = get_historical_prices(sess, symbol)
        time.sleep(delay)
        if hist.empty or len(hist) < 21:
            continue

        rsi = compute_rsi(hist["close"])
        vol_chg = compute_vol_chg_pct(hist["volume"])

        if pd.isna(rsi) or pd.isna(vol_chg):
            continue
        if rsi > rsi_thresh and vol_chg > volchg_thresh:
            results.append({
                "Symbol": symbol,
                "Open": open_p,
                "LTP": ltp,
                "PrevClose": prev_close,
                "Chg%": round(chg_pct, 2) if chg_pct is not None else None,
                "Gap%": round(gap_pct, 2),
                "RSI(14)": round(rsi, 2),
                "VolChg%": round(vol_chg, 2),
            })

    return pd.DataFrame(results).sort_values("VolChg%", ascending=False).reset_index(drop=True)


def send_telegram_message(text: str):
    """Sends a message via the Telegram Bot API. Splits long messages
    since Telegram caps a single message at 4096 characters."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[warn] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, skipping Telegram send.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [text]

    for chunk in chunks:
        try:
            r = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if r.status_code != 200:
                print(f"[warn] Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"[warn] Telegram send failed: {e}", file=sys.stderr)
        time.sleep(0.5)


def format_results_message(index_name: str, df: pd.DataFrame, rsi_thresh, volchg_thresh, gap_thresh) -> str:
    header = (
        f"*NSE Swing Screener*\n"
        f"Index: `{index_name}`\n"
        f"Filters: Open>LTP, Gap%>{gap_thresh}, RSI>{rsi_thresh}, VolChg%>{volchg_thresh}\n"
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST\n\n"
    )
    if df.empty:
        return header + "No stocks matched all filters this run."

    lines = [f"{len(df)} match(es):\n"]
    for _, row in df.iterrows():
        lines.append(
            f"*{row['Symbol']}*  LTP {row['LTP']}  "
            f"Gap {row['Gap%']}%  RSI {row['RSI(14)']}  Vol {row['VolChg%']}%"
        )
    return header + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NSE swing trade screener")
    parser.add_argument("--index", type=str, default=None,
                         help="Force a specific index instead of auto-detecting the top one")
    parser.add_argument("--rsi", type=float, default=60, help="RSI(14) threshold (default 60)")
    parser.add_argument("--volchg", type=float, default=100, help="Volume change %% threshold (default 100)")
    parser.add_argument("--gap", type=float, default=0.01, help="Gap %% threshold (default 0.01)")
    args = parser.parse_args()

    sess = NSESession()

    index_name = args.index.upper() if args.index else find_top_broad_market_index(sess)

    print(f"\nScreening constituents of: {index_name}")
    print(f"Filters -> Open>LTP, Gap%>{args.gap}, RSI>{args.rsi}, VolChg%>{args.volchg}\n")

    df = screen(index_name, sess, args.rsi, args.volchg, args.gap)

    print("\n" + "=" * 60)
    if df.empty:
        print("No stocks matched all filters right now.")
    else:
        print(f"{len(df)} stock(s) matched — swing trade shortlist:\n")
        print(df.to_string(index=False))
    print("=" * 60)
    print("Note: This is a screening tool for research purposes only, not investment advice.")

    message = format_results_message(index_name, df, args.rsi, args.volchg, args.gap)
    send_telegram_message(message)


if __name__ == "__main__":
    main()
