#!/usr/bin/env python3
"""
NSE Swing Trade Screener (Yahoo Finance edition)
==================================================
1. Compares today's average % change across each tracked broad market index
   (a proxy for "which index is hottest today", computed from the index's
   own constituents since Yahoo doesn't give a clean index-level feed for
   every NSE index).
2. Auto-selects the best-performing index.
3. Screens that index's stocks for swing-trade candidates.
4. Notifies a Telegram bot with the results.

Why Yahoo Finance instead of the NSE website's own API?
--------------------------------------------------------
NSE's website (nseindia.com) blocks requests from datacenter/cloud IP
ranges -- including GitHub-hosted Actions runners -- via Akamai
bot-detection, regardless of headers/cookies used. Yahoo Finance's public
data endpoints (via the `yfinance` package) are free and generally
reachable from cloud runners, so this uses that instead.

Currently tracked indices: NIFTY 50, NIFTY NEXT 50 (both with officially
sourced, dated constituent lists -- see INDEX_UNIVERSES below). More
indices (Midcap 50, Smallcap 50, etc.) can be added once their constituent
lists are verified; get the mapping wrong and the screen silently runs on
the wrong stocks, so new indices are added deliberately, not guessed.

Filters applied
-----------------
  - Open > LTP         -> today's Open is above today's Close (pulled back / red day)
  - Gap % > threshold   -> (Open - PrevClose) / PrevClose * 100
  - RSI(14) > threshold -> daily RSI from closing prices
  - Vol chg % > threshold -> today's volume vs YESTERDAY's volume (previous trading day)

This is a research/screening tool, NOT investment advice. Always do your
own due diligence before trading.

Usage
-----
    python3 nse_swing_screener.py
    python3 nse_swing_screener.py --rsi 60 --volchg 100 --gap 0.01
    python3 nse_swing_screener.py --index "NIFTY 50"   # skip auto-detect

Environment variables (for Telegram notification)
--------------------------------------------------
    BOT_TOKEN   - Telegram bot token
    CHAT_ID     - Telegram chat id to send results to
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID")

# --- Index universes -------------------------------------------------------
# .NS suffix is Yahoo Finance's convention for NSE-listed stocks.
# Sourced from official NSE-published constituent PDFs (dated below).
# NSE rebalances these lists around 31-Jan and 31-Jul each year -- refresh
# periodically if it drifts.

# NIFTY 50 as of 17-Aug-2026
NIFTY_50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TATAMOTORS", "TRENT", "ULTRACEMCO", "WIPRO",
]

# NIFTY NEXT 50 as of 30-Mar-2026
NIFTY_NEXT_50_SYMBOLS = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "BAJAJHLDNG", "BANKBARODA", "BOSCHLTD", "BPCL", "BRITANNIA",
    "CANBK", "CGPOWER", "CHOLAFIN", "CUMMINSIND", "DIVISLAB", "DLF",
    "DMART", "SIEMENSENERGY", "GAIL", "GODREJCP", "HAL", "HDFCAMC",
    "HINDZINC", "HYUNDAI", "INDHOTEL", "IOC", "IRFC", "JINDALSTEL",
    "LODHA", "LTIM", "MAZDOCK", "MOTHERSON", "MUTHOOTFIN", "PFC",
    "PIDILITIND", "PNB", "RECLTD", "SHREECEM", "SIEMENS", "SOLARINDS",
    "TATACAPITAL", "TATAPOWER", "TATAMOTORS", "TORNTPHARM", "TVSMOTOR",
    "UNIONBANK", "MCDOWELL-N", "VBL", "VEDL", "ZYDUSLIFE",
]

INDEX_UNIVERSES = {
    "NIFTY 50": NIFTY_50_SYMBOLS,
    "NIFTY NEXT 50": NIFTY_NEXT_50_SYMBOLS,
}


def send_telegram_message(text: str):
    """Sends a message via the Telegram Bot API. Splits long messages
    since Telegram caps a single message at 4096 characters."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[warn] BOT_TOKEN / CHAT_ID not set, skipping Telegram send.")
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


def fetch_all_history(symbols, period="4mo"):
    """One batched call for all tickers (much faster/kinder than per-symbol calls)."""
    tickers = [f"{s}.NS" for s in symbols]
    print(f"Downloading history for {len(tickers)} symbols...")
    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    return data


def find_top_index(data: pd.DataFrame) -> str:
    """Picks the index whose constituents have the highest average
    day-over-day % change today (a proxy for 'which index is hottest')."""
    scores = {}
    for index_name, symbols in INDEX_UNIVERSES.items():
        pct_changes = []
        for symbol in symbols:
            ticker = f"{symbol}.NS"
            try:
                df = data[ticker].dropna(how="all")
            except KeyError:
                continue
            if len(df) < 2:
                continue
            today_close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2]
            if pd.isna(today_close) or pd.isna(prev_close) or prev_close == 0:
                continue
            pct_changes.append((today_close - prev_close) / prev_close * 100)

        if pct_changes:
            scores[index_name] = float(np.mean(pct_changes))
            print(f"  {index_name}: avg daily change {scores[index_name]:+.2f}% "
                  f"(from {len(pct_changes)} stocks)")

    if not scores:
        raise RuntimeError("Could not compute index performance from downloaded data.")

    top_index = max(scores, key=scores.get)
    print(f"\nTop index right now: {top_index} ({scores[top_index]:+.2f}%)")
    return top_index


def screen(symbols, data, rsi_thresh, volchg_thresh, gap_thresh):
    results = []

    for symbol in symbols:
        ticker = f"{symbol}.NS"
        try:
            df = data[ticker].dropna(how="all")
        except KeyError:
            print(f"[warn] No data returned for {ticker}, skipping.")
            continue

        if df.empty or len(df) < 16:
            continue

        today = df.iloc[-1]
        yesterday = df.iloc[-2] if len(df) >= 2 else None
        if yesterday is None:
            continue

        open_p = today["Open"]
        close_p = today["Close"]
        prev_close = yesterday["Close"]
        today_vol = today["Volume"]
        prev_vol = yesterday["Volume"]

        if any(pd.isna(v) for v in [open_p, close_p, prev_close, today_vol, prev_vol]):
            continue
        if prev_close == 0 or prev_vol == 0:
            continue

        gap_pct = (open_p - prev_close) / prev_close * 100
        vol_chg = (today_vol - prev_vol) / prev_vol * 100
        chg_pct = (close_p - prev_close) / prev_close * 100

        # Cheap filters first
        if not (open_p > close_p and gap_pct > gap_thresh):
            continue

        rsi = compute_rsi(df["Close"])
        if pd.isna(rsi):
            continue

        if rsi > rsi_thresh and vol_chg > volchg_thresh:
            results.append({
                "Symbol": symbol,
                "Open": round(float(open_p), 2),
                "LTP": round(float(close_p), 2),
                "PrevClose": round(float(prev_close), 2),
                "Chg%": round(chg_pct, 2),
                "Gap%": round(gap_pct, 2),
                "RSI(14)": round(rsi, 2),
                "VolChg%": round(vol_chg, 2),
            })

    return pd.DataFrame(results).sort_values("VolChg%", ascending=False).reset_index(drop=True) \
        if results else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="NSE swing trade screener (Yahoo Finance)")
    parser.add_argument("--index", type=str, default=None,
                         help="Force a specific index instead of auto-detecting the top one "
                              f"(choices: {list(INDEX_UNIVERSES.keys())})")
    parser.add_argument("--rsi", type=float, default=60, help="RSI(14) threshold (default 60)")
    parser.add_argument("--volchg", type=float, default=100, help="Volume change %% threshold (default 100)")
    parser.add_argument("--gap", type=float, default=0.01, help="Gap %% threshold (default 0.01)")
    args = parser.parse_args()

    # Download history for every symbol across every tracked index in one batch.
    all_symbols = sorted(set(s for syms in INDEX_UNIVERSES.values() for s in syms))
    data = fetch_all_history(all_symbols)

    if args.index:
        index_name = args.index.upper()
        if index_name not in INDEX_UNIVERSES:
            print(f"Unknown index '{index_name}'. Choices: {list(INDEX_UNIVERSES.keys())}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Comparing today's performance across tracked indices...")
        index_name = find_top_index(data)

    print(f"\nScreening constituents of: {index_name}")
    print(f"Filters -> Open>LTP, Gap%>{args.gap}, RSI>{args.rsi}, VolChg%>{args.volchg}\n")

    df = screen(INDEX_UNIVERSES[index_name], data, args.rsi, args.volchg, args.gap)

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
