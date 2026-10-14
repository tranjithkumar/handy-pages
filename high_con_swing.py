#!/usr/bin/env python3
"""
High-Conviction Swing Scanner (Yahoo Finance edition)
=======================================================
Daily-timeframe version of the intraday stacked-confirmation scanner.
Moving to daily bars removes the "data delay" concern entirely -- a daily
close is a daily close regardless of when you check it, so this scan is
more trustworthy than the intraday version, not less.

IMPORTANT
---------
- No screener guarantees outcomes. This stacks four independently
  well-known, battle-tested techniques to raise signal QUALITY; it does
  not eliminate risk. Treat results as a shortlist to research further.
- This is a research/screening tool, NOT investment advice. Always do
  your own due diligence before trading.

The four stacked conditions (all must be true)
-------------------------------------------------
  1. EMA trend alignment -> EMA9 > EMA21 > EMA50 (full bullish stack,
                             stricter than a simple 2-EMA crossover).
  2. 20-day breakout       -> today's close is above the highest close of
                             the prior 20 trading days -- the core entry
                             signal from the Turtle Trading system
                             (Richard Dennis, 1980s), one of the most
                             famous and widely replicated trend-following
                             setups in trading history.
  3. RSI momentum band     -> RSI(14) between 50 and 70 -- bullish
                             momentum without being dangerously overbought.
  4. Relative Volume       -> today's volume is at least RVOL_THRESHOLD
                             times the prior 20-day average volume
                             (confirms the breakout has real participation,
                             not a low-volume drift).

Results are ranked by Relative Volume and only the top N are reported.

Usage
-----
    python3 swing_conviction_scanner.py
    python3 swing_conviction_scanner.py --rvol 1.5 --top 10

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
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID")

NSE_NIFTY200_CSV = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 50
BREAKOUT_LOOKBACK = 20   # Turtle-style 20-day high
RSI_PERIOD = 14
RVOL_LOOKBACK = 20


# --- Universe ---------------------------------------------------------------

def fetch_nifty200_symbols(timeout: int = 10):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
    }
    r = requests.get(NSE_NIFTY200_CSV, headers=headers, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
    if len(symbols) < 30:
        raise RuntimeError("NIFTY 200 list from NSE archive looks too short -- aborting.")
    return symbols


# --- Telegram ----------------------------------------------------------------

def send_telegram_message(text: str):
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
                data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"[warn] Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"[warn] Telegram send failed: {e}", file=sys.stderr)
        time.sleep(0.5)


# --- Indicators ----------------------------------------------------------------

def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    if len(closes) < period + 1:
        return np.nan
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return np.nan
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def fetch_all_daily(symbols, period="6mo"):
    tickers = [f"{s}.NS" for s in symbols]
    print(f"Downloading daily history for {len(tickers)} symbols (last {period})...")
    data = yf.download(
        tickers, period=period, interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )
    return data


def analyze_symbol(symbol, df, rvol_thresh, rsi_low=50, rsi_high=70):
    """Returns a result dict if the stock passes all 4 stacked conditions,
    else None."""
    if df.empty:
        return None
    df = df.dropna(how="all")
    min_needed = max(EMA_SLOW, BREAKOUT_LOOKBACK, RVOL_LOOKBACK, RSI_PERIOD) + 5
    if len(df) < min_needed:
        return None

    close = df["Close"]
    volume = df["Volume"]

    last_close = close.iloc[-1]
    if pd.isna(last_close):
        return None

    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    ema_mid = close.ewm(span=EMA_MID, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]

    # Highest CLOSE of the prior N days, excluding today -- the Turtle-style
    # breakout reference level.
    prior_high = close.iloc[-(BREAKOUT_LOOKBACK + 1):-1].max()

    rsi = compute_rsi(close)

    today_vol = volume.iloc[-1]
    avg_prior_vol = volume.iloc[-(RVOL_LOOKBACK + 1):-1].mean()
    if pd.isna(today_vol) or pd.isna(avg_prior_vol) or avg_prior_vol == 0:
        return None
    rvol = today_vol / avg_prior_vol

    if any(pd.isna(v) for v in [ema_fast, ema_mid, ema_slow, prior_high, rsi]):
        return None

    passed = {
        "ema_stack": ema_fast > ema_mid > ema_slow,
        "breakout": last_close > prior_high,
        "rsi_band": rsi_low <= rsi <= rsi_high,
        "rvol": rvol >= rvol_thresh,
    }

    if all(passed.values()):
        return {
            "Symbol": symbol,
            "LTP": round(float(last_close), 2),
            "20D_High": round(float(prior_high), 2),
            "RSI(14)": round(float(rsi), 2),
            "RVOL": round(float(rvol), 2),
        }
    return None


def scan(symbols, data, rvol_thresh, rsi_low, rsi_high):
    results = []
    checked, no_data = 0, 0
    for symbol in symbols:
        checked += 1
        ticker = f"{symbol}.NS"
        try:
            df = data[ticker]
        except KeyError:
            no_data += 1
            continue
        res = analyze_symbol(symbol, df, rvol_thresh, rsi_low, rsi_high)
        if res:
            results.append(res)

    print(f"[diagnostic] {checked} checked, {no_data} had no data, "
          f"{len(results)} passed all 4 stacked conditions.")
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("RVOL", ascending=False).reset_index(drop=True)


def format_message(df: pd.DataFrame, top_n: int, rvol_thresh: float) -> str:
    header = (
        f"*High-Conviction Swing Scanner*\n"
        f"Universe: NIFTY 200 | Run: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST\n"
        f"Rules: EMA9>21>50 + 20D breakout + RSI 50-70 + RVOL>{rvol_thresh}x\n\n"
    )
    if df.empty:
        return header + "No stocks matched all 4 conditions right now."
    shortlist = df.head(top_n)
    lines = [f"{len(shortlist)} of {len(df)} match(es) (top {top_n} by RVOL):\n"]
    for _, row in shortlist.iterrows():
        lines.append(
            f"*{row['Symbol']}*  LTP {row['LTP']}  20D-High {row['20D_High']}  "
            f"RSI {row['RSI(14)']}  RVOL {row['RVOL']}x"
        )
    return header + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="High-conviction daily swing scanner (NIFTY 200)")
    parser.add_argument("--rvol", type=float, default=1.3,
                         help="Relative volume threshold, e.g. 1.3 = 30%% above 20-day average (default 1.3)")
    parser.add_argument("--rsi-low", type=float, default=50, help="RSI lower bound (default 50)")
    parser.add_argument("--rsi-high", type=float, default=70, help="RSI upper bound (default 70)")
    parser.add_argument("--top", type=int, default=10, help="Max candidates to report (default 10)")
    args = parser.parse_args()

    print("Fetching live NIFTY 200 constituent list from NSE...")
    try:
        symbols = fetch_nifty200_symbols()
        print(f"  Got {len(symbols)} symbols.")
    except Exception as e:
        print(f"[error] Could not fetch NIFTY 200 list: {e}", file=sys.stderr)
        sys.exit(1)

    data = fetch_all_daily(symbols)

    print(f"\nScanning for high-conviction swing setup "
          f"(RVOL>{args.rvol}x, RSI {args.rsi_low}-{args.rsi_high})...")
    df = scan(symbols, data, args.rvol, args.rsi_low, args.rsi_high)

    print("\n" + "=" * 60)
    if df.empty:
        print("No stocks matched all 4 conditions right now.")
    else:
        print(df.head(args.top).to_string(index=False))
    print("=" * 60)
    print("Reminder: screening tool only, not investment advice.")

    if df.empty:
        print("\nNo matches -- staying silent, no Telegram message sent.")
    else:
        message = format_message(df, args.top, args.rvol)
        send_telegram_message(message)


if __name__ == "__main__":
    main()
