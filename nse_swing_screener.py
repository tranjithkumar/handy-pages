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

Currently tracked indices: NIFTY 200, NIFTY MIDCAP 150. Constituents are
fetched live every run from NSE's CSV archive (see NSE_ARCHIVE_CSV_URLS
below) -- no stale hardcoded list, since these are large (150-200 stock)
indices that would be impractical to maintain by hand. If NSE blocks this
archive endpoint, the script fails loudly rather than silently falling
back to a wrong/incomplete list.

Two swing setups are screened in parallel, each with its own heading in
the output and Telegram message:

  Pullback-buy       -> gapped up, then faded to close below the open
                         (cheaper entry within an uptrend), RSI still
                         reasonably bullish, elevated volume.
  Momentum breakout   -> closed green with strong RSI and a clear volume
                         pickup -- actively breaking out with conviction.

Shared metric definitions:
  - Gap %      -> (Open - PrevClose) / PrevClose * 100
  - RSI(14)    -> daily RSI from closing prices
  - Vol chg %  -> today's volume vs YESTERDAY's volume (previous trading day)

This is a research/screening tool, NOT investment advice. Always do your
own due diligence before trading.

Usage
-----
    python3 nse_swing_screener.py
    python3 nse_swing_screener.py --rsi-pullback 50 --rsi-breakout 60 --volchg 50 --gap 0.01
    python3 nse_swing_screener.py --index "NIFTY 200"   # skip auto-detect

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
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

import numpy as np
import pandas as pd
import yfinance as yf
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID")

# --- Index universes -------------------------------------------------------
# .NS suffix is Yahoo Finance's convention for NSE-listed stocks.
# These are large lists (200 / 150 stocks), so unlike earlier versions of
# this script there's no hardcoded fallback copy -- that's too big to
# maintain reliably by hand. Constituents are fetched LIVE every run from
# NSE's CSV archive, which has proven reachable from this GitHub Actions
# runner. If NSE ever blocks this endpoint too, the script fails loudly
# (see get_index_universe) rather than silently screening a stale/wrong list.

NSE_ARCHIVE_CSV_URLS = {
    "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY MIDCAP 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}

INDEX_UNIVERSES = list(NSE_ARCHIVE_CSV_URLS.keys())


def try_fetch_live_constituents(index_name: str, timeout: int = 10):
    """Fetches the current constituent list straight from NSE's CSV archive.
    Returns a list of symbols, or None if it fails for any reason."""
    url = NSE_ARCHIVE_CSV_URLS.get(index_name)
    if not url:
        return None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,*/*",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
        if not symbol_col:
            return None
        symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
        if len(symbols) < 30:  # sanity check -- these indices have 150-200 names
            return None
        return symbols
    except Exception:
        return None


def get_index_universe(index_name: str):
    """Returns the live symbol list for an index. Raises if NSE's archive
    is unreachable -- there is no stale fallback for these large indices."""
    live = try_fetch_live_constituents(index_name)
    if live:
        print(f"  [{index_name}] using LIVE constituent list ({len(live)} symbols) from NSE archive.")
        return live
    raise RuntimeError(
        f"Could not fetch live constituents for '{index_name}' from NSE's archive "
        f"({NSE_ARCHIVE_CSV_URLS[index_name]}). NSE may be blocking this runner, "
        "or the file format/URL has changed."
    )


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


def _format_section(title: str, filters_desc: str, df: pd.DataFrame) -> str:
    lines = [f"*{title}*", f"_{filters_desc}_"]
    if df.empty:
        lines.append("No matches.")
    else:
        lines.append(f"{len(df)} match(es):")
        for _, row in df.iterrows():
            idx_tag = f" [{row['Index']}]" if row.get("Index") else ""
            lines.append(
                f"*{row['Symbol']}*{idx_tag}  LTP {row['LTP']}  "
                f"Gap {row['Gap%']}%  RSI {row['RSI(14)']}  Vol {row['VolChg%']}%"
            )
    return "\n".join(lines)


def format_results_message(index_name: str, pullback_df: pd.DataFrame, breakout_df: pd.DataFrame,
                            rsi_pullback, rsi_breakout, volchg_thresh, gap_thresh) -> str:
    header = (
        f"*NSE Swing Screener*\n"
        f"Index: `{index_name}`\n"
        f"Run: {datetime.now(IST).strftime('%Y-%m-%d %H:%M')} IST\n\n"
    )
    pullback_section = _format_section(
        "Pullback-buy",
        f"Open>LTP (faded), Gap%>{gap_thresh}, RSI>{rsi_pullback}, VolChg%>{volchg_thresh}",
        pullback_df,
    )
    breakout_section = _format_section(
        "Momentum breakout",
        f"Close>Open (green), RSI>{rsi_breakout}, VolChg%>{volchg_thresh}",
        breakout_df,
    )
    return header + pullback_section + "\n\n" + breakout_section


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
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
        return 50.0  # perfectly flat window -- neutral
    if avg_loss == 0:
        return 100.0  # no down days at all -- maximally overbought/strong
    if avg_gain == 0:
        return 0.0  # no up days at all -- maximally oversold/weak

    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


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


def find_top_index(data: pd.DataFrame, universes: dict) -> str:
    """Picks the index whose constituents have the highest average
    day-over-day % change today (a proxy for 'which index is hottest')."""
    scores = {}
    for index_name, symbols in universes.items():
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


def compute_all_metrics(symbols, data, symbol_index_map=None):
    """One pass over every symbol computing the shared metrics both setups
    need (gap%, vol chg%, RSI, etc). Returns a list of dicts; each setup's
    filter function below just applies its own conditions to this list.

    symbol_index_map: optional dict of symbol -> "Index Name(s)" label,
    used when screening a combined universe so results can still show
    which index each stock came from."""
    metrics = []
    no_data = 0
    symbol_index_map = symbol_index_map or {}

    for symbol in symbols:
        ticker = f"{symbol}.NS"
        try:
            df = data[ticker].dropna(how="all")
        except KeyError:
            no_data += 1
            continue

        if df.empty or len(df) < 16:
            no_data += 1
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

        rsi = compute_rsi(df["Close"])
        if pd.isna(rsi):
            continue

        metrics.append({
            "Symbol": symbol,
            "Index": symbol_index_map.get(symbol, ""),
            "Open": float(open_p),
            "Close": float(close_p),
            "PrevClose": float(prev_close),
            "Gap%": (open_p - prev_close) / prev_close * 100,
            "VolChg%": (today_vol - prev_vol) / prev_vol * 100,
            "Chg%": (close_p - prev_close) / prev_close * 100,
            "RSI(14)": rsi,
        })

    print(f"[diagnostic] {len(symbols)} stocks in universe, {no_data} had no/insufficient data, "
          f"{len(metrics)} had usable metrics computed.")
    return metrics


def _to_result_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["Open", "Close", "PrevClose", "Gap%", "VolChg%", "Chg%", "RSI(14)"]:
        df[col] = df[col].round(2)
    df = df.rename(columns={"Close": "LTP"})
    return df.sort_values("VolChg%", ascending=False).reset_index(drop=True)


def filter_pullback_buy(metrics, rsi_thresh, volchg_thresh, gap_thresh):
    """Gapped up, then faded to close below the open (cheaper entry within
    an uptrend), but RSI still reasonably bullish."""
    passed_gap = 0
    passed_rsi = 0
    rows = []
    for m in metrics:
        if not (m["Open"] > m["Close"] and m["Gap%"] > gap_thresh):
            continue
        passed_gap += 1
        if m["RSI(14)"] <= rsi_thresh:
            continue
        passed_rsi += 1
        if m["VolChg%"] > volchg_thresh:
            rows.append(m)
    print(f"[diagnostic:pullback] {passed_gap} passed Open>LTP & Gap% filter, "
          f"{passed_rsi} of those also passed RSI filter, {len(rows)} passed all filters.")
    return _to_result_df(rows)


def filter_momentum_breakout(metrics, rsi_thresh, volchg_thresh):
    """Closed green with strong RSI and a clear pickup in volume --
    a stock actively breaking out with conviction, not fading."""
    passed_green = 0
    passed_rsi = 0
    rows = []
    for m in metrics:
        if not (m["Close"] > m["Open"]):
            continue
        passed_green += 1
        if m["RSI(14)"] <= rsi_thresh:
            continue
        passed_rsi += 1
        if m["VolChg%"] > volchg_thresh:
            rows.append(m)
    print(f"[diagnostic:breakout] {passed_green} passed Close>Open filter, "
          f"{passed_rsi} of those also passed RSI filter, {len(rows)} passed all filters.")
    return _to_result_df(rows)


def main():
    parser = argparse.ArgumentParser(description="NSE swing trade screener (Yahoo Finance)")
    parser.add_argument("--index", type=str, default=None,
                         help="Restrict screening to a single index instead of the combined "
                              f"universe of both (choices: {INDEX_UNIVERSES})")
    parser.add_argument("--rsi-pullback", type=float, default=50,
                         help="RSI(14) threshold for the pullback-buy setup (default 50)")
    parser.add_argument("--rsi-breakout", type=float, default=60,
                         help="RSI(14) threshold for the momentum breakout setup (default 60)")
    parser.add_argument("--volchg", type=float, default=35,
                         help="Volume change %% threshold, shared by both setups (default 35)")
    parser.add_argument("--gap", type=float, default=0.01,
                         help="Gap %% threshold, used by the pullback-buy setup only (default 0.01)")
    parser.add_argument("--window-start-hour", type=int, default=6,
                         help="If the script fires before this IST hour, skip Telegram (likely a "
                              "misfired/early run) (default 6)")
    parser.add_argument("--window-end-hour", type=int, default=11,
                         help="If the script fires after this IST hour, skip Telegram -- this "
                              "catches GitHub Actions scheduling delays so results don't land at "
                              "an unexpected hour like the middle of the night (default 11)")
    args = parser.parse_args()

    now_ist = datetime.now(IST)
    within_window = args.window_start_hour <= now_ist.hour < args.window_end_hour
    if not within_window:
        print(f"[warn] Script is running at {now_ist.strftime('%Y-%m-%d %H:%M')} IST, outside the "
              f"intended {args.window_start_hour}:00-{args.window_end_hour}:00 IST morning window "
              f"(likely a delayed GitHub Actions trigger). Will still screen and log results, but "
              f"will NOT send a Telegram message this run.")

    print("Fetching live index constituent lists from NSE...")
    live_universes = {name: get_index_universe(name) for name in INDEX_UNIVERSES}

    # Download history for every symbol across every tracked index in one batch.
    all_symbols = sorted(set(s for syms in live_universes.values() for s in syms))
    data = fetch_all_history(all_symbols)

    # Informational only (not restricting the scan) unless --index is passed:
    # shows which index is performing best today for context in the logs.
    print("\nComparing today's performance across tracked indices (for context only)...")
    try:
        find_top_index(data, live_universes)
    except RuntimeError as e:
        print(f"[warn] {e}")

    if args.index:
        index_name = args.index.upper()
        if index_name not in INDEX_UNIVERSES:
            print(f"Unknown index '{index_name}'. Choices: {INDEX_UNIVERSES}", file=sys.stderr)
            sys.exit(1)
        screen_symbols = live_universes[index_name]
        symbol_index_map = {s: index_name for s in screen_symbols}
        scope_label = index_name
    else:
        # Combined universe: every stock in either index, tagged with which
        # index (or both) it belongs to so results stay traceable.
        symbol_index_map = {}
        for name, syms in live_universes.items():
            for s in syms:
                symbol_index_map[s] = symbol_index_map.get(s, "") + ("+" if symbol_index_map.get(s) else "") + name
        screen_symbols = sorted(symbol_index_map.keys())
        scope_label = " + ".join(INDEX_UNIVERSES)

    print(f"\nScreening combined universe: {scope_label} ({len(screen_symbols)} stocks)")
    metrics = compute_all_metrics(screen_symbols, data, symbol_index_map)

    print(f"\n--- Pullback-buy setup (Open>LTP, Gap%>{args.gap}, RSI>{args.rsi_pullback}, VolChg%>{args.volchg}) ---")
    pullback_df = filter_pullback_buy(metrics, args.rsi_pullback, args.volchg, args.gap)

    print(f"\n--- Momentum breakout setup (Close>Open, RSI>{args.rsi_breakout}, VolChg%>{args.volchg}) ---")
    breakout_df = filter_momentum_breakout(metrics, args.rsi_breakout, args.volchg)

    print("\n" + "=" * 60)
    print("PULLBACK-BUY")
    print(pullback_df.to_string(index=False) if not pullback_df.empty else "No matches.")
    print("\nMOMENTUM BREAKOUT")
    print(breakout_df.to_string(index=False) if not breakout_df.empty else "No matches.")
    print("=" * 60)
    print("Note: This is a screening tool for research purposes only, not investment advice.")

    if pullback_df.empty and breakout_df.empty:
        print("\nNo matches in either setup -- staying silent, no Telegram message sent.")
    elif not within_window:
        print("\nMatches found, but suppressing Telegram send -- outside the intended morning window.")
    else:
        message = format_results_message(
            scope_label, pullback_df, breakout_df,
            args.rsi_pullback, args.rsi_breakout, args.volchg, args.gap,
        )
        send_telegram_message(message)


if __name__ == "__main__":
    main()
