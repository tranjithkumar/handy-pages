"""
FNO Gap-RSI Swing Screener
===========================
Rule being scanned (from "The System"):

    IF  RSI < 10   AND  gap UP    ->  BUY signal   (deeply oversold, next-day gap up)
    IF  RSI > 90   AND  gap DOWN  ->  SELL signal  (deeply overbought, next-day gap down)

Stop-loss on every signal = the gap-fill level, EXACTLY (i.e. the prior day's close,
the price that would "fill" the gap).

This is a low-frequency, low-accuracy, high-reward setup by design (per the source
material: "Low accuracy. Monster risk-reward. Ruthless sizing."). It will flag very
few signals -- that's expected, not a bug. RSI<10 or >90 on NSE names is rare.

Usage
-----
    python fno_gap_rsi_screener.py                     # scan built-in FNO list
    python fno_gap_rsi_screener.py --symbols mylist.txt # scan your own list (one symbol per line, no .NS needed)
    python fno_gap_rsi_screener.py --rsi-period 14 --lookback 200

Requirements
------------
    pip install yfinance pandas --break-system-packages

Notes
-----
- Data source is Yahoo Finance via yfinance (NSE symbols suffixed with .NS).
  This script must be run on a machine with internet access to Yahoo Finance
  (it will not run inside a sandboxed/offline environment).
- The built-in FNO_STOCKS list is a snapshot and WILL drift out of date as NSE
  revises its F&O ban/inclusion list quarterly. Update FNO_STOCKS below, or
  pass your own file with --symbols.
- "Gap" here is defined as today's OPEN vs. the PREVIOUS close (standard gap
  definition). Since this scans daily bars, run it after market open (or on
  historical data) to catch the trigger day itself, or run it end-of-day to
  see if yesterday's setup triggered a gap today.
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Run: pip install yfinance pandas --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Built-in NSE F&O universe (update periodically -- NSE revises this list).
# Trim/extend freely, or supply --symbols path/to/file.txt instead.
# ---------------------------------------------------------------------------
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "AXISBANK", "BAJFINANCE", "MARUTI",
    "ASIANPAINT", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL",
    "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "GRASIM",
    "HINDALCO", "DRREDDY", "CIPLA", "DIVISLAB", "EICHERMOT", "BRITANNIA",
    "APOLLOHOSP", "HEROMOTOCO", "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE",
    "INDUSINDBK", "TECHM", "UPL", "BPCL", "SHREECEM", "TATACONSUM", "GODREJCP",
    "PIDILITIND", "DABUR", "MARICO", "COLPAL", "BERGEPAINT", "HAVELLS",
    "SIEMENS", "ABB", "DLF", "GODREJPROP", "OBEROIRLTY", "LODHA",
    "ZOMATO", "NAUKRI", "PAYTM", "POLICYBZR", "NYKAA", "DMART", "TRENT",
    "PAGEIND", "MUTHOOTFIN", "CHOLAFIN", "BAJAJHLDNG", "LICHSGFIN", "PFC",
    "RECLTD", "IRFC", "IRCTC", "IEX", "MCX", "CDSL", "BSE", "ANGELONE",
    "MOTILALOFS", "CANBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK",
    "AUBANK", "BANDHANBNK", "RBLBANK", "YESBANK", "TATAPOWER", "TATACOMM",
    "VOLTAS", "BLUESTARCO", "DIXON", "AMBER", "POLYCAB", "KEI", "CUMMINSIND",
    "BHEL", "BEL", "HAL", "MAZDOCK", "COCHINSHIP", "GRSE", "SAIL", "NMDC",
    "VEDL", "JINDALSTEL", "APLAPOLLO", "ASTRAL", "SUPREMEIND", "AARTIIND",
    "SRF", "PIIND", "DEEPAKNTR", "UPLTD", "GNFC", "CHAMBLFERT", "COROMANDEL",
    "GLENMARK", "LUPIN", "ALKEM", "TORNTPHARM", "ZYDUSLIFE", "AUROPHARMA",
    "BIOCON", "ABBOTINDIA", "IPCALAB", "LAURUSLABS", "GLAND", "SYNGENE",
    "MPHASIS", "COFORGE", "PERSISTENT", "LTIM", "LTTS", "OFSS", "TATAELXSI",
    "KPITTECH", "CYIENT", "SONACOMS", "BALKRISIND", "MRF", "APOLLOTYRE",
    "CEATLTD", "EXIDEIND", "AMARAJABAT", "BOSCHLTD", "MOTHERSON", "BHARATFORG",
    "ESCORTS", "ASHOKLEY", "TVSMOTOR", "SAIL", "JSL", "RATNAMANI", "WELCORP",
    "JKCEMENT", "AMBUJACEM", "ACC", "DALBHARAT", "RAMCOCEM", "INDIACEM",
    "PERSISTENT", "INDIGO", "SPICEJET", "GMRINFRA", "GAIL", "PETRONET",
    "IGL", "MGL", "ATGL", "OIL", "HINDPETRO", "IOC", "MFSL", "ICICIPRULI",
    "ICICIGI", "GICRE", "NIACL", "STARHEALTH", "MAXHEALTH", "FORTIS",
    "METROPOLIS", "LALPATHLAB", "SUNTV", "ZEEL", "PVRINOX", "SAREGAMA",
    "DELHIVERY", "IDEA", "INDUSTOWER", "HFCL", "TATACHEM", "PIDILITIND",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100
    out[(avg_gain == 0) & (avg_loss == 0)] = 50
    return out


def scan_symbol(symbol: str, rsi_period: int, lookback_days: int):
    """
    Returns a dict signal if the most recent completed setup fired, else None.

    Logic:
      Day T (setup day):   RSI(close) < 10  or  > 90
      Day T+1 (trigger):   open gaps up (>prev close) or gaps down (<prev close)
      Stop loss:           exactly the Day T close (the gap-fill level)

    We check the LAST TWO bars in the fetched window: second-to-last = setup day,
    last = trigger/current day.
    """
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d",
                          progress=False, auto_adjust=False)
    except Exception as e:
        print(f"  [warn] {symbol}: download failed ({e})")
        return None

    if df is None or len(df) < rsi_period + 2:
        return None

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["RSI"] = rsi(df["Close"], rsi_period)
    df = df.dropna(subset=["RSI"])
    if len(df) < 2:
        return None

    setup = df.iloc[-2]   # day T
    trig = df.iloc[-1]    # day T+1 (today / most recent bar)

    setup_rsi = setup["RSI"]
    prev_close = setup["Close"]
    today_open = trig["Open"]

    gap_up = today_open > prev_close
    gap_down = today_open < prev_close

    if setup_rsi < 10 and gap_up:
        return {
            "symbol": symbol,
            "signal": "BUY",
            "setup_date": setup.name.date(),
            "trigger_date": trig.name.date(),
            "setup_rsi": round(float(setup_rsi), 2),
            "prev_close": round(float(prev_close), 2),
            "gap_open": round(float(today_open), 2),
            "gap_pts": round(float(today_open - prev_close), 2),
            "stop_loss (gap-fill level)": round(float(prev_close), 2),
        }
    if setup_rsi > 90 and gap_down:
        return {
            "symbol": symbol,
            "signal": "SELL",
            "setup_date": setup.name.date(),
            "trigger_date": trig.name.date(),
            "setup_rsi": round(float(setup_rsi), 2),
            "prev_close": round(float(prev_close), 2),
            "gap_open": round(float(today_open), 2),
            "gap_pts": round(float(today_open - prev_close), 2),
            "stop_loss (gap-fill level)": round(float(prev_close), 2),
        }
    return None


DEFAULT_SYMBOL_FILE = "data/fno_symbols.txt"


def load_symbol_list(path: str | None):
    # Explicit --symbols path always wins.
    if path:
        with open(path) as f:
            return [line.strip().upper() for line in f if line.strip()]
    # Otherwise prefer the maintained data file (easier to update than code).
    if os.path.exists(DEFAULT_SYMBOL_FILE):
        with open(DEFAULT_SYMBOL_FILE) as f:
            return [line.strip().upper() for line in f if line.strip()]
    # Last resort: built-in snapshot baked into this script.
    return FNO_STOCKS


def main():
    ap = argparse.ArgumentParser(description="RSI-extreme + gap screener (NSE F&O)")
    ap.add_argument("--symbols", help="Path to a text file of symbols (one per line, no .NS needed). Defaults to built-in FNO list.")
    ap.add_argument("--rsi-period", type=int, default=14, help="RSI period (default 14)")
    ap.add_argument("--lookback", type=int, default=60, help="Days of history to pull per symbol (default 60)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: outputs/signals_YYYY-MM-DD_HHMM.csv)")
    args = ap.parse_args()

    if args.out is None:
        os.makedirs("outputs", exist_ok=True)
        args.out = f"outputs/signals_{datetime.now():%Y-%m-%d_%H%M}.csv"

    symbols = load_symbol_list(args.symbols)
    print(f"Scanning {len(symbols)} NSE F&O symbols | RSI period={args.rsi_period} | {datetime.now():%Y-%m-%d %H:%M}")
    print("-" * 70)

    hits = []
    for i, sym in enumerate(symbols, 1):
        sig = scan_symbol(sym, args.rsi_period, args.lookback)
        if sig:
            hits.append(sig)
            print(f"  >>> SIGNAL: {sym} -> {sig['signal']} "
                  f"(setup RSI {sig['setup_rsi']}, gap {sig['gap_pts']} pts, "
                  f"SL {sig['stop_loss (gap-fill level)']})")
        if i % 25 == 0:
            print(f"  ...scanned {i}/{len(symbols)}")

    print("-" * 70)
    if hits:
        out_df = pd.DataFrame(hits)
        out_df.to_csv(args.out, index=False)
        print(f"{len(hits)} signal(s) found. Saved to {args.out}")
        print(out_df.to_string(index=False))
    else:
        print("No signals today. Setup is rare by design (RSI<10 or >90 + matching gap).")

    # Append a one-line record to a running log so history survives in git
    # even on no-signal days.
    log_path = "outputs/run_log.txt"
    os.makedirs("outputs", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | scanned={len(symbols)} | "
                 f"hits={len(hits)} | {[h['symbol']+':'+h['signal'] for h in hits]}\n")

    # Write a clean, human-readable summary for posting as a GitHub Issue / notification.
    summary_lines = [f"# Gap-RSI Scan Result — {datetime.now():%Y-%m-%d %H:%M IST}", ""]
    summary_lines.append(f"Scanned **{len(symbols)}** F&O symbols.\n")
    if hits:
        summary_lines.append(f"## {len(hits)} signal(s) found\n")
        for h in hits:
            summary_lines.append(
                f"- **{h['signal']} — {h['symbol']}**  \n"
                f"  Setup RSI: {h['setup_rsi']} | Prev close: {h['prev_close']} | "
                f"Gap open: {h['gap_open']} ({h['gap_pts']:+} pts)  \n"
                f"  Stop-loss (gap-fill level): **{h['stop_loss (gap-fill level)']}**"
            )
    else:
        summary_lines.append("No signals today. This setup is rare by design "
                              "(RSI<10/>90 + matching gap) — no result is a normal outcome.")
    with open("outputs/last_result.md", "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
