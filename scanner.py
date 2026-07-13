"""
Weekly Supertrend + EMA9 Scanner — F&O Universe  (HIGH CONVICTION ONLY)
Strategy : Weekly Supertrend(7,2) = BUY  +  Weekly Close > Weekly EMA(9)
           + hard quality gates (liquidity, trend, momentum)
Signal   : STRONG BUY  → fresh Supertrend flip to buy ≤3 weeks + all gates passed
           BUY         → Supertrend buy confirmed (not fresh) + gates passed
           (everything else is silently dropped)
Delivery : Telegram (≤5 picks, richly formatted) + scanner_results.csv

Run modes:
    python weekly_supertrend_ema9_scanner.py            -> live run, sends Telegram
    python weekly_supertrend_ema9_scanner.py --debug     -> prints gate-by-gate
                                                             failure breakdown,
                                                             no Telegram sent
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_HOUR_IST = 15
MARKET_CLOSE_MIN_IST  = 30

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID        = os.environ.get("CHAT_ID")

INDEX = "^NSEI"   # Nifty 50 used as benchmark

ST_PERIOD     = 7      # Supertrend ATR period
ST_MULT       = 2      # Supertrend multiplier
EMA_LEN       = 9      # Weekly EMA length
FRESH_WEEKS   = 3      # "fresh flip" window, in weeks

FNO_CACHE_FILE = "fno_list_cache.csv"
FNO_CACHE_MAX_AGE_DAYS = 7   # reuse cache if fresher than this, when live fetch fails

# ── Emergency fallback ONLY ──
# Used only if BOTH the live NSE fetch and the local cache fail (e.g. no
# internet, NSE blocking the IP). This is deliberately small — just Nifty 50 —
# so a stale/incomplete run is obvious rather than silently passing off an
# old F&O list as current.
EMERGENCY_FALLBACK_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "BHARTIARTL.NS", "ITC.NS", "WIPRO.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "COALINDIA.NS", "TITAN.NS", "TECHM.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "INDUSINDBK.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "ASIANPAINT.NS", "BRITANNIA.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "M&M.NS", "BPCL.NS", "TATACONSUM.NS", "SHRIRAMFIN.NS",
]


def fetch_fno_list_from_nse(timeout: int = 10) -> list[str] | None:
    """
    Fetch the live 'Securities in F&O' list from NSE's JSON API.

    NSE blocks bare requests with a 401/403 unless the request carries
    browser-like headers AND cookies obtained from an initial visit to the
    site. We replicate that here with a requests.Session().

    Returns a list of tickers like ['RELIANCE.NS', 'TCS.NS', ...] on success,
    or None if the fetch fails for any reason (network blocked, NSE changed
    its endpoint/anti-bot check, rate-limited, etc).
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
    }
    api_url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    try:
        session = requests.Session()
        session.headers.update(headers)
        # Visit homepage first to pick up the cookies NSE's API checks for
        session.get("https://www.nseindia.com", timeout=timeout)
        session.get("https://www.nseindia.com/market-data/securities-available-for-trading",
                    timeout=timeout)

        resp = session.get(api_url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("data", [])
        symbols = sorted({row["symbol"].strip() for row in rows if row.get("symbol")})
        if not symbols:
            return None

        tickers = [f"{s}.NS" for s in symbols]
        return tickers

    except Exception as e:
        print(f"[F&O fetch] Live NSE fetch failed: {e}")
        return None


def load_fno_cache() -> list[str] | None:
    """Load the cached F&O list if it exists and isn't too stale."""
    if not os.path.exists(FNO_CACHE_FILE):
        return None
    try:
        cache_df = pd.read_csv(FNO_CACHE_FILE, parse_dates=["fetched_at"])
        if cache_df.empty:
            return None
        age_days = (pd.Timestamp.now() - cache_df["fetched_at"].iloc[0]).days
        if age_days > FNO_CACHE_MAX_AGE_DAYS:
            print(f"[F&O cache] Cache is {age_days}d old (> {FNO_CACHE_MAX_AGE_DAYS}d) — treating as stale.")
            return None
        tickers = cache_df["ticker"].dropna().tolist()
        print(f"[F&O cache] Using cached list from {cache_df['fetched_at'].iloc[0].date()} "
              f"({len(tickers)} tickers, {age_days}d old).")
        return tickers
    except Exception as e:
        print(f"[F&O cache] Failed to read cache: {e}")
        return None


def save_fno_cache(tickers: list[str]) -> None:
    try:
        pd.DataFrame({
            "ticker": tickers,
            "fetched_at": pd.Timestamp.now(),
        }).to_csv(FNO_CACHE_FILE, index=False)
    except Exception as e:
        print(f"[F&O cache] Failed to save cache: {e}")


def get_fno_universe() -> list[str]:
    """
    Resolution order:
      1. Live fetch from NSE (freshest, reflects latest quarterly F&O review)
      2. Local cache, if fresher than FNO_CACHE_MAX_AGE_DAYS
      3. Hardcoded Nifty-50 emergency fallback (logs a loud warning)
    """
    tickers = fetch_fno_list_from_nse()
    if tickers:
        print(f"[F&O universe] Live fetch OK — {len(tickers)} F&O stocks from NSE.")
        save_fno_cache(tickers)
        return tickers

    tickers = load_fno_cache()
    if tickers:
        return tickers

    print("⚠️  [F&O universe] Live fetch AND cache both unavailable — "
          "falling back to a small hardcoded Nifty-50 list. "
          "Results will NOT reflect the full F&O universe. "
          "Check your network / NSE access.")
    return EMERGENCY_FALLBACK_TICKERS


TICKERS = get_fno_universe()

# de-duplicate while preserving order
seen = set()
TICKERS = [t for t in TICKERS if not (t in seen or seen.add(t))]

# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[Telegram] Token/Chat ID missing – skipping.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": chunk}, timeout=10)
        except Exception as e:
            print("Telegram error:", e)

# ─────────────────────────────────────────────
#  DATA FETCH  (3 years daily -> gives ~150 weekly bars, enough for SMA50 weekly)
# ─────────────────────────────────────────────
def get_daily(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="3y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < 260:      # need at least ~1yr of daily bars
            return None
        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = drop_incomplete_today(df)
        return df
    except Exception:
        return None


def drop_incomplete_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the last row is TODAY and the NSE market hasn't closed yet (15:30 IST),
    drop it. yfinance returns a live/partial candle for the current session
    while the market is open — its Volume is only 'so far today', which makes
    same-day-vs-20d-average volume checks fail almost every stock even when
    the real setup is fine. Running the scanner after close (or on a day it
    already closed) is unaffected by this.
    """
    if df.empty:
        return df
    now_ist = datetime.now(IST)
    last_bar_date = df.index[-1].date()
    if last_bar_date == now_ist.date():
        market_closed = (now_ist.hour, now_ist.minute) >= (MARKET_CLOSE_HOUR_IST, MARKET_CLOSE_MIN_IST)
        if not market_closed:
            df = df.iloc[:-1]
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars, week ending Friday."""
    wk = df.resample("W-FRI").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
    return wk

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 7, mult: float = 2.0) -> pd.DataFrame:
    """
    Classic Supertrend implementation.
    Returns df with added columns: 'ST' (the line) and 'ST_dir'
    ('buy' = price above line / uptrend, 'sell' = downtrend).
    """
    hl2 = (df["High"] + df["Low"]) / 2
    atr = atr_wilder(df, period)

    upperband = hl2 + mult * atr
    lowerband = hl2 - mult * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    st_dir = pd.Series(index=df.index, dtype=object)
    st_line = pd.Series(index=df.index, dtype=float)

    close = df["Close"]

    for i in range(len(df)):
        if i == 0:
            st_dir.iloc[i] = "buy"
            st_line.iloc[i] = final_lower.iloc[i]
            continue

        # carry forward bands (standard Supertrend band-locking rule)
        if upperband.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lowerband.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        prev_dir = st_dir.iloc[i - 1]
        if prev_dir == "buy":
            if close.iloc[i] < final_lower.iloc[i]:
                st_dir.iloc[i] = "sell"
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_dir.iloc[i] = "buy"
                st_line.iloc[i] = final_lower.iloc[i]
        else:  # prev_dir == "sell"
            if close.iloc[i] > final_upper.iloc[i]:
                st_dir.iloc[i] = "buy"
                st_line.iloc[i] = final_lower.iloc[i]
            else:
                st_dir.iloc[i] = "sell"
                st_line.iloc[i] = final_upper.iloc[i]

    out = df.copy()
    out["ST"]     = st_line
    out["ST_dir"] = st_dir
    return out

# ─────────────────────────────────────────────
#  BENCHMARK RETURN  (20 trading days ~ 4 weeks, on daily data)
# ─────────────────────────────────────────────
def market_return() -> float:
    df = get_daily(INDEX)
    if df is None or len(df) < 20:
        return 0.0
    close = df["Close"]
    return float((close.iloc[-1] / close.iloc[-20] - 1) * 100)

# ─────────────────────────────────────────────
#  HARD GATES  (all must pass -> else drop silently)
# ─────────────────────────────────────────────
# G1  Weekly Supertrend(7,2) direction == "buy"
# G2  Weekly Close > Weekly EMA(9)
# G3  Daily Close  > 50                       (no penny stocks)
# G4  Daily Volume (10d avg) > 1,00,000        (liquidity)
# G5  Daily Volume > 1.2x 20d avg volume       (interest / expansion)
# G6  Weekly Close > Weekly SMA(20)            (medium-term trend)
# G7  Weekly Close > Weekly SMA(50)            (longer-term trend)
# G8  Weekly RSI(14) between 50 and 75         (strength, not exhausted)
#
# RS-vs-Nifty, momentum, 52w-high proximity -> scoring only (rank, don't gate)

GATE_LABELS = {
    "G1": "Weekly Supertrend = BUY",
    "G2": "Weekly Close > Weekly EMA9",
    "G3": "Price > 50               (no penny stocks)",
    "G4": "10d Avg Vol > 1,00,000   (liquidity)",
    "G5": "Vol > 1.2x 20d Avg Vol   (volume expansion)",
    "G6": "Weekly Close > Weekly SMA20",
    "G7": "Weekly Close > Weekly SMA50",
    "G8": "Weekly RSI(14) 50-75     (strength zone)",
}


def compute_all(ticker: str):
    """Fetch data and compute everything needed for both live run and debug."""
    daily = get_daily(ticker)
    if daily is None:
        return None

    weekly = to_weekly(daily)
    if len(weekly) < 55:      # need enough bars for weekly SMA50 + Supertrend warmup
        return None

    weekly = supertrend(weekly, period=ST_PERIOD, mult=ST_MULT)
    weekly["EMA9"]  = weekly["Close"].ewm(span=EMA_LEN, adjust=False).mean()
    weekly["SMA20"] = weekly["Close"].rolling(20).mean()
    weekly["SMA50"] = weekly["Close"].rolling(50).mean()
    weekly["RSI14"] = rsi(weekly["Close"], 14)

    daily["VOL_SMA20"] = daily["Volume"].rolling(20).mean()

    return daily, weekly


def analyze(ticker: str, mkt_ret: float) -> dict | None:
    data = compute_all(ticker)
    if data is None:
        return None
    daily, weekly = data

    close_now   = float(weekly["Close"].iloc[-1])
    ema9_now    = float(weekly["EMA9"].iloc[-1])
    sma20_now   = float(weekly["SMA20"].iloc[-1])
    sma50_now   = float(weekly["SMA50"].iloc[-1])
    rsi_now     = float(weekly["RSI14"].iloc[-1])
    st_dir_now  = weekly["ST_dir"].iloc[-1]
    st_line_now = float(weekly["ST"].iloc[-1])

    price_now   = float(daily["Close"].iloc[-1])
    vol_10d     = float(daily["Volume"].iloc[-10:].mean())
    vol_20d_avg = float(daily["VOL_SMA20"].iloc[-1])
    vol_today   = float(daily["Volume"].iloc[-1])

    # ── HARD GATES ──
    if st_dir_now != "buy":                       return None   # G1
    if close_now <= ema9_now:                      return None   # G2
    if price_now <= 50:                            return None   # G3
    if vol_10d <= 100_000:                          return None   # G4
    # G5: recent volume trend vs longer average — trailing 10d avg vs 20d avg
    # (not a single day's volume) so a partial/quiet session doesn't wipe out
    # an otherwise valid setup.
    if pd.isna(vol_20d_avg) or vol_10d <= 1.1 * vol_20d_avg:
        return None
    if close_now <= sma20_now:                      return None   # G6
    if close_now <= sma50_now:                       return None   # G7
    if not (50 <= rsi_now <= 75):                    return None   # G8

    # ── Fresh Supertrend flip? (buy started within FRESH_WEEKS) ──
    dirs = weekly["ST_dir"]
    fresh_flip = False
    for back in range(1, FRESH_WEEKS + 1):
        if len(dirs) > back and dirs.iloc[-1 - back] == "sell":
            fresh_flip = True
            break

    # ── Relative strength vs Nifty (20 trading days) ──
    stock_ret_20 = float((daily["Close"].iloc[-1] / daily["Close"].iloc[-20] - 1) * 100)
    rel_strength = round(stock_ret_20 - mkt_ret, 2)

    # ── 52-week high proximity ──
    high_52w   = float(daily["High"].iloc[-252:].max()) if len(daily) >= 252 else float(daily["High"].max())
    from_52w_h = round((price_now / high_52w - 1) * 100, 2)

    # ── 1-week momentum ──
    mom_1w = round(float((weekly["Close"].iloc[-1] / weekly["Close"].iloc[-2] - 1) * 100), 2) \
        if len(weekly) >= 2 else 0.0

    gap_to_st_pct = round((close_now - st_line_now) / st_line_now * 100, 2)

    # ── Quality score (gates already passed -> now rank) ──
    score = 3                              # base: all gates passed
    if fresh_flip:            score += 3
    if 50 <= rsi_now < 65:    score += 2   # sweet spot
    elif 65 <= rsi_now <= 75: score += 1
    if rel_strength > 0:      score += 2
    if rel_strength > 5:      score += 1
    if mom_1w > 1.5:          score += 1
    if from_52w_h >= -10:     score += 1
    if vol_10d > 1_000_000:   score += 1

    if score >= 10 and fresh_flip:
        signal = "🚀 STRONG BUY"
    elif score >= 7:
        signal = "✅ BUY"
    else:
        signal = "👀 WATCH"

    return {
        "ticker":       ticker.replace(".NS", ""),
        "signal":       signal,
        "score":        score,
        "close":        round(price_now, 2),
        "weekly_close": round(close_now, 2),
        "ema9":         round(ema9_now, 2),
        "st_line":      round(st_line_now, 2),
        "gap_to_st_%":  gap_to_st_pct,
        "fresh_flip":   fresh_flip,
        "rsi":          round(rsi_now, 2),
        "rs_vs_nifty":  rel_strength,
        "mom_1w_%":     mom_1w,
        "from_52wh_%":  from_52w_h,
        "vol_10d_k":    round(vol_10d / 1000, 0),
    }

# ─────────────────────────────────────────────
#  DEBUG MODE — gate-by-gate breakdown for every ticker
# ─────────────────────────────────────────────
def debug_gates(mkt_ret: float) -> None:
    rows = []
    for t in TICKERS:
        name = t.replace(".NS", "")
        data = compute_all(t)
        if data is None:
            rows.append({"ticker": name, "failed": "NO_DATA",
                         **{g: "?" for g in GATE_LABELS},
                         "rsi": "-", "rs": "-", "52wh": "-", "vol_k": "-"})
            continue
        daily, weekly = data

        close_now   = float(weekly["Close"].iloc[-1])
        ema9_now    = float(weekly["EMA9"].iloc[-1])
        sma20_now   = float(weekly["SMA20"].iloc[-1])
        sma50_now   = float(weekly["SMA50"].iloc[-1])
        rsi_now     = float(weekly["RSI14"].iloc[-1])
        st_dir_now  = weekly["ST_dir"].iloc[-1]

        price_now   = float(daily["Close"].iloc[-1])
        vol_10d     = float(daily["Volume"].iloc[-10:].mean())
        vol_20d_avg = float(daily["VOL_SMA20"].iloc[-1])
        vol_today   = float(daily["Volume"].iloc[-1])

        stock_ret = float((daily["Close"].iloc[-1] / daily["Close"].iloc[-20] - 1) * 100)
        rs        = round(stock_ret - mkt_ret, 2)
        high_52w  = float(daily["High"].iloc[-252:].max()) if len(daily) >= 252 else float(daily["High"].max())
        from_ath  = round((price_now / high_52w - 1) * 100, 2)

        g1 = "✅" if st_dir_now == "buy" else "❌"
        g2 = "✅" if close_now > ema9_now else "❌"
        g3 = "✅" if price_now > 50 else "❌"
        g4 = "✅" if vol_10d > 100_000 else "❌"
        g5 = "✅" if (not pd.isna(vol_20d_avg) and vol_10d > 1.1 * vol_20d_avg) else "❌"
        g6 = "✅" if close_now > sma20_now else "❌"
        g7 = "✅" if close_now > sma50_now else "❌"
        g8 = "✅" if 50 <= rsi_now <= 75 else "❌"

        gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "G8": g8}
        failed = [k for k, v in gates.items() if v == "❌"]

        rows.append({
            "ticker": name,
            "failed": ",".join(failed) if failed else "NONE — qualifies!",
            **gates,
            "rsi": round(rsi_now, 1) if not pd.isna(rsi_now) else "-",
            "rs": rs,
            "52wh": from_ath,
            "vol_k": round(vol_10d / 1000, 0),
        })
 
    dbg = pd.DataFrame(rows).sort_values("failed")
 
    print(f"\n{'═'*72}")
    print(f"  DEBUG — Weekly Supertrend(7,2) + EMA9 Gate Analysis  |  Nifty 20d: {round(mkt_ret,2):+.2f}%")
    print(f"{'═'*72}")
    for gate, label in GATE_LABELS.items():
        fails = dbg[dbg[gate] == "❌"].shape[0]
        bar = "█" * (fails // 2)
        print(f"  {gate}  {label:35s}  {fails:3d} fails  {bar}")
    print(f"{'─'*72}")
 
    dbg["fail_count"] = dbg["failed"].apply(lambda x: 0 if x == "NONE — qualifies!" else len(x.split(",")))
    near_misses = dbg[dbg["fail_count"].between(1, 2)].sort_values("fail_count")
    winners = dbg[dbg["fail_count"] == 0]
 
    print(f"\n  QUALIFIED  (0 gate failures)")
    if winners.empty:
        print("    — none —")
    else:
        for _, r in winners.iterrows():
            print(f"    ✔ {r['ticker']:18s}  RSI {r['rsi']}  RS {r['rs']:+.1f}%  52wH {r['52wh']}%  Vol {int(r['vol_k'])}k")
 
    print(f"\n  NEAR MISSES  (1–2 gate failures)")
    if near_misses.empty:
        print("    — none —")
    else:
        for _, r in near_misses.iterrows():
            print(f"    ✗ {r['ticker']:18s}  FAILED: {r['failed']:20s}  RSI {r['rsi']}  RS {r['rs']:+.1f}%  52wH {r['52wh']}%  Vol {int(r['vol_k'])}k")
 
    print(f"\n{'═'*72}\n")
    dbg.to_csv("debug_gates.csv", index=False)
    print("  Full breakdown saved → debug_gates.csv")
 
# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    mkt_ret     = market_return()
    weak_market = mkt_ret < -1
 
    results = []
    for t in TICKERS:
        try:
            res = analyze(t, mkt_ret)
            if res:
                results.append(res)
                print(f"  ✔ {res['ticker']:18s}  score={res['score']}  {res['signal']}")
        except Exception as e:
            print(f"  Skip {t}: {e}")
 
    if not results:
        msg = "No high-conviction setups today. Sit tight. 🧘"
        print(msg)
        send_telegram(msg)
        return
 
    df = pd.DataFrame(results)
    if weak_market:
        df["score"] = df["score"] - 1
 
    df = df.sort_values(["score", "rs_vs_nifty"], ascending=False)
    df.to_csv("scanner_results.csv", index=False)
 
    picks = df[df["score"] >= 7].head(5)
 
    if picks.empty:
        msg = "No high-conviction setups today. Sit tight. 🧘"
    else:
        lines = []
        for rank, (_, r) in enumerate(picks.iterrows(), 1):
            fresh_tag = "  ⚡ FRESH FLIP" if r["fresh_flip"] else ""
            lines.append(
                f"#{rank}  {r['ticker']}{fresh_tag}\n"
                f"    {r['signal']}  •  Score {r['score']}\n"
                f"    CMP ₹{r['close']}  |  Wk Close ₹{r['weekly_close']}  |  EMA9 ₹{r['ema9']}\n"
                f"    ST line ₹{r['st_line']}  (gap {r['gap_to_st_%']}%)\n"
                f"    RSI {r['rsi']}  |  RS {r['rs_vs_nifty']:+.1f}%  |  1w {r['mom_1w_%']:+.1f}%\n"
                f"    52wH gap {r['from_52wh_%']}%  |  Vol {int(r['vol_10d_k'])}k\n"
            )
        msg = "\n".join(lines)
 
    print("\n" + msg)
    send_telegram(msg)
 
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if "--debug" in sys.argv:
        print("Running in DEBUG mode — no Telegram messages will be sent.")
        mkt_ret = market_return()
        debug_gates(mkt_ret)
    else:
        if not TELEGRAM_TOKEN or not CHAT_ID:
            raise ValueError("Set BOT_TOKEN and CHAT_ID env vars before running.")
        run()
