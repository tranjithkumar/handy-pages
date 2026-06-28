"""
Golden Cross Scanner — Nifty 200 / F&O Universe  (HIGH CONVICTION ONLY)
Strategy : EMA 50 / EMA 200 Golden Cross + hard quality gates
Signal   : STRONG BUY  → fresh cross ≤10 bars + all gates passed
           BUY         → cross confirmed + gates passed
           (everything else is silently dropped)
Delivery : Telegram (≤5 picks, richly formatted) + scanner_results.csv
"""

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID        = os.environ.get("CHAT_ID")

INDEX = "^NSEI"   # Nifty 50 used as benchmark

# ── Nifty 200 + active F&O universe ──────────
# Core Nifty 50 + midcap / F&O names commonly traded.
# Extend this list freely; all tickers are .NS (NSE).
TICKERS = [
    # ── Large-cap / Nifty 50 ──
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "MARUTI.NS",
    "TATAMOTORS.NS", "BHARTIARTL.NS", "ITC.NS",
    "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "TITAN.NS",
    "NESTLEIND.NS", "TECHM.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "INDUSINDBK.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "CIPLA.NS",
    "DIVISLAB.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "BRITANNIA.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "M&M.NS", "BPCL.NS",
    "TATACONSUM.NS", "SHRIRAMFIN.NS",

    # ── Nifty Next 50 / Midcap F&O ──
    "PIDILITIND.NS", "SIEMENS.NS", "HAVELLS.NS", "VOLTAS.NS",
    "BERGEPAINT.NS", "COLPAL.NS", "MARICO.NS", "GODREJCP.NS",
    "DABUR.NS", "EMAMILTD.NS", "UBL.NS", "MCDOWELL-N.NS",
    "TATAPOWER.NS", "ADANIGREEN.NS", "ADANITRANS.NS",
    "ZOMATO.NS", "PAYTM.NS", "DMART.NS", "NYKAA.NS",
    "POLICYBZR.NS", "IRCTC.NS", "IRFC.NS", "RVNL.NS",
    "RAILVIKAS.NS", "PFC.NS", "RECLTD.NS",
    "BANKBARODA.NS", "CANARABANK.NS", "PNB.NS", "UNIONBANK.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS",
    "CHOLAFIN.NS", "BAJAJHFL.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
    "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
    "LTTS.NS", "KPITTECH.NS", "TATAELXSI.NS",
    "ASTRAL.NS", "AAPL.NS", "SUPREMEIND.NS",
    "PIIND.NS", "UPL.NS", "CHAMBALFERT.NS", "COROMANDEL.NS",
    "SYNGENE.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
    "MAXHEALTH.NS", "FORTIS.NS", "NARAYANAMRN.NS",
    "PAGEIND.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS",
    "JINDALSTEL.NS", "RATNAMANI.NS", "KALYANKJIL.NS",
    "INDIAMART.NS", "NAUKRI.NS", "JUSTDIAL.NS",
    "OBEROIRLTY.NS", "DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS",
    "PHOENIXLTD.NS", "CONCOR.NS", "BLUEDART.NS", "DELHIVERY.NS",
    "ZYDUSLIFE.NS", "ALKEM.NS", "TORNTPHARM.NS", "AUROPHARMA.NS",
    "GLENMARK.NS", "IPCALAB.NS", "NATCOPHARM.NS",
    "HINDPETRO.NS", "IOC.NS", "PETRONET.NS", "GAIL.NS",
    "OIL.NS", "MGL.NS", "IGL.NS", "CESC.NS",
    "TRENT.NS", "ABFRL.NS", "VBL.NS", "RADICO.NS",
    "BALKRISIND.NS", "CEAT.NS", "MRF.NS", "APOLLOTYRE.NS",
    "ASHOKLEY.NS", "ESCORTS.NS", "TIINDIA.NS", "MOTHERSON.NS",
    "BHARATFORG.NS", "BOSCHLTD.NS", "SCHAEFFLER.NS",
    "LINDEINDIA.NS", "DEEPAKNTR.NS", "AARTIIND.NS", "ATUL.NS",
    "SRF.NS", "GALAXYSURF.NS", "NAVINFLUOR.NS",
    "ABCAPITAL.NS", "ICICIGI.NS", "ICICIPRU.NS", "HDFCAMC.NS",
    "NIPPONLIFE.NS", "SUNDARMFIN.NS", "LICHSGFIN.NS",
    "STARHEALTH.NS", "GODIGIT.NS",
    "GMRINFRA.NS", "AIAENGINEERING.NS", "GRINDWELL.NS",
    "KFINTECH.NS", "CDSL.NS", "BSE.NS",
]

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
    # Telegram caps a single message at ~4096 chars; chunk if needed
    for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": chunk}, timeout=10)
        except Exception as e:
            print("Telegram error:", e)

# ─────────────────────────────────────────────
#  DATA FETCH  (1 year for reliable EMA 200)
# ─────────────────────────────────────────────
def get_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="15mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        df = df[~df.index.duplicated(keep="last")]
        # Need at least 210 bars for EMA 200 to be meaningful
        if len(df) < 210:
            return None
        return df
    except Exception:
        return None

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ─────────────────────────────────────────────
#  BENCHMARK RETURN
# ─────────────────────────────────────────────
def market_return() -> float:
    df = get_data(INDEX)
    if df is None or len(df) < 20:
        return 0.0
    close = pd.Series(df["Close"].values.flatten())
    return float((close.iloc[-1] / close.iloc[-20] - 1) * 100)

# ─────────────────────────────────────────────
#  HARD GATES  (all must pass → else drop)
# ─────────────────────────────────────────────
# G1  EMA 50 > EMA 200  (golden cross active)
# G2  Price > EMA 50    (price above short MA)
# G3  RSI 45–78         (momentum not overbought/dead)
# G4  RS vs Nifty > 0   (outperforming benchmark)
# G5  Within 20% of 52w high  (uptrend not broken)
# G6  Volume 10d avg > 500 k  (liquid enough)

# ─────────────────────────────────────────────
#  CORE ANALYSIS — High Conviction Golden Cross
# ─────────────────────────────────────────────
def analyze(ticker: str, mkt_ret: float) -> dict | None:
    df = get_data(ticker)
    if df is None:
        return None

    close_s = df["Close"].squeeze()
    vol_s   = df["Volume"].squeeze()

    # ── Indicators ────────────────────────────
    df["EMA50"]  = close_s.ewm(span=50,  adjust=False).mean()
    df["EMA200"] = close_s.ewm(span=200, adjust=False).mean()
    df["RSI"]    = rsi(close_s)
    df["ATR"]    = atr(df)

    ema50  = df["EMA50"].squeeze()
    ema200 = df["EMA200"].squeeze()

    close_now = float(close_s.iloc[-1])
    e50_now   = float(ema50.iloc[-1])
    e200_now  = float(ema200.iloc[-1])
    rsi_val   = float(df["RSI"].iloc[-1])
    atr_val   = float(df["ATR"].iloc[-1])
    vol_10d   = float(vol_s.iloc[-10:].mean())

    # Relative strength vs Nifty (20-day)
    stock_ret_20 = float((close_s.iloc[-1] / close_s.iloc[-20] - 1) * 100)
    rel_strength = round(stock_ret_20 - mkt_ret, 2)

    # 52-week high proximity
    high_52w   = float(df["High"].iloc[-252:].max()) if len(df) >= 252 else float(df["High"].max())
    from_52w_h = round((close_now / high_52w - 1) * 100, 2)

    # ── HARD GATES — exit immediately if any fail ──
    if e50_now <= e200_now:          return None   # G1: no golden cross
    if close_now <= e50_now:         return None   # G2: price below EMA50
    if not (45 <= rsi_val <= 78):    return None   # G3: bad RSI zone
    if rel_strength <= 0:            return None   # G4: underperforming Nifty
    if from_52w_h < -20:             return None   # G5: trend broken (>20% off high)
    if vol_10d < 500_000:            return None   # G6: illiquid

    # ── Fresh cross detection (≤15 bars) ──────
    FRESH_WINDOW = 15
    prev_e50  = float(ema50.iloc[-FRESH_WINDOW - 1])
    prev_e200 = float(ema200.iloc[-FRESH_WINDOW - 1])
    fresh_cross = prev_e50 < prev_e200          # was below → now above

    cross_gap_pct = round((e50_now - e200_now) / e200_now * 100, 2)

    # 5-day momentum
    mom_5d = round(float((close_s.iloc[-1] / close_s.iloc[-5] - 1) * 100), 2)

    # ── Quality score (on top of hard gates) ──
    score = 0

    score += 3                              # base: all gates cleared
    if fresh_cross:         score += 3      # freshness premium
    if 55 < rsi_val < 70:  score += 2      # sweet-spot RSI
    if rel_strength > 3:   score += 2      # strong outperformer
    if rel_strength > 7:   score += 1      # exceptional RS
    if mom_5d > 1.5:       score += 1      # short-term thrust
    if from_52w_h >= -8:   score += 1      # near ATH expansion
    if cross_gap_pct < 3:  score += 1      # EMA50 just cleared 200 (not stretched)

    # ── Signal ────────────────────────────────
    if score >= 10 and fresh_cross:
        signal = "🚀 STRONG BUY"
    elif score >= 7:
        signal = "✅ BUY"
    else:
        signal = "👀 WATCH"

    # Swing target: 2.5× ATR (tighter = more conservative)
    upside_target = round(close_now + 2.5 * atr_val, 2)
    upside_pct    = round((upside_target / close_now - 1) * 100, 2)

    return {
        "ticker":       ticker.replace(".NS", ""),
        "signal":       signal,
        "score":        score,
        "close":        round(close_now, 2),
        "ema50":        round(e50_now, 2),
        "ema200":       round(e200_now, 2),
        "cross_gap_%":  cross_gap_pct,
        "fresh_cross":  fresh_cross,
        "rsi":          round(rsi_val, 2),
        "rs_vs_nifty":  rel_strength,
        "mom_5d_%":     mom_5d,
        "from_52wh_%":  from_52w_h,
        "vol_10d_k":    round(vol_10d / 1000, 0),
        "upside_tgt":   upside_target,
        "upside_%":     upside_pct,
    }

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    print("── Golden Cross Scanner (High Conviction) ──")
    mkt_ret     = market_return()
    weak_market = mkt_ret < -1

    results = []
    for t in TICKERS:
        try:
            res = analyze(t, mkt_ret)
            if res:
                results.append(res)
                print(f"  ✔ {res['ticker']:18s}  score={res['score']}  {res['signal']}")
            # else: silently dropped by hard gates
        except Exception as e:
            print(f"  Skip {t}: {e}")

    if not results:
        msg = "Golden Cross Scanner: No high-conviction setups today. Sit tight. 🧘"
        print(msg)
        send_telegram(msg)
        return

    df = pd.DataFrame(results)

    # Penalise in weak market
    if weak_market:
        df["score"] = df["score"] - 1

    df = df.sort_values(["score", "rs_vs_nifty"], ascending=False)
    df.to_csv("scanner_results.csv", index=False)

    # ── Only TRUE buys go to Telegram (score ≥ 7, max 5) ──
    picks = df[df["score"] >= 7].head(5)

    nifty_tag = f"Nifty 20d: {round(mkt_ret, 2):+.2f}%"
    if weak_market:
        nifty_tag += "  ⚠️ Weak market — size down"

    if picks.empty:
        msg = f"📈 Golden Cross Scanner  |  {nifty_tag}\n\nNo high-conviction buys today. All candidates failed quality gates. Cash is a position. 💰"
    else:
        lines = [f"📈 Golden Cross Scanner  |  {nifty_tag}\n"]
        lines.append(f"{'─'*36}")
        for rank, (_, r) in enumerate(picks.iterrows(), 1):
            fresh_tag  = "  ⚡ FRESH CROSS" if r["fresh_cross"] else ""
            lines.append(
                f"#{rank}  {r['ticker']}{fresh_tag}\n"
                f"    {r['signal']}  •  Score {r['score']}\n"
                f"    CMP ₹{r['close']}  |  EMA50 ₹{r['ema50']}  |  EMA200 ₹{r['ema200']}\n"
                f"    RSI {r['rsi']}  |  RS {r['rs_vs_nifty']:+.1f}%  |  5d {r['mom_5d_%']:+.1f}%\n"
                f"    52wH gap {r['from_52wh_%']}%  |  Vol {int(r['vol_10d_k'])}k\n"
                f"    🎯 Swing tgt ₹{r['upside_tgt']} (+{r['upside_%']}%)\n"
            )
        lines.append(f"{'─'*36}")
        lines.append(f"Scanned {len(TICKERS)} stocks  •  {len(results)} passed gates  •  {len(picks)} actionable")
        msg = "\n".join(lines)

    print("\n" + msg)
    send_telegram(msg)

# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise ValueError("Set BOT_TOKEN and CHAT_ID env vars before running.")
    run()
