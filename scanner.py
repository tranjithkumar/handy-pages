import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- CONFIG ----------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS",
    "ITC.NS", "BHARTIARTL.NS"
]

LOOKBACK = "6mo"
MIN_VOLUME_RATIO = 1.2  # volume spike filter

# ---------------- INDICATORS ----------------
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ---------------- SCAN LOGIC ----------------
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period=LOOKBACK, interval="1d", progress=False)

        if df is None or len(df) < 60:
            return None

        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI"] = rsi(df["Close"])
        df["VolAvg20"] = df["Volume"].rolling(20).mean()

        latest = df.iloc[-1]

        close = latest["Close"]
        ema20 = latest["EMA20"]
        ema50 = latest["EMA50"]
        rsi_val = latest["RSI"]
        volume = latest["Volume"]
        vol_avg = latest["VolAvg20"]

        # 52W / lookback high proxy
        high_6m = df["High"].max()
        breakout_strength = (close / high_6m - 1) * 100

        # volume condition
        volume_ratio = volume / vol_avg if vol_avg > 0 else 0

        score = 0

        # Trend scoring
        if close > ema20:
            score += 2
        if ema20 > ema50:
            score += 2

        # Momentum
        if 50 < rsi_val < 70:
            score += 2
        elif 70 <= rsi_val <= 80:
            score += 1
        elif rsi_val > 80:
            score -= 1  # overheated

        # Volume confirmation
        if volume_ratio > MIN_VOLUME_RATIO:
            score += 2

        # breakout positioning
        if breakout_strength > -5:
            score += 1
        if breakout_strength > 0:
            score += 2

        return {
            "ticker": ticker,
            "close": round(close, 2),
            "rsi": round(rsi_val, 2),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "volume_ratio": round(volume_ratio, 2),
            "breakout_%": round(breakout_strength, 2),
            "score": score
        }

    except Exception as e:
        return None

# ---------------- RUN SCANNER ----------------
def run_scanner():
    results = []

    for t in TICKERS:
        res = analyze_stock(t)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    if df.empty:
        print("No candidates found")
        return

    df = df.sort_values(by="score", ascending=False)

    print("\n=== TOP SCANNER PICKS ===")
    print(df.to_string(index=False))

    df.to_csv("scanner_results.csv", index=False)
    print("\nSaved: scanner_results.csv")

if __name__ == "__main__":
    run_scanner()
