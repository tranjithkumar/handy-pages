import yfinance as yf
import pandas as pd
import requests

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

INDEX = "^NSEI"

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "MARUTI.NS",
    "TATAMOTORS.NS", "BHARTIARTL.NS", "ITC.NS"
]

# ---------------- TELEGRAM ----------------
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# ---------------- INDICATORS ----------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    return df.dropna()

# ---------------- SCORING CORE ----------------
def analyze(ticker, market_ret):
    df = get_data(ticker)
    if len(df) < 60:
        return None

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = rsi(df["Close"])

    close = df["Close"].iloc[-1]
    ema20 = df["EMA20"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]
    rsi_val = df["RSI"].iloc[-1]

    stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100
    rel_strength = stock_ret - market_ret

    score = 0

    # trend structure
    score += 2 if close > ema20 else 0
    score += 2 if ema20 > ema50 else 0

    # momentum zone
    if 50 < rsi_val < 70:
        score += 2
    elif 70 <= rsi_val < 80:
        score += 1

    # relative strength (core edge)
    score += 3 if rel_strength > 0 else 0
    score += 2 if rel_strength > 5 else 0

    # signal engine
    if score >= 8:
        signal = "STRONG BUY"
    elif score >= 6:
        signal = "WATCH"
    else:
        signal = "IGNORE"

    return {
        "ticker": ticker,
        "score": score,
        "signal": signal,
        "rel_strength": round(rel_strength, 2),
        "rsi": round(rsi_val, 2)
    }

# ---------------- MAIN ----------------
def run():
    idx = get_data(INDEX)
    market_ret = (idx["Close"].iloc[-1] / idx["Close"].iloc[-20] - 1) * 100

    results = []

    for t in TICKERS:
        res = analyze(t, market_ret)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    actionable = df[df["signal"] != "IGNORE"].head(5)

    # ---------------- MESSAGE ----------------
    if actionable.empty:
        msg = "Scanner: No actionable setups this week."
    else:
        msg = "📊 Weekly Scanner Picks\n\n"
        for _, r in actionable.iterrows():
            msg += (
                f"{r['ticker']} | {r['signal']} | "
                f"Score {r['score']} | RS {r['rel_strength']} | RSI {r['rsi']}\n"
            )

    print(msg)
    send_telegram(msg)

    df.to_csv("scanner_results.csv", index=False)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    run()
