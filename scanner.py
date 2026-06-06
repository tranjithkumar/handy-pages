import json
import pandas as pd
import yfinance as yf

OUTPUT_FILE = "data/swing.json"

# TEMPORARY STARTER UNIVERSE
# Replace with dynamic fetch later
UNIVERSE = [
    "RELIANCE.NS",
    "SBIN.NS",
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "TATAMOTORS.NS",
    "TRENT.NS",
    "BEL.NS",
    "HAL.NS"
]


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def get_otm_call(price):

    if price < 100:
        step = 5
    elif price < 1000:
        step = 10
    elif price < 5000:
        step = 50
    else:
        step = 100

    atm = round(price / step) * step
    otm = atm + step

    return f"{int(otm)} CE"


print("Loading Nifty...")

nifty = yf.download(
    "^NSEI",
    period="3mo",
    auto_adjust=True,
    progress=False
)

nifty_close = nifty["Close"]

nifty_monthly_change = (
    (nifty_close.iloc[-1] / nifty_close.iloc[-22]) - 1
) * 100

results = []

for symbol in UNIVERSE:

    try:

        print(f"Checking {symbol}")

        df = yf.download(
            symbol,
            period="1y",
            auto_adjust=True,
            progress=False
        )

        if len(df) < 220:
            continue

        close = df["Close"]

        current_price = float(close.iloc[-1])

        monthly_change = (
            (close.iloc[-1] / close.iloc[-22]) - 1
        ) * 100

        weekly_change = (
            (close.iloc[-1] / close.iloc[-6]) - 1
        ) * 100

        daily_change = (
            (close.iloc[-1] / close.iloc[-2]) - 1
        ) * 100

        rsi = calculate_rsi(close).iloc[-1]

        dma50 = close.rolling(50).mean().iloc[-1]
        dma200 = close.rolling(200).mean().iloc[-1]

        high_52w = close.max()

        high_gap = (
            (high_52w - current_price) / high_52w
        ) * 100

        avg_volume = (
            df["Volume"]
            .tail(20)
            .mean()
        )

        relative_strength = (
            monthly_change - float(nifty_monthly_change)
        )

        # RULES

        if monthly_change <= 5:
            continue

        if rsi <= 60:
            continue

        if not (
            (-0.5 <= daily_change <= 0.5)
            or
            (-2 <= weekly_change <= 2)
        ):
            continue

        if current_price <= dma50:
            continue

        if current_price <= dma200:
            continue

        if high_gap >= 15:
            continue

        if relative_strength <= 0:
            continue

        if avg_volume <= 500000:
            continue

        results.append({
            "symbol": symbol.replace(".NS", ""),
            "close": round(current_price, 2),
            "monthly_change": round(float(monthly_change), 2),
            "rsi": round(float(rsi), 1),
            "high_gap": round(float(high_gap), 1),
            "strike": get_otm_call(current_price)
        })

    except Exception as e:
        print(symbol, e)

results.sort(
    key=lambda x: x["monthly_change"],
    reverse=True
)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"{len(results)} candidates written")
