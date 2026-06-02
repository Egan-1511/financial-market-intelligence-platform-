import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ====================================
# LOAD NIFTY DATA
# ====================================

df = pd.read_csv(
    "data/raw/nifty50.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    "Date"
)

# ====================================
# RETURNS
# ====================================

df["NIFTY_Return_1D"] = (
    df["Close"].pct_change(1)
)

df["NIFTY_Return_5D"] = (
    df["Close"].pct_change(5)
)

df["NIFTY_Return_20D"] = (
    df["Close"].pct_change(20)
)

# ====================================
# SMA50
# ====================================

df["NIFTY_SMA50"] = SMAIndicator(
    close=df["Close"],
    window=50
).sma_indicator()

# ====================================
# RSI
# ====================================

df["NIFTY_RSI"] = RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# ====================================
# KEEP ONLY USEFUL COLUMNS
# ====================================

nifty_features = df[
    [
        "Date",
        "NIFTY_Return_1D",
        "NIFTY_Return_5D",
        "NIFTY_Return_20D",
        "NIFTY_SMA50",
        "NIFTY_RSI"
    ]
]

nifty_features = nifty_features.dropna()

# ====================================
# SAVE
# ====================================

nifty_features.to_csv(
    "data/processed/nifty_features.csv",
    index=False
)

print("\nNIFTY Features Created")

print("\nShape:")
print(nifty_features.shape)

print("\nColumns:")
print(nifty_features.columns)

print("\nSample:")
print(nifty_features.head())