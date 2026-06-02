import pandas as pd

from ta.momentum import RSIIndicator

# ====================================
# LOAD VIX DATA
# ====================================

df = pd.read_csv(
    "data/raw/vix.csv"
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

df["VIX_Return_1D"] = (
    df["Close"].pct_change(1)
)

df["VIX_Return_5D"] = (
    df["Close"].pct_change(5)
)

df["VIX_Return_20D"] = (
    df["Close"].pct_change(20)
)

# ====================================
# RSI
# ====================================

df["VIX_RSI"] = RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# ====================================
# KEEP ONLY FEATURES
# ====================================

vix_features = df[
    [
        "Date",
        "VIX_Return_1D",
        "VIX_Return_5D",
        "VIX_Return_20D",
        "VIX_RSI"
    ]
]

vix_features = vix_features.dropna()

# ====================================
# SAVE
# ====================================

vix_features.to_csv(
    "data/processed/vix_features.csv",
    index=False
)

print("\nVIX Features Created")

print("\nShape:")
print(vix_features.shape)

print("\nColumns:")
print(vix_features.columns)

print("\nSample:")
print(vix_features.head())