import pandas as pd

# =====================================
# LOAD FILES
# =====================================

stock_df = pd.read_csv(
    "data/processed/stock_features.csv"
)

nifty_df = pd.read_csv(
    "data/processed/nifty_features.csv"
)

# =====================================
# DATE CONVERSION
# =====================================

stock_df["Date"] = pd.to_datetime(
    stock_df["Date"]
)

nifty_df["Date"] = pd.to_datetime(
    nifty_df["Date"]
)

# =====================================
# MERGE
# =====================================

merged_df = stock_df.merge(
    nifty_df,
    on="Date",
    how="left"
)

# =====================================
# REMOVE MISSING ROWS
# =====================================

merged_df = merged_df.dropna()

# =====================================
# SAVE
# =====================================

merged_df.to_csv(
    "data/processed/stock_features_market.csv",
    index=False
)

print("\nMerge Complete")

print("\nShape:")
print(merged_df.shape)

print("\nUnique Stocks:")
print(merged_df["Ticker"].nunique())

print("\nNew Columns Added:")

market_cols = [
    "NIFTY_Return_1D",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "NIFTY_SMA50",
    "NIFTY_RSI"
]

print(market_cols)

print("\nSample:")

print(
    merged_df[
        [
            "Ticker",
            "Date",
            "RSI",
            "NIFTY_RSI"
        ]
    ].head()
)