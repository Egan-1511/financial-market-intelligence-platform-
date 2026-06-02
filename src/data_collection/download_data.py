import yfinance as yf
import pandas as pd
from nifty50_stocks import NIFTY50_STOCKS

all_data = []

for stock in NIFTY50_STOCKS:

    print(f"Downloading {stock}")

    df = yf.download(
        stock,
        start="2015-01-01",
        end="2026-01-01",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        continue

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df["Ticker"] = stock

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker"
        ]
    ]

    all_data.append(df)

master_df = pd.concat(
    all_data,
    ignore_index=True
)

master_df.to_csv(
    "data/raw/master_stock_data.csv",
    index=False
)

print("\nShape:", master_df.shape)
print("\nColumns:")
print(master_df.columns)

print("\nMissing Values:")
print(master_df.isnull().sum())

print("\nUnique Stocks:")
print(master_df["Ticker"].nunique())

print(master_df.head())

print(master_df.shape)

print(master_df.columns)

print(master_df.isnull().sum())