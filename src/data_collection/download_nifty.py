import yfinance as yf
import pandas as pd

print("Downloading NIFTY50 Data...")

nifty = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Flatten columns if needed
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty = nifty.reset_index()

nifty.to_csv(
    "data/raw/nifty50.csv",
    index=False
)

print("\nDownload Complete")

print("\nShape:")
print(nifty.shape)

print("\nColumns:")
print(nifty.columns)

print("\nSample:")
print(nifty.head())