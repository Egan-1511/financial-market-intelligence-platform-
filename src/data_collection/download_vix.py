import yfinance as yf
import pandas as pd

print("Downloading India VIX Data...")

vix = yf.download(
    "^INDIAVIX",
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Flatten columns if needed
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

vix = vix.reset_index()

vix.to_csv(
    "data/raw/vix.csv",
    index=False
)

print("\nDownload Complete")

print("\nShape:")
print(vix.shape)

print("\nColumns:")
print(vix.columns)

print("\nSample:")
print(vix.head())