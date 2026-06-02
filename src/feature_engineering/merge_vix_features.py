import pandas as pd

market_df = pd.read_csv(
    "data/processed/stock_features_market.csv"
)

vix_df = pd.read_csv(
    "data/processed/vix_features.csv"
)

market_df["Date"] = pd.to_datetime(
    market_df["Date"]
)

vix_df["Date"] = pd.to_datetime(
    vix_df["Date"]
)

final_df = market_df.merge(
    vix_df,
    on="Date",
    how="left"
)

final_df = final_df.dropna()

final_df.to_csv(
    "data/processed/stock_features_market_vix.csv",
    index=False
)

print("\nMerge Complete")
print(final_df.shape)
print(final_df["Ticker"].nunique())