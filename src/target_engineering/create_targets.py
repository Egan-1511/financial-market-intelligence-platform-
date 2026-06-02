import pandas as pd

df = pd.read_csv(
    "data/processed/stock_features_market_vix.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    ["Ticker", "Date"]
)

target_data = []

for ticker in df["Ticker"].unique():

    stock_df = df[
        df["Ticker"] == ticker
    ].copy()

    # 7 day future return
    stock_df["Future_7D_Return"] = (
        (
            stock_df["Close"].shift(-7)
            - stock_df["Close"]
        )
        / stock_df["Close"]
    ) * 100

    # 30 day future return
    stock_df["Future_30D_Return"] = (
        (
            stock_df["Close"].shift(-30)
            - stock_df["Close"]
        )
        / stock_df["Close"]
    ) * 100

    # 60 day future return
    stock_df["Future_60D_Return"] = (
        (
            stock_df["Close"].shift(-60)
            - stock_df["Close"]
        )
        / stock_df["Close"]
    ) * 100

    target_data.append(stock_df)

final_df = pd.concat(
    target_data,
    ignore_index=True
)

# Remove rows with missing future values
final_df = final_df.dropna()

# ==========================================
# CLASSIFICATION TARGET
# ==========================================

# Buy if expected 30-day return > 5%
final_df["Buy_Signal"] = (
    final_df["Future_30D_Return"] > 5
).astype(int)

# Save dataset
final_df.to_csv(
    "data/processed/model_dataset.csv",
    index=False
)

print("\nTarget Creation Complete")

print("\nDataset Shape:")
print(final_df.shape)

print("\nBuy Signal Distribution:")
print(
    final_df["Buy_Signal"]
    .value_counts()
)

print("\nBuy Signal Percentage:")
print(
    final_df["Buy_Signal"]
    .value_counts(normalize=True) * 100
)

print(
    final_df[
        [
            "Close",
            "Future_30D_Return",
            "Buy_Signal"
        ]
    ].head()
)

print("\nColumns:")
print(final_df.columns)