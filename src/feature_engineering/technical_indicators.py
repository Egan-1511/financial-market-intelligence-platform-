import pandas as pd

from ta.momentum import RSIIndicator

from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD
)

from ta.volatility import (
    BollingerBands,
    AverageTrueRange
)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/raw/master_stock_data.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
)

processed_data = []

# ==================================================
# PROCESS EACH STOCK SEPARATELY
# ==================================================

for ticker in df["Ticker"].unique():

    print(f"Processing {ticker}")

    stock_df = df[
        df["Ticker"] == ticker
    ].copy()

    # ==================================================
    # TECHNICAL INDICATORS
    # ==================================================

    stock_df["RSI"] = RSIIndicator(
        close=stock_df["Close"],
        window=14
    ).rsi()

    stock_df["SMA20"] = SMAIndicator(
        close=stock_df["Close"],
        window=20
    ).sma_indicator()

    stock_df["SMA50"] = SMAIndicator(
        close=stock_df["Close"],
        window=50
    ).sma_indicator()

    stock_df["EMA20"] = EMAIndicator(
        close=stock_df["Close"],
        window=20
    ).ema_indicator()

    # ==================================================
    # MACD
    # ==================================================

    macd = MACD(
        close=stock_df["Close"]
    )

    stock_df["MACD"] = (
        macd.macd()
    )

    stock_df["MACD_Signal"] = (
        macd.macd_signal()
    )

    # ==================================================
    # BOLLINGER BANDS
    # ==================================================

    bb = BollingerBands(
        close=stock_df["Close"]
    )

    stock_df["BB_High"] = (
        bb.bollinger_hband()
    )

    stock_df["BB_Low"] = (
        bb.bollinger_lband()
    )

    # ==================================================
    # ATR
    # ==================================================

    atr = AverageTrueRange(
        high=stock_df["High"],
        low=stock_df["Low"],
        close=stock_df["Close"]
    )

    stock_df["ATR"] = (
        atr.average_true_range()
    )

    # ==================================================
    # RETURN FEATURES
    # ==================================================

    stock_df["Return_1D"] = (
        stock_df["Close"].pct_change(1)
    )

    stock_df["Return_5D"] = (
        stock_df["Close"].pct_change(5)
    )

    stock_df["Return_10D"] = (
        stock_df["Close"].pct_change(10)
    )

    stock_df["Return_20D"] = (
        stock_df["Close"].pct_change(20)
    )

    # ==================================================
    # MOMENTUM FEATURES
    # ==================================================

    stock_df["Momentum_10"] = (
        stock_df["Close"] /
        stock_df["Close"].shift(10)
    )

    stock_df["Momentum_20"] = (
        stock_df["Close"] /
        stock_df["Close"].shift(20)
    )

    # ==================================================
    # VOLATILITY FEATURES
    # ==================================================

    stock_df["Volatility_10"] = (
        stock_df["Return_1D"]
        .rolling(10)
        .std()
    )

    stock_df["Volatility_20"] = (
        stock_df["Return_1D"]
        .rolling(20)
        .std()
    )

    # ==================================================
    # VOLUME FEATURES
    # ==================================================

    stock_df["Volume_MA20"] = (
        stock_df["Volume"]
        .rolling(20)
        .mean()
    )

    stock_df["Volume_Ratio"] = (
        stock_df["Volume"]
        /
        stock_df["Volume_MA20"]
    )

    # ==================================================
    # ADVANCED FEATURES
    # ==================================================

    stock_df["Price_vs_SMA50"] = (
        stock_df["Close"]
        /
        stock_df["SMA50"]
    )

    stock_df["Price_vs_EMA20"] = (
        stock_df["Close"]
        /
        stock_df["EMA20"]
    )

    stock_df["BB_Position"] = (
        (stock_df["Close"] - stock_df["BB_Low"])
        /
        (stock_df["BB_High"] - stock_df["BB_Low"])
    )

    stock_df["ATR_Pct"] = (
        stock_df["ATR"]
        /
        stock_df["Close"]
    )

    stock_df["EMA_SMA_Ratio"] = (
        stock_df["EMA20"]
        /
        stock_df["SMA50"]
    )

    # ==================================================
    # SAVE STOCK DATA
    # ==================================================

    processed_data.append(
        stock_df
    )

# ==================================================
# COMBINE ALL STOCKS
# ==================================================

final_df = pd.concat(
    processed_data,
    ignore_index=True
)

# Remove rows with NaN values
final_df = final_df.dropna()

# Save output
final_df.to_csv(
    "data/processed/stock_features.csv",
    index=False
)

# ==================================================
# REPORT
# ==================================================

print("\nFeature Engineering Completed")

print("\nDataset Shape:")
print(final_df.shape)

print("\nUnique Stocks:")
print(final_df["Ticker"].nunique())

print("\nColumns:")
print(final_df.columns)

print("\nSample Data:")
print(final_df.head())