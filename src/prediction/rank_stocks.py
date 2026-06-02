import pandas as pd
import joblib

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

# Keep latest record for each stock
latest_data = (
    df.sort_values("Date")
      .groupby("Ticker")
      .tail(1)
      .copy()
)

# =====================================
# PREPARE FEATURES
# =====================================

latest_data_encoded = pd.get_dummies(
    latest_data,
    columns=["Ticker"]
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/xgboost_buy_classifier.pkl"
)

# =====================================
# MATCH TRAINING FEATURES
# =====================================

exclude_columns = [
    "Date",
    "Future_7D_Return",
    "Future_30D_Return",
    "Future_60D_Return",
    "Buy_Signal"
]

features = [
    col
    for col in latest_data_encoded.columns
    if col not in exclude_columns
]

X = latest_data_encoded[features]

# =====================================
# PREDICT
# =====================================

probabilities = model.predict_proba(X)[:, 1]

latest_data["Buy_Probability"] = (
    probabilities * 100
)

# =====================================
# RANK STOCKS
# =====================================

ranking = latest_data[
    [
        "Ticker",
        "Close",
        "Buy_Probability"
    ]
]

ranking = ranking.sort_values(
    by="Buy_Probability",
    ascending=False
)

# =====================================
# OUTPUT
# =====================================

print("\nTOP 10 STOCKS\n")

print(
    ranking.head(10)
)

ranking.to_csv(
    "predictions/stock_rankings.csv",
    index=False
)

print(
    "\nRanking saved to:"
)

print(
    "predictions/stock_rankings.csv"
)