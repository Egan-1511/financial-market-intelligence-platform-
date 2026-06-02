import pandas as pd
import joblib

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/xgboost_buy_classifier.pkl"
)

features = joblib.load(
    "models/model_features.pkl"
)

# =====================================
# ENCODE TICKERS
# =====================================

df_encoded = pd.get_dummies(
    df,
    columns=["Ticker"]
)

# Add missing columns if needed
for col in features:

    if col not in df_encoded.columns:

        df_encoded[col] = 0

df_encoded = df_encoded.reindex(
    columns=df_encoded.columns,
    fill_value=0
)

# =====================================
# PREDICT
# =====================================

X = df_encoded[features]

df_encoded["Buy_Probability"] = (
    model.predict_proba(X)[:, 1]
)

# =====================================
# BACKTEST
# =====================================

results = []

dates = sorted(
    df_encoded["Date"].unique()
)

test_dates = dates[::30]

for current_date in test_dates:

    current_data = df_encoded[
        df_encoded["Date"] == current_date
    ].copy()

    if len(current_data) < 5:
        continue

    top5 = current_data.sort_values(
        "Buy_Probability",
        ascending=False
    ).head(5)

    avg_return = (
        top5["Future_30D_Return"]
        .mean()
    )

    results.append(
        {
            "Date": current_date,
            "Return": avg_return
        }
    )

# =====================================
# RESULTS
# =====================================

results_df = pd.DataFrame(
    results
)

average_return = (
    results_df["Return"]
    .mean()
)

win_rate = (
    (
        results_df["Return"] > 0
    ).mean()
    * 100
)

total_periods = len(
    results_df
)

print("\nREAL AI BACKTEST")

print(
    f"\nPeriods Tested : {total_periods}"
)

print(
    f"Average Return : {average_return:.2f}%"
)

print(
    f"Win Rate       : {win_rate:.2f}%"
)

print("\nSample Results:")

print(
    results_df.head()
)

# =====================================
# SAVE
# =====================================

results_df.to_csv(
    "predictions/real_backtest_results.csv",
    index=False
)

print(
    "\nBacktest Saved Successfully"
)