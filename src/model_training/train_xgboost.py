import pandas as pd


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib

# Load dataset
df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

print("Dataset Shape:", df.shape)

print("\nUnique Stocks:")
print(df["Ticker"].nunique())

print("\nStocks:")
print(df["Ticker"].unique())

df = pd.get_dummies(
    df,
    columns=["Ticker"]
)

exclude_columns = [
    "Date",
    "Future_7D_Return",
    "Future_30D_Return",
    "Future_60D_Return"
]

features = [
    col
    for col in df.columns
    if col not in exclude_columns
]

print(f"Total Features: {len(features)}")
print(features)

# Target
target = "Future_30D_Return"

X = df[features]

y = df[target]

# Time-series split
split_index = int(
    len(df) * 0.8
)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Model
model = XGBRegressor(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

# Predictions
predictions = model.predict(
    X_test
)

# Metrics
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Performance")

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

# Save model
joblib.dump(
    model,
    "models/xgboost_return_predictor.pkl"
)

print("\nModel Saved Successfully")