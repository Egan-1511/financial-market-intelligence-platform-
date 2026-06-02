import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from catboost import CatBoostClassifier

import joblib

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

# One-Hot Encode Stocks
df = pd.get_dummies(
    df,
    columns=["Ticker"]
)

# ==========================================
# FEATURES
# ==========================================

exclude_columns = [
    "Date",
    "Future_7D_Return",
    "Future_30D_Return",
    "Future_60D_Return",
    "Buy_Signal"
]

features = [
    col
    for col in df.columns
    if col not in exclude_columns
]

print(f"Total Features: {len(features)}")

# ==========================================
# TARGET
# ==========================================

X = df[features]

y = df["Buy_Signal"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# MODEL
# ==========================================

model = CatBoostClassifier(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100
)

print("\nTraining CatBoost...")

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nCatBoost Performance")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.get_feature_importance()
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")

print(
    feature_importance.head(20)
)

feature_importance.to_csv(
    "models/catboost_feature_importance.csv",
    index=False
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/catboost_buy_classifier.pkl"
)

print("\nCatBoost Model Saved Successfully")