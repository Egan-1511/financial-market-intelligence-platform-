import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

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

target = "Buy_Signal"

X = df[features]
y = df[target]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(
    len(df) * 0.8
)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

print("\nTraining Classifier...")

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

print("\nClassifier Performance")

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
    "Importance": model.feature_importances_
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
    "models/classifier_feature_importance.csv",
    index=False
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/xgboost_buy_classifier.pkl"
)

joblib.dump(
    features,
    "models/model_features.pkl"
)
print("\nClassifier Saved Successfully")