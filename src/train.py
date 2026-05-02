import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from src.features import extract_features

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/indian_address_dataset.csv")

# Basic checks
print("Initial shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna(subset=["address", "label"])

# Normalize text
df["address"] = df["address"].astype(str).str.lower().str.strip()

# Ensure labels are 0/1
df["label"] = df["label"].astype(int)

# Remove duplicates
df = df.drop_duplicates(subset=["address"])

print("After cleaning:", df.shape)

# =========================
# 3. FEATURES
# =========================
X_text = df["address"]
y = df["label"]

# TF-IDF (IMPORTANT: keep consistent later)
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X_text).toarray()

# Numerical features
X_num = np.array([extract_features(x) for x in X_text])

# Combine features
X = np.hstack([X_tfidf, X_num])

print("Final feature shape:", X.shape)

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # ⚠️ VERY IMPORTANT
    random_state=42
)

# =========================
# 5. MODEL
# =========================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# =========================
# 6. EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 7. SAVE MODEL
# =========================
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model and vectorizer saved!")