import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5,
}
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
TARGET = "diabetes"
SEED = 42


def load(path):
    df = pd.read_csv(path)
    df["gender"] = df["gender"].map(GENDER_MAP)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def split(df):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def top_features(model, features, k=5):
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_
    ranking = sorted(zip(features, importances), key=lambda x: -x[1])
    print("\nTop features:")
    for f, w in ranking[:k]:
        print(f"  {f:22s} {w:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="diabetes_prediction_dataset.csv")
    args = parser.parse_args()

    df = load(Path(args.data))
    X_train, X_test, y_train, y_test = split(df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logit = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    logit.fit(X_train_s, y_train)
    evaluate("Logistic Regression (L1, balanced)", logit, X_test_s, y_test)
    top_features(logit, FEATURES)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)
    top_features(rf, FEATURES)


if __name__ == "__main__":
    main()
