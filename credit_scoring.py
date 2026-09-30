

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
)

SEED = 42
DATA_PATH = "german_credit.csv"
TARGET_COL = "credit_risk"  # 1 = good credit risk, 0 = bad credit risk


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

    # One-hot encode: each category becomes its own 0/1 column.
    # drop_first=True drops one category per column to avoid redundancy
    # (e.g. if "housing" is not "own" and not "for free", it must be "rent").
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Columns after one-hot encoding: {X_encoded.shape[1]} (was {X.shape[1]})")

    return X_encoded, y


def tune_model(name, model, param_grid, X_train, y_train):
    """Runs GridSearchCV (5-fold cross-validation) to find the best
    hyperparameters for this model, scored by ROC-AUC."""
    search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"{name}: best params = {search.best_params_}, "
          f"best CV ROC-AUC = {search.best_score_:.3f}")
    return search.best_estimator_


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Bad (0)", "Good (1)"]))
    print(f"ROC-AUC: {auc:.3f}")

    return {
        "name": name, "precision": precision, "recall": recall,
        "f1": f1, "auc": auc, "y_pred": y_pred, "y_proba": y_proba,
    }


def main():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\nTrain samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTuning hyperparameters (this runs 5-fold cross-validation per model)...")

    # Logistic Regression: tune regularization strength C
    lr_best = tune_model(
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10]},
        X_train_scaled, y_train
    )

    # Decision Tree: tune depth and minimum samples per leaf (controls overfitting)
    dt_best = tune_model(
        "Decision Tree",
        DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20]},
        X_train_scaled, y_train
    )

    # Random Forest: tune number of trees and depth
    rf_best = tune_model(
        "Random Forest",
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
        {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8, None]},
        X_train_scaled, y_train
    )

    models = {
        "Logistic Regression": lr_best,
        "Decision Tree": dt_best,
        "Random Forest": rf_best,
    }

    results = []
    for name, model in models.items():
        result = evaluate_model(name, model, X_test_scaled, y_test)
        results.append(result)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (on held-out test set)")
    print("=" * 60)
    print(f"{'Model':<22}{'Precision':<12}{'Recall':<10}{'F1':<10}{'ROC-AUC':<10}")
    for r in results:
        print(f"{r['name']:<22}{r['precision']:<12.3f}{r['recall']:<10.3f}{r['f1']:<10.3f}{r['auc']:<10.3f}")

    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves -- Credit Scoring Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    print("\nSaved roc_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad", "Good"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(r["name"])
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    print("Saved confusion_matrices.png")

    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 most important features (Random Forest):")
    print(importances.head(10))

    plt.figure(figsize=(8, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature_importance.png")

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model by ROC-AUC: {best['name']} ({best['auc']:.3f})")


if __name__ == "__main__":
    main()
