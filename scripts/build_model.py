"""
Train a member-vs-casual rider classifier and write the results behind the
site's "Predict" section (docs/data/model.json + .js). Loading, cleaning and
feature engineering come from scripts/pipeline.py, shared with the notebook,
so the notebook, site charts, and this model all agree on the same trips.

The features are deliberately the same signals the rest of the site already
tells a story about: duration/distance ("who's riding" -- casual trips run
longer), hour/weekend/rush-hour/holiday ("when they ride" -- commute vs.
leisure), and station popularity ("where they ride" -- Manhattan
concentration), plus a couple of behavior signals (round trip, bike type).
The model asks whether those threads, taken together, can predict which type
of rider took a given trip.

Both models use class_weight="balanced". With an ~89/11 class split, an
unweighted model mostly learns to say "member" and finds few casual riders;
balancing trades raw accuracy for casual-class recall. The naive "always
guess member" accuracy is reported alongside so raw accuracy is never read as
skill on its own, and ROC-AUC plus casual-class precision/recall/F1 carry the
evaluation.

Usage: python scripts/build_model.py [YYYYMM]
       python scripts/build_model.py 202401   # defaults to this if omitted
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import DEFAULT_MONTH, FEATURE_LABELS, FEATURES, engineer_features, load_clean_month, month_label

REPO = Path(__file__).resolve().parent.parent


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df["is_member"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # With an ~89/11 class split, always guessing "member" already scores
    # ~89% accuracy while identifying zero casual riders -- report that
    # explicitly so the models' accuracy is never read as skill on its own.
    majority_label = int(y_train.mode()[0])
    majority_preds = np.full(len(y_test), majority_label)
    majority_accuracy = float((majority_preds == y_test).mean())

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    logistic.fit(X_train, y_train)
    logistic_proba = logistic.predict_proba(X_test)[:, 1]
    logistic_preds = logistic.predict(X_test)

    forest = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    forest.fit(X_train, y_train)
    forest_proba = forest.predict_proba(X_test)[:, 1]
    forest_preds = forest.predict(X_test)

    def model_summary(preds, proba) -> dict:
        accuracy = float((preds == y_test).mean())
        auc = float(roc_auc_score(y_test, proba))
        # casual = label 0, the minority class the majority baseline misses entirely
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, labels=[0], zero_division=0
        )
        fpr, tpr, _ = roc_curve(y_test, proba)
        idx = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
        return {
            "accuracy": accuracy,
            "beats_baseline": accuracy > majority_accuracy,
            "roc_auc": auc,
            "casual_precision": float(precision[0]),
            "casual_recall": float(recall[0]),
            "casual_f1": float(f1[0]),
            "roc_curve": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
        }

    cm = confusion_matrix(y_test, forest_preds, labels=[0, 1]).tolist()

    importances = sorted(
        (
            {"feature": f, "label": FEATURE_LABELS[f], "importance": float(imp)}
            for f, imp in zip(FEATURES, forest.feature_importances_)
        ),
        key=lambda d: -d["importance"],
    )

    return {
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "majority_baseline_accuracy": majority_accuracy,
        "logistic_regression": model_summary(logistic_preds, logistic_proba),
        "random_forest": model_summary(forest_preds, forest_proba),
        "confusion_matrix": {
            "labels": ["casual", "member"],
            "matrix": cm,
        },
        "feature_importances": importances,
    }


def write_model_data(results: dict, label: str) -> None:
    out = {
        "title": "Predicting Member vs. Casual Riders",
        "subtitle": f"{label}, random forest vs. logistic regression vs. naive baseline",
        **results,
    }
    data_dir = REPO / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "model.json").write_text(json.dumps(out))
    (data_dir / "model.js").write_text("window.MODEL_DATA = " + json.dumps(out) + ";\n")
    print("Wrote docs/data/model.json and model.js", flush=True)
    summary = {
        "sample_size": results["sample_size"],
        "majority_baseline_accuracy": results["majority_baseline_accuracy"],
        "logistic_regression": {k: v for k, v in results["logistic_regression"].items() if k != "roc_curve"},
        "random_forest": {k: v for k, v in results["random_forest"].items() if k != "roc_curve"},
        "confusion_matrix": results["confusion_matrix"],
        "top_features": results["feature_importances"][:3],
    }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    yyyymm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MONTH
    label = month_label(yyyymm)
    df = engineer_features(load_clean_month(yyyymm), yyyymm)

    results = train_and_evaluate(df)
    write_model_data(results, label=label)


if __name__ == "__main__":
    main()
