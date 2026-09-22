"""Logistic churn-model baselines using StreamStay's existing preprocessing."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from preprocessing import preprocess_data


CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
RANDOM_STATE = 42
MAX_ITERATIONS = 2000


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def _confusion_plot(name, y_test, predictions, filename):
    fig, axis = plt.subplots(figsize=(5, 4), dpi=120)
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions, labels=[0, 1]),
                           display_labels=["No Churn", "Churn"]).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close(fig)


def _roc_plot(name, y_test, probabilities, filename):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    score = roc_auc_score(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(5, 4), dpi=120)
    axis.plot(false_positive_rate, true_positive_rate, color="#28c7ff", linewidth=2,
              label=f"ROC-AUC = {score:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#718096", label="Random baseline")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{name} — ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close(fig)


def _comparison_plot(rows):
    frame = pd.DataFrame(rows)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, axis = plt.subplots(figsize=(9, 5), dpi=120)
    positions = np.arange(len(frame))
    width = 0.15
    for index, metric in enumerate(metrics):
        axis.bar(positions + (index - 2) * width, frame[metric], width, label=metric)
    axis.set_xticks(positions, frame["Model"], rotation=12, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Logistic Model Comparison")
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(_chart_path("logistic_regression_model_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def run_logistic_regression():
    """Fit comparable classifiers using the one StreamStay preprocessing split."""
    processed = preprocess_data()
    X_train, X_test = processed["X_train"], processed["X_test"]
    y_train, y_test = processed["y_train"], processed["y_test"]
    models = {
        "Logistic Regression": (
            LogisticRegression(penalty=None, solver="lbfgs", max_iter=MAX_ITERATIONS,
                               class_weight="balanced", random_state=RANDOM_STATE),
            "No regularization; penalty=None, solver=lbfgs, max_iter=2000"
        ),
        "Ridge Logistic Regression": (
            LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=MAX_ITERATIONS,
                               class_weight="balanced", random_state=RANDOM_STATE),
            "L2 regularization; C=1.0, solver=lbfgs, max_iter=2000"
        ),
        "Lasso Logistic Regression": (
            LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=MAX_ITERATIONS,
                               class_weight="balanced", random_state=RANDOM_STATE),
            "L1 regularization; C=0.1, solver=liblinear, max_iter=2000"
        ),
    }
    results, rows = {}, []
    filename_bases = ["logistic_regression", "ridge_logistic", "lasso_logistic"]
    for (name, (model, configuration)), base in zip(models.items(), filename_bases):
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        _confusion_plot(name, y_test, metrics["predictions"], f"{base}_confusion_matrix.png")
        _roc_plot(name, y_test, metrics["probabilities"], f"{base}_roc_curve.png")
        metrics["configuration"] = configuration
        results[name] = metrics
        rows.append({"Model": name, "Accuracy": metrics["accuracy"], "Precision": metrics["precision"],
                     "Recall": metrics["recall"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]})
    _comparison_plot(rows)
    target_counts = {
        "no_churn": int((y_train == 0).sum() + (y_test == 0).sum()),
        "churn": int((y_train == 1).sum() + (y_test == 1).sum()),
    }
    total_count = target_counts["no_churn"] + target_counts["churn"]
    return {
        "models": results,
        "comparison_rows": rows,
        "feature_count": len(processed["feature_names"]),
        "training_shape": tuple(X_train.shape),
        "testing_shape": tuple(X_test.shape),
        "training_target_shape": tuple(y_train.shape),
        "testing_target_shape": tuple(y_test.shape),
        "target_counts": target_counts,
        "no_churn_percentage": float(target_counts["no_churn"] / total_count * 100),
        "churn_percentage": float(target_counts["churn"] / total_count * 100),
        "threshold": 0.5,
        "class_weight": "balanced",
    }
