"""Stage 7 tree-based churn classifiers using StreamStay's approved preprocessing."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import preprocess_data


CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
RANDOM_STATE = 42
TOP_FEATURE_COUNT = 15


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _model_definitions():
    return {
        "decision_tree": {"name": "Decision Tree", "configuration": "max_depth=12; min_samples_leaf=10; class_weight=balanced; random_state=42", "factory": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE)},
        "random_forest": {"name": "Random Forest", "configuration": "n_estimators=100; max_depth=15; min_samples_leaf=5; class_weight=balanced; random_state=42; n_jobs=-1", "factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)},
        "extra_trees": {"name": "Extra Trees", "configuration": "n_estimators=100; max_depth=15; min_samples_leaf=5; class_weight=balanced; random_state=42; n_jobs=-1", "factory": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)},
        "gradient_boosting": {"name": "Gradient Boosting", "configuration": "n_estimators=100; learning_rate=0.10; max_depth=3; random_state=42", "factory": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.10, max_depth=3, random_state=RANDOM_STATE)},
        "adaboost": {"name": "AdaBoost", "configuration": "n_estimators=100; learning_rate=0.50; random_state=42", "factory": lambda: AdaBoostClassifier(n_estimators=100, learning_rate=0.50, random_state=RANDOM_STATE)},
        "xgboost": {"name": "XGBoost", "configuration": "n_estimators=100; learning_rate=0.10; max_depth=6; subsample=0.8; colsample_bytree=0.8; eval_metric=logloss; random_state=42; n_jobs=-1", "factory": lambda: XGBClassifier(n_estimators=100, learning_rate=0.10, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)},
        "lightgbm": {"name": "LightGBM", "configuration": "n_estimators=100; learning_rate=0.10; max_depth=-1; num_leaves=31; random_state=42; n_jobs=-1", "factory": lambda: LGBMClassifier(n_estimators=100, learning_rate=0.10, max_depth=-1, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)},
    }


def _evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {"accuracy": float(accuracy_score(y_test, predictions)), "precision": float(precision_score(y_test, predictions, zero_division=0)), "recall": float(recall_score(y_test, predictions, zero_division=0)), "f1": float(f1_score(y_test, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, probabilities)), "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(), "predictions": predictions, "probabilities": probabilities}


def _plot_confusion(name, y_test, predictions):
    fig, axis = plt.subplots(figsize=(6, 4.8), dpi=120)
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions, labels=[0, 1]), display_labels=["No Churn", "Churn"]).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(_chart_path("tree_based_confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_roc(name, y_test, probabilities):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    score = roc_auc_score(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(6, 4.8), dpi=120)
    axis.plot(false_positive_rate, true_positive_rate, color="#167eaa", linewidth=2, label=f"ROC-AUC = {score:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#718096", label="Random classifier")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{name} — ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(_chart_path("tree_based_roc_curve.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(rows):
    frame = pd.DataFrame(rows)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    figure, axis = plt.subplots(figsize=(12, 6), dpi=120)
    positions = np.arange(len(frame))
    width = 0.15
    for index, metric in enumerate(metrics):
        axis.bar(positions + (index - 2) * width, frame[metric], width, label=metric)
    axis.set_xticks(positions, frame["Model"], rotation=16, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Tree-Based Model Comparison")
    axis.legend(ncol=5, loc="upper center")
    figure.tight_layout()
    figure.savefig(_chart_path("tree_based_model_comparison.png"), bbox_inches="tight")
    plt.close(figure)


def _feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return [], None
    values = np.asarray(model.feature_importances_)
    pairs = sorted(zip(feature_names, values), key=lambda pair: pair[1], reverse=True)[:TOP_FEATURE_COUNT]
    figure, axis = plt.subplots(figsize=(9, 6), dpi=120)
    labels, scores = zip(*reversed(pairs))
    axis.barh(labels, scores, color="#167eaa")
    axis.set_xlabel("Feature Importance")
    axis.set_title(f"Top Feature Importances — {model_name}")
    figure.tight_layout()
    figure.savefig(_chart_path("tree_based_feature_importance.png"), bbox_inches="tight")
    plt.close(figure)
    return [{"feature": feature, "importance": float(score)} for feature, score in pairs], "tree_based_feature_importance.png"


def run_tree_models(selected_algorithm):
    """Train all required models once per explicit POST and return selected-model details."""
    definitions = _model_definitions()
    if selected_algorithm not in definitions:
        raise ValueError(f"Unknown tree-based algorithm: {selected_algorithm}")
    processed = preprocess_data()
    X_train, X_test = processed["X_train"], processed["X_test"]
    y_train, y_test = processed["y_train"], processed["y_test"]
    if X_train.shape != (40000, 61) or X_test.shape != (10000, 61):
        raise ValueError(f"Unexpected approved preprocessing shapes: {X_train.shape}, {X_test.shape}")
    all_results, comparison_rows, selected_result = {}, [], None
    for key, definition in definitions.items():
        model = definition["factory"]()
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        entry = {"algorithm": definition["name"], "configuration": definition["configuration"], "model": model, **metrics}
        all_results[key] = entry
        comparison_rows.append({"Model": definition["name"], "Accuracy": metrics["accuracy"], "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]})
        if key == selected_algorithm:
            selected_result = entry
    _plot_comparison(comparison_rows)
    _plot_confusion(selected_result["algorithm"], y_test, selected_result["predictions"])
    _plot_roc(selected_result["algorithm"], y_test, selected_result["probabilities"])
    importance, importance_filename = _feature_importance(selected_result["model"], processed["feature_names"], selected_result["algorithm"])
    selected_result.update({"feature_importance": importance, "feature_importance_plot": importance_filename, "confusion_matrix_plot": "tree_based_confusion_matrix.png", "roc_curve_plot": "tree_based_roc_curve.png"})
    no_churn = int((y_train == 0).sum() + (y_test == 0).sum())
    churn = int((y_train == 1).sum() + (y_test == 1).sum())
    return {"selected": selected_result, "comparison_rows": comparison_rows, "training_shape": tuple(X_train.shape), "testing_shape": tuple(X_test.shape), "feature_count": len(processed["feature_names"]), "target_counts": {"no_churn": no_churn, "churn": churn}, "no_churn_percentage": float(no_churn / (no_churn + churn) * 100), "churn_percentage": float(churn / (no_churn + churn) * 100), "algorithm_count": len(definitions), "comparison_plot": "tree_based_model_comparison.png"}
