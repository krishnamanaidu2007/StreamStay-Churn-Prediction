"""Linear, Ridge, and Lasso regression using StreamStay's processed split."""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, r2_score, recall_score, roc_auc_score

from preprocessing import preprocess_data


RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.001
CLASSIFICATION_THRESHOLD = 0.5
TOP_COEFFICIENT_COUNT = 15
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _validate_processed_data(X_train, X_test):
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Preprocessing returned an empty training or test matrix.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Processed training and test data have different feature counts.")


def evaluate_model(y_true, predictions, threshold=CLASSIFICATION_THRESHOLD):
    """Calculate regression and threshold-based classification metrics."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced non-finite predictions.")

    predicted_classes = (predictions >= threshold).astype(int)
    metrics = {
        "mse": float(mean_squared_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predicted_classes)),
        "precision": float(precision_score(y_true, predicted_classes, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_classes, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_classes, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted_classes, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            predicted_classes,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0
        ),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, predictions))
        if len(np.unique(y_true)) == 2 else None
    )
    return metrics


def _coefficient_summary(model, feature_names, model_name):
    coefficients = np.asarray(model.coef_).ravel()
    if len(coefficients) != len(feature_names):
        raise ValueError("Coefficient count does not match preprocessed feature names.")
    frame = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    return {
        "model": model_name,
        "coefficient_count": int(len(coefficients)),
        "zero_coefficients": int(np.isclose(coefficients, 0.0, atol=1e-8).sum()),
        "mean_absolute_coefficient": float(np.abs(coefficients).mean()),
        "top_coefficients": frame.sort_values("absolute_coefficient", ascending=False)
        .head(TOP_COEFFICIENT_COUNT).to_dict("records"),
    }


def _plot_actual_vs_predicted(model_name, y_test, predictions, filename):
    plt.figure(figsize=(8, 5), dpi=120)
    plt.scatter(y_test, predictions, alpha=0.18, color="#167eaa", edgecolors="none")
    bounds = [min(0, float(np.min(predictions))), max(1, float(np.max(predictions)))]
    plt.plot(bounds, bounds, "--", color="#ef626c", label="Ideal prediction")
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("Actual Churn (0 = No, 1 = Yes)")
    plt.ylabel("Continuous Regression Prediction")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close()


def _plot_model_comparison(comparison):
    names, positions = comparison["Model"].tolist(), np.arange(len(comparison))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    axes[0].bar(positions, comparison["RMSE"], color=["#167eaa", "#28c7ff", "#68b984"])
    axes[0].set(title="Regression Error Comparison", ylabel="RMSE (lower is better)")
    axes[0].set_xticks(positions, names, rotation=12, ha="right")
    for column, color in [("Accuracy", "#167eaa"), ("F1", "#ef9f3e"), ("ROC-AUC", "#68b984")]:
        axes[1].plot(names, comparison[column].fillna(0), marker="o", linewidth=2,
                     label=column, color=color)
    axes[1].set(title="Threshold-Based Classification Metrics", ylabel="Score", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(_chart_path("linear_regression_model_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_coefficients(summaries):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=120)
    for axis, (name, summary), color in zip(
        axes, summaries.items(), ["#167eaa", "#28c7ff", "#68b984"]
    ):
        top = pd.DataFrame(summary["top_coefficients"]).sort_values("coefficient")
        axis.barh(top["feature"], top["coefficient"], color=color)
        axis.axvline(0, color="#16283d", linewidth=1)
        axis.set_title(f"Top coefficients — {name}")
        axis.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(_chart_path("linear_regression_top_coefficients.png"), bbox_inches="tight")
    plt.close(fig)


def generate_linear_regression_charts(results, y_test):
    charts = [
        ("Linear Regression", "linear_regression_actual_vs_predicted.png"),
        ("Ridge Regression", "ridge_regression_actual_vs_predicted.png"),
        ("Lasso Regression", "lasso_regression_actual_vs_predicted.png"),
    ]
    for name, filename in charts:
        _plot_actual_vs_predicted(name, y_test, results[name]["predictions"], filename)
    _plot_model_comparison(results["comparison"])
    _plot_coefficients(results["coefficient_summaries"])
    return [filename for _, filename in charts] + [
        "linear_regression_model_comparison.png", "linear_regression_top_coefficients.png"
    ]


def run_linear_regression():
    """Train three regressors from exactly one existing preprocessing result."""
    processed = preprocess_data()
    X_train, X_test = processed["X_train"], processed["X_test"]
    y_train, y_test = processed["y_train"], processed["y_test"]
    feature_names = processed["feature_names"]
    _validate_processed_data(X_train, X_test)

    models = {
        "Linear Regression": (LinearRegression(), "No", "None", None),
        "Ridge Regression": (Ridge(alpha=RIDGE_ALPHA), "Yes", "L2", RIDGE_ALPHA),
        "Lasso Regression": (Lasso(alpha=LASSO_ALPHA, max_iter=10000), "Yes", "L1", LASSO_ALPHA),
    }
    results, summaries, rows = {}, {}, []
    for name, (model, regularization, regularization_type, alpha) in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test, predictions)
        results[name] = {"model": model, "predictions": predictions,
                         "regularization": regularization, "regularization_type": regularization_type,
                         "alpha": alpha, "metrics": metrics}
        summaries[name] = _coefficient_summary(model, feature_names, name)
        rows.append({"Model": name, "Regularization": regularization,
                     "Regularization Type": regularization_type, "Alpha": "—" if alpha is None else alpha,
                     "MSE": metrics["mse"], "RMSE": metrics["rmse"], "MAE": metrics["mae"],
                     "R²": metrics["r2"], "Accuracy": metrics["accuracy"],
                     "Precision": metrics["precision"], "Recall": metrics["recall"],
                     "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]})
    results["comparison"] = pd.DataFrame(rows)
    results["comparison_rows"] = results["comparison"].to_dict("records")
    results["coefficient_summaries"] = summaries
    results["threshold"] = CLASSIFICATION_THRESHOLD
    results["processed_feature_count"] = len(feature_names)
    results["training_shape"] = tuple(X_train.shape)
    results["testing_shape"] = tuple(X_test.shape)
    results["training_target_shape"] = tuple(y_train.shape)
    results["testing_target_shape"] = tuple(y_test.shape)
    results["target"] = "is_churn"
    results["charts"] = generate_linear_regression_charts(results, y_test)
    return results


if __name__ == "__main__":
    output = run_linear_regression()
    print("Linear regression stage completed successfully.")
    print(output["comparison"].to_string(index=False))
