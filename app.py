from flask import Flask, render_template, request
import os
import traceback

from load_data import get_data_summary
from streamstay_eda import run_eda
from preprocessing import preprocess_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import run_tree_models
from kmeans_clustering import run_kmeans_clustering
from hierarchical_clustering import run_hierarchical_clustering
from dbscan_clustering import run_dbscan_clustering
from best_model import run_best_model


app = Flask(__name__)


@app.context_processor
def stylesheet_version():
    """Use the stylesheet modification time so browsers receive UI updates."""
    stylesheet = os.path.join(app.static_folder, "style.css")
    return {"css_version": os.stat(stylesheet).st_mtime_ns}


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        traceback.print_exc()

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = f"Unexpected error: {e}"


    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    error = None
    eda_output = None

    try:

        eda_output = run_eda()

    except FileNotFoundError as e:

        traceback.print_exc()

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = f"Unexpected error: {e}"


    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error
    )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    preprocessing_output = None

    try:

        preprocessing_output = preprocess_data()

    except FileNotFoundError as e:

        traceback.print_exc()

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = f"Unexpected error: {e}"


    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocessing_output,
        error=error
    )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression():

    error = None
    linear_regression_output = None

    try:
        linear_regression_output = run_linear_regression()

    except FileNotFoundError as e:
        traceback.print_exc()
        error = str(e)

    except Exception as e:
        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=linear_regression_output,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression():
    error = None
    logistic_output = None

    try:
        logistic_output = run_logistic_regression()
    except Exception as e:
        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=logistic_output,
        error=error
    )


# =========================================================
# TREE BASED MODELS
# =========================================================

@app.route("/tree-based", methods=["GET", "POST"])
def tree_based():
    error = None
    result = None
    selected_algorithm = "random_forest"

    if request.method == "POST":
        selected_algorithm = request.form.get("algorithm", selected_algorithm)
        try:
            result = run_tree_models(selected_algorithm)
        except Exception as e:
            traceback.print_exc()
            error = f"Unexpected error: {e}"

    return render_template(
        "tree_based.html",
        active="tree-based",
        selected_algorithm=selected_algorithm,
        result=result,
        error=error
    )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route("/kmeans-clustering", methods=["GET", "POST"])
def kmeans_clustering():
    error = None
    results = None
    method = request.form.get("method", "silhouette")
    manual_k = request.form.get("k", "3")

    try:
        manual_k_value = int(manual_k)
    except (TypeError, ValueError):
        error = "Manual K must be an integer from 2 to 6."
    else:
        try:
            results = run_kmeans_clustering(method, manual_k_value)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            traceback.print_exc()
            error = f"Unable to complete K-Means clustering: {e}"

    return render_template(
        "kmeans_clustering.html",
        active="kmeans",
        results=results,
        error=error,
        method=method,
        manual_k=manual_k,
    )


@app.route("/hierarchical-clustering", methods=["GET", "POST"])
def hierarchical_clustering():
    error = None
    results = None
    linkage_method = request.form.get("linkage", "ward")
    selected_k = request.form.get("k")
    try:
        requested_k = int(selected_k) if selected_k else None
    except (TypeError, ValueError):
        error = "Number of clusters must be an integer from 2 to 5."
    else:
        try:
            results = run_hierarchical_clustering(linkage_method, requested_k)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            traceback.print_exc()
            error = f"Unable to complete hierarchical clustering: {e}"
    return render_template(
        "hierarchical_clustering.html",
        active="hierarchical",
        results=results,
        error=error,
        linkage_method=linkage_method,
        selected_k=selected_k,
    )


@app.route("/dbscan", methods=["GET", "POST"])
def dbscan_clustering():
    error = None
    results = None
    eps = request.form.get("eps")
    min_samples = request.form.get("min_samples", "10")
    try:
        eps_value = float(eps) if eps else None
    except (TypeError, ValueError):
        error = "EPS must be a number from 0.05 to 10."
    else:
        try:
            min_samples_value = int(min_samples)
        except (TypeError, ValueError):
            error = "Minimum samples must be an integer from 2 to 50."
        else:
            try:
                results = run_dbscan_clustering(eps_value, min_samples_value)
            except ValueError as e:
                error = str(e)
            except Exception as e:
                traceback.print_exc()
                error = f"Unable to complete DBSCAN clustering: {e}"
    return render_template(
        "dbscan_clustering.html",
        active="dbscan",
        results=results,
        error=error,
        eps=eps,
        min_samples=min_samples,
    )


@app.route("/best-model", methods=["GET", "POST"])
def best_model():
    error = None
    results = None
    if request.method == "POST":
        try:
            results = run_best_model()
        except Exception as e:
            traceback.print_exc()
            error = f"Unable to build the classifier leaderboard: {e}"
    return render_template(
        "best_model.html",
        active="best-model",
        results=results,
        error=error,
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("========================================")
    print("      STREAMSTAY FLASK APPLICATION")
    print("========================================")

    print(
        "Starting StreamStay on "
        "http://127.0.0.1:5001"
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
