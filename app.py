from flask import Flask, render_template, request
import traceback

from load_data import get_data_summary
from streamstay_eda import run_eda
from preprocessing import preprocess_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from tree_based import run_tree_models


app = Flask(__name__)


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
