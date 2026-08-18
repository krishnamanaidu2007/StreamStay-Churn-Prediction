from flask import Flask, render_template
import traceback

from load_data import get_data_summary
from streamstay_eda import run_eda


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