import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from load_data import load_data


# =========================================================
# SETTINGS
# =========================================================

sns.set_theme(style="whitegrid")

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


# =========================================================
# CHART PATH
# =========================================================

def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


# =========================================================
# SAVE CHART
# =========================================================

def _save(filename: str):
    plt.tight_layout()

    plt.savefig(
        _chart_path(filename),
        bbox_inches="tight"
    )

    plt.close("all")


# =========================================================
# MAIN EDA FUNCTION
# =========================================================

def run_eda() -> dict:

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    data = load_data()

    charts = []


    # =====================================================
    # 2. BASIC INFO / STRUCTURE
    # =====================================================

    # Data types
    data.dtypes

    # Numeric description
    data.describe()

    # Categorical description
    try:
        data.describe(include=["object"])
    except ValueError:
        pass


    # =====================================================
    # 3. MISSING VALUES
    # =====================================================

    missing = data.isnull().sum()

    missing_pct = (
        missing / len(data)
    ) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_pct"] > 0
    ].sort_values(
        by="missing_count",
        ascending=False
    )


    # Missing values chart
    if not missing_df.empty:

        plt.figure(
            figsize=(12, 6),
            dpi=100
        )

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel(
            "Percentage of Missing Values"
        )

        plt.xlabel("Column")

        plt.title(
            "Missing Values by Column"
        )

        _save("missing_values.png")

        charts.append(
            "missing_values.png"
        )


    # =====================================================
    # 4. DUPLICATE ROWS
    # =====================================================

    duplicate_count = int(
        data.duplicated().sum()
    )


    # =====================================================
    # 5. TARGET VARIABLE - CHURN
    # =====================================================

    target_counts = {}

    if "is_churn" in data.columns:

        target_counts = (
            data["is_churn"]
            .value_counts()
            .to_dict()
        )

        plt.figure(
            figsize=(8, 5),
            dpi=125
        )

        sns.countplot(
            x="is_churn",
            data=data
        )

        plt.xlabel(
            "Churn Status (0 = No Churn, 1 = Churn)"
        )

        plt.ylabel("Count")

        plt.title(
            "Churn Distribution"
        )

        _save(
            "target_distribution.png"
        )

        charts.append(
            "target_distribution.png"
        )


    # =====================================================
    # 6. NUMERIC FEATURE DISTRIBUTIONS
    # =====================================================

    numeric_features = [
        "age",
        "transaction_count",
        "avg_plan_days",
        "avg_plan_price",
        "avg_amount_paid",
        "auto_renew_rate",
        "cancel_rate"
    ]

    numeric_features = [
        col
        for col in numeric_features
        if col in data.columns
    ]


    for column in numeric_features:

        plt.figure(
            figsize=(8, 5),
            dpi=100
        )

        sns.histplot(
            data[column].dropna(),
            kde=True
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        filename = (
            f"distribution_{column}.png"
        )

        _save(filename)

        charts.append(filename)


    # =====================================================
    # 7. OUTLIER DETECTION
    # =====================================================

    for column in numeric_features:

        plt.figure(
            figsize=(8, 4),
            dpi=100
        )

        sns.boxplot(
            x=data[column].dropna()
        )

        plt.title(
            f"Outlier Detection - {column}"
        )

        plt.xlabel(column)

        filename = (
            f"outliers_{column}.png"
        )

        _save(filename)

        charts.append(filename)


    # =====================================================
    # 8. CORRELATION ANALYSIS
    # =====================================================

    correlation_columns = [
        "is_churn"
    ] + numeric_features

    correlation_columns = list(
        dict.fromkeys(correlation_columns)
    )

    correlation_df = data[
        correlation_columns
    ]

    correlation_matrix = (
        correlation_df.corr(
            numeric_only=True
        )
    )

    plt.figure(
        figsize=(10, 8),
        dpi=100
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Matrix - StreamStay"
    )

    _save(
        "correlation_matrix.png"
    )

    charts.append(
        "correlation_matrix.png"
    )


    # =====================================================
    # 9. RELATIONSHIP PLOTS
    # =====================================================

    relationship_features = [
        "age",
        "transaction_count",
        "avg_plan_price",
        "avg_amount_paid"
    ]

    relationship_features = [
        col
        for col in relationship_features
        if col in data.columns
    ]


    for column in relationship_features:

        plt.figure(
            figsize=(8, 5),
            dpi=100
        )

        sns.boxplot(
            x="is_churn",
            y=column,
            data=data
        )

        plt.title(
            f"{column} vs Churn"
        )

        plt.xlabel(
            "Churn Status (0 = No Churn, 1 = Churn)"
        )

        plt.ylabel(column)

        filename = (
            f"{column}_vs_churn.png"
        )

        _save(filename)

        charts.append(filename)


    # =====================================================
    # 10. CATEGORICAL FEATURE COUNTS
    # =====================================================

    categorical_features = [
        "gender",
        "city",
        "registered_via"
    ]

    categorical_features = [
        col
        for col in categorical_features
        if col in data.columns
    ]


    for column in categorical_features:

        plot_data = (
            data[column]
            .fillna("Missing")
            .astype(str)
        )

        counts = (
            plot_data
            .value_counts()
            .head(20)
        )

        plt.figure(
            figsize=(10, 6),
            dpi=100
        )

        sns.barplot(
            x=counts.index,
            y=counts.values
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        filename = (
            f"categorical_{column}.png"
        )

        _save(filename)

        charts.append(filename)


    # =====================================================
    # 11. GENDER VS CHURN
    # =====================================================

    if "gender" in data.columns:

        gender_data = data.copy()

        gender_data["gender"] = (
            gender_data["gender"]
            .fillna("Missing")
            .astype(str)
        )

        plt.figure(
            figsize=(8, 5),
            dpi=100
        )

        sns.barplot(
            x="gender",
            y="is_churn",
            data=gender_data
        )

        plt.title(
            "Gender vs Churn"
        )

        plt.xlabel("Gender")
        plt.ylabel("Churn Rate")

        _save(
            "gender_vs_churn.png"
        )

        charts.append(
            "gender_vs_churn.png"
        )


    # =====================================================
    # 12. REGISTRATION METHOD VS CHURN
    # =====================================================

    if "registered_via" in data.columns:

        registration_data = data.copy()

        registration_data["registered_via"] = (
            registration_data["registered_via"]
            .fillna("Missing")
            .astype(str)
        )

        registration_churn = (
            registration_data
            .groupby("registered_via")["is_churn"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        plt.figure(
            figsize=(10, 6),
            dpi=100
        )

        sns.barplot(
            x=registration_churn.index,
            y=registration_churn.values
        )

        plt.title(
            "Registration Method vs Churn"
        )

        plt.xlabel(
            "Registration Method"
        )

        plt.ylabel(
            "Churn Rate (%)"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        _save(
            "registration_method_vs_churn.png"
        )

        charts.append(
            "registration_method_vs_churn.png"
        )


    # =====================================================
    # 13. REGISTRATION YEAR VS CHURN
    # =====================================================

    if "registration_year" in data.columns:

        year_data = data.dropna(
            subset=["registration_year"]
        ).copy()

        year_churn = (
            year_data
            .groupby("registration_year")["is_churn"]
            .mean()
            * 100
        )

        plt.figure(
            figsize=(10, 6),
            dpi=100
        )

        sns.lineplot(
            x=year_churn.index,
            y=year_churn.values,
            marker="o"
        )

        plt.title(
            "Registration Year vs Churn"
        )

        plt.xlabel(
            "Registration Year"
        )

        plt.ylabel(
            "Churn Rate (%)"
        )

        _save(
            "registration_year_vs_churn.png"
        )

        charts.append(
            "registration_year_vs_churn.png"
        )


    # =====================================================
    # 14. PAYMENT & SUBSCRIPTION ANALYSIS
    # =====================================================

    payment_features = [
        "avg_plan_days",
        "avg_plan_price",
        "avg_amount_paid",
        "auto_renew_rate",
        "cancel_rate"
    ]

    payment_features = [
        col
        for col in payment_features
        if col in data.columns
    ]


    if payment_features and "is_churn" in data.columns:

        payment_summary = (
            data
            .groupby("is_churn")[payment_features]
            .mean()
        )


        for column in payment_features:

            plt.figure(
                figsize=(8, 5),
                dpi=100
            )

            sns.boxplot(
                x="is_churn",
                y=column,
                data=data
            )

            plt.title(
                f"{column} by Churn Status"
            )

            plt.xlabel(
                "Churn Status (0 = No Churn, 1 = Churn)"
            )

            plt.ylabel(column)

            filename = (
                f"payment_{column}_vs_churn.png"
            )

            _save(filename)

            charts.append(filename)


    # =====================================================
    # 15. PAIRPLOT
    # =====================================================

    pairplot_features = [
        "age",
        "transaction_count",
        "avg_plan_days",
        "avg_plan_price",
        "is_churn"
    ]

    pairplot_features = [
        col
        for col in pairplot_features
        if col in data.columns
    ]

    pairplot_data = data[
        pairplot_features
    ].dropna()


    # Limit rows for faster plotting
    if len(pairplot_data) > 5000:

        pairplot_data = (
            pairplot_data
            .sample(
                n=5000,
                random_state=42
            )
        )


    if (
        len(pairplot_data) > 0
        and "is_churn" in pairplot_data.columns
    ):

        pair_plot = sns.pairplot(
            pairplot_data,
            hue="is_churn",
            diag_kind="hist"
        )

        pair_plot.fig.suptitle(
            "StreamStay Feature Pairplot",
            y=1.02
        )

        pair_plot.savefig(
            _chart_path("pairplot.png"),
            bbox_inches="tight"
        )

        plt.close("all")

        charts.append(
            "pairplot.png"
        )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    numeric_columns = [
        col
        for col in data.select_dtypes(
            include=[np.number]
        ).columns
        if col not in [
            "is_churn",
            "city",
            "registered_via",
            "bd",
            "registration_init_time",
            "registration_year",
            "registration_month"
        ]
    ]


    categorical_columns = list(
        data.select_dtypes(
            include=["object", "category"]
        ).columns
    )


    # =====================================================
    # RETURN RESULTS TO FLASK
    # =====================================================

    return {
        "n_rows": len(data),

        "n_cols": len(data.columns),

        "duplicate_count": duplicate_count,

        "missing": {
            col: int(cnt)
            for col, cnt in missing.items()
            if cnt > 0
        },

        "target_counts": {
            str(k): int(v)
            for k, v in target_counts.items()
        },

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "charts": charts
    }


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    results = run_eda()

    print("EDA completed successfully.")
    print("Rows:", results["n_rows"])
    print("Columns:", results["n_cols"])
    print("Duplicate rows:", results["duplicate_count"])
    print("Charts generated:", len(results["charts"]))