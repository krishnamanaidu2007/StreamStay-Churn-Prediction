import os
import pandas as pd


# =========================================================
# SAFE PRINT
# =========================================================

def safe_print(*args, **kwargs):
    """
    Prevent console-print errors from crashing the Flask app.
    """
    try:
        print(*args, **kwargs)
    except (OSError, ValueError):
        pass


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
MEMBERS_PATH = os.path.join(DATA_DIR, "members_v3.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions_v2.csv")

# Our final 50K working dataset
FINAL_50K_PATH = os.path.join(
    DATA_DIR,
    "streamstay_50k.csv"
)


# =========================================================
# CREATE FULL COMBINED DATASET
# =========================================================

def build_full_dataset():

    safe_print("\n========== LOADING RAW DATASETS ==========")

    # Check files
    for path in [
        TRAIN_PATH,
        MEMBERS_PATH,
        TRANSACTIONS_PATH
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    # -----------------------------------------------------
    # LOAD DATASETS
    # -----------------------------------------------------

    safe_print("Loading train dataset...")
    train = pd.read_csv(TRAIN_PATH)

    safe_print("Loading members dataset...")
    members = pd.read_csv(MEMBERS_PATH)

    safe_print("Loading transactions dataset...")
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    safe_print("Raw datasets loaded successfully.")

    # -----------------------------------------------------
    # MEMBER DATA CLEANING
    # -----------------------------------------------------

    safe_print("\n========== PROCESSING MEMBER DATA ==========")

    members["bd"] = pd.to_numeric(
        members["bd"],
        errors="coerce"
    )

    # Create clean age
    members["age"] = members["bd"].where(
        (members["bd"] >= 10) &
        (members["bd"] <= 100)
    )

    # Convert registration date
    members["registration_date"] = pd.to_datetime(
        members["registration_init_time"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # Date features
    members["registration_year"] = (
        members["registration_date"].dt.year
    )

    members["registration_month"] = (
        members["registration_date"].dt.month
    )

    # -----------------------------------------------------
    # TRANSACTION FEATURE ENGINEERING
    # -----------------------------------------------------

    safe_print(
        "\n========== CREATING TRANSACTION FEATURES =========="
    )

    transaction_features = transactions.groupby("msno").agg(

        transaction_count=(
            "msno",
            "count"
        ),

        avg_plan_days=(
            "payment_plan_days",
            "mean"
        ),

        avg_plan_price=(
            "plan_list_price",
            "mean"
        ),

        avg_amount_paid=(
            "actual_amount_paid",
            "mean"
        ),

        auto_renew_rate=(
            "is_auto_renew",
            "mean"
        ),

        cancel_rate=(
            "is_cancel",
            "mean"
        )

    ).reset_index()

    safe_print(
        "Transaction features created:",
        transaction_features.shape
    )

    # -----------------------------------------------------
    # MERGE TRAIN + MEMBERS
    # -----------------------------------------------------

    safe_print("\n========== MERGING DATASETS ==========")

    df = train.merge(
        members,
        on="msno",
        how="left"
    )

    # -----------------------------------------------------
    # ADD TRANSACTION FEATURES
    # -----------------------------------------------------

    df = df.merge(
        transaction_features,
        on="msno",
        how="left"
    )

    safe_print(
        "Full combined dataset shape:",
        df.shape
    )

    return df


# =========================================================
# CREATE 50K WORKING DATASET
# =========================================================

def create_50k_dataset():

    safe_print("\n==============================================")
    safe_print("     CREATING STREAMSTAY 50K DATASET")
    safe_print("==============================================")

    # Build complete dataset
    df = build_full_dataset()

    safe_print("\nFull dataset:")
    safe_print("Rows:", len(df))
    safe_print("Columns:", len(df.columns))

    # -----------------------------------------------------
    # CHECK TARGET
    # -----------------------------------------------------

    if "is_churn" not in df.columns:
        raise ValueError(
            "Target column 'is_churn' not found."
        )

    # -----------------------------------------------------
    # CALCULATE STRATIFIED SAMPLE SIZE
    # -----------------------------------------------------

    TARGET_SIZE = 50000

    churn_count = int(
        (df["is_churn"] == 1).sum()
    )

    non_churn_count = int(
        (df["is_churn"] == 0).sum()
    )

    total_count = churn_count + non_churn_count

    if total_count == 0:
        raise ValueError(
            "No valid target values found in is_churn."
        )

    # Preserve original churn percentage
    churn_ratio = churn_count / total_count

    churn_sample_size = round(
        TARGET_SIZE * churn_ratio
    )

    non_churn_sample_size = (
        TARGET_SIZE - churn_sample_size
    )

    safe_print("\n========== ORIGINAL TARGET ==========")

    safe_print(
        "Non-churn:",
        non_churn_count
    )

    safe_print(
        "Churn:",
        churn_count
    )

    safe_print(
        "Churn percentage:",
        round(churn_ratio * 100, 2),
        "%"
    )

    safe_print("\n========== 50K SAMPLE ==========")

    safe_print(
        "Non-churn sample:",
        non_churn_sample_size
    )

    safe_print(
        "Churn sample:",
        churn_sample_size
    )

    # -----------------------------------------------------
    # STRATIFIED RANDOM SAMPLING
    # -----------------------------------------------------

    churn_df = df[
        df["is_churn"] == 1
    ].sample(
        n=churn_sample_size,
        random_state=42
    )

    non_churn_df = df[
        df["is_churn"] == 0
    ].sample(
        n=non_churn_sample_size,
        random_state=42
    )

    # Combine both groups
    sample_df = pd.concat(
        [
            churn_df,
            non_churn_df
        ],
        ignore_index=True
    )

    # Shuffle the final dataset
    sample_df = sample_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # MOVE TARGET TO LAST COLUMN
    # -----------------------------------------------------

    columns = [
        col
        for col in sample_df.columns
        if col != "is_churn"
    ]

    columns.append("is_churn")

    sample_df = sample_df[
        columns
    ]

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    sample_df.to_csv(
        FINAL_50K_PATH,
        index=False
    )

    safe_print("\n==============================================")
    safe_print("       50K DATASET CREATED SUCCESSFULLY")
    safe_print("==============================================")

    safe_print(
        "Saved to:",
        FINAL_50K_PATH
    )

    safe_print(
        "Rows:",
        len(sample_df)
    )

    safe_print(
        "Columns:",
        len(sample_df.columns)
    )

    safe_print(
        "\nColumns:"
    )

    safe_print(
        list(sample_df.columns)
    )

    safe_print(
        "\nTarget distribution:"
    )

    safe_print(
        sample_df["is_churn"].value_counts()
    )

    safe_print(
        "\nTarget percentage:"
    )

    safe_print(
        sample_df["is_churn"]
        .value_counts(normalize=True)
        .mul(100)
    )

    safe_print(
        "\nFirst 5 rows:"
    )

    safe_print(
        sample_df.head()
    )

    return sample_df


# =========================================================
# LOAD DATA FOR OUR PROJECT
# =========================================================

def load_data():

    # If 50K dataset already exists,
    # use it directly.
    if os.path.exists(FINAL_50K_PATH):

        safe_print(
            "\n========== LOADING 50K DATASET =========="
        )

        df = pd.read_csv(
            FINAL_50K_PATH
        )

        safe_print(
            "Rows:",
            len(df)
        )

        safe_print(
            "Columns:",
            len(df.columns)
        )

        return df

    # Otherwise create it
    safe_print(
        "\n50K dataset not found."
    )

    safe_print(
        "Creating it from the original datasets..."
    )

    return create_50k_dataset()


# =========================================================
# DATA SUMMARY
# =========================================================

def get_data_summary() -> dict:

    df = load_data()

    summary = {

        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(
            df.columns
        ),

        "dtypes": {
            col: str(
                df[col].dtype
            )
            for col in df.columns
        },

        "missing_counts": {
            col: int(
                df[col].isnull().sum()
            )
            for col in df.columns
        },

        "preview": (
            df.head(10)
            .to_dict("records")
        ),

    }

    return summary


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    safe_print(
        "\n=============================================="
    )

    safe_print(
        "       STREAMSTAY 50K DATASET CREATOR"
    )

    safe_print(
        "=============================================="
    )

    # Create the dataset
    df = create_50k_dataset()

    safe_print(
        "\n========== FINAL VERIFICATION =========="
    )

    safe_print(
        "Rows:",
        df.shape[0]
    )

    safe_print(
        "Columns:",
        df.shape[1]
    )

    safe_print(
        "\nFinal columns:"
    )

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        safe_print(
            i,
            "->",
            column
        )

    safe_print(
        "\nTarget column:",
        df.columns[-1]
    )

    safe_print(
        "\nTarget distribution:"
    )

    safe_print(
        df["is_churn"].value_counts()
    )

    safe_print(
        "\n========== DONE =========="
    )