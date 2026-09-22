from load_data import load_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# =========================================================
# STREAMSTAY DATA PREPROCESSING
# =========================================================

def preprocess_data():

    print("=" * 80)
    print("STREAMSTAY DATA PREPROCESSING")
    print("=" * 80)


    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    df = load_data()

    print("\n1. DATA LOADED")
    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


    # =====================================================
    # 2. CHECK AND REMOVE DUPLICATES
    # =====================================================

    print("\n" + "=" * 80)
    print("2. DUPLICATE ROWS")
    print("=" * 80)

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows before removal:", duplicate_count)

    df = df.drop_duplicates()

    print("Duplicate rows after removal:",
          df.duplicated().sum())


    # =====================================================
    # 3. DEFINE TARGET
    # =====================================================

    print("\n" + "=" * 80)
    print("3. TARGET VARIABLE")
    print("=" * 80)

    target = "is_churn"

    print("Target column:", target)

    print("\nTarget distribution:")
    print(df[target].value_counts())


    # =====================================================
    # 4. DROP UNNECESSARY COLUMNS
    # =====================================================

    print("\n" + "=" * 80)
    print("4. REMOVING UNNECESSARY COLUMNS")
    print("=" * 80)

    columns_to_drop = [
        "msno",
        "bd",
        "registration_init_time",
        "registration_date"
    ]

    existing_columns = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df = df.drop(columns=existing_columns)

    print("Dropped columns:")
    print(existing_columns)


    # =====================================================
    # 5. SEPARATE FEATURES AND TARGET
    # =====================================================

    print("\n" + "=" * 80)
    print("5. SEPARATING FEATURES AND TARGET")
    print("=" * 80)

    X = df.drop(columns=[target])

    y = df[target]

    print("Feature shape:", X.shape)

    print("Target shape:", y.shape)


    # =====================================================
    # 6. DEFINE NUMERICAL AND CATEGORICAL FEATURES
    # =====================================================

    print("\n" + "=" * 80)
    print("6. IDENTIFYING FEATURE TYPES")
    print("=" * 80)

    numerical_columns = [
        "age",
        "transaction_count",
        "avg_plan_days",
        "avg_plan_price",
        "avg_amount_paid",
        "auto_renew_rate",
        "cancel_rate"
    ]

    categorical_columns = [
        "city",
        "gender",
        "registered_via",
        "registration_year",
        "registration_month"
    ]

    # Keep only columns that actually exist
    numerical_columns = [
        col for col in numerical_columns
        if col in X.columns
    ]

    categorical_columns = [
        col for col in categorical_columns
        if col in X.columns
    ]

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)


    # =====================================================
    # 7. CHECK MISSING VALUES
    # =====================================================

    print("\n" + "=" * 80)
    print("7. MISSING VALUES BEFORE PREPROCESSING")
    print("=" * 80)

    missing_values = X.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    print(missing_values)


    # =====================================================
    # 8. TRAIN / TEST SPLIT
    # =====================================================

    print("\n" + "=" * 80)
    print("8. TRAIN TEST SPLIT")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("Training features:", X_train.shape)

    print("Testing features:", X_test.shape)

    print("Training target:", y_train.shape)

    print("Testing target:", y_test.shape)


    # =====================================================
    # 9. NUMERICAL PREPROCESSING
    # =====================================================

    print("\n" + "=" * 80)
    print("9. NUMERICAL PREPROCESSING")
    print("=" * 80)

    numerical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )

        ]

    )

    print("Numerical features:")
    print("1. Missing values -> Median")
    print("2. Feature scaling -> StandardScaler")


    # =====================================================
    # 10. CATEGORICAL PREPROCESSING
    # =====================================================

    print("\n" + "=" * 80)
    print("10. CATEGORICAL PREPROCESSING")
    print("=" * 80)

    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )

    print("Categorical features:")
    print("1. Missing values -> Most frequent value")
    print("2. Encoding -> OneHotEncoder")


    # =====================================================
    # 11. COMBINE PREPROCESSING
    # =====================================================

    print("\n" + "=" * 80)
    print("11. COMBINING PREPROCESSING")
    print("=" * 80)

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",

                numerical_transformer,

                numerical_columns
            ),

            (
                "categorical",

                categorical_transformer,

                categorical_columns
            )

        ]

    )


    # =====================================================
    # 12. FIT AND TRANSFORM TRAINING DATA
    # =====================================================

    print("\n" + "=" * 80)
    print("12. PROCESSING TRAINING DATA")
    print("=" * 80)

    X_train_processed = preprocessor.fit_transform(
        X_train
    )


    # =====================================================
    # 13. TRANSFORM TEST DATA
    # =====================================================

    print("\n" + "=" * 80)
    print("13. PROCESSING TEST DATA")
    print("=" * 80)

    X_test_processed = preprocessor.transform(
        X_test
    )


    # =====================================================
    # 14. FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nOriginal feature shape:")
    print(X.shape)

    print("\nProcessed training shape:")
    print(X_train_processed.shape)

    print("\nProcessed testing shape:")
    print(X_test_processed.shape)

    print("\nTraining target shape:")
    print(y_train.shape)

    print("\nTesting target shape:")
    print(y_test.shape)

    # Reuse names from this fitted transformer in later modelling stages.
    # This does not fit any additional preprocessing transformations.
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        feature_names = [
            f"processed_feature_{index}"
            for index in range(X_train_processed.shape[1])
        ]


    # =====================================================
    # RETURN PROCESSED DATA
    # =====================================================

    return {

        "X_train": X_train_processed,

        "X_test": X_test_processed,

        "y_train": y_train,

        "y_test": y_test,

        "preprocessor": preprocessor,

        "numerical_columns": numerical_columns,

        "categorical_columns": categorical_columns,

        "feature_names": feature_names,

        "original_feature_count": X.shape[1],

        "transformed_feature_count": X_train_processed.shape[1],

        "removed_columns": existing_columns,

        "target": target,

        "split_details": {
            "training_percentage": 80,
            "testing_percentage": 20,
            "random_state": 42,
            "stratified": True
        }

    }


# =========================================================
# RUN PREPROCESSING
# =========================================================

if __name__ == "__main__":

    processed_data = preprocess_data()
