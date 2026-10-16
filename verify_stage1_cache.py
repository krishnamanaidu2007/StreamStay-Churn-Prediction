"""Focused verification for Stage 1 process-local data/preprocessing reuse.

Run from the StreamStay directory:
    .venv/Scripts/python.exe -B verify_stage1_cache.py
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import load_data
import preprocessing


def _clear_caches() -> None:
    load_data._DATA_CACHE["fingerprint"] = None
    load_data._DATA_CACHE["data"] = None
    preprocessing._PREPROCESSING_CACHE["fingerprint"] = None
    preprocessing._PREPROCESSING_CACHE["result"] = None


def _assert_processed_values_equal(first: dict, second: dict) -> None:
    assert first["X_train"].shape == second["X_train"].shape
    assert first["X_test"].shape == second["X_test"].shape
    assert first["X_train"].shape == (40000, 61)
    assert first["X_test"].shape == (10000, 61)
    for key in ("X_train", "X_test"):
        if sparse.issparse(first[key]):
            assert (first[key] != second[key]).nnz == 0
        else:
            np.testing.assert_allclose(first[key], second[key])
    pd.testing.assert_series_equal(first["y_train"], second["y_train"])
    pd.testing.assert_series_equal(first["y_test"], second["y_test"])
    assert first["feature_names"] == second["feature_names"]


def main() -> None:
    original_path = load_data.FINAL_50K_PATH

    workspace_directory = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=workspace_directory) as temporary_directory:
        temporary_csv = Path(temporary_directory) / "streamstay_50k.csv"
        shutil.copy2(original_path, temporary_csv)
        load_data.FINAL_50K_PATH = str(temporary_csv)
        _clear_caches()

        try:
            started = time.perf_counter()
            first_data = load_data.load_data()
            cold_load_seconds = time.perf_counter() - started

            started = time.perf_counter()
            second_data = load_data.load_data()
            warm_load_seconds = time.perf_counter() - started

            assert first_data.equals(second_data)
            first_data.iloc[0, 0] = "caller-mutation"
            assert load_data.load_data().iloc[0, 0] != "caller-mutation"

            started = time.perf_counter()
            first_processed = preprocessing.preprocess_data()
            cold_preprocessing_seconds = time.perf_counter() - started

            started = time.perf_counter()
            second_processed = preprocessing.preprocess_data()
            warm_preprocessing_seconds = time.perf_counter() - started

            _assert_processed_values_equal(first_processed, second_processed)

            original_matrix_value = second_processed["X_train"][0, 0]
            original_scaler_value = second_processed["preprocessor"].named_transformers_[
                "numerical"
            ].named_steps["scaler"].mean_[0]
            second_processed["X_train"][0, 0] = original_matrix_value + 12345
            second_processed["preprocessor"].named_transformers_["numerical"].named_steps[
                "scaler"
            ].mean_[0] = original_scaler_value + 12345
            third_processed = preprocessing.preprocess_data()
            assert third_processed["X_train"][0, 0] == original_matrix_value
            assert (
                third_processed["preprocessor"].named_transformers_["numerical"]
                .named_steps["scaler"].mean_[0]
                == original_scaler_value
            )

            previous_fingerprint = preprocessing._PREPROCESSING_CACHE["fingerprint"]
            changed_data = pd.read_csv(temporary_csv)
            changed_index = changed_data["transaction_count"].first_valid_index()
            changed_data.loc[changed_index, "transaction_count"] += 1
            changed_data.to_csv(temporary_csv, index=False)

            changed_processed = preprocessing.preprocess_data()
            current_fingerprint = preprocessing._PREPROCESSING_CACHE["fingerprint"]
            assert current_fingerprint != previous_fingerprint
            assert changed_processed["X_train"].shape == (40000, 61)
            assert changed_processed["X_test"].shape == (10000, 61)

            print("Stage 1 cache verification passed.")
            print(f"CSV load: cold={cold_load_seconds:.3f}s, warm={warm_load_seconds:.3f}s")
            print(
                "Preprocessing: "
                f"cold={cold_preprocessing_seconds:.3f}s, "
                f"warm={warm_preprocessing_seconds:.3f}s"
            )
            print("Repeated results: equal shapes and values")
            print("Caller mutation protection: passed")
            print("Changed-data SHA-256 invalidation: passed")
        finally:
            load_data.FINAL_50K_PATH = original_path
            _clear_caches()


if __name__ == "__main__":
    main()
