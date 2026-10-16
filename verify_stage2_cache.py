"""Focused Stage 2 cache verification using only a disposable CSV and chart folder."""

import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

import app
import linear_regression
import load_data
import logistic_regression
import preprocessing
import streamstay_eda


def _clear_caches() -> None:
    load_data._DATA_CACHE.update({"fingerprint": None, "data": None})
    preprocessing._PREPROCESSING_CACHE.update({"fingerprint": None, "result": None})
    streamstay_eda._EDA_CACHE.update(
        {"data_fingerprint": None, "signature": None, "result": None}
    )
    linear_regression._LINEAR_CACHE.update(
        {"data_fingerprint": None, "signature": None, "result": None}
    )
    logistic_regression._LOGISTIC_CACHE.update(
        {"data_fingerprint": None, "signature": None, "result": None}
    )


def _timed(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def _assert_charts_exist(charts_dir: Path, result: dict) -> None:
    assert result["charts"]
    assert all((charts_dir / filename).is_file() for filename in result["charts"])


def main() -> None:
    original_csv = load_data.FINAL_50K_PATH
    original_chart_dirs = (
        streamstay_eda.CHARTS_DIR,
        linear_regression.CHARTS_DIR,
        logistic_regression.CHARTS_DIR,
    )
    workspace = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory(dir=workspace) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_csv = temporary_root / "streamstay_50k.csv"
        temporary_charts = temporary_root / "charts"
        shutil.copy2(original_csv, temporary_csv)
        load_data.FINAL_50K_PATH = str(temporary_csv)
        streamstay_eda.CHARTS_DIR = str(temporary_charts)
        linear_regression.CHARTS_DIR = str(temporary_charts)
        logistic_regression.CHARTS_DIR = str(temporary_charts)
        _clear_caches()

        try:
            eda_first, eda_cold = _timed(streamstay_eda.run_eda)
            eda_second, eda_warm = _timed(streamstay_eda.run_eda)
            linear_first, linear_cold = _timed(linear_regression.run_linear_regression)
            linear_second, linear_warm = _timed(linear_regression.run_linear_regression)
            logistic_first, logistic_cold = _timed(logistic_regression.run_logistic_regression)
            logistic_second, logistic_warm = _timed(logistic_regression.run_logistic_regression)

            for first, second in (
                (eda_first, eda_second),
                (linear_first, linear_second),
                (logistic_first, logistic_second),
            ):
                assert first["charts"] == second["charts"]
                _assert_charts_exist(temporary_charts, second)

            assert len(eda_first["charts"]) == 33
            assert len(linear_first["charts"]) == 5
            assert len(logistic_first["charts"]) == 7

            client = app.app.test_client()
            for route in (
                "/",
                "/eda",
                "/preprocessing",
                "/linear-regression",
                "/logistic-regression",
                "/tree-based",
            ):
                response = client.get(route)
                assert response.status_code == 200, route

            old_versions = (
                eda_first["charts"],
                linear_first["charts"],
                logistic_first["charts"],
            )
            changed_data = pd.read_csv(temporary_csv)
            changed_index = changed_data["transaction_count"].first_valid_index()
            changed_data.loc[changed_index, "transaction_count"] += 1
            changed_data.to_csv(temporary_csv, index=False)

            eda_changed = streamstay_eda.run_eda()
            linear_changed = linear_regression.run_linear_regression()
            logistic_changed = logistic_regression.run_logistic_regression()
            for previous, changed in zip(
                old_versions, (eda_changed["charts"], linear_changed["charts"], logistic_changed["charts"])
            ):
                assert previous != changed
                _assert_charts_exist(temporary_charts, changed)

            print("Stage 2 cache verification passed.")
            print(f"EDA: cold={eda_cold:.3f}s, warm={eda_warm:.3f}s")
            print(f"Linear: cold={linear_cold:.3f}s, warm={linear_warm:.3f}s")
            print(f"Logistic: cold={logistic_cold:.3f}s, warm={logistic_warm:.3f}s")
            print("Chart existence, changed-data invalidation, and route HTTP 200 checks: passed")
        finally:
            load_data.FINAL_50K_PATH = original_csv
            (
                streamstay_eda.CHARTS_DIR,
                linear_regression.CHARTS_DIR,
                logistic_regression.CHARTS_DIR,
            ) = original_chart_dirs
            _clear_caches()


if __name__ == "__main__":
    main()
