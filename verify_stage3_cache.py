"""Focused Stage 3 verification using a disposable CSV and chart directory."""

import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

import app
import load_data
import preprocessing
import tree_based


def _clear_caches() -> None:
    load_data._DATA_CACHE.update({"fingerprint": None, "data": None})
    preprocessing._PREPROCESSING_CACHE.update({"fingerprint": None, "result": None})
    tree_based._TREE_SUITE_CACHE.update(
        {"data_fingerprint": None, "signature": None, "suite": None}
    )


def _timed(call):
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def _assert_charts_exist(charts_dir: Path, result: dict) -> None:
    assert len(result["charts"]) == 4
    assert all((charts_dir / filename).is_file() for filename in result["charts"])


def main() -> None:
    original_csv = load_data.FINAL_50K_PATH
    original_charts_dir = tree_based.CHARTS_DIR
    workspace = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory(dir=workspace) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_csv = temporary_root / "streamstay_50k.csv"
        temporary_charts = temporary_root / "charts"
        shutil.copy2(original_csv, temporary_csv)
        load_data.FINAL_50K_PATH = str(temporary_csv)
        tree_based.CHARTS_DIR = str(temporary_charts)
        _clear_caches()

        try:
            first, cold_seconds = _timed(
                lambda: tree_based.run_tree_models("random_forest")
            )
            selected, cached_seconds = _timed(
                lambda: tree_based.run_tree_models("xgboost")
            )
            assert first["algorithm_count"] == 7
            assert first["selected"]["algorithm"] == "Random Forest"
            assert selected["selected"]["algorithm"] == "XGBoost"
            assert first["comparison_rows"] == selected["comparison_rows"]
            assert first["comparison_plot"] == selected["comparison_plot"]
            assert first["charts"] != selected["charts"]
            _assert_charts_exist(temporary_charts, first)
            _assert_charts_exist(temporary_charts, selected)

            client = app.app.test_client()
            assert client.get("/tree-based").status_code == 200
            assert client.post(
                "/tree-based", data={"algorithm": "random_forest"}
            ).status_code == 200
            assert client.post(
                "/tree-based", data={"algorithm": "xgboost"}).status_code == 200

            prior_fingerprint = tree_based._TREE_SUITE_CACHE["data_fingerprint"]
            changed_data = pd.read_csv(temporary_csv)
            changed_index = changed_data["transaction_count"].first_valid_index()
            changed_data.loc[changed_index, "transaction_count"] += 1
            changed_data.to_csv(temporary_csv, index=False)
            invalidated = tree_based.run_tree_models("lightgbm")
            assert tree_based._TREE_SUITE_CACHE["data_fingerprint"] != prior_fingerprint
            assert invalidated["selected"]["algorithm"] == "LightGBM"
            assert invalidated["comparison_plot"] != first["comparison_plot"]
            _assert_charts_exist(temporary_charts, invalidated)

            # A changed source signature must take the same cache-miss path.
            tree_based._TREE_SUITE_CACHE["signature"] = "obsolete-source-signature"
            source_invalidated = tree_based.run_tree_models("decision_tree")
            assert tree_based._TREE_SUITE_CACHE["signature"] != "obsolete-source-signature"
            assert source_invalidated["selected"]["algorithm"] == "Decision Tree"
            _assert_charts_exist(temporary_charts, source_invalidated)

            print("Stage 3 cache verification passed.")
            print(f"Tree suite: cold={cold_seconds:.3f}s, cached selection={cached_seconds:.3f}s")
            print("Selection reuse, data/source invalidation, versioned charts, and HTTP 200: passed")
        finally:
            load_data.FINAL_50K_PATH = original_csv
            tree_based.CHARTS_DIR = original_charts_dir
            _clear_caches()


if __name__ == "__main__":
    main()
