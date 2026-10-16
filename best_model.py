"""A read-only leaderboard built from StreamStay's existing classifier results."""

from __future__ import annotations

import copy
import os
import threading

from load_data import get_data_fingerprint
from logistic_regression import run_logistic_regression
from result_cache import source_signature
from tree_based import run_tree_models


_CACHE = {"fingerprint": None, "signature": None, "result": None}
_CACHE_LOCK = threading.RLock()


def _build_leaderboard() -> dict:
    logistic = run_logistic_regression()
    tree = run_tree_models("random_forest")
    rows = []
    for row in logistic["comparison_rows"]:
        rows.append({"family": "Logistic Regression", **row})
    for row in tree["comparison_rows"]:
        rows.append({"family": "Tree-Based", **row})
    ranked = sorted(rows, key=lambda row: (row["ROC-AUC"], row["F1"], row["Recall"]), reverse=True)
    return {
        "leaderboard": ranked,
        "best": ranked[0],
        "eligible_model_count": len(ranked),
        "test_shape": logistic["testing_shape"],
        "feature_count": logistic["feature_count"],
    }


def run_best_model() -> dict:
    """Rank already-approved classifiers without changing their configuration."""
    with _CACHE_LOCK:
        fingerprint = get_data_fingerprint()
        signature = source_signature(
            __file__,
            os.path.join(os.path.dirname(__file__), "logistic_regression.py"),
            os.path.join(os.path.dirname(__file__), "tree_based.py"),
            os.path.join(os.path.dirname(__file__), "preprocessing.py"),
        )
        if (
            _CACHE["fingerprint"] == fingerprint
            and _CACHE["signature"] == signature
            and _CACHE["result"] is not None
        ):
            return copy.deepcopy(_CACHE["result"])
        result = _build_leaderboard()
        if get_data_fingerprint() != fingerprint:
            return run_best_model()
        _CACHE.update({"fingerprint": fingerprint, "signature": signature, "result": result})
        return copy.deepcopy(result)


def clear_best_model_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.update({"fingerprint": None, "signature": None, "result": None})
