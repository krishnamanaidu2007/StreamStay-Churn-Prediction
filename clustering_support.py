"""Shared, target-free inputs for StreamStay's exploratory clustering pages."""

from __future__ import annotations

import copy
import threading

import numpy as np
import pandas as pd

from load_data import load_data_with_fingerprint
from preprocessing import preprocess_data


RANDOM_STATE = 42

_INPUT_CACHE: dict[tuple[str, int], dict] = {}
_INPUT_CACHE_LOCK = threading.RLock()


def _copy_input(result: dict) -> dict:
    """Return independent values so page code cannot mutate cached inputs."""
    return {
        "features": result["features"].copy(),
        "raw_profiles": result["raw_profiles"].copy(deep=True),
        "feature_names": list(result["feature_names"]),
        "sample_size": result["sample_size"],
        "total_rows": result["total_rows"],
        "fingerprint": result["fingerprint"],
    }


def get_clustering_input(sample_limit: int) -> dict:
    """Prepare a reproducible feature-only customer sample for clustering.

    The existing fitted numerical transformer is reused without changing its
    imputation, scaling, columns, or fit.  The seven continuous customer
    behavior fields are appropriate for Euclidean-distance clustering; the
    encoded categorical fields remain descriptive profile context rather than
    arbitrary numeric distances. ``is_churn`` and the customer identifier are
    never passed to the transformer or returned as clustering inputs.
    """
    if sample_limit < 3:
        raise ValueError("Clustering requires a sample limit of at least three rows.")

    with _INPUT_CACHE_LOCK:
        raw_data, fingerprint = load_data_with_fingerprint()
        cache_key = (fingerprint, sample_limit)
        if cache_key in _INPUT_CACHE:
            return _copy_input(_INPUT_CACHE[cache_key])

        prepared = preprocess_data()
        feature_columns = prepared["numerical_columns"]
        missing_columns = [name for name in feature_columns if name not in raw_data]
        if missing_columns:
            raise ValueError(
                "The dataset is missing approved clustering features: "
                + ", ".join(missing_columns)
            )

        # Match the approved preprocessing data-cleaning rule without using
        # the target to choose the customer sample.
        clean_data = raw_data.drop_duplicates()
        if len(clean_data) < 3:
            raise ValueError("Clustering requires at least three distinct customer rows.")

        sample_size = min(sample_limit, len(clean_data))
        sample = clean_data.sample(n=sample_size, random_state=RANDOM_STATE)
        numerical_transformer = prepared["preprocessor"].named_transformers_["numerical"]
        transformed = numerical_transformer.transform(sample[feature_columns])
        features = (
            transformed.toarray()
            if hasattr(transformed, "toarray")
            else np.asarray(transformed)
        )
        features = np.asarray(features, dtype=float)

        if features.ndim != 2 or features.shape[1] < 2:
            raise ValueError("Clustering requires at least two transformed features.")
        if not np.isfinite(features).all():
            raise ValueError("Clustering inputs must contain only finite values.")

        raw_profiles = sample[
            [
                "age",
                "transaction_count",
                "avg_plan_days",
                "avg_amount_paid",
                "auto_renew_rate",
                "cancel_rate",
                "city",
                "gender",
                "registered_via",
            ]
        ].copy()

        result = {
            "features": features,
            "raw_profiles": raw_profiles,
            "feature_names": list(feature_columns),
            "sample_size": sample_size,
            "total_rows": len(clean_data),
            "fingerprint": fingerprint,
        }
        _INPUT_CACHE.clear()
        _INPUT_CACHE[cache_key] = result
        return _copy_input(result)


def profile_clusters(raw_profiles: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Return descriptive, non-target customer summaries for assigned groups."""
    profile_data = raw_profiles.copy(deep=True)
    profile_data["cluster"] = labels
    profiles = []
    for cluster, group in profile_data.groupby("cluster", sort=True):
        profiles.append(
            {
                "cluster": int(cluster),
                "customers": int(len(group)),
                "share": float(len(group) / len(profile_data) * 100),
                "mean_age": _mean(group, "age"),
                "mean_transactions": _mean(group, "transaction_count"),
                "mean_plan_days": _mean(group, "avg_plan_days"),
                "mean_amount_paid": _mean(group, "avg_amount_paid"),
                "auto_renew_rate": _mean(group, "auto_renew_rate", as_percent=True),
                "cancel_rate": _mean(group, "cancel_rate", as_percent=True),
                "common_city": _mode(group, "city"),
                "common_gender": _mode(group, "gender"),
                "common_registration_method": _mode(group, "registered_via"),
            }
        )
    return profiles


def _mean(group: pd.DataFrame, column: str, as_percent: bool = False) -> float | None:
    value = group[column].mean()
    if pd.isna(value):
        return None
    return float(value * 100) if as_percent else float(value)


def _mode(group: pd.DataFrame, column: str) -> str:
    values = group[column].dropna()
    if values.empty:
        return "Not available"
    return str(values.mode().iloc[0])


def clear_clustering_input_cache() -> None:
    """Test-only helper for verifying cache invalidation behaviour."""
    with _INPUT_CACHE_LOCK:
        _INPUT_CACHE.clear()
