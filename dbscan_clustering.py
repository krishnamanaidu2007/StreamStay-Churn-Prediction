"""Bounded DBSCAN customer exploration with target-free parameter selection."""

from __future__ import annotations

import copy
import hashlib
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from clustering_support import RANDOM_STATE, get_clustering_input, profile_clusters
from load_data import get_data_fingerprint
from result_cache import (
    chart_files_exist,
    result_version,
    save_chart_atomically,
    source_signature,
    versioned_chart_filename,
)


SAMPLE_LIMIT = 2500
MIN_SAMPLES = 10
EPS_QUANTILE = 0.90
SILHOUETTE_LIMIT = 1500

_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.RLock()
_CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")


def _chart_path(filename: str) -> str:
    return os.path.join(_CHARTS_DIR, filename)


def _select_eps(features: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    if len(features) < min_samples:
        raise ValueError(f"DBSCAN requires at least {min_samples} customer rows.")
    distances, _ = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(features).kneighbors(features)
    k_distances = np.sort(distances[:, -1])
    eps = float(np.quantile(k_distances, EPS_QUANTILE))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("DBSCAN could not derive a positive finite epsilon from the customer sample.")
    return eps, k_distances


def _save_k_distance_chart(k_distances: np.ndarray, eps: float, min_samples: int, filename: str) -> None:
    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        axis.plot(np.arange(1, len(k_distances) + 1), k_distances, color="#2d7dd2", linewidth=1.7)
        axis.axhline(eps, color="#db5a42", linestyle="--", label=f"Selected eps = {eps:.3f}")
        axis.set_title(f"{min_samples}-Nearest-Neighbor Distance Distribution")
        axis.set_xlabel("Customers sorted by distance")
        axis.set_ylabel(f"Distance to {min_samples}th nearest neighbor")
        axis.legend()
        axis.grid(alpha=.22)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_chart_path(filename), save)


def _save_projection(features: np.ndarray, labels: np.ndarray, eps: float, filename: str) -> list[float]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(features)
    coordinates = pca.transform(features)

    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.8))
        unique_labels = sorted(np.unique(labels))
        palette = plt.get_cmap("tab10")
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                axis.scatter(coordinates[mask, 0], coordinates[mask, 1], marker="x", color="#db5a42", s=20, alpha=.72, label=f"Noise ({int(mask.sum()):,})")
            else:
                axis.scatter(coordinates[mask, 0], coordinates[mask, 1], color=palette(int(label) % 10), s=15, alpha=.65, label=f"Cluster {label}")
        axis.set_title(f"DBSCAN Customer Groups in PCA Space (eps = {eps:.3f})")
        axis.set_xlabel(f"Principal component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axis.set_ylabel(f"Principal component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_chart_path(filename), save)
    return [float(value) for value in pca.explained_variance_ratio_]


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    assigned = labels != -1
    assigned_labels = labels[assigned]
    if len(np.unique(assigned_labels)) < 2 or len(assigned_labels) <= len(np.unique(assigned_labels)):
        return None
    assigned_features = features[assigned]
    if len(assigned_features) > SILHOUETTE_LIMIT:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(assigned_features), size=SILHOUETTE_LIMIT, replace=False)
        assigned_features = assigned_features[indices]
        assigned_labels = assigned_labels[indices]
    return float(silhouette_score(assigned_features, assigned_labels))


def _build_result(chart_version: str, requested_eps: float | None, min_samples: int) -> dict:
    if not 2 <= min_samples <= 50:
        raise ValueError("Minimum samples must be an integer from 2 to 50.")
    clustering_input = get_clustering_input(SAMPLE_LIMIT)
    features = clustering_input["features"]
    automatic_eps, k_distances = _select_eps(features, min_samples)
    eps = automatic_eps if requested_eps is None else requested_eps
    if not np.isfinite(eps) or not 0.05 <= eps <= 10:
        raise ValueError("EPS must be a number from 0.05 to 10.")
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(features)
    cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_count = int(np.count_nonzero(labels == -1))

    k_distance_filename = versioned_chart_filename("dbscan_k_distance.png", chart_version)
    projection_filename = versioned_chart_filename("dbscan_pca_clusters.png", chart_version)
    _save_k_distance_chart(k_distances, eps, min_samples, k_distance_filename)
    pca_variance = _save_projection(features, labels, eps, projection_filename)

    return {
        "eps": eps,
        "min_samples": min_samples,
        "automatic_eps": automatic_eps,
        "eps_quantile": EPS_QUANTILE,
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_share": float(noise_count / len(labels) * 100),
        "silhouette": _safe_silhouette(features, labels),
        "sample_size": clustering_input["sample_size"],
        "total_rows": clustering_input["total_rows"],
        "feature_count": features.shape[1],
        "profiles": profile_clusters(clustering_input["raw_profiles"], labels),
        "chart_files": {"k_distance": k_distance_filename, "projection": projection_filename},
        "pca_variance": pca_variance,
        "version": chart_version,
    }


def run_dbscan_clustering(eps: float | None = None, min_samples: int = MIN_SAMPLES) -> dict:
    """Return cached DBSCAN results for the current data and source code."""
    with _CACHE_LOCK:
        fingerprint = get_data_fingerprint()
        signature = source_signature(
            __file__,
            os.path.join(os.path.dirname(__file__), "clustering_support.py"),
            os.path.join(os.path.dirname(__file__), "preprocessing.py"),
            os.path.join(os.path.dirname(__file__), "result_cache.py"),
        )
        signature = hashlib.sha256(f"{signature}:{eps}:{min_samples}".encode()).hexdigest()
        version = result_version(fingerprint, signature)
        chart_names = [
            versioned_chart_filename("dbscan_k_distance.png", version),
            versioned_chart_filename("dbscan_pca_clusters.png", version),
        ]
        if version in _CACHE and chart_files_exist(_CHARTS_DIR, chart_names):
            return copy.deepcopy(_CACHE[version])
        result = _build_result(version, eps, min_samples)
        if get_data_fingerprint() != fingerprint:
            return run_dbscan_clustering(eps, min_samples)
        _CACHE.clear()
        _CACHE[version] = result
        return copy.deepcopy(result)


def clear_dbscan_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
