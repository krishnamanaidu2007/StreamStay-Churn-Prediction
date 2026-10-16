"""Bounded hierarchical customer clustering for StreamStay."""

from __future__ import annotations

import copy
import hashlib
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from clustering_support import RANDOM_STATE, get_clustering_input, profile_clusters
from load_data import get_data_fingerprint
from result_cache import (
    chart_files_exist,
    result_version,
    save_chart_atomically,
    source_signature,
    versioned_chart_filename,
)


SAMPLE_LIMIT = 1500
SILHOUETTE_LIMIT = 1200
K_VALUES = tuple(range(2, 6))
LINKAGE_METHOD = "ward"
DISTANCE_METRIC = "euclidean"

_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.RLock()
_CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")


def _chart_path(filename: str) -> str:
    return os.path.join(_CHARTS_DIR, filename)


def _silhouette_sample(features: np.ndarray) -> np.ndarray:
    if len(features) <= SILHOUETTE_LIMIT:
        return features
    rng = np.random.default_rng(RANDOM_STATE)
    return features[rng.choice(len(features), size=SILHOUETTE_LIMIT, replace=False)]


def _save_dendrogram(features: np.ndarray, selected_k: int, linkage_method: str, filename: str) -> None:
    # A deterministic 200-customer display keeps the dendrogram readable while
    # the cluster assignments themselves use the full bounded sample.
    display_size = min(200, len(features))
    rng = np.random.default_rng(RANDOM_STATE)
    display = features[rng.choice(len(features), size=display_size, replace=False)]
    linkage_matrix = linkage(display, method=linkage_method, metric=DISTANCE_METRIC)
    cutoff_index = max(0, len(linkage_matrix) - selected_k)
    cutoff = float(linkage_matrix[cutoff_index, 2])

    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(10, 5.8))
        dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=axis)
        axis.axhline(cutoff, color="#db5a42", linestyle="--", label=f"Reference cut for K = {selected_k}")
        axis.set_title(f"Hierarchical Customer Grouping — {linkage_method.title()} Linkage")
        axis.set_xlabel("Merged customer groups (leaf labels show group sizes)")
        axis.set_ylabel("Euclidean linkage distance")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_chart_path(filename), save)


def _save_projection(features: np.ndarray, labels: np.ndarray, selected_k: int, filename: str) -> list[float]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(features)
    coordinates = pca.transform(features)

    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.8))
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=15, alpha=.65)
        handles, _ = scatter.legend_elements()
        axis.legend(handles, [f"Cluster {k}" for k in range(selected_k)], title="Group", loc="best")
        axis.set_title(f"Hierarchical Customer Groups in PCA Space (K = {selected_k})")
        axis.set_xlabel(f"Principal component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axis.set_ylabel(f"Principal component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_chart_path(filename), save)
    return [float(value) for value in pca.explained_variance_ratio_]


def _build_result(chart_version: str, linkage_method: str, requested_k: int | None) -> dict:
    if linkage_method not in {"ward", "complete", "average"}:
        raise ValueError("Linkage method must be Ward, Complete, or Average.")
    if requested_k is not None and requested_k not in K_VALUES:
        raise ValueError("Number of clusters must be an integer from 2 to 5.")
    clustering_input = get_clustering_input(SAMPLE_LIMIT)
    features = clustering_input["features"]
    if len(features) < max(K_VALUES) + 1:
        raise ValueError("Not enough customer records are available for hierarchical clustering.")

    evaluation_features = _silhouette_sample(features)
    scores = []
    for k in K_VALUES:
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=DISTANCE_METRIC).fit_predict(evaluation_features)
        scores.append(float(silhouette_score(evaluation_features, labels)))
    selected_index = int(np.argmax(scores))
    selected_k = requested_k if requested_k is not None else K_VALUES[selected_index]
    labels = AgglomerativeClustering(
        n_clusters=selected_k, linkage=linkage_method, metric=DISTANCE_METRIC
    ).fit_predict(features)

    dendrogram_filename = versioned_chart_filename("hierarchical_dendrogram.png", chart_version)
    projection_filename = versioned_chart_filename("hierarchical_pca_clusters.png", chart_version)
    _save_dendrogram(features, selected_k, linkage_method, dendrogram_filename)
    pca_variance = _save_projection(features, labels, selected_k, projection_filename)

    return {
        "selected_k": selected_k,
        "silhouette_scores": [{"k": k, "score": score} for k, score in zip(K_VALUES, scores)],
        "best_silhouette": scores[selected_index],
        "sample_size": clustering_input["sample_size"],
        "total_rows": clustering_input["total_rows"],
        "feature_count": features.shape[1],
        "linkage_method": linkage_method.title(),
        "distance_metric": "Euclidean",
        "profiles": profile_clusters(clustering_input["raw_profiles"], labels),
        "chart_files": {"dendrogram": dendrogram_filename, "projection": projection_filename},
        "pca_variance": pca_variance,
        "version": chart_version,
    }


def run_hierarchical_clustering(linkage_method: str = LINKAGE_METHOD, k: int | None = None) -> dict:
    """Return cached hierarchical clustering results for current data and source."""
    with _CACHE_LOCK:
        fingerprint = get_data_fingerprint()
        signature = source_signature(
            __file__,
            os.path.join(os.path.dirname(__file__), "clustering_support.py"),
            os.path.join(os.path.dirname(__file__), "preprocessing.py"),
            os.path.join(os.path.dirname(__file__), "result_cache.py"),
        )
        signature = hashlib.sha256(f"{signature}:{linkage_method}:{k}".encode()).hexdigest()
        version = result_version(fingerprint, signature)
        chart_names = [
            versioned_chart_filename("hierarchical_dendrogram.png", version),
            versioned_chart_filename("hierarchical_pca_clusters.png", version),
        ]
        if version in _CACHE and chart_files_exist(_CHARTS_DIR, chart_names):
            return copy.deepcopy(_CACHE[version])
        result = _build_result(version, linkage_method, k)
        if get_data_fingerprint() != fingerprint:
            return run_hierarchical_clustering(linkage_method, k)
        _CACHE.clear()
        _CACHE[version] = result
        return copy.deepcopy(result)


def clear_hierarchical_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
