"""Cached, bounded K-Means customer exploration for StreamStay."""

from __future__ import annotations

import copy
import hashlib
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
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


SAMPLE_LIMIT = 5000
SILHOUETTE_LIMIT = 2000
K_VALUES = tuple(range(2, 7))

_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.RLock()
_CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")


def _plot_path(filename: str) -> str:
    return os.path.join(_CHARTS_DIR, filename)


def _silhouette_sample(features: np.ndarray) -> np.ndarray:
    if len(features) <= SILHOUETTE_LIMIT:
        return features
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(features), size=SILHOUETTE_LIMIT, replace=False)
    return features[indices]


def _select_elbow_k(k_values: list[int], inertia: list[float]) -> int:
    """Choose the interior WCSS point farthest from the end-point trend line."""
    points = np.column_stack((np.asarray(k_values, dtype=float), np.asarray(inertia, dtype=float)))
    start, end = points[0], points[-1]
    line = end - start
    line_length = float(np.linalg.norm(line))
    if line_length == 0:
        return k_values[1]
    offsets = points - start
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / line_length
    # The endpoints always lie on the trend line, so an elbow must be internal.
    selected_index = int(np.argmax(distances[1:-1])) + 1
    return k_values[selected_index]


def _save_silhouette_chart(k_values: list[int], scores: list[float], selected_k: int, filename: str) -> None:
    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        axis.plot(k_values, scores, marker="o", linewidth=2, color="#2d7dd2")
        axis.axvline(selected_k, color="#db5a42", linestyle="--", label=f"Selected K = {selected_k}")
        axis.set_title("K-Means Silhouette Scores")
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel("Silhouette score")
        axis.set_xticks(k_values)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_plot_path(filename), save)


def _save_projection_chart(features: np.ndarray, labels: np.ndarray, centers: np.ndarray, selected_k: int, filename: str) -> list[float]:
    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit(features)
    coordinates = projection.transform(features)
    projected_centers = projection.transform(centers)

    def save(destination: str) -> None:
        figure, axis = plt.subplots(figsize=(8.5, 5.8))
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=14, alpha=0.62
        )
        axis.scatter(
            projected_centers[:, 0], projected_centers[:, 1], marker="X", s=170,
            color="#172a3a", edgecolor="white", linewidth=1.0, label="Centroids"
        )
        axis.set_title(f"K-Means Customer Groups in PCA Space (K = {selected_k})")
        axis.set_xlabel(f"Principal component 1 ({projection.explained_variance_ratio_[0]:.1%} variance)")
        axis.set_ylabel(f"Principal component 2 ({projection.explained_variance_ratio_[1]:.1%} variance)")
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=plt.get_cmap("tab10")(k),
                   markersize=7, label=f"Cluster {k}")
            for k in range(selected_k)
        ]
        handles.append(
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#172a3a",
                   markeredgecolor="white", markersize=9, label="Centroids")
        )
        axis.legend(handles=handles, title="Group", loc="best")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    save_chart_atomically(_plot_path(filename), save)
    return [float(value) for value in projection.explained_variance_ratio_]


def _build_result(chart_version: str, method: str, manual_k: int) -> dict:
    clustering_input = get_clustering_input(SAMPLE_LIMIT)
    features = clustering_input["features"]
    if len(features) < max(K_VALUES) + 1:
        raise ValueError("Not enough customer records are available to compare K-Means clusters.")

    evaluation_features = _silhouette_sample(features)
    k_values: list[int] = []
    scores: list[float] = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(evaluation_features)
        k_values.append(k)
        scores.append(float(silhouette_score(evaluation_features, labels)))

    selected_index = int(np.argmax(scores))
    silhouette_k = k_values[selected_index]
    inertia = []
    for k in (1, *K_VALUES):
        inertia.append(float(KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(features).inertia_))
    elbow_k = _select_elbow_k(list(range(1, 7)), inertia)
    if method == "manual":
        if manual_k not in K_VALUES:
            raise ValueError("Manual K must be an integer from 2 to 6.")
        selected_k = manual_k
    elif method == "elbow":
        selected_k = elbow_k
    elif method == "silhouette":
        selected_k = silhouette_k
    else:
        raise ValueError("Choose Silhouette Analysis, Elbow Method, or Manual K Selection.")
    final_model = KMeans(n_clusters=selected_k, n_init=10, random_state=RANDOM_STATE)
    labels = final_model.fit_predict(features)

    silhouette_filename = versioned_chart_filename("kmeans_silhouette.png", chart_version)
    elbow_filename = versioned_chart_filename("kmeans_elbow.png", chart_version)
    projection_filename = versioned_chart_filename("kmeans_pca_clusters.png", chart_version)
    _save_silhouette_chart(k_values, scores, selected_k, silhouette_filename)
    _save_elbow_chart(list(range(1, 7)), inertia, selected_k, elbow_filename)
    explained_variance = _save_projection_chart(
        features, labels, final_model.cluster_centers_, selected_k, projection_filename
    )

    return {
        "selected_k": selected_k,
        "method": method,
        "silhouette_k": silhouette_k,
        "elbow_k": elbow_k,
        "silhouette_scores": [
            {"k": k, "score": score} for k, score in zip(k_values, scores)
        ],
        "best_silhouette": scores[selected_index],
        "sample_size": clustering_input["sample_size"],
        "total_rows": clustering_input["total_rows"],
        "feature_count": features.shape[1],
        "profiles": profile_clusters(clustering_input["raw_profiles"], labels),
        "chart_files": {
            "silhouette": silhouette_filename,
            "elbow": elbow_filename,
            "projection": projection_filename,
        },
        "pca_variance": explained_variance,
        "version": chart_version,
    }


def _save_elbow_chart(k_values, inertia, selected_k, filename):
    def save(destination):
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        axis.plot(k_values, inertia, marker="o", linewidth=2, color="#2d7dd2")
        axis.axvline(selected_k, color="#db5a42", linestyle="--", label=f"Selected K = {selected_k}")
        axis.set(title="K-Means Elbow Curve (WCSS)", xlabel="Number of clusters (K)", ylabel="Within-cluster sum of squares")
        axis.set_xticks(k_values); axis.grid(alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure)
    save_chart_atomically(_plot_path(filename), save)


def run_kmeans_clustering(method: str = "silhouette", manual_k: int = 3) -> dict:
    """Return a complete cached K-Means analysis for the current source CSV."""
    with _CACHE_LOCK:
        data_fingerprint = get_data_fingerprint()
        signature = source_signature(
            __file__,
            os.path.join(os.path.dirname(__file__), "clustering_support.py"),
            os.path.join(os.path.dirname(__file__), "preprocessing.py"),
            os.path.join(os.path.dirname(__file__), "result_cache.py"),
        )
        signature = hashlib.sha256(f"{signature}:{method}:{manual_k}".encode()).hexdigest()
        version = result_version(data_fingerprint, signature)
        chart_names = [
            versioned_chart_filename("kmeans_silhouette.png", version),
            versioned_chart_filename("kmeans_elbow.png", version),
            versioned_chart_filename("kmeans_pca_clusters.png", version),
        ]
        if version in _CACHE and chart_files_exist(_CHARTS_DIR, chart_names):
            return copy.deepcopy(_CACHE[version])

        result = _build_result(version, method, manual_k)
        if get_data_fingerprint() != data_fingerprint:
            return run_kmeans_clustering(method, manual_k)
        _CACHE.clear()
        _CACHE[version] = result
        return copy.deepcopy(result)


def clear_kmeans_cache() -> None:
    """Test-only helper for process-local K-Means cache checks."""
    with _CACHE_LOCK:
        _CACHE.clear()
