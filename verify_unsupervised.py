"""Focused executable checks for StreamStay's three clustering pages."""

import os
import re

import pandas as pd

import clustering_support
from clustering_support import clear_clustering_input_cache, get_clustering_input
from dbscan_clustering import run_dbscan_clustering
from hierarchical_clustering import run_hierarchical_clustering
from kmeans_clustering import run_kmeans_clustering
from load_data import get_data_fingerprint
from result_cache import result_version
from app import app


def _assert_charts(result):
    for filename in result["chart_files"].values():
        assert os.path.isfile(os.path.join("static", "charts", filename)), filename


def _check_target_free_defensive_input():
    first = get_clustering_input(20)
    assert first["features"].shape[0] == 20
    assert all("is_churn" not in name for name in first["feature_names"])
    assert "is_churn" not in first["raw_profiles"]
    assert "msno" not in first["raw_profiles"]
    original = first["features"][0, 0]
    first["features"][0, 0] = original + 999
    second = get_clustering_input(20)
    assert second["features"][0, 0] == original

    try:
        get_clustering_input(2)
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid clustering sample size was accepted.")


def _check_missing_feature_error():
    original_loader = clustering_support.load_data_with_fingerprint
    try:
        clear_clustering_input_cache()
        clustering_support.load_data_with_fingerprint = lambda: (pd.DataFrame({"age": [20, 30, 40]}), "missing")
        try:
            get_clustering_input(3)
        except ValueError as error:
            assert "missing approved clustering features" in str(error)
        else:
            raise AssertionError("Missing approved features were accepted.")
    finally:
        clustering_support.load_data_with_fingerprint = original_loader
        clear_clustering_input_cache()


def _check_cache_and_signature():
    for runner in (run_kmeans_clustering, run_hierarchical_clustering, run_dbscan_clustering):
        first = runner()
        second = runner()
        assert first == second
        _assert_charts(first)
    fingerprint = get_data_fingerprint()
    assert result_version(fingerprint, "source-a") != result_version(fingerprint, "source-b")
    assert result_version("different-data", "source-a") != result_version(fingerprint, "source-a")


def _check_interactive_parameters():
    k_manual_two = run_kmeans_clustering("manual", 2)
    k_manual_three = run_kmeans_clustering("manual", 3)
    k_elbow = run_kmeans_clustering("elbow", 3)
    assert k_manual_two["selected_k"] == 2
    assert k_manual_three["selected_k"] == 3
    assert k_elbow["method"] == "elbow"
    assert len({k_manual_two["version"], k_manual_three["version"], k_elbow["version"]}) == 3

    ward = run_hierarchical_clustering("ward", 2)
    complete = run_hierarchical_clustering("complete", 3)
    assert ward["selected_k"] == 2 and ward["linkage_method"] == "Ward"
    assert complete["selected_k"] == 3 and complete["linkage_method"] == "Complete"
    assert ward["version"] != complete["version"]

    dbscan_tight = run_dbscan_clustering(1.0, 5)
    dbscan_wide = run_dbscan_clustering(3.0, 10)
    assert dbscan_tight["eps"] == 1.0 and dbscan_tight["min_samples"] == 5
    assert dbscan_wide["eps"] == 3.0 and dbscan_wide["min_samples"] == 10
    assert dbscan_tight["version"] != dbscan_wide["version"]

    invalid_calls = (
        (run_kmeans_clustering, ("manual", 1)),
        (run_hierarchical_clustering, ("invalid", 3)),
        (run_dbscan_clustering, (0.01, 10)),
        (run_dbscan_clustering, (1.0, 1)),
    )
    for runner, arguments in invalid_calls:
        try:
            runner(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid parameters were accepted: {arguments}")


def _check_flask_parameter_posts():
    client = app.test_client()
    submissions = (
        ("/kmeans-clustering", {"method": "manual", "k": "2"}, b"Manual K Selection used your requested K = 2"),
        ("/kmeans-clustering", {"method": "elbow", "k": "3"}, b"Elbow Method selected K ="),
        ("/hierarchical-clustering", {"linkage": "ward", "k": "2"}, b"Ward linkage was used"),
        ("/hierarchical-clustering", {"linkage": "complete", "k": "3"}, b"Complete linkage was used"),
        ("/dbscan", {"eps": "1.0", "min_samples": "5"}, b"submitted <code>eps</code> value is used directly"),
        ("/dbscan", {"eps": "3.0", "min_samples": "10"}, b"submitted <code>eps</code> value is used directly"),
    )
    for route, form, expected in submissions:
        response = client.post(route, data=form)
        assert response.status_code == 200, (route, response.status_code)
        assert expected in response.data, route
        chart_urls = re.findall(rb'src="(/static/charts/[^"]+)"', response.data)
        assert chart_urls, route
        for chart_url in chart_urls:
            assert client.get(chart_url.decode()).status_code == 200, chart_url

    invalid_submissions = (
        ("/kmeans-clustering", {"method": "manual", "k": "1"}, b"Manual K must be an integer from 2 to 6."),
        ("/hierarchical-clustering", {"linkage": "ward", "k": "invalid"}, b"Number of clusters must be an integer from 2 to 5."),
        ("/dbscan", {"eps": "bad", "min_samples": "10"}, b"EPS must be a number from 0.05 to 10."),
        ("/dbscan", {"eps": "1.0", "min_samples": "1"}, b"Minimum samples must be an integer from 2 to 50."),
    )
    for route, form, expected in invalid_submissions:
        response = client.post(route, data=form)
        assert response.status_code == 200, (route, response.status_code)
        assert expected in response.data, route


if __name__ == "__main__":
    _check_target_free_defensive_input()
    _check_missing_feature_error()
    _check_cache_and_signature()
    _check_interactive_parameters()
    _check_flask_parameter_posts()
    print("Unsupervised verification passed.")
