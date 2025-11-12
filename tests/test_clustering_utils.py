import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.disteta_batch.utils.clustering_utils import (
    calculate_hdr_for_cluster,
    calculate_wcss_and_silhouette,
    filter_silhouette_scores,
    find_optimal_k_values,
    perform_clustering,
    perform_clustering_and_aggregation,
)
from src.disteta_batch.utils.constants import CLUSTER_COL, QUANT_PREFIX


def test_perform_clustering():
    """Tests that perform_clustering returns a KMeans object with the correct number of clusters."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    k = 2
    clusterer = perform_clustering(X, k)
    assert isinstance(clusterer, KMeans)
    assert clusterer.n_clusters == k


def test_calculate_wcss_and_silhouette():
    """Tests that calculate_wcss_and_silhouette returns the correct data structures."""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    k_range = range(2, 4)
    wcss, silh, labels = calculate_wcss_and_silhouette(X, k_range)
    assert isinstance(wcss, list)
    assert isinstance(silh, dict)
    assert isinstance(labels, dict)
    assert len(wcss) == 2
    assert len(silh) == 2
    assert len(labels) == 2


def test_filter_silhouette_scores():
    """Tests that filter_silhouette_scores correctly filters scores."""
    scores = {2: 0.5, 3: 0.6, 4: 0.55, 5: 0.4}
    filtered_scores = filter_silhouette_scores(scores)
    assert filtered_scores == {3: 0.6, 4: 0.55, 5: 0.4}


def test_find_optimal_k_values():
    """Tests that find_optimal_k_values correctly identifies the optimal K."""
    all_valid_scores = {
        "segment1": {2: 0.5, 3: 0.6, 4: 0.55, 5: 0.4},
        "segment2": {2: 0.3, 3: 0.35, 4: 0.32, 5: 0.2},
    }
    percent_drop_threshold = 0.1
    optimal_k = find_optimal_k_values(all_valid_scores, percent_drop_threshold)
    assert optimal_k == {"segment1": 4, "segment2": 4}


def test_perform_clustering_and_aggregation():
    """Tests that perform_clustering_and_aggregation returns correct aggregated data."""
    df_aggregated_by_combination = pd.DataFrame(
        {
            "comb": [1, 1, 2, 2],
            f"{QUANT_PREFIX}1": [0.1, 0.2, 0.7, 0.8],
            f"{QUANT_PREFIX}2": [0.9, 0.8, 0.3, 0.2],
        }
    )
    combination_cols = [f"{QUANT_PREFIX}1", f"{QUANT_PREFIX}2"]
    k = 2
    df_sum, df_map = perform_clustering_and_aggregation(
        df_aggregated_by_combination, combination_cols, k
    )
    assert isinstance(df_sum, pd.DataFrame)
    assert isinstance(df_map, pd.DataFrame)
    assert CLUSTER_COL in df_sum.columns
    assert CLUSTER_COL in df_map.columns
    assert df_sum.shape[0] == k


def test_calculate_hdr_for_cluster():
    """Tests that calculate_hdr_for_cluster returns the correct HDR information."""
    cluster_data = pd.Series(
        {
            f"{QUANT_PREFIX}1": 10,
            f"{QUANT_PREFIX}2": 20,
            f"{QUANT_PREFIX}3": 50,
            f"{QUANT_PREFIX}4": 15,
            f"{QUANT_PREFIX}5": 5,
        }
    )
    n_classes = 5
    hdr_threshold_percentage_from_config = 90.0
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    unit = "units"

    hdr_result = calculate_hdr_for_cluster(
        cluster_data, n_classes, hdr_threshold_percentage_from_config, bin_edges, unit
    )

    assert isinstance(hdr_result, dict)
    assert "hdr_threshold_count" in hdr_result
    assert "hdr_intervals" in hdr_result
    assert len(hdr_result["hdr_intervals"]) == 1
    assert hdr_result["hdr_intervals"][0]["start_bin_label"] == 3
    assert hdr_result["hdr_intervals"][0]["end_bin_label"] == 3
