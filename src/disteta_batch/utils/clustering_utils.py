# This module provides functions for performing K-Means clustering, evaluating
# cluster quality, and identifying optimal cluster counts.
import logging
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .constants import CLUSTER_COL, HDR_THRESHOLD_MIN_COUNT, QUANT_PREFIX

logger = logging.getLogger(__name__)


def perform_clustering(X: pd.DataFrame | np.ndarray, k: int) -> KMeans:
    """
    Performs K-Means clustering with robust input validation.

    This function wraps scikit-learn's KMeans, setting a fixed random_state and a
    higher n_init for reproducible and stable results. It includes checks for
    data size, k value, and other common edge cases.

    Args:
        X: The input data (features) to cluster.
        k: The number of clusters to form.

    Returns:
        A fitted scikit-learn KMeans object.
    """
    if not hasattr(X, "shape"):
        raise TypeError("Input data X must be array-like.")
    n_samples = X.shape[0]
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Number of clusters (k) must be positive, got {k}.")
    if n_samples == 0:
        raise ValueError("Cannot cluster empty data.")
    if k > n_samples:
        raise ValueError(
            f"Number of clusters (k={k}) cannot be greater than number of samples ({n_samples})."
        )
    try:
        clusterer = KMeans(n_clusters=k, random_state=10, n_init=20)  # type: ignore
        clusterer.fit(X)
        return clusterer
    except Exception as e:
        logger.error(f"Error during KMeans fitting for k={k}: {e}")
        raise


def calculate_wcss_and_silhouette(
    X: pd.DataFrame | np.ndarray, range_n_clusters: Iterable[int]
) -> Tuple[List[float], Dict[int, float], Dict[int, Optional[np.ndarray]]]:
    """
    Calculates WCSS and silhouette scores for a range of K values.

    This function iterates through a given range of cluster numbers, performs
    K-Means clustering for each, and computes the Within-Cluster Sum of Squares
    (WCSS) and the average silhouette score. It is designed to be robust,
    handling cases where clustering or scoring might fail for specific K values.

    Args:
        X: The input data (features) to cluster.
        range_n_clusters: An iterable of integers representing the K values to test.

    Returns:
        A tuple containing:
        - A list of WCSS scores for each K.
        - A dictionary mapping each K to its average silhouette score.
        - A dictionary mapping each K to the resulting cluster labels.
    """
    wcss_results, silhouette_results, labels_results = [], {}, {}
    if not hasattr(X, "shape"):
        raise TypeError("Input data X must be array-like.")
    n_samples = X.shape[0]
    if n_samples <= 1:
        logger.warning("Cannot calculate WCSS/Silhouette with <= 1 sample.")
        try:
            k_values = list(range_n_clusters)
        except TypeError:
            k_values = []
        return (
            [np.nan] * len(k_values),
            {k: np.nan for k in k_values},
            {k: None for k in k_values},
        )

    for n_clusters in range_n_clusters:
        if not isinstance(n_clusters, int) or n_clusters <= 1:
            wcss_results.append(np.nan)
            silhouette_results[n_clusters] = np.nan
            labels_results[n_clusters] = None
            continue
        cluster_labels = None
        try:
            clusterer = perform_clustering(X, n_clusters)
            wcss_results.append(clusterer.inertia_)
            cluster_labels = clusterer.labels_
            labels_results[n_clusters] = cluster_labels
        except ValueError as ve:
            logger.info(f"Skipping K={n_clusters} due to validation error: {ve}")
            wcss_results.append(np.nan)
            silhouette_results[n_clusters] = np.nan
            labels_results[n_clusters] = None
            continue
        except Exception as clust_e:
            logger.error(f"Error during clustering for K={n_clusters}: {clust_e}")
            wcss_results.append(np.nan)
            silhouette_results[n_clusters] = np.nan
            labels_results[n_clusters] = None
            continue

        if 1 < n_clusters < n_samples:
            try:
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_results[n_clusters] = silhouette_avg
            except Exception as sil_e:
                logger.warning(
                    f"Could not calculate silhouette score for K={n_clusters}: {sil_e}"
                )
                silhouette_results[n_clusters] = np.nan
        else:
            silhouette_results[n_clusters] = np.nan
    return wcss_results, silhouette_results, labels_results


def filter_silhouette_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Filters silhouette scores to start the analysis from the first peak.

    This function identifies the K value with the highest silhouette score (or the
    first K if multiple have the same max score) and returns a dictionary
    containing only the scores from that K value onwards. This is a preprocessing
    step to ensure the subsequent drop-off analysis starts from the most
    promising region of K values.

    Args:
        scores: A dictionary mapping K values to their silhouette scores.

    Returns:
        A sorted dictionary of filtered scores.
    """
    if not scores:
        logger.warning("Input scores dictionary is empty.")
        return {}
    try:
        score_df = pd.DataFrame(list(scores.items()), columns=["k_value", "score"])  # type: ignore
        score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
        score_df.dropna(subset=["score"], inplace=True)
    except Exception as e:
        logger.error(f"Error processing scores DataFrame: {e}")
        return {}
    if score_df.empty:
        logger.warning("No valid numeric scores found after filtering.")
        return {}
    try:
        score_df["rounded_score"] = score_df["score"].round(2)
        max_rounded_score = score_df["rounded_score"].max()
        peak_k_candidates = score_df[score_df["rounded_score"] == max_rounded_score][
            "k_value"
        ]
        max_score_k = peak_k_candidates.min()
    except (ValueError, KeyError) as e:
        logger.warning(
            f"Could not determine peak K for silhouette scores: {e}. Defaulting to first available K."
        )
        if not score_df.empty:
            return dict(
                sorted(score_df.set_index("k_value")["score"].to_dict().items())
            )
        return {}

    filtered_scores = {
        k: v
        for k, v in scores.items()
        if isinstance(k, int) and k >= max_score_k and pd.notna(v)
    }
    return dict(sorted(filtered_scores.items()))


def find_optimal_k_values(
    all_valid_scores: Dict[str, Dict[int, float]], percent_drop_threshold: float
) -> Dict[str, int]:
    """Determines the optimal K for each segment based on silhouette score drop-off."""
    optimal_k_per_key = {}
    for combination_key, scores_dict in all_valid_scores.items():
        if not scores_dict:
            continue

        sorted_k = sorted(scores_dict.keys())
        optimal_k_for_this_key = sorted_k[0]

        for i in range(1, len(sorted_k)):
            current_k = sorted_k[i]
            previous_k = sorted_k[i - 1]
            current_score = scores_dict[current_k]
            previous_score = scores_dict[previous_k]

            drop_percentage = 0.0
            if previous_score != 0:
                drop = (previous_score - current_score) / abs(previous_score)
                drop_percentage = drop
            elif current_score < 0:
                drop_percentage = 1.0

            if drop_percentage <= percent_drop_threshold:
                optimal_k_for_this_key = current_k
            else:
                logger.info(
                    f"  Stopping at K={current_k} for {combination_key}: Drop ({drop_percentage:.2%}) > Threshold ({percent_drop_threshold:.2%})"
                )
                break

        optimal_k_per_key[combination_key] = int(optimal_k_for_this_key)

    if optimal_k_per_key:
        logger.info(f"Determined optimal K values per combination: {optimal_k_per_key}")
    return optimal_k_per_key


def perform_clustering_and_aggregation(
    df_aggregated_by_combination: pd.DataFrame, combination_cols: List[str], k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs final clustering on combination profiles and aggregates results."""
    if df_aggregated_by_combination.empty:
        raise ValueError("Input DataFrame 'df_aggregated_by_combination' is empty.")

    X = df_aggregated_by_combination[combination_cols].copy()
    X = cast(pd.DataFrame, X)
    try:
        clusterer = perform_clustering(X, k)

        df_with_clusters = df_aggregated_by_combination.copy()
        df_with_clusters[CLUSTER_COL] = clusterer.labels_

        df_final_aggregation = (
            df_with_clusters.groupby(CLUSTER_COL, observed=True)[combination_cols]
            .sum()
            .reset_index()
        )
        return df_final_aggregation, df_with_clusters
    except Exception as e:
        logger.error(f"Error during final clustering or aggregation for K={k}: {e}")
        empty_df_agg = pd.DataFrame(columns=[CLUSTER_COL] + combination_cols)  # type: ignore
        empty_df_map = pd.DataFrame(
            columns=df_aggregated_by_combination.columns.tolist() + [CLUSTER_COL]  # type: ignore
        )
        return empty_df_agg, empty_df_map


def calculate_hdr_for_cluster(
    cluster_data: pd.Series,
    n_classes: int,
    hdr_threshold_percentage_from_config: float,
    bin_edges: np.ndarray,
    unit: str,
) -> Dict:
    """
    Calculates the single most significant High-Density Region (HDR) for a cluster's distribution.
    Even if multiple regions are above the density threshold, this function identifies and
    returns only the one containing the most data points (the highest "mass").
    """
    c_cols = cluster_data.filter(like=QUANT_PREFIX)
    if c_cols.size == 0:
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    c_cols.index = c_cols.index.str.replace(QUANT_PREFIX, "").astype(int)
    counts_per_bin = c_cols.reindex(range(1, n_classes + 1), fill_value=0.0)
    total_cluster_count = counts_per_bin.sum()

    if total_cluster_count == 0:
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    target_visual_hdr_mass = (
        total_cluster_count * (100.0 - hdr_threshold_percentage_from_config) / 100.0
    )
    if target_visual_hdr_mass <= 0:
        max_y = counts_per_bin.max() if not counts_per_bin.empty else 0.0
        return {"hdr_threshold_count": max_y + 1, "hdr_intervals": []}

    positive_counts_series = counts_per_bin[counts_per_bin > 0]
    if positive_counts_series.size == 0:
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    max_bin_count = int(positive_counts_series.max())
    best_y_line, min_overshoot_mass = 0.0, float("inf")
    for y_candidate_int in range(max_bin_count, -1, -1):
        y_candidate = float(y_candidate_int)
        current_visual_hdr_mass = (
            (positive_counts_series - y_candidate).clip(lower=0).sum()  # type: ignore
        )
        if current_visual_hdr_mass >= target_visual_hdr_mass:
            if current_visual_hdr_mass < min_overshoot_mass:
                min_overshoot_mass, best_y_line = current_visual_hdr_mass, y_candidate
            elif (
                current_visual_hdr_mass == min_overshoot_mass
                and y_candidate > best_y_line
            ):
                best_y_line = y_candidate

    hdr_line_y_value = float(best_y_line)
    hdr_bins_for_intervals = counts_per_bin[counts_per_bin > hdr_line_y_value]

    if hdr_bins_for_intervals.sum() < HDR_THRESHOLD_MIN_COUNT:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    hdr_bin_indices = sorted(cast(pd.Series, hdr_bins_for_intervals).index.tolist())
    if not hdr_bin_indices:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    # Find contiguous blocks of bins to form intervals
    hdr_bin_indices_np = np.array(hdr_bin_indices)
    splits = np.where(np.diff(hdr_bin_indices_np) != 1)[0] + 1
    consecutive_blocks = np.split(hdr_bin_indices_np, splits)
    all_intervals = [
        (block[0], block[-1]) for block in consecutive_blocks if block.size > 0
    ]

    best_interval, max_mass = None, -1.0
    for start, end in all_intervals:
        interval_mass = counts_per_bin.loc[start:end].sum()
        if interval_mass > max_mass:
            max_mass = interval_mass
            best_interval = (start, end)

    if best_interval is None:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    start, end = best_interval
    if not (0 <= start - 1 < len(bin_edges) and 0 <= end < len(bin_edges)):
        logger.warning(
            f"Best interval bin edges out of bounds for interval [{start}, {end}]. Skipping."
        )
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    start_edge = bin_edges[start - 1]
    end_edge = bin_edges[end]
    interval_counts = counts_per_bin.loc[start:end]
    max_bin_in_interval = interval_counts.idxmax()
    prob_start, prob_end = (
        bin_edges[max_bin_in_interval - 1],
        bin_edges[max_bin_in_interval],
    )

    final_hdr_interval_data = {
        "start_bin_label": int(start),
        "end_bin_label": int(end),
        "median_bin_label": (start + end) / 2.0,
        "peak_bin_label": int(max_bin_in_interval),
        "start_edge_orig_unit": float(start_edge),
        "end_edge_orig_unit": float(end_edge),
        "most_probable_orig_unit": float((prob_start + prob_end) / 2.0),
        "unit": unit,
        "total_mass_in_interval": float(max_mass),
    }

    return {
        "hdr_threshold_count": hdr_line_y_value,
        "hdr_intervals": [final_hdr_interval_data],
    }
