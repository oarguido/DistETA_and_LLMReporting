"""
This module provides functions for performing K-Means clustering, evaluating cluster quality, and identifying the optimal number of clusters (K).

Key functionalities include:
- Performing K-Means clustering with robust input validation.
- Calculating WCSS and silhouette scores across a range of K values.
- Filtering silhouette scores to find the most promising region for analysis.
- Determining the optimal K for each data segment based on a score drop-off heuristic.
- Calculating High-Density Regions (HDR) for final cluster profiles.
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .constants import CLUSTER_COL, HDR_THRESHOLD_MIN_COUNT, QUANT_PREFIX

logger = logging.getLogger(__name__)


def perform_clustering(X: Union[pd.DataFrame, pd.Series, np.ndarray], k: int) -> KMeans:
    """
    Performs K-Means clustering with robust settings for reproducibility.

    Args:
        X: The input data (features) to cluster. Can be a DataFrame, Series, or ndarray.
        k: The number of clusters to form.

    Returns:
        A fitted scikit-learn KMeans object.

    Raises:
        ValueError: If k is invalid or data is empty or too small.
    """
    X_data = X
    if isinstance(X, pd.Series):
        X_data = X.to_frame()

    if not hasattr(X_data, "shape"):
        raise TypeError("Input data X must be array-like.")
    n_samples = X_data.shape[0]
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Number of clusters (k) must be positive, got {k}.")
    if n_samples == 0:
        raise ValueError("Cannot cluster empty data.")
    if k > n_samples:
        raise ValueError(
            f"Number of clusters (k={k}) cannot be greater than number of samples ({n_samples})."
        )

    try:
        clusterer = KMeans(n_clusters=k, random_state=10, n_init="auto")
        clusterer.fit(X_data)
        return clusterer
    except Exception as e:
        logger.error(f"Error during KMeans fitting for k={k}: {e}")
        raise


def calculate_wcss_and_silhouette(
    X: Union[pd.DataFrame, pd.Series, np.ndarray], range_n_clusters: Iterable[int]
) -> Tuple[List[float], Dict[int, float], Dict[int, Optional[np.ndarray]]]:
    """
    Calculates WCSS and silhouette scores for a range of K values.

    Args:
        X: The input data (features) to cluster.
        range_n_clusters: An iterable of integers representing the K values to test.

    Returns:
        A tuple containing:
        - A list of WCSS scores for each K.
        - A dictionary mapping each K to its average silhouette score.
        - A dictionary mapping each K to the resulting cluster labels.
    """
    wcss_results: List[float] = []
    silhouette_results: Dict[int, float] = {}
    labels_results: Dict[int, Optional[np.ndarray]] = {}

    if not hasattr(X, "shape"):
        raise TypeError("Input data X must be array-like.")
    n_samples = X.shape[0]

    if n_samples <= 1:
        logger.warning("Cannot calculate WCSS/Silhouette with <= 1 sample.")
        k_values = list(range_n_clusters)
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

        try:
            clusterer = perform_clustering(X, n_clusters)
            wcss_results.append(cast(float, clusterer.inertia_))
            cluster_labels = clusterer.labels_
            labels_results[n_clusters] = cluster_labels

            if 1 < n_clusters < n_samples:
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_results[n_clusters] = silhouette_avg
            else:
                silhouette_results[n_clusters] = np.nan

        except (ValueError, Exception) as e:
            logger.info(f"Skipping K={n_clusters} due to clustering/scoring error: {e}")
            wcss_results.append(np.nan)
            silhouette_results[n_clusters] = np.nan
            labels_results[n_clusters] = None

    return wcss_results, silhouette_results, labels_results


def filter_silhouette_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Filters silhouette scores to start the analysis from the first peak.

    This helps the drop-off analysis focus on the most promising region of K values.

    Args:
        scores: A dictionary mapping K values to their silhouette scores.

    Returns:
        A sorted dictionary of filtered scores, starting from the first K with the max score.
    """
    if not scores:
        return {}

    try:
        score_df = pd.DataFrame(list(scores.items()), columns=["k_value", "score"])  # type: ignore
        score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
        score_df.dropna(subset=["score"], inplace=True)
    except Exception as e:
        logger.error(f"Error processing scores DataFrame: {e}")
        return {}

    if score_df.empty:
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
            f"Could not determine peak K for silhouette scores: {e}. Using all valid scores."
        )
        return dict(sorted(score_df.set_index("k_value")["score"].to_dict().items()))

    filtered_scores = {
        k: v
        for k, v in scores.items()
        if isinstance(k, int) and k >= max_score_k and pd.notna(v)
    }
    return dict(sorted(filtered_scores.items()))


def find_optimal_k_values(
    all_valid_scores: Dict[str, Dict[int, float]], percent_drop_threshold: float
) -> Dict[str, int]:
    """
    Determines the optimal K for each segment based on a silhouette score drop-off.

    Args:
        all_valid_scores: A dictionary where keys are segment names and values are dicts of {k: score}.
        percent_drop_threshold: The maximum allowed percentage drop in score to continue increasing K.

    Returns:
        A dictionary mapping each segment name to its determined optimal K.
    """
    optimal_k_per_key: Dict[str, int] = {}
    for combination_key, scores_dict in all_valid_scores.items():
        if not scores_dict:
            continue

        sorted_k = sorted(scores_dict.keys())
        optimal_k_for_this_key = sorted_k[0]

        for i in range(1, len(sorted_k)):
            current_k, previous_k = sorted_k[i], sorted_k[i - 1]
            current_score, previous_score = (
                scores_dict[current_k],
                scores_dict[previous_k],
            )

            drop = (
                (previous_score - current_score) / abs(previous_score)
                if previous_score != 0
                else (1.0 if current_score < 0 else 0.0)
            )

            if drop <= percent_drop_threshold:
                optimal_k_for_this_key = current_k
            else:
                logger.info(
                    f"  Stopping at K={current_k} for {combination_key}: Drop ({drop:.2%}) > Threshold ({percent_drop_threshold:.2%})"
                )
                break

        optimal_k_per_key[combination_key] = int(optimal_k_for_this_key)

    if optimal_k_per_key:
        logger.info(f"Determined optimal K values per combination: {optimal_k_per_key}")
    return optimal_k_per_key


def perform_clustering_and_aggregation(
    df_aggregated_by_combination: pd.DataFrame, combination_cols: List[str], k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs final clustering on combination profiles and aggregates results.

    Args:
        df_aggregated_by_combination: DataFrame of aggregated combination profiles.
        combination_cols: List of columns representing the combination features.
        k: The final number of clusters to use.

    Returns:
        A tuple containing:
        - A DataFrame with cluster profiles aggregated by the final cluster ID.
        - A DataFrame mapping original combinations to their final cluster ID.
    """
    if df_aggregated_by_combination.empty:
        raise ValueError("Input DataFrame is empty.")

    X = df_aggregated_by_combination[combination_cols].copy()
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
        empty_agg = pd.DataFrame(columns=[CLUSTER_COL] + combination_cols)  # type: ignore
        empty_map = pd.DataFrame(
            columns=df_aggregated_by_combination.columns.tolist() + [CLUSTER_COL]  # type: ignore
        )
        return empty_agg, empty_map


def calculate_hdr_for_cluster(
    cluster_data: pd.Series,
    n_classes: int,
    hdr_threshold_percentage: float,
    bin_edges: np.ndarray,
    unit: str,
) -> Dict:
    """
    Calculates the single most significant High-Density Region (HDR) for a cluster.

    This identifies the contiguous block of bins with the highest data concentration
    (mass) that is above a calculated density threshold.

    Args:
        cluster_data: A Series representing a single cluster's distribution profile.
        n_classes: The total number of bins.
        hdr_threshold_percentage: The percentage of data to exclude from the HDR (e.g., 90.0 for top 10%).
        bin_edges: The edges of the bins for converting back to original units.
        unit: The unit of the original measurement for annotation.

    Returns:
        A dictionary containing the HDR threshold count and a list with the single most significant interval.
    """
    c_cols = cluster_data.filter(like=QUANT_PREFIX)
    if c_cols.empty:
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    c_cols.index = c_cols.index.str.replace(QUANT_PREFIX, "").astype(int)
    counts_per_bin = c_cols.reindex(range(1, n_classes + 1), fill_value=0.0)
    total_cluster_count = counts_per_bin.sum()

    if total_cluster_count == 0:
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    # Determine the y-axis line for the HDR threshold
    target_hdr_mass = total_cluster_count * (100.0 - hdr_threshold_percentage) / 100.0
    if target_hdr_mass <= 0:
        return {"hdr_threshold_count": counts_per_bin.max() + 1, "hdr_intervals": []}

    positive_counts = counts_per_bin[counts_per_bin > 0]
    if positive_counts.empty:  # type: ignore
        return {"hdr_threshold_count": 0.0, "hdr_intervals": []}

    # Find the highest y-value (density) that captures at least the target mass
    best_y_line, min_overshoot = 0.0, float("inf")
    for y_candidate in range(int(positive_counts.max()), -1, -1):
        mass_above_line = (positive_counts - y_candidate).clip(lower=0).sum()  # type: ignore
        if mass_above_line >= target_hdr_mass and mass_above_line < min_overshoot:
            min_overshoot, best_y_line = mass_above_line, float(y_candidate)

    hdr_line_y_value = best_y_line
    hdr_bins = counts_per_bin[counts_per_bin > hdr_line_y_value]

    if hdr_bins.sum() < HDR_THRESHOLD_MIN_COUNT:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    # Find all contiguous blocks of bins above the threshold
    hdr_indices = np.array(sorted(hdr_bins.index.tolist()))  # type: ignore
    if hdr_indices.size == 0:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    splits = np.where(np.diff(hdr_indices) != 1)[0] + 1
    consecutive_blocks = np.split(hdr_indices, splits)

    # Find the single block with the most mass
    best_interval, max_mass = None, -1.0
    for block in consecutive_blocks:
        if block.size > 0:
            start, end = block[0], block[-1]
            interval_mass = counts_per_bin.loc[start:end].sum()
            if interval_mass > max_mass:
                max_mass, best_interval = interval_mass, (start, end)

    if best_interval is None:
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    # Format the final result for the best interval
    start, end = best_interval
    if not (0 <= start - 1 < len(bin_edges) and 0 <= end < len(bin_edges)):
        logger.warning(
            f"Best interval bin edges out of bounds for [{start}, {end}]. Skipping."
        )
        return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": []}

    interval_counts = counts_per_bin.loc[start:end]
    peak_bin = interval_counts.idxmax()

    final_hdr_data = {
        "start_bin_label": int(start),
        "end_bin_label": int(end),
        "median_bin_label": (start + end) / 2.0,
        "peak_bin_label": int(peak_bin),
        "start_edge_orig_unit": float(bin_edges[start - 1]),
        "end_edge_orig_unit": float(bin_edges[end]),
        "most_probable_orig_unit": float(
            (bin_edges[peak_bin - 1] + bin_edges[peak_bin]) / 2.0
        ),
        "unit": unit,
        "total_mass_in_interval": float(max_mass),
        "peak_count": float(interval_counts.max()),
    }

    return {"hdr_threshold_count": hdr_line_y_value, "hdr_intervals": [final_hdr_data]}
