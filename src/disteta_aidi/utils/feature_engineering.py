# This module provides functions for feature engineering, including categorical
# encoding, data quantization, and aggregation, which are essential steps in
# preparing the data for distributional analysis.

import logging
import re
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from kneed import KneeLocator

from .constants import CLASS_COL, COMB_COL, QUANT_PREFIX

logger = logging.getLogger(__name__)


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    grouping_column: Optional[str] = None,
    continuous_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Encodes specified categorical columns using one-hot encoding.

    This function takes a DataFrame, applies pandas.get_dummies to the specified
    categorical columns, and returns a new DataFrame with the original continuous
    and grouping columns preserved alongside the new encoded columns.

    Args:
        df: The input DataFrame.
        categorical_columns: A list of column names to be one-hot encoded.
        grouping_column: An optional column name to preserve at the start of the output.
        continuous_columns: A list of continuous columns to preserve.

    Returns:
        A new DataFrame with categorical columns encoded.
    """
    df_to_encode = df.copy()
    continuous_columns = continuous_columns or []
    valid_categorical = [
        col for col in categorical_columns if col in df_to_encode.columns
    ]
    if not valid_categorical:
        logger.warning("No valid categorical columns found for encoding.")
        cols_to_keep = [
            col
            for col in ([grouping_column] + continuous_columns)
            if col and col in df.columns
        ]
        result_df = (
            df_to_encode[cols_to_keep].copy() if cols_to_keep else pd.DataFrame()
        )
        return cast(pd.DataFrame, result_df)
    encoded_df = pd.get_dummies(df_to_encode, columns=valid_categorical, dtype=float)
    final_order = []
    if grouping_column and grouping_column in encoded_df.columns:
        final_order.append(grouping_column)
    final_order.extend([col for col in continuous_columns if col in encoded_df.columns])
    cat_prefixes = tuple(f"{cat_col}_" for cat_col in valid_categorical)
    final_order.extend(
        sorted([col for col in encoded_df.columns if col.startswith(cat_prefixes)])
    )
    return cast(pd.DataFrame, encoded_df[list(dict.fromkeys(final_order))])


def get_encoded_column_names(
    df: pd.DataFrame, original_categorical_columns: List[str]
) -> List[str]:
    """
    Retrieves the names of columns created by one-hot encoding.

    Given a DataFrame and the original list of categorical columns, this function
    identifies and returns the names of the dummified columns based on their prefixes.

    Args:
        df: The DataFrame containing the encoded columns.
        original_categorical_columns: The list of original categorical column names.

    Returns:
        A sorted list of the dummified column names.
    """
    if not original_categorical_columns:
        return []
    prefixes = tuple(f"{col}_" for col in original_categorical_columns)
    encoded_columns = [col for col in df.columns if col.startswith(prefixes)]
    return sorted(encoded_columns)


def generate_label(
    row: pd.Series, combination_columns: List[str], label_mapping_dict: dict
) -> int:
    """
    Generates a unique integer label for each distinct row-wise combination of values.

    This function is used to create a single identifier ('comb') for each unique
    categorical profile in the data, using a memoization dictionary for efficiency.

    Args:
        row: A row of a DataFrame (as a pandas Series).
        combination_columns: The list of columns that define a unique combination.
        label_mapping_dict: A dictionary used for memoization to store and retrieve labels.

    Returns:
        An integer label for the combination.
    """
    combination = tuple(row[col] for col in combination_columns)
    if combination not in label_mapping_dict:
        label_mapping_dict[combination] = len(label_mapping_dict) + 1
    return label_mapping_dict[combination]


def calculate_combination_threshold(
    df: pd.DataFrame, threshold_config: str | int
) -> int:
    """
    Calculates the minimum size threshold for feature combinations.

    This function determines the minimum number of occurrences a combination must have
    to be included in the analysis. It supports several methods specified via a
    configuration string: a direct integer, a percentile (e.g., 'p20'), or an
    automatic knee-point detection ('auto-knee').

    Args:
        df: The DataFrame containing the 'comb' column.
        threshold_config: The configuration string or integer for the threshold.

    Returns:
        The calculated integer threshold.
    """
    if isinstance(threshold_config, int):
        logger.info(
            f"Using manually specified min combination size: {threshold_config}"
        )
        return threshold_config

    if COMB_COL not in df.columns:
        logger.warning(
            f"'{COMB_COL}' not found for threshold calculation. Defaulting to 2."
        )
        return 2

    combination_sizes = df.groupby(COMB_COL).size()
    if combination_sizes.empty:
        logger.warning("No combinations found. Defaulting to a threshold of 2.")
        return 2

    if threshold_config.lower().startswith("p"):
        try:
            percentile_value = int(threshold_config[1:])
            if not (0 <= percentile_value <= 100):
                raise ValueError("Percentile must be between 0 and 100.")

            threshold = np.percentile(combination_sizes, percentile_value)
            final_threshold = max(2, int(np.ceil(threshold)))
            logger.info(
                f"Calculated {percentile_value}th percentile threshold: {final_threshold}"
            )
            return final_threshold
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid percentile format: '{threshold_config}'. Use 'pXX' (e.g., 'p20')."
            ) from e

    elif threshold_config.lower().startswith("auto-knee"):
        sensitivity = 1.0
        match = re.search(r"S=([\d.]+)", threshold_config, re.IGNORECASE)
        if match:
            try:
                sensitivity = float(match.group(1))
            except ValueError:
                logger.warning(
                    f"Could not parse sensitivity from '{threshold_config}'. Using default S=1.0."
                )

        logger.info(f"Using 'auto-knee' method with sensitivity S={sensitivity}.")
        sorted_sizes = combination_sizes.sort_values(ascending=False).values  # type: ignore

        if len(sorted_sizes) < 3:
            logger.warning(
                "Not enough unique combinations (< 3) to find a knee. Defaulting to 2."
            )
            return 2

        x_values = range(len(sorted_sizes))
        kneedle = KneeLocator(
            x_values,
            sorted_sizes,
            S=sensitivity,
            curve="convex",
            direction="decreasing",
        )

        if kneedle.knee is not None:
            threshold_value = sorted_sizes[kneedle.knee]
            final_threshold = max(2, int(threshold_value))
            logger.info(
                f"Found knee point at rank {kneedle.knee}, corresponding to a minimum size of {final_threshold}."
            )
            return final_threshold
        else:
            logger.warning(
                f"Could not find a knee with S={sensitivity}. Falling back to p25 percentile."
            )
            return calculate_combination_threshold(df, "p25")

    raise ValueError(f"Invalid threshold configuration string: {threshold_config}.")


def calculate_optimal_bins(data: pd.Series) -> int:
    """
    Calculates the optimal number of bins for a histogram using the Freedman-Diaconis rule.

    This method is robust to outliers and adapts to the data's size and
    interquartile range to determine a suitable bin count for quantization.

    Args:
        data: A pandas Series of numeric data.

    Returns:
        The calculated optimal number of bins as an integer.
    """
    data = data.dropna()
    n = len(data)
    if n < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        bin_width = 2 * iqr / (n ** (1 / 3))
        if bin_width > 0:
            num_bins = (data.max() - data.min()) / bin_width
            return int(np.ceil(num_bins))
    num_bins = 1 + np.log2(n)
    return int(np.ceil(num_bins))


def quantize_and_dummify(
    df: pd.DataFrame,
    column_name: str,
    n_classes: int,
    labels: Sequence[int] | Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Quantizes a continuous column into discrete bins and then creates dummy variables.

    This function first uses pandas.cut to discretize a numeric column and then
    applies one-hot encoding to the resulting bins, preparing the data for
    distributional analysis.

    Args:
        df: The input DataFrame.
        column_name: The name of the continuous column to process.
        n_classes: The number of bins to create.
        labels: The labels for the bins.

    Returns:
        A tuple containing the processed DataFrame with new dummy columns and an
        array of the calculated bin edges.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric.")
    if not isinstance(n_classes, int) or n_classes <= 0:
        raise ValueError(f"n_classes must be a positive integer, got {n_classes}.")
    if len(labels) != n_classes:
        raise ValueError(
            f"Labels length ({len(labels)}) must match n_classes ({n_classes})."
        )
    df_processed = df.copy()
    try:
        df_processed[CLASS_COL], actual_bin_edges = pd.cut(
            df_processed[column_name],
            bins=n_classes,
            labels=labels,
            include_lowest=True,
            retbins=True,
            duplicates="drop",
        )
        df_processed[CLASS_COL] = df_processed[CLASS_COL].astype("category")
    except Exception as e:
        logger.error(f"Error during pd.cut for column '{column_name}': {e}")
        raise
    try:
        dummies = pd.get_dummies(
            df_processed[CLASS_COL],
            dtype=float,
            prefix=QUANT_PREFIX,
            prefix_sep="",
            dummy_na=False,
        )
        df_processed = pd.concat([df_processed, dummies], axis=1)
    except Exception as e:
        logger.error(
            f"Error during pd.get_dummies for '{CLASS_COL}' (from '{column_name}'): {e}"
        )
        raise
    return df_processed, cast(np.ndarray, actual_bin_edges)


def aggregate_by_comb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the dummified quantized features by the combination label.

    This function groups the DataFrame by the 'comb' column and sums the values
    of the quantized feature columns (prefixed with 'c_'), creating a single
    profile for each unique combination.

    Args:
        df: The DataFrame containing 'comb' and dummified 'c_*' columns.

    Returns:
        An aggregated DataFrame where each row represents a unique combination profile.
    """
    if COMB_COL not in df.columns:
        logger.error(f"'{COMB_COL}' column not found for aggregation.")
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=np.number).columns
    combination_cols = [col for col in numeric_cols if col.startswith(QUANT_PREFIX)]
    if not combination_cols:
        logger.warning(f"No numeric '{QUANT_PREFIX}*' columns found for aggregation.")
        result_df = df[[COMB_COL]].drop_duplicates().reset_index(drop=True)
        return cast(pd.DataFrame, result_df)
    try:
        df_aggregated = (
            df.groupby(COMB_COL, observed=True)[combination_cols].sum().reset_index()
        )
        return df_aggregated
    except Exception as e:
        logger.error(f"Error during aggregation by '{COMB_COL}': {e}")
        return pd.DataFrame()
