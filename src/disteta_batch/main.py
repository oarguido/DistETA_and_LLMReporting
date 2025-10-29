"""
This module, part of the DistETA project, is the core analysis pipeline.

It defines the `DistetaBatch` class, which orchestrates a configurable workflow
for analyzing distributional data. The key responsibilities of this module are:

1.  **Data Loading and Preparation**: Ingests data and prepares it for analysis.
2.  **Data Segmentation**: Groups data into meaningful segments for targeted analysis.
3.  **Quantization and Clustering**: Discretizes continuous features and uses
    K-Means clustering to identify dominant distributional patterns.
4.  **Artifact Generation**: Produces a comprehensive set of outputs, including
    processed data, JSON summaries, and visualizations.

The analysis is driven by a central YAML configuration file, allowing for
flexible and repeatable execution. The main entry point is the `run_all_analyses`
function, which can be called from other scripts or executed directly.

Execution:
    To run the analysis, execute this module as a script:
    $ python -m src.disteta_batch.main
"""

# =============================================================================
# HEADER (Imports, Constants, Logger)
# =============================================================================
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .. import constants
from .utils.clustering_utils import (
    calculate_hdr_for_cluster,
    calculate_wcss_and_silhouette,
    filter_silhouette_scores,
    find_optimal_k_values,
    perform_clustering_and_aggregation,
)
from .utils.constants import (
    AGG_DF_PREFIX,
    ALL_DATA_GROUP_KEY,
    CFG_CLUSTERING_MAX_K,
    CFG_CLUSTERING_MIN_K,
    CFG_COLS_CATEGORICAL,
    CFG_COLS_CONTINUOUS,
    CFG_COLS_GROUPING,
    CFG_COLS_UNITS,
    CFG_FILTERING_VALUES,
    CFG_HDR_THRESHOLD,
    CFG_IO_INPUT_DATA_PATH,
    CFG_MAPPING_COLUMN,
    CFG_MAPPING_VALUE,
    CFG_PLOTTING_QUANT_LABEL,
    CFG_PLOTTING_X_LABEL,
    CFG_PREPROC_MIN_COMB_SIZE,
    CFG_PREPROC_N_CLASSES,
    CFG_PREPROC_SILHOUETTE_DROP,
    CLUSTER_COL,
    COMB_COL,
    NAN_GROUP_KEY,
    QUANT_PREFIX,
)
from .utils.data_utils import identify_numeric_columns, load_config, load_data
from .utils.feature_engineering import (
    aggregate_by_comb,
    calculate_combination_threshold,
    calculate_optimal_bins,
    encode_categorical_features,
    get_encoded_column_names,
    quantize_and_dummify,
)
from .utils.plotting import (
    plot_cluster_distributions,
    plot_continuous_histograms,
    plot_silhouette_and_elbow,
)

# --- Constants ---
DEFAULT_CONFIG_PATH = os.path.join(
    constants.CONFIG_DIR, constants.DISTETA_CONFIG_FILENAME
)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Silence noisy third-party loggers
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("shutil").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATION MODEL (The AnalysisSettings dataclass)
# =============================================================================
@dataclass(frozen=True)
class AnalysisSettings:
    """Holds all configuration parameters for a single analysis run."""

    # Required from config
    input_path_resolved: str
    min_comb_size_input: Union[str, int]
    n_classes_input: Union[str, int]
    percent_drop_threshold_input: Union[int, float]

    # Optional with defaults
    categorical_cols: List[str] = field(default_factory=list)
    continuous_cols_config: List[str] = field(default_factory=list)
    grouping_col_config: Optional[str] = None
    filter_values_config: List[str] = field(default_factory=list)
    value_mapping: Dict = field(default_factory=dict)
    column_name_mapping: Dict = field(default_factory=dict)
    continuous_units_map: Dict = field(default_factory=dict)
    x_axis_label_config: str = "Value"
    quantized_axis_label_config: str = "Quantized Bin"
    hdr_threshold_percentage_config: float = 90.0
    clustering_min_k: int = 2
    clustering_max_k: int = 10

    @classmethod
    def from_dict(cls, config: dict) -> "AnalysisSettings":
        """Factory method to create an instance from a config dictionary."""
        CONFIG_PARAMS_MAP = {
            # attribute_name: (config_path, default_value, is_required)
            "input_path_resolved": (CFG_IO_INPUT_DATA_PATH, None, True),
            "categorical_cols": (CFG_COLS_CATEGORICAL, [], False),
            "continuous_cols_config": (CFG_COLS_CONTINUOUS, [], False),
            "grouping_col_config": (CFG_COLS_GROUPING, None, False),
            "filter_values_config": (CFG_FILTERING_VALUES, [], False),
            "min_comb_size_input": (CFG_PREPROC_MIN_COMB_SIZE, None, True),
            "n_classes_input": (CFG_PREPROC_N_CLASSES, None, True),
            "percent_drop_threshold_input": (CFG_PREPROC_SILHOUETTE_DROP, None, True),
            "value_mapping": (CFG_MAPPING_VALUE, {}, False),
            "column_name_mapping": (CFG_MAPPING_COLUMN, {}, False),
            "continuous_units_map": (CFG_COLS_UNITS, {}, False),
            "x_axis_label_config": (CFG_PLOTTING_X_LABEL, "Value", False),
            "quantized_axis_label_config": (
                CFG_PLOTTING_QUANT_LABEL,
                "Quantized Bin",
                False,
            ),
            "hdr_threshold_percentage_config": (CFG_HDR_THRESHOLD, 90.0, False),
            "clustering_min_k": (CFG_CLUSTERING_MIN_K, 2, False),
            "clustering_max_k": (CFG_CLUSTERING_MAX_K, 10, False),
        }

        def get_cfg(path, default=None):
            keys = path.split(".")
            val = config
            for key in keys:
                if not isinstance(val, dict):
                    return default
                val = val.get(key)
                if val is None:
                    return default
            return val

        kwargs = {}
        for attr, (path, default, required) in CONFIG_PARAMS_MAP.items():
            value = get_cfg(path, default)
            if required and value is None:
                raise ValueError(f"'{path}' is a required configuration key.")
            # We only pass non-None values to the constructor to allow dataclass defaults to work
            if value is not None:
                kwargs[attr] = value

        return cls(**kwargs)


# =============================================================================
# MAIN ANALYSIS CLASS (The DistetaBatch class)
# =============================================================================
class DistetaBatch:
    """Orchestrates the DistETA (Distributional ETA) analysis pipeline.

    This class is the core of the analysis, executing a configurable workflow:
    1.  **Data Loading & Preparation**: Ingests data, identifies column types,
        and encodes categorical features.
    2.  **Data Segmentation**: Groups data for targeted analysis based on a
        specified column and filters.
    3.  **Quantization**: Discretizes continuous features into bins.
    4.  **Clustering**: Applies K-Means to find dominant distributional patterns,
        automatically determining the optimal number of clusters (K).
    5.  **High-Density Region (HDR) Calculation**: Summarizes each cluster's
        distribution by calculating its high-density region.
    6.  **Artifact Generation**: Produces a comprehensive set of outputs,
        including timestamped directories, processed data (CSVs), detailed
        JSON summaries, and interactive/static plots (HTML/PNG).

    The process is driven by a YAML configuration for flexibility and
    repeatability.

    Attributes:
        settings: An `AnalysisSettings` object with all run parameters.
        run_config_name: The name of the configuration block being executed.
        base_output_path: The root directory for all output folders.
        logger: The logger instance for this class.
        run_timestamp: The timestamp string for the current run.
        generated_figures: A list to store generated Plotly figures before saving.
    """

    def __init__(
        self,
        settings: AnalysisSettings,
        base_output_path: str,
        run_config_name: str = "unknown_run",
    ):
        self.settings = settings
        self.run_config_name = run_config_name
        self.base_output_path = base_output_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.generated_figures = []

    def _setup_output_directories(self):
        """
        Creates the unique, timestamped directory structure for the current run.

        The structure is:
        <base_output_path>/<timestamp>_run_<config_name>/
            - data/
            - graphics/
            - logs/
        """
        run_folder_name = f"{self.run_timestamp}_run_{self.run_config_name}"
        self.run_specific_output_dir = os.path.join(
            self.base_output_path, run_folder_name
        )
        self.data_output_path = os.path.join(
            self.run_specific_output_dir, constants.DATA_DIR_NAME
        )
        self.graphics_output_path = os.path.join(
            self.run_specific_output_dir, constants.GRAPHICS_DIR_NAME
        )
        self.logs_output_path = os.path.join(
            self.run_specific_output_dir, constants.LOGS_DIR_NAME
        )
        os.makedirs(self.data_output_path, exist_ok=True)
        os.makedirs(self.graphics_output_path, exist_ok=True)
        os.makedirs(self.logs_output_path, exist_ok=True)
        self.logger.info(
            f"Outputs for run '{self.run_config_name}' will be saved to: "
            f"{self.run_specific_output_dir}"
        )

    def _get_groups_to_process(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segments a DataFrame into groups based on the grouping column and filter values.

        This method first groups the DataFrame by the `grouping_col_config`.
        If `filter_values_config` is provided, only the specified groups are
        processed. It handles NaN values in the grouping column as a separate
        group, which can be targeted using the key 'NaN'. If no grouping column
        is specified, it returns a single group containing the entire DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be segmented.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are group names and
                                     values are the corresponding DataFrame segments.
        """
        grouping_col = self.settings.grouping_col_config

        if not grouping_col or grouping_col not in df.columns:
            return {ALL_DATA_GROUP_KEY: df.copy()}

        # Group by the specified column, including NaN values as a separate group
        grouped = df.groupby(grouping_col, dropna=False)

        # Get the names of all available groups from the groupby object
        available_group_keys = list(grouped.groups.keys())

        filter_list = self.settings.filter_values_config

        target_group_keys = []
        if not filter_list:
            # If no filter is specified, process all groups
            target_group_keys = available_group_keys
        else:
            # If a filter is specified, select only the groups present in the filter list.
            # The filter list uses the string "NaN" to represent null values.
            for key in available_group_keys:
                # The key from groupby for the NaN group is np.nan.
                # Use np.isnan for explicit NaN check on scalar floats, combined
                # with isinstance for type safety.
                if isinstance(key, float) and np.isnan(key):
                    if NAN_GROUP_KEY in filter_list:
                        target_group_keys.append(key)
                elif key in filter_list:
                    target_group_keys.append(key)

        groups_to_process = {}
        for key in target_group_keys:
            group_df = grouped.get_group(key)
            if not group_df.empty:
                # Use NAN_GROUP_KEY for the dictionary key if the group key is NaN.
                # Use np.isnan for explicit NaN check on scalar floats, combined
                # with isinstance for type safety.
                dict_key = (
                    NAN_GROUP_KEY
                    if isinstance(key, float) and np.isnan(key)
                    else str(key)
                )
                groups_to_process[dict_key] = group_df
        return groups_to_process

    def _prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Loads, prepares, and filters the initial DataFrame for analysis.

        This method performs several key steps:
        1.  Loads the data from the path specified in the settings.
        2.  Identifies the continuous columns to be analyzed.
        3.  Generates and saves initial distribution plots for these columns.
        4.  Encodes categorical features into a numerical format.
        5.  Creates a 'combination' column based on the encoded categorical features.
        6.  Filters the DataFrame to remove combinations with fewer members than
            the `min_comb_size_input` threshold.
        7.  Saves the filtered, pre-quantization DataFrame.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing the prepared
                                             DataFrame and the list of
                                             continuous columns to be analyzed.
        """
        df = load_data(self.settings.input_path_resolved)
        continuous_cols = (
            self.settings.continuous_cols_config
            or identify_numeric_columns(df, exclude_cols=self.settings.categorical_cols)
        )
        if not continuous_cols:
            raise ValueError("No continuous columns specified or auto-detected.")
        self.logger.info(f"Analyzing continuous columns: {continuous_cols}")

        self._plot_initial_distributions(df, continuous_cols)

        expanded_df = encode_categorical_features(
            df,
            self.settings.categorical_cols,
            self.settings.grouping_col_config,
            continuous_cols,
        )
        encoded_col_list = get_encoded_column_names(
            expanded_df, self.settings.categorical_cols
        )

        if encoded_col_list:
            # Vectorized approach to create combination labels, much faster than df.apply
            comb_series = (
                expanded_df[encoded_col_list].astype(str).agg("-".join, axis=1)
            )
            expanded_df[COMB_COL] = pd.factorize(comb_series)[0] + 1
        else:
            expanded_df[COMB_COL] = 1

        if COMB_COL in expanded_df.columns:
            self.logger.info("--- Analyzing Combination Size Distribution ---")
            combination_sizes = expanded_df.groupby(COMB_COL).size()
            self.logger.info(
                f"\n{combination_sizes.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])}"
            )
            self.logger.info("---------------------------------------------")

        min_size_threshold = calculate_combination_threshold(
            expanded_df, self.settings.min_comb_size_input
        )

        expanded_df_filtered = expanded_df[
            expanded_df.groupby(COMB_COL)[COMB_COL].transform("size")
            >= min_size_threshold
        ].copy()

        if expanded_df_filtered.empty:
            raise ValueError(
                f"No data meets the minimum combination size threshold of {min_size_threshold}."
            )

        retained_rows = len(expanded_df_filtered)
        total_rows = len(expanded_df)
        percent_retained = (retained_rows / total_rows * 100) if total_rows > 0 else 0
        self.logger.info(
            f"Filtering by min combination size of {min_size_threshold} "
            f"retained {retained_rows:,} of {total_rows:,} rows "
            f"({percent_retained:.2f}%)."
        )

        self._save_dataframe(
            cast(pd.DataFrame, expanded_df_filtered),
            "01_filtered_pre_quantization_data",
        )

        return cast(pd.DataFrame, expanded_df_filtered), continuous_cols

    def _quantize_and_aggregate_segments(
        self,
        df: pd.DataFrame,
        continuous_cols: List[str],  # noqa: E501
    ) -> Tuple[Dict, Dict]:
        """
        Quantizes continuous features and aggregates data for each segment.

        For each data group and each continuous feature, this method:
        1.  Determines the optimal number of bins for quantization (or uses the
            configured value).
        2.  Discretizes the continuous feature into these bins (quantization).
        3.  Dummifies the quantized feature, creating a one-hot encoded representation.
        4.  Aggregates the dummified data by the 'combination' column, creating
            a profile of distributions for each combination.
        5.  Stores the aggregated DataFrames and the quantization parameters
            (bin edges, units) for later use.

        Args:
            df (pd.DataFrame): The filtered DataFrame from the preparation step.
            continuous_cols (List[str]): The list of continuous columns to process.

        Returns:
            Tuple[Dict, Dict]: A tuple containing:
                               - A dictionary of aggregated DataFrames for each segment.
                               - A dictionary of quantization parameters for each segment.
        """
        groups_to_process = self._get_groups_to_process(df)
        if not groups_to_process:
            raise ValueError("No data groups left after filtering.")

        quantization_params, aggregated_dfs = {}, {}
        self.logger.info("\n--- Quantizing and Aggregating Data ---")
        for group_key, group_df in groups_to_process.items():
            for feature_col in continuous_cols:
                if (
                    feature_col not in group_df.columns
                    or group_df[feature_col].nunique(dropna=True) <= 1
                ):
                    continue

                n_classes_to_use: int
                if self.settings.n_classes_input == "auto":
                    series_to_quantize = cast(pd.Series, group_df[feature_col])
                    n_classes_to_use = calculate_optimal_bins(series_to_quantize)
                    self.logger.info(
                        f"For group '{group_key}', feature '{feature_col}': "
                        f"Auto-calculated optimal bins = {n_classes_to_use}"
                    )
                elif isinstance(self.settings.n_classes_input, int):
                    n_classes_to_use = self.settings.n_classes_input
                else:
                    self.logger.error(
                        f"Invalid value for 'quantization_n_classes_input': "
                        f"'{self.settings.n_classes_input}'. Must be 'auto' or an "
                        f"integer. Skipping processing for '{feature_col}' in "
                        f"group '{group_key}'."
                    )
                    continue

                if n_classes_to_use <= 1:
                    self.logger.warning(
                        f"Skipping '{feature_col}' for group '{group_key}': not enough "
                        f"data variance to create more than 1 bin."
                    )
                    continue

                quant_labels = [str(i) for i in range(1, n_classes_to_use + 1)]

                try:
                    df_quant, bin_edges = quantize_and_dummify(
                        group_df, feature_col, n_classes_to_use, quant_labels
                    )
                    agg_df = aggregate_by_comb(df_quant)
                    if not agg_df.empty:
                        segment_key = f"{AGG_DF_PREFIX}{group_key}_{feature_col}"
                        aggregated_dfs[segment_key] = agg_df
                        quantization_params[segment_key] = {
                            "bin_edges": bin_edges.tolist(),
                            "unit": self.settings.continuous_units_map.get(
                                feature_col, "units"
                            ),
                            "original_n_classes_input": n_classes_to_use,
                        }
                except (ValueError, TypeError) as e:
                    self.logger.error(
                        f"Data processing error for '{feature_col}' in group '{group_key}': {e}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"An unexpected error occurred while processing '{feature_col}' "
                        f"in group '{group_key}': {e}",
                        exc_info=True,
                    )

        if not aggregated_dfs:
            raise ValueError("No data to cluster after aggregation.")
        return aggregated_dfs, quantization_params

    def _find_and_plot_optimal_k(
        self, aggregated_dfs: Dict, continuous_cols: List[str]
    ) -> Tuple[Dict, Dict]:
        """
        Performs silhouette analysis to find and plot the optimal K for clustering.

        For each aggregated data segment, this method iterates through a range of
        possible K values (number of clusters):
        1.  Calculates the Within-Cluster Sum of Squares (WCSS) for the elbow method.
        2.  Calculates the silhouette score for each K.
        3.  Generates and saves plots showing the silhouette score and elbow curve
            for each K, providing a visual aid for K-selection.
        4.  Automatically determines the optimal K based on a configurable
            silhouette score drop-off threshold.

        Args:
            aggregated_dfs (Dict): The dictionary of aggregated DataFrames.
            continuous_cols (List[str]): The list of continuous columns.

        Returns:
            Tuple[Dict, Dict]: A tuple containing:
                               - A dictionary of the determined optimal K for each segment.
                               - A dictionary of all calculated silhouette scores for each segment.
        """
        self.logger.info("\n--- Performing Clustering Analysis (Elbow/Silhouette) ---")
        all_silh_scores, wcss_data, labels_data = {}, {}, {}

        if not aggregated_dfs:
            self.logger.warning(
                "No aggregated data to perform clustering on. Skipping K-selection."
            )
            return {}, {}

        for df_name, df_agg in aggregated_dfs.items():
            combination_cols = sorted(
                [c for c in df_agg.columns if c.startswith(QUANT_PREFIX)],
                key=lambda x: int(x.split(QUANT_PREFIX)[1]),
            )
            if not combination_cols:
                self.logger.warning(
                    f"No '{QUANT_PREFIX}*' columns in segment {df_name}. Skipping."
                )
                continue

            X = df_agg[combination_cols]
            max_k_adj = min(self.settings.clustering_max_k, X.shape[0] - 1)
            min_k_adj = max(2, self.settings.clustering_min_k)
            if min_k_adj > max_k_adj:
                self.logger.warning(
                    f"Not enough samples in {df_name} to test K range. Skipping."
                )
                continue

            k_range = range(min_k_adj, max_k_adj + 1)
            wcss, silh, labels = calculate_wcss_and_silhouette(X, k_range)
            all_silh_scores[df_name], wcss_data[df_name], labels_data[df_name] = (
                silh,
                wcss,
                labels,
            )
            group_key, feature_name = self._parse_group_and_feature(
                df_name.replace(AGG_DF_PREFIX, ""),
                self.settings.grouping_col_config,
                continuous_cols,
            )
            clustering_metrics = {
                "wcss_data": wcss_data,
                "k_range": k_range,
                "all_silh_scores": all_silh_scores,
                "labels_data": labels_data,
            }
            for k in k_range:
                if k in labels_data[df_name] and labels_data[df_name][k] is not None:
                    self._plot_silhouette_analysis(
                        X,
                        k,
                        df_name,
                        group_key,
                        feature_name,
                        continuous_cols,
                        clustering_metrics,
                    )
        optimal_k = find_optimal_k_values(
            {k: filter_silhouette_scores(v) for k, v in all_silh_scores.items()},
            self.settings.percent_drop_threshold_input / 100.0,
        )
        return optimal_k, all_silh_scores

    def _perform_final_clustering_and_hdr(
        self,
        aggregated_dfs: Dict,
        optimal_k_values: Dict,
        df_filtered: pd.DataFrame,
        continuous_cols: List[str],
        quantization_params: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Performs final clustering and calculates High-Density Regions (HDR).

        Using the optimal K value determined for each segment, this method:
        1.  Runs K-Means clustering on the aggregated data.
        2.  Creates a mapping from each original 'combination' to its assigned cluster.
        3.  Generates a summary profile for each final cluster.
        4.  Saves the combination-to-cluster mapping and the final cluster profiles.
        5.  Analyzes the composition of each cluster in terms of the original
            categorical features.
        6.  Calculates the High-Density Region (HDR) for each cluster's distribution,
            identifying the range of bins that contains a specified percentage of
            the distribution's mass.

        Args:
            aggregated_dfs (Dict): The dictionary of aggregated DataFrames.
            optimal_k_values (Dict): The dictionary of optimal K values for each segment.
            df_filtered (pd.DataFrame): The filtered DataFrame from the preparation step.
            continuous_cols (List[str]): The list of continuous columns.
            quantization_params (Dict): The dictionary of quantization parameters.

        Returns:
            Tuple[Dict, Dict]: A tuple containing:
                               - A dictionary of the final cluster profile DataFrames.
                               - A dictionary containing the HDR results for each cluster.
        """
        self.logger.info("\n--- Performing Final Clustering with Optimal K ---")
        final_cluster_profiles, hdr_results = {}, {}
        for df_name, df_agg in aggregated_dfs.items():
            if df_name not in optimal_k_values:
                self.logger.warning(
                    f"No optimal K found for {df_name}, skipping final clustering."
                )
                continue
            k_final = optimal_k_values[df_name]
            combination_cols = sorted(
                [c for c in df_agg.columns if c.startswith(QUANT_PREFIX)],
                key=lambda x: int(x.split(QUANT_PREFIX)[1]),
            )
            df_sum, df_map = perform_clustering_and_aggregation(
                df_agg, combination_cols, k_final
            )
            final_df_key = df_name.replace(AGG_DF_PREFIX, "")
            final_cluster_profiles[final_df_key] = df_sum
            self._save_dataframe(
                df_map, f"02_combination_to_cluster_mapping_{final_df_key}"
            )
            self._save_dataframe(df_sum, f"03_final_cluster_profiles_{final_df_key}")
            self._analyze_and_save_cluster_composition(
                df_filtered, df_map, final_df_key, continuous_cols
            )

            hdr_results[final_df_key] = {}
            quant_params_for_segment = quantization_params.get(df_name, {})
            bin_edges, unit, n_classes = (
                np.array(quant_params_for_segment.get("bin_edges", [])),
                quant_params_for_segment.get("unit", "units"),
                quant_params_for_segment.get("original_n_classes_input", 0),
            )
            if n_classes == 0:
                self.logger.warning(f"No n_classes info for {df_name}, skipping HDR.")
                continue

            for _, row in df_sum.iterrows():
                cluster_id = str(int(row[CLUSTER_COL]))
                hdr_results[final_df_key][cluster_id] = calculate_hdr_for_cluster(
                    row,
                    n_classes,
                    self.settings.hdr_threshold_percentage_config,
                    bin_edges,
                    unit,
                )
        return final_cluster_profiles, hdr_results

    def _plot_final_cluster_distributions(
        self,
        final_cluster_profiles: Dict,
        continuous_cols: List[str],
        quantization_params: Dict,
        hdr_results: Dict,
    ):
        """
        Generates and saves plots of the final cluster distributions.

        For each segment, this method creates a plot showing the distribution
        (as a bar chart of quantized bins) for each of the identified clusters.
        It also overlays the calculated High-Density Region (HDR) on each
        distribution.

        Args:
            final_cluster_profiles (Dict): The dictionary of final cluster profile DataFrames.
            continuous_cols (List[str]): The list of continuous columns.
            quantization_params (Dict): The dictionary of quantization parameters.
            hdr_results (Dict): The dictionary of HDR results.
        """
        self.logger.info("\n--- Plotting Final Cluster Distributions ---")
        for final_df_name, df_final_sum in final_cluster_profiles.items():
            group_key, feature_name = self._parse_group_and_feature(
                final_df_name, self.settings.grouping_col_config, continuous_cols
            )
            df_name_key = f"{AGG_DF_PREFIX}{final_df_name}"
            quant_params_for_plot = quantization_params.get(df_name_key, {})
            n_classes_for_plot = quant_params_for_plot.get(
                "original_n_classes_input", 20
            )

            fig = plot_cluster_distributions(
                df_cluster_c_sum=df_final_sum,
                df_name_suffix=final_df_name,
                group_key=group_key,
                feature_name=feature_name,
                n_classes=n_classes_for_plot,
                value_mapping_config=self.settings.value_mapping,
                grouping_column_name=self.settings.grouping_col_config,
                column_name_mapping=self.settings.column_name_mapping,
                x_axis_label=self.settings.quantized_axis_label_config,
                hdr_info_for_plot=hdr_results.get(final_df_name, {}),
                hdr_threshold_percentage=(
                    self.settings.hdr_threshold_percentage_config
                ),
                save_non_interactive=True,
            )
            if fig:
                self.generated_figures.append(
                    (fig, f"03_cluster_distributions_{final_df_name}")
                )

    def run_analysis(self):
        """Executes the full DistetaBatch analysis workflow."""
        self._setup_output_directories()
        log_filepath = os.path.join(
            self.logs_output_path, f"analysis_log_{self.run_config_name}.log"
        )
        done_filepath = os.path.join(constants.OUTPUT_DIR, ".analysis_done")

        if os.path.exists(done_filepath):
            os.remove(done_filepath)

        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self.logger.info(f"Logging for this run is being saved to: {log_filepath}")
        try:
            start_time = time.time()
            self.logger.info(
                f"--- Starting DistetaBatch Analysis for config '{self.run_config_name}' ---"
            )
            df_filtered, continuous_cols = self._prepare_data()
            aggregated_dfs, quantization_params = self._quantize_and_aggregate_segments(
                df_filtered, continuous_cols
            )
            optimal_k_values, all_silh_scores = self._find_and_plot_optimal_k(
                aggregated_dfs, continuous_cols
            )
            final_cluster_profiles, hdr_results = (
                self._perform_final_clustering_and_hdr(
                    aggregated_dfs,
                    optimal_k_values,
                    df_filtered,
                    continuous_cols,
                    quantization_params,
                )
            )
            self._plot_final_cluster_distributions(
                final_cluster_profiles,
                continuous_cols,
                quantization_params,
                hdr_results,
            )

            # Save all data artifacts (JSONs, CSVs are saved elsewhere)
            self._save_json_artifacts(
                continuous_cols,
                optimal_k_values,
                all_silh_scores,
                final_cluster_profiles,
                hdr_results,
                quantization_params,
            )
            # Save all generated figures to disk
            self._save_all_generated_figures()
            end_time = time.time()
            self.logger.info(
                f"\nAnalysis for config '{self.run_config_name}' complete. "
                f"Total time: {end_time - start_time:.2f}s."
            )

            with open(done_filepath, "w") as f:
                f.write(self.run_specific_output_dir)

        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

    def _plot_initial_distributions(self, df: pd.DataFrame, continuous_cols: List[str]):
        """Helper to generate and store the initial data distribution histograms."""
        plot_groups_df = (
            pd.concat(self._get_groups_to_process(df).values())
            if self.settings.grouping_col_config
            else df
        )
        fig_hist = plot_continuous_histograms(
            df=plot_groups_df,
            continuous_columns=continuous_cols,
            grouping_column_name=self.settings.grouping_col_config,
            value_mapping_config=self.settings.value_mapping,
            column_name_mapping=self.settings.column_name_mapping,
            x_axis_label=self.settings.x_axis_label_config,
            save_non_interactive=True,
        )
        if fig_hist:
            self.generated_figures.append((fig_hist, "01_initial_distributions"))

    def _plot_silhouette_analysis(
        self,
        X,
        k,
        df_name,
        group_key,
        feature_name,
        continuous_cols,
        clustering_metrics: Dict,
    ):
        """Helper to plot the silhouette analysis for a given K."""
        fig_silh = plot_silhouette_and_elbow(
            X=X,
            cluster_labels=clustering_metrics["labels_data"][df_name][k],
            n_clusters=k,
            df_name_suffix=df_name.replace(AGG_DF_PREFIX, ""),
            group_key=group_key,
            feature_name=feature_name,
            wcss_values=clustering_metrics["wcss_data"][df_name],
            range_n_clusters=clustering_metrics["k_range"],
            silh_scores=clustering_metrics["all_silh_scores"][df_name],
            value_mapping_config=self.settings.value_mapping,
            grouping_column_name=self.settings.grouping_col_config,
            column_name_mapping=self.settings.column_name_mapping,
            save_non_interactive=True,
        )
        if fig_silh:
            self.generated_figures.append(
                (
                    fig_silh,
                    f"02_silhouette_elbow_{df_name.replace(AGG_DF_PREFIX, '')}_K{k}",
                )
            )

    def _save_all_generated_figures(self):
        """Saves all generated Plotly figures to both PNG and HTML files."""
        self.logger.info(f"Saving plots to '{self.graphics_output_path}'...")
        for fig, base_filename in self.generated_figures:
            safe_filename = base_filename.replace(" ", "_").replace("/", "-")

            # --- Save as PNG (static) ---
            png_filepath = os.path.join(
                self.graphics_output_path, f"{safe_filename}.png"
            )
            try:
                # Static PNGs require a fixed size for consistent layout in reports.
                fig.write_image(png_filepath, width=1280, height=720)
            except Exception as e:
                self.logger.error(
                    f"Failed to save PNG {png_filepath}: {e}. Ensure 'kaleido' "
                    f"is installed and up-to-date."
                )

            html_filepath = os.path.join(
                self.graphics_output_path, f"{safe_filename}.html"
            )
            try:
                # For interactive HTML, we remove fixed sizes and set autosize=True.
                # This allows the plot to fill its container (e.g., an iframe)
                # and be responsive in the final HTML report.
                fig.update_layout(width=None, height=None, autosize=True)
                fig.write_html(html_filepath, include_plotlyjs="cdn", full_html=False)
            except Exception as e:
                self.logger.error(f"Failed to save HTML {html_filepath}: {e}.")

    def _save_json_artifacts(
        self,
        continuous_cols,
        optimal_k,
        k_metrics,
        cluster_profiles,
        hdr_results,
        quant_params,
    ):
        """Saves all non-plot, non-CSV artifacts (JSON summaries) for the run."""
        self.logger.info(
            f"Saving JSON summary artifacts to '{self.logs_output_path}'..."
        )

        # Create the detailed summary dictionary for final cluster profiles
        final_profiles_for_json = {}
        for final_df_name, df_final_sum in cluster_profiles.items():
            if df_final_sum is not None and not df_final_sum.empty:
                final_profiles_for_json[final_df_name] = (
                    self._create_segment_profile_summary(
                        final_df_name,
                        df_final_sum,
                        optimal_k,
                        quant_params,
                        hdr_results,
                    )
                )

        k_selection_summary = {
            "optimal_k_values": optimal_k,
            "silhouette_scores_by_segment": k_metrics,
        }
        self._save_json(self._create_run_summary(continuous_cols), "run_summary.json")
        self._save_json(k_selection_summary, "k_selection_summary.json")
        self._save_json(final_profiles_for_json, "final_cluster_profiles.json")

    def _create_run_summary(self, continuous_cols):
        """Creates a dictionary summarizing the configuration for this run."""
        return {
            "run_timestamp": self.run_timestamp,
            "config_name_used": self.run_config_name,
            "input_data_path": self.settings.input_path_resolved,
            "categorical_columns_defined": self.settings.categorical_cols,
            "continuous_columns_analyzed": continuous_cols,
            "grouping_column_used": self.settings.grouping_col_config or "N/A",
            "filter_values_for_grouping_column": (
                self.settings.filter_values_config or "all_available_groups_processed"
            ),
            "min_combination_size_input": self.settings.min_comb_size_input,
            "quantization_n_classes_input": self.settings.n_classes_input,
            "clustering_k_range_tested": {
                "min_k_config": self.settings.clustering_min_k,
                "max_k_config": self.settings.clustering_max_k,
            },
            "silhouette_score_drop_threshold_input_percentage": (
                self.settings.percent_drop_threshold_input
            ),
            "hdr_threshold_percentage_input": (
                self.settings.hdr_threshold_percentage_config
            ),
        }

    def _parse_group_and_feature(
        self, base_name: str, grouping_col: Optional[str], continuous_cols: List[str]
    ) -> Tuple[Optional[str], str]:
        """Helper to parse group key and feature name from a segment name."""
        if grouping_col:
            for feature in continuous_cols:
                if base_name.endswith(f"_{feature}"):
                    return base_name[: -len(f"_{feature}")], feature
        if base_name.startswith(f"{ALL_DATA_GROUP_KEY}_"):
            return ALL_DATA_GROUP_KEY, base_name[len(ALL_DATA_GROUP_KEY) + 1 :]
        return ALL_DATA_GROUP_KEY, base_name

    def _save_dataframe(self, df: pd.DataFrame, base_filename: str):
        """Saves a DataFrame to a CSV file."""
        if df is None or df.empty:
            self.logger.warning(
                f"DataFrame for '{base_filename}' is empty. Skipping save."
            )
            return
        safe_filename = base_filename.replace(" ", "_").replace("/", "-") + ".csv"
        filepath = os.path.join(self.data_output_path, safe_filename)
        try:
            df.to_csv(filepath, index=False)
            self.logger.info(f"  Saved DataFrame: {filepath}")
        except Exception as e:
            self.logger.error(f"    Error saving DataFrame to {filepath}: {e}")

    def _analyze_and_save_cluster_composition(
        self,
        original_filtered_df: pd.DataFrame,
        combination_to_cluster_map: pd.DataFrame,
        segment_name: str,
        continuous_cols: List[str],
    ):
        """
        Analyzes cluster composition by inspecting the categorical features
        that define each combination. Saves a detailed CSV and a
        user-specified structured JSON.
        """
        self.logger.info(f"  Analyzing cluster composition for segment: {segment_name}")
        try:
            if (
                COMB_COL not in original_filtered_df.columns
                or CLUSTER_COL not in combination_to_cluster_map.columns
            ):
                self.logger.error(
                    f"Missing required columns for composition analysis in {segment_name}."
                )
                return

            df_with_clusters = original_filtered_df.merge(
                combination_to_cluster_map[[COMB_COL, CLUSTER_COL]],
                on=COMB_COL,
                how="left",
            )
            df_with_clusters.dropna(subset=[CLUSTER_COL], inplace=True)
            df_with_clusters[CLUSTER_COL] = df_with_clusters[CLUSTER_COL].astype(int)

            dummified_cols = get_encoded_column_names(
                df_with_clusters, self.settings.categorical_cols
            )
            if not dummified_cols:
                self.logger.warning(
                    f"No dummified categorical columns found for composition analysis "
                    f"in {segment_name}."
                )
                return

            # --- Prepare outputs for both JSON and CSV ---
            json_output = {}
            summary_list_for_csv = []

            for cluster_id_numpy in sorted(df_with_clusters[CLUSTER_COL].unique()):
                cluster_id = int(cluster_id_numpy)
                cluster_key = f"cluster_{cluster_id}"
                cluster_df = df_with_clusters[
                    df_with_clusters[CLUSTER_COL] == cluster_id
                ]
                total_rows_in_cluster = len(cluster_df)

                json_output[cluster_key] = {
                    "total_rows_in_cluster": total_rows_in_cluster,
                    "composition": {},
                }

                # --- 1. Process DUMMIFIED columns ---
                # These columns have values of 1 or 0. The sum is the count.
                composition_counts = cluster_df[dummified_cols].sum()

                for feature_value_key, count in composition_counts.items():
                    percentage = (
                        (count / total_rows_in_cluster * 100)
                        if total_rows_in_cluster > 0
                        else 0
                    )

                    # Populate the JSON
                    json_output[cluster_key]["composition"][feature_value_key] = {
                        "count": int(count),
                        "percentage": round(float(percentage), 2),
                    }

                # --- 2. Create the detailed CSV ---
                csv_cat_summary = composition_counts.reset_index()
                csv_cat_summary.columns = ["feature_value", "count"]
                csv_cat_summary["percentage"] = (
                    csv_cat_summary["count"] / total_rows_in_cluster * 100
                ).round(2)
                csv_cat_summary["cluster"] = cluster_id
                summary_list_for_csv.append(csv_cat_summary)

                if continuous_cols:
                    continuous_data_for_describe = cast(
                        pd.DataFrame, cluster_df[continuous_cols]
                    )
                    desc_stats = continuous_data_for_describe.describe().transpose()
                    desc_df = desc_stats.reset_index()
                    desc_df.rename(columns={"index": "column"}, inplace=True)
                    desc_df["cluster"] = cluster_id
                    summary_list_for_csv.append(desc_df)

            if summary_list_for_csv:
                full_summary_df = pd.concat(summary_list_for_csv, ignore_index=True)
                self._save_dataframe(
                    full_summary_df, f"04_cluster_composition_summary_{segment_name}"
                )

            if json_output:
                filepath = f"cluster_composition_summary_{segment_name}.json"
                self._save_json(json_output, filepath)
        except Exception as e:
            self.logger.error(
                f"    Error during cluster composition analysis for {segment_name}: {e}",
                exc_info=True,
            )

    def _save_json(self, data: dict, filename: str):
        """Saves a dictionary to a JSON file, handling numpy types."""
        if not data:
            self.logger.info(f"No data to save for {filename}. Skipping.")
            return

        filepath = os.path.join(self.data_output_path, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:

                def default_converter(o):
                    if isinstance(o, np.integer):
                        return int(o)
                    if isinstance(o, np.floating):
                        return float(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    raise TypeError(
                        f"Object of type {o.__class__.__name__} is not JSON "
                        f"serializable"
                    )

                json.dump(data, f, indent=4, default=default_converter)
            self.logger.info(f"  Saved JSON: {filepath}")
        except Exception as e:
            self.logger.error(f"    Error saving JSON {filepath}: {e}")

    def _create_segment_profile_summary(
        self,
        final_df_name,
        df_final_sum,
        optimal_k_values,
        quant_params,
        hdr_results,
    ):
        """Helper to create the detailed summary for final_cluster_profiles.json."""
        segment_key = f"{AGG_DF_PREFIX}{final_df_name}"  # noqa: E501
        quant_params_for_segment = quant_params.get(segment_key, {})
        n_classes_used = quant_params_for_segment.get("original_n_classes_input", "N/A")

        current_segment_profile = {
            "segment_name": final_df_name,
            "optimal_k_used": int(optimal_k_values.get(segment_key, 0)),
            "quantization_bins": n_classes_used,
            "bin_edges": quant_params_for_segment.get("bin_edges", []),
            "unit": quant_params_for_segment.get("unit", "units"),
            "clusters": [],
        }
        for _, cluster_row in df_final_sum.iterrows():
            cluster_id = int(cluster_row[CLUSTER_COL])
            cluster_c_cols = cluster_row.filter(like=QUANT_PREFIX)
            total_items_in_cluster = cluster_c_cols.sum()
            bin_profiles = []
            for bin_col, count in cluster_c_cols.items():
                bin_num = int(bin_col.replace(QUANT_PREFIX, ""))
                bin_profiles.append(
                    {
                        "bin_label": bin_num,
                        "count": float(count),
                        "percentage_of_cluster": (
                            (float(count) / total_items_in_cluster * 100)
                            if total_items_in_cluster > 0
                            else 0
                        ),
                    }
                )

            hdr_info_for_cluster = hdr_results.get(final_df_name, {}).get(
                str(cluster_id), {}
            )
            current_segment_profile["clusters"].append(
                {
                    "cluster_id": cluster_id,
                    "total_items": float(total_items_in_cluster),
                    "hdr_profile": hdr_info_for_cluster,
                    "bin_profiles": sorted(bin_profiles, key=lambda x: x["bin_label"]),
                }
            )
        return current_segment_profile


# =============================================================================
# ORCHESTRATION (The run_all_analyses function)
# =============================================================================
def run_all_analyses():
    """
    Main orchestration function to load configurations and run the analysis
    for all active configurations specified in the main config file.
    This function is designed to be callable from other scripts.
    """
    main_logger = logging.getLogger(__name__)

    try:
        main_logger.info(f"Loading multi-config file from: {DEFAULT_CONFIG_PATH}")
        multi_config = load_config(DEFAULT_CONFIG_PATH)

        active_configs = multi_config.get("active_config_names")
        if not active_configs or not isinstance(active_configs, list):
            raise ValueError(
                "Config file must contain a list key 'active_config_names' "
                "with at least one config block name."
            )

        main_logger.info(f"Found active configurations to run: {active_configs}")

        for config_name in active_configs:
            main_logger.info(f"\n--- Running analysis for: '{config_name}' ---")

            run_config = multi_config.get(config_name)
            if not run_config:
                main_logger.error(
                    f"Could not find configuration block for '{config_name}'. Skipping."
                )
                continue

            common_settings = multi_config.get("common_settings", {})

            def merge_dicts(base, override):
                for k, v in override.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        base[k] = merge_dicts(base[k], v)
                    else:
                        base[k] = v
                return base

            import copy

            final_config = copy.deepcopy(common_settings)
            final_config = merge_dicts(final_config, run_config)

            settings = AnalysisSettings.from_dict(final_config)
            analyzer = DistetaBatch(
                settings=settings,
                base_output_path=constants.OUTPUT_DIR,
                run_config_name=config_name,
            )
            analyzer.run_analysis()
            main_logger.info(f"--- Finished analysis for: '{config_name}' ---\n")

    except FileNotFoundError:
        main_logger.error(
            f"FATAL: The main configuration file was not found at "
            f"'{DEFAULT_CONFIG_PATH}'"
        )
        raise  # Re-raise to allow the caller to handle it
    except Exception as e:
        main_logger.error(
            f"An unexpected error occurred during analysis execution: {e}",
            exc_info=True,
        )
        raise  # Re-raise


# =============================================================================
# SCRIPT EXECUTION (The if __name__ == "__main__" block)
# =============================================================================
if __name__ == "__main__":
    # Basic logging configuration to see output in the console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        run_all_analyses()
    except Exception as e:
        # Errors are logged in detail within run_all_analyses, but we catch here
        # to ensure the script exits with a non-zero status code on failure.
        logging.critical(
            f"A critical error occurred, and the process will terminate. Error: {e}"
        )
        exit(1)
