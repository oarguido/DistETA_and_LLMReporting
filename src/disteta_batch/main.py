"""
This module is the core of the DistETA batch analysis pipeline.

It defines the `DistetaBatch` class, which orchestrates a configurable workflow
for analyzing distributional data. The key responsibilities of this module are:

1.  **Data Loading and Preparation**: Ingests and prepares data for analysis.
2.  **Data Segmentation**: Groups data into meaningful segments.
3.  **Quantization and Clustering**: Discretizes continuous features and uses
    K-Means clustering to identify dominant distributional patterns.
4.  **Artifact Generation**: Produces a comprehensive set of outputs, including
    processed data, JSON summaries, visualizations, and trained models.

The analysis is driven by a central YAML configuration file, allowing for
flexible and repeatable execution. The main entry point is the `run_all_analyses`
function, which can be called from other scripts or executed directly.

Execution:
    To run the analysis for all active configurations:
    $ python -m src.disteta_batch.main
"""

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

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
    CLUSTER_COL,
    COMB_COL,
    NAN_GROUP_KEY,
    QUANT_PREFIX,
)
from .utils.data_utils import (
    _default_json_converter,
    identify_numeric_columns,
    load_config,
    load_data,
)
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
    constants.PROJECT_ROOT, constants.CONFIG_DIR, constants.DISTETA_CONFIG_FILENAME
)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("shutil").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATION DATA MODELS
# =============================================================================
@dataclass
class IOOptions:
    """Specifies input/output paths for an analysis."""

    input_data_path: str = "data/truck_arrival_data.csv"

    @property
    def input_path_resolved(self) -> str:
        """Returns the absolute path to the input data file."""
        return os.path.join(constants.PROJECT_ROOT, self.input_data_path)


@dataclass
class ColumnConfig:
    """Defines all column-related configurations for an analysis."""

    grouping_column: Optional[str] = None
    filter_values: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    continuous_to_analyze: List[str] = field(default_factory=list)
    continuous_units_map: Dict[str, str] = field(default_factory=dict)
    value_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    column_name_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class PreprocessingOptions:
    """Defines parameters for the data preprocessing stage."""

    min_combination_size_input: Union[str, int] = "auto-knee"
    quantization_n_classes_input: Union[str, int] = "auto"


@dataclass
class ClusteringOptions:
    """Defines parameters for the clustering analysis stage."""

    clustering_min_k: int = 2
    clustering_max_k: int = 10
    percent_drop_threshold_input: float = 5.0


@dataclass
class HDROptions:
    """Defines parameters for High-Density Region (HDR) analysis."""

    hdr_threshold_percentage: float = 90.0


@dataclass
class VisualizationOptions:
    """Defines parameters for plot visualization."""

    x_axis_label: str = "Value"
    quantized_axis_label: str = "Bin"


@dataclass
class AnalysisSettings:
    """A typed dataclass that aggregates all settings for a single analysis run."""

    io_options: IOOptions
    column_config: ColumnConfig
    preprocessing_options: PreprocessingOptions
    clustering_options: ClusteringOptions
    hdr_options: HDROptions
    visualization_options: VisualizationOptions

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AnalysisSettings":
        """Factory method to create an AnalysisSettings instance from a dictionary."""
        return cls(
            io_options=IOOptions(**config.get("io", {})),
            column_config=ColumnConfig(**config.get("columns", {})),
            preprocessing_options=PreprocessingOptions(
                **config.get("preprocessing", {})
            ),
            clustering_options=ClusteringOptions(**config.get("clustering", {})),
            hdr_options=HDROptions(**config.get("hdr_analysis", {})),
            visualization_options=VisualizationOptions(
                **config.get("visualization", {})
            ),
        )

    # --- Convenience Properties for direct access to nested settings ---
    @property
    def input_path_resolved(self) -> str:
        return self.io_options.input_path_resolved

    @property
    def grouping_col_config(self) -> Optional[str]:
        return self.column_config.grouping_column

    @property
    def filter_values_config(self) -> List[str]:
        return self.column_config.filter_values

    @property
    def categorical_cols(self) -> List[str]:
        return self.column_config.categorical

    @property
    def continuous_cols_config(self) -> List[str]:
        return self.column_config.continuous_to_analyze

    @property
    def continuous_units_map(self) -> Dict[str, str]:
        return self.column_config.continuous_units_map

    @property
    def value_mapping(self) -> Dict[str, Dict[str, str]]:
        return self.column_config.value_mapping

    @property
    def column_name_mapping(self) -> Dict[str, str]:
        return self.column_config.column_name_mapping

    @property
    def min_comb_size_input(self) -> Union[str, int]:
        return self.preprocessing_options.min_combination_size_input

    @property
    def n_classes_input(self) -> Union[str, int]:
        return self.preprocessing_options.quantization_n_classes_input

    @property
    def clustering_min_k(self) -> int:
        return self.clustering_options.clustering_min_k

    @property
    def clustering_max_k(self) -> int:
        return self.clustering_options.clustering_max_k

    @property
    def percent_drop_threshold_input(self) -> float:
        return self.clustering_options.percent_drop_threshold_input

    @property
    def hdr_threshold_percentage_config(self) -> float:
        return self.hdr_options.hdr_threshold_percentage

    @property
    def x_axis_label_config(self) -> str:
        return self.visualization_options.x_axis_label

    @property
    def quantized_axis_label_config(self) -> str:
        return self.visualization_options.quantized_axis_label


# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================
class DistetaBatch:
    """Orchestrates the entire DistETA (Distributional ETA) analysis pipeline."""

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
        self.generated_figures: List[Tuple[Any, str]] = []

    def run_analysis(self):
        """Main entry point to execute the full analysis pipeline for the given settings."""
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
            aggregated_dfs, quantization_params, dummified_dfs = (
                self._quantize_and_aggregate_segments(df_filtered, continuous_cols)
            )
            optimal_k_values, all_silh_scores = self._find_and_plot_optimal_k(
                aggregated_dfs, continuous_cols
            )
            final_cluster_profiles, hdr_results, cluster_mappings = (
                self._perform_final_clustering_and_hdr(
                    aggregated_dfs,
                    optimal_k_values,
                    df_filtered,
                    continuous_cols,
                    quantization_params,
                )
            )
            self._train_and_save_classifiers(
                dummified_dfs, cluster_mappings, optimal_k_values, quantization_params
            )
            self._plot_final_cluster_distributions(
                final_cluster_profiles,
                continuous_cols,
                quantization_params,
                hdr_results,
            )
            self._save_json_artifacts(
                continuous_cols,
                optimal_k_values,
                all_silh_scores,
                final_cluster_profiles,
                hdr_results,
                quantization_params,
            )
            self._save_all_generated_figures()

            end_time = time.time()
            self.logger.info(
                f"\nAnalysis for config '{self.run_config_name}' complete. Total time: {end_time - start_time:.2f}s."
            )

            with open(done_filepath, "w") as f:
                f.write(self.run_specific_output_dir)

        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

    def _setup_output_directories(self):
        """Creates the directory structure for all analysis outputs."""
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
            f"Outputs for run '{self.run_config_name}' will be saved to: {self.run_specific_output_dir}"
        )

    def _get_groups_to_process(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Segments the DataFrame into groups based on the configuration."""
        grouping_col = self.settings.grouping_col_config
        if not grouping_col or grouping_col not in df.columns:
            return {ALL_DATA_GROUP_KEY: df.copy()}

        grouped = df.groupby(grouping_col, dropna=False)
        filter_list = self.settings.filter_values_config
        target_group_keys = [
            key
            for key in grouped.groups.keys()
            if not filter_list
            or (
                isinstance(key, float)
                and np.isnan(key)
                and NAN_GROUP_KEY in filter_list
            )
            or (key in filter_list)
        ]

        groups_to_process = {}
        for key in target_group_keys:
            group_df = grouped.get_group(key)
            if not group_df.empty:
                dict_key = (
                    NAN_GROUP_KEY
                    if isinstance(key, float) and np.isnan(key)
                    else str(key)
                )
                groups_to_process[dict_key] = group_df
        return groups_to_process

    def _prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Loads, preprocesses, and filters the main dataset."""
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

        expanded_df[COMB_COL] = (
            (
                pd.factorize(
                    expanded_df[encoded_col_list].astype(str).agg("- ".join, axis=1)
                )[0]
                + 1
            )
            if encoded_col_list
            else 1
        )

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

        retained_rows, total_rows = len(expanded_df_filtered), len(expanded_df)
        percent_retained = (retained_rows / total_rows * 100) if total_rows > 0 else 0
        self.logger.info(
            f"Filtering by min combination size of {min_size_threshold} retained {retained_rows:,} of {total_rows:,} rows ({percent_retained:.2f}%)."
        )

        self._save_dataframe(
            cast(pd.DataFrame, expanded_df_filtered),
            "01_filtered_pre_quantization_data",
        )
        return cast(pd.DataFrame, expanded_df_filtered), continuous_cols

    def _quantize_and_aggregate_segments(
        self, df: pd.DataFrame, continuous_cols: List[str]
    ) -> Tuple[Dict, Dict, Dict]:
        """Quantizes continuous features and aggregates them for each data segment."""
        groups_to_process = self._get_groups_to_process(df)
        if not groups_to_process:
            raise ValueError("No data groups left after filtering.")

        quantization_params, aggregated_dfs, dummified_dfs = {}, {}, {}
        self.logger.info("\n--- Quantizing and Aggregating Data ---")
        for group_key, group_df in groups_to_process.items():
            for feature_col in continuous_cols:
                if (
                    feature_col not in group_df.columns
                    or group_df[feature_col].nunique(dropna=True) <= 1
                ):
                    continue

                if self.settings.n_classes_input == "auto":
                    n_classes_to_use = calculate_optimal_bins(
                        cast(pd.Series, group_df[feature_col])
                    )
                    self.logger.info(
                        f"For group '{group_key}', feature '{feature_col}': Auto-calculated optimal bins = {n_classes_to_use}"
                    )
                elif isinstance(self.settings.n_classes_input, int):
                    n_classes_to_use = self.settings.n_classes_input
                else:
                    self.logger.error(
                        f"Invalid value for 'quantization_n_classes_input': '{self.settings.n_classes_input}'. Skipping."
                    )
                    continue

                if n_classes_to_use <= 1:
                    self.logger.warning(
                        f"Skipping '{feature_col}' for group '{group_key}': not enough data variance for >1 bin."
                    )
                    continue

                try:
                    df_quant, bin_edges = quantize_and_dummify(
                        group_df,
                        feature_col,
                        n_classes_to_use,
                        [str(i) for i in range(1, n_classes_to_use + 1)],
                    )
                    agg_df = aggregate_by_comb(df_quant)
                    if not agg_df.empty:
                        segment_key = f"{AGG_DF_PREFIX}{group_key}_{feature_col}"
                        aggregated_dfs[segment_key] = agg_df
                        dummified_dfs[segment_key] = df_quant
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
                        f"Unexpected error processing '{feature_col}' in group '{group_key}': {e}",
                        exc_info=True,
                    )

        if not aggregated_dfs:
            raise ValueError("No data to cluster after aggregation.")
        return aggregated_dfs, quantization_params, dummified_dfs

    def _find_and_plot_optimal_k(
        self, aggregated_dfs: Dict, continuous_cols: List[str]
    ) -> Tuple[Dict, Dict]:
        """Performs clustering analysis (Elbow/Silhouette) to find the best K for each segment."""
        self.logger.info("\n--- Performing Clustering Analysis (Elbow/Silhouette) ---")
        all_silh_scores, wcss_data, labels_data = {}, {}, {}
        if not aggregated_dfs:
            self.logger.warning(
                "No aggregated data for clustering. Skipping K-selection."
            )
            return {}, {}

        for df_name, df_agg in aggregated_dfs.items():
            combination_cols = sorted(
                [c for c in df_agg.columns if c.startswith(QUANT_PREFIX)],
                key=lambda x: int(x.split(QUANT_PREFIX)[1]),
            )
            if not combination_cols:
                continue

            X = df_agg[combination_cols]
            max_k_adj = min(self.settings.clustering_max_k, X.shape[0] - 1)
            min_k_adj = max(2, self.settings.clustering_min_k)
            if min_k_adj > max_k_adj:
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
    ) -> Tuple[Dict, Dict, Dict]:
        """Performs final clustering with optimal K and calculates HDR for each cluster."""
        self.logger.info("\n--- Performing Final Clustering with Optimal K ---")
        final_cluster_profiles, hdr_results, cluster_mappings = {}, {}, {}
        for df_name, df_agg in aggregated_dfs.items():
            if df_name not in optimal_k_values:
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
            final_cluster_profiles[final_df_key], cluster_mappings[df_name] = (
                df_sum,
                df_map,
            )

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
        return final_cluster_profiles, hdr_results, cluster_mappings

    def _train_and_save_classifiers(
        self,
        dummified_dfs: Dict,
        cluster_mappings: Dict,
        optimal_k_values: Dict,
        quantization_params: Dict,
    ):
        """Trains and saves an XGBoost classifier for each data segment."""
        self.logger.info("\n--- Training and Saving Cluster Prediction Models ---")
        os.makedirs(constants.MODELS_DIR, exist_ok=True)

        for df_name, df_map in cluster_mappings.items():
            if (
                df_name not in optimal_k_values
                or df_name not in dummified_dfs
                or df_name not in quantization_params
            ):
                continue

            try:
                df_quant = dummified_dfs[df_name]
                df_train_full = pd.merge(
                    df_quant, df_map[[COMB_COL, CLUSTER_COL]], on=COMB_COL, how="inner"
                )
                if df_train_full.empty:
                    continue

                feature_cols = sorted(
                    [c for c in df_train_full.columns if c.startswith(QUANT_PREFIX)],
                    key=lambda x: int(x.split(QUANT_PREFIX)[1]),
                )
                y_train_raw, X_train = (
                    df_train_full[CLUSTER_COL],
                    df_train_full[feature_cols],
                )

                le = LabelEncoder()
                y_train = le.fit_transform(y_train_raw)

                if not hasattr(le, "classes_") or le.classes_ is None:
                    self.logger.warning(
                        f"Segment '{df_name}' failed to produce classes after encoding. Skipping model training."
                    )
                    continue

                num_classes = len(le.classes_)
                if num_classes < 2:
                    self.logger.warning(
                        f"Segment '{df_name}' has fewer than 2 classes ({num_classes}). Skipping model training."
                    )
                    continue

                self.logger.info(
                    f"  Training model for segment '{df_name}' with {len(X_train)} samples and {num_classes} classes."
                )
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_classes,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_train)
                y_pred_proba = model.predict_proba(X_train)
                report = classification_report(
                    y_train,
                    y_pred,
                    output_dict=True,
                    zero_division=0.0,  # type: ignore
                )
                auroc = roc_auc_score(
                    y_train, y_pred_proba, multi_class="ovr", average="weighted"
                )

                model_metadata = {
                    "training_timestamp": self.run_timestamp,
                    "config_name": self.run_config_name,
                    "segment_name": df_name.replace(f"{AGG_DF_PREFIX}", ""),
                    "num_samples_trained": len(X_train),
                    "num_features": X_train.shape[1],
                    "num_classes": num_classes,
                    "accuracy": accuracy_score(y_train, y_pred),
                    "auroc_weighted_ovr": auroc,
                    "classification_report": report,
                    "label_mapping": dict(
                        zip(
                            range(num_classes),
                            le.inverse_transform(range(num_classes)),
                        )
                    ),
                }

                model_filename = f"{self.run_config_name}_{df_name.replace(f'{AGG_DF_PREFIX}', '')}.json"
                model_filepath = os.path.join(constants.MODELS_DIR, model_filename)
                model.save_model(model_filepath)
                self.logger.info(f"  Saved trained model to: {model_filepath}")

                metadata_filepath = os.path.join(
                    constants.MODELS_DIR,
                    model_filename.replace(".json", "_metadata.json"),
                )
                with open(metadata_filepath, "w", encoding="utf-8") as f:
                    json.dump(
                        model_metadata, f, indent=4, default=_default_json_converter
                    )
                self.logger.info(f"  Saved model metadata to: {metadata_filepath}")

            except Exception as e:
                self.logger.error(
                    f"Failed to train or save model for segment {df_name}: {e}",
                    exc_info=True,
                )

    def _plot_final_cluster_distributions(
        self,
        final_cluster_profiles: Dict,
        continuous_cols: List[str],
        quantization_params: Dict,
        hdr_results: Dict,
    ):
        """Generates and stores plots for the final cluster distributions."""
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
                hdr_threshold_percentage=self.settings.hdr_threshold_percentage_config,
                save_non_interactive=True,
            )
            if fig:
                self.generated_figures.append(
                    (fig, f"03_cluster_distributions_{final_df_name}")
                )

    def _plot_initial_distributions(self, df: pd.DataFrame, continuous_cols: List[str]):
        """Generates and stores plots for the initial data distributions."""
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
        """Generates and stores plots for the silhouette and elbow analysis."""
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
        """Saves all generated figures to PNG and HTML files."""
        self.logger.info(
            f"Saving {len(self.generated_figures)} plots to '{self.graphics_output_path}'..."
        )
        for fig, base_filename in self.generated_figures:
            safe_filename = base_filename.replace(" ", "_").replace("/", "-")
            try:
                fig.write_image(
                    os.path.join(self.graphics_output_path, f"{safe_filename}.png"),
                    width=1280,
                    height=720,
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to save PNG {safe_filename}.png: {e}. Ensure 'kaleido' is installed."
                )
            try:
                fig.update_layout(width=None, height=None, autosize=True)
                fig.write_html(
                    os.path.join(self.graphics_output_path, f"{safe_filename}.html"),
                    include_plotlyjs="cdn",
                    full_html=False,
                )
            except Exception as e:
                self.logger.error(f"Failed to save HTML {safe_filename}.html: {e}.")

    def _save_json_artifacts(
        self,
        continuous_cols,
        optimal_k,
        k_metrics,
        cluster_profiles,
        hdr_results,
        quant_params,
    ):
        """Saves all JSON summary artifacts for the run."""
        self.logger.info(
            f"Saving JSON summary artifacts to '{self.data_output_path}'..."
        )
        final_profiles_for_json = {
            name: self._create_segment_profile_summary(
                name, df, optimal_k, quant_params, hdr_results
            )
            for name, df in cluster_profiles.items()
            if df is not None and not df.empty
        }
        self._save_json(self._create_run_summary(continuous_cols), "run_summary.json")
        self._save_json(
            {"optimal_k_values": optimal_k, "silhouette_scores_by_segment": k_metrics},
            "k_selection_summary.json",
        )
        self._save_json(final_profiles_for_json, "final_cluster_profiles.json")

    def _create_run_summary(self, continuous_cols: List[str]) -> Dict[str, Any]:
        """Creates a dictionary summarizing the main configuration settings for the run."""
        return {
            "run_timestamp": self.run_timestamp,
            "config_name_used": self.run_config_name,
            "input_data_path": self.settings.input_path_resolved,
            "categorical_columns_defined": self.settings.categorical_cols,
            "continuous_columns_analyzed": continuous_cols,
            "grouping_column_used": self.settings.grouping_col_config or "N/A",
            "filter_values_for_grouping_column": self.settings.filter_values_config
            or "all_available_groups_processed",
            "min_combination_size_input": self.settings.min_comb_size_input,
            "quantization_n_classes_input": self.settings.n_classes_input,
            "clustering_k_range_tested": {
                "min_k_config": self.settings.clustering_min_k,
                "max_k_config": self.settings.clustering_max_k,
            },
            "silhouette_score_drop_threshold_input_percentage": self.settings.percent_drop_threshold_input,
            "hdr_threshold_percentage_input": self.settings.hdr_threshold_percentage_config,
            "x_axis_label_config": self.settings.x_axis_label_config,
            "quantized_axis_label_config": self.settings.quantized_axis_label_config,
        }

    def _parse_group_and_feature(
        self, base_name: str, grouping_col: Optional[str], continuous_cols: List[str]
    ) -> Tuple[Optional[str], str]:
        """Utility to extract group and feature names from a segment name string."""
        if grouping_col:
            for feature in continuous_cols:
                if base_name.endswith(f"_{feature}"):
                    return base_name[: -len(f"_{feature}")], feature
        if base_name.startswith(f"{ALL_DATA_GROUP_KEY}_"):
            return ALL_DATA_GROUP_KEY, base_name[len(ALL_DATA_GROUP_KEY) + 1 :]
        return ALL_DATA_GROUP_KEY, base_name

    def _save_dataframe(self, df: pd.DataFrame, base_filename: str):
        """Saves a DataFrame to a CSV file in the data output directory."""
        if df is None or df.empty:
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
        """Analyzes and saves the categorical and continuous composition of each cluster."""
        self.logger.info(f"  Analyzing cluster composition for segment: {segment_name}")
        try:
            if (
                COMB_COL not in original_filtered_df.columns
                or CLUSTER_COL not in combination_to_cluster_map.columns
            ):
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
                return

            json_output: Dict[str, Any] = {}
            summary_list_for_csv: List[pd.DataFrame] = []

            for cluster_id in sorted(df_with_clusters[CLUSTER_COL].unique()):
                cluster_key = f"cluster_{cluster_id}"
                cluster_df = df_with_clusters[
                    df_with_clusters[CLUSTER_COL] == cluster_id
                ]
                total_rows_in_cluster = len(cluster_df)

                json_output[cluster_key] = {
                    "total_rows_in_cluster": total_rows_in_cluster,
                    "composition": {},
                }
                composition_counts = cluster_df[dummified_cols].sum()

                for feature_value_key, count in composition_counts.items():
                    percentage = (
                        (count / total_rows_in_cluster * 100)
                        if total_rows_in_cluster > 0
                        else 0
                    )
                    json_output[cluster_key]["composition"][feature_value_key] = {
                        "count": int(count),
                        "percentage": round(float(percentage), 2),
                    }

                csv_cat_summary = composition_counts.reset_index()
                csv_cat_summary.columns = ["feature_value", "count"]
                csv_cat_summary["percentage"] = (
                    csv_cat_summary["count"] / total_rows_in_cluster * 100
                ).round(2)
                csv_cat_summary["cluster"] = cluster_id
                summary_list_for_csv.append(csv_cat_summary)

                if continuous_cols:
                    desc_stats = (
                        cast(pd.DataFrame, cluster_df[continuous_cols])
                        .describe()
                        .transpose()
                        .reset_index()
                    )
                    desc_stats.rename(columns={"index": "column"}, inplace=True)
                    desc_stats["cluster"] = cluster_id
                    summary_list_for_csv.append(desc_stats)

            if summary_list_for_csv:
                self._save_dataframe(
                    pd.concat(summary_list_for_csv, ignore_index=True),
                    f"04_cluster_composition_summary_{segment_name}",
                )
            if json_output:
                self._save_json(
                    json_output, f"cluster_composition_summary_{segment_name}.json"
                )
        except Exception as e:
            self.logger.error(
                f"    Error during cluster composition analysis for {segment_name}: {e}",
                exc_info=True,
            )

    def _save_json(self, data: dict, filename: str):
        """Saves a dictionary to a JSON file with custom numpy type handling."""
        if not data:
            return
        filepath = os.path.join(self.data_output_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, default=_default_json_converter)
            self.logger.info(f"  Saved JSON: {filepath}")
        except Exception as e:
            self.logger.error(f"    Error saving JSON {filepath}: {e}")

    def _create_segment_profile_summary(
        self, final_df_name, df_final_sum, optimal_k_values, quant_params, hdr_results
    ) -> Dict[str, Any]:
        """Creates a detailed summary dictionary for a single segment's cluster profiles."""
        segment_key = f"{AGG_DF_PREFIX}{final_df_name}"
        quant_params_for_segment = quant_params.get(segment_key, {})
        current_segment_profile: Dict[str, Any] = {
            "segment_name": final_df_name,
            "optimal_k_used": int(optimal_k_values.get(segment_key, 0)),
            "quantization_bins": quant_params_for_segment.get(
                "original_n_classes_input", "N/A"
            ),
            "bin_edges": quant_params_for_segment.get("bin_edges", []),
            "unit": quant_params_for_segment.get("unit", "units"),
            "clusters": [],
        }
        for _, cluster_row in df_final_sum.iterrows():
            cluster_id = int(cluster_row[CLUSTER_COL])
            cluster_c_cols = cluster_row.filter(like=QUANT_PREFIX)
            total_items_in_cluster = cluster_c_cols.sum()
            bin_profiles = [
                {
                    "bin_label": int(bin_col.replace(QUANT_PREFIX, "")),
                    "count": float(count),
                    "percentage_of_cluster": (
                        float(count) / total_items_in_cluster * 100
                    )
                    if total_items_in_cluster > 0
                    else 0,
                }
                for bin_col, count in cluster_c_cols.items()
            ]

            current_segment_profile["clusters"].append(
                {
                    "cluster_id": cluster_id,
                    "total_items": float(total_items_in_cluster),
                    "hdr_profile": hdr_results.get(final_df_name, {}).get(
                        str(cluster_id), {}
                    ),
                    "bin_profiles": sorted(bin_profiles, key=lambda x: x["bin_label"]),
                }
            )
        return current_segment_profile


# =============================================================================
# ORCHESTRATION
# =============================================================================
def run_all_analyses():
    """
    Main orchestration function to load configurations and run the analysis for all active configurations.
    """
    main_logger = logging.getLogger(__name__)
    try:
        main_logger.info(f"Loading multi-config file from: {DEFAULT_CONFIG_PATH}")
        multi_config = load_config(DEFAULT_CONFIG_PATH)

        active_configs = multi_config.get("active_config_names")
        if not active_configs or not isinstance(active_configs, list):
            raise ValueError(
                "Config file must contain a list key 'active_config_names'."
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

            # Deep merge common settings with specific run config
            final_config = copy.deepcopy(multi_config.get("common_settings", {}))
            # Ensure run_config is a dict before merging
            final_config = merge_dicts(final_config, run_config or {})

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
            f"FATAL: The main configuration file was not found at '{DEFAULT_CONFIG_PATH}'"
        )
        raise
    except Exception as e:
        main_logger.error(
            f"An unexpected error occurred during analysis execution: {e}",
            exc_info=True,
        )
        raise


def merge_dicts(base: Dict, override: Optional[Dict]) -> Dict:
    """Recursively merges two dictionaries."""
    if override is None:
        return base
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        run_all_analyses()
    except Exception as e:
        logging.critical(
            f"A critical error occurred, and the process will terminate. Error: {e}"
        )
        exit(1)
