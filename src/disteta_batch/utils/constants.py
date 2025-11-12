"""
This module centralizes all shared constants for the DistETA batch analysis package.

Using a dedicated constants module ensures consistency, avoids magic strings,
and makes the code easier to maintain and understand.

<details>
<summary>Deprecated Configuration Keys</summary>

The following keys were previously used for parsing the YAML configuration file.
The current implementation uses a typed dataclass (`AnalysisSettings`) for
configuration management, which is safer and more explicit. These are kept for
historical reference.

- `CFG_IO_INPUT_DATA_PATH = "io.input_data_path"`
- `CFG_COLS_CATEGORICAL = "columns.categorical"`
- `CFG_COLS_CONTINUOUS = "columns.continuous_to_analyze"`
- `CFG_COLS_GROUPING = "columns.grouping_column"`
- `CFG_COLS_UNITS = "columns.continuous_units"`
- `CFG_FILTERING_VALUES = "filtering.filter_group_value"`
- `CFG_PREPROC_MIN_COMB_SIZE = "preprocessing.min_combination_size_input"`
- `CFG_PREPROC_N_CLASSES = "preprocessing.quantization_n_classes_input"`
- `CFG_PREPROC_SILHOUETTE_DROP = "preprocessing.silhouette_score_drop_percentage_input"`
- `CFG_MAPPING_VALUE = "mapping.column_value"`
- `CFG_MAPPING_COLUMN = "mapping.column_name"`
- `CFG_PLOTTING_X_LABEL = "plotting.x_axis_label"`
- `CFG_PLOTTING_QUANT_LABEL = "plotting.quantized_axis_label"`
- `CFG_HDR_THRESHOLD = "hdr_analysis.hdr_threshold_percentage"`
- `CFG_CLUSTERING_MIN_K = "clustering.cluster_range_min"`
- `CFG_CLUSTERING_MAX_K = "clustering.cluster_range_max"`

</details>
"""

# =============================================================================
# Core Analysis and Segmentation Keys
# =============================================================================

AGG_DF_PREFIX = "df_comb_"  # Prefix for aggregated DataFrame names.
ALL_DATA_GROUP_KEY = (
    "all_data"  # Key for analysis on the entire dataset without grouping.
)
NAN_GROUP_KEY = "NaN"  # String representation for NaN groups.

# =============================================================================
# Standardized Column Names
# =============================================================================

COMB_COL = "comb"  # Column for unique categorical combination IDs.
CLASS_COL = "class"  # Temporary column for quantized bin labels before dummification.
CLASS_COL_NUM = "class_num"  # Column for numeric bin labels used in plotting.
CLUSTER_COL = "cluster"  # Column for final cluster assignments.

# =============================================================================
# Prefixes
# =============================================================================

QUANT_PREFIX = "c_"  # Prefix for dummified (one-hot encoded) quantized columns.

# =============================================================================
# Analysis Thresholds
# =============================================================================

HDR_THRESHOLD_MIN_COUNT = (
    2  # Minimum number of data points required for a valid High-Density Region.
)
