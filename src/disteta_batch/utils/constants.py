# This module centralizes all shared constants for the DistETA package
# to ensure consistency and adherence to the DRY (Don't Repeat Yourself)
# principle.

# --- Analysis & Segmentation Keys ---
AGG_DF_PREFIX = "df_comb_"
ALL_DATA_GROUP_KEY = "all_data"  # Key for analysis on the entire dataset
NAN_GROUP_KEY = "NaN"

# --- Column Names ---
COMB_COL = "comb"  # For unique categorical combinations
CLASS_COL = "class"  # For quantized bin labels before dummification
CLASS_COL_NUM = "class_num"  # For numeric bin labels in plots
CLUSTER_COL = "cluster"  # For cluster assignments

# --- Prefixes ---
QUANT_PREFIX = "c_"  # For dummified quantized columns

# --- Thresholds ---
HDR_THRESHOLD_MIN_COUNT = 2  # Min data points for a valid HDR

# --- Configuration Keys ---
# Used to parse the YAML configuration file in a type-safe way.
CFG_IO_INPUT_DATA_PATH = "io.input_data_path"
CFG_COLS_CATEGORICAL = "columns.categorical"
CFG_COLS_CONTINUOUS = "columns.continuous_to_analyze"
CFG_COLS_GROUPING = "columns.grouping_column"
CFG_COLS_UNITS = "columns.continuous_units"
CFG_FILTERING_VALUES = "filtering.filter_group_value"
CFG_PREPROC_MIN_COMB_SIZE = "preprocessing.min_combination_size_input"
CFG_PREPROC_N_CLASSES = "preprocessing.quantization_n_classes_input"
CFG_PREPROC_SILHOUETTE_DROP = "preprocessing.silhouette_score_drop_percentage_input"
CFG_MAPPING_VALUE = "mapping.column_value"
CFG_MAPPING_COLUMN = "mapping.column_name"
CFG_PLOTTING_X_LABEL = "plotting.x_axis_label"
CFG_PLOTTING_QUANT_LABEL = "plotting.quantized_axis_label"
CFG_HDR_THRESHOLD = "hdr_analysis.hdr_threshold_percentage"
CFG_CLUSTERING_MIN_K = "clustering.cluster_range_min"
CFG_CLUSTERING_MAX_K = "clustering.cluster_range_max"
