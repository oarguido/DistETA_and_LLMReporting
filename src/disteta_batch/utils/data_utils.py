"""
This module provides core utilities for data and configuration handling.

It includes robust functions for:
1.  **Loading YAML configurations**: Safely loads and parses YAML files with
    detailed error handling.
2.  **Loading data**: Reads data from various file formats (CSV, Parquet, Excel)
    into pandas DataFrames, automatically detecting the file type.
3.  **Type identification**: Automatically identifies numeric columns in a
    DataFrame, which is useful for downstream processing.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _default_json_converter(o):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file with robust error handling.

    Args:
        config_path: The absolute path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is invalid or cannot be parsed.
        RuntimeError: For other unexpected errors during file reading.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return {} if config is None else config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_path}': {e}")
        raise ValueError(f"Invalid YAML in {config_path}") from e
    except Exception as e:
        logger.error(f"Error reading configuration file '{config_path}': {e}")
        raise RuntimeError(f"Could not read {config_path}") from e


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a file, supporting CSV, Parquet, and Excel formats.

    The function automatically detects the file type based on its extension.

    Args:
        filepath: The absolute path to the data file.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file is empty or has an unsupported format.
        RuntimeError: For other unexpected errors during data loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Data file is empty: {filepath}")

    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: .csv, .parquet, .xlsx, .xls"
            )
        logger.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load or parse data from {filepath}: {e}")
        raise RuntimeError(f"Error loading data from {filepath}") from e


def identify_numeric_columns(
    df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Identifies all numeric (integer and float) columns in a DataFrame.

    Args:
        df: The DataFrame to analyze.
        exclude_cols: A list of column names to explicitly exclude from the search.

    Returns:
        A list of column names that are identified as numeric.
    """
    if exclude_cols is None:
        exclude_cols = []

    valid_exclude_cols = {col for col in exclude_cols if col in df.columns}

    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return [col for col in numeric_cols if col not in valid_exclude_cols]
    except Exception as e:
        logger.warning(f"Error during numeric column detection: {e}")
        return []
