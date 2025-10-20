# This module provides core utilities for data and configuration handling,
# including loading YAML configuration files and reading data from various file
# formats like CSV, Parquet, and Excel.
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
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
    """Loads data from a file, supporting CSV, Parquet, and Excel formats."""
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
    """Identifies numeric columns (integer and float types) in a DataFrame."""
    if exclude_cols is None:
        exclude_cols = []
    valid_exclude_cols = {col for col in exclude_cols if col in df.columns}
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return [col for col in numeric_cols if col not in valid_exclude_cols]
    except Exception as e:
        logger.warning(f"Error during numeric column detection: {e}")
        return []
