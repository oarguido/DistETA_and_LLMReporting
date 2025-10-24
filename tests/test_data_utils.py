import os
import pytest
import pandas as pd
from src.disteta_aidi.utils.data_utils import load_config, load_data, identify_numeric_columns


def test_load_config():
    """Tests that the load_config function can correctly load a YAML file."""
    # Get the absolute path to the test config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "test_config.yaml")

    # Load the config
    config = load_config(config_path)

    # Check that the config is a dictionary and has the expected key-value pair
    assert isinstance(config, dict)
    assert config.get("key") == "value"


def test_load_data_csv():
    """Tests that the load_data function can correctly load a CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test.csv")
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_parquet():
    """Tests that the load_data function can correctly load a Parquet file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test.parquet")
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_excel():
    """Tests that the load_data function can correctly load an Excel file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test.xlsx")
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_non_existent():
    """Tests that load_data raises an error for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def test_load_data_empty_file():
    """Tests that load_data raises an error for an empty file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "empty.txt")
    with pytest.raises(ValueError):
        load_data(file_path)


def test_load_data_unsupported_file():
    """Tests that load_data raises an error for an unsupported file type."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "test.xyz")
    with pytest.raises(RuntimeError):
        load_data(file_path)


def test_identify_numeric_columns():
    """Tests that identify_numeric_columns correctly identifies numeric columns."""
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
    df = pd.DataFrame(data)
    numeric_cols = identify_numeric_columns(df)
    assert numeric_cols == ["col1", "col3"]


def test_identify_numeric_columns_with_exclude():
    """Tests that identify_numeric_columns correctly excludes specified columns."""
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
    df = pd.DataFrame(data)
    numeric_cols = identify_numeric_columns(df, exclude_cols=["col1"])
    assert numeric_cols == ["col3"]
