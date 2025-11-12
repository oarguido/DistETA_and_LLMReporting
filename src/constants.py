"""
Centralized definitions for all project-wide constants.

This module consolidates file paths, directory names, and other static values
to ensure consistency and ease of maintenance. By defining these constants in one
place, we avoid hardcoding strings in other modules, making the codebase more
robust and easier to reconfigure.

Attributes:
    PROJECT_ROOT (str): The absolute path to the project's root directory.
    CONFIG_DIR (str): The name of the configuration directory.
    DATA_DIR (str): The name of the data directory.
    OUTPUT_DIR (str): The name of the main output directory.
    MODELS_DIR (str): The name of the directory for storing trained models.
    ASSETS_DIR (str): The name of the directory for static assets like CSS.
    STREAMING_PLOTS_DIR (str): The path to the directory for streaming plots.
    DISTETA_CONFIG_FILENAME (str): The filename for the batch analysis config.
    REPORT_GEN_CONFIG_FILENAME (str): The filename for the report generator config.
    DATA_DIR_NAME (str): The name for the data subdirectory within a run output.
    GRAPHICS_DIR_NAME (str): The name for the graphics subdirectory within a run output.
    LOGS_DIR_NAME (str): The name for the logs subdirectory within a run output.
    REPORTS_DIR_NAME (str): The name for the reports subdirectory within a run output.
"""

import os

# --- Project Root ---
# Resolves the absolute path to the project's root directory, allowing for
# consistent pathing regardless of where the script is executed from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Top-Level Directory Names ---
CONFIG_DIR = "config"
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
ASSETS_DIR = "assets"

# --- Derived and Specific Paths ---
STREAMING_PLOTS_DIR = os.path.join(OUTPUT_DIR, "streaming_plots")

# --- Configuration Filenames ---
DISTETA_CONFIG_FILENAME = "config_disteta.yaml"
REPORT_GEN_CONFIG_FILENAME = "config_report_generator.yaml"

# --- Internal Directory Names (used within a run-specific output folder) ---
DATA_DIR_NAME = "data"
GRAPHICS_DIR_NAME = "graphics"
LOGS_DIR_NAME = "logs"
REPORTS_DIR_NAME = "reports"