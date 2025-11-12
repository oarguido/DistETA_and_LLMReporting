"""
This module provides the real-time data processing and visualization component of the DistETA project.

It defines the `DistetaStreaming` class, which is responsible for loading artifacts
from a completed batch analysis run, processing new, individual data instances as
they arrive, and generating detailed prediction plots.

The primary workflow is as follows:
1.  **Artifact Loading**: On initialization, it finds the latest batch run and loads
    the analysis settings, cluster profiles, and trained XGBoost models.
2.  **Instance Processing**: The `process_instance` method takes a new data point,
    predicts its cluster using the appropriate model, and generates a plot.
3.  **Visualization**: A sophisticated 2-row plot is created, showing both an
    overview of the assigned cluster and a detailed, three-part decomposed view.

This module is designed to be run as a standalone script to simulate a real-time
stream of data from a CSV file, generating a new plot for each row.

Execution:
    $ python -m src.disteta_streaming.main
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from plotly.subplots import make_subplots

from .. import constants
from ..disteta_batch.main import AnalysisSettings
from ..disteta_batch.utils.constants import AGG_DF_PREFIX
from ..disteta_batch.utils.data_utils import load_config

# --- Logger Setup ---
logger = logging.getLogger(__name__)


@dataclass
class StreamingArtifacts:
    """A data container for all the artifacts required for the streaming process."""

    analysis_settings: AnalysisSettings
    quantization_params: Dict
    cluster_profiles: Dict
    xgboost_models: Dict[str, xgb.XGBClassifier]


def plot_decomposed_prediction(
    cluster_profile: Dict,
    bin_edges: np.ndarray,
    output_path: str,
    subplot_titles: List[str],
    ranges: Dict[str, Tuple[int, int]],
    main_title: str,
    xaxis_title: str,
    instance_info: Optional[Dict] = None,
):
    """
    Generates and saves a comprehensive 2-row prediction visualization.

    The top row shows the complete distribution for the predicted cluster. The
    bottom row shows three subplots, each focusing on a different section
    (head, body, tail) of the distribution.

    Args:
        cluster_profile: The profile data for the predicted cluster.
        bin_edges: The bin edges used for quantization.
        output_path: The file path to save the generated PNG image.
        subplot_titles: The titles for all four subplots.
        ranges: A dictionary defining the bin ranges for the decomposed subplots.
        main_title: The main title for the entire figure.
        xaxis_title: The title for the shared x-axis of the decomposed plots.
        instance_info: A dictionary containing the raw value and binned value of
            the current instance being plotted.
    """
    try:
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=subplot_titles,
            specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        bin_counts = [b["count"] for b in cluster_profile["bin_profiles"]]
        bin_indices = np.arange(len(bin_counts))

        # --- Top Row: Overall Distribution ---
        fig.add_trace(
            go.Bar(
                x=bin_indices,
                y=bin_counts,
                name="Cluster Dist.",
                showlegend=False,
                marker_color="#1f77b4",
            ),
            row=1,
            col=1,
        )

        # --- Bottom Row: Decomposed Plots (Base) ---
        for i, (name, (start_idx, end_idx)) in enumerate(ranges.items()):
            col = i + 1
            fig.add_trace(
                go.Bar(
                    x=bin_indices,
                    y=bin_counts,
                    name=name,
                    showlegend=False,
                    marker_color="#1f77b4",
                ),
                row=2,
                col=col,
            )
            fig.update_xaxes(range=[start_idx - 0.5, end_idx - 0.5], row=2, col=col)

        # --- Overlays for the current instance ---
        if instance_info:
            instance_value = instance_info["value"]
            binned_value = instance_info["binned_value"]
            bin_lower_bound = bin_edges[binned_value]
            bin_upper_bound = bin_edges[binned_value + 1]

            annotation_text = (
                f"<b>Current Instance</b><br>"
                f"Raw Value: {instance_value:.2f}<br>"
                f"Bin Range: {bin_lower_bound:.2f} - {bin_upper_bound:.2f}"
            )

            # Add vertical line and annotation to the top plot.
            fig.add_vline(
                x=binned_value,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=1,  # type: ignore
                col=1,  # type: ignore
            )
            fig.add_annotation(
                text=annotation_text,
                xref="x1",
                yref="y domain",
                x=binned_value,
                y=0.95,
                xanchor="left",
                xshift=10,
                showarrow=False,
                align="left",
                font=dict(size=10),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="rgba(255,255,255,0.7)",
                borderwidth=1,
                row=1,
                col=1,
            )

            # Add a second annotation for the prediction probabilities.
            if "probabilities" in instance_info and "model_classes" in instance_info:
                probs_text = "<b>Cluster Probabilities:</b><br>" + "<br>".join(
                    [
                        f"- Cluster {int(cls)}: {prob:.1%}"
                        for cls, prob in zip(
                            instance_info["model_classes"],
                            instance_info["probabilities"],
                        )
                    ]
                )
                fig.add_annotation(
                    text=probs_text,
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    align="left",
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="rgba(255,255,255,0.7)",
                    borderwidth=1,
                )

            # Add vline/annotation to the correct subplot in the bottom row.
            for i, (name, (start_idx, end_idx)) in enumerate(ranges.items()):
                if start_idx <= binned_value < end_idx:
                    col = i + 1
                    fig.add_vline(
                        x=binned_value,
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        row=2,  # type: ignore
                        col=col,  # type: ignore
                    )
                    fig.add_annotation(
                        text=annotation_text,
                        xref=f"x{col + 1}",
                        yref="y domain",
                        x=binned_value,
                        y=0.95,
                        xanchor="left",
                        xshift=10,
                        showarrow=False,
                        align="left",
                        font=dict(size=10),
                        bgcolor="rgba(0,0,0,0.7)",
                        bordercolor="rgba(255,255,255,0.7)",
                        borderwidth=1,
                        row=2,
                        col=col,
                    )
                    break  # Stop after finding the correct subplot

        # --- Final Layout Updates ---
        fig.update_layout(
            title_text=main_title,
            template="plotly_dark",
            width=1200,
            height=800,
        )
        fig.update_xaxes(title_text=xaxis_title, row=2, col=2)  # Center bottom title

        # Y-Axis for top row (linear).
        fig.update_yaxes(title_text="Count", type="linear", row=1, col=1)

        # Y-Axis for bottom row (log scale).
        fig.update_yaxes(title_text="Count (log scale)", type="log", row=2, col=1)
        fig.update_yaxes(showticklabels=False, type="log", row=2, col=2)
        fig.update_yaxes(
            title_text="Count (log scale)",
            side="right",
            ticklabelposition="outside right",
            type="log",
            row=2,
            col=3,
        )

        fig.write_image(output_path)
        logger.info(f"Saved prediction plot to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


class DistetaStreaming:
    """Orchestrates the real-time DistETA analysis and plotting pipeline."""

    def __init__(self, artifacts: StreamingArtifacts):
        """
        Initializes the DistetaStreaming processor.

        Args:
            artifacts: A dataclass containing all loaded artifacts from a batch run.
        """
        self.artifacts = artifacts
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        os.makedirs(constants.STREAMING_PLOTS_DIR, exist_ok=True)

    @classmethod
    def from_latest_run(cls) -> "DistetaStreaming":
        """
        A factory method to create a DistetaStreaming instance from the latest batch run.

        Returns:
            An instance of DistetaStreaming, ready to process instances.
        """
        latest_run_dir = cls._find_latest_run_dir()
        artifacts = cls._load_artifacts(latest_run_dir)
        return cls(artifacts)

    @staticmethod
    def _find_latest_run_dir() -> str:
        """
        Finds the most recent, valid run directory in the output folder.

        Returns:
            The absolute path to the latest run directory.

        Raises:
            FileNotFoundError: If no run directories are found.
        """
        output_dir = constants.OUTPUT_DIR
        all_run_dirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and "run" in d
        ]
        if not all_run_dirs:
            raise FileNotFoundError("No run directories found in the output folder.")
        latest_run_dir = max(all_run_dirs)
        return os.path.join(output_dir, latest_run_dir)

    @staticmethod
    def _load_artifacts(run_dir: str) -> StreamingArtifacts:
        """
        Loads all necessary artifacts from a specific run directory.

        This includes the analysis configuration, cluster profiles, quantization
        parameters, and trained XGBoost models.

        Args:
            run_dir: The path to the batch run directory.

        Returns:
            A StreamingArtifacts dataclass instance.

        Raises:
            FileNotFoundError: If essential artifacts like models are not found.
        """
        data_dir = os.path.join(run_dir, constants.DATA_DIR_NAME)
        models_dir = constants.MODELS_DIR

        # Load the original run summary to identify the config used.
        run_summary_path = os.path.join(data_dir, "run_summary.json")
        run_summary = load_config(run_summary_path)
        config_name = run_summary["config_name_used"]

        # Load the full configuration to get all settings.
        config_path = os.path.join(
            constants.CONFIG_DIR, constants.DISTETA_CONFIG_FILENAME
        )
        config = load_config(config_path)
        final_config = {
            **config.get("common_settings", {}),
            **config.get(config_name, {}),
        }
        analysis_settings = AnalysisSettings.from_dict(final_config)

        # Load cluster profiles.
        cluster_profiles_path = os.path.join(data_dir, "final_cluster_profiles.json")
        with open(cluster_profiles_path, "r") as f:
            cluster_profiles = json.load(f)

        # Extract quantization parameters from profiles.
        quantization_params = {}
        for segment, profile in cluster_profiles.items():
            quantization_params[segment] = {
                "bin_edges": np.array(profile["bin_edges"]),
                "n_bins": profile["quantization_bins"],
            }

        # Load all relevant XGBoost models.
        xgboost_models = {}
        for model_file in os.listdir(models_dir):
            if (
                model_file.startswith(config_name)
                and model_file.endswith(".json")
                and not model_file.endswith("_metadata.json")
            ):
                segment_name = (
                    model_file.replace(f"{config_name}_", "")
                    .replace(".json", "")
                    .replace(f"{AGG_DF_PREFIX}", "")
                )
                model = xgb.XGBClassifier()
                model.load_model(os.path.join(models_dir, model_file))
                xgboost_models[segment_name] = model
                logger.info(f"Loaded model for segment: {segment_name}")

        if not xgboost_models:
            raise FileNotFoundError(f"No models found for config '{config_name}'.")

        return StreamingArtifacts(
            analysis_settings=analysis_settings,
            quantization_params=quantization_params,
            cluster_profiles=cluster_profiles,
            xgboost_models=xgboost_models,
        )

    def process_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single data instance: predicts its cluster and generates a plot.

        Args:
            instance: A dictionary representing the data instance.

        Returns:
            A dictionary containing the prediction results and the path to the plot.
        """
        self.logger.info(f"Processing instance: {instance}")
        start_time = time.time()

        instance_config = instance.get("config", {})
        if not instance_config:
            self.logger.warning("Instance has no 'config' key. Skipping.")
            return {"error": "Missing 'config' in instance."}

        prediction_result = self.predict_cluster(instance_config)

        plot_path = None
        predicted_cluster_id = None
        if prediction_result:
            predicted_cluster_id, pred_probabilities = prediction_result
            plot_path = self._plot_instance_in_cluster(
                instance_config, predicted_cluster_id, pred_probabilities
            )

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "instance": instance,
            "predicted_cluster_id": predicted_cluster_id,
            "prediction_plot_path": plot_path,
            "processing_time": f"{processing_time:.4f}s",
        }

    def predict_cluster(
        self, instance_config: Dict[str, Any]
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        Predicts the cluster for a given data instance using the appropriate model.

        Args:
            instance_config: A dictionary containing the feature values for the instance.

        Returns:
            A tuple containing the predicted cluster ID and the array of prediction
            probabilities, or None if prediction fails.
        """
        try:
            grouping_col = self.artifacts.analysis_settings.grouping_col_config
            continuous_col = self.artifacts.analysis_settings.continuous_cols_config[0]

            if grouping_col is None:
                self.logger.error(
                    "Grouping column configuration is None. Cannot predict cluster."
                )
                return None

            if grouping_col not in instance_config:
                self.logger.warning(
                    f"Grouping column '{grouping_col}' not in instance."
                )
                return None

            group_val = instance_config[cast(str, grouping_col)]
            segment_name = f"{group_val}_{continuous_col}"

            model = self.artifacts.xgboost_models.get(segment_name)
            q_params = self.artifacts.quantization_params.get(segment_name)

            if not model or not q_params:
                self.logger.error(f"No model or params for segment '{segment_name}'.")
                return None

            raw_value = instance_config[continuous_col]
            bin_edges = q_params["bin_edges"]
            n_bins = q_params["n_bins"]

            # One-hot encode the binned value for the model.
            bin_index = np.digitize(raw_value, bins=bin_edges) - 1
            bin_index = np.clip(bin_index, 0, n_bins - 1)

            column_names = [f"c_{i + 1}" for i in range(n_bins)]
            feature_df = pd.DataFrame(np.zeros((1, n_bins)), columns=column_names)  # type: ignore
            if 0 <= bin_index < n_bins:
                feature_df.iloc[0, bin_index] = 1

            pred_probabilities = model.predict_proba(feature_df)[0]
            predicted_class_index = np.argmax(pred_probabilities)
            predicted_cluster_id = model.classes_[predicted_class_index]

            return int(predicted_cluster_id), pred_probabilities

        except Exception as e:
            self.logger.error(
                f"Failed to predict cluster for instance: {e}", exc_info=True
            )
            return None

    def _plot_instance_in_cluster(
        self,
        instance_config: Dict[str, Any],
        cluster_id: int,
        pred_probabilities: np.ndarray,
    ) -> Optional[str]:
        """
        Generates and saves the detailed plot for a predicted instance.

        Args:
            instance_config: The feature dictionary for the instance.
            cluster_id: The predicted cluster ID for the instance.
            pred_probabilities: An array of prediction probabilities for each cluster.

        Returns:
            The absolute path to the saved plot image, or None on failure.
        """
        try:
            grouping_col = self.artifacts.analysis_settings.grouping_col_config
            continuous_col = self.artifacts.analysis_settings.continuous_cols_config[0]
            group_val = instance_config[cast(str, grouping_col)]
            segment_name = f"{group_val}_{continuous_col}"

            cluster_profiles = self.artifacts.cluster_profiles.get(segment_name)
            q_params = self.artifacts.quantization_params.get(segment_name)
            model = self.artifacts.xgboost_models.get(segment_name)

            if not cluster_profiles or not q_params or not model:
                self.logger.warning(f"Artifacts missing for segment '{segment_name}'.")
                return None

            cluster_profile = next(
                (
                    c
                    for c in cluster_profiles["clusters"]
                    if c["cluster_id"] == cluster_id
                ),
                None,
            )

            if not cluster_profile:
                self.logger.warning(f"No profile found for cluster {cluster_id}.")
                return None

            bin_edges = q_params["bin_edges"]
            n_bins = q_params["n_bins"]
            instance_value = instance_config[continuous_col]

            # --- 1. Partition bins and create dynamic titles ---
            partition_size = n_bins // 3
            p1_end_idx = partition_size
            p2_end_idx = 2 * partition_size

            ranges = {
                f"Bins 0-{p1_end_idx - 1}": (0, p1_end_idx),
                f"Bins {p1_end_idx}-{p2_end_idx - 1}": (p1_end_idx, p2_end_idx),
                f"Bins {p2_end_idx}-{n_bins - 1}": (p2_end_idx, n_bins),
            }

            main_title = f"Instance Prediction: Raw Value {instance_value:.2f} classified as Cluster {cluster_id}"
            subplot_titles = ["Overall Distribution"] + list(ranges.keys())
            xaxis_title = self.artifacts.analysis_settings.quantized_axis_label_config

            # --- 2. Create and save the plot ---
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_pred_{segment_name}_c{cluster_id}.png"
            output_path = os.path.join(constants.STREAMING_PLOTS_DIR, output_filename)

            binned_value = np.clip(
                np.digitize(instance_value, bins=bin_edges) - 1, 0, n_bins - 1
            )
            instance_info = {
                "value": instance_value,
                "binned_value": binned_value,
                "probabilities": pred_probabilities,
                "model_classes": model.classes_,
            }

            plot_decomposed_prediction(
                cluster_profile=cluster_profile,
                bin_edges=bin_edges,
                output_path=output_path,
                ranges=ranges,
                main_title=main_title,
                subplot_titles=subplot_titles,
                xaxis_title=xaxis_title,
                instance_info=instance_info,
            )

            return output_path

        except Exception as e:
            self.logger.error(f"Failed during plot generation: {e}", exc_info=True)
            return None


def simulate_streaming():
    """
    Simulates a real-time data stream by processing a CSV file row by row.

    This function initializes the streaming processor, reads a sample data file,
    and processes each row as if it were a new, incoming data instance, generating
    a plot for each one.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        streaming_processor = DistetaStreaming.from_latest_run()

        data_path = os.path.join(constants.DATA_DIR, "truck_arrival_data.csv")
        for chunk in pd.read_csv(data_path, chunksize=1):
            instance_config = chunk.to_dict(orient="records")[0]
            instance = {
                "config": instance_config,
                "context": {"source": "truck_arrival_data.csv"},
            }
            result = streaming_processor.process_instance(instance)
            logger.info(f"Result: {result}")
            time.sleep(1)  # Pause to simulate a real-time interval.

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to initialize streaming processor: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during streaming simulation: {e}")


def main():
    """
    Main entry point for the streaming module.

    Parses command-line arguments to either run the full end-to-end pipeline
    (batch analysis, report generation, and streaming) or just the streaming
    simulation.
    """
    # Configure logging at the very beginning to capture logs from all modules.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run the DistETA streaming simulation or the full pipeline."
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full end-to-end pipeline: batch analysis, report generation, and then streaming.",
    )
    args = parser.parse_args()

    if args.run_pipeline:
        logger.info("--- Starting Full End-to-End Pipeline ---")

        # 1. Run Batch Analysis
        logger.info("--- Step 1: Running Batch Analysis ---")
        from ..disteta_batch.main import run_all_analyses

        run_all_analyses()
        logger.info("--- Batch Analysis Complete ---")

        # 2. Generate Report
        logger.info("--- Step 2: Generating Analysis Report ---")
        from ..report_generator.main import AgnosticReportGenerator, load_report_config
        import webbrowser

        html_report_path = None
        try:
            report_config = load_report_config()
            generator = AgnosticReportGenerator(
                config=report_config, base_output_dir=constants.OUTPUT_DIR
            )
            relative_report_path = generator.generate_report()
            if relative_report_path:
                html_report_path = os.path.join(
                    constants.OUTPUT_DIR, relative_report_path
                )
                logger.info("--- Report Generation Complete ---")
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)

        # Open the generated report in the browser if it exists.
        if html_report_path and os.path.exists(html_report_path):
            absolute_report_path = os.path.realpath(html_report_path)
            logger.info(f"Opening report in browser: {absolute_report_path}")
            webbrowser.open(f"file://{absolute_report_path}")
            time.sleep(2)  # Give the browser a moment to open.

        # 3. Run Streaming Simulation
        logger.info("--- Step 3: Starting Streaming Simulation ---")
        simulate_streaming()
        logger.info("--- Streaming Simulation Complete ---")

    else:
        # Default behavior: just run the streaming simulation.
        simulate_streaming()


if __name__ == "__main__":
    import argparse

    main()
