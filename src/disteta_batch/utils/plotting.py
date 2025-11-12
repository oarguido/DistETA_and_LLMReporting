"""
This module provides functions for generating all visualizations for the DistETA analysis pipeline using Plotly.

It includes high-level functions for creating:
1.  **Initial Distributions**: Histograms of continuous variables, optionally grouped.
2.  **K-Means Analysis Plots**: A 3-panel plot including Silhouette analysis, a 2D PCA
    projection of clusters, and the WCSS Elbow Method plot.
3.  **Final Cluster Profiles**: A grid of bar plots showing the distribution for
    each discovered cluster, including High-Density Region (HDR) information.

All plots are designed to be saved as artifacts in the batch analysis output.
"""

import logging
from typing import Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from . import clustering_utils
from .constants import (
    ALL_DATA_GROUP_KEY,
    CLASS_COL_NUM,
    CLUSTER_COL,
    NAN_GROUP_KEY,
    QUANT_PREFIX,
)

logger = logging.getLogger(__name__)


def _generate_plot_title(
    base_plot_title: str,
    df_name_suffix: str,
    group_key: Optional[str],
    feature_name: str,
    n_clusters: Optional[int] = None,
    grouping_column_name: Optional[str] = None,
    value_mapping_config: Optional[dict] = None,
    column_name_mapping: Optional[dict] = None,
    hdr_threshold_percentage: Optional[float] = None,
) -> str:
    """
    Generates a formatted, descriptive title for a plot.

    This function dynamically constructs a title string based on the analysis
    parameters, applying user-friendly mappings for column and value names if provided.

    Args:
        base_plot_title: The base text for the title (e.g., "Cluster Distributions").
        df_name_suffix: The internal suffix used for the data segment.
        group_key: The specific group being plotted (e.g., a specific port name).
        feature_name: The continuous feature being plotted.
        n_clusters: The number of clusters (K), if applicable.
        grouping_column_name: The name of the column used for grouping.
        value_mapping_config: A dictionary to map grouping values to display names.
        column_name_mapping: A dictionary to map feature names to display names.
        hdr_threshold_percentage: The HDR threshold, if used.

    Returns:
        A formatted title string for the plot.
    """
    try:
        display_group_name = None
        if group_key is not None and group_key != ALL_DATA_GROUP_KEY:
            display_group_name = group_key
            if (
                value_mapping_config
                and grouping_column_name in value_mapping_config
                and isinstance(value_mapping_config.get(grouping_column_name), dict)
                and group_key in value_mapping_config[grouping_column_name]
            ):
                display_group_name = value_mapping_config[grouping_column_name][
                    group_key
                ]

        display_time_type = feature_name
        if column_name_mapping and feature_name in column_name_mapping:
            display_time_type = column_name_mapping[feature_name]

        if display_group_name:
            title = f"{base_plot_title} for Group '{display_group_name}' ({display_time_type})"
        else:
            title = f"{base_plot_title} for '{display_time_type}'"

        if n_clusters is not None:
            title += f", K = {n_clusters}"
        if hdr_threshold_percentage is not None:
            actual_hdr_percentage = 100.0 - hdr_threshold_percentage
            title += f" (Top {actual_hdr_percentage:.1f}% HDR)"
        return title
    except Exception as e:
        logger.warning(f"Error generating title details for '{df_name_suffix}': {e}")
        fallback_title = f"{base_plot_title} (Error in Title)"
        if n_clusters is not None:
            fallback_title += f", K = {n_clusters}"
        if hdr_threshold_percentage is not None:
            actual_hdr_percentage = 100.0 - hdr_threshold_percentage
            fallback_title += f" (Top {actual_hdr_percentage:.1f}% HDR)"
        return fallback_title


def plot_continuous_histograms(
    df: pd.DataFrame,
    continuous_columns: List[str],
    grouping_column_name: Optional[str] = None,
    value_mapping_config: Optional[dict] = None,
    column_name_mapping: Optional[dict] = None,
    x_axis_label: str = "Value",
    save_non_interactive: bool = False,
) -> Optional[go.Figure]:
    """
    Plots histograms of continuous variables, creating subplots for each group.

    Args:
        df: The DataFrame containing the data to plot.
        continuous_columns: A list of continuous column names to generate histograms for.
        grouping_column_name: The name of the column to group by for creating subplots.
        value_mapping_config: A dictionary to map grouping values to display names.
        column_name_mapping: A dictionary to map column names to display names.
        x_axis_label: The label for the x-axis.
        save_non_interactive: If True, returns the figure object instead of showing it.

    Returns:
        A Plotly Figure object, or None if no columns are provided.
    """
    if not continuous_columns:
        logger.warning("No continuous columns provided for plotting histograms.")
        return None

    if grouping_column_name and grouping_column_name in df.columns:
        try:
            groups = sorted(df[grouping_column_name].dropna().unique().tolist())
            if bool(df[grouping_column_name].isnull().any()):
                groups.append(np.nan)
            is_grouped = True
        except Exception as e:
            logger.warning(f"Error getting unique groups: {e}. Plotting overall data.")
            groups, is_grouped, grouping_column_name = [None], False, None
    else:
        groups, is_grouped, grouping_column_name = [None], False, None

    n_groups = len(groups)
    n_cols_per_group = len(continuous_columns)
    fig = make_subplots(
        rows=n_groups,
        cols=n_cols_per_group,
        subplot_titles=["Analyzing..." for _ in range(n_groups * n_cols_per_group)],
        vertical_spacing=0.2,
    )

    colors = px.colors.qualitative.Plotly
    for i, current_group_value in enumerate(groups):
        group_df = df
        is_nan_group = isinstance(current_group_value, float) and np.isnan(
            current_group_value
        )

        if is_grouped and grouping_column_name is not None:
            group_df = (
                df[df[grouping_column_name].isnull()]
                if is_nan_group
                else df[df[grouping_column_name] == current_group_value]
            )

        count_str = f"n = {len(group_df):,}"
        for j, cont_col in enumerate(continuous_columns):
            subplot_row, subplot_col = i + 1, j + 1

            display_col_name = (
                column_name_mapping.get(cont_col, cont_col)
                if column_name_mapping
                else cont_col
            )
            display_group_name = (
                str(current_group_value) if not is_nan_group else NAN_GROUP_KEY
            )
            if value_mapping_config and grouping_column_name in value_mapping_config:
                lookup_key = NAN_GROUP_KEY if is_nan_group else current_group_value
                display_group_name = value_mapping_config[grouping_column_name].get(
                    lookup_key, display_group_name
                )

            title_line1 = f"{display_col_name}"
            if is_grouped:
                title_line1 += f" for {display_group_name}"

            has_data = (
                cont_col in group_df.columns
                and cast(pd.Series, group_df[cont_col]).notna().any()
            )
            if not has_data:
                title_line1 += " (No Data)"
                fig.update_xaxes(showticklabels=False, row=subplot_row, col=subplot_col)
                fig.update_yaxes(showticklabels=False, row=subplot_row, col=subplot_col)
            else:
                try:
                    fig.add_trace(
                        go.Histogram(
                            x=group_df[cont_col],
                            name=f"{display_col_name} ({display_group_name})",
                            marker_color=colors[i % len(colors)],
                            histnorm="probability density",
                            showlegend=False,
                        ),
                        row=subplot_row,
                        col=subplot_col,
                    )
                except Exception as e:
                    logger.error(
                        f"Error plotting histogram for {cont_col} in group {display_group_name}: {e}"
                    )
                    title_line1 = f"Error plotting {display_col_name}"

            # Update subplot title after adding trace
            full_subplot_title = f"{title_line1}<br>{count_str}"
            fig.update_xaxes(title_text=x_axis_label, row=subplot_row, col=subplot_col)
            fig.update_yaxes(title_text="Density", row=subplot_row, col=subplot_col)

            annotation_index = i * n_cols_per_group + j
            layout_obj = cast(go.Layout, fig.layout)
            if layout_obj.annotations and annotation_index < len(
                layout_obj.annotations  # type: ignore
            ):
                layout_obj.annotations[annotation_index].text = full_subplot_title  # type: ignore

    # Update main figure layout
    overall_fig_title = "Distribution of Continuous Variables"
    if grouping_column_name:
        overall_fig_title += f" (Grouped by {grouping_column_name})"

    fig.update_layout(
        title_text=overall_fig_title,
        height=max(500, n_groups * 350),
        width=max(800, n_cols_per_group * 450),
        bargap=0.1,
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=120, b=50, l=50, r=50),
    )

    layout_obj = cast(go.Layout, fig.layout)
    if layout_obj.annotations:
        for annotation in layout_obj.annotations:
            if annotation.font:
                annotation.font.size = 12

    if not save_non_interactive:
        fig.show()
    return fig


def _plot_silhouette(
    fig: go.Figure,
    X: pd.DataFrame | np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    silh_scores: Dict[int, float],
):
    """Inner function to render the silhouette plot onto a subplot figure."""
    n_samples = X.shape[0]
    silhouette_avg = silh_scores.get(n_clusters, np.nan)

    layout_obj = cast(go.Layout, fig.layout)

    if pd.notna(silhouette_avg) and 1 < n_clusters < n_samples:
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower, cluster_colors = 0, px.colors.qualitative.Plotly
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            fig.add_trace(
                go.Bar(
                    y=np.arange(y_lower, y_upper),
                    x=ith_cluster_silhouette_values,
                    name=f"Cluster {i}",
                    orientation="h",
                    marker_color=cluster_colors[i % len(cluster_colors)],
                    marker_line_width=0,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=-0.05,
                y=(y_lower + y_upper) / 2,
                text=str(i),
                showarrow=False,
                xref="x domain",
                yref="y",
                row=1,
                col=1,
            )
            y_lower = y_upper

        fig.add_vline(
            x=silhouette_avg,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=1,  # type: ignore
            col=1,  # type: ignore
        )
        fig.update_xaxes(
            title_text="Silhouette Coefficient", range=[-0.1, 1.0], row=1, col=1
        )
        fig.update_yaxes(
            title=dict(text="Cluster Samples", standoff=30),
            showticklabels=False,
            row=1,
            col=1,
        )

        # Update the specific subplot title with the average score
        if layout_obj.annotations and len(layout_obj.annotations) > 0:  # type: ignore
            layout_obj.annotations[
                0
            ].text = f"Silhouette Plot (Avg: {silhouette_avg:.2f})"  # type: ignore
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Score N/A for K={n_clusters}",
            showarrow=False,
            row=1,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        if layout_obj.annotations and len(layout_obj.annotations) > 0:  # type: ignore
            layout_obj.annotations[
                0
            ].text = f"Silhouette Plot (K={n_clusters} Invalid/NA)"  # type: ignore


def _plot_scatter_2d(
    fig: go.Figure,
    X: pd.DataFrame | np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
):
    """Inner function to render a 2D PCA projection of clustered data."""
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    cluster_colors = px.colors.qualitative.Plotly

    layout_obj = cast(go.Layout, fig.layout)

    if X_vals.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=10)
        X_2d = pca.fit_transform(X_vals)
        fig.add_trace(
            go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode="markers",
                marker=dict(
                    color=[
                        cluster_colors[label % len(cluster_colors)]
                        for label in cluster_labels
                    ],
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color="DarkSlateGrey"),
                ),
                showlegend=False,
                name="Data Points",
            ),
            row=1,
            col=2,
        )
        try:
            clusterer = clustering_utils.perform_clustering(X, n_clusters)
            centers = clusterer.cluster_centers_
            centers_2d = pca.transform(centers)
            fig.add_trace(
                go.Scatter(
                    x=centers_2d[:, 0],
                    y=centers_2d[:, 1],
                    mode="markers+text",
                    marker=dict(
                        color="white", size=12, line=dict(color="black", width=1)
                    ),
                    text=[str(i) for i in range(n_clusters)],
                    textposition="middle center",
                    textfont=dict(color="black", size=10),
                    showlegend=False,
                    name="Centers",
                ),
                row=1,
                col=2,
            )
        except Exception as e:
            logger.warning(f"Could not plot cluster centers for K={n_clusters}: {e}")
        fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
        fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)
        if layout_obj.annotations and len(layout_obj.annotations) > 1:  # type: ignore
            layout_obj.annotations[1].text = "Clustered Data (PCA Projection)"  # type: ignore
    else:
        fig.add_annotation(
            x=0.5, y=0.5, text="Data has < 2 dims", showarrow=False, row=1, col=2
        )
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        if layout_obj.annotations and len(layout_obj.annotations) > 1:  # type: ignore
            layout_obj.annotations[1].text = "Clustered Data (1D)"  # type: ignore


def _plot_elbow(
    fig: go.Figure,
    wcss_values: List[float],
    range_n_clusters: Iterable[int],
    n_clusters: int,
):
    """Inner function to render the WCSS Elbow Method plot."""
    k_list = list(range_n_clusters)
    valid_points = [(k_list[i], w) for i, w in enumerate(wcss_values) if pd.notna(w)]

    layout_obj = cast(go.Layout, fig.layout)

    if valid_points:
        valid_k, valid_wcss = zip(*valid_points)
        fig.add_trace(
            go.Scatter(
                x=valid_k,
                y=valid_wcss,
                mode="lines+markers",
                name="WCSS",
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        if n_clusters in valid_k:
            current_wcss = valid_wcss[valid_k.index(n_clusters)]
            fig.add_trace(
                go.Scatter(
                    x=[n_clusters],
                    y=[current_wcss],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name=f"Current K={n_clusters}",
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
            fig.add_annotation(
                x=n_clusters,
                y=current_wcss,
                text=f"K={n_clusters}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
                row=1,
                col=3,
            )

        fig.update_xaxes(
            title_text="Number of Clusters (K)", tickvals=k_list, row=1, col=3
        )
        fig.update_yaxes(title_text="WCSS", row=1, col=3)
        if layout_obj.annotations and len(layout_obj.annotations) > 2:  # type: ignore
            layout_obj.annotations[2].text = "Elbow Method for KMeans"  # type: ignore
    else:
        fig.add_annotation(
            x=0.5, y=0.5, text="No WCSS data", showarrow=False, row=1, col=3
        )
        fig.update_xaxes(showticklabels=False, row=1, col=3)
        fig.update_yaxes(showticklabels=False, row=1, col=3)
        if layout_obj.annotations and len(layout_obj.annotations) > 2:  # type: ignore
            layout_obj.annotations[2].text = "Elbow Method (No Valid Data)"  # type: ignore


def plot_silhouette_and_elbow(
    X: pd.DataFrame | np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    df_name_suffix: str,
    group_key: Optional[str],
    feature_name: str,
    wcss_values: List[float],
    range_n_clusters: Iterable[int],
    silh_scores: Dict[int, float],
    value_mapping_config: Optional[dict] = None,
    grouping_column_name: Optional[str] = None,
    column_name_mapping: Optional[dict] = None,
    save_non_interactive: bool = False,
) -> go.Figure:
    """Generates a unified 3-panel plot for K-Means clustering analysis."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Silhouette Plot",
            "Clustered Data (2D Projection)",
            "Elbow Method",
        ),
    )

    # Populate each subplot
    _plot_silhouette(fig, X, cluster_labels, n_clusters, silh_scores)
    _plot_scatter_2d(fig, X, cluster_labels, n_clusters)
    _plot_elbow(fig, wcss_values, range_n_clusters, n_clusters)

    # Generate a comprehensive title for the entire figure
    full_title = _generate_plot_title(
        base_plot_title="KMeans Analysis",
        df_name_suffix=df_name_suffix,
        group_key=group_key,
        feature_name=feature_name,
        n_clusters=n_clusters,
        grouping_column_name=grouping_column_name,
        value_mapping_config=value_mapping_config,
        column_name_mapping=column_name_mapping,
    )

    # Final layout adjustments
    fig.update_layout(
        title_text=full_title,
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=80, b=50, l=50, r=50),
    )
    layout_obj = cast(go.Layout, fig.layout)
    if layout_obj.annotations:
        for annotation in layout_obj.annotations:
            if annotation.font:
                annotation.font.size = 12

    if not save_non_interactive:
        fig.show()
    return fig


def plot_cluster_distributions(
    df_cluster_c_sum: pd.DataFrame,
    df_name_suffix: str,
    group_key: Optional[str],
    feature_name: str,
    n_classes: int,
    value_mapping_config: Optional[dict] = None,
    grouping_column_name: Optional[str] = None,
    column_name_mapping: Optional[dict] = None,
    x_axis_label: str = "Quantized Bin",
    hdr_info_for_plot: Optional[Dict] = None,
    hdr_threshold_percentage: Optional[float] = None,
    save_non_interactive: bool = False,
) -> Optional[go.Figure]:
    """
    Generates a grid of bar plots showing the distribution for each final cluster.

    Also overlays High-Density Region (HDR) information if provided.

    Args:
        df_cluster_c_sum: DataFrame with final aggregated cluster profiles.
        df_name_suffix: A suffix for the plot title, identifying the data segment.
        group_key: The key for the group being plotted.
        feature_name: The name of the continuous feature being plotted.
        n_classes: The number of bins used in quantization.
        value_mapping_config: Dictionary to map group values to display names.
        grouping_column_name: The name of the column used for grouping.
        column_name_mapping: Dictionary to map column names to display names.
        x_axis_label: The label for the x-axis.
        hdr_info_for_plot: Dictionary containing HDR analysis results.
        hdr_threshold_percentage: The HDR threshold used.
        save_non_interactive: If True, returns the figure object.

    Returns:
        A Plotly Figure object, or None on error.
    """
    if CLUSTER_COL not in df_cluster_c_sum.columns:
        logger.error(f"'{CLUSTER_COL}' column missing in data for {df_name_suffix}.")
        return None
    try:
        id_vars = [CLUSTER_COL]
        value_vars = sorted(
            [col for col in df_cluster_c_sum.columns if col.startswith(QUANT_PREFIX)],
            key=lambda x: int(x.split("_")[1]),
        )
        if not value_vars:
            logger.error(f"No '{QUANT_PREFIX}*' columns found for {df_name_suffix}.")
            return None

        # Reshape data for plotting
        melted_df = df_cluster_c_sum.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="classes",
            value_name="count",
        )
        melted_df[CLASS_COL_NUM] = cast(
            pd.Series,
            pd.to_numeric(
                melted_df["classes"].str.replace(QUANT_PREFIX, ""), errors="coerce"
            ),
        ).astype(int)
    except Exception as e:
        logger.error(f"Error processing DataFrame for {df_name_suffix}: {e}")
        return None

    unique_clusters = sorted(melted_df[CLUSTER_COL].unique())
    if not unique_clusters:
        logger.warning(f"No valid clusters found to plot for {df_name_suffix}.")
        return None

    # Create subplot grid
    num_clusters = len(unique_clusters)
    num_cols = 2 if num_clusters > 1 else 1
    num_rows = int(np.ceil(num_clusters / num_cols))

    subplot_titles_with_n = [
        f"Cluster {cid}<br>n = {int(melted_df[melted_df[CLUSTER_COL] == cid]['count'].sum()):,}"
        for cid in unique_clusters
    ]
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles_with_n,
        vertical_spacing=0.25,
    )

    cluster_colors = px.colors.qualitative.T10
    for i, cluster_id in enumerate(unique_clusters):
        subplot_row, subplot_col = (i // num_cols) + 1, (i % num_cols) + 1
        cluster_data_for_plot = melted_df[melted_df[CLUSTER_COL] == cluster_id]

        fig.add_trace(
            go.Bar(
                x=cluster_data_for_plot[CLASS_COL_NUM],
                y=cluster_data_for_plot["count"],
                name=f"Cluster {cluster_id}",
                marker_color=cluster_colors[i % len(cluster_colors)],
                showlegend=False,
            ),
            row=subplot_row,
            col=subplot_col,
        )

        # Add HDR annotations if available
        if hdr_info_for_plot and str(cluster_id) in hdr_info_for_plot:
            cluster_hdr_data = hdr_info_for_plot.get(str(cluster_id), {})

            # Draw HDR threshold line
            hdr_line_y = cluster_hdr_data.get("hdr_threshold_count", 0)
            fig.add_shape(
                type="line",
                x0=0,
                y0=hdr_line_y,
                x1=1,
                y1=hdr_line_y,
                xref="x domain",
                yref="y",
                line=dict(color="cyan", width=2, dash="dot"),
                row=subplot_row,
                col=subplot_col,
            )

            # Annotate HDR intervals
            for interval in cluster_hdr_data.get("hdr_intervals", []):
                most_probable = interval.get("most_probable_orig_unit")
                start_edge, end_edge = (
                    interval.get("start_edge_orig_unit"),
                    interval.get("end_edge_orig_unit"),
                )
                annotation_x_bin = interval.get(
                    "peak_bin_label", interval.get("median_bin_label")
                )

                if all(
                    v is not None
                    for v in [most_probable, start_edge, end_edge, annotation_x_bin]
                ):
                    annotation_text = f"<b>HDR Mode: {most_probable:.2f} {interval.get('unit', '')}</b><br>Interval: [{start_edge:.2f} - {end_edge:.2f}]"
                    max_y_in_interval = interval.get("peak_count", 0)

                    fig.add_annotation(
                        x=annotation_x_bin,
                        y=max_y_in_interval,
                        text=annotation_text,
                        showarrow=True,
                        arrowhead=4,
                        arrowcolor="#c7c7c7",
                        ax=0,
                        ay=-40,
                        row=subplot_row,
                        col=subplot_col,
                        bgcolor="rgba(40, 40, 40, 0.85)",
                        bordercolor="cyan",
                        borderwidth=1,
                        font=dict(color="white", size=10),
                        align="left",
                    )

        # Configure axes for each subplot
        tick_step = 50 if n_classes > 200 else 20 if n_classes > 50 else 5
        tick_vals = list(range(tick_step, n_classes + 1, tick_step))
        fig.update_xaxes(
            title_text=x_axis_label,
            tickvals=tick_vals,
            tickangle=0,
            row=subplot_row,
            col=subplot_col,
        )
        fig.update_yaxes(title_text="Summed Count", row=subplot_row, col=subplot_col)

    # Generate main title with HDR info
    base_title = _generate_plot_title(
        base_plot_title="Cluster Distributions",
        df_name_suffix=df_name_suffix,
        group_key=group_key,
        feature_name=feature_name,
        column_name_mapping=column_name_mapping,
    )
    if hdr_threshold_percentage is not None:
        base_title += f" (Top {100.0 - hdr_threshold_percentage:.1f}% HDR)"

    fig.update_layout(
        title_text=base_title,
        bargap=0.1,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=100, b=50, l=80, r=50),
    )
    return fig
