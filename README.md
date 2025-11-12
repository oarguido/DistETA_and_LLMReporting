# DistETA: Distributional ETA Analysis, Real-Time Prediction & LLM Reporting

This project provides a robust, two-part pipeline for analyzing time-based operational data, making real-time predictions, and generating insightful, human-readable reports using Large Language Models (LLMs). It moves beyond simple averages to understand the full *distribution* of outcomes, identifying distinct, recurring operational patterns.

**This software is provided for demonstration and evaluation purposes only. It is not intended for commercial use. See the [LICENSE](LICENSE) file for more details.**

For a detailed explanation of the project's workflow, architecture, and configuration, please see the [**Detailed Workflow Documentation**](docs/WORKFLOW.md).

## Core Modules

1.  **`disteta_batch`**: Analyzes historical data to find distributional patterns using K-Means clustering and trains a classifier (XGBoost) to predict which pattern a new data point belongs to.
2.  **`disteta_streaming`**: Simulates a real-time environment, processing individual data points, predicting their cluster with the trained model, and generating a detailed plot. The plot visualizes the instance against its predicted cluster's distribution and displays the prediction probabilities for all clusters.
3.  **`report_generator`**: Creates a comprehensive HTML report from the batch analysis results.

## Getting Started

### Prerequisites

-   Python 3.10+
-   Docker Desktop (recommended)
-   An API key for Google Gemini.

### Installation & Usage (Docker)

1.  **Clone the repository**
2.  **Create a `.env` file** in the project root and add your Google API key:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```
3.  **Run the full pipeline (batch analysis and reporting):**
    ```bash
    docker-compose --profile analyze_report up --build
    ```
4.  **View the report** at [http://localhost:5001](http://localhost:5001).

### Local Installation

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the batch analysis and model training:**
    ```bash
    python -m src.disteta_batch.main
    ```
4.  **Run the real-time streaming simulation:**
    ```bash
    python -m src.disteta_streaming.main
    ```
5.  **Generate the summary report:**
    ```bash
    python -m src.report_generator.main
    ```

## Project Structure

*   **`config/`**: YAML configuration files for the analysis and reporting.
*   **`data/`**: Input data files.
*   **`docs/`**: Detailed project documentation.
*   **`models/`**: Saved XGBoost models and their metadata.
*   **`output/`**: Generated reports, analysis results, and streaming plots.
*   **`src/`**: Main source code.
*   **`tests/`**: Pytest tests.