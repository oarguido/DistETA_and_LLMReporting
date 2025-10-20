# DistETA-AIDI: Distributional ETA Analysis & LLM Reporting

This project provides a robust, two-part pipeline for analyzing time-based operational data and generating insightful, human-readable reports using Large Language Models (LLMs). It moves beyond simple averages to understand the full *distribution* of outcomes, identifying distinct, recurring operational patterns.

**This software is provided for demonstration and evaluation purposes only. It is not intended for commercial use. See the [LICENSE](LICENSE) file for more details.**

For a detailed explanation of the project's workflow, architecture, and configuration, please see the [**Detailed Workflow Documentation**](docs/WORKFLOW.md).

## Getting Started

### Prerequisites

-   Python 3.10+
-   Docker Desktop (recommended)
-   An API key for Google Gemini (if using the Gemini provider).
-   Ollama installed and running with a multimodal model (if using the Ollama provider).

### Installation & Usage (Docker)

1.  **Clone the repository**
2.  **Create a `.env` file** in the project root and add your Google API key:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```
3.  **Run the full pipeline (analysis and reporting):**
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
3.  **Run the analysis:**
    ```bash
    python -m src.disteta_aidi.main
    ```
4.  **Generate the report:**
    ```bash
    python -m src.report_generator.main
    ```

## Project Structure

*   **`config/`**: YAML configuration files for the analysis and reporting.
*   **`data/`**: Input data files.
*   **`docs/`**: Detailed project documentation.
*   **`output/`**: Generated reports and analysis results.
*   **`src/`**: Main source code.
*   **`tests/`**: Pytest tests.