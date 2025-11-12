# DistETA: Distributional Analysis, Real-Time Prediction & LLM Reporting

This project provides a robust, multi-stage pipeline for analyzing time-based operational data, making real-time predictions, and generating insightful, human-readable reports using Large Language Models (LLMs). It moves beyond simple averages to understand the full *distribution* of outcomes, identifying distinct, recurring operational patterns.

The pipeline is composed of three main components:

1.  **`disteta_batch` (The Analyzer & Trainer):** A configurable application that performs a distributional analysis on historical data. It uses K-Means clustering to identify distinct operational patterns and then trains an XGBoost classifier to predict which pattern a new data point belongs to.
2.  **`disteta_streaming` (The Predictor):** Simulates a real-time environment by processing individual data points as they arrive. It uses the pre-trained models to predict the cluster for each new instance and generates a plot visualizing the instance against its cluster's distribution.
3.  **`report_generator` (The Reporter):** An agnostic reporting engine that automatically finds the latest analysis output, synthesizes all data and plots using an LLM, and generates a polished, self-contained HTML report that explains the findings in natural language.

## Key Features

-   **Deep Distributional Analysis:** Analyzes the entire probability distribution of a variable, capturing nuances like skewness and multi-modality.
-   **Automated Pattern Discovery:** Automatically segments data into distinct, meaningful clusters representing different performance patterns.
-   **Predictive Modeling:** Trains an XGBoost model on the discovered patterns, enabling real-time classification of new data.
-   **Advanced Predictive Metrics:** Automatically calculates and saves key performance indicators, including accuracy and a weighted AUROC (Area Under the Receiver Operating Characteristic curve) score, providing a nuanced view of model performance.
-   **Probabilistic Predictions:** The classifier outputs the probability for each potential cluster, not just the final prediction. These probabilities are visualized in the real-time plot, offering deeper insight into prediction confidence.
-   **Real-Time Simulation:** A streaming module that processes individual instances and predicts their cluster.
-   **Configurable & Modular:** The entire workflow is controlled via central YAML files.
-   **Multi-LLM Reporting Engine:** Seamlessly switch between Google Gemini and local Ollama models for report generation.
-   **Reproducible Environment:** Ensures a consistent and reliable setup via `pip-tools` and Docker.

## System Workflow

### End-to-End Pipeline Execution

For convenience, the entire pipeline can be orchestrated with a single command. This is the recommended way to run the project.

```bash
python -m src.disteta_streaming.main --run-pipeline
```

This command executes the following steps in sequence:
1.  **Runs `disteta_batch`**: Performs the full analysis and model training.
2.  **Runs `report_generator`**: Generates the HTML report from the new artifacts.
3.  **Runs `disteta_streaming`**: Starts the real-time simulation using the newly trained models.

```mermaid
graph TD
    subgraph "Unified Command: --run-pipeline"
        A[Run src.disteta_streaming.main --run-pipeline] --> B["Step 1: Execute disteta_batch"];
        B --> C["Step 2: Execute report_generator"];
        C --> D["Step 3: Execute disteta_streaming"];
    end

    style A fill:#bde4ff,stroke:#367ab3,stroke-width:3px;
```

### High-Level Pipeline

The project operates as a multi-stage pipeline. First, `disteta_batch` runs to analyze data and train models. Then, the `disteta_streaming` and `report_generator` modules can consume these artifacts.

```mermaid
graph TD
    subgraph "Step 1: Analysis & Training"
        A[Run src.disteta_batch.main] --> B["Performs clustering & trains XGBoost models"];
        B --> C["Saves models, plots, & JSON results to 'output/run_folder' and 'models/'"];
    end

    subgraph "Step 2: Real-Time Prediction"
        D[Run src.disteta_streaming.main] --> E["Finds latest run artifacts"];
        E --> F["Loads models & cluster profiles"];
        F --> G["Processes single instance, predicts cluster"];
        G --> H["Generates plot"];
    end

    subgraph "Step 3: Reporting (Optional)"
        I[Run src.report_generator.main] --> J["Finds latest 'run_folder'"];
        J --> K["Gathers all plots & JSON artifacts"];
        K --> L["Calls LLM to generate narrative"];
        L --> M["Generates & serves HTML report"];
    end

    C -- "Provides artifacts for" --> F;
    C -- "Provides artifacts for" --> K;

    style A fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px;
    style D fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px;
    style I fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px;
```

### Detailed Batch Workflow (`disteta_batch.py`)

The `disteta_batch.py` application follows a sophisticated, multi-stage process to uncover hidden patterns and train a predictive model.

```mermaid
graph TD
    A[Start main] --> B{"Load Main Config File"};
    B --> C{"Get active_config_names"};
    C --> D{More active configs?};
    D -- "Yes" --> E["Instantiate DistetaBatch class"];
    E --> F["Call run_analysis()"];
    
    subgraph "run_analysis() Workflow"
        F --> S1["Stage 1: Prepare Data"];
        S1 --> S1_1["Load data & identify columns"];
        S1_1 --> S1_2["Plot initial distributions"];
        S1_2 --> S1_3["Encode categoricals & create 'comb' label"];
        S1_3 --> S1_4["Filter by min combination size"];

        S1_4 --> S2["Stage 2: Quantize & Aggregate"];
        S2 --> S2_1["Segment data by group"];
        S2_1 --> S2_2{More segments/features?};
        S2_2 -- "Yes" --> S2_3["Quantize feature into bins (df_quant)"];
        S2_3 --> S2_4["Aggregate by 'comb' label (df_agg)"];
        S2_4 --> S2_2;

        S2_2 -- "No" --> S3["Stage 3: Find Optimal K"];
        S3 --> S3_1[Use df_agg to find optimal clusters];
        S3_1 --> S4["Stage 4: Final Clustering & HDR"];
        S4 --> S4_1["Perform final clustering on df_agg"];
        S4_1 --> S4_2["Generate cluster profiles & HDR"];

        S4_2 --> S5["Stage 5: Train Classifier"];
        S5 --> S5_1["Map cluster IDs back to df_quant"];
        S5_1 --> S5_2["Train probabilistic XGBoost classifier"];
        S5_2 --> S5_3["Save model & metadata (with AUROC score) to /models"];

        S5_3 --> S6["Stage 6: Save All Outputs"];
        S6 --> S6_1["Save plots as PNG"];
        S6_1 --> S6_2["Save results as JSON"];
        S6_2 --> G[End of run for current config];
    end

    G --> D;
    D -- "No" --> H[End Program];

    style A fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
    style H fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
    style F fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
```

### Detailed Streaming Workflow (`disteta_streaming.py`)

The `disteta_streaming.py` application simulates a real-time environment, processing one data point at a time.

```mermaid
graph TD
    A[Start main] --> B{"Instantiate DistetaStreaming class"};
    B --> C{"Find and load latest run artifacts (models, cluster profiles, etc.)"};
    C --> D{For each incoming data instance...};
    D --> E["Call process_instance()"];
    
    subgraph "process_instance() Workflow"
        E --> S1["Stage 1: Predict Cluster"];
        S1 --> S1_1["Determine correct model & parameters from instance data"];
        S1_1 --> S1_2["Pre-process instance into one-hot encoded vector"];
        S1_2 --> S1_3["Predict cluster probabilities with XGBoost model"];

        S1_3 --> S2["Stage 2: Generate Plot"];
        S2 --> S2_1["Load cluster distribution profile"];
        S2_1 --> S2_2["Create plot with distribution, instance line, and probabilities"];
        S2_2 --> S2_3["Save plot to output/streaming_plots"];
    end
    
    S2_3 --> F[Return results & wait for next instance];
    F --> D;
    
    style A fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
    style D fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
    style E fill:#cde4ff,stroke:#6a8ebf,stroke-width:2px
```

## Project Structure

*   **`assets/`**: Contains static resource files like images and audio clips used in documentation.
*   **`config/`**: Contains all YAML configuration files.
    *   `config_disteta.yaml`: Configures the analysis application.
    *   `config_report_generator.yaml`: Configures the reporting application and LLM prompts.
*   **`data/`**: Directory to store your input data files (e.g., `.csv`, `.parquet`).
*   **`models/`**: Contains the saved XGBoost models and their corresponding metadata files.
*   **`output/`**: Default directory where all generated run folders are saved.
    *   **`streaming_plots/`**: Contains the plots generated by the real-time streaming module.
*   **`scripts/`**: Holds small, one-off utility or testing scripts.
*   **`src/`**: The main source code directory.
    *   `constants.py`: A centralized file for shared, project-wide constants.
    *   `disteta_batch/`: The main analysis and model training application.
    *   `disteta_streaming/`: The real-time prediction and plotting application.
    *   `report_generator/`: The LLM-based reporting application.
*   **`tests/`**: Pytest tests.
*   **`README.md`**: The main project readme.