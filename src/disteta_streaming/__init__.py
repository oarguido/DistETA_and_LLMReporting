"""
This package contains the real-time data processing and visualization component of the DistETA project.

It is designed to load artifacts from a completed batch analysis run, process new,
individual data instances as they arrive, and generate detailed prediction plots.

Usage:
------
This module can be run in two ways:

1.  **Run only the streaming simulation:**
    This assumes that a batch analysis has already been completed. It will load the
    latest artifacts and process the sample data from `data/truck_arrival_data.csv`.

    $ python -m src.disteta_streaming.main

2.  **Run the full end-to-end pipeline:**
    This command orchestrates the entire workflow: it runs the batch analysis,
    generates the analysis report, and then starts the streaming simulation.

    $ python -m src.disteta_streaming.main --run-pipeline

    Via Docker Compose:
    $ docker-compose --profile pipeline up --build
"""