# =============================================================================
# Dockerfile for DistETA-AIDI
# =============================================================================
# This Dockerfile creates a single image that can run both the analysis
# (`disteta_aidi`) and the report generation (`report_generator`) modules.

# --- Stage 1: Build Environment ---
# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by kaleido for static plot generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libxss1 libasound2 libxtst6 libexpat1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
# This is done in a separate step to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Test Environment ---
# This stage inherits from the builder and adds test-specific dependencies and code.
FROM builder as test

# Copy development requirements and install them
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy test-specific files and source code
COPY pytest.ini .
COPY tests/ ./tests/
COPY src/ ./src/

CMD ["pytest"]
# --- Stage 3: Final Image ---
# The final image is built FROM the 'builder' stage, which already contains
# the python interpreter, all system dependencies, and all pip packages.
FROM builder

# Copy application source code, config, and data
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY assets/ ./assets/

# Create the output directory and set permissions
RUN mkdir -p output && chown -R 1000:1000 output

# Switch to a non-root user for security
USER 1000

# Environment variable for the Google API Key (for Gemini)
# This should be provided at runtime (e.g., via docker-compose.yml)
ENV GOOGLE_API_KEY=""

# Default command (can be overridden)
CMD ["python", "-m", "src.disteta_aidi.main"]