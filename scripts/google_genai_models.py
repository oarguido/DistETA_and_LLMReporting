"""
This script provides utility functions to interact with Google's Generative AI models.

It allows users to:
1.  Test if their `GOOGLE_API_KEY` is correctly set up and working.
2.  List all available Gemini models in both a detailed list and a clean,
    tabular format (if pandas is installed).

The script is designed to be run directly from the command line and provides
clear, formatted output to help users verify their environment and explore
available models.

Execution:
    $ python scripts/google_genai_models.py
"""

import logging
import os

import google.generativeai as genai

# Configure a logger for this script
logger = logging.getLogger(__name__)

# Optional: Import pandas for tabular output
try:
    import pandas as pd
except ImportError:
    pd = None

# Optional: Load API key from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.info("dotenv not installed, skipping .env file load.")

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


def list_gemini_models_formatted():
    """
    Fetches and prints available Gemini models with key attributes.

    This function retrieves all models accessible via the configured API key
    and presents them in two formats:
    1. A detailed, multi-line list for each model.
    2. A compact, tabular summary (if pandas is installed).
    """
    if not GEMINI_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set. Cannot list models.")
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore

    logger.info("Fetching available Gemini models...")
    try:
        models = list(genai.list_models())  # type: ignore
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return

    if not models:
        logger.warning("No models found for the provided API key.")
        return

    # --- Detailed List Output ---
    print("\n--- Available Gemini Models (Detailed List) ---")
    for i, model in enumerate(models):
        print(f"\nModel {i + 1}:")
        print(f"  Name: {model.name}")
        print(f"  Display Name: {model.display_name}")
        description = getattr(model, "description", "N/A")
        print(f"  Description: {description}")
        print(f"  Input Token Limit: {getattr(model, 'input_token_limit', 'N/A')}")
        print(f"  Output Token Limit: {getattr(model, 'output_token_limit', 'N/A')}")
        supported_methods = getattr(model, "supported_generation_methods", [])
        print(
            f"  Supported Methods: {', '.join(supported_methods) if supported_methods else 'N/A'}"
        )
        print(f"  Version: {getattr(model, 'version', 'N/A')}")
    print("\n" + "-" * 45)

    # --- Tabular Output (if pandas is installed) ---
    if pd:
        print("\n--- Available Gemini Models (Tabular Format) ---")
        model_data = [
            {
                "Name": model.name,
                "Display Name": model.display_name,
                "Description": f"{model.description[:72]}..."
                if len(model.description) > 75
                else model.description,
                "Input Limit": model.input_token_limit,
                "Output Limit": model.output_token_limit,
                "Methods": ", ".join(model.supported_generation_methods),
                "Version": model.version,
            }
            for model in models
        ]
        df = pd.DataFrame(model_data)
        print(df.to_string(index=False))
        print("\n" + "-" * 45)
    else:
        logger.info(
            "Tip: Install pandas (`pip install pandas`) for a structured tabular output."
        )


def test_api_key() -> bool:
    """
    Tests the configured GOOGLE_API_KEY to ensure it is valid.

    Returns:
        True if the API key is working, False otherwise.
    """
    if not GEMINI_API_KEY:
        logger.error("❌ Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key using one of these methods:")
        print("1. Export in terminal: export GOOGLE_API_KEY='your_key_here'")
        print(
            "2. Create a .env file in your project root with: GOOGLE_API_KEY=your_key_here"
        )
        return False

    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        models = list(genai.list_models())  # type: ignore
        logger.info(f"✅ API key is working! Found {len(models)} available models.")
        return True
    except Exception as e:
        logger.error(f"❌ API key test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Testing Google API key...")
    if test_api_key():
        print("\n" + "=" * 50)
        list_gemini_models_formatted()
    else:
        logger.error("\nPlease fix the API key issue and try again.")
