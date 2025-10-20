import os

from google import genai

# Optional: Import pandas for tabular output
try:
    import pandas as pd
except ImportError:
    pd = None

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


def list_gemini_models_formatted():
    """
    Fetches and prints available Gemini models with key attributes in a readable format.
    Optionally uses pandas for tabular output if available.
    """
    if not GEMINI_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set. Cannot list models.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    print("Fetching available Gemini models...")
    try:
        # Convert generator to list so we can iterate over models multiple times if needed
        models = list(client.models.list())
    except Exception as e:
        print(f"Error fetching models: {e}")
        return

    if not models:
        print("No models found.")
        return

    # --- Detailed List Output ---
    print("\n--- Available Gemini Models (Detailed List) ---")
    for i, model in enumerate(models):
        print(f"\nModel {i + 1}:")
        print(f"  Name: {model.name}")
        print(f"  Display Name: {model.display_name}")
        # Safely get description, which could be None even if attribute exists
        description = getattr(model, "description", None)
        print(f"  Description: {description if description else 'N/A'}")
        print(f"  Input Token Limit: {getattr(model, 'input_token_limit', 'N/A')}")
        print(f"  Output Token Limit: {getattr(model, 'output_token_limit', 'N/A')}")
        supported_methods = getattr(model, "supported_generation_methods", [])
        print(
            f"  Supported Methods: {', '.join(supported_methods) if supported_methods else 'N/A'}"
        )
        print(f"  Version: {getattr(model, 'version', 'N/A')}")
    print("\n---------------------------------------------")

    # --- Tabular Output (if pandas is installed) ---
    if pd:
        print("\n--- Available Gemini Models (Tabular Format) ---")
        model_data = []
        for model in models:
            # Get description, which might be None even if the attribute exists.
            description = getattr(model, "description", None)

            # Prepare a display-ready description, handling None and long strings.
            display_description = "N/A"
            if isinstance(description, str):
                display_description = (
                    f"{description[:72]}..." if len(description) > 75 else description
                )

            model_data.append(
                {
                    "Name": model.name,
                    "Display Name": model.display_name,
                    "Description": display_description,
                    "Input Limit": getattr(model, "input_token_limit", "N/A"),
                    "Output Limit": getattr(model, "output_token_limit", "N/A"),
                    "Methods": ", ".join(
                        getattr(model, "supported_generation_methods", [])
                    ),
                    "Version": getattr(model, "version", "N/A"),
                }
            )
        df = pd.DataFrame(model_data)
        print(df.to_string(index=False))

        print("\n---------------------------------------------")
    else:
        print(
            "\nTip: Install pandas (`pip install pandas`) for a more structured tabular output."
        )


if __name__ == "__main__":
    list_gemini_models_formatted()
