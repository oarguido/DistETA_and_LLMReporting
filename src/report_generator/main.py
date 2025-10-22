"""
This module, part of the DistETA-AIDI project, is the LLM-based report generator.

It defines the `AgnosticReportGenerator` class, which is designed to be
decoupled from the analysis script. It operates on the output artifacts
(JSON summaries, plots, and logs) from a completed analysis run.

The key responsibilities of this module are:

1.  **Configuration Loading**: Loads and validates settings from a dedicated
    YAML file using Pydantic models.
2.  **Artifact Discovery**: Automatically finds the latest analysis output
    directory or processes a specified one.
3.  **LLM Orchestration**: Gathers all artifacts, formats them into a detailed
    prompt, and uses LangChain to invoke a configured Large Language Model
    (e.g., Google Gemini, local Ollama) to generate a narrative report.
4.  **Report Generation & Serving**: Saves the final report as both Markdown
    and styled HTML, and can serve it via a local FastAPI web server.

Execution:
    To generate a report and serve it, execute this module as a script:
    $ python -m src.report_generator.main
"""

# =============================================================================
# HEADER (Imports, Constants, Logger)
# =============================================================================
import argparse
import base64
import glob
import io
import json
import logging
import os
import re
import time
import webbrowser
from datetime import datetime
from typing import Dict, Optional

import markdown
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from starlette.staticfiles import StaticFiles

from src.disteta_aidi.main import run_all_analyses

from .. import constants

# --- Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence noisy third-party loggers
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("shutil").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("google.api_core").setLevel(logging.ERROR)


# Ollama is an optional dependency, so we handle its import gracefully.
try:
    import ollama
except ImportError:
    ollama = None


# --- Application Configuration ---
DEFAULT_REPORT_CONFIG_PATH = os.path.join(
    constants.CONFIG_DIR, "config_report_generator.yaml"
)
STYLE_SHEET_PATH = os.path.join(constants.ASSETS_DIR, "styles.css")


# --- Default LLM Configuration (used if not in config file) ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_DEFAULT_MODEL_NAME = "gemini-flash-latest"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")


# =============================================================================
# CONFIGURATION MODELS (Pydantic)
# =============================================================================
class GeminiSettings(BaseModel):
    """Pydantic model for Gemini settings."""

    model_name: str = GEMINI_DEFAULT_MODEL_NAME


class OllamaSettings(BaseModel):
    """Pydantic model for Ollama settings."""

    model_name: Optional[str] = None


class LLMConfig(BaseModel):
    """Pydantic model for LLM provider and model configuration."""

    provider: str = "gemini"
    gemini_settings: GeminiSettings = Field(default_factory=GeminiSettings)
    ollama_settings: OllamaSettings = Field(default_factory=OllamaSettings)


class DirectoryStructureConfig(BaseModel):
    """Pydantic model for directory and file path configurations."""

    run_dir_regex: str = r"^\d{8}_\d{6}_run_.*"
    data_dir: str = constants.DATA_DIR_NAME
    graphics_dir: str = constants.GRAPHICS_DIR_NAME
    logs_dir: str = constants.LOGS_DIR_NAME
    reports_dir: str = constants.REPORTS_DIR_NAME


class VisualsConfig(BaseModel):
    """Pydantic model for visual settings like report width."""

    report_width_px: int = 1100


class ReportingConfig(BaseModel):
    """Pydantic model for report appearance and content settings."""

    plot_type: str = Field("static", pattern=r"^(static|interactive)$")
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)


class ReportGeneratorConfig(BaseModel):
    """Main Pydantic model for the entire report generator configuration."""

    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    llm_prompt_template: str
    directory_structure: DirectoryStructureConfig = Field(
        default_factory=DirectoryStructureConfig
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_report_config(
    config_path: str = DEFAULT_REPORT_CONFIG_PATH,
) -> ReportGeneratorConfig:
    """
    Loads and validates report configuration from a YAML file using Pydantic.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                raise ValueError("Configuration file is empty.")
        return ReportGeneratorConfig(**config_data)
    except FileNotFoundError:
        logger.error(f"Report configuration file not found at {config_path}.")
        raise
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing report configuration YAML file '{config_path}': {e}"
        )
        raise
    except ValidationError as e:
        logger.error(f"Error validating configuration from '{config_path}':\n{e}")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading config '{config_path}': {e}"
        )
        raise


# =============================================================================
# MAIN REPORT GENERATION CLASS
# =============================================================================
class AgnosticReportGenerator:
    """
    Generates a comprehensive, LLM-driven analysis report.

    This class is 'agnostic' to the analysis itself, operating on the output
    artifacts (logs, JSON summaries, plots) from a DistetaAIDI run. It finds
    the specified or latest run directory, gathers all relevant data, and
    constructs a detailed prompt for a large
    language model (LLM) to generate a human-readable report in Markdown.

    The key steps are:
    1.  **Find Run Directory**: Locates the target analysis output directory.
    2.  **Gather Artifacts**: Collects all JSON data, plots (static or interactive),
        and log files from the directory.
    3.  **Invoke LLM**: Initializes the configured LLM (e.g., Gemini, Ollama)
        and passes the gathered artifacts along with a sophisticated prompt
        template to generate a Markdown report.
    4.  **Post-process**: Corrects file paths in the generated Markdown to ensure
        they are accessible when served.
    5.  **Save Artifacts**: Saves the final report in both Markdown (.md) and
        styled HTML (.html) formats.

    Attributes:
        config (ReportGeneratorConfig): The validated configuration object.
        base_output_dir (str): The root directory where run outputs are stored.
    """

    def __init__(self, config: ReportGeneratorConfig, base_output_dir: str):
        self.config = config
        self.base_output_dir = base_output_dir

    def _get_html_styles(self) -> str:
        """Loads and formats the CSS styles for the HTML report."""
        report_width_px = self.config.reporting.visuals.report_width_px
        try:
            with open(STYLE_SHEET_PATH, "r", encoding="utf-8") as f:
                css_content = f.read()
        except FileNotFoundError:
            logger.error(
                f"CSS file not found at {STYLE_SHEET_PATH}. Using empty styles."
            )
            css_content = ""

        return f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --report-width: {report_width_px}px;
            }}
            {css_content}
        </style>
        """

    def _find_latest_run_dir(self) -> Optional[str]:
        """
        Finds the most recent timestamped run directory in the base output path.

        It identifies directories matching the `run_dir_regex` from the config,
        sorts them lexicographically (which works for `YYYYMMDD_HHMMSS` format),
        and returns the path to the latest one.

        Returns:
            The absolute path to the latest run directory, or None if not found.
        """
        if not os.path.isdir(self.base_output_dir):
            logger.error(f"Output directory '{self.base_output_dir}' not found.")
            return None
        dir_pattern = re.compile(self.config.directory_structure.run_dir_regex.strip())
        potential_dirs = [
            d
            for d in os.listdir(self.base_output_dir)
            if os.path.isdir(os.path.join(self.base_output_dir, d))
            and dir_pattern.match(d)
        ]
        if not potential_dirs:
            logger.warning(
                f"No valid run directories found in '{self.base_output_dir}'."
            )
            return None
        latest_run_subdir = sorted(potential_dirs, reverse=True)[0]
        return os.path.join(self.base_output_dir, latest_run_subdir)

    def _gather_run_artifacts(self, run_dir: str) -> Dict:
        """
        Gathers all logs, JSON, and plot files from a specific run directory.

        This method scans the 'data', 'logs', and 'graphics' subdirectories of a given
        run folder to collect all necessary files for the LLM prompt. It
        selectively excludes verbose JSON files to manage token count.

        Args:
            run_dir: The absolute path to the run directory to scan.

        Returns:
            A dictionary containing the collected artifacts.
        """
        artifacts = {
            "json_data": {},
            "plots": [],
            "logs": "",
            "plot_type": self.config.reporting.plot_type,
        }

        # --- Gather JSON Artifacts ---
        json_subdir = os.path.join(run_dir, self.config.directory_structure.data_dir)
        if os.path.isdir(json_subdir):
            for json_file in sorted(glob.glob(os.path.join(json_subdir, "*.json"))):
                filename = os.path.basename(json_file)
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        artifacts["json_data"][filename] = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse JSON file {filename}: {e}")
                    artifacts["json_data"][filename] = f"Error parsing JSON: {e}"
                except Exception as e:
                    logger.warning(f"Could not read JSON file {filename}: {e}")
                    artifacts["json_data"][filename] = f"Error reading file: {e}"

        # --- Gather Plot Artifacts ---
        graphics_subdir = os.path.join(
            run_dir, self.config.directory_structure.graphics_dir
        )
        plot_extension = (
            "html" if self.config.reporting.plot_type == "interactive" else "png"
        )
        if os.path.isdir(graphics_subdir):
            artifacts["plots"] = sorted(
                glob.glob(os.path.join(graphics_subdir, f"*.{plot_extension}"))
            )

        # --- Gather Log Files ---
        logs_subdir = os.path.join(run_dir, self.config.directory_structure.logs_dir)
        log_files = glob.glob(os.path.join(logs_subdir, "*.log"))
        if log_files:
            try:
                with open(log_files[0], "r", encoding="utf-8") as f:
                    artifacts["logs"] = f.read()
            except Exception as e:
                logger.warning(f"Could not read log file {log_files[0]}: {e}")
                artifacts["logs"] = f"Could not read log file {log_files[0]}: {e}"

        return artifacts

    def _get_llm_instance(self) -> Runnable:
        """
        Initializes and returns an instance of the configured LLM via LangChain.

        Returns:
            A LangChain `Runnable` object for the configured LLM.
        """
        provider = self.config.llm.provider.lower()
        if provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set for Gemini."
                )
            model_name = self.config.llm.gemini_settings.model_name
            logger.info(f"Initializing LangChain Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name, google_api_key=GEMINI_API_KEY
            )
        elif provider == "ollama":
            if not ollama:
                raise ImportError(
                    "'ollama' library not installed. Cannot use Ollama provider."
                )
            model_name = self.config.llm.ollama_settings.model_name
            if not model_name:
                raise ValueError("Ollama provider selected, but no model name is set.")
            logger.info(f"Initializing LangChain Ollama model: {model_name}")
            init_kwargs = {"model": model_name}
            if OLLAMA_BASE_URL:
                init_kwargs["base_url"] = OLLAMA_BASE_URL
                logger.info(f"  Connecting to Ollama at: {OLLAMA_BASE_URL}")
            return ChatOllama(**init_kwargs)
        else:
            raise ValueError(f"Invalid LLM provider in config: '{provider}'")

    def _create_report_generation_chain(self, llm: Runnable) -> Runnable:
        """
        Creates the full LangChain Runnable chain for generating the report.

        This chain takes the prepared artifacts, formats them into a multimodal
        `HumanMessage` (text + images), and pipes it to the LLM for inference.
        """

        def _prepare_llm_input(input_dict: dict) -> list[HumanMessage]:
            """
            Prepares the final prompt and image parts for the LLM call.
            This function constructs the HumanMessage with multimodal content.
            """
            prompt_template = PromptTemplate.from_template(
                self.config.llm_prompt_template
            )
            final_prompt_str = prompt_template.format(**input_dict)

            # LangChain's multimodal input is a list of content blocks (text, image).
            # The content for a HumanMessage can be a list of parts (text, images).
            # The first part is always the text prompt.
            message_content = [{"type": "text", "text": final_prompt_str}]

            # For static plots, load images and add to the message content
            if (
                self.config.reporting.plot_type == "static"
                and Image
                and input_dict.get("plots")
            ):
                logger.info("Encoding static plots for multimodal LLM input...")
                for img_path in input_dict["plots"]:
                    try:
                        # For multimodal LLMs like Gemini, images must be base64 encoded
                        # and passed as a data URI.
                        # 1. Open the image with PIL
                        img = Image.open(img_path)
                        # 2. Save it to a bytes buffer in memory
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        # 3. Encode the bytes to a base64 string
                        img_base64 = base64.b64encode(buffered.getvalue()).decode(
                            "utf-8"
                        )
                        # 4. Format as a data URI and append to the message
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_base64}",
                            }
                        )
                    except Exception as e_img:
                        logger.warning(
                            f"Could not load image {img_path} for LLM: {e_img}"
                        )

            # Chat models expect a list of messages, even if it's just one.
            return [HumanMessage(content=message_content)]

        # This chain defines the sequence of operations
        return RunnableLambda(_prepare_llm_input) | llm

    def _create_placeholder_report(self, artifacts: Dict, run_dir_name: str) -> str:
        """
        Generates a simple, raw-data fallback report if the LLM call fails.

        Args:
            artifacts: The dictionary of gathered run artifacts.
            run_dir_name: The name of the run directory.

        Returns:
            A string containing the placeholder report content in Markdown/HTML.
        """
        logger.warning(
            "LLM report generation failed or was skipped. Generating a placeholder report."
        )
        report_text = f"# Analysis Report for {run_dir_name}\n\n"
        report_text += "*(LLM report generation failed. This is a placeholder containing raw data.)*\n\n"

        # Add JSON data to placeholder
        all_json_data_parts = [
            f"<pre><code>{json.dumps(data, indent=2)}</code></pre>"
            for data in artifacts["json_data"].values()
        ]
        all_json_data_str = "\n".join(all_json_data_parts)
        report_text += "## JSON Data Summary\n\n"
        # The LLM is instructed to use ```json ... ```, but for placeholder we directly use HTML
        # to ensure the header is applied correctly.
        report_text += f"{all_json_data_str}\n\n"

        # Add plots to placeholder
        report_text += "## Plots\n\n"
        if artifacts["plots"]:
            is_interactive = self.config.reporting.plot_type == "interactive"
            for plot_path in artifacts["plots"]:
                plot_filename = os.path.basename(plot_path)
                correct_plot_path = f"{run_dir_name}/{self.config.directory_structure.graphics_dir}/{plot_filename}"
                report_text += f"### {plot_filename}\n"
                if is_interactive:
                    report_text += f'<iframe src="{correct_plot_path}" width="100%" height="620px" style="border:none;"></iframe>\n\n'
                else:
                    report_text += f"!{plot_filename}\n\n"
        else:
            report_text += "No plots found."
        return report_text

    def _save_markdown_report(
        self, report_text: str, run_dir: str, token_usage: Optional[Dict]
    ) -> tuple[str, str]:
        """
        Saves the final report text to a Markdown file, prepending usage info.

        Args:
            report_text: The Markdown content generated by the LLM.
            run_dir: The path to the run directory where the report will be saved.
            token_usage: A dictionary with token usage statistics.

        Returns:
            A tuple containing the file path and the final content of the saved report.
        """
        if token_usage:
            model_name = (
                self.config.llm.ollama_settings.model_name
                if self.config.llm.provider == "ollama"
                else self.config.llm.gemini_settings.model_name
            )
            usage_summary = (
                f"### LLM Usage Information\n\n"
                f"- **Model:** `{model_name}`\n"
                f"- **Input Tokens:** `{token_usage.get('input_tokens', 'N/A')}`\n"
                f"- **Output Tokens:** `{token_usage.get('output_tokens', 'N/A')}`\n"
                f"- **Total Tokens:** `{token_usage.get('total_tokens', 'N/A')}`\n\n---\n\n"
            )
            report_text = usage_summary + report_text

        reports_subdir = os.path.join(
            run_dir, self.config.directory_structure.reports_dir
        )
        os.makedirs(reports_subdir, exist_ok=True)

        model_name = (
            self.config.llm.ollama_settings.model_name
            if self.config.llm.provider == "ollama"
            else self.config.llm.gemini_settings.model_name
        )
        sanitized_model_name = re.sub(r"[/: ]", "_", model_name).replace(".", "")
        report_filename = f"llm_analysis_report_{sanitized_model_name}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        report_filepath = os.path.join(reports_subdir, report_filename)

        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Report saved to: {report_filepath}")
        return report_filepath, report_text

    def _prepare_llm_chain_input(self, artifacts: Dict, run_dir: str) -> Dict:
        """
        Prepares the dictionary of variables to be passed to the LLM prompt template.

        Args:
            artifacts: The dictionary of gathered run artifacts.
            run_dir: The path to the run directory.

        Returns:
            A dictionary of formatted strings ready for the prompt template.
        """
        run_dir_name = os.path.basename(run_dir)

        # --- Extract and format timestamp ---
        timestamp_str = run_dir_name[:15]  # e.g., "20251020_163947"
        try:
            dt_object = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")  # type: ignore
            human_readable_timestamp = dt_object.strftime(
                "%B %d, %Y, at %H:%M:%S UTC"
            )  # Added UTC for clarity
        except ValueError:
            human_readable_timestamp = "an unknown time"  # Fallback

        all_json_data_parts = [
            f"```json\n{json.dumps(data, indent=2)}"
            for filename, data in artifacts["json_data"].items()
        ]
        all_json_data_str = (
            "\n\n".join(all_json_data_parts)
            if all_json_data_parts
            else "No JSON data found."
        )

        plot_embedding_instructions = ""
        plot_files_summary = "No plot files found in the 'graphics' directory."
        plot_extension = (
            "html" if self.config.reporting.plot_type == "interactive" else "png"
        )

        plot_list_str = (
            "\n".join([f"- `{os.path.basename(p)}`" for p in artifacts["plots"]])
            if artifacts["plots"]
            else "No plot files available."
        )
        plot_files_summary = (
            f"The following plot files (.{plot_extension}) were generated. "
            "When you embed a plot, YOU MUST use the format specified in the rules.\n\n"
            "Available plot files:\n" + plot_list_str
        )

        if self.config.reporting.plot_type == "interactive":
            plot_embedding_instructions = f'You are generating a report with **interactive** plots. To embed a plot, you MUST use an `<iframe>` tag with a full path starting with `/{constants.OUTPUT_DIR}/` like this:\n`<iframe src="/{constants.OUTPUT_DIR}/{{run_directory_name}}/graphics/filename.html" width="100%" height="620px" style="border:none;"></iframe>`'
        else:  # static
            plot_embedding_instructions = f"You are generating a report with **static** images. To embed a plot, you MUST use the standard Markdown image format with a full path starting with `/{constants.OUTPUT_DIR}/` like this:\n`![Descriptive Alt Text](/{constants.OUTPUT_DIR}/{{run_directory_name}}/graphics/filename.png)`"

        return {
            "run_directory_name": run_dir_name,
            "run_timestamp_str": human_readable_timestamp,
            "log_content": artifacts.get("logs", "No execution log available."),
            "all_json_data_str": all_json_data_str,
            "plot_files_summary": plot_files_summary,
            "plot_embedding_instructions": plot_embedding_instructions,
            "plots": artifacts.get("plots", []),
        }

    def _generate_llm_report(
        self, artifacts: Dict, run_dir: str
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Orchestrates the LLM report generation process using LangChain.

        Args:
            artifacts: The dictionary of gathered run artifacts.
            run_dir: The path to the run directory.

        Returns:
            A tuple containing the generated report text and a dictionary of
            token usage statistics.
        """
        logger.info("--- Generating LLM Analysis Report ---")

        report_text: Optional[str] = None
        token_usage: Optional[dict] = None

        try:
            llm = self._get_llm_instance()
            report_chain = self._create_report_generation_chain(llm)

            chain_input = self._prepare_llm_chain_input(artifacts, run_dir)

            # Invoke the chain to get the response message
            response_message = report_chain.invoke(chain_input)
            report_text = response_message.content

            # --- Unified Token Extraction Logic ---
            # For recent LangChain versions, token usage is consistently in response_metadata.
            response_meta = getattr(response_message, "response_metadata", {})
            usage_data = response_meta.get("token_usage")

            # Fallback for older Gemini-specific metadata structure
            if not usage_data and hasattr(response_message, "usage_metadata"):
                usage_data = response_message.usage_metadata

            if usage_data:
                logger.info(
                    f"--- {self.config.llm.provider.capitalize()} Token Usage ---"
                )
                token_usage = {
                    "input_tokens": usage_data.get("prompt_token_count")
                    or usage_data.get("input_tokens", 0),
                    "output_tokens": usage_data.get("candidates_token_count")
                    or usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("total_token_count", 0),
                }
                if not token_usage.get("total_tokens") and token_usage.get(
                    "input_tokens"
                ):
                    token_usage["total_tokens"] = token_usage.get(
                        "input_tokens", 0
                    ) + token_usage.get("output_tokens", 0)

            if token_usage:
                logger.info(
                    f"  Input:  {token_usage.get('input_tokens', 'N/A')} tokens"
                )
                logger.info(
                    f"  Output: {token_usage.get('output_tokens', 'N/A')} tokens"
                )
                logger.info(
                    f"  Total:  {token_usage.get('total_tokens', 'N/A')} tokens"
                )
                logger.info("-----------------------")

        except Exception as e:
            logger.error(f"Error invoking LangChain chain: {e}", exc_info=True)
            report_text = self._create_placeholder_report(
                artifacts, os.path.basename(run_dir)
            )
            token_usage = None

        if report_text:
            logger.info(f"--- Generated Report (Preview) ---\n{report_text[:500]}...")

        return report_text, token_usage

    def _postprocess_llm_output(self, report_text: str, run_dir_name: str) -> str:
        """
        Fixes plot paths in the LLM's Markdown output to be relative to the server root.

        Args:
            report_text: The raw Markdown output from the LLM.
            run_dir_name: The name of the run directory.

        Returns:
            The processed Markdown with corrected image and iframe paths.
        """
        logger.info("Post-processing LLM output to ensure correct plot paths...")

        # Regex for Markdown images: ![alt text](path)
        markdown_image_regex = r"!\\[(.*?)\\]\((.*?)\""
        # Regex for iframes: <iframe src="path" ...>
        iframe_regex = r'(<iframe\s+src=")(.*?)(".*?></iframe>)'

        def fix_path(match, is_iframe=False):
            if is_iframe:
                prefix, malformed_path, suffix = match.groups()
                filename = os.path.basename(malformed_path)
                correct_path = f"/{constants.OUTPUT_DIR}/{run_dir_name}/{self.config.directory_structure.graphics_dir}/{filename}"
                logger.info(
                    f"  Fixing iframe path: '{malformed_path}' -> '{correct_path}'"
                )
                return f"{prefix}{correct_path}{suffix}"
            else:  # Markdown image
                alt_text, malformed_path = match.groups()
                filename = os.path.basename(malformed_path)
                correct_path = f"/{constants.OUTPUT_DIR}/{run_dir_name}/{self.config.directory_structure.graphics_dir}/{filename}"
                logger.info(
                    f"  Fixing image path: '{malformed_path}' -> '{correct_path}'"
                )
                return f"![{alt_text}]({correct_path})"

        # Process both types of paths
        processed_text = re.sub(
            markdown_image_regex, lambda m: fix_path(m, is_iframe=False), report_text
        )
        processed_text = re.sub(
            iframe_regex, lambda m: fix_path(m, is_iframe=True), processed_text
        )

        return processed_text

    def _convert_md_to_html(
        self, md_filepath: str, md_content_with_usage: str
    ) -> Optional[str]:
        """
        Converts Markdown to a styled, self-contained HTML file.

        Args:
            md_filepath: The path to the saved Markdown file.
            md_content_with_usage: The full content of the Markdown file.

        Returns:
            The path to the generated HTML file, or None on failure."""
        if not os.path.exists(md_filepath):
            logger.error(f"Markdown file not found at {md_filepath}")
            return None
        html_filepath = md_filepath.replace(".md", ".html")

        run_dir_name = os.path.basename(os.path.dirname(os.path.dirname(md_filepath)))
        processed_md_content = self._postprocess_llm_output(
            md_content_with_usage, run_dir_name
        )

        html_body_content = markdown.markdown(
            processed_md_content,
            extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        )
        full_html_output = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
        <title>Analysis Report</title>{self._get_html_styles()}</head>
        <body><div class="report-content-wrapper">{html_body_content}</div></body></html>
        """
        with open(html_filepath, "w", encoding="utf-8") as f_html:
            f_html.write(full_html_output)
        return html_filepath

    def _postprocess_and_get_html_content(self, report_path: str) -> Optional[str]:
        """
        Reads a Markdown file, post-processes paths, and returns a full HTML string.

        Args:
            report_path: The path to the Markdown report file.

        Returns:
            A string containing the complete, styled HTML for serving.
        """
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            run_dir_name = os.path.basename(
                os.path.dirname(os.path.dirname(report_path))
            )
            processed_md_content = self._postprocess_llm_output(
                md_content, run_dir_name
            )

            html_body_content = markdown.markdown(
                processed_md_content,
                extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
            )
            full_html_output = f"""
            <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
            <title>Analysis Report</title>{self._get_html_styles()}</head>
            <body><div class="report-content-wrapper">{html_body_content}</div></body></html>
            """
            return full_html_output
        except FileNotFoundError:
            logger.error(f"Markdown file not found at {report_path} for serving.")
            return None

    def generate_report(self, target_run_dir: Optional[str] = None) -> Optional[str]:
        """
        Main orchestration method to generate and save a complete analysis report.

        This method executes the full report generation pipeline:
        1.  Finds the appropriate run directory (either the one specified or the latest).
        2.  Gathers all analysis artifacts from that directory.
        3.  Invokes the LLM to generate the report content in Markdown.
        4.  Saves the raw Markdown report.
        5.  Converts the Markdown to a final, styled HTML report.

        Args:
            target_run_dir (Optional[str]): The absolute path to a specific run
                directory to process. If None, the latest run directory found in
                the base output path will be used.

        Returns:
            Optional[str]: The relative path to the generated HTML report from the
                           base output directory if successful, otherwise None.
        """
        run_dir_to_process = target_run_dir or self._find_latest_run_dir()
        if not run_dir_to_process or not os.path.isdir(run_dir_to_process):
            logger.error("Could not find or access a valid run directory.")
            return None

        if target_run_dir:
            logger.info(f"Processing specified run directory: {run_dir_to_process}")
        else:
            logger.info(f"Found latest run directory to process: {run_dir_to_process}")

        artifacts = self._gather_run_artifacts(run_dir_to_process)
        if not artifacts["json_data"] and not artifacts["plots"]:
            logger.warning(
                "No data or plot artifacts found in the run directory. Cannot generate a meaningful report."
            )
            return None

        final_report_text, token_usage = self._generate_llm_report(
            artifacts, run_dir_to_process
        )

        if not final_report_text:
            logger.error("Report text generation failed.")
            return None

        md_report_path, final_md_content = self._save_markdown_report(
            final_report_text, run_dir_to_process, token_usage
        )
        html_report_path = self._convert_md_to_html(md_report_path, final_md_content)
        if not html_report_path:
            logger.error("Failed to convert Markdown report to HTML.")
            return None

        logger.info(f"Successfully generated HTML report: {html_report_path}")
        return os.path.relpath(html_report_path, self.base_output_dir)


# =============================================================================
# WEB SERVER LOGIC (FastAPI)
# =============================================================================
def create_fastapi_app(state: Dict) -> FastAPI:
    """
    Creates and configures the FastAPI application, injecting state.
    This factory pattern avoids using global variables for state management.
    """
    app = FastAPI()

    # Mount the entire 'output' directory to be served at the '/output' URL path.
    # This is the standard and most robust way to handle static files.
    if os.path.isdir(constants.OUTPUT_DIR):
        app.mount(
            f"/{constants.OUTPUT_DIR}",
            StaticFiles(directory=constants.OUTPUT_DIR),
            name=constants.OUTPUT_DIR,
        )

    @app.get("/", response_class=HTMLResponse)
    async def serve_index_page():
        generator: Optional[AgnosticReportGenerator] = state.get("generator")

        report_filename = state.get("latest_html_report_filename")
        if not report_filename:
            return HTMLResponse(
                content="<h1>Report Not Found</h1><p>No report has been generated or specified to serve. Please run the script without --serve-only first.",
                status_code=404,
            )

        report_path = os.path.join(constants.OUTPUT_DIR, report_filename)
        if not os.path.exists(report_path):
            return HTMLResponse(
                content=f"<h1>Error</h1><p>Report file not found at: {report_path}</p>",
                status_code=404,
            )

        md_path = report_path.replace(".html", ".md")
        if generator:
            full_html_output = generator._postprocess_and_get_html_content(md_path)

            return HTMLResponse(content=full_html_output)

        return HTMLResponse(
            content="<h1>Server Error</h1><p>Report generator not initialized.</p>",
            status_code=500,
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Returns an empty response to prevent 404 errors for the favicon."""
        return Response(status_code=204)

    return app


# =============================================================================
# SCRIPT EXECUTION (The if __name__ == "__main__" block)
# =============================================================================
def main():
    """Parses command-line arguments and orchestrates the report generation and serving."""
    parser = argparse.ArgumentParser(
        description="Generate an LLM-based report from existing analysis results."
    )
    parser.add_argument(
        "--run-directory",
        type=str,
        help="Path to a specific run directory to process. If not provided, the latest one will be used.",
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="Port for the FastAPI server."
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for the FastAPI server."
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Generate the report but do not start the web server.",
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Skip report generation, only start the server for the latest found report.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run the full 'disteta_aidi' analysis before generating the report.",
    )
    args = parser.parse_args()

    try:
        if args.run_analysis:
            if args.run_directory:
                logger.warning(
                    "Both --run-analysis and --run-directory were specified. "
                    "The analysis will run, and the report will be generated for the NEWEST run, not the one specified."
                )
            logger.info("--- STEP 1: Running DistETA-AIDI Analysis ---")
            run_all_analyses()
            logger.info("--- Analysis complete. Proceeding to report generation. ---")
        report_config = load_report_config()
        generator = AgnosticReportGenerator(
            config=report_config, base_output_dir=constants.OUTPUT_DIR
        )
    except (ValueError, ImportError) as e:
        logger.critical(f"Initialization Error: {e}")
        return  # Exit gracefully

    app_state = {
        "latest_html_report_filename": None,
        "config": report_config,
        "generator": generator,
    }

    if not args.serve_only:
        report_path = generator.generate_report(target_run_dir=args.run_directory)
        app_state["latest_html_report_filename"] = report_path
    else:
        logger.info("Serve-only mode: Finding latest report to serve.")
        latest_dir = generator._find_latest_run_dir()
        if latest_dir:
            reports_path = os.path.join(
                latest_dir, generator.config.directory_structure.reports_dir
            )
            if os.path.isdir(reports_path):
                html_files = sorted(glob.glob(os.path.join(reports_path, "*.html")))
                if html_files:
                    app_state["latest_html_report_filename"] = os.path.relpath(
                        html_files[-1], constants.OUTPUT_DIR
                    )
                    logger.info(
                        f"Will serve latest found report: {app_state['latest_html_report_filename']}"
                    )
                else:
                    logger.warning(f"No HTML reports found in {reports_path}")
            else:
                logger.warning(f"No 'reports' directory found in {latest_dir}")
        else:
            logger.warning("No run directories found to serve from.")

    if args.no_serve:
        logger.info("Report generation complete. --no-serve flag is set, so exiting.")
        return

    if app_state["latest_html_report_filename"]:
        app = create_fastapi_app(app_state)
        logger.info(
            f"FastAPI server starting. Access report at: http://{args.host}:{args.port}/"
        )
        try:
            if not args.serve_only:
                webbrowser.open_new_tab(f"http://{args.host}:{args.port}/")
        except Exception as e:
            logger.warning(f"Could not open web browser: {e}")
        logger.info("Press CTRL+C to stop.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        logger.info("No report available to serve. Exiting.")


if __name__ == "__main__":
    main()
