"""Test package for ConstitutionGPT."""

import logging
import warnings

# Suppress noisy Streamlit ScriptRunContext warnings and resource warnings during command-line testing
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=ResourceWarning)
