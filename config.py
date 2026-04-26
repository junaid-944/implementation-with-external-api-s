"""
Configuration for the Agentic Planning Study.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration (Google Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 1024

# --- Experiment Configuration ---
NUM_RUNS_PER_TASK = 1          # Reduced from 3 to control API spend
MAX_STEPS_PER_TASK = 12        # Lower safety limit to reduce looping
TIMEOUT_SECONDS = 120          # Per-task timeout
MAX_TOOL_CALLS_PER_TASK = 8    # Hard cap to reduce looping / cost
AGENT_RECURSION_LIMIT = 15     # LangGraph recursion cap for ReAct-style agents

# --- API Rate Limiting ---
LLM_CALL_DELAY = 1.0           # Seconds between LLM calls
TOOL_CALL_DELAY = 0.5          # Seconds between real API calls (polite rate limiting)
MAX_RETRIES = 2                # Retries on 429/503 errors
RETRY_BASE_DELAY = 10          # Base delay for exponential backoff (seconds)

# --- Tree of Thoughts Configuration ---
TOT_BRANCHES = 1               # Reduced from 3 to control branching cost
TOT_MAX_DEPTH = 2              # Shallower search keeps ToT focused and cheaper
TOT_EVALUATION_STRATEGY = "vote"  # "vote" or "score"

# --- Output Paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
