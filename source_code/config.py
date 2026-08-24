"""Central configuration: paths, credentials, tunables, and scoring weights.

All values here can be overridden with environment variables so the same
codebase runs unmodified in different environments (local dev, CI, a
teammate's machine) without editing source.
"""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
TAXONOMY_FILE = APP_DIR / "taxonomy.json"

# All pipeline output (checkpoints, scores, knowledge graphs, reports, ...)
# is written under this single root. See output_paths.py for the layout.
OUTPUT_DIR = APP_DIR / "output"

# --- Credentials -------------------------------------------------------
# Read from the environment only. Never hardcode a real token here: doing
# so leaks the credential to anyone who has this file (git history,
# shared zips, etc). Missing tokens just mean lower API rate limits.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "")

LLM_URL = os.environ.get("LLM_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-e4b-it")

GITHUB_API = "https://api.github.com"
GITLAB_API = "https://gitlab.com/api/v4"
RANDOM_SEED = 42

# Temporal snapshot settings (SecoDash 1 parity).
MAX_SNAPSHOTS = 20

# Domain classification and file-hint heuristics.
AIOPS_KW = {"aiops", "root cause analysis", "self-healing", "auto-remediation", "log anomaly detection"}
MLOPS_KW = {"mlops", "kubeflow", "dvc", "ml pipeline", "model serving", "mlflow", "feature store"}
CODE_EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".rs", ".php", ".swift", ".kt", ".r")
STOP_WORDS = {"and", "or", "the", "a", "an", "of", "for", "with", "to", "in", "on", "by", "from", "at", "as", "is", "are"}

# Pipeline limits and tuning.
RECENCY_MONTHS = 12
README_MIN_CHARS = 100
MAX_RETRIES = 2000
MAX_THREADS = 4
LLM_MAX_WORKERS = 6
REQUEST_TIMEOUT = 45
MAX_STALL_ROUNDS = 3
WEIGHT_PERTURBATION_ITERS = 200

# Scoring weights for each composite score category. Percentages within a
# category should sum to 100 (the "ai" category deliberately sums to 110 -
# see generate_perturbed_weights in analysis.py for why).
BASELINE_WEIGHTS = {
    "popularity": {"stars": 40.0, "forks": 30.0, "watchers": 20.0, "citation_bonus": 10.0},
    "maturity": {"readme": 15.0, "license": 15.0, "packages": 15.0, "branches": 15.0, "ci": 20.0, "tests": 20.0},
    "sustainability": {"contributors": 40.0, "recent_update": 30.0, "commit_volume": 30.0},
    "fair": {"fair_license": 25.0, "fair_citation": 20.0, "fair_packages": 20.0, "fair_repro": 20.0, "fair_docs": 15.0},
    "ai": {"ai_mlops": 35.0, "ai_aiops": 25.0, "ai_data": 15.0, "ai_citation": 15.0, "ai_genai": 10.0},
}

# Expected maximum declared-dependency count per language, used to normalize
# the "packages" scoring term on a per-language basis (a JS project's
# typical dependency count dwarfs a Rust project's, so both are compared
# against their own denominator rather than one global constant).
LANG_MAX_DEPS = {
    "javascript": 80,
    "typescript": 80,
    "python": 40,
    "java": 50,
    "ruby": 40,
    "php": 40,
    "rust": 20,
    "go": 20,
    "c++": 15,
    "c": 10,
    # Placeholder in the same order of magnitude as Rust/Go; not empirically
    # calibrated. See Section 7.6 (Construct validity) in the paper.
    "erlang": 15,
    "default": 30,
}

# Manifest files recognized by extraction.py's fallback package-detection path.
MANIFEST_FILES = {
    "requirements.txt": "python",
    "package.json": "javascript",
    "pyproject.toml": "python",
    "pipfile": "python",
    "cargo.toml": "rust",
    "go.mod": "go",
    "CMakeLists.txt": "cpp",
    "Makefile": "cpp",
    "DESCRIPTION": "r",
    "Project.toml": "julia",
    "*.csproj": "csharp",
    "*.fsproj": "fsharp",
    "build.gradle": "java",
    "pom.xml": "java",
    "composer.json": "php",
    "Cargo.lock": "rust",
    "go.sum": "go",
    "environment.yml": "conda",
    "conda.recipe": "conda",
    "rebar.config": "erlang",
}

DEPENDENCY_PATTERNS = {
    "r": ["DESCRIPTION", "NAMESPACE"],
    "julia": ["Project.toml", "Manifest.toml"],
    "cpp": ["CMakeLists.txt", "Makefile", "*.mk"],
    "csharp": ["*.csproj", "*.fsproj"],
    "php": ["composer.json"],
    "scala": ["build.sbt", "*.scala"],
    "lua": ["*.rockspec"],
    "perl": ["Makefile.PL", "Build.PL", "cpanfile"],
    "erlang": ["rebar.config"],
}
