"""Project paths shared by the CLI tools and tests."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
GOLD_DIR = DATA_DIR / "gold"
REPORT_DIR = PROJECT_ROOT / "evals" / "reports"

DEFAULT_POLICY_PDF = POLICY_DIR / "synthetic_health_policy.pdf"
DEFAULT_GOLD_FILE = GOLD_DIR / "gold_questions.json"
