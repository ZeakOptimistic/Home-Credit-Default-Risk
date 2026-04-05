from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42

for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)