from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SQLITE_DB_FILE = DATA_DIR / "routes.db"
SQLITE_URL = f"sqlite:///{SQLITE_DB_FILE}"

MODEL_SYNC_CHECK_INTERVAL_SECONDS = 300
