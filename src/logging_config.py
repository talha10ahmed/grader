import logging
import os

from pathlib import Path

# Ensure log directory exists inside grader-main
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = str(LOGS_DIR / "exam_processing.log")

# Configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Create a named logger for your app
logger = logging.getLogger("exam_pipeline")
