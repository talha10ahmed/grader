import logging
import os

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", "exam_processing.log")

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
