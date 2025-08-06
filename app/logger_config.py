import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Define log file path
log_file_path = os.path.join(log_dir, "training.log")

# Create logger
logger = logging.getLogger("lora_logger")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:
    # File handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=1_000_000, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
