# logger_config.py
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup application-wide logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "service.log",
        maxBytes=50*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:  # Only setup once
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Reduce noise
        logging.getLogger('elasticsearch').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def get_logger(name):
    """Get a logger with proper configuration"""
    setup_logging()  # Ensure logging is setup
    return logging.getLogger(name)