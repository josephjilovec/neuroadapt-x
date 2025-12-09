"""
Logging configuration for NeuroAdapt-X

Provides structured logging with different levels and handlers
for console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "neuroadapt",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (default: ./logs)
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str = "neuroadapt") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


# Default logger instance
logger = get_logger()

