"""
Logging utility for the cyber insurance ML pipeline.
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optionally file output.

    Args:
        name (str): Name of the logger
        log_file (Path, optional): Path to log file. If None, only console output is used
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # Console handler (with simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified (with detailed format)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger
