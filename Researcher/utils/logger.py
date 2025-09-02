import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger instance that writes logs into logs/session_<timestamp>.log
    
    Args:
        name (str): Name of the logger (e.g. module or agent name).
    
    Returns:
        logging.Logger: Configured logger object.
    """
    os.makedirs('logs', exist_ok= True)
    log_filename = datetime.now().strftime('logs/session_%Y-%m-%d_%H-%M-%S.log')

    # logging configuration
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # writing logs to file
    file_handler = logging.FileHandler(log_filename, mode= 'a', encoding= 'utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Common formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers when calling multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
