import logging
from typing import Optional

def get_console_logger(name: Optional[str] = "standard") -> logging.Logger:
    """This function aims to create a logger for debuggign purposes"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create consolo handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Define logger formating properties
        formater = logging.Formatter(
            fmt="{asctime}s - {levelname}s - {lineno}d - {message}s",
            datefmt="%d/%m %H:%M:%S",
            style="{"
        )
        console_handler.setFormatter(formater)

        # Add console handler to the logger
        logger.addHandler(console_handler)
    
    return logger


