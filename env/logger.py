"""
env/logger.py
=============
Centralized logging utility to prevent duplicate handlers when modules
are re-imported dynamically in production (e.g. by uvicorn).
"""

import logging

def get_engine_logger(name: str) -> logging.Logger:
    """Return a strictly isolated logger for the environment engine.

    Sets propagate = False so that messages don't bubble up to root
    loggers (preventing duplication), and attaches exactly one StreamHandler
    per named logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        # Dynamically create format prefix based on submodule (e.g., ENGINE, REWARD, GRADER)
        submodule = name.split(".")[-1].upper()
        formatter = logging.Formatter(f"[%(levelname)s] {submodule} - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
