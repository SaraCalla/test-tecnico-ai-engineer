"""Logger configuration using loguru."""
from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom handler with colorized output and better formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

__all__ = ["logger"]
