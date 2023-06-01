import logging
import os

# Default logger used throughout the package
logger = logging.getLogger(os.environ.get("BBO_LOGGER_NAME", "blackboxopt"))
