"""upi-fraud package

Expose minimal public API for the project.
"""

__version__ = "0.0.1"

from .data_pipeline import load_data  # noqa: F401
from .features import build_features  # noqa: F401
