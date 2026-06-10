"""PyQt6 transistor curve plotting app."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "transistor_plotter_mpl"))

__version__ = "0.1.0"
