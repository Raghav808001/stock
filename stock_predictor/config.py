"""Central configuration for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class Config:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str | None = None          # None -> up to today
    test_start: str | None = None   # None -> the last 365 days are the test set
    forecast_days: int = 30         # business days to forecast past the last close
    refresh: bool = False           # ignore the local CSV cache and re-download
    show_plot: bool = True
    random_state: int = 42
