"""Run full, raw-image baseline or configurable source ablations."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.cli import main

if __name__ == "__main__":
    main(default_mode="leave_one_out")
