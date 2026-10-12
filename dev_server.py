#!/usr/bin/env python3
"""Convenient repository-root launcher: python dev_server.py."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    backend = Path(__file__).resolve().parent / "src" / "ai-origin"
    sys.path.insert(0, str(backend))
    runpy.run_path(str(backend / "dev.py"), run_name="__main__")
